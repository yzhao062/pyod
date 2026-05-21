"""LLM-client Protocol, prompt builder, and routing-response parser.

This module powers pyod 3.5.3's :meth:`ADEngine.plan_detection` Surface 2
extension. When a user passes ``llm_client=callable``, the engine
invokes :func:`build_routing_prompt` and :func:`parse_routing_response`
through this module; when ``llm_client=None``, the rules path is
unchanged.

Public:
    LLMCallable -- typing.Protocol; any (prompt: str) -> str
    RoutingParseError -- raised when the parser cannot extract a plan
    build_routing_prompt(kb_context, top_k) -> str
    parse_routing_response(response, kb, top_k) -> (list[str], list[str])

No optional dependencies are imported at module load; PyOD does not ship
any provider-specific adapters.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol


logger = logging.getLogger(__name__)


class LLMCallable(Protocol):
    """Any ``(prompt: str) -> str`` callable.

    Users supply an instance wrapping their preferred LLM SDK. Example
    (Anthropic SDK):

    .. code-block:: python

        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        def my_llm(prompt: str) -> str:
            return client.messages.create(
                model='claude-opus-4-7',
                max_tokens=4096,
                messages=[{'role': 'user', 'content': prompt}],
            ).content[0].text

        plan = engine.plan_detection(profile, llm_client=my_llm)

    PyOD ships no provider-specific adapter classes; users wrap their
    own SDK in a ``(prompt) -> str`` callable.
    """

    def __call__(self, prompt: str) -> str: ...


class RoutingParseError(ValueError):
    """Raised when :func:`parse_routing_response` cannot extract a plan.

    The engine catches this and falls back to rule-driven routing unless
    the environment variable ``PYOD3_LLM_STRICT=1`` is set.
    """


def build_routing_prompt(kb_context: dict, top_k: int = 3) -> str:
    """Render a routing prompt from a knowledge-base context dict.

    Parameters
    ----------
    kb_context : dict
        Output of :meth:`ADEngine.get_kb_for_routing`. Carries
        ``task_profile`` and ``available_detectors`` lists.
    top_k : int
        Number of detectors the LLM should select. Default 3.

    Returns
    -------
    str
        A self-contained prompt instructing the LLM to return a JSON
        array of ``{"detector": ..., "justification": ...}`` objects.

    Notes
    -----
    The template avoids chain-of-thought scaffolding so the same prompt
    works across diverse LLMs (Claude, GPT, Gemini, open-weight).
    """
    profile = kb_context.get("task_profile", {})
    detectors = kb_context.get("available_detectors", [])
    # Soft-KB view (opt-in): get_kb_for_routing(scoring_view="soft_kb") stamps
    # a top-level kb_version plus per-detector kb_* fields. When present, render
    # the calibrated within-modality score + uncertainty so the agent reasons
    # over benchmark evidence rather than the legacy citation rank. The legacy
    # view (no kb_version) falls through to the byte-identical path below.
    if "kb_version" in kb_context:
        return _build_soft_kb_prompt(kb_context, profile, detectors, top_k)
    # Compress each detector entry to a single line: name + 1-line
    # best_for + 1-line avoid_when + benchmark_rank (resolved per-modality).
    # Prefer `resolved_rank` / `resolved_rank_key` that get_kb_for_routing
    # stamped on each entry; fall back to the modality-title-overall key
    # for older callers that bypass get_kb_for_routing.
    lines = []
    for d in detectors:
        rank = d.get("resolved_rank")
        rank_key = d.get("resolved_rank_key")
        if rank is None:
            bench = d.get("benchmark_rank") or {}
            rank = bench.get(
                f"{str(profile.get('data_type', 'tabular')).title()}_overall"
            ) or bench.get("ADBench_overall")
            rank_key = None
        rank_str = (f" rank={rank} ({rank_key})" if rank is not None
                    and rank_key else
                    (f" rank={rank}" if rank is not None else ""))
        strengths = "; ".join((d.get("strengths") or [])[:2])
        weaknesses = "; ".join((d.get("weaknesses") or [])[:2])
        best_for = d.get("best_for") or ""
        avoid_when = d.get("avoid_when") or ""
        lines.append(
            f"- {d['name']} ({d.get('category', 'unknown')}{rank_str}): "
            f"best_for={best_for!r}; avoid_when={avoid_when!r}; "
            f"strengths=[{strengths}]; weaknesses=[{weaknesses}]")

    profile_str = (
        f"data_type={profile.get('data_type', 'tabular')}, "
        f"n_samples={profile.get('n_samples', '?')}, "
        f"n_features={profile.get('n_features', '?')}, "
        f"contamination_estimate={profile.get('contamination_estimate', '?')}"
    )

    return (
        "You are an anomaly-detection routing expert. Given the task "
        "profile and a list of available detectors annotated with "
        "strengths, weaknesses, best_for, avoid_when, and benchmark "
        "rank, choose the ordered top-K detectors most likely to "
        "succeed on this task.\n\n"
        f"TASK PROFILE: {profile_str}\n\n"
        f"AVAILABLE DETECTORS ({len(detectors)}):\n"
        + "\n".join(lines) + "\n\n"
        f"Return exactly {top_k} detectors as a JSON array of objects, "
        'each shaped {"detector": "<name>", "justification": "<one '
        'sentence>"}. Detector names are case-sensitive and must come '
        "from the list above. Return ONLY the JSON array (no prose, "
        "no markdown fences).\n"
    )


# Routing guidance for the soft-KB view: turns the calibrated scores into
# behavior (build step 3, "uncertainty must be actionable") rather than
# leaving them as decorative numbers.
_SOFT_KB_GUIDE = (
    "KB SCORES: kb_score is a calibrated within-modality percentile in "
    "[0,1] (1 = best) over the listed metric scope; it is NOT comparable "
    "across modalities. ci=[low,high] is a bootstrap uncertainty interval "
    "and n is the number of supporting datasets. Use the scores as "
    "evidence, not a hard ranking: when the top candidates' intervals "
    "overlap, do not over-commit to one detector -- prefer a small set to "
    "evaluate. Treat low n or a 'modality'-level fallback as weaker "
    "evidence and weigh task fit (best_for / avoid_when) more heavily. "
    "Detectors marked 'no KB score' are unbenchmarked for this modality, "
    "not necessarily weak."
)


def _build_soft_kb_prompt(kb_context: dict, profile: dict,
                          detectors: list, top_k: int) -> str:
    """Render the routing prompt for the opt-in soft-KB view.

    Surfaces each detector's calibrated within-modality ``kb_score`` with its
    bootstrap interval, modality rank, support count, metric scope, and
    fallback level, plus the actionable-uncertainty guidance in
    :data:`_SOFT_KB_GUIDE`. Detectors with no score for this modality are
    rendered as ``no KB score`` (unbenchmarked, never a silent zero).
    """
    n_scored = sum(1 for d in detectors if d.get("kb_score") is not None)
    lines = []
    for d in detectors:
        score = d.get("kb_score")
        if score is None:
            annot = ", no KB score"
        else:
            interval = d.get("kb_interval") or [None, None]
            lo, hi = interval[0], interval[1]
            ci = (f" ci=[{lo:.3f},{hi:.3f}]"
                  if lo is not None and hi is not None else "")
            rank = d.get("kb_score_rank")
            rank_str = f" rank={rank}/{n_scored}" if rank is not None else ""
            n = d.get("kb_coverage_n")
            n_str = f" n={n}" if n is not None else ""
            scope = d.get("kb_metric_scope")
            scope_str = (f" scope={'+'.join(scope)}"
                         if isinstance(scope, list) and scope else "")
            fb = d.get("kb_fallback_level")
            fb_str = f" {fb}-level" if fb else ""
            annot = (f", kb_score={score:.3f}{ci}{rank_str}{n_str}"
                     f"{scope_str}{fb_str}")
        strengths = "; ".join((d.get("strengths") or [])[:2])
        weaknesses = "; ".join((d.get("weaknesses") or [])[:2])
        best_for = d.get("best_for") or ""
        avoid_when = d.get("avoid_when") or ""
        lines.append(
            f"- {d['name']} ({d.get('category', 'unknown')}{annot}): "
            f"best_for={best_for!r}; avoid_when={avoid_when!r}; "
            f"strengths=[{strengths}]; weaknesses=[{weaknesses}]")

    profile_str = (
        f"data_type={profile.get('data_type', 'tabular')}, "
        f"n_samples={profile.get('n_samples', '?')}, "
        f"n_features={profile.get('n_features', '?')}, "
        f"contamination_estimate={profile.get('contamination_estimate', '?')}"
    )

    # Never ask the agent for more detectors than the candidate set holds:
    # the image and text modalities expose fewer than the default top_k, and
    # an impossible exact count pushes the LLM toward duplicate or invented
    # names (which would add noise to the Phase B soft-KB condition).
    n_to_return = min(top_k, len(detectors))
    return (
        "You are an anomaly-detection routing expert. Given the task "
        "profile and a list of available detectors annotated with "
        "strengths, weaknesses, best_for, avoid_when, and a calibrated "
        "within-modality KB score with an uncertainty interval, choose the "
        "ordered top-K detectors most likely to succeed on this task.\n\n"
        f"{_SOFT_KB_GUIDE}\n\n"
        f"TASK PROFILE: {profile_str}\n\n"
        f"AVAILABLE DETECTORS ({len(detectors)}):\n"
        + "\n".join(lines) + "\n\n"
        f"Return exactly {n_to_return} detectors as a JSON array of objects, "
        'each shaped {"detector": "<name>", "justification": "<one '
        'sentence>"}. Detector names are case-sensitive and must come '
        "from the list above. Return ONLY the JSON array (no prose, "
        "no markdown fences).\n"
    )


# Matches a balanced top-level JSON array. We do not parse arbitrary
# nested arrays defensively; the spec asks for a flat list of objects.
_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*?\]", re.DOTALL)


def _extract_first_array(response: str) -> str | None:
    """Return the first ``[...]`` substring that parses as a JSON list.

    Tolerates surrounding prose, markdown fences, or repeated arrays.
    Returns ``None`` if no parseable array is found.
    """
    # Strip ```json fences if present.
    fenced = re.sub(r"```(?:json)?\s*", "", response)
    fenced = re.sub(r"```", "", fenced)

    # Try greedy first: from first '[' to last ']'.
    first = fenced.find("[")
    last = fenced.rfind("]")
    if 0 <= first < last:
        candidate = fenced[first: last + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return candidate
        except json.JSONDecodeError:
            pass

    # Fall back to balanced matches without nested arrays.
    for m in _JSON_ARRAY_RE.findall(fenced):
        try:
            data = json.loads(m)
            if isinstance(data, list):
                return m
        except json.JSONDecodeError:
            continue
    return None


def parse_routing_response(response: str, kb: Any,
                           top_k: int = 3) -> tuple[list[str], list[str]]:
    """Parse an LLM routing response into ``(detector_choices, justifications)``.

    Parameters
    ----------
    response : str
        The raw LLM text. Expected to be a JSON array of
        ``{"detector": str, "justification": str}`` objects, but
        tolerates surrounding prose and markdown fences.
    kb : pyod.utils.knowledge.KnowledgeBase
        Used to validate detector names. Unknown names are skipped
        with a warning.
    top_k : int
        Truncate to at most this many detectors. Default 3.

    Returns
    -------
    detector_choices : list[str]
        Validated, ordered list (length >= 1). Trimmed to ``top_k``.
    justifications : list[str]
        Parallel list, one short sentence per detector. Empty string
        when the LLM omitted the field.

    Raises
    ------
    RoutingParseError
        If no JSON array can be extracted OR if fewer than 1 detector
        survives validation against ``kb``.
    """
    if not isinstance(response, str):
        raise RoutingParseError(
            f"response must be a string; got {type(response).__name__}")

    candidate = _extract_first_array(response)
    if candidate is None:
        raise RoutingParseError(
            "no JSON array found in LLM response (response head: "
            f"{response[:120]!r})")

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as ex:
        raise RoutingParseError(f"JSON parse error: {ex}") from ex

    if not isinstance(data, list):
        raise RoutingParseError(
            f"expected JSON array; got {type(data).__name__}")

    detector_choices: list[str] = []
    justifications: list[str] = []
    seen: set[str] = set()
    for entry in data:
        if isinstance(entry, str):
            name = entry
            just = ""
        elif isinstance(entry, dict):
            name = entry.get("detector") or entry.get("name") or ""
            just = (entry.get("justification") or entry.get("reason")
                    or "")
        else:
            continue
        if not isinstance(name, str) or not name:
            continue
        # Drop duplicates so the LLM cannot pad top_k with repeats.
        if name in seen:
            continue
        algo = kb.get_algorithm(name)
        if algo is None:
            logger.warning(
                "parse_routing_response: skipping unknown detector %r "
                "(not in KB)", name)
            continue
        if algo.get("status") != "shipped":
            logger.warning(
                "parse_routing_response: skipping non-shipped detector "
                "%r (status=%r)", name, algo.get("status"))
            continue
        detector_choices.append(name)
        justifications.append(just if isinstance(just, str) else "")
        seen.add(name)
        if len(detector_choices) >= top_k:
            break

    if not detector_choices:
        raise RoutingParseError(
            "no valid detector names in LLM response after KB "
            f"validation (raw array: {candidate[:200]!r})")

    return detector_choices, justifications
