# -*- coding: utf-8 -*-
"""API-reference safety net for packaged pyod skill content.

Validates that every ``state.X`` / ``state.X['a']['b']`` / ``engine.X(...)``
reference in the skill Markdown files maps to a real attribute / nested
dict key chain / method-and-kwargs on a live ``ADEngine`` /
``InvestigationState`` pair.

This catches the v3.2.0 regression that shipped ``state.plan['detectors']``,
``state.scores`` etc. (the v3.2.1 Round-0 scan), AND the v3.2.1 Round-1
findings that the first version of this test still missed:

- Only one ``['key']`` segment was captured; nested refs like
  ``state.analysis['consensus_analysis']['top_anomalies']`` were not
  walked at all, so a bad nested key would slip past the validator.
- ``engine.method`` was checked for name existence only; invalid kwargs
  like ``engine.start(X, y=labels)`` (there is no ``y`` parameter on
  ``ADEngine.start``) passed the test silently.

The v3.2.1 test adds:

1. A nested-key walker that reads ``state.attr['a']['b']['c']`` chains
   and validates each level against the live ground truth collected
   from a dry run (``_build_interface_ground_truth`` already populates
   ``valid_dict_keys[f"{attr}.{k}"]`` nested keys; now those are
   actually consulted).
2. A kwargs validator that uses ``inspect.signature`` on every
   ``engine.<name>(...)`` call site scraped from the Markdown and
   rejects any keyword argument not accepted by the real method.
3. Explicit negative tests that fabricate the exact regression shapes
   (``state.analysis['consensus_analysis']['not_a_real_key']`` and
   ``engine.start(X, y=labels)``) and run the same scanner against a
   synthetic Markdown blob, asserting the scanner flags them.
"""
import inspect
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "pyod" / "skills"
AD_ENGINE_SRC = REPO_ROOT / "pyod" / "utils" / "ad_engine.py"

# Match ``state.attr`` followed by zero or more ``['key']`` / ``["key"]``
# segments. Captures the attribute name in group 1 and the full chain of
# key segments in group 2 (to be parsed below). Negative lookbehind keeps
# us from matching substrings like ``my_state.foo``.
_STATE_REF_RE = re.compile(
    r"(?<![a-zA-Z_0-9.])state\.([a-zA-Z_]\w*)((?:\[['\"][^'\"]+['\"]\])*)"
)

# Extract each individual ``['key']`` segment from the chain captured above.
_KEY_SEGMENT_RE = re.compile(r"\[['\"]([^'\"]+)['\"]\]")

# Match ``engine.<method>(`` — we want the call site so we can parse the
# kwargs that follow. The parenthesis confirms it's a call, not a
# plain attribute reference.
_ENGINE_CALL_RE = re.compile(
    r"(?<![a-zA-Z_0-9.])engine\.([a-zA-Z_]\w*)\("
)

# Match a bare ``engine.<method>`` reference not followed by ``(``. These
# are attribute references in prose (e.g. "call `engine.start`") that we
# only validate by name, not by kwargs.
_ENGINE_ATTR_RE = re.compile(
    r"(?<![a-zA-Z_0-9.])engine\.([a-zA-Z_]\w*)(?!\w|\()"
)


def _build_interface_ground_truth():
    """Walk ADEngine + InvestigationState via multi-modality dry runs.

    Returns a tuple of:
        (engine_attrs, state_attrs, valid_dict_keys, engine_signatures)

    ``valid_dict_keys`` maps both top-level attribute names (``profile``,
    ``quality``, ...) and dotted nested paths
    (``analysis.consensus_analysis``) to the UNION of keys observed inside
    that dict across multiple modality-specific dry runs. We walk:

    - ``start(tabular_X) -> plan -> run -> analyze`` (primary)
    - ``start(univariate_y, data_type='time_series') -> plan`` for
      time-series-only profile keys (``n_timestamps``, ``channels``)
    - ``start(list_of_strings)`` for text-only profile keys

    Graph is deliberately skipped because it requires ``torch_geometric``
    which is not available on core installs.

    ``engine_signatures`` maps method names to the set of keyword-argument
    names each method accepts (plus a sentinel ``"**"`` if the method
    declares ``**kwargs``, meaning any kwarg is allowed).
    """
    from pyod.utils.ad_engine import ADEngine

    engine = ADEngine()
    engine_attrs = {a for a in dir(engine) if not a.startswith("_")}
    state_attrs = set()
    valid_dict_keys = {}

    def _walk_dict(prefix, d, out):
        """Recursively populate out[prefix] with d.keys(), then recurse."""
        if not isinstance(d, dict):
            return
        out.setdefault(prefix, set()).update(d.keys())
        for k, v in d.items():
            if isinstance(v, dict):
                _walk_dict(f"{prefix}.{k}", v, out)

    def _collect(state):
        for a in dir(state):
            if a.startswith("_"):
                continue
            state_attrs.add(a)
            try:
                val = getattr(state, a)
            except Exception:
                continue
            if isinstance(val, dict):
                _walk_dict(a, val, valid_dict_keys)

    # Tabular dry run: full lifecycle start -> plan -> run -> analyze.
    np.random.seed(0)
    X_tab = np.random.randn(50, 5)
    state = engine.start(X_tab)
    state = engine.plan(state)
    state = engine.run(state)
    state = engine.analyze(state)
    _collect(state)

    # Time-series dry run: only start + plan. We just need the
    # time-series-specific profile keys (n_timestamps, channels).
    try:
        engine_ts = ADEngine()
        y_ts = np.random.randn(200)
        state_ts = engine_ts.start(y_ts, data_type="time_series")
        state_ts = engine_ts.plan(state_ts)
        _collect(state_ts)
    except Exception:
        pass  # best-effort; if ts planner errors, we just miss those keys

    # Text dry run: only start + plan. Needed so text profiles are
    # represented in the union (even though current skill content does
    # not reference text-specific keys, being defensive is cheap).
    try:
        engine_text = ADEngine()
        state_text = engine_text.start(
            ["sample text %d" % i for i in range(5)]
        )
        state_text = engine_text.plan(state_text)
        _collect(state_text)
    except Exception:
        pass

    # Source scrape: next_action can have keys set in branches the dry run
    # did not trigger (iterate / confirm_with_user). Walk the ad_engine.py
    # source for literal 'key' assignments on next_action dicts.
    src = AD_ENGINE_SRC.read_text(encoding="utf-8")
    extra_next_action_keys = set()
    for m in re.finditer(
        r"next_action\s*=\s*\{([^}]*)\}", src, re.DOTALL
    ):
        block = m.group(1)
        for km in re.finditer(r"['\"](\w+)['\"]\s*:", block):
            extra_next_action_keys.add(km.group(1))
    for m in re.finditer(r"next_action\[['\"](\w+)['\"]\]", src):
        extra_next_action_keys.add(m.group(1))
    if extra_next_action_keys:
        valid_dict_keys.setdefault("next_action", set()).update(
            extra_next_action_keys
        )

    # Collect accepted keyword-argument names per engine method. Methods
    # declaring **kwargs are flagged with the sentinel "**" so any kwarg
    # passes (fall back to name-only validation for those).
    engine_signatures = {}
    for name in engine_attrs:
        try:
            obj = getattr(engine, name)
        except Exception:
            continue
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        accepted = set()
        has_var_keyword = False
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted.add(p.name)
        if has_var_keyword:
            accepted.add("**")
        engine_signatures[name] = accepted

    return engine_attrs, state_attrs, valid_dict_keys, engine_signatures


def _all_skill_files():
    return sorted(SKILLS_DIR.rglob("*.md"))


def _short_path(path):
    """Render *path* relative to the repo root if possible, else absolute.

    Used when building error messages. The negative-test fixture points
    SKILLS_DIR at a pytest tmp_path outside the repo, so path.relative_to
    would raise ValueError there.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _walk_state_ref(text, attr, key_chain_source, valid_dict_keys,
                    state_attrs):
    """Validate a single ``state.attr[..]..`` occurrence against ground truth.

    Returns (ok, problem_message) where problem_message is None on success.
    """
    if attr not in state_attrs:
        return False, f"state.{attr} — attribute not on InvestigationState"

    segments = _KEY_SEGMENT_RE.findall(key_chain_source)
    if not segments:
        return True, None

    # Walk the segments; each step must resolve to a dict whose keys
    # contain the next segment.
    path = attr
    for depth, seg in enumerate(segments):
        if path not in valid_dict_keys:
            # Previous level is not a dict, or no keys were observed.
            # If this is the very first segment and the attribute is
            # simply not a dict, produce a cleaner message.
            if depth == 0:
                return False, (
                    f"state.{attr}[{seg!r}] — state.{attr} is not a dict"
                )
            return False, (
                f"state.{attr}{_render_chain(segments[:depth + 1])} — "
                f"parent state.{attr}{_render_chain(segments[:depth])} is not a dict"
            )
        if seg not in valid_dict_keys[path]:
            return False, (
                f"state.{attr}{_render_chain(segments[:depth + 1])} — "
                f"{seg!r} not in state.{attr}{_render_chain(segments[:depth])}. "
                f"Valid: {sorted(valid_dict_keys[path])}"
            )
        path = f"{path}.{seg}"
    return True, None


def _render_chain(segments):
    return "".join(f"[{s!r}]" for s in segments)


def _extract_engine_call_kwargs(text, call_start):
    """Given the position just after ``engine.<name>(`` in *text*, return
    the set of kwarg names used in the call. Crude but sufficient for
    skill content: scans forward to the balancing ``)`` and finds
    identifier-followed-by-``=`` tokens.

    Handles nested parens and simple string literals, not full Python
    syntax; good enough for the 1-line call snippets the skill content
    uses.
    """
    depth = 1
    i = call_start
    start = call_start
    in_str = None  # None or the quote char
    while i < len(text) and depth > 0:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < len(text):
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in ("'", '"'):
            in_str = c
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    call_text = text[start:i]
    kwargs = set()
    # kwarg pattern: identifier followed by '=' (but not '==')
    for m in re.finditer(r"\b([a-zA-Z_]\w*)\s*=(?!=)", call_text):
        kwargs.add(m.group(1))
    return kwargs


def test_skill_api_refs_exist_on_adengine_and_state():
    """Every ``state.X`` / ``state.X['a']['b']`` / ``engine.X(...)`` ref
    in the skill Markdown must match a real attribute / nested dict key /
    method signature.

    Catches the v3.2.0 / v3.2.1 regression families: invented attributes
    (``state.plan``, ``state.scores``, ``state.best_detector``), invented
    nested keys (``state.profile['estimated_contamination']``), and
    impossible call signatures (``engine.start(X, y=labels)``).
    """
    (engine_attrs, state_attrs, valid_dict_keys,
     engine_signatures) = _build_interface_ground_truth()

    problems = []
    seen = set()
    for path in _all_skill_files():
        text = path.read_text(encoding="utf-8")

        # --- state.X refs with optional nested dict-key chain ---
        for m in _STATE_REF_RE.finditer(text):
            attr = m.group(1)
            chain_source = m.group(2) or ""
            lineno = text[: m.start()].count("\n") + 1
            dedupe = (str(path), lineno, "state", attr, chain_source)
            if dedupe in seen:
                continue
            seen.add(dedupe)

            ok, msg = _walk_state_ref(
                text, attr, chain_source, valid_dict_keys, state_attrs
            )
            if not ok:
                problems.append(
                    f"{_short_path(path)}:{lineno}: {msg}"
                )

        # --- engine.X(...) call sites: validate method name + kwargs ---
        for m in _ENGINE_CALL_RE.finditer(text):
            name = m.group(1)
            lineno = text[: m.start()].count("\n") + 1
            dedupe = (str(path), lineno, "engine_call", name)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            if name not in engine_attrs:
                problems.append(
                    f"{_short_path(path)}:{lineno}: "
                    f"engine.{name}(...) — method not on ADEngine"
                )
                continue
            accepted = engine_signatures.get(name)
            if accepted is None:
                # Not inspectable (e.g. a property or non-callable attr).
                # Skip kwarg validation but keep the name check above.
                continue
            if "**" in accepted:
                # Method declares **kwargs; anything goes.
                continue
            call_start = m.end()
            kwargs = _extract_engine_call_kwargs(text, call_start)
            # The first argument is usually positional (X, state, fb, ...);
            # we only flag kwargs that are NOT in the accepted set.
            unknown = {k for k in kwargs if k not in accepted}
            if unknown:
                problems.append(
                    f"{_short_path(path)}:{lineno}: "
                    f"engine.{name}(...) — unknown keyword arg(s) "
                    f"{sorted(unknown)}. Accepted: {sorted(accepted - {'self'})}"
                )

        # --- bare engine.X attribute refs in prose (no call) ---
        for m in _ENGINE_ATTR_RE.finditer(text):
            name = m.group(1)
            lineno = text[: m.start()].count("\n") + 1
            dedupe = (str(path), lineno, "engine_attr", name)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            if name not in engine_attrs:
                problems.append(
                    f"{_short_path(path)}:{lineno}: "
                    f"engine.{name} — attribute not on ADEngine"
                )

    assert not problems, (
        "API reference errors in skill files:\n  "
        + "\n  ".join(problems)
        + "\n\nEither fix the reference to match the real API, or if a "
        + "legitimate new attribute / key / method was added, update the "
        + "interface ground truth in this test."
    )


def test_interface_ground_truth_snapshot_is_non_empty():
    """Sanity check: the live dry run actually populated the state, and
    ``engine_signatures`` captured the signatures we rely on for kwarg
    validation.
    """
    (engine_attrs, state_attrs, valid_dict_keys,
     engine_signatures) = _build_interface_ground_truth()
    assert "start" in engine_attrs
    assert "plan" in engine_attrs
    assert "run" in engine_attrs
    assert "analyze" in engine_attrs
    assert "iterate" in engine_attrs
    assert "report" in engine_attrs
    assert "plans" in state_attrs
    assert "consensus" in state_attrs
    assert "quality" in state_attrs
    assert "analysis" in state_attrs
    assert valid_dict_keys["profile"], "profile dict is empty after analyze"
    assert valid_dict_keys["quality"], "quality dict is empty after analyze"
    assert valid_dict_keys["consensus"], "consensus dict is empty after analyze"
    assert valid_dict_keys["analysis"], "analysis dict is empty after analyze"
    assert "summary" in valid_dict_keys["next_action"], (
        "next_action source scrape missed 'summary'"
    )
    # Signature capture: engine.start's accepted kwargs should include
    # 'data_type' but NOT 'y' (invented in v3.2.0).
    start_kwargs = engine_signatures["start"]
    assert "data_type" in start_kwargs, (
        f"engine.start should accept data_type kwarg. Got: {sorted(start_kwargs)}"
    )
    assert "y" not in start_kwargs, (
        f"engine.start should NOT accept y kwarg. Got: {sorted(start_kwargs)}"
    )
    # Nested dict walk: analysis.consensus_analysis should have its
    # observed keys populated.
    assert "analysis.consensus_analysis" in valid_dict_keys, (
        "nested dict walk missed analysis.consensus_analysis"
    )
    assert "top_anomalies" in valid_dict_keys["analysis.consensus_analysis"], (
        "consensus_analysis did not expose top_anomalies"
    )


def test_validator_catches_regression_shapes(tmp_path):
    """Synthetic negative tests: the validator must flag the exact
    regression shapes Codex's Round 1 review raised.

    If any of these assertions fail, the safety net has a gap and new
    skill content could ship the same bugs.
    """
    # Write a synthetic skill file with several deliberate bugs.
    bad = tmp_path / "od_expert" / "bad.md"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "<!-- synthetic test fixture -->\n"
        "First, call `state.plan['detectors']` (invented attribute).\n"
        "Check `state.analysis['consensus_analysis']['not_a_real_key']` (bad nested key).\n"
        "Switch to supervised via `engine.start(X, y=labels)` (bad kwarg).\n"
        "Also call `engine.no_such_method_exists(X)` (missing method).\n"
        "And read `engine.made_up_attr` (missing attr in prose).\n",
        encoding="utf-8",
    )

    # Point the scanner at the temp dir by monkey-patching SKILLS_DIR.
    import pyod.test.test_skill_api_refs as mod
    original_skills_dir = mod.SKILLS_DIR
    mod.SKILLS_DIR = tmp_path
    try:
        try:
            mod.test_skill_api_refs_exist_on_adengine_and_state()
        except AssertionError as e:
            msg = str(e)
        else:
            raise AssertionError(
                "The validator did not flag ANY of the synthetic regression "
                "shapes. The safety net is broken."
            )
    finally:
        mod.SKILLS_DIR = original_skills_dir

    # Each of the five bugs must be named in the assertion message.
    expected_fragments = [
        "state.plan",                       # invented attribute
        "'not_a_real_key'",                 # nested bad key
        "unknown keyword arg",              # engine.start y=...
        "no_such_method_exists",            # missing method at call site
        "made_up_attr",                     # missing attr in prose
    ]
    missing = [f for f in expected_fragments if f not in msg]
    assert not missing, (
        f"Validator message missed regression fragments: {missing}\n\n"
        f"Full message:\n{msg}"
    )
