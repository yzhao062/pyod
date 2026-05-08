"""ADEngine iterate() feedback parsing.

Extracts `_iterate_structured` and `_iterate_nl` from
`pyod.utils.ad_engine.ADEngine` in 2026-05.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .investigation import _make_history_entry

if TYPE_CHECKING:
    from pyod.utils.investigation import InvestigationState
    from pyod.utils.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


_CONTAMINATION_INCREASE_FACTOR: float = 1.5
_CONTAMINATION_DECREASE_FACTOR: float = 0.5
_CONTAMINATION_MAX: float = 0.5
_CONTAMINATION_MIN: float = 0.01
_NL_HIGH_CONFIDENCE_THRESHOLD: float = 0.8
"""NL feedback parsed with confidence above this is auto-applied."""


_VALID_ACTIONS: frozenset[str] = frozenset({
    'adjust_contamination', 'exclude', 'include', 'rerun',
})

_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    'adjust_contamination': frozenset({'value'}),
    'exclude': frozenset({'detectors'}),
    'include': frozenset({'detectors'}),
    'rerun': frozenset(),
}


def validate_structured_feedback(feedback: dict) -> None:
    """Raise ValueError if feedback dict is malformed.

    Validates only structure and required fields. Does not validate
    semantic content (e.g., whether a detector name actually exists in
    the KB; that is the responsibility of `apply_structured_feedback`).

    Parameters
    ----------
    feedback : dict
        Structured feedback per design spec section 4.3 of
        `docs/superpowers/specs/2026-04-12-v3-agentic-design.md`.

    Raises
    ------
    ValueError
        If 'action' is missing, action is not in `_VALID_ACTIONS`, or
        required fields for the chosen action are missing.
    """
    if not isinstance(feedback, dict):
        raise ValueError(
            f'feedback must be a dict, got {type(feedback).__name__}')
    action = feedback.get('action')
    if action is None:
        raise ValueError(
            "feedback dict missing 'action' key; "
            f"valid actions: {sorted(_VALID_ACTIONS)}")
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"unknown action {action!r}; "
            f"valid actions: {sorted(_VALID_ACTIONS)}")
    missing = _REQUIRED_FIELDS[action] - feedback.keys()
    if missing:
        raise ValueError(
            f"action {action!r} requires fields {sorted(missing)}, "
            f"got fields {sorted(feedback.keys())}")


def adjust_contamination_up(current: float) -> float:
    """Increase contamination by `_CONTAMINATION_INCREASE_FACTOR`, capped at `_CONTAMINATION_MAX`."""
    return min(current * _CONTAMINATION_INCREASE_FACTOR, _CONTAMINATION_MAX)


def adjust_contamination_down(current: float) -> float:
    """Decrease contamination by `_CONTAMINATION_DECREASE_FACTOR`, floored at `_CONTAMINATION_MIN`."""
    return max(current * _CONTAMINATION_DECREASE_FACTOR, _CONTAMINATION_MIN)


def apply_structured_feedback(
        state: 'InvestigationState',
        feedback: dict,
        kb: 'KnowledgeBase',
        plan_detection_fn: Callable,
        make_plan_fn: Callable) -> 'InvestigationState':
    """Handle structured feedback dict.

    Mutates `state` in place to apply the feedback action, then
    resets detection-side fields (``results``, ``consensus``,
    ``analysis``, ``quality``) so the next ``run()`` starts clean.
    Recognized actions:

    - ``'adjust_contamination'``: rewrite the ``contamination``
      param on every plan to ``feedback['value']``.
    - ``'exclude'``: drop plans whose detector name appears in
      ``feedback['detectors']``. Re-plans (with the excluded
      detectors as a constraint) when the result is empty.
    - ``'include'``: append plans for detectors in
      ``feedback['detectors']`` that are not already present, are
      shipped or experimental in the KB, and fit under the v1 cap of
      three plans.
    - ``'rerun'``: keep plans unchanged; signals the agent to run
      the same plan again.

    Unknown actions raise ``ValueError`` via
    `validate_structured_feedback` (called as the first step).

    Parameters
    ----------
    state : InvestigationState
        Mutable session state. Modified in place.
    feedback : dict
        Structured feedback. Must contain ``'action'``; other keys
        depend on the action.
    kb : KnowledgeBase
        Knowledge-base instance, used for algorithm lookup on
        ``'include'``.
    plan_detection_fn : callable
        Planner used for the ``'exclude'`` re-plan branch. Called as
        ``plan_detection_fn(profile, constraints={...})``.
    make_plan_fn : callable
        Factory matching `make_plan`. Called by the ``'include'``
        branch to build new plans.

    Returns
    -------
    InvestigationState
        The same `state` object, mutated.
    """
    validate_structured_feedback(feedback)
    action = feedback.get('action', '')
    state.iteration += 1

    if action == 'adjust_contamination':
        value = feedback['value']
        for p in state.plans:
            params = dict(p.get('params', {}))
            params['contamination'] = value
            p['params'] = params
        detail = 'Adjusted contamination to %.3f' % value

    elif action == 'exclude':
        to_exclude = set(feedback.get('detectors', []))
        state.plans = [
            p for p in state.plans
            if p['detector_name'] not in to_exclude]
        if not state.plans:
            # Re-plan without excluded detectors
            result = plan_detection_fn(
                state.profile,
                constraints={'exclude_detectors': list(to_exclude)})
            state.plans = [result]
            for alt in result.get('alternatives', []):
                if alt.get('detector_name'):
                    state.plans.append(alt)
        detail = 'Excluded: %s' % ', '.join(to_exclude)

    elif action == 'include':
        to_include = feedback.get('detectors', [])
        existing = {p['detector_name'] for p in state.plans}
        added, already, capped = [], [], []
        for name in to_include:
            if name in existing:
                already.append(name)
            elif len(state.plans) >= 3:
                capped.append(name)
            else:
                algo = kb.get_algorithm(name)
                if algo and algo.get('status') in (
                        'shipped', 'experimental'):
                    state.plans.append(make_plan_fn(
                        detector_name=name, params={},
                        reason='Added by user', confidence=0.5))
                    added.append(name)
        parts = []
        if added:
            parts.append('Included: %s' % ', '.join(added))
        if already:
            parts.append('Already present: %s'
                         % ', '.join(already))
        if capped:
            parts.append('Could not add %s (v1 cap: 3)'
                         % ', '.join(capped))
        detail = '. '.join(parts) if parts else 'No changes'

    elif action == 'rerun':
        detail = 'Re-running same plan'

    state.phase = 'planned'
    state.results = []
    state.consensus = None
    state.analysis = None
    state.quality = None
    state.next_action = {
        'action': 'run',
        'reason': 'Plan adjusted. ' + detail,
        'adjustment': detail,
    }
    state.history.append(_make_history_entry(
        'planned', 'iterate', state.iteration, detail))
    return state


@dataclass(frozen=True)
class _NLPattern:
    """A pattern that matches user feedback and produces a structured action."""
    pattern: re.Pattern[str]
    confidence: float
    builder: Callable[[object, str], dict]


def _build_exclude_action(state: 'InvestigationState', feedback_lower: str) -> dict:
    for r in state.results:
        name = r.get('detector_name', '')
        if name and name.lower() in feedback_lower:
            return {'action': 'exclude', 'detectors': [name]}
    return {'action': 'exclude', 'detectors': []}


def _build_decrease_contamination(state: 'InvestigationState', _feedback_lower: str) -> dict:
    current = (state.plans[0].get('params', {}).get('contamination', 0.1)
               if state.plans else 0.1)
    return {
        'action': 'adjust_contamination',
        'value': adjust_contamination_down(current),
    }


def _build_increase_contamination(state: 'InvestigationState', _feedback_lower: str) -> dict:
    current = (state.plans[0].get('params', {}).get('contamination', 0.1)
               if state.plans else 0.1)
    return {
        'action': 'adjust_contamination',
        'value': adjust_contamination_up(current),
    }


def _build_rerun(_state: 'InvestigationState', _feedback_lower: str) -> dict:
    return {'action': 'rerun'}


_NL_PATTERNS: list[_NLPattern] = [
    _NLPattern(re.compile(r'\b(without|exclude)\b'), 0.9, _build_exclude_action),
    _NLPattern(re.compile(r'\b(false positive|too many)\b'), 0.7, _build_decrease_contamination),
    _NLPattern(re.compile(r'\b(missed|false negative)\b'), 0.7, _build_increase_contamination),
    _NLPattern(re.compile(r'\b(rerun|again)\b'), 0.9, _build_rerun),
]


def parse_nl_to_structured(state: 'InvestigationState', feedback: str) -> tuple[dict, float]:
    """Match feedback against the pattern table; return (proposed, confidence).

    Parameters
    ----------
    state : InvestigationState
    feedback : str

    Returns
    -------
    tuple of (dict, float)
        The proposed structured-feedback dict (action + fields) and the
        confidence score in [0, 1]. A confidence at or above
        `_NL_HIGH_CONFIDENCE_THRESHOLD` triggers auto-apply.
    """
    feedback_lower = feedback.lower()
    for entry in _NL_PATTERNS:
        if entry.pattern.search(feedback_lower):
            proposed = entry.builder(state, feedback_lower)
            if (entry.builder is _build_exclude_action
                    and not proposed['detectors']):
                return proposed, 0.3
            return proposed, entry.confidence
    return {'action': 'rerun'}, 0.0


def apply_nl_feedback(
        state: 'InvestigationState',
        feedback: str,
        kb: 'KnowledgeBase',
        plan_detection_fn: Callable,
        make_plan_fn: Callable) -> 'InvestigationState':
    """Parse natural-language feedback into a structured action.

    Maps a free-text string to a structured-feedback dict using a
    small set of keyword heuristics, then dispatches to
    `apply_structured_feedback` when confident.

    - ``'without' | 'exclude'`` plus a known detector name (case
      insensitive): action ``'exclude'`` with that detector
      (confidence 0.9). Without a detector match, falls back to an
      ``'exclude'`` shell with confidence 0.3.
    - ``'false positive' | 'too many'``: ``'adjust_contamination'``
      to half the current value, floored at 0.01 (confidence 0.7).
    - ``'missed' | 'false negative'``: ``'adjust_contamination'`` to
      1.5 times the current value, capped at 0.5 (confidence 0.7).
    - ``'rerun' | 'again'``: ``'rerun'`` (confidence 0.9).
    - Otherwise: ``'rerun'`` with confidence 0.0.

    When confidence is at least 0.8, applies the action immediately
    via `apply_structured_feedback`. Otherwise sets
    ``next_action`` to ``confirm_with_user`` so the caller can ask
    for confirmation and appends an entry to ``state.history``.

    Parameters
    ----------
    state : InvestigationState
        Mutable session state. Modified in place.
    feedback : str
        Free-form natural-language feedback from the user.
    kb : KnowledgeBase
        Knowledge-base instance, forwarded to
        `apply_structured_feedback` on high-confidence dispatch.
    plan_detection_fn : callable
        Planner forwarded to `apply_structured_feedback`.
    make_plan_fn : callable
        Plan factory forwarded to `apply_structured_feedback`.

    Returns
    -------
    InvestigationState
        The same `state` object, mutated.
    """
    proposed, confidence = parse_nl_to_structured(state, feedback)
    if confidence >= _NL_HIGH_CONFIDENCE_THRESHOLD:
        return apply_structured_feedback(
            state, proposed, kb, plan_detection_fn, make_plan_fn)

    state.next_action = {
        'action': 'confirm_with_user',
        'reason': f'Interpreted "{feedback}" as: {proposed.get("action", "?")} '
                  f'(confidence={confidence:.1f}).',
        'suggestion': f'Proposed: {proposed}. Proceed?',
        'proposed_change': proposed,
    }
    state.history.append(_make_history_entry(
        state.phase, 'iterate_nl', state.iteration,
        f'NL feedback: "{feedback}" -> confidence={confidence:.1f}'))
    return state
