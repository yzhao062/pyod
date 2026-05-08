"""ADEngine iterate() feedback parsing.

Extracts `_iterate_structured` and `_iterate_nl` from
`pyod.utils.ad_engine.ADEngine` in 2026-05.
"""

from .investigation import _make_history_entry


def apply_structured_feedback(
        state, feedback, kb, plan_detection_fn, make_plan_fn):
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

    Unknown actions set ``next_action`` to ``confirm_with_user`` and
    return early without resetting detection state.

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

    else:
        state.next_action = {
            'action': 'confirm_with_user',
            'reason': 'Unknown action: %s' % action,
        }
        return state

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


def apply_nl_feedback(
        state, feedback, kb, plan_detection_fn, make_plan_fn):
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
    lower = feedback.lower()
    proposed = None
    confidence = 0.0

    # High-confidence patterns
    if 'without' in lower or 'exclude' in lower:
        # Try to extract detector name
        for r in state.results:
            name = r.get('detector_name', '')
            if name.lower() in lower:
                proposed = {'action': 'exclude',
                            'detectors': [name]}
                confidence = 0.9
                break
        if proposed is None and ('without' in lower
                                 or 'exclude' in lower):
            proposed = {'action': 'exclude', 'detectors': []}
            confidence = 0.3

    elif ('false positive' in lower or 'too many' in lower):
        current = state.plans[0].get('params', {}).get(
            'contamination', 0.1) if state.plans else 0.1
        proposed = {'action': 'adjust_contamination',
                    'value': max(current * 0.5, 0.01)}
        confidence = 0.7

    elif ('missed' in lower or 'false negative' in lower):
        current = state.plans[0].get('params', {}).get(
            'contamination', 0.1) if state.plans else 0.1
        proposed = {'action': 'adjust_contamination',
                    'value': min(current * 1.5, 0.5)}
        confidence = 0.7

    elif 'rerun' in lower or 'again' in lower:
        proposed = {'action': 'rerun'}
        confidence = 0.9

    if proposed is None:
        proposed = {'action': 'rerun'}
        confidence = 0.0

    if confidence >= 0.8:
        return apply_structured_feedback(
            state, proposed, kb, plan_detection_fn, make_plan_fn)

    # Low confidence → ask for confirmation
    state.next_action = {
        'action': 'confirm_with_user',
        'reason': 'Interpreted "%s" as: %s (confidence=%.1f).'
                  % (feedback, proposed.get('action', '?'),
                     confidence),
        'suggestion': 'Proposed: %s. Proceed?' % str(proposed),
        'proposed_change': proposed,
    }
    state.history.append(_make_history_entry(
        state.phase, 'iterate_nl', state.iteration,
        'NL feedback: "%s" -> confidence=%.1f'
        % (feedback, confidence)))
    return state
