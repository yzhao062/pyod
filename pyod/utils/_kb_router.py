"""KB rule engine for ADEngine planning.

Extracts the `_evaluate_rules` / `_rule_matches` / `_eval_condition`
chain plus `_make_plan` and `_suggest_alternative` from
`pyod.utils.ad_engine.ADEngine` in 2026-05. Not part of the public API.
"""


def evaluate_rules(profile, priority, kb):
    """Evaluate routing rules against a data profile.

    Walks every rule in the KB's routing-rules table; for each rule
    whose conditions match the profile, attaches the rule's
    ``reason`` and ``evidence`` fields to each recommendation as the
    private keys ``_reason`` and ``_evidence``. When the same
    detector appears in multiple matched rules, keeps the
    recommendation with the highest ``confidence``. The final list
    is sorted by descending confidence.

    Parameters
    ----------
    profile : dict
        Data profile from `profile_data()`. Provides fields like
        ``'n_samples'``, ``'n_features'``, ``'modality'``, etc.
    priority : str
        Caller-supplied priority axis (e.g., ``'speed'``,
        ``'accuracy'``, ``'memory'``). Matched against rule
        conditions whose ``field == 'priority'``.
    kb : KnowledgeBase
        Knowledge-base instance exposing ``routing_rules``.

    Returns
    -------
    list of dict
        Recommendation dicts sorted by descending confidence. Each
        dict has keys ``'detector'``, ``'confidence'``, plus rule
        context under ``'_reason'`` and ``'_evidence'``. Empty when
        no rule matches.
    """
    rules = kb.routing_rules.get('rules', [])
    all_recs = []

    for rule in rules:
        if rule_matches(rule, profile, priority):
            reason = rule.get('reason', '')
            evidence = rule.get('evidence', [])
            for rec in rule.get('recommendations', []):
                enriched = dict(rec)
                enriched['_reason'] = reason
                enriched['_evidence'] = evidence
                all_recs.append(enriched)

    seen = {}
    for rec in all_recs:
        name = rec['detector']
        if name not in seen or \
                rec.get('confidence', 0) > seen[name].get('confidence', 0):
            seen[name] = rec
    return sorted(seen.values(),
                  key=lambda r: r.get('confidence', 0),
                  reverse=True)


def rule_matches(rule, profile, priority):
    """Check if all conditions in a rule match the profile.

    Each rule's ``conditions`` list is treated as a logical AND. A
    condition's ``field`` is read from `priority` when equal to
    ``'priority'``, otherwise from `profile`. A missing profile field
    fails the match.

    Parameters
    ----------
    rule : dict
        Rule dict with a ``'conditions'`` list. Each condition has
        keys ``'field'``, ``'op'``, and ``'value'``.
    profile : dict
        Data profile from `profile_data()`.
    priority : str
        Caller-supplied priority axis.

    Returns
    -------
    bool
        ``True`` only if every condition evaluates to true and no
        referenced profile field is missing; ``False`` otherwise.
    """
    for cond in rule.get('conditions', []):
        field = cond['field']
        op = cond['op']
        value = cond['value']

        if field == 'priority':
            actual = priority
        else:
            actual = profile.get(field)

        if actual is None:
            return False
        if not eval_condition(actual, op, value):
            return False
    return True


def eval_condition(actual, op, value):
    """Evaluate a single condition predicate.

    Supported operators: ``'eq'`` (equality), ``'lt'``, ``'lte'``,
    ``'gt'``, ``'gte'`` (numeric comparisons after `float` coercion),
    and ``'in'`` (membership test). Unknown operators evaluate to
    ``False``.

    Parameters
    ----------
    actual : object
        Value taken from the profile or priority.
    op : str
        Operator name. One of ``'eq'``, ``'lt'``, ``'lte'``,
        ``'gt'``, ``'gte'``, ``'in'``.
    value : object
        Right-hand operand. For numeric ops, must be coercible to
        `float`. For ``'in'``, must be a container.

    Returns
    -------
    bool
        Result of the predicate. Returns ``False`` when `op` is not
        recognized.
    """
    if op == 'eq':
        return actual == value
    if op == 'lt':
        return float(actual) < float(value)
    if op == 'lte':
        return float(actual) <= float(value)
    if op == 'gt':
        return float(actual) > float(value)
    if op == 'gte':
        return float(actual) >= float(value)
    if op == 'in':
        return actual in value
    return False


def make_plan(detector_name, params=None, preset=None,
              reason='', evidence=None, confidence=0.5,
              alternatives=None, note=None):
    """Construct a closed-schema DetectionPlan dict.

    The closed schema means downstream code can rely on the keys
    ``'detector_name'``, ``'params'``, ``'reason'``, ``'evidence'``,
    ``'confidence'``, and ``'alternatives'`` always being present.
    Optional keys (``'preset'``, ``'note'``) are added only when the
    caller supplies a non-empty value.

    Parameters
    ----------
    detector_name : str
        Class name of the detector (e.g., ``'IForest'``, ``'KNN'``).
    params : dict, optional
        Constructor kwargs for the detector. Defaults to ``{}``.
    preset : str, optional
        Factory preset name (e.g., ``'for_text'``). Only included in
        the returned plan when non-empty.
    reason : str, optional
        Human-readable explanation for the choice. Defaults to ``''``.
    evidence : list, optional
        Supporting evidence items (rule IDs, profile fields, etc.).
        Defaults to ``[]``.
    confidence : float, optional
        Confidence in the plan, in [0, 1]. Defaults to 0.5.
    alternatives : list of dict, optional
        Backup plans, each itself a DetectionPlan dict. Defaults to
        ``[]``.
    note : str, optional
        Free-form note. Only included in the returned plan when
        non-empty.

    Returns
    -------
    dict
        DetectionPlan dict with closed-schema keys
        ``'detector_name'``, ``'params'``, ``'reason'``,
        ``'evidence'``, ``'confidence'``, ``'alternatives'``, and
        optional ``'preset'`` and ``'note'``.
    """
    plan = {
        'detector_name': detector_name,
        'params': params or {},
        'reason': reason,
        'evidence': evidence or [],
        'confidence': confidence,
        'alternatives': alternatives or [],
    }
    if preset:
        plan['preset'] = preset
    if note:
        plan['note'] = note
    return plan


def suggest_alternative(result, kb, make_plan_fn):
    """Suggest an alternative detector different from the current one.

    Resolution order:

    1. First entry in the current plan's ``alternatives`` list whose
       ``detector_name`` differs from the current detector.
    2. First detector from a hard-coded fallback chain (``IForest``,
       ``ECOD``, ``KNN``, ``LOF``, ``HBOS``, ``COPOD``, ``CBLOF``)
       that is not the current one and is marked ``shipped`` in the
       knowledge base.
    3. ``IForest`` as a last-resort default.

    Parameters
    ----------
    result : dict
        Detector result dict containing the current ``'plan'``.
    kb : KnowledgeBase
        Knowledge-base instance, used to check algorithm status.
    make_plan_fn : callable
        Factory matching the signature of `make_plan`. Called for the
        fallback-chain and last-resort branches.

    Returns
    -------
    dict
        DetectionPlan dict for an alternative detector.
    """
    current = result['plan'].get('detector_name', '')

    alternatives = result['plan'].get('alternatives', [])
    for alt in alternatives:
        if alt.get('detector_name') and alt['detector_name'] != current:
            return alt

    fallback_order = ['IForest', 'ECOD', 'KNN', 'LOF', 'HBOS',
                      'COPOD', 'CBLOF']
    for name in fallback_order:
        if name != current:
            algo = kb.get_algorithm(name)
            if algo and algo.get('status') == 'shipped':
                return make_plan_fn(
                    detector_name=name, params={},
                    reason='Alternative to %s' % current,
                    evidence=[], confidence=0.6)

    return make_plan_fn(
        detector_name='IForest', params={},
        reason='Default fallback', evidence=[], confidence=0.5)
