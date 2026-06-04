# -*- coding: utf-8 -*-
"""ADEngine: anomaly detection lifecycle engine.

Handles data profiling, detection planning, detector construction,
and knowledge queries. Works as a standalone Python API (no LLM
required) or as the backend for MCP/agent interfaces.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .knowledge import KnowledgeBase
from pyod.utils._quality_metrics import (
    compute_consensus,
    compute_feature_importance,
    compute_quality,
    feature_contributions,
    label_metrics,
    select_best_detector,
)
from pyod.utils._kb_router import (
    evaluate_rules,
    make_plan,
    suggest_alternative,
)
from pyod.utils._detector_factory import (
    build_detector_from_plan,
)
from pyod.utils._nl_feedback import (
    adjust_contamination_down,
    adjust_contamination_up,
    apply_nl_feedback,
    apply_structured_feedback,
)

if TYPE_CHECKING:
    from pyod.utils.investigation import InvestigationState

logger = logging.getLogger(__name__)


class ADEngine:
    """Anomaly detection lifecycle engine.

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge base directory. If None, uses bundled.
    random_state : int or None, optional
        Random seed forwarded to every detector that declares an
        explicit ``random_state`` parameter when the engine instantiates
        it from a plan. Detectors without ``random_state`` in their
        signature (e.g., ABOD, KNN, LOF, SOD) are deterministic by
        construction (distance, angle, or density based, with no internal
        sampling) and need no seed. With this set, the shallow-detector
        pipeline is reproducible: a run-to-run audit of the shipped
        shallow detectors found every one either honors the seed or is
        deterministic by construction, with no nondeterministic cases.
        Deep detectors additionally depend on framework-level seeding
        (e.g., ``torch.manual_seed``). Set this to a fixed integer for
        byte-identical flagged sets across re-runs on the same input.
    """

    def __init__(self, knowledge_dir: str | None = None,
                 random_state: int | None = None) -> None:
        self.kb = KnowledgeBase(knowledge_dir=knowledge_dir)
        self.random_state = random_state

    def profile_data(self, X: Any, data_type: str | None = None) -> dict:
        """Profile the input data.

        Parameters
        ----------
        X : array-like, list, or dict
            Input data.
        data_type : str or None
            Explicit override. One of 'tabular', 'text', 'image',
            'time_series', 'multimodal', 'graph'.

        Returns
        -------
        profile : dict
        """
        if data_type is not None:
            detected_type = data_type
        else:
            detected_type = self._sniff_data_type(X)

        profile = {'data_type': detected_type}

        if detected_type == 'text':
            profile['n_samples'] = len(X)
        elif detected_type == 'image':
            profile['n_samples'] = len(X)
        elif detected_type == 'multimodal':
            first_key = next(iter(X))
            first_val = X[first_key]
            profile['n_samples'] = len(first_val)
            profile['modalities'] = list(X.keys())
        elif detected_type == 'graph':
            # PyG Data object (only supported graph input for ADEngine)
            profile['n_nodes'] = X.num_nodes
            profile['n_edges'] = X.edge_index.shape[1]
            profile['n_features'] = (
                X.x.shape[1] if X.x is not None else 0)
            profile['has_features'] = X.x is not None
            profile['n_samples'] = X.num_nodes
        else:
            # tabular or time_series
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            profile['n_samples'] = arr.shape[0]
            profile['n_features'] = arr.shape[1]
            profile['has_nan'] = bool(np.isnan(arr).any())
            profile['dtype'] = str(arr.dtype)

            n_feat = arr.shape[1]
            if n_feat <= 10:
                profile['dimensionality_class'] = 'low'
            elif n_feat <= 100:
                profile['dimensionality_class'] = 'medium'
            else:
                profile['dimensionality_class'] = 'high'

            if detected_type == 'time_series':
                profile['n_timestamps'] = arr.shape[0]
                profile['channels'] = arr.shape[1]

        return profile

    def _sniff_data_type(self, X: Any) -> str:
        """Conservative data type detection."""
        # Check for PyG Data object
        try:
            from torch_geometric.data import Data
            if isinstance(X, Data):
                return 'graph'
        except ImportError:
            pass

        if isinstance(X, dict):
            return 'multimodal'
        if isinstance(X, (list, tuple)) and len(X) > 0:
            sample = X[:min(20, len(X))]
            if all(isinstance(x, str) for x in sample):
                if self._looks_like_image_paths(sample[:5]):
                    return 'image'
                return 'text'
        return 'tabular'

    @staticmethod
    def _looks_like_image_paths(samples: list[str]) -> bool:
        """Check if string samples look like image file paths."""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif',
                      '.tiff', '.webp'}
        for s in samples:
            ext = os.path.splitext(s)[1].lower()
            if ext not in image_exts:
                return False
        return True

    def _with_contamination(self, detector_name: str,
                            params: dict) -> dict:
        """Ensure plan params expose an explicit contamination value (TA2).

        The MCP `plan_detection` -> `build_detector` chain serializes the
        plan to JSON. When `params` does not include `contamination`, the
        emitted code snippet inherits the detector class's own default,
        which is invisible to MCP-only agents. Always include a value
        sourced from the KB `default_params` when the KB confirms the
        detector accepts a `contamination` kwarg; otherwise leave params
        unchanged so we do not paper over detectors that use a different
        threshold mechanism.
        """
        if 'contamination' in params:
            return dict(params)
        algo = self.kb.get_algorithm(detector_name)
        if algo is None:
            return dict(params)
        kb_default = algo.get('default_params', {}).get('contamination')
        if kb_default is None:
            return dict(params)
        out = dict(params)
        out['contamination'] = kb_default
        return out

    def plan_detection(self, profile: dict, priority: str = 'balanced',
                       constraints: dict | None = None, *,
                       top_k: int = 3,
                       llm_client=None,
                       llm_strict: bool | None = None) -> dict:
        """Plan a detection pipeline.

        Parameters
        ----------
        profile : dict
            Output of profile_data().
        priority : str
            'speed', 'accuracy', or 'balanced'.
        constraints : dict or None
            Optional: {'exclude_detectors': [...]}
        top_k : int, default 3
            Number of detectors in the returned plan (primary + ``top_k - 1``
            alternatives). Default ``3`` preserves the v3.5.2 behaviour
            (``valid[1:3]`` produced two alternatives plus the primary).
            Values < 1 are clamped to 1.
        llm_client : callable or None, default None
            Optional ``(prompt: str) -> str`` callable (see
            :class:`pyod.utils._llm.LLMCallable`). When provided, routing
            consults the LLM with the KB context and parses its response
            into a plan via :func:`pyod.utils._llm.parse_routing_response`.
            If the LLM call or parser raises, falls back to rule routing
            with a :class:`RuntimeWarning` (see ``llm_strict``). When
            ``None`` (default), v3.5.2 rule routing is unchanged.
        llm_strict : bool or None, default None
            Per-call control for LLM-routing failure mode. ``True``
            re-raises any exception from ``llm_client`` or the response
            parser; ``False`` falls back to rule routing with a
            :class:`RuntimeWarning`; ``None`` defers to the
            ``PYOD3_LLM_STRICT`` environment variable
            (``"1"`` re-raises, anything else falls back). The explicit
            kwarg takes precedence so concurrent callers in the same
            process can choose independently.

        Returns
        -------
        plan : dict (DetectionPlan, closed schema)
        """
        constraints = constraints or {}
        top_k = max(1, int(top_k))

        if llm_client is not None:
            try:
                return self._plan_via_llm(profile, top_k, llm_client,
                                          constraints)
            except Exception as ex:  # noqa: BLE001
                if llm_strict is None:
                    import os
                    strict = os.environ.get('PYOD3_LLM_STRICT') == '1'
                else:
                    strict = bool(llm_strict)
                if strict:
                    raise
                import warnings
                warnings.warn(
                    f"plan_detection: llm_client routing failed "
                    f"({type(ex).__name__}: {ex}); falling back to "
                    "rule routing. Pass llm_strict=True (or set "
                    "PYOD3_LLM_STRICT=1) to re-raise.",
                    RuntimeWarning, stacklevel=2)

        exclude = set(constraints.get('exclude_detectors', []))

        matched = evaluate_rules(profile, priority, self.kb)

        valid = []
        for rec in matched:
            name = rec['detector']
            algo = self.kb.get_algorithm(name)
            if algo is None:
                continue
            if algo.get('status') != 'shipped':
                continue
            if name in exclude:
                continue
            valid.append(rec)

        if not valid:
            # Fallback: pick first non-excluded shipped detector
            fallback_order = ['IForest', 'ECOD', 'KNN', 'HBOS', 'LOF',
                              'COPOD', 'PCA']
            fallback_name = None
            for fb in fallback_order:
                if fb not in exclude:
                    algo = self.kb.get_algorithm(fb)
                    if algo and algo.get('status') == 'shipped':
                        fallback_name = fb
                        break
            if fallback_name is None:
                return make_plan(
                    detector_name='',
                    params={},
                    reason='No valid detector available: all candidates '
                           'excluded or no matching rule found',
                    evidence=[],
                    confidence=0.0,
                    alternatives=[],
                    note='no_valid_plan')

            return make_plan(
                detector_name=fallback_name,
                params=self._with_contamination(fallback_name, {}),
                reason='Fallback: no routing rule matched or all '
                       'candidates excluded',
                evidence=['ADBench'], confidence=0.5,
                alternatives=[], note='No specific rule matched')

        best = valid[0]
        alternatives = [make_plan(
            detector_name=r['detector'],
            params=self._with_contamination(
                r['detector'], r.get('params', {})),
            preset=r.get('preset'),
            reason=r.get('_reason', ''),
            evidence=r.get('_evidence', []),
            confidence=r.get('confidence', 0.5),
            alternatives=[]) for r in valid[1:top_k]]

        return make_plan(
            detector_name=best['detector'],
            params=self._with_contamination(
                best['detector'], best.get('params', {})),
            preset=best.get('preset'),
            reason=best.get('_reason', ''),
            evidence=best.get('_evidence', []),
            confidence=best.get('confidence', 0.7),
            alternatives=alternatives)

    # ------------------------------------------------------------------
    # Surface 1: KB exposure for caller-driven (agent / LLM) routing
    # ------------------------------------------------------------------

    def get_kb_for_routing(self, profile: dict, top_k: int = 3,
                           constraints: dict | None = None,
                           scoring_view: str = "legacy") -> dict:
        """Return a structured KB snapshot for caller-driven detector
        selection.

        This is the agent-facing companion to :meth:`plan_detection`.
        ``plan_detection`` consumes the KB through hand-coded rules and
        returns a single plan; ``get_kb_for_routing`` exposes the KB
        directly so a caller (LLM agent, MCP tool client, ...) can
        reason over each detector's strengths, weaknesses, complexity,
        and benchmark rank, then call :meth:`make_plan` to commit a
        plan.

        Parameters
        ----------
        profile : dict
            Output of :meth:`profile_data`. Must include ``data_type``;
            ``n_samples`` / ``n_features`` are passed through unchanged.
        top_k : int, default 3
            The number of detectors the caller intends to select. The KB
            snapshot itself is returned in full (filtered + sorted); the
            field is included in the returned dict so the response-format
            hint can reference it.
        constraints : dict or None, optional
            ``{'exclude_detectors': list[str], 'data_type_strict': bool}``.
            ``exclude_detectors`` is a hard filter. ``data_type_strict``
            (default ``True``) drops detectors whose KB ``data_types``
            field does not include ``profile['data_type']``.

        Returns
        -------
        dict
            ``{'task_profile': {...}, 'available_detectors': [...],
            'top_k_requested': int, 'response_format_hint': str,
            'n_available': int}``.

        Notes
        -----
        Pure function; no LLM calls, no state mutation.
        """
        if not isinstance(profile, dict):
            raise ValueError("profile must be a dict from profile_data()")
        if scoring_view not in ("legacy", "soft_kb"):
            raise ValueError(
                "scoring_view must be 'legacy' (default) or 'soft_kb'")
        top_k = max(1, int(top_k))
        constraints = constraints or {}
        exclude = set(constraints.get('exclude_detectors') or [])
        data_type_strict = constraints.get('data_type_strict', True)
        target_modality = profile.get('data_type', 'tabular')

        catalog = self.list_detectors(data_type=None, status='shipped')
        available: list[dict] = []
        for entry in catalog:
            name = entry.get('name') if isinstance(entry, dict) else str(entry)
            if name in exclude:
                continue
            dts = entry.get('data_types') or []
            modality_match = (target_modality in dts) if dts else True
            if data_type_strict and not modality_match:
                continue
            complexity = entry.get('complexity') or {}
            available.append({
                'name': name,
                'category': entry.get('category', 'unknown'),
                'complexity_time': complexity.get('time'),
                'complexity_space': complexity.get('space'),
                'strengths': entry.get('strengths') or [],
                'weaknesses': entry.get('weaknesses') or [],
                'best_for': entry.get('best_for'),
                'avoid_when': entry.get('avoid_when'),
                'benchmark_rank': entry.get('benchmark_rank') or {},
                'modality_match': modality_match,
            })

        # Modality-aware benchmark-rank keys. Each modality lists its
        # preferred KB rank fields in priority order; the first non-None
        # value sets the sort key. `ADBench_overall` is the universal
        # fallback because the KB ships rank for nearly every tabular
        # detector there. Detectors missing every key sort last (999).
        _MODALITY_RANK_KEYS = {
            'tabular': ['ADBench_overall'],
            'time_series': ['TSB_AD_overall', 'TSB_AD_overall_iforest',
                            'ADBench_overall'],
            'timeseries': ['TSB_AD_overall', 'TSB_AD_overall_iforest',
                           'ADBench_overall'],
            'graph': ['BOND_deep', 'BOND_overall', 'ADBench_overall'],
            'text': ['NLP_ADBench_overall', 'ADBench_overall'],
            'image': ['MVTec_overall', 'ADBench_overall'],
            'synthetic': ['ADBench_overall'],
        }
        rank_key_candidates = _MODALITY_RANK_KEYS.get(
            str(target_modality).lower(),
            [f"{str(target_modality).title()}_overall", 'ADBench_overall'])

        def _rank(d):
            br = d.get('benchmark_rank') or {}
            for k in rank_key_candidates:
                v = br.get(k)
                if v is not None:
                    return v
            return 999

        # Stamp the resolved (rank, rank_key) on each entry so downstream
        # consumers (e.g., build_routing_prompt) can render the modality-
        # specific rank without re-doing the lookup. None when no rank
        # field is present in the KB for this detector under this modality.
        for d in available:
            br = d.get('benchmark_rank') or {}
            resolved = None
            resolved_key = None
            for k in rank_key_candidates:
                v = br.get(k)
                if v is not None:
                    resolved = v
                    resolved_key = k
                    break
            d['resolved_rank'] = resolved
            d['resolved_rank_key'] = resolved_key

        available.sort(key=lambda d: (_rank(d), d['name']))

        # Soft-KB view (opt-in): attach calibrated within-modality scores and
        # re-sort by them. The legacy default path above is left byte-identical
        # because the published section 5.4 evidence depends on it (see the
        # golden test pyod/test/test_kb_routing_golden.py).
        if scoring_view == "soft_kb":
            self._attach_soft_kb(available, target_modality)

        # Strip non-JSON-safe fields from the profile copy
        profile_safe = {k: v for k, v in profile.items() if k != 'data'}

        result = {
            'task_profile': profile_safe,
            'available_detectors': available,
            'top_k_requested': top_k,
            'response_format_hint': (
                "To commit your selection, call ADEngine.make_plan with "
                "detector_choices=['detName1', ...] (ordered list of "
                f"top-{top_k} names from available_detectors[*].name; "
                "case-sensitive) and justifications=['why1', ...] "
                "(parallel list, one short sentence each)."
            ),
            'n_available': len(available),
        }
        if scoring_view == "soft_kb":
            meta = self.kb.kb_scores_meta or {}
            result['kb_version'] = meta.get('kb_version', 'unknown')
            result['kb_scoring_method'] = meta.get('kb_scoring_method', 'unknown')
        return result

    # Modality name as used in the soft-KB scores artifact (kb_scores.json).
    _KB_MODALITY = {
        'tabular': 'tabular', 'time_series': 'timeseries',
        'timeseries': 'timeseries', 'graph': 'graph', 'text': 'text',
        'image': 'image', 'synthetic': 'synthetic',
    }

    def _attach_soft_kb(self, available: list, target_modality) -> None:
        """Attach soft-KB within-modality scores in place and re-sort
        ``available`` by ``kb_score`` (descending; unscored detectors last).

        Scores come from the redesigned KB (``_raw/kb_scores.json``, v3.6).
        Strictly within modality: a detector with no score for this modality
        gets ``kb_score=None`` rather than a cross-modality fallback.
        """
        kb_mod = self._KB_MODALITY.get(
            str(target_modality).lower(), str(target_modality).lower())
        for d in available:
            roll = self.kb.get_kb_scores(d['name'], kb_mod)
            if roll:
                d['kb_score'] = roll.get('kb_score')
                d['kb_interval'] = [roll.get('ci_low'), roll.get('ci_high')]
                d['kb_coverage_n'] = roll.get('coverage_n')
                d['kb_eligible_n'] = roll.get('eligible_n')
                d['kb_failure_n'] = roll.get('failure_n')
                d['kb_missing_n'] = roll.get('missing_n')
                d['kb_effective_n'] = roll.get('effective_n')
                d['kb_metric_scope'] = roll.get('metric_scopes')
                d['kb_source_benchmark'] = roll.get('source_benchmark')
                d['kb_fallback_level'] = 'modality'
            else:
                d['kb_score'] = None
                d['kb_interval'] = None
                d['kb_coverage_n'] = 0
                d['kb_eligible_n'] = 0
                d['kb_failure_n'] = 0
                d['kb_missing_n'] = 0
                d['kb_effective_n'] = 0
                d['kb_metric_scope'] = None
                d['kb_source_benchmark'] = None
                d['kb_fallback_level'] = None
        available.sort(key=lambda d: (
            0 if d.get('kb_score') is not None else 1,
            -(d.get('kb_score') or 0.0),
            d['name']))
        # Stamp a 1-based rank among scored detectors in this (post-sort)
        # order; unscored detectors get None. Lets downstream consumers cite
        # the within-modality rank without re-sorting.
        rank = 0
        for d in available:
            if d.get('kb_score') is not None:
                rank += 1
                d['kb_score_rank'] = rank
            else:
                d['kb_score_rank'] = None

    def make_plan(self, detector_choices: list,
                  justifications: list | None = None,
                  params: list | None = None) -> dict:
        """Commit a caller-driven detector plan and return a DetectionPlan.

        Companion to :meth:`get_kb_for_routing`. The caller (LLM agent,
        rule engine, human script) selects ``len(detector_choices)``
        detectors and this method validates names against the KB, fills
        per-detector defaults, and packages the result as a
        :func:`pyod.utils._kb_router.make_plan`-shaped dict so existing
        consumers (``build_detector``, ``run``, downstream MCP clients)
        keep working unchanged.

        Parameters
        ----------
        detector_choices : list of str
            Ordered list of detector class names. ``detector_choices[0]``
            is the primary; the rest become ``alternatives`` in plan
            order. Length must be >= 1. Names must match KB entries
            (case-sensitive) with ``status='shipped'``; otherwise
            ``ValueError`` is raised.
        justifications : list of str, optional
            Parallel to ``detector_choices``. One short sentence per
            choice. ``None`` is accepted and yields autogenerated
            reasons.
        params : list of dict, optional
            Parallel to ``detector_choices``. Per-detector constructor
            kwargs. ``None`` -> KB defaults overlaid with the
            engine's contamination resolution.

        Returns
        -------
        dict
            Closed-schema DetectionPlan: ``{'detector_name',
            'params', 'reason', 'evidence', 'confidence',
            'alternatives', 'note'}``.

        Raises
        ------
        ValueError
            If ``detector_choices`` is empty or any name is unknown /
            not ``status='shipped'`` in the KB.
        """
        if not detector_choices:
            raise ValueError(
                "detector_choices must be non-empty; got an empty list")
        if not isinstance(detector_choices, list):
            raise ValueError(
                "detector_choices must be a list of strings; "
                f"got {type(detector_choices).__name__}")

        justifications = list(justifications or [])
        params_list = list(params or [])
        while len(justifications) < len(detector_choices):
            justifications.append('')
        while len(params_list) < len(detector_choices):
            params_list.append({})

        unknown = []
        not_shipped = []
        for name in detector_choices:
            algo = self.kb.get_algorithm(name)
            if algo is None:
                unknown.append(name)
                continue
            if algo.get('status') != 'shipped':
                not_shipped.append(name)
        if unknown:
            raise ValueError(
                "Unknown detector name(s) (case-sensitive). Names must "
                "match KB entries from ADEngine.list_detectors(): "
                f"{unknown!r}")
        if not_shipped:
            raise ValueError(
                f"Detector(s) not shipped (cannot be built): {not_shipped!r}")

        primary = detector_choices[0]
        primary_params = self._with_contamination(
            primary, params_list[0] or {})
        alternatives = []
        for i, det in enumerate(detector_choices[1:], start=1):
            alt_params = self._with_contamination(det, params_list[i] or {})
            alt_reason = (justifications[i] or
                          'caller-selected via make_plan')
            alternatives.append(make_plan(
                detector_name=det,
                params=alt_params,
                reason=alt_reason,
                evidence=['caller_selection'],
                confidence=0.5,
                alternatives=[]))

        primary_reason = (justifications[0] or
                          'caller-selected via make_plan')
        return make_plan(
            detector_name=primary,
            params=primary_params,
            reason=primary_reason,
            evidence=['caller_selection'],
            confidence=0.7,
            alternatives=alternatives,
            note='caller-driven via make_plan')

    def _plan_via_llm(self, profile: dict, top_k: int, llm_client,
                      constraints: dict | None = None) -> dict:
        """Route via an LLM client (internal; see plan_detection)."""
        from ._llm import (
            RoutingParseError,
            build_routing_prompt,
            parse_routing_response,
        )
        kb_context = self.get_kb_for_routing(
            profile, top_k=top_k, constraints=constraints or {})
        prompt = build_routing_prompt(kb_context, top_k=top_k)
        response = llm_client(prompt)
        detector_choices, justifications = parse_routing_response(
            response, self.kb, top_k=top_k)
        # LLM output is untrusted: enforce the constrained KB context
        # (exclude_detectors + data_type_strict) after parsing. Without
        # this, a hostile or buggy client could return an excluded or
        # modality-mismatched detector and get an LLM-sourced plan.
        # parse_routing_response only validates against the global KB.
        allowed = {d['name'] for d in kb_context.get(
            'available_detectors', [])}
        blocked = [name for name in detector_choices if name not in allowed]
        if blocked:
            raise RoutingParseError(
                "LLM selected detector(s) outside the constrained KB "
                f"context: {blocked!r}. The constrained context "
                f"excluded {sorted(constraints.get('exclude_detectors') or [])!r}.")
        plan = self.make_plan(
            detector_choices=detector_choices,
            justifications=justifications)
        # Tag the plan so downstream code can distinguish LLM-sourced
        # plans from caller-driven or rule-driven ones.
        plan['note'] = 'llm-driven via plan_detection(llm_client=...)'
        plan['evidence'] = ['llm_routing']
        return plan

    # ------------------------------------------------------------------
    # Detector construction
    # ------------------------------------------------------------------

    def build_detector(self, plan: dict) -> Any:
        """Build and return an unfitted detector from a plan.

        Parameters
        ----------
        plan : dict (DetectionPlan)
            Output of plan_detection().

        Returns
        -------
        detector : BaseDetector
        """
        return build_detector_from_plan(plan, self.kb,
                                        random_state=self.random_state)

    # ------------------------------------------------------------------
    # One-shot detection
    # ------------------------------------------------------------------

    def detect(self, X_train: Any, X_test: Any = None,
               data_type: str | None = None,
               priority: str = 'balanced') -> dict:
        """One-shot anomaly detection: profile -> plan -> run -> analyze.

        Parameters
        ----------
        X_train : array-like
            Training data.
        X_test : array-like or None
            Optional test data.
        data_type : str or None
            Explicit data type override.
        priority : str
            'speed', 'accuracy', or 'balanced'.

        Returns
        -------
        result : dict
            Output of run_detection() enriched with analysis.
            Compatible with all Tier B methods (analyze_results,
            explain_findings, suggest_next_step, generate_report).
        """
        profile = self.profile_data(X_train, data_type=data_type)
        plan = self.plan_detection(profile, priority=priority)
        result = self.run_detection(X_train, plan, X_test=X_test)
        result['analysis'] = self.analyze_results(result, X=X_train)
        return result

    # ------------------------------------------------------------------
    # Structured detection
    # ------------------------------------------------------------------

    def run_detection(self, X_train: Any, plan: dict,
                      X_test: Any = None) -> dict:
        """Execute a detection plan.

        Parameters
        ----------
        X_train : array-like
            Training data.
        plan : dict (DetectionPlan)
            Output of plan_detection().
        X_test : array-like or None
            Optional test data.

        Returns
        -------
        result : dict
            Keys: 'plan', 'scores_train', 'labels_train', 'threshold',
            'n_anomalies', 'anomaly_ratio', 'detector', 'runtime_seconds',
            'score_summary'. If X_test: also 'scores_test', 'labels_test'.
        """
        import time
        start = time.time()

        clf = self.build_detector(plan)
        clf.fit(X_train)

        elapsed = time.time() - start

        scores = clf.decision_scores_
        labels = clf.labels_
        n_anomalies = int(labels.sum())

        result = {
            'plan': plan,
            'scores_train': scores,
            'labels_train': labels,
            'threshold': float(clf.threshold_),
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / len(labels),
            'detector': clf,
            'runtime_seconds': elapsed,
            'score_summary': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
            },
        }

        if X_test is not None:
            try:
                result['scores_test'] = clf.decision_function(X_test)
                result['labels_test'] = clf.predict(X_test)
            except NotImplementedError:
                result['scores_test'] = None
                result['labels_test'] = None

        return result

    # ------------------------------------------------------------------
    # Result analysis
    # ------------------------------------------------------------------

    def analyze_results(self, result: dict, X: Any = None,
                        top_k: int = 10) -> dict:
        """Analyze detection results.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        X : array-like or None
            Original training data for feature-level analysis.
        top_k : int
            Number of top anomalies to return.

        Returns
        -------
        analysis : dict
        """
        top_k = max(0, int(top_k))
        scores = result['scores_train']
        labels = result['labels_train']
        n_anomalies = int(labels.sum())

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_anomalies = [{'index': int(i), 'score': float(scores[i])}
                         for i in top_indices]

        score_dist = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
        }

        detector_name = result['plan'].get('detector_name', 'unknown')
        ratio = n_anomalies / len(labels) if len(labels) > 0 else 0
        summary = (
            "%d anomalies detected out of %d samples (%.1f%%) "
            "using %s. Scores range from %.4f to %.4f "
            "(mean=%.4f, std=%.4f). Threshold: %.4f."
            % (n_anomalies, len(labels), ratio * 100,
               detector_name,
               score_dist['min'], score_dist['max'],
               score_dist['mean'], score_dist['std'],
               result['threshold']))

        analysis = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': ratio,
            'score_distribution': score_dist,
            'top_anomalies': top_anomalies,
            'summary': summary,
        }

        if X is not None:
            fi = compute_feature_importance(result, X)
            if fi is not None:
                analysis['feature_importance'] = fi

        return analysis

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_findings(self, result: dict,
                         indices: list[int] | None = None,
                         top_k: int = 5, X: Any = None,
                         feature_names: list[str] | None = None
                         ) -> list[dict]:
        """Explain why specific samples were flagged as anomalies.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        indices : list of int or None
            Specific sample indices. If None, explains top-k.
        top_k : int
            Number of top anomalies to explain if indices is None.
        X : array-like or None
            Original data for feature-level explanations.
        feature_names : list of str or None
            Optional feature labels in column order, threaded through
            to ``feature_contributions`` so each contributing feature
            has a human-readable name. When omitted, names default to
            ``f'feature_{column_index}'``.

        Returns
        -------
        explanations : list of dict
            Each entry has ``'index'``, ``'score'``, ``'percentile'``,
            ``'label'``, ``'narrative'``. When ``X`` is provided, also
            includes ``'contributing_features'``: a list of dicts with
            ``'feature'``, ``'name'``, ``'value'``, ``'mean'``,
            ``'z_score'``, and ``'direction'``.
        """
        top_k = max(0, int(top_k))
        scores = result['scores_train']

        if indices is None:
            indices = list(np.argsort(scores)[::-1][:top_k])

        # Validate indices: must be integers (not bool) and in range
        n_samples = len(scores)
        validated = []
        for idx in indices:
            if isinstance(idx, bool):
                continue
            if not isinstance(idx, (int, np.integer)):
                continue
            if 0 <= idx < n_samples:
                validated.append(int(idx))
        indices = validated

        explanations = []
        for idx in indices:
            score = float(scores[idx])
            pctile = float(np.mean(scores <= score) * 100)
            label = 'anomaly' if score > result['threshold'] else 'normal'

            narrative = (
                "Sample %d has anomaly score %.4f (percentile: %.1f%%), "
                "classified as %s (threshold: %.4f)."
                % (idx, score, pctile, label, result['threshold']))

            entry = {
                'index': int(idx),
                'score': score,
                'percentile': pctile,
                'label': label,
                'narrative': narrative,
            }

            if X is not None:
                contribs = feature_contributions(
                    X, idx, scores, feature_names=feature_names)
                if contribs is not None:
                    entry['contributing_features'] = contribs

            explanations.append(entry)

        return explanations

    # ------------------------------------------------------------------
    # Next-step suggestions
    # ------------------------------------------------------------------

    def suggest_next_step(self, result: dict, analysis: dict,
                          feedback: str | None = None) -> dict:
        """Suggest what to try next.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        analysis : dict
            Output of analyze_results().
        feedback : str or None
            User feedback like 'too many false positives'.

        Returns
        -------
        suggestion : dict
            Keys: 'action', 'reason', optionally 'new_plan',
            'threshold_adjustment'.
        """
        feedback_lower = (feedback or '').lower()
        ratio = analysis.get('anomaly_ratio', 0)

        # Specific intents first (before generic keyword matches)
        if 'ensemble' in feedback_lower:
            return {
                'action': 'try_alternative',
                'reason': 'Consider running multiple detectors and '
                          'combining scores.',
                'new_plan': suggest_alternative(result, self.kb, make_plan),
            }

        # "more sensitive" intent: lower threshold / increase contamination
        _more_sensitive = (
            'false negative' in feedback_lower
            or 'missed' in feedback_lower
            or 'lower threshold' in feedback_lower
            or 'decrease threshold' in feedback_lower
            or 'increase contamination' in feedback_lower
            or 'higher contamination' in feedback_lower
        )
        if _more_sensitive:
            current_contam = result['plan'].get('params', {}).get(
                'contamination', 0.1)
            new_contam = adjust_contamination_up(current_contam)
            return {
                'action': 'adjust_threshold',
                'reason': 'Missed anomalies reported. Try increasing '
                          'contamination from %.2f to %.2f.'
                          % (current_contam, new_contam),
                'threshold_adjustment': {
                    'current_contamination': current_contam,
                    'suggested_contamination': new_contam,
                    'direction': 'increase',
                },
            }

        # "less sensitive" intent: raise threshold / decrease contamination
        _less_sensitive = (
            'false positive' in feedback_lower
            or 'too many' in feedback_lower
            or 'raise threshold' in feedback_lower
            or 'increase threshold' in feedback_lower
            or 'reduce contamination' in feedback_lower
            or 'decrease contamination' in feedback_lower
            or 'lower contamination' in feedback_lower
        )
        if _less_sensitive:
            current_contam = result['plan'].get('params', {}).get(
                'contamination', 0.1)
            new_contam = adjust_contamination_down(current_contam)
            return {
                'action': 'adjust_threshold',
                'reason': 'High false positive rate reported. Try reducing '
                          'contamination from %.2f to %.2f.'
                          % (current_contam, new_contam),
                'threshold_adjustment': {
                    'current_contamination': current_contam,
                    'suggested_contamination': new_contam,
                    'direction': 'decrease',
                },
            }

        if ('different' in feedback_lower or 'another' in feedback_lower
                or 'switch' in feedback_lower):
            new_plan = suggest_alternative(result, self.kb, make_plan)
            return {
                'action': 'try_alternative',
                'reason': 'Trying an alternative detector.',
                'new_plan': new_plan,
            }

        # No feedback: heuristic based on results
        if ratio > 0.3:
            current_contam = result['plan'].get('params', {}).get(
                'contamination', 0.1)
            new_contam = adjust_contamination_down(current_contam)
            return {
                'action': 'adjust_threshold',
                'reason': '%.0f%% flagged as anomalies, which is unusually '
                          'high. Consider reducing contamination to %.2f.'
                          % (ratio * 100, new_contam),
                'threshold_adjustment': {
                    'current_contamination': current_contam,
                    'suggested_contamination': new_contam,
                    'direction': 'decrease',
                },
            }
        if ratio == 0:
            new_plan = suggest_alternative(result, self.kb, make_plan)
            return {
                'action': 'try_alternative',
                'reason': 'No anomalies detected. Try a different detector.',
                'new_plan': new_plan,
            }

        return {
            'action': 'done',
            'reason': 'Results look reasonable (%.1f%% anomaly rate). '
                      'Review the top anomalies to validate.'
                      % (ratio * 100),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, result: dict, analysis: dict,
                        format: str = 'text') -> str:
        """Generate a summary report.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        analysis : dict
            Output of analyze_results().
        format : str
            'text' (markdown) or 'json'.

        Returns
        -------
        report : str
        """
        import json as json_mod

        if format == 'json':
            report_dict = {
                'detector': result['plan'].get('detector_name', ''),
                'reason': result['plan'].get('reason', ''),
                'n_samples': len(result['scores_train']),
                'n_anomalies': analysis['n_anomalies'],
                'anomaly_ratio': analysis['anomaly_ratio'],
                'threshold': result['threshold'],
                'runtime_seconds': result.get('runtime_seconds', 0),
                'score_distribution': analysis['score_distribution'],
                'top_anomalies': analysis['top_anomalies'][:10],
            }
            return json_mod.dumps(report_dict, indent=2, default=str)

        if format == 'text':
            lines = []
            lines.append('# Anomaly Detection Report')
            lines.append('')
            det = result['plan'].get('detector_name', 'unknown')
            lines.append('## Configuration')
            lines.append('- **Detector:** %s' % det)
            lines.append('- **Reason:** %s' % result['plan'].get('reason', ''))
            lines.append('- **Samples:** %d' % len(result['scores_train']))
            lines.append('- **Runtime:** %.2fs'
                         % result.get('runtime_seconds', 0))
            lines.append('')
            lines.append('## Results')
            lines.append('- **Anomalies found:** %d (%.1f%%)'
                         % (analysis['n_anomalies'],
                            analysis['anomaly_ratio'] * 100))
            lines.append('- **Threshold:** %.4f' % result['threshold'])
            dist = analysis['score_distribution']
            lines.append('- **Score range:** %.4f to %.4f'
                         % (dist['min'], dist['max']))
            lines.append('- **Score mean/std:** %.4f / %.4f'
                         % (dist['mean'], dist['std']))
            lines.append('')
            lines.append('## Top Anomalies')
            lines.append('')
            lines.append('| Rank | Index | Score |')
            lines.append('|------|-------|-------|')
            for rank, entry in enumerate(analysis['top_anomalies'][:10], 1):
                lines.append('| %d | %d | %.4f |'
                             % (rank, entry['index'], entry['score']))
            lines.append('')
            return '\n'.join(lines)

        raise ValueError("Unknown report format: '%s'. "
                         "Use 'text' or 'json'." % format)

    # ------------------------------------------------------------------
    # V3 Session workflow
    # ------------------------------------------------------------------

    def start(self, X: Any,
              data_type: str | None = None) -> InvestigationState:
        """Start an investigation session.

        Profiles the data and returns an InvestigationState.

        Parameters
        ----------
        X : array-like, Data, list, or dict
            Input data (any modality).
        data_type : str or None
            Explicit type override.

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import InvestigationState, _make_history_entry

        profile = self.profile_data(X, data_type=data_type)
        state = InvestigationState(
            phase='profiled',
            data=X,
            profile=profile,
            next_action={
                'action': 'plan',
                'reason': 'Data profiled as %s with %d samples. '
                          'Ready to select detectors.'
                          % (profile['data_type'],
                             profile.get('n_samples', 0)),
            },
        )
        state.history.append(_make_history_entry(
            'profiled', 'start', 0,
            'Profiled %s data' % profile['data_type']))
        return state

    def plan(self, state: InvestigationState,
             priority: str = 'balanced',
             constraints: dict | None = None) -> InvestigationState:
        """Plan detection: select top-N detectors.

        Wraps ``plan_detection()`` and extracts primary + alternatives
        into ``state.plans`` (up to 3 detectors, v1 limit).

        Parameters
        ----------
        state : InvestigationState
        priority : str
        constraints : dict or None

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import _make_history_entry

        # Clear downstream state if re-planning from later phase
        state.results = []
        state.consensus = None
        state.analysis = None
        state.quality = None

        constraints = constraints or {}
        result = self.plan_detection(
            state.profile, priority=priority, constraints=constraints)

        # Extract primary + alternatives into flat list
        plans = []
        if result.get('detector_name'):
            plans.append(result)
        for alt in result.get('alternatives', []):
            if alt.get('detector_name'):
                plans.append(alt)

        # Honor max_detectors (v1 cap at 3)
        max_det = max(1, min(
            int(constraints.get('max_detectors', 3)), 3))
        state.plans = plans[:max_det]
        state.phase = 'planned'
        names = [p['detector_name'] for p in state.plans]
        state.next_action = {
            'action': 'run',
            'reason': 'Top %d detectors selected: %s. Ready to run.'
                      % (len(state.plans), ', '.join(names)),
        }
        state.history.append(_make_history_entry(
            'planned', 'plan', state.iteration,
            'Selected %d detectors: %s' % (len(plans), ', '.join(names))))
        return state

    @staticmethod
    def _require_phase(state: InvestigationState, expected: str) -> None:
        """Enforce workflow phase precondition."""
        if state.phase != expected:
            raise ValueError(
                "Expected phase '%s', got '%s'. Call the "
                "workflow methods in order: start -> plan -> "
                "run -> analyze -> iterate/report."
                % (expected, state.phase))

    def run(self, state: InvestigationState) -> InvestigationState:
        """Run detection with all planned detectors.

        Wraps ``run_detection()`` per plan. Computes consensus via
        rank normalization and majority vote. Records errors per
        detector without stopping.

        Parameters
        ----------
        state : InvestigationState

        Returns
        -------
        state : InvestigationState
        """
        self._require_phase(state, 'planned')
        from .investigation import _make_history_entry

        results = []
        for plan in state.plans:
            try:
                raw = self.run_detection(state.data, plan)
                entry = dict(raw)
                entry['detector_name'] = plan['detector_name']
                entry['status'] = 'success'
                entry['error'] = None
                results.append(entry)
            except Exception as exc:
                logger.warning(
                    'Detector %s raised %s during run(): %s',
                    plan['detector_name'], type(exc).__name__, exc)
                results.append({
                    'detector_name': plan['detector_name'],
                    'status': 'error',
                    'error': str(exc),
                    'plan': plan,
                })

        state.results = results
        state.phase = 'detected'

        # Compute consensus from successful detectors
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        state.consensus = compute_consensus(successful)

        if state.consensus is None:
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'All %d detectors failed. Check data format '
                          'or try a different detector family.'
                          % len(results),
            }
        elif failed:
            failed_names = [r['detector_name'] for r in failed]
            successful_names = [r['detector_name'] for r in successful]
            substitutes = self._suggest_substitutes(
                state.profile,
                exclude=failed_names + successful_names,
                n_needed=len(failed_names))
            state.next_action = {
                'action': 'recover_detector_failure',
                'reason': '%d/%d detectors failed (%s); consensus '
                          'currently uses %d detector(s).'
                          % (len(failed_names), len(results),
                             ', '.join(failed_names),
                             state.consensus['n_detectors']),
                'failed_detectors': failed_names,
                'suggested_replacements': substitutes,
                'suggestion': "iterate(state, {'action': 'recover'}) "
                              "to substitute failed detectors with %s, "
                              "or call analyze(state) to proceed with the "
                              "%d successful detector(s)."
                              % (substitutes if substitutes
                                 else '<no substitutes available>',
                                 state.consensus['n_detectors']),
            }
        elif state.consensus['n_detectors'] == 1:
            state.next_action = {
                'action': 'analyze',
                'reason': 'Detection complete (1 detector).',
            }
        else:
            state.next_action = {
                'action': 'analyze',
                'reason': 'Detection complete (%d detectors, '
                          'agreement=%.2f).' % (state.consensus['n_detectors'],
                                                state.consensus['agreement']),
            }

        state.history.append(_make_history_entry(
            'detected', 'run', state.iteration,
            '%d/%d detectors succeeded'
            % (len(successful), len(results))))
        return state

    def _suggest_substitutes(self, profile: dict, exclude: list,
                             n_needed: int) -> list:
        """Suggest substitute detector names for failed slots.

        Calls ``plan_detection`` with ``exclude_detectors`` set to the
        union of failed and already-running detector names, then takes
        the top ``n_needed`` names from ``best + alternatives``. Best
        effort: returns ``[]`` if planning raises or yields no
        candidates.
        """
        if n_needed <= 0:
            return []
        try:
            plan = self.plan_detection(
                profile,
                constraints={'exclude_detectors': list(exclude)})
        except Exception as exc:
            logger.warning(
                'plan_detection failed during substitute '
                'suggestion with %s: %s',
                type(exc).__name__, exc)
            return []
        candidates = []
        if plan and plan.get('detector_name'):
            candidates.append(plan['detector_name'])
        for alt in plan.get('alternatives', []) if plan else []:
            name = alt.get('detector_name')
            if name and name not in candidates:
                candidates.append(name)
        return candidates[:n_needed]

    def analyze(self, state: InvestigationState) -> InvestigationState:
        """Analyze detection results with quality assessment.

        Computes per-detector analysis, consensus analysis, quality
        metrics (separation, agreement, stability), and selects
        the best detector.

        Parameters
        ----------
        state : InvestigationState

        Returns
        -------
        state : InvestigationState
        """
        self._require_phase(state, 'detected')
        from .investigation import _make_history_entry

        state.phase = 'analyzed'

        # All-error path
        successful = [r for r in state.results
                      if r['status'] == 'success']
        if not successful:
            state.analysis = None
            state.quality = {
                'separation': 0.0, 'agreement': 0.0,
                'stability': 0.0, 'overall': 0.0,
                'verdict': 'low',
                'explanation': 'All detectors failed.',
            }
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'All detectors failed. Check data format '
                          'or try a different detector family.',
            }
            state.history.append(_make_history_entry(
                'analyzed', 'analyze', state.iteration,
                'All detectors failed'))
            return state

        # Per-detector analysis (aligned with state.results)
        per_det = []
        for r in state.results:
            if r['status'] == 'success':
                try:
                    a = self.analyze_results(r, X=state.data)
                except Exception as exc:
                    logger.warning(
                        'analyze_results failed for %s with %s: %s',
                        r.get('detector_name', '<unknown>'),
                        type(exc).__name__, exc)
                    a = None
                per_det.append(a)
            else:
                per_det.append(None)

        # Consensus analysis (lightweight, not via analyze_results)
        c = state.consensus
        c_scores = c['scores']
        c_labels = c['labels']
        n_anomalies = int(c_labels.sum())
        n_samples = len(c_labels)
        top_k = min(10, n_samples)
        top_indices = np.argsort(c_scores)[::-1][:top_k]
        consensus_analysis = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / max(n_samples, 1),
            'score_distribution': {
                'mean': float(np.mean(c_scores)),
                'std': float(np.std(c_scores)),
                'min': float(np.min(c_scores)),
                'max': float(np.max(c_scores)),
                'median': float(np.median(c_scores)),
                'q25': float(np.percentile(c_scores, 25)),
                'q75': float(np.percentile(c_scores, 75)),
            },
            'top_anomalies': [
                {'index': int(i), 'score': float(c_scores[i])}
                for i in top_indices],
            'summary': '%d anomalies detected out of %d samples '
                       '(%.1f%%) by consensus of %d detectors.'
                       % (n_anomalies, n_samples,
                          100 * n_anomalies / max(n_samples, 1),
                          c['n_detectors']),
        }

        # Best detector selection
        best_idx = select_best_detector(
            state.results, c_scores)

        state.analysis = {
            'consensus_analysis': consensus_analysis,
            'per_detector_analysis': per_det,
            'best_detector': state.results[best_idx]['detector_name'],
            'best_detector_index': best_idx,
            'summary': consensus_analysis['summary'],
        }

        # Quality metrics
        state.quality = compute_quality(
            c_scores, c_labels, state.results, c)
        state.analysis['summary'] += (
            ' Quality: %s (%.2f).'
            % (state.quality['verdict'], state.quality['overall']))

        # Next action based on quality
        if state.quality['overall'] >= 0.4:
            state.next_action = {
                'action': 'report_to_user',
                'reason': 'Results ready (quality=%s, %.2f).'
                          % (state.quality['verdict'],
                             state.quality['overall']),
                'summary': state.analysis['summary'],
                'confidence': state.quality['overall'],
            }
        else:
            state.next_action = {
                'action': 'iterate',
                'reason': 'Low result quality (%.2f). Consider '
                          'trying different detectors.'
                          % state.quality['overall'],
                'suggestion': 'Exclude lowest-agreement detector '
                              'and re-run.',
            }

        state.history.append(_make_history_entry(
            'analyzed', 'analyze', state.iteration,
            'Quality: %s (%.2f)' % (
                state.quality['verdict'],
                state.quality['overall'])))
        return state

    # ------------------------------------------------------------------
    # V3 Session workflow: iterate
    # ------------------------------------------------------------------

    def iterate(self, state: InvestigationState,
                feedback: str | dict) -> InvestigationState:
        """Iterate based on feedback.

        Structured dicts execute immediately. NL strings are
        parsed with confidence; ambiguous feedback triggers
        ``'confirm_with_user'``.

        Most actions require phase ``'analyzed'``. The ``'recover'``
        action also accepts phase ``'detected'`` so the agent can
        substitute failed detectors immediately after ``run()``
        without first calling ``analyze()``.

        Parameters
        ----------
        state : InvestigationState
        feedback : str or dict

        Returns
        -------
        state : InvestigationState
        """
        action = feedback.get('action') if isinstance(feedback, dict) else None
        if action == 'recover':
            if state.phase not in ('detected', 'analyzed'):
                raise ValueError(
                    "Recover requires phase 'detected' or "
                    "'analyzed', got '%s'. Call run() first."
                    % state.phase)
        else:
            self._require_phase(state, 'analyzed')
        if isinstance(feedback, dict):
            return apply_structured_feedback(
                state, feedback, self.kb, self.plan_detection, make_plan)
        return apply_nl_feedback(
            state, str(feedback), self.kb, self.plan_detection, make_plan)

    # ------------------------------------------------------------------
    # V3 Session workflow: report and investigate
    # ------------------------------------------------------------------

    def report(self, state: InvestigationState,
               format: str = 'text') -> str | dict:
        """Generate investigation report.

        Text format wraps ``generate_report()`` for best detector,
        prepending session-level context. JSON format returns a
        native dict.

        Parameters
        ----------
        state : InvestigationState
        format : str
            'text' or 'json'.

        Returns
        -------
        report : str or dict
        """
        self._require_phase(state, 'analyzed')
        if format not in ('text', 'json'):
            raise ValueError(
                "Unknown report format: '%s'. Use 'text' or 'json'."
                % format)
        if state.analysis is None:
            raise ValueError(
                "No successful detectors to report on. "
                "Use iterate() to adjust the plan.")

        best_idx = state.analysis['best_detector_index']
        best_result = state.results[best_idx]
        best_analysis = state.analysis['per_detector_analysis'][
            best_idx]

        if format == 'json':
            return {
                'session': {
                    'consensus': {
                        'scores': state.consensus[
                            'scores'].tolist(),
                        'labels': state.consensus[
                            'labels'].tolist(),
                        'n_detectors': state.consensus[
                            'n_detectors'],
                        'agreement': state.consensus[
                            'agreement'],
                        'disagreements': state.consensus[
                            'disagreements'],
                    },
                    'quality': state.quality,
                    'comparison': {
                        'agreement': state.consensus[
                            'agreement'],
                        'disagreements': state.consensus[
                            'disagreements'],
                    },
                },
                'best_detector': {
                    'name': best_result['detector_name'],
                    'scores': best_result[
                        'scores_train'].tolist(),
                    'labels': best_result[
                        'labels_train'].tolist(),
                    'threshold': best_result['threshold'],
                    'analysis': best_analysis,
                },
            }

        # Text format
        lines = []
        lines.append('# Investigation Report')
        lines.append('')

        # Session section
        lines.append('## Session Summary')
        c = state.consensus
        q = state.quality
        lines.append('- **Detectors run:** %d' % c['n_detectors'])
        lines.append('- **Detector agreement:** %.2f'
                     % c['agreement'])
        lines.append('- **Quality verdict:** %s (%.2f)'
                     % (q['verdict'], q['overall']))
        lines.append('- **Iterations:** %d' % state.iteration)
        if c['disagreements']:
            lines.append('- **Disagreements:** %d samples'
                         % len(c['disagreements']))
        lines.append('')

        # Best detector report (via generate_report)
        detector_report = self.generate_report(
            best_result, best_analysis, format='text')
        lines.append(detector_report)

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Contamination diagnostics (O5 narrowed)
    # ------------------------------------------------------------------

    _CONTAMINATION_DIAGNOSTIC_PERCENTILES: tuple[int, ...] = (
        50, 75, 90, 95, 99)

    def contamination_diagnostics(
            self,
            state: InvestigationState,
            threshold_sweep: list[float] | None = None) -> dict:
        """Diagnostic helper for contamination calibration.

        Reports the contamination value the run actually used, the
        actual flagged rate from the consensus, the score-percentile
        distribution, and (optionally) a threshold sweep showing what
        fraction would be flagged at each candidate contamination
        value. The agent can use these numbers to choose a sensible
        next contamination before iterating.

        This helper does NOT estimate contamination automatically and
        does NOT mutate state. It is purely a read-only diagnostic the
        agent uses to inform a subsequent
        `engine.iterate(state, {'action': 'adjust_contamination',
        'value': <rate>})` call.

        Parameters
        ----------
        state : InvestigationState
            Must be in the 'analyzed' phase.
        threshold_sweep : list of float or None
            Optional sequence of candidate contamination values in
            (0, 1). For each value c, the result includes the
            corresponding threshold (the (1 - c) quantile of consensus
            scores) and the resulting flagged rate. Use this to preview
            how the flagged set would change before deciding to
            iterate. Values outside (0, 1) are skipped.

        Returns
        -------
        diagnostics : dict
            Keys:

            - ``effective_contamination`` (float or None): contamination
              value from the primary plan's params, or ``None`` if the
              plan has no contamination set.
            - ``flagged_rate`` (float): actual fraction flagged by the
              consensus labels.
            - ``score_percentiles`` (dict[int, float]): consensus-score
              percentiles at the 50th, 75th, 90th, 95th, and 99th.
            - ``threshold_sweep`` (list of dict, optional): present only
              when ``threshold_sweep`` was passed; each entry has
              ``contamination``, ``threshold``, and ``flagged_rate``.
        """
        self._require_phase(state, 'analyzed')

        primary_plan = state.plans[0] if state.plans else {}
        effective = primary_plan.get('params', {}).get('contamination')

        if state.consensus is None:
            diagnostics: dict = {
                'effective_contamination': effective,
                'flagged_rate': 0.0,
                'score_percentiles': {},
            }
            if threshold_sweep:
                diagnostics['threshold_sweep'] = []
            return diagnostics

        scores = state.consensus['scores']
        labels = state.consensus['labels']
        n = len(labels)
        flagged_rate = float(labels.sum()) / n if n > 0 else 0.0

        score_percentiles = {
            p: float(np.percentile(scores, p))
            for p in self._CONTAMINATION_DIAGNOSTIC_PERCENTILES
        }

        diagnostics = {
            'effective_contamination': effective,
            'flagged_rate': flagged_rate,
            'score_percentiles': score_percentiles,
        }

        if threshold_sweep:
            sweep_results = []
            for c in threshold_sweep:
                if not (0.0 < c < 1.0):
                    # Skip invalid candidates rather than raise; this
                    # is a lenient diagnostic, not a strict validator.
                    continue
                threshold = float(np.quantile(scores, 1.0 - c))
                n_flagged = int((scores > threshold).sum())
                sweep_results.append({
                    'contamination': float(c),
                    'threshold': threshold,
                    'flagged_rate': (
                        n_flagged / n if n > 0 else 0.0),
                })
            diagnostics['threshold_sweep'] = sweep_results

        return diagnostics

    # ------------------------------------------------------------------
    # Hindsight validation (O8)
    # ------------------------------------------------------------------

    def validate(self,
                 state: InvestigationState,
                 y: Any) -> dict:
        """Hindsight validation of consensus and per-detector results.

        Computes label-based metrics from `y` against the consensus
        labels and each successful detector, plus a
        consensus-vs-best-detector diagnostic so the agent can see
        whether consensus actually helped.

        Pure functional; does not mutate state. Use after `analyze`
        when held-out labels become available (e.g., a labeled cohort
        opened post-hoc for hindsight evaluation). For routine
        unsupervised detection runs, this method is unnecessary.

        Parameters
        ----------
        state : InvestigationState
            Must be in the 'analyzed' phase.
        y : array-like, shape (n_samples,)
            Held-out binary labels (0 = inlier, 1 = anomaly). Length
            must match the consensus.

        Returns
        -------
        validation : dict
            Keys:

            - ``consensus`` (dict): label_metrics for the consensus
              labels and scores.
            - ``per_detector`` (dict[str, dict]): label_metrics per
              successful detector, keyed by detector name.
            - ``best_detector`` (dict or None): label_metrics for the
              detector picked by `analyze` as best (or None when
              `state.analysis` does not name one).
            - ``consensus_vs_best`` (dict): comparison summary with
              keys ``consensus_f1``, ``best_detector_f1`` (or None),
              and ``consensus_helped`` (True if consensus F1 is at
              least the best-detector F1; None when no best detector).
            - ``false_positives`` (list[int]): row indices flagged by
              consensus but inlier in `y`.
            - ``false_negatives`` (list[int]): row indices not flagged
              by consensus but anomaly in `y`.

        Raises
        ------
        ValueError
            If `state` is not in 'analyzed' phase, if the consensus is
            missing (all detectors failed), or if `len(y)` does not
            match the consensus length.
        """
        self._require_phase(state, 'analyzed')
        if state.consensus is None:
            raise ValueError(
                "Cannot validate: state.consensus is None (all "
                "detectors failed). Use iterate() to recover first.")

        y_arr = np.asarray(y).astype(int)
        consensus_labels = state.consensus['labels']
        consensus_scores = state.consensus['scores']
        n = len(consensus_labels)
        if len(y_arr) != n:
            raise ValueError(
                f"y has {len(y_arr)} samples but the consensus has "
                f"{n}; lengths must match.")

        consensus_metrics = label_metrics(
            y_arr, consensus_labels, consensus_scores)

        fp_indices = np.where(
            (consensus_labels == 1) & (y_arr == 0))[0].tolist()
        fn_indices = np.where(
            (consensus_labels == 0) & (y_arr == 1))[0].tolist()

        per_detector: dict[str, dict] = {}
        for r in state.results:
            if r.get('status') == 'success':
                per_detector[r['detector_name']] = label_metrics(
                    y_arr, r['labels_train'], r['scores_train'])

        best_metrics = None
        if state.analysis and 'best_detector' in state.analysis:
            best_name = state.analysis['best_detector']
            best_metrics = per_detector.get(best_name)

        if best_metrics is not None:
            consensus_helped = (
                consensus_metrics['f1'] >= best_metrics['f1'])
            best_f1 = best_metrics['f1']
        else:
            consensus_helped = None
            best_f1 = None

        return {
            'consensus': consensus_metrics,
            'per_detector': per_detector,
            'best_detector': best_metrics,
            'consensus_vs_best': {
                'consensus_f1': consensus_metrics['f1'],
                'best_detector_f1': best_f1,
                'consensus_helped': consensus_helped,
            },
            'false_positives': [int(i) for i in fp_indices],
            'false_negatives': [int(i) for i in fn_indices],
        }

    def investigate(self, X: Any, data_type: str | None = None,
                    priority: str = 'balanced') -> InvestigationState:
        """One-shot investigation: start → plan → run → analyze.

        Parameters
        ----------
        X : array-like
            Input data.
        data_type : str or None
        priority : str

        Returns
        -------
        state : InvestigationState
        """
        state = self.start(X, data_type=data_type)
        state = self.plan(state, priority=priority)
        state = self.run(state)
        state = self.analyze(state)
        return state

    # ------------------------------------------------------------------
    # Knowledge queries
    # ------------------------------------------------------------------

    def list_detectors(self, data_type: str | None = None,
                       status: str = 'shipped') -> list[dict]:
        """List available detectors.

        Parameters
        ----------
        data_type : str or None
            Filter by data type (e.g. 'tabular', 'text').
        status : str
            Filter by status. Use 'all' to list everything.

        Returns
        -------
        detectors : list of dict
        """
        if data_type:
            return self.kb.list_by_data_type(data_type, status=status)
        if status == 'all':
            return [{'name': k, **v}
                    for k, v in self.kb.algorithms.items()]
        return self.kb.list_by_status(status)

    def explain_detector(self, name: str) -> dict:
        """Explain a detector.

        Parameters
        ----------
        name : str
            Detector short name (e.g. 'ECOD').

        Returns
        -------
        info : dict
        """
        algo = self.kb.get_algorithm(name)
        if algo is None:
            raise ValueError("Unknown detector '%s'" % name)
        return {'name': name, **algo}

    # Maps a data_type to (benchmark name, ranking key) for
    # `compare_detectors` when the KB benchmark's top-level ranking
    # already uses PyOD detector names (e.g., ADBench `overall_top_5`).
    _COMPARE_BENCHMARK_RANKINGS: dict[str, tuple[str, str]] = {
        'tabular': ('ADBench', 'overall_top_5'),
    }

    # Maps a data_type to benchmark-rank keys stored on each shipped
    # detector's `benchmark_rank` metadata. Used when the benchmark's
    # top-level ranking lists paper method names that do not match the
    # PyOD detector names (e.g., TSB-AD lists "POLY", "KShapeAD", which
    # do not match the shipped `KShape`, `MatrixProfile`, etc.). Lower
    # rank value = better. When a detector carries multiple matching
    # keys, the minimum (best) rank wins.
    _COMPARE_BENCHMARK_RANK_KEYS: dict[str, tuple[str, ...]] = {
        'time_series': ('TSB_AD_overall', 'TSB_AD_overall_iforest'),
    }

    def _benchmark_ranked_detectors(self, data_type: str,
                                    top_k: int) -> list[str] | None:
        """Return up to `top_k` shipped detector names for `data_type`,
        ranked by the modality-specific benchmark from the KB.

        Two ranking sources are consulted in order. First, when
        `_COMPARE_BENCHMARK_RANKINGS` lists the data_type, use the
        benchmark's top-level overall ranking and filter to shipped
        detectors. Second, when `_COMPARE_BENCHMARK_RANK_KEYS` lists
        the data_type, read each shipped detector's `benchmark_rank`
        metadata and sort ascending by best rank. In both modes,
        detectors without an applicable rank are appended in catalog
        order to fill `top_k`. Returns `None` when no applicable
        ranking exists, signalling the caller to fall back to catalog
        order. Used by `compare_detectors` (TA1).
        """
        bench_lookup = self._COMPARE_BENCHMARK_RANKINGS.get(data_type)
        if bench_lookup is not None:
            bench_name, ranking_key = bench_lookup
            bench = self.kb.benchmarks.get(bench_name)
            if not bench:
                return None
            ranked = bench.get('rankings', {}).get(ranking_key, [])
            if not ranked:
                return None
            shipped_dicts = self.list_detectors(data_type=data_type)
            shipped_set = {d['name'] for d in shipped_dicts}
            ranked_shipped = [n for n in ranked if n in shipped_set]
            if not ranked_shipped:
                return None
            remaining = [d['name'] for d in shipped_dicts
                         if d['name'] not in ranked_shipped]
            return (ranked_shipped + remaining)[:top_k]

        rank_keys = self._COMPARE_BENCHMARK_RANK_KEYS.get(data_type)
        if rank_keys is None:
            return None
        shipped_dicts = self.list_detectors(data_type=data_type)
        ranked_pairs: list[tuple[int, str]] = []
        unranked: list[str] = []
        for detector in shipped_dicts:
            ranks = detector.get('benchmark_rank', {})
            values = [ranks[key] for key in rank_keys if key in ranks]
            if values:
                ranked_pairs.append((min(values), detector['name']))
            else:
                unranked.append(detector['name'])
        if not ranked_pairs:
            return None
        ranked_names = [name for _, name in sorted(ranked_pairs)]
        return (ranked_names + unranked)[:top_k]

    def compare_detectors(self, names: list[str] | None = None,
                          data_type: str | None = None,
                          top_k: int = 3) -> list[dict]:
        """Compare detectors.

        When `names` is provided, returns explanations for those
        detectors in input order.

        When `names` is omitted and `data_type` has a benchmark-backed
        ranking in the KB, returns up to `top_k` detectors ranked by
        that benchmark, then appends remaining shipped detectors in
        catalog order until `top_k` is reached. Two ranking sources are
        supported: top-level `overall_top_5` for benchmarks whose names
        match PyOD detector names (currently `tabular` via ADBench);
        per-detector `benchmark_rank` metadata when the benchmark lists
        paper method names (currently `time_series` via TSB-AD, sorted
        ascending by the best matching rank key). For modalities
        without an applicable ranking (`graph`, `text`, `image`,
        `multimodal`) or when no `data_type` is given, falls back to
        the catalog order from `list_detectors`.

        Parameters
        ----------
        names : list of str or None
            Explicit list of detector names to compare.
        data_type : str or None
            Filter by data type.
        top_k : int
            Number of detectors to return when not using explicit names.

        Returns
        -------
        comparison : list of dict
        """
        if names:
            return [self.explain_detector(n) for n in names]
        if data_type:
            ranked = self._benchmark_ranked_detectors(data_type, top_k)
            if ranked is not None:
                return [self.explain_detector(n) for n in ranked]
        detectors = self.list_detectors(data_type=data_type)
        return detectors[:top_k]

    def get_benchmarks(self, benchmark: str = 'all') -> dict:
        """Get benchmark results.

        Parameters
        ----------
        benchmark : str
            Benchmark name, or 'all' for everything.

        Returns
        -------
        benchmarks : dict
        """
        if benchmark == 'all':
            return self.kb.benchmarks
        return {benchmark: self.kb.benchmarks.get(benchmark)}
