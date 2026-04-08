# -*- coding: utf-8 -*-
"""ADEngine: Intelligent anomaly detection lifecycle engine.

Handles data profiling, detection planning, detector construction,
and knowledge queries. Works as a standalone Python API (no LLM
required) or as the backend for MCP/agent interfaces.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import importlib
import os

import numpy as np

from .knowledge import KnowledgeBase


class ADEngine:
    """Anomaly detection lifecycle engine.

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge base directory. If None, uses bundled.
    """

    def __init__(self, knowledge_dir=None):
        self.kb = KnowledgeBase(knowledge_dir=knowledge_dir)

    def profile_data(self, X, data_type=None):
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
        else:
            # tabular, time_series, or graph
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

    def _sniff_data_type(self, X):
        """Conservative data type detection."""
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
    def _looks_like_image_paths(samples):
        """Check if string samples look like image file paths."""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif',
                      '.tiff', '.webp'}
        for s in samples:
            ext = os.path.splitext(s)[1].lower()
            if ext not in image_exts:
                return False
        return True

    def plan_detection(self, profile, priority='balanced',
                       constraints=None):
        """Plan a detection pipeline.

        Parameters
        ----------
        profile : dict
            Output of profile_data().
        priority : str
            'speed', 'accuracy', or 'balanced'.
        constraints : dict or None
            Optional: {'exclude_detectors': [...]}

        Returns
        -------
        plan : dict (DetectionPlan, closed schema)
        """
        constraints = constraints or {}
        exclude = set(constraints.get('exclude_detectors', []))

        matched = self._evaluate_rules(profile, priority)

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
                return self._make_plan(
                    detector_name='',
                    params={},
                    reason='No valid detector available: all candidates '
                           'excluded or no matching rule found',
                    evidence=[],
                    confidence=0.0,
                    alternatives=[],
                    note='no_valid_plan')

            return self._make_plan(
                detector_name=fallback_name, params={},
                reason='Fallback: no routing rule matched or all '
                       'candidates excluded',
                evidence=['ADBench'], confidence=0.5,
                alternatives=[], note='No specific rule matched')

        best = valid[0]
        alternatives = [self._make_plan(
            detector_name=r['detector'],
            params=r.get('params', {}),
            preset=r.get('preset'),
            reason=r.get('_reason', ''),
            evidence=r.get('_evidence', []),
            confidence=r.get('confidence', 0.5),
            alternatives=[]) for r in valid[1:3]]

        return self._make_plan(
            detector_name=best['detector'],
            params=best.get('params', {}),
            preset=best.get('preset'),
            reason=best.get('_reason', ''),
            evidence=best.get('_evidence', []),
            confidence=best.get('confidence', 0.7),
            alternatives=alternatives)

    def _evaluate_rules(self, profile, priority):
        """Evaluate routing rules against profile. Returns matched
        recommendations with their rule context."""
        rules = self.kb.routing_rules.get('rules', [])
        all_recs = []

        for rule in rules:
            if self._rule_matches(rule, profile, priority):
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

    def _rule_matches(self, rule, profile, priority):
        """Check if all conditions in a rule match the profile."""
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
            if not self._eval_condition(actual, op, value):
                return False
        return True

    @staticmethod
    def _eval_condition(actual, op, value):
        """Evaluate a single condition predicate."""
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

    @staticmethod
    def _make_plan(detector_name, params=None, preset=None,
                   reason='', evidence=None, confidence=0.5,
                   alternatives=None, note=None):
        """Construct a closed-schema DetectionPlan dict."""
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

    # ------------------------------------------------------------------
    # Detector construction
    # ------------------------------------------------------------------

    def build_detector(self, plan):
        """Build and return an unfitted detector from a plan.

        Parameters
        ----------
        plan : dict (DetectionPlan)
            Output of plan_detection().

        Returns
        -------
        detector : BaseDetector
        """
        name = plan['detector_name']
        algo = self.kb.get_algorithm(name)
        if algo is None:
            raise ValueError("Unknown detector '%s'" % name)
        if algo.get('status') not in ('shipped', 'experimental'):
            raise ValueError(
                "Detector '%s' has status '%s' and cannot be built"
                % (name, algo.get('status', 'unknown')))

        preset = plan.get('preset')
        if preset:
            return self._build_from_preset(name, preset,
                                           plan.get('params', {}))

        class_path = algo['class_path']
        module_path, class_name = class_path.rsplit('.', 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        params = plan.get('params', {})
        return cls(**params)

    @staticmethod
    def _build_from_preset(detector_name, preset, extra_params):
        """Build a detector using a factory preset."""
        if detector_name == 'EmbeddingOD':
            from pyod.models.embedding import EmbeddingOD
            if preset == 'for_text':
                return EmbeddingOD.for_text(**extra_params)
            elif preset == 'for_image':
                return EmbeddingOD.for_image(**extra_params)
        raise ValueError("Unknown preset '%s' for '%s'"
                         % (preset, detector_name))

    # ------------------------------------------------------------------
    # One-shot detection
    # ------------------------------------------------------------------

    def detect(self, X_train, X_test=None, data_type=None,
               priority='balanced'):
        """One-shot anomaly detection: profile -> plan -> build -> fit.

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
            Keys: 'plan', 'scores', 'labels', 'threshold',
            'n_anomalies', 'detector'.
        """
        profile = self.profile_data(X_train, data_type=data_type)
        plan = self.plan_detection(profile, priority=priority)
        clf = self.build_detector(plan)
        clf.fit(X_train)

        result = {
            'plan': plan,
            'scores': clf.decision_scores_,
            'labels': clf.labels_,
            'threshold': clf.threshold_,
            'n_anomalies': int(clf.labels_.sum()),
            'detector': clf,
        }

        if X_test is not None:
            test_scores = clf.decision_function(X_test)
            test_labels = clf.predict(X_test)
            result['scores_test'] = test_scores
            result['labels_test'] = test_labels

        return result

    # ------------------------------------------------------------------
    # Knowledge queries
    # ------------------------------------------------------------------

    def list_detectors(self, data_type=None, status='shipped'):
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

    def explain_detector(self, name):
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

    def compare_detectors(self, names=None, data_type=None, top_k=3):
        """Compare detectors.

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
        detectors = self.list_detectors(data_type=data_type)
        return detectors[:top_k]

    def get_benchmarks(self, benchmark='all'):
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
