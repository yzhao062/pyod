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

    def _sniff_data_type(self, X):
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

    def run_detection(self, X_train, plan, X_test=None):
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

    def analyze_results(self, result, X=None, top_k=10):
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
            fi = self._compute_feature_importance(result, X)
            if fi is not None:
                analysis['feature_importance'] = fi

        return analysis

    @staticmethod
    def _compute_feature_importance(result, X):
        """Estimate per-feature contribution to anomaly scores."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2:
                return None
            scores = result['scores_train']
            if len(scores) != X_arr.shape[0]:
                return None

            means = np.mean(X_arr, axis=0)
            stds = np.std(X_arr, axis=0)
            stds[stds == 0] = 1.0
            z_scores = np.abs((X_arr - means) / stds)

            importances = []
            for j in range(X_arr.shape[1]):
                corr = np.corrcoef(z_scores[:, j], scores)[0, 1]
                importances.append(float(corr) if np.isfinite(corr) else 0.0)

            return importances
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_findings(self, result, indices=None, top_k=5, X=None):
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

        Returns
        -------
        explanations : list of dict
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
                contribs = self._feature_contributions(X, idx, scores)
                if contribs is not None:
                    entry['contributing_features'] = contribs

            explanations.append(entry)

        return explanations

    @staticmethod
    def _feature_contributions(X, idx, scores):
        """Compute per-feature z-score for a specific sample."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2:
                return None
            means = np.mean(X_arr, axis=0)
            stds = np.std(X_arr, axis=0)
            stds[stds == 0] = 1.0
            z = np.abs((X_arr[idx] - means) / stds)
            top_feat = np.argsort(z)[::-1][:5]
            return [{'feature': int(f), 'z_score': float(z[f])}
                    for f in top_feat]
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Next-step suggestions
    # ------------------------------------------------------------------

    def suggest_next_step(self, result, analysis, feedback=None):
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
                'new_plan': self._suggest_alternative(result),
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
            new_contam = min(current_contam * 1.5, 0.5)
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
            new_contam = max(current_contam * 0.5, 0.01)
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
            new_plan = self._suggest_alternative(result)
            return {
                'action': 'try_alternative',
                'reason': 'Trying an alternative detector.',
                'new_plan': new_plan,
            }

        # No feedback: heuristic based on results
        if ratio > 0.3:
            current_contam = result['plan'].get('params', {}).get(
                'contamination', 0.1)
            new_contam = max(current_contam * 0.5, 0.01)
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
            new_plan = self._suggest_alternative(result)
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

    def _suggest_alternative(self, result):
        """Suggest an alternative detector different from the current one."""
        current = result['plan'].get('detector_name', '')

        alternatives = result['plan'].get('alternatives', [])
        for alt in alternatives:
            if alt.get('detector_name') and alt['detector_name'] != current:
                return alt

        fallback_order = ['IForest', 'ECOD', 'KNN', 'LOF', 'HBOS',
                          'COPOD', 'CBLOF']
        for name in fallback_order:
            if name != current:
                algo = self.kb.get_algorithm(name)
                if algo and algo.get('status') == 'shipped':
                    return self._make_plan(
                        detector_name=name, params={},
                        reason='Alternative to %s' % current,
                        evidence=[], confidence=0.6)

        return self._make_plan(
            detector_name='IForest', params={},
            reason='Default fallback', evidence=[], confidence=0.5)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, result, analysis, format='text'):
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

    def start(self, X, data_type=None):
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

    def plan(self, state, priority='balanced', constraints=None):
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

    def run(self, state):
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
        from .investigation import _make_history_entry
        from scipy.stats import rankdata, spearmanr

        results = []
        for plan in state.plans:
            try:
                raw = self.run_detection(state.data, plan)
                entry = dict(raw)
                entry['detector_name'] = plan['detector_name']
                entry['status'] = 'success'
                entry['error'] = None
                results.append(entry)
            except Exception as e:
                results.append({
                    'detector_name': plan['detector_name'],
                    'status': 'error',
                    'error': str(e),
                    'plan': plan,
                })

        state.results = results
        state.phase = 'detected'

        # Compute consensus from successful detectors
        successful = [r for r in results if r['status'] == 'success']

        if len(successful) == 0:
            state.consensus = None
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'All %d detectors failed. Check data format '
                          'or try a different detector family.'
                          % len(results),
            }
        elif len(successful) == 1:
            r = successful[0]
            state.consensus = {
                'scores': r['scores_train'],
                'labels': r['labels_train'],
                'n_detectors': 1,
                'agreement': 0.5,
                'disagreements': [],
            }
            state.next_action = {
                'action': 'analyze',
                'reason': 'Detection complete (1 detector).',
            }
        else:
            n_samples = len(successful[0]['scores_train'])
            # Rank-normalize scores per detector
            rank_scores = np.array([
                rankdata(r['scores_train']) / n_samples
                for r in successful
            ])
            consensus_scores = np.mean(rank_scores, axis=0)

            # Majority-vote labels
            all_labels = np.array([
                r['labels_train'] for r in successful])
            vote_count = np.sum(all_labels, axis=0)
            consensus_labels = (
                vote_count > len(successful) / 2).astype(int)

            # Pairwise Spearman agreement
            correlations = []
            for i in range(len(successful)):
                for j in range(i + 1, len(successful)):
                    rho, _ = spearmanr(
                        successful[i]['scores_train'],
                        successful[j]['scores_train'])
                    correlations.append(
                        max(0.0, rho) if np.isfinite(rho) else 0.0)
            agreement = float(np.mean(correlations)) if correlations else 0.5

            # Disagreements: indices where detectors disagree
            disagreements = []
            for idx in range(n_samples):
                votes = all_labels[:, idx]
                if not (votes.all() or not votes.any()):
                    disagreements.append(int(idx))

            state.consensus = {
                'scores': consensus_scores,
                'labels': consensus_labels,
                'n_detectors': len(successful),
                'agreement': agreement,
                'disagreements': disagreements,
            }
            state.next_action = {
                'action': 'analyze',
                'reason': 'Detection complete (%d detectors, '
                          'agreement=%.2f).'
                          % (len(successful), agreement),
            }

        state.history.append(_make_history_entry(
            'detected', 'run', state.iteration,
            '%d/%d detectors succeeded'
            % (len(successful), len(results))))
        return state

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
