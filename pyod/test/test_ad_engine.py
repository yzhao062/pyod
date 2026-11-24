# -*- coding: utf-8 -*-

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine
from pyod.models.base import BaseDetector


class TestProfileData(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_tabular_array(self):
        X = np.random.randn(1000, 20)
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'tabular'
        assert profile['n_samples'] == 1000
        assert profile['n_features'] == 20

    def test_tabular_1d_is_tabular_not_ts(self):
        X = np.random.randn(500)
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'tabular'

    def test_explicit_time_series_override(self):
        X = np.random.randn(500)
        profile = self.engine.profile_data(X, data_type='time_series')
        assert profile['data_type'] == 'time_series'

    def test_text_list(self):
        X = ["hello world", "anomaly detection", "test sentence"]
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'text'
        assert profile['n_samples'] == 3

    def test_dict_is_multimodal(self):
        X = {'text': ["hello"], 'tabular': np.array([[1, 2]])}
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'multimodal'

    def test_has_nan_detection(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        profile = self.engine.profile_data(X)
        assert profile['has_nan'] is True

    def test_no_nan(self):
        X = np.random.randn(100, 5)
        profile = self.engine.profile_data(X)
        assert profile['has_nan'] is False

    def test_dimensionality_class_high(self):
        X = np.random.randn(100, 200)
        profile = self.engine.profile_data(X)
        assert profile['dimensionality_class'] == 'high'

    def test_dimensionality_class_low(self):
        X = np.random.randn(100, 5)
        profile = self.engine.profile_data(X)
        assert profile['dimensionality_class'] == 'low'


class TestPlanDetection(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_tabular_high_dim_speed(self):
        profile = {'data_type': 'tabular', 'n_samples': 10000,
                   'n_features': 200, 'dimensionality_class': 'high'}
        plan = self.engine.plan_detection(profile, priority='speed')
        assert plan['detector_name'] == 'ECOD'
        assert plan['confidence'] > 0
        assert 'reason' in plan
        assert 'evidence' in plan

    def test_tabular_low_dim_small(self):
        profile = {'data_type': 'tabular', 'n_samples': 1000,
                   'n_features': 5, 'dimensionality_class': 'low'}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] in ('KNN', 'LOF', 'CBLOF',
                                          'IForest', 'ECOD')

    def test_text_routes_to_embedding(self):
        profile = {'data_type': 'text', 'n_samples': 100}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] == 'EmbeddingOD'
        assert plan.get('preset') == 'for_text'

    def test_image_routes_to_embedding(self):
        profile = {'data_type': 'image', 'n_samples': 50}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] == 'EmbeddingOD'
        assert plan.get('preset') == 'for_image'

    def test_plan_has_alternatives(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(profile)
        assert 'alternatives' in plan
        assert isinstance(plan['alternatives'], list)

    def test_time_series_routes_to_shipped_detector(self):
        profile = {'data_type': 'time_series', 'n_samples': 1000,
                   'n_features': 1}
        plan = self.engine.plan_detection(profile)
        # Should route to a shipped TS detector (KShape is #2 in TSB-AD)
        assert plan['detector_name'] in ('KShape', 'TimeSeriesOD',
                                          'SpectralResidual', 'LSTMAD')
        assert plan['confidence'] >= 0.7

    def test_constraints_exclude_detector(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(
            profile, constraints={'exclude_detectors': ['IForest', 'ECOD']})
        assert plan['detector_name'] not in ('IForest', 'ECOD')

    def test_fallback_respects_exclusions(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(
            profile,
            constraints={'exclude_detectors': ['IForest', 'ECOD', 'KNN']})
        assert plan['detector_name'] not in ('IForest', 'ECOD', 'KNN')

    def test_all_fallbacks_excluded_returns_no_plan(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(
            profile,
            constraints={'exclude_detectors': [
                'IForest', 'ECOD', 'KNN', 'HBOS', 'LOF', 'COPOD', 'PCA']})
        assert plan['note'] == 'no_valid_plan'
        assert plan['confidence'] == 0.0

    def test_plan_is_closed_schema(self):
        profile = {'data_type': 'tabular', 'n_samples': 1000,
                   'n_features': 10, 'dimensionality_class': 'low'}
        plan = self.engine.plan_detection(profile)
        allowed_keys = {'detector_name', 'preset', 'params',
                        'preprocessing', 'threshold_strategy',
                        'threshold_value', 'reason', 'evidence',
                        'confidence', 'alternatives', 'note'}
        for key in plan:
            assert key in allowed_keys, \
                f"Unexpected key '{key}' in plan"


class TestPlanDetectionContamination(unittest.TestCase):
    """TA2: plan_detection always exposes the effective contamination in
    `params` so the MCP `plan_detection` -> `build_detector` chain emits
    a code snippet that names the value an MCP-only agent would actually
    run with."""

    def setUp(self):
        self.engine = ADEngine()

    def test_primary_plan_includes_contamination_for_iforest(self):
        # Routing rule for medium-tabular returns IForest with empty
        # params; contamination must be filled from KB defaults.
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(profile)
        assert 'contamination' in plan['params'], \
            "primary plan params must include contamination (TA2)"
        assert plan['params']['contamination'] == 0.1

    def test_alternatives_include_contamination(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(profile)
        for alt in plan['alternatives']:
            if alt.get('detector_name'):
                assert 'contamination' in alt['params'], \
                    f"alt {alt['detector_name']} missing contamination"

    def test_fallback_plan_includes_contamination(self):
        # Force the routing-fallback path: profile that no rule matches.
        profile = {'data_type': 'tabular', 'n_samples': 50,
                   'n_features': 3, 'dimensionality_class': 'low'}
        plan = self.engine.plan_detection(
            profile,
            constraints={'exclude_detectors': [
                'IForest', 'KNN', 'LOF', 'CBLOF']})
        if plan.get('detector_name'):  # may be empty if everything excluded
            assert 'contamination' in plan['params'], \
                "fallback plan params must include contamination (TA2)"

    def test_user_supplied_contamination_is_preserved(self):
        # Only the routing layer needs to backfill; if a rule (or a
        # caller through structured feedback) ever sets contamination
        # explicitly, the fix must not overwrite it.
        engine = ADEngine()
        # Simulate by calling _with_contamination directly with an
        # explicit value.
        params = engine._with_contamination(
            'IForest', {'contamination': 0.25})
        assert params['contamination'] == 0.25

    def test_detectors_without_kb_contamination_left_unchanged(self):
        # Conservative behavior: if KB has no contamination default,
        # do not invent one. Pick a detector_name that does not exist
        # so kb.get_algorithm returns None.
        engine = ADEngine()
        params = engine._with_contamination('NotADetector', {})
        assert 'contamination' not in params


class TestBuildDetector(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_build_returns_base_detector(self):
        plan = {'detector_name': 'IForest', 'params': {}}
        clf = self.engine.build_detector(plan)
        assert isinstance(clf, BaseDetector)

    def test_build_with_params(self):
        plan = {'detector_name': 'KNN', 'params': {'n_neighbors': 10}}
        clf = self.engine.build_detector(plan)
        assert clf.n_neighbors == 10

    def test_build_unknown_detector_raises(self):
        plan = {'detector_name': 'NonExistentDetector', 'params': {}}
        with self.assertRaises(ValueError):
            self.engine.build_detector(plan)

    def test_build_planned_detector_raises(self):
        plan = {'detector_name': 'LLMAD', 'params': {}}
        with self.assertRaises(ValueError):
            self.engine.build_detector(plan)


class TestDetectShortcut(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(200, 10)

    def test_detect_returns_result(self):
        result = self.engine.detect(self.X_train)
        assert 'plan' in result
        assert 'scores_train' in result
        assert 'labels_train' in result
        assert 'n_anomalies' in result
        assert 'analysis' in result
        assert len(result['scores_train']) == 200

    def test_detect_with_explicit_type(self):
        result = self.engine.detect(self.X_train, data_type='tabular')
        assert result['plan']['detector_name'] in (
            'IForest', 'ECOD', 'KNN', 'LOF', 'CBLOF', 'HBOS',
            'COPOD', 'INNE')

    def test_detect_compatible_with_tier_b(self):
        """detect() output works with all Tier B methods."""
        result = self.engine.detect(self.X_train)
        # Should not raise
        analysis = self.engine.analyze_results(result)
        assert 'n_anomalies' in analysis
        explanations = self.engine.explain_findings(result, top_k=2)
        assert len(explanations) == 2
        suggestion = self.engine.suggest_next_step(result, analysis)
        assert 'action' in suggestion
        report = self.engine.generate_report(result, analysis)
        assert len(report) > 0


class TestKnowledgeQueries(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_list_detectors(self):
        detectors = self.engine.list_detectors()
        assert len(detectors) >= 40
        names = [d['name'] for d in detectors]
        assert 'ECOD' in names
        assert 'IForest' in names

    def test_list_detectors_by_type(self):
        text_dets = self.engine.list_detectors(data_type='text')
        names = [d['name'] for d in text_dets]
        assert 'EmbeddingOD' in names

    def test_explain_detector(self):
        info = self.engine.explain_detector('ECOD')
        assert info['full_name'] is not None
        assert 'strengths' in info
        assert 'weaknesses' in info

    def test_explain_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.engine.explain_detector('FakeDetector')

    def test_compare_detectors(self):
        comparison = self.engine.compare_detectors(
            names=['ECOD', 'IForest', 'KNN'])
        assert len(comparison) == 3

    def test_get_benchmarks(self):
        benchmarks = self.engine.get_benchmarks()
        assert 'ADBench' in benchmarks

    def test_deep_sad_is_listed_but_not_routed(self):
        # DeepSAD needs labeled anomalies and ADEngine passes no labels,
        # so it is listed as experimental and no routing path selects it.
        info = self.engine.explain_detector('DeepSAD')
        assert info['status'] == 'experimental'
        profile = self.engine.profile_data(np.random.randn(500, 10))
        routing = self.engine.get_kb_for_routing(profile)
        names = [d['name'] for d in routing['available_detectors']]
        assert 'DeepSAD' not in names
        with self.assertRaises(ValueError):
            self.engine.make_plan(['DeepSAD'])


class TestRunDetection(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.rng = np.random.RandomState(42)
        self.X_train = self.rng.randn(200, 10)
        self.X_test = self.rng.randn(50, 10)
        profile = self.engine.profile_data(self.X_train)
        self.plan = self.engine.plan_detection(profile)

    def test_returns_required_keys(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        required = {'plan', 'scores_train', 'labels_train', 'threshold',
                    'n_anomalies', 'anomaly_ratio', 'detector',
                    'runtime_seconds', 'score_summary'}
        for key in required:
            assert key in result, f"Missing key '{key}'"

    def test_scores_shape(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert len(result['scores_train']) == 200
        assert len(result['labels_train']) == 200

    def test_with_test_data(self):
        result = self.engine.run_detection(
            self.X_train, self.plan, X_test=self.X_test)
        assert 'scores_test' in result
        assert 'labels_test' in result
        assert len(result['scores_test']) == 50

    def test_score_summary_has_stats(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        summary = result['score_summary']
        for key in ('mean', 'std', 'min', 'max', 'q25', 'q75'):
            assert key in summary, f"Missing stat '{key}'"

    def test_anomaly_ratio_is_fraction(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert 0.0 <= result['anomaly_ratio'] <= 1.0

    def test_runtime_is_positive(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert result['runtime_seconds'] >= 0.0

    def test_detector_is_fitted(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert hasattr(result['detector'], 'decision_scores_')


class TestAnalyzeResults(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(self.X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(self.X_train, plan)

    def test_returns_required_keys(self):
        analysis = self.engine.analyze_results(self.result)
        required = {'n_anomalies', 'anomaly_ratio', 'score_distribution',
                    'top_anomalies', 'summary'}
        for key in required:
            assert key in analysis, f"Missing key '{key}'"

    def test_top_anomalies_sorted_by_score(self):
        analysis = self.engine.analyze_results(self.result)
        top = analysis['top_anomalies']
        assert len(top) > 0
        scores = [a['score'] for a in top]
        assert scores == sorted(scores, reverse=True)

    def test_top_anomalies_have_index_and_score(self):
        analysis = self.engine.analyze_results(self.result)
        for entry in analysis['top_anomalies']:
            assert 'index' in entry
            assert 'score' in entry

    def test_score_distribution_has_stats(self):
        analysis = self.engine.analyze_results(self.result)
        dist = analysis['score_distribution']
        for key in ('mean', 'std', 'min', 'max', 'median', 'q25', 'q75'):
            assert key in dist

    def test_summary_is_string(self):
        analysis = self.engine.analyze_results(self.result)
        assert isinstance(analysis['summary'], str)
        assert len(analysis['summary']) > 0

    def test_with_feature_data(self):
        analysis = self.engine.analyze_results(self.result, X=self.X_train)
        assert 'n_anomalies' in analysis

    def test_top_k_parameter(self):
        analysis = self.engine.analyze_results(self.result, top_k=3)
        assert len(analysis['top_anomalies']) <= 3


class TestExplainFindings(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(self.X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(self.X_train, plan)

    def test_default_top_k(self):
        explanations = self.engine.explain_findings(self.result)
        assert len(explanations) == 5

    def test_custom_top_k(self):
        explanations = self.engine.explain_findings(self.result, top_k=3)
        assert len(explanations) == 3

    def test_specific_indices(self):
        explanations = self.engine.explain_findings(
            self.result, indices=[0, 5, 10])
        assert len(explanations) == 3
        assert explanations[0]['index'] == 0
        assert explanations[1]['index'] == 5

    def test_entry_has_required_fields(self):
        explanations = self.engine.explain_findings(self.result)
        for entry in explanations:
            assert 'index' in entry
            assert 'score' in entry
            assert 'percentile' in entry
            assert 'narrative' in entry

    def test_percentile_range(self):
        explanations = self.engine.explain_findings(self.result)
        for entry in explanations:
            assert 0.0 <= entry['percentile'] <= 100.0

    def test_narrative_is_string(self):
        explanations = self.engine.explain_findings(self.result)
        for entry in explanations:
            assert isinstance(entry['narrative'], str)
            assert len(entry['narrative']) > 0

    def test_with_feature_data(self):
        explanations = self.engine.explain_findings(
            self.result, X=self.X_train, top_k=2)
        for entry in explanations:
            assert 'contributing_features' in entry

    def test_out_of_range_indices_skipped(self):
        explanations = self.engine.explain_findings(
            self.result, indices=[0, 999, 5])
        assert len(explanations) == 2
        assert explanations[0]['index'] == 0
        assert explanations[1]['index'] == 5

    def test_non_integer_indices_skipped(self):
        explanations = self.engine.explain_findings(
            self.result, indices=[0, 1.5, '2', True, 5])
        # Only 0 and 5 are valid int indices
        assert len(explanations) == 2
        assert explanations[0]['index'] == 0
        assert explanations[1]['index'] == 5

    # ----- O9: enriched contributing_features -----

    def test_contributing_features_have_enriched_keys(self):
        explanations = self.engine.explain_findings(
            self.result, X=self.X_train, top_k=2)
        for entry in explanations:
            for cf in entry.get('contributing_features', []):
                assert {'feature', 'name', 'value', 'mean',
                        'z_score', 'direction'} <= set(cf.keys())
                assert cf['direction'] in ('high', 'low')

    def test_contributing_features_use_provided_names(self):
        names = [f'col_{i}' for i in range(self.X_train.shape[1])]
        explanations = self.engine.explain_findings(
            self.result, X=self.X_train, top_k=1,
            feature_names=names)
        for cf in explanations[0].get('contributing_features', []):
            assert cf['name'].startswith('col_')

    def test_contributing_features_default_name(self):
        explanations = self.engine.explain_findings(
            self.result, X=self.X_train, top_k=1)
        for cf in explanations[0].get('contributing_features', []):
            assert cf['name'].startswith('feature_')

    def test_direction_high_when_value_above_mean(self):
        rng = np.random.RandomState(0)
        X = np.zeros((100, 3))
        X[:, 0] = rng.randn(100)
        X[5, 0] = 10.0  # extreme positive
        scores = np.zeros(100)
        scores[5] = 1.0
        result = {'scores_train': scores, 'threshold': 0.5}
        explanations = self.engine.explain_findings(
            result, indices=[5], X=X)
        cf_top = explanations[0]['contributing_features'][0]
        assert cf_top['feature'] == 0
        assert cf_top['direction'] == 'high'

    def test_direction_low_when_value_below_mean(self):
        rng = np.random.RandomState(0)
        X = np.zeros((100, 3))
        X[:, 1] = rng.randn(100)
        X[5, 1] = -10.0  # extreme negative
        scores = np.zeros(100)
        scores[5] = 1.0
        result = {'scores_train': scores, 'threshold': 0.5}
        explanations = self.engine.explain_findings(
            result, indices=[5], X=X)
        cf_top = explanations[0]['contributing_features'][0]
        assert cf_top['feature'] == 1
        assert cf_top['direction'] == 'low'

    def test_value_and_mean_match_input(self):
        # Sanity check: value and mean should match what the input
        # actually contains, not be invented numbers.
        rng = np.random.RandomState(7)
        X = rng.randn(100, 4)
        scores = np.zeros(100)
        scores[10] = 1.0
        result = {'scores_train': scores, 'threshold': 0.5}
        explanations = self.engine.explain_findings(
            result, indices=[10], X=X)
        for cf in explanations[0]['contributing_features']:
            f = cf['feature']
            assert abs(cf['value'] - float(X[10, f])) < 1e-9
            assert abs(cf['mean'] - float(np.mean(X[:, f]))) < 1e-9


class TestSuggestNextStep(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(X_train, plan)
        self.analysis = self.engine.analyze_results(self.result)

    def test_returns_required_keys(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis)
        assert 'action' in suggestion
        assert 'reason' in suggestion

    def test_too_many_false_positives(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='too many false positives')
        assert suggestion['action'] == 'adjust_threshold'
        assert 'threshold_adjustment' in suggestion
        assert suggestion['threshold_adjustment']['direction'] == 'decrease'

    def test_missed_anomalies(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='missed some anomalies')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'increase'

    def test_try_different_detector(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='switch to a different detector')
        assert suggestion['action'] == 'try_alternative'
        assert 'new_plan' in suggestion

    def test_ensemble_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='try ensemble')
        assert suggestion['action'] == 'try_alternative'
        assert 'new_plan' in suggestion

    def test_lower_threshold_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='lower threshold')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'increase'

    def test_reduce_contamination_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='reduce contamination')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'decrease'

    def test_decrease_threshold_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='decrease threshold')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'increase'

    def test_higher_contamination_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='higher contamination')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'increase'

    def test_increase_threshold_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='increase threshold')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'decrease'

    def test_lower_contamination_feedback(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='lower contamination')
        assert suggestion['action'] == 'adjust_threshold'
        assert suggestion['threshold_adjustment']['direction'] == 'decrease'

    def test_negative_top_k_clamped(self):
        analysis = self.engine.analyze_results(self.result, top_k=-1)
        assert len(analysis['top_anomalies']) == 0

    def test_no_feedback_suggests_done_or_alternative(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis)
        assert suggestion['action'] in ('done', 'try_alternative',
                                         'adjust_threshold')

    def test_new_plan_is_valid(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='switch to a different detector')
        if 'new_plan' in suggestion:
            plan = suggestion['new_plan']
            assert 'detector_name' in plan
            assert plan['detector_name'] != self.result['plan']['detector_name']


class TestGenerateReport(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(X_train, plan)
        self.analysis = self.engine.analyze_results(self.result)

    def test_text_format(self):
        report = self.engine.generate_report(
            self.result, self.analysis, format='text')
        assert isinstance(report, str)
        assert 'Anomaly Detection Report' in report
        assert self.result['plan']['detector_name'] in report

    def test_json_format(self):
        import json
        report = self.engine.generate_report(
            self.result, self.analysis, format='json')
        parsed = json.loads(report)
        assert 'detector' in parsed
        assert 'n_anomalies' in parsed

    def test_report_contains_key_info(self):
        report = self.engine.generate_report(
            self.result, self.analysis, format='text')
        assert 'anomal' in report.lower()
        assert str(self.analysis['n_anomalies']) in report

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            self.engine.generate_report(
                self.result, self.analysis, format='pdf')


class TestRandomStateDeterminism(unittest.TestCase):
    """Regression test for issue #686: ADEngine non-determinism on identical input.

    Before the fix, repeated ``ADEngine().investigate(X)`` calls on
    byte-identical X produced different flagged sets and different
    ``anomaly_ratio`` values because detectors with stochastic internals
    (e.g., IForest random subsample) fell back to numpy's module-level
    random state. The fix wires a ``random_state`` parameter through
    ``ADEngine.__init__`` to ``build_detector_from_plan``, which injects
    it into each detector class that declares an explicit ``random_state``
    parameter.
    """

    def setUp(self):
        self.X = np.random.RandomState(42).randn(200, 5)

    def test_investigate_is_deterministic_with_fixed_seed(self):
        first_labels = None
        ratios = []
        for _ in range(5):
            state = ADEngine(random_state=42).investigate(
                self.X, data_type='tabular')
            labels = np.asarray(state.consensus['labels'])
            ratios.append(
                state.analysis['consensus_analysis']['anomaly_ratio'])
            if first_labels is None:
                first_labels = labels
            else:
                assert np.array_equal(labels, first_labels), (
                    "flagged set drifted across same-seed calls")
        assert len(set(ratios)) == 1, (
            "anomaly_ratio drifted across same-seed calls: %s" % ratios)

    def test_different_seeds_can_differ(self):
        s1 = ADEngine(random_state=1).investigate(
            self.X, data_type='tabular')
        s2 = ADEngine(random_state=2).investigate(
            self.X, data_type='tabular')
        # We do not assert the flagged sets MUST differ (the data is
        # well-separated and consensus may agree), but at minimum the
        # call must succeed for both seeds.
        assert s1.consensus is not None
        assert s2.consensus is not None

    def test_default_constructor_unchanged(self):
        # Backward compatibility: ADEngine() with no seed still works and
        # returns a usable result (no determinism guarantee, see #686).
        state = ADEngine().investigate(self.X, data_type='tabular')
        assert state.consensus is not None
        assert state.analysis is not None

    def test_engine_seed_flows_to_lunar_plan(self):
        # LUNAR is a torch-based stochastic detector. Before #686 plus the
        # LUNAR random_state plumbing, an ADEngine seed was silently dropped
        # for LUNAR plans because LUNAR did not declare random_state in its
        # __init__, so _accepts_random_state() returned False. After the fix,
        # the engine seed reaches LUNAR and same-seed reruns produce
        # bit-identical flagged sets.
        plan = {
            'detector_name': 'LUNAR',
            'params': {
                'n_epochs': 2,
                'n_neighbours': 5,
                'contamination': 0.1,
            },
        }
        labels = []
        for _ in range(3):
            result = ADEngine(random_state=42).run_detection(self.X, plan)
            labels.append(np.asarray(result['labels_train']))
        assert all(np.array_equal(labels[0], arr) for arr in labels[1:])


class TestRandomStateFactory(unittest.TestCase):
    """Lower-level regression tests for the factory contract that
    ``ADEngine.build_detector`` and ``build_detector_from_plan`` use to
    inject ``random_state`` (issue #686). The output-level
    TestRandomStateDeterminism above could keep passing if the consensus
    happens to settle on deterministic detectors while seed injection
    silently regresses; these tests pin the factory behavior directly.
    """

    def setUp(self):
        self.engine = ADEngine(random_state=42)

    def test_seed_is_injected_into_iforest(self):
        clf = self.engine.build_detector({
            'detector_name': 'IForest',
            'params': {},
        })
        assert clf.random_state == 42

    def test_plan_level_seed_wins(self):
        # An explicit plan['params']['random_state'] must override the
        # engine default. Preserves caller intent (e.g., a per-plan seed
        # for a multi-fold cross-validation that overrides the global).
        clf = self.engine.build_detector({
            'detector_name': 'IForest',
            'params': {'random_state': 7},
        })
        assert clf.random_state == 7

    def test_knn_is_not_given_random_state(self):
        # KNN does not declare random_state in its __init__ (#685 cleanup).
        # _accepts_random_state() must refuse to inject and the call must
        # not raise.
        clf = self.engine.build_detector({
            'detector_name': 'KNN',
            'params': {},
        })
        assert not hasattr(clf, 'random_state')

    def test_abod_is_not_given_random_state(self):
        clf = self.engine.build_detector({
            'detector_name': 'ABOD',
            'params': {},
        })
        assert not hasattr(clf, 'random_state')

    def test_sod_is_not_given_random_state(self):
        clf = self.engine.build_detector({
            'detector_name': 'SOD',
            'params': {},
        })
        assert not hasattr(clf, 'random_state')

    def test_input_plan_not_mutated(self):
        # build_detector_from_plan does dict(plan['params']) before adding
        # random_state. The caller's plan must not pick up an unwanted
        # random_state key after the call.
        plan = {'detector_name': 'IForest', 'params': {}}
        _ = self.engine.build_detector(plan)
        assert 'random_state' not in plan.get('params', {})

    def test_no_seed_when_engine_random_state_is_none(self):
        # An ADEngine constructed without random_state must not inject one,
        # preserving v3.5.1 behavior end-to-end.
        engine = ADEngine()
        clf = engine.build_detector({
            'detector_name': 'IForest',
            'params': {},
        })
        # IForest's default random_state is None.
        assert clf.random_state is None

    def test_seed_propagates_through_embedding_preset(self):
        # Codex round-2 finding: build_from_preset bypassed the seed
        # injection because the preset branch returned before random_state
        # was added to params. EmbeddingOD.for_text() defaults to LUNAR
        # internally, so an ADEngine seed silently dropped on text routes
        # before the fix. After the round-2 fix, the engine seed reaches
        # the preset and the inner LUNAR via build_from_preset -> for_text.
        clf = ADEngine(random_state=42).build_detector({
            'detector_name': 'EmbeddingOD',
            'preset': 'for_text',
            'params': {'quality': 'balanced'},
        })
        assert clf.random_state == 42

    def test_preset_plan_level_seed_wins(self):
        clf = ADEngine(random_state=42).build_detector({
            'detector_name': 'EmbeddingOD',
            'preset': 'for_text',
            'params': {'quality': 'balanced', 'random_state': 99},
        })
        assert clf.random_state == 99

    def test_preset_no_seed_unchanged(self):
        # An ADEngine constructed without random_state must not inject one
        # on the preset path either.
        clf = ADEngine().build_detector({
            'detector_name': 'EmbeddingOD',
            'preset': 'for_text',
            'params': {'quality': 'balanced'},
        })
        assert clf.random_state is None

    def test_embedding_preset_seed_reaches_pca_preprocessing(self):
        # Codex round-3 finding: EmbeddingOD._preprocess_fit constructed
        # PCA without random_state, so a preset plan with reduce_dim could
        # still be non-deterministic in the preprocessing step even with
        # the inner detector seeded. The fix forwards EmbeddingOD's
        # random_state to PCA. We monkeypatch the PCA constructor so we
        # do not need to fit a real embedding pipeline in CI.
        from unittest.mock import patch
        captured = {}

        class _FakePCA:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_transform(self, X):
                return X[:, :captured.get('n_components', X.shape[1])]

            def transform(self, X):
                return X[:, :captured.get('n_components', X.shape[1])]

        import pyod.models.embedding as _emb
        from pyod.models.embedding import EmbeddingOD
        # Drive the preprocessing branch directly so the test does not
        # depend on a real encoder.
        clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', reduce_dim=2,
                          random_state=42)
        X_emb = np.random.RandomState(0).randn(20, 8).astype(np.float32)
        with patch.object(_emb, 'PCA', _FakePCA):
            clf._preprocess_fit(X_emb)
        assert captured.get('random_state') == 42


if __name__ == '__main__':
    unittest.main()
