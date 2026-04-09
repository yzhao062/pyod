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

    def test_planned_detector_filtered_out(self):
        profile = {'data_type': 'time_series', 'n_samples': 1000,
                   'n_features': 1}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] != 'TimeSeriesOD'

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
        plan = {'detector_name': 'TimeSeriesOD', 'params': {}}
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


if __name__ == '__main__':
    unittest.main()
