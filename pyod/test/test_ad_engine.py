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
        assert 'scores' in result
        assert 'labels' in result
        assert 'n_anomalies' in result
        assert len(result['scores']) == 200

    def test_detect_with_explicit_type(self):
        result = self.engine.detect(self.X_train, data_type='tabular')
        assert result['plan']['detector_name'] in (
            'IForest', 'ECOD', 'KNN', 'LOF', 'CBLOF', 'HBOS',
            'COPOD', 'INNE')


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


if __name__ == '__main__':
    unittest.main()
