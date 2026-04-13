# -*- coding: utf-8 -*-
"""Tests for TimeSeriesOD."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_od import TimeSeriesOD
from pyod.utils.data import generate_ts_data


class TestTimeSeriesOD(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=500, n_test=200, contamination=0.05,
                             random_state=42)
        _, _, _, _ = generate_ts_data(
            n_train=500, n_test=200, n_channels=3, contamination=0.05,
            random_state=42)
        self.X_multi = generate_ts_data(
            n_train=500, n_test=200, n_channels=3, contamination=0.05,
            random_state=42)[0]

    def test_fit_univariate(self):
        clf = TimeSeriesOD(window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert hasattr(clf, 'decision_scores_')
        assert len(clf.decision_scores_) == 500
        assert hasattr(clf, 'labels_')

    def test_fit_multivariate(self):
        clf = TimeSeriesOD(window_size=20)
        clf.fit(self.X_multi)
        assert len(clf.decision_scores_) == 500

    def test_decision_function(self):
        clf = TimeSeriesOD(window_size=20)
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert len(scores) == 200

    def test_predict(self):
        clf = TimeSeriesOD(window_size=20)
        clf.fit(self.X_train)
        labels = clf.predict(self.X_test)
        assert len(labels) == 200
        assert set(labels).issubset({0, 1})

    def test_string_detector(self):
        clf = TimeSeriesOD(detector='ECOD', window_size=20)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_detector_instance(self):
        from pyod.models.iforest import IForest
        clf = TimeSeriesOD(detector=IForest(), window_size=20)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            clf = TimeSeriesOD(window_size=100)
            clf.fit(np.random.randn(50))

    def test_score_aggregation_mean(self):
        clf = TimeSeriesOD(window_size=20, score_aggregation='mean')
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_step_parameter(self):
        clf = TimeSeriesOD(window_size=20, step=5)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500


if __name__ == '__main__':
    unittest.main()
