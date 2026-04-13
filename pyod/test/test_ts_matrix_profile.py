# -*- coding: utf-8 -*-
"""Tests for MatrixProfile."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_matrix_profile import MatrixProfile
from pyod.utils.data import generate_ts_data


class TestMatrixProfile(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=300, n_test=100, contamination=0.05,
                             random_state=42)

    def test_fit(self):
        clf = MatrixProfile(window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')

    def test_multivariate(self):
        X_multi = generate_ts_data(
            n_train=300, n_test=100, n_channels=3, contamination=0.05,
            random_state=42)[0]
        clf = MatrixProfile(window_size=20)
        clf.fit(X_multi)
        assert len(clf.decision_scores_) == 300

    def test_transductive_no_decision_function(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(np.random.randn(100))

    def test_transductive_no_predict(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        with self.assertRaises(NotImplementedError):
            clf.predict(np.random.randn(100))

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            clf = MatrixProfile(window_size=50)
            clf.fit(np.random.randn(30))

    def test_anomaly_scores_nonnegative(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        # After threshold fill, some boundary scores equal threshold
        # All valid MP scores should be >= 0
        assert np.all(clf.decision_scores_ >= 0)


if __name__ == '__main__':
    unittest.main()
