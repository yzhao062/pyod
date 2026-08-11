"""Tests for pyod.models.small_n.SmallN, matching the structure of
pyod/test/test_mcd.py.
"""

import unittest

import numpy as np
from numpy.testing import assert_equal

from pyod.models.small_n import SmallN


class TestSmallN(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1.0, 2.1], [1.1, 1.9], [0.9, 2.0]])
        self.X_test = np.array([[1.0, 2.0], [8.0, 8.0]])
        self.clf = SmallN(contamination=0.3)
        self.clf.fit(self.X_train)

    def test_fit(self):
        assert hasattr(self.clf, 'decision_scores_')
        assert_equal(len(self.clf.decision_scores_), len(self.X_train))

    def test_decision_function_shape(self):
        scores = self.clf.decision_function(self.X_test)
        assert_equal(len(scores), len(self.X_test))

    def test_outlier_scores_higher(self):
        scores = self.clf.decision_function(self.X_test)
        assert scores[1] > scores[0]

    def test_predict_labels(self):
        labels = self.clf.predict(self.X_test)
        assert_equal(len(labels), len(self.X_test))
        assert set(labels.tolist()).issubset({0, 1})

    def test_wrong_feature_count_raises(self):
        with self.assertRaises(ValueError):
            self.clf.decision_function(np.array([[1.0], [2.0]]))

    def test_degenerate_reference_set_still_flags_outliers(self):
        X_degenerate = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        clf = SmallN(contamination=0.3)
        clf.fit(X_degenerate)
        score = clf.decision_function(np.array([[100.0, 100.0]]))
        assert score[0] > 0

    def test_rejection_stats_raises_clear_error_for_small_n(self):
        with self.assertRaises(ValueError):
            self.clf.compute_rejection_stats()

    def test_unify_proba_no_nan_on_degenerate_reference(self):
        X_degenerate = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        clf = SmallN(contamination=0.3)
        clf.fit(X_degenerate)
        probs = clf.predict_proba(
            np.array([[1.0, 2.0], [100.0, 100.0]]), method='unify')
        assert not np.isnan(probs).any()


class TestSmallNRouting(unittest.TestCase):
    """Covers ADEngine routing behavior specific to SmallN, not just the
    detector class in isolation.
    """

    def test_tiny_low_dim_routes_to_small_n(self):
        from pyod.utils.ad_engine import ADEngine
        X = np.array([[1.0, 2.1], [1.1, 1.9], [0.9, 2.0]])
        result = ADEngine().detect(X)
        self.assertEqual(result['plan']['detector_name'], 'SmallN')
        alt_names = [a['detector_name']
                     for a in result['plan']['alternatives']]
        self.assertNotIn('KNN', alt_names)

    def test_tiny_high_dim_does_not_route_to_small_n(self):
        from pyod.utils.ad_engine import ADEngine
        X = np.random.RandomState(0).randn(4, 2000)
        result = ADEngine().detect(X)
        self.assertNotEqual(result['plan']['detector_name'], 'SmallN')


if __name__ == '__main__':
    unittest.main()
