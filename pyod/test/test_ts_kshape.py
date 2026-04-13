# -*- coding: utf-8 -*-
"""Tests for KShape."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_kshape import KShape
from pyod.utils.data import generate_ts_data


class TestKShape(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=300, n_test=200, contamination=0.05,
                             random_state=42)

    def test_fit(self):
        clf = KShape(n_clusters=3, window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300

    def test_decision_function(self):
        clf = KShape(n_clusters=3, window_size=20)
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert len(scores) == 200

    def test_multivariate(self):
        X_multi = generate_ts_data(
            n_train=300, n_test=200, n_channels=2, contamination=0.05,
            random_state=42)[0]
        clf = KShape(n_clusters=3, window_size=20)
        clf.fit(X_multi)
        assert len(clf.decision_scores_) == 300

    def test_has_labels_and_threshold(self):
        clf = KShape(n_clusters=3, window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.labels_) == 300

    def test_scores_nonnegative(self):
        clf = KShape(n_clusters=3, window_size=20)
        clf.fit(self.X_train)
        # SBD is in [0, 2], so scores should be non-negative
        assert np.all(clf.decision_scores_ >= 0)

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            clf = KShape(window_size=50)
            clf.fit(np.random.randn(30))

    def test_kshape_centroid_correctness(self):
        """Identical members should produce a centroid with near-zero SBD."""
        from pyod.models.ts_kshape import _znormalize, _sbd, _compute_centroid
        x = _znormalize(np.sin(np.arange(20, dtype=np.float64)))
        members = np.vstack([x, x, x])
        centroid = _compute_centroid(members)
        dist, _ = _sbd(x, centroid)
        assert dist < 0.1, f"Centroid SBD too high: {dist}"

    def test_kshape_too_few_subsequences_raises(self):
        with self.assertRaises(ValueError):
            clf = KShape(n_clusters=10, window_size=20)
            clf.fit(np.random.randn(25))


if __name__ == '__main__':
    unittest.main()
