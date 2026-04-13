# -*- coding: utf-8 -*-
"""Tests for SAND."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_sand import SAND
from pyod.utils.data import generate_ts_data


class TestSAND(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=500, n_test=200, contamination=0.05,
                             random_state=42)

    def test_fit(self):
        clf = SAND(n_clusters=3, window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_decision_function(self):
        clf = SAND(n_clusters=3, window_size=20)
        clf.fit(self.X_train)
        scores = clf.decision_function(
            generate_ts_data(n_train=200, n_test=100, contamination=0.05,
                             random_state=42)[0])
        assert len(scores) == 200

    def test_drift_adaptation(self):
        clf = SAND(n_clusters=3, window_size=20, batch_size=50)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_sand_channel_mismatch_raises(self):
        X_multi = generate_ts_data(
            n_train=300, n_test=100, n_channels=2, contamination=0.05,
            random_state=42)[0]
        clf = SAND(n_clusters=3, window_size=20)
        clf.fit(X_multi)
        with self.assertRaises(ValueError):
            clf.decision_function(np.random.RandomState(0).randn(100, 3))

    def test_sand_too_few_init_batch_raises(self):
        with self.assertRaises(ValueError):
            clf = SAND(n_clusters=10, window_size=20, batch_size=5)
            clf.fit(np.random.randn(100))


if __name__ == '__main__':
    unittest.main()
