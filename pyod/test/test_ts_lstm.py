# -*- coding: utf-8 -*-
"""Tests for LSTMAD."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_ts_data


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


class TestLSTMAD(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=300, n_test=200, contamination=0.05,
                             random_state=42)

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_fit(self):
        from pyod.models.ts_lstm import LSTMAD
        clf = LSTMAD(window_size=20, epochs=2, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_decision_function(self):
        from pyod.models.ts_lstm import LSTMAD
        clf = LSTMAD(window_size=20, epochs=2)
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert len(scores) == 200

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_multivariate(self):
        from pyod.models.ts_lstm import LSTMAD
        X_multi = generate_ts_data(
            n_train=300, n_test=200, n_channels=3, contamination=0.05,
            random_state=42)[0]
        clf = LSTMAD(window_size=20, epochs=2)
        clf.fit(X_multi)
        assert len(clf.decision_scores_) == 300

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_causal_boundary(self):
        from pyod.models.ts_lstm import LSTMAD
        clf = LSTMAD(window_size=20, epochs=2)
        clf.fit(self.X_train)
        # First window_size scores should be threshold-filled
        assert np.allclose(clf.decision_scores_[:20], clf.threshold_)


if __name__ == '__main__':
    unittest.main()
