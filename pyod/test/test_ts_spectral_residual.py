# -*- coding: utf-8 -*-
"""Tests for SpectralResidual."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_spectral_residual import SpectralResidual
from pyod.utils.data import generate_ts_data


class TestSpectralResidual(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=500, n_test=200, contamination=0.05,
                             random_state=42)

    def test_fit(self):
        clf = SpectralResidual(contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_decision_function(self):
        clf = SpectralResidual()
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert len(scores) == 200

    def test_multivariate(self):
        X_multi = generate_ts_data(
            n_train=500, n_test=200, n_channels=3, contamination=0.05,
            random_state=42)[0]
        clf = SpectralResidual()
        clf.fit(X_multi)
        assert len(clf.decision_scores_) == 500

    def test_scores_nonnegative(self):
        clf = SpectralResidual()
        clf.fit(self.X_train)
        assert np.all(clf.decision_scores_ >= 0)

    def test_dense_no_nan(self):
        clf = SpectralResidual()
        clf.fit(self.X_train)
        assert not np.any(np.isnan(clf.decision_scores_))


if __name__ == '__main__':
    unittest.main()
