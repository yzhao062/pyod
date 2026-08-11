"""Tests for pyod.models.small_n.SmallN.
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


if __name__ == '__main__':
    unittest.main()
