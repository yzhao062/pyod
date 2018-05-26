import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from sklearn.utils import shuffle
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.metrics import roc_auc_score

import numpy as np
from pyod.models.combination import aom
from pyod.models.combination import moa


class TestAOM(unittest.TestCase):
    def setUp(self):
        self.scores = np.asarray([[0.5, 0.8, 0.6, 0.9, 0.7, 0.6],
                                  [0.8, 0.75, 0.25, 0.6, 0.45, 0.8],
                                  [0.8, 0.3, 0.28, 0.99, 0.28, 0.3],
                                  [0.74, 0.85, 0.38, 0.47, 0.27, 0.69]])

    def test_aom_static_norepeat(self):
        score = aom(self.scores, 3, method='static', replace=False,
                    random_state=42)

        assert_equal(score.shape, (4,))

        shuffled_list = shuffle(list(range(0, 6, 1)), random_state=42)
        manual_scores = np.zeros([4, 3])
        manual_scores[:, 0] = np.max(self.scores[:, shuffled_list[0:2]],
                                     axis=1)
        manual_scores[:, 1] = np.max(self.scores[:, shuffled_list[2:4]],
                                     axis=1)
        manual_scores[:, 2] = np.max(self.scores[:, shuffled_list[4:6]],
                                     axis=1)

        manual_score = np.mean(manual_scores, axis=1)
        assert_array_equal(score, manual_score)

    def test_aom_static_repeat(self):
        score = aom(self.scores, 3, method='static', replace=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

        # TODO: add more complicated testcases

    def test_aom_dynamic_repeat(self):
        score = aom(self.scores, 3, method='dynamic', replace=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

        # TODO: add more complicated testcases

    def tearDown(self):
        pass


class TestMOA(unittest.TestCase):
    def setUp(self):
        self.scores = np.asarray([[0.5, 0.8, 0.6, 0.9, 0.7, 0.6],
                                  [0.8, 0.75, 0.25, 0.6, 0.45, 0.8],
                                  [0.8, 0.3, 0.28, 0.99, 0.28, 0.3],
                                  [0.74, 0.85, 0.38, 0.47, 0.27, 0.69]])

    def test_moa_static_norepeat(self):
        score = moa(self.scores, 3, method='static', replace=False,
                    random_state=42)

        assert_equal(score.shape, (4,))

        shuffled_list = shuffle(list(range(0, 6, 1)), random_state=42)
        manual_scores = np.zeros([4, 3])
        manual_scores[:, 0] = np.mean(self.scores[:, shuffled_list[0:2]],
                                      axis=1)
        manual_scores[:, 1] = np.mean(self.scores[:, shuffled_list[2:4]],
                                      axis=1)
        manual_scores[:, 2] = np.mean(self.scores[:, shuffled_list[4:6]],
                                      axis=1)

        manual_score = np.max(manual_scores, axis=1)
        assert_array_equal(score, manual_score)

    def test_moa_static_repeat(self):
        score = moa(self.scores, 3, method='static', replace=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

        # TODO: add more complicated testcases

    def test_moa_dynamic_repeat(self):
        score = moa(self.scores, 3, method='dynamic', replace=True,
                    random_state=42)
        assert_equal(score.shape, (4,))

        # TODO: add more complicated testcases

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
