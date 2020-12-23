# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.utils.stat_models import wpearsonr
from pyod.utils.stat_models import pearsonr_mat


class TestStatModels(unittest.TestCase):
    def setUp(self):
        self.a = [1, 2, 3, 2, 3, 1, 0, 5]
        self.b = [1, 2, 1, 2, 2, 1, 0, 2]
        self.w = [2, 2, 1, 2, 4, 1, 0, 2]

        self.mat = np.random.rand(10, 20)
        self.w_mat = np.random.rand(10, 1)

        self.X = np.array([[1, 2, 3],
                           [3, 4, 5],
                           [3, 6, 7],
                           [4, 1, 1]])
        self.Y = np.array([[2, 2, 2],
                           [3, 3, 3],
                           [4, 4, 3],
                           [0, 1, 2]])

    def test_pairwise_distances_no_broadcast(self):
        assert_allclose(pairwise_distances_no_broadcast(self.X, self.Y),
                        [1.41421356, 2.23606798, 4.58257569, 4.12310563])

        with assert_raises(ValueError):
            pairwise_distances_no_broadcast([1, 2, 3], [6])

    def test_wpearsonr(self):
        # TODO: if unweight version changes, wp[0] format should be changed
        wp = wpearsonr(self.a, self.b)
        assert_allclose(wp[0], 0.6956083, atol=0.01)

        wp = wpearsonr(self.a, self.b, w=self.w)
        assert_allclose(wp, 0.5477226, atol=0.01)

    def test_pearsonr_mat(self):
        pear_mat = pearsonr_mat(self.mat)
        assert_equal(pear_mat.shape, (10, 10))

        pear_mat = pearsonr_mat(self.mat, self.w_mat)
        assert_equal(pear_mat.shape, (10, 10))

        assert (np.min(pear_mat) >= -1)
        assert (np.max(pear_mat) <= 1)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
