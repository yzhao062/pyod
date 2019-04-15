# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
from sklearn.utils.testing import assert_equal
# noinspection PyProtectedMember
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data import generate_data
from utils.data import evaluate_print
from utils.data import get_outliers_inliers
from utils.data import check_consistent_shape
from utils.data import generate_data_clusters

class TestData(unittest.TestCase):
    def setUp(self):
        self.n_train = 1000
        self.n_test = 500
        self.contamination = 0.1
        self.n_samples = 1000
        self.test_size = 0.2
        self.value_lists = [0.1, 0.3, 0.2, -2, 1.5, 0, 1, -1, -0.5, 11]

    def test_data_generate(self):
        X_train, y_train, X_test, y_test = \
            generate_data(n_train=self.n_train,
                          n_test=self.n_test,
                          contamination=self.contamination)

        assert_equal(y_train.shape[0], X_train.shape[0])
        assert_equal(y_test.shape[0], X_test.shape[0])

        assert_less_equal(self.n_train - X_train.shape[0], 1)
        assert_equal(X_train.shape[1], 2)

        assert_less_equal(self.n_test - X_test.shape[0], 1)
        assert_equal(X_test.shape[1], 2)

        out_perc = np.sum(y_train) / self.n_train
        assert_allclose(self.contamination, out_perc, atol=0.01)

        out_perc = np.sum(y_test) / self.n_test
        assert_allclose(self.contamination, out_perc, atol=0.01)

    def test_data_generate2(self):
        X_train, y_train, X_test, y_test = \
            generate_data(n_train=self.n_train,
                          n_test=self.n_test,
                          n_features=3,
                          contamination=self.contamination)
        assert_allclose(X_train.shape, (self.n_train, 3))
        assert_allclose(X_test.shape, (self.n_test, 3))

    def test_data_generate3(self):
        X_train, y_train, X_test, y_test = \
            generate_data(n_train=self.n_train,
                          n_test=self.n_test,
                          n_features=2,
                          contamination=self.contamination,
                          random_state=42)

        X_train2, y_train2, X_test2, y_test2 = \
            generate_data(n_train=self.n_train,
                          n_test=self.n_test,
                          n_features=2,
                          contamination=self.contamination,
                          random_state=42)

        assert_allclose(X_train, X_train2)
        assert_allclose(X_test, X_test2)
        assert_allclose(y_train, y_train2)
        assert_allclose(y_test, y_test2)

    def test_data_generate_clusters(self):
        x_train, y_train, x_test, y_test = \
            generate_data_clusters(n_samples=self.n_samples,
                                   test_size=self.test_size,
                                   n_features=2,
                                   contamination=self.contamination,
                                   random_state=0,
                                   return_in_clusters=False)

        assert_equal(y_train.shape[0], x_train.shape[0])
        assert_equal(y_test.shape[0], x_test.shape[0])

        assert_less_equal(self.n_samples - x_train.shape[0], x_test.shape[0])
        assert_equal(x_train.shape[1], 2)

        assert_less_equal((self.test_size * x_train.shape[0]) - x_test.shape[0], 1)
        assert_equal(x_test.shape[1], 2)

        out_perc = np.add(np.sum(y_train), np.sum(y_test)) / self.n_samples
        assert_allclose(self.contamination, out_perc, atol=0.01)

        x_train2, y_train2, x_test2, y_test2 = \
            generate_data_clusters(n_samples=self.n_samples,
                                   test_size=self.test_size,
                                   n_features=2,
                                   contamination=self.contamination,
                                   random_state=42,
                                   return_in_clusters=False)

        assert_equal(x_train.shape[0], x_train2.shape[0])
        assert_equal(x_test.shape[0], x_test2.shape[0])
        assert_equal(y_train.shape[0], y_train2.shape[0])
        assert_equal(y_test.shape[0], y_test2.shape[0])

        x_train3, y_train3, x_test3, y_test3 = \
            generate_data_clusters(n_samples=self.n_samples,
                                   test_size=self.test_size,
                                   n_features=2,
                                   n_clusters=3,
                                   contamination=self.contamination,
                                   random_state=10,
                                   return_in_clusters=True)
        # if return_in_clusters: it returns a list of numpy arrays
        # where each index represents a cluster
        assert_equal(len(x_train3), len(x_test3), 3)
        assert_equal(len(y_train3), len(y_test3), 3)

        with assert_raises(ValueError):
            generate_data_clusters(n_samples=-1,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=-1,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=-1,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=0,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.9,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='something',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='something',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=1.1,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters='yes')
        with assert_raises(ValueError):
            generate_data_clusters(n_samples='not int',
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size='not float',
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters='not int',
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features='not int',
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination='not float',
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=20,
                                   n_features=2,
                                   contamination=0.1,
                                   size='different',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=0.25,
                                   n_clusters=9,
                                   n_features=2,
                                   contamination=0.4,
                                   size='different',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)
        with assert_raises(ValueError):
            generate_data_clusters(n_samples=100,
                                   test_size=1.1,
                                   n_clusters=2,
                                   n_features=2,
                                   contamination=0.1,
                                   size='same',
                                   density='same',
                                   dist=0.25,
                                   random_state=None,
                                   return_in_clusters=False)

    def test_evaluate_print(self):
        X_train, y_train, X_test, y_test = generate_data(
            n_train=self.n_train,
            n_test=self.n_test,
            contamination=self.contamination)
        evaluate_print('dummy', y_train, y_train * 0.1)

    def test_get_outliers_inliers(self):
        X_train, y_train = generate_data(
            n_train=self.n_train, train_only=True,
            contamination=self.contamination)

        X_outliers, X_inliers = get_outliers_inliers(X_train, y_train)

        inlier_index = int(self.n_train * (1 - self.contamination))

        assert_allclose(X_train[0:inlier_index, :], X_inliers)
        assert_allclose(X_train[inlier_index:, :], X_outliers)

    def test_check_consistent_shape(self):
        X_train, y_train, X_test, y_test = generate_data(
            n_train=self.n_train,
            n_test=self.n_test,
            contamination=self.contamination)

        X_train_n, y_train_n, X_test_n, y_test_n, y_train_pred_n, y_test_pred_n \
            = check_consistent_shape(X_train, y_train, X_test, y_test,
                                     y_train, y_test)

        assert_allclose(X_train_n, X_train)
        assert_allclose(y_train_n, y_train)
        assert_allclose(X_test_n, X_test)
        assert_allclose(y_test_n, y_test)
        assert_allclose(y_train_pred_n, y_train)
        assert_allclose(y_test_pred_n, y_test)

        # test shape difference
        with assert_raises(ValueError):
            check_consistent_shape(X_train, y_train, y_train, y_test,
                                   y_train, y_test)

    def tearDown(self):
        pass

    if __name__ == '__main__':
        unittest.main()
