# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
from pyod.utils.data import generate_data_categorical

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.data import get_outliers_inliers
from pyod.utils.data import check_consistent_shape
from pyod.utils.data import generate_data_clusters


class TestData(unittest.TestCase):
    def setUp(self):
        self.n_train = 1000
        self.n_test = 500
        self.contamination = 0.1
        self.n_samples = 1000
        self.test_size = 0.2
        self.value_lists = [0.1, 0.3, 0.2, -2, 1.5, 0, 1, -1, -0.5, 11]
        self.random_state = 42

    def test_data_generate(self):
        X_train, X_test, y_train, y_test = \
            generate_data(n_train=self.n_train,
                          n_test=self.n_test,
                          contamination=self.contamination)

        assert_equal(y_train.shape[0], X_train.shape[0])
        assert_equal(y_test.shape[0], X_test.shape[0])

        assert (self.n_train - X_train.shape[0] <= 1)
        assert_equal(X_train.shape[1], 2)

        assert (self.n_test - X_test.shape[0] <= 1)
        assert_equal(X_test.shape[1], 2)

        out_perc = np.sum(y_train) / self.n_train
        assert_allclose(self.contamination, out_perc, atol=0.01)

        out_perc = np.sum(y_test) / self.n_test
        assert_allclose(self.contamination, out_perc, atol=0.01)

    def test_data_generate2(self):
        X_train, X_test, y_train, y_test = \
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

    def test_data_generate_outliers_have_spread(self):
        # Regression test for GH #141: for certain seeds the internal offset
        # was drawn as 0, which collapsed every outlier onto the origin
        # (uniform(-0, 0) == 0) and produced zero-variance outliers. Sweep a
        # range of seeds (41, 48 and 50 previously triggered the collapse)
        # and confirm outliers always keep a non-zero spread.
        for seed in range(60):
            X, y = generate_data(
                n_features=2,
                contamination=0.05,
                train_only=True,
                random_state=seed,
            )
            outliers = X[y == 1]
            assert outliers.var(axis=0).min() > 0, \
                "outliers collapsed to zero variance for random_state=%d" % seed

    def test_data_generate_float_offset(self):
        # A float offset in (1, 2) used to raise ValueError, because randint
        # truncates its bounds and the redraw got low == high. It is now drawn
        # continuously with a floor of 1.0. Assert separation, not merely a
        # non-zero variance: a spread test alone would pass on data whose
        # labelled outliers sit inside the inlier cloud.
        for offset in (1.1, 1.5, np.nextafter(2.0, 1.0)):
            with self.subTest(offset=offset):
                for seed in range(5):
                    X, y = generate_data(
                        n_features=2,
                        contamination=0.05,
                        train_only=True,
                        offset=offset,
                        random_state=seed,
                    )
                    outliers = X[y == 1]
                    assert outliers.var(axis=0).min() > 0
                    centrality = np.linalg.norm(X - X.mean(axis=0), axis=1)
                    assert roc_auc_score(y, centrality) > 0.7, \
                        "outliers are not separated for offset=%r seed=%d" % (
                            offset, seed)

    def test_data_generate_redraw_branches_stable(self):
        # The other golden test pins three default-offset seeds whose first
        # draw is non-zero, so it exercises neither branch the GH #141 fix
        # actually touches. Both seeds below take the zero-then-redraw path
        # (the first randint returns 0): offset=1 hits the fixed offset_=1
        # branch, and offset=2 hits the integer redraw.
        #
        # Pin an array-wide sum alongside the first and last rows, with a
        # tolerance, rather than hashing the raw bytes. A byte-exact pin is
        # not portable: randn draws Gaussians by the polar method, whose log
        # is not required to be correctly rounded, so libm implementations
        # disagree in the last ulp and the same seed yields different low bits
        # on macOS than on Linux or Windows. The sum still covers every element,
        # so an inserted or reordered RNG draw, which moves values by O(1),
        # cannot hide inside this tolerance.
        golden = {
            (1, 0): (1888.646949950,
                     [1.407737153, 1.853812935], [0.716955137, -0.505085628]),
            (2, 0): (1907.119693504,
                     [1.070368485, 1.067679162], [-0.467865797, 0.251365755]),
        }
        for (offset, seed), (total, first, last) in golden.items():
            with self.subTest(offset=offset, seed=seed):
                X, _ = generate_data(n_features=2, contamination=0.05,
                                     train_only=True, offset=offset,
                                     random_state=seed)[:2]
                assert_equal(X.shape, (1000, 2))
                assert_allclose(X.sum(), total, rtol=0, atol=1e-6)
                assert_allclose(X[0], first, rtol=0, atol=1e-9)
                assert_allclose(X[-1], last, rtol=0, atol=1e-9)

    def test_data_generate_offset_below_one_rejected(self):
        # An offset below 1 must keep raising. coef_, the inlier spread, is
        # drawn from [0.001, 1.001) independently of offset, so an outlier box
        # of half-width < 1 lands inside an O(1) inlier cloud and the labelled
        # outliers become the densest points in the sample.
        for offset in (0.01, 0.5, np.nextafter(1.0, 0.0)):
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    generate_data(train_only=True, offset=offset,
                                  random_state=0)

    def test_data_generate_reproducibility(self):
        # Golden values pinned from the pre-fix implementation for seeds whose
        # offset was already non-zero. Redrawing only when the offset comes out
        # as 0 leaves these untouched, so this guards against a future change
        # silently altering long-standing fixed-seed output.
        golden = {
            0: ([5.059894904, 5.061739412], [-0.263919547, 3.009107520]),
            1: ([3.401186423, 3.524852969], [0.181525489, 3.650202520]),
            42: ([6.433658544, 5.509168303], [-3.206743915, -4.912722786]),
        }
        for seed, (first, last) in golden.items():
            X, _ = generate_data(n_train=10, n_test=5, n_features=2,
                                 contamination=0.2, train_only=True,
                                 random_state=seed)
            assert_allclose(X[0], first, rtol=0, atol=1e-9)
            assert_allclose(X[-1], last, rtol=0, atol=1e-9)

    def test_data_generate_cluster(self):
        X_train, X_test, y_train, y_test = \
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=2,
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        assert_equal(y_train.shape[0], X_train.shape[0])
        assert_equal(y_test.shape[0], X_test.shape[0])

        assert (self.n_train - X_train.shape[0] <= 1)
        assert_equal(X_train.shape[1], 2)

        assert (self.n_test - X_test.shape[0] <= 1)
        assert_equal(X_test.shape[1], 2)

        out_perc = (np.sum(y_train) + np.sum(y_test)) / (
                self.n_train + self.n_test)
        assert_allclose(self.contamination, out_perc, atol=0.01)

    def test_data_generate_cluster2(self):
        X_train, X_test, y_train, y_test = \
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=4,
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        assert_allclose(X_train.shape, (self.n_train, 4))
        assert_allclose(X_test.shape, (self.n_test, 4))

    def test_data_generate_cluster3(self):
        X_train, y_train, X_test, y_test = \
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=3,
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        X_train2, y_train2, X_test2, y_test2 = \
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=3,
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        assert_allclose(X_train, X_train2)
        assert_allclose(X_test, X_test2)
        assert_allclose(y_train, y_train2)
        assert_allclose(y_test, y_test2)

    def test_data_generate_cluster5(self):
        with assert_raises(ValueError):
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=3,
                                   n_clusters='e',
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features='e',
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=3,
                                   contamination='e',
                                   random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=3,
                                   contamination=self.contamination,
                                   dist='e',
                                   random_state=self.random_state)

    def test_data_generate_cluster6(self):
        X_train, X_test, y_train, y_test = \
            generate_data_clusters(n_train=self.n_train,
                                   n_test=self.n_test,
                                   n_features=2,
                                   size='different',
                                   density='different',
                                   contamination=self.contamination,
                                   random_state=self.random_state)

        assert_equal(y_train.shape[0], X_train.shape[0])
        assert_equal(y_test.shape[0], X_test.shape[0])

        assert (self.n_train - X_train.shape[0] <= 1)
        assert_equal(X_train.shape[1], 2)

        assert (self.n_test - X_test.shape[0] <= 1)
        assert_equal(X_test.shape[1], 2)

        out_perc = (np.sum(y_train) + np.sum(y_test)) / (
                self.n_train + self.n_test)
        assert_allclose(self.contamination, out_perc, atol=0.01)

    def test_data_generate_categorical(self):
        X_train, X_test, y_train, y_test = \
            generate_data_categorical(n_train=self.n_train,
                                      n_test=self.n_test,
                                      n_features=2,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        assert_equal(y_train.shape[0], X_train.shape[0])
        assert_equal(y_test.shape[0], X_test.shape[0])

        assert (self.n_train - X_train.shape[0] <= 1)
        assert_equal(X_train.shape[1], 2)

        assert (self.n_test - X_test.shape[0] <= 1)
        assert_equal(X_test.shape[1], 2)

        out_perc = (np.sum(y_train) + np.sum(y_test)) / (
                self.n_train + self.n_test)
        assert_allclose(self.contamination, out_perc, atol=0.01)

    def test_data_generate_categorical2(self):
        X_train, X_test, y_train, y_test = \
            generate_data_categorical(n_train=self.n_train,
                                      n_test=self.n_test,
                                      n_features=4,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        assert_allclose(X_train.shape, (self.n_train, 4))
        assert_allclose(X_test.shape, (self.n_test, 4))

    def test_data_generate_categorical3(self):
        X_train, y_train, X_test, y_test = \
            generate_data_categorical(n_train=self.n_train,
                                      n_test=self.n_test,
                                      n_features=3,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        X_train2, y_train2, X_test2, y_test2 = \
            generate_data_categorical(n_train=self.n_train,
                                      n_test=self.n_test,
                                      n_features=3,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        assert np.array_equal(X_train, X_train2)
        assert np.array_equal(X_train, X_train2)
        assert np.array_equal(X_test, X_test2)
        assert np.array_equal(y_train, y_train2)
        assert np.array_equal(y_test, y_test2)

    def test_data_generate_categorical5(self):
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=-1)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=0, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=-1,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train='not int', n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test='not int',
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=0,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features='not int',
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=-1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative='not int', n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=0.6,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination='not float',
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=-1, n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in='not int',
                                      n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=self.n_train + self.n_test + 1,
                                      n_category_out=3,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5, n_category_out=-1,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5,
                                      n_category_out='not int',
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)
        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5,
                                      n_category_out=self.n_train + self.n_test + 1,
                                      n_informative=1, n_features=1,
                                      contamination=self.contamination,
                                      random_state=self.random_state)

        with assert_raises(ValueError):
            generate_data_categorical(n_train=self.n_train, n_test=self.n_test,
                                      n_category_in=5,
                                      n_category_out=5,
                                      n_informative=2, n_features=2,
                                      contamination=self.contamination,
                                      shuffle='not bool',
                                      random_state=self.random_state)

    def test_evaluate_print(self):
        X_train, X_test, y_train, y_test = generate_data(
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
        X_train, X_test, y_train, y_test = generate_data(
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

        # test shape difference between X_train and X_test
        X_test = np.hstack((X_test, np.zeros(
            (X_test.shape[0], 1))))  # add extra column/feature
        with assert_raises(ValueError):
            check_consistent_shape(X_train, y_train, X_test, y_test,
                                   y_train_pred_n, y_test_pred_n)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
