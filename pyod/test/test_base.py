# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys

import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.base import BaseDetector
from pyod.utils.data import generate_data


# Check sklearn\tests\test_base
# A few test classes
# noinspection PyMissingConstructor,PyPep8Naming
class MyEstimator(BaseDetector):

    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class K(BaseDetector):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class T(BaseDetector):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class ModifyInitParams(BaseDetector):
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])):
        self.a = a.copy()

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


# noinspection PyMissingConstructor
class VargEstimator(BaseDetector):
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        pass

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


class Dummy1(BaseDetector):
    def __init__(self, contamination=0.1):
        super(Dummy1, self).__init__(contamination=contamination)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        pass


class Dummy2(BaseDetector):
    def __init__(self, contamination=0.1):
        super(Dummy2, self).__init__(contamination=contamination)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        return X


class Dummy3(BaseDetector):
    def __init__(self, contamination=0.1):
        super(Dummy3, self).__init__(contamination=contamination)

    def decision_function(self, X):
        pass

    def fit(self, X, y=None):
        self.labels_ = X


class TestBASE(unittest.TestCase):
    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination)

    def test_init(self):
        """
        Test base class initialization

        :return:
        """
        self.dummy_clf = Dummy1()
        assert_equal(self.dummy_clf.contamination, 0.1)

        self.dummy_clf = Dummy1(contamination=0.2)
        assert_equal(self.dummy_clf.contamination, 0.2)

        with assert_raises(ValueError):
            Dummy1(contamination=0.51)

        with assert_raises(ValueError):
            Dummy1(contamination=0)

        with assert_raises(ValueError):
            Dummy1(contamination=-0.5)

    def test_fit(self):
        self.dummy_clf = Dummy2()
        assert_equal(self.dummy_clf.fit(0), 0)

    def test_fit_predict(self):
        # TODO: add more testcases

        self.dummy_clf = Dummy3()

        assert_equal(self.dummy_clf.fit_predict(0), 0)

    def test_predict_proba(self):
        # TODO: create uniform testcases
        pass

    def test_rank(self):
        # TODO: create uniform testcases
        pass

    def test_repr(self):
        # Smoke test the repr of the base estimator.
        my_estimator = MyEstimator()
        repr(my_estimator)
        test = T(K(), K())
        assert_equal(
            repr(test),
            "T(a=K(c=None, d=None), b=K(c=None, d=None))"
        )

        some_est = T(a=["long_params"] * 1000)
        assert_equal(len(repr(some_est)), 415)

    def test_str(self):
        # Smoke test the str of the base estimator
        my_estimator = MyEstimator()
        str(my_estimator)

    def test_get_params(self):
        test = T(K(), K())

        assert ('a__d' in test.get_params(deep=True))
        assert ('a__d' not in test.get_params(deep=False))

        test.set_params(a__d=2)
        assert (test.a.d == 2)
        assert_raises(ValueError, test.set_params, a__a=2)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
