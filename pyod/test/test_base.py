# -*- coding: utf-8 -*-
import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from sklearn.utils import deprecated
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_warns_message

import numpy as np
import scipy.sparse as sp

from pyod.models.base import BaseDetector
from pyod.models.base import clone
from pyod.utils.load_data import generate_data


# Check sklearn\tests\test_base
# A few test classes
class MyEstimator(BaseDetector):

    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


class K(BaseDetector):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


class T(BaseDetector):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


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


class DeprecatedAttributeEstimator(BaseDetector):
    def __init__(self, a=None, b=None):
        self.a = a
        if b is not None:
            DeprecationWarning("b is deprecated and renamed 'a'")
            self.a = b

    @property
    @deprecated("Parameter 'b' is deprecated and renamed to 'a'")
    def b(self):
        return self._b


class Buggy(BaseDetector):
    """
    A buggy estimator that does not set its parameters right.
    """

    def __init__(self, a=None):
        self.a = 1

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


class NoEstimator(object):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def predict(self, X=None):
        return None


class VargEstimator(BaseDetector):
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs):
        pass

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass


class TestBASE(unittest.TestCase):
    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.X_train, self.y_train, _, self.X_test, self.y_test, _ = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination)

    def test_init(self):
        """
        Test base class initialization

        :return:
        """

        class DummyClass(BaseDetector):
            def __init__(self, contamination=0.1):
                super().__init__(contamination=contamination)

            def decision_function(self, X):
                pass

            def fit(self, X, y=None):
                pass

        self.dummy_clf = DummyClass()
        assert_equal(self.dummy_clf.contamination, 0.1)

        self.dummy_clf = DummyClass(contamination=0.2)
        assert_equal(self.dummy_clf.contamination, 0.2)

        with assert_raises(ValueError):
            DummyClass(contamination=0.51)

        with assert_raises(ValueError):
            DummyClass(contamination=0)

        with assert_raises(ValueError):
            DummyClass(contamination=-0.5)

    def test_fit(self):
        class dummy(BaseDetector):
            def __init__(self, contamination=0.1):
                super().__init__(contamination=contamination)

            def decision_function(self, X):
                pass

            def fit(self, X, y=None):
                return X

        self.dummy_clf = dummy()

        assert_equal(self.dummy_clf.fit(0), 0)

    def test_fit_predict(self):
        # TODO: add more testcases
        class dummy(BaseDetector):
            def __init__(self, contamination=0.1):
                super().__init__(contamination=contamination)

            def decision_function(self, X):
                pass

            def fit(self, X, y=None):
                self.labels_ = X

        self.dummy_clf = dummy()

        assert_equal(self.dummy_clf.fit_predict(0), 0)

    def test_predict_proba(self):
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

        assert_true('a__d' in test.get_params(deep=True))
        assert_true('a__d' not in test.get_params(deep=False))

        test.set_params(a__d=2)
        assert_true(test.a.d == 2)
        assert_raises(ValueError, test.set_params, a__a=2)

    def tearDown(self):
        pass


#############################################################################
# The tests
class TestSklearnBase(unittest.TestCase):

    def test_clone(self):
        # Tests that clone creates a correct deep copy.
        # We create an estimator, make a copy of its original state
        # (which, in this case, is the current state of the estimator),
        # and check that the obtained copy is a correct deep copy.

        from sklearn.feature_selection import SelectFpr, f_classif

        selector = SelectFpr(f_classif, alpha=0.1)
        new_selector = clone(selector)
        assert_true(selector is not new_selector)
        assert_equal(selector.get_params(), new_selector.get_params())

        selector = SelectFpr(f_classif, alpha=np.zeros((10, 2)))
        new_selector = clone(selector)
        assert_true(selector is not new_selector)

    def test_clone_2(self):
        # Tests that clone doesn't copy everything.
        # We first create an estimator, give it an own attribute, and
        # make a copy of its original state. Then we check that the copy doesn't
        # have the specific attribute we manually added to the initial estimator.

        from sklearn.feature_selection import SelectFpr, f_classif

        selector = SelectFpr(f_classif, alpha=0.1)
        selector.own_attribute = "test"
        new_selector = clone(selector)
        assert_false(hasattr(new_selector, "own_attribute"))

    def test_clone_buggy(self):
        # Check that clone raises an error on buggy estimators.
        buggy = Buggy()
        buggy.a = 2
        assert_raises(RuntimeError, clone, buggy)

        no_estimator = NoEstimator()
        assert_raises(TypeError, clone, no_estimator)

        varg_est = VargEstimator()
        assert_raises(RuntimeError, clone, varg_est)

    def test_clone_empty_array(self):
        # Regression test for cloning estimators with empty arrays
        clf = MyEstimator(empty=np.array([]))
        clf2 = clone(clf)
        assert_array_equal(clf.empty, clf2.empty)

        clf = MyEstimator(empty=sp.csr_matrix(np.array([[0]])))
        clf2 = clone(clf)
        assert_array_equal(clf.empty.data, clf2.empty.data)

    def test_clone_nan(self):
        # Regression test for cloning estimators with default parameter as np.nan
        clf = MyEstimator(empty=np.nan)
        clf2 = clone(clf)

        assert_true(clf.empty is clf2.empty)

    def test_clone_copy_init_params(self):
        # test for deprecation warning when copying or casting an init parameter
        est = ModifyInitParams()
        message = (
            "Estimator ModifyInitParams modifies parameters in __init__. "
            "This behavior is deprecated as of 0.18 and support "
            "for this behavior will be removed in 0.20.")

        assert_warns_message(DeprecationWarning, message, clone, est)

    def test_clone_sparse_matrices(self):
        sparse_matrix_classes = [
            getattr(sp, name)
            for name in dir(sp) if name.endswith('_matrix')]

        for cls in sparse_matrix_classes:
            sparse_matrix = cls(np.eye(5))
            clf = MyEstimator(empty=sparse_matrix)
            clf_cloned = clone(clf)
            assert_true(clf.empty.__class__ is clf_cloned.empty.__class__)
            assert_array_equal(clf.empty.toarray(), clf_cloned.empty.toarray())


if __name__ == '__main__':
    unittest.main()
