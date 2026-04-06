# -*- coding: utf-8 -*-

import os
import sys
import unittest

# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.cd import CD
from pyod.utils.data import generate_data


class TestCD(unittest.TestCase):
    """
    Notes: GAN may yield unstable results, so the test is design for running
    models only, without any performance check.
    """

    def setUp(self):
        self.n_train = 1000
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = CD(contamination=self.contamination)
        self.clf.fit(self.X_train, self.y_train)

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        # assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test,
                                                      return_stats=False)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_with_rejection_stats(self):
        _, [expected_rejrate, ub_rejrate,
            ub_cost] = self.clf.predict_with_rejection(self.X_test,
                                                       return_stats=True)
        assert (expected_rejrate >= 0)
        assert (expected_rejrate <= 1)
        assert (ub_rejrate >= 0)
        assert (ub_rejrate <= 1)
        assert (ub_cost >= 0)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape, self.y_test.shape)
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_decision_function_uses_train_models(self):
        # Verify that decision_function uses models fitted on training
        # data rather than re-fitting on test data.
        import numpy as np
        from sklearn.linear_model import LinearRegression

        clf = CD(contamination=self.contamination)
        clf.fit(self.X_train)

        # The fitted models should exist and match the number of features
        assert hasattr(clf, '_models')
        assert_equal(len(clf._models), self.X_train.shape[1])

        # Each stored model should already have coefficients (fitted)
        for mod in clf._models:
            assert hasattr(mod, 'coef_')

        # Replace stored models AND self.model with sentinels that
        # raise on fit(). This catches both the refit=False path and
        # any regression to the old buggy path that clones self.model.
        class NoRefitRegressor(LinearRegression):
            def fit(self, X, y=None, **kwargs):
                raise AssertionError(
                    "fit() called during decision_function")

        for i, mod in enumerate(clf._models):
            sentinel = NoRefitRegressor()
            sentinel.coef_ = mod.coef_
            sentinel.intercept_ = mod.intercept_
            clf._models[i] = sentinel

        clf.model = NoRefitRegressor()

        # decision_function must succeed without calling fit()
        test_scores = clf.decision_function(self.X_test)
        assert_equal(len(test_scores), self.n_test)

    def test_sklearn_clone_compatibility(self):
        # Verify that fit() uses sklearn.base.clone (not deepcopy),
        # so estimators with non-picklable state still work.
        import threading
        import numpy as np
        from sklearn.base import BaseEstimator, RegressorMixin

        class LockingRegressor(BaseEstimator, RegressorMixin):
            def __init__(self):
                self._lock = threading.Lock()

            def fit(self, X, y=None):
                self.coef_ = np.zeros(X.shape[1])
                self.intercept_ = 0.0
                return self

            def predict(self, X):
                return X @ self.coef_ + self.intercept_

        clf = CD(contamination=self.contamination, model=LockingRegressor())
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert_equal(len(scores), self.n_test)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
