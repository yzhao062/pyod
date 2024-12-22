# -*- coding: utf-8 -*-


import os
import sys
import unittest

# noinspection PyProtectedMember
from numpy.testing import (assert_array_less, assert_equal,
                           assert_raises)
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from pyod.models.gmm import GMM
from pyod.utils.data import generate_data_clusters

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestGMM(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.n_components = 4
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data_clusters(
            n_train=self.n_train,
            n_test=self.n_test,
            n_clusters=self.n_components,
            contamination=self.contamination,
            random_state=42,
        )

        self.clf = GMM(n_components=self.n_components,
                       contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert (
                hasattr(self.clf, "decision_scores_")
                and self.clf.decision_scores_ is not None
        )
        assert hasattr(self.clf, "labels_") and self.clf.labels_ is not None
        assert hasattr(self.clf,
                       "threshold_") and self.clf.threshold_ is not None
        assert hasattr(self.clf, "weights_") and self.clf.weights_ is not None
        assert hasattr(self.clf, "means_") and self.clf.means_ is not None
        assert hasattr(self.clf,
                       "covariances_") and self.clf.covariances_ is not None
        assert hasattr(self.clf,
                       "precisions_") and self.clf.precisions_ is not None
        assert (
                hasattr(self.clf, "precisions_cholesky_")
                and self.clf.precisions_cholesky_ is not None
        )
        assert hasattr(self.clf,
                       "converged_") and self.clf.converged_ is not None
        assert hasattr(self.clf, "n_iter_") and self.clf.n_iter_ is not None
        assert hasattr(self.clf,
                       "lower_bound_") and self.clf.lower_bound_ is not None

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method="linear")
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method="unify")
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method="something")

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

    def test_predict_rank(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        # assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=2)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        # assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=2)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
