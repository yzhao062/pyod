# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np
# noinspection PyProtectedMember
from numpy.testing import *
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.stats import rankdata
from sklearn.base import clone

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.rod import ROD, rod_3D, rod_nD, angle, sigmoid, process_sub, \
    mad
from pyod.utils.data import generate_data


class TestROD(unittest.TestCase):
    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.gm = None
        self.median = None
        self.data_scaler = None
        self.angles_scalers1 = None
        self.angles_scalers2 = None
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=4,
            contamination=self.contamination, random_state=42)

        self.clf = ROD()
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        with assert_raises(TypeError):
            ROD(parallel_execution='str')

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

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

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

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

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_predict_rank(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)
        print(pred_ranks)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=2)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_invocation(self):
        X_2D = self.X_train[:, 0:2]
        X_3D = self.X_train[:, 0:3]
        X_4D = self.X_train

        with assert_raises(IndexError):
            rod_3D(X_2D)
        scores = ROD().fit(X_2D).decision_scores_
        assert_array_equal(scores,
                           rod_3D(np.hstack((X_2D, np.zeros(
                               shape=(X_2D.shape[0], 3 - X_2D.shape[1])))))[0])
        scores = ROD().fit(X_3D).decision_scores_
        assert_array_equal(scores, rod_3D(X_3D)[0])
        scores = ROD().fit(X_4D).decision_scores_
        assert_array_equal(scores,
                           rod_nD(X_4D, False, self.gm, self.data_scaler,
                                  self.angles_scalers1, self.angles_scalers2)[
                               0])

    def test_angle(self):
        assert_equal(0.0, angle(v1=[0, 0, 1], v2=[0, 0, 1]))

    def test_sigmoid(self):
        assert_equal(0.5, sigmoid(np.array([0.0])))

    def test_process_sub(self):
        subspace = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        assert_equal([0.5, 0.5, 0.5],
                     process_sub(subspace, self.gm, self.median,
                                 self.angles_scalers1, self.angles_scalers2)[
                         0])

    def test_parallel_vs_non_parallel(self):
        assert_equal(rod_nD(self.X_train, False, self.gm, self.data_scaler,
                            self.angles_scalers1, self.angles_scalers2)[0],
                     rod_nD(self.X_train, True, self.gm, self.data_scaler,
                            self.angles_scalers1, self.angles_scalers2)[0])

    def test_mad(self):
        gm, _ = mad(np.array([1, 2, 3]))
        assert_equal([0.6745, 0.0, 0.6745], gm)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
