# -*- coding: utf-8 -*-


import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.deep_sad import DeepSAD, optimizer_dict
from pyod.utils.data import generate_data


class TestDeepSAD(unittest.TestCase):
    def setUp(self):
        self.n_train = 6000
        self.n_test = 1000
        self.n_features = 300
        self.contamination = 0.1
        self.roc_floor = 0.5
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        # Deep SAD is semi-supervised: reveal a subset of the training
        # anomalies as labeled (1) and leave everything else unlabeled (0).
        self.semi_y = np.zeros(self.n_train, dtype=int)
        anomaly_idx = np.where(self.y_train == 1)[0]
        revealed = anomaly_idx[:len(anomaly_idx) // 2]
        self.semi_y[revealed] = 1

        self.clf = DeepSAD(n_features=self.n_features, epochs=10,
                           hidden_neurons=[64, 32],
                           contamination=self.contamination,
                           random_state=2021)
        self.clf.fit(self.X_train, self.semi_y)

        # a second detector trained fully unsupervised (y=None) exercises
        # the Deep SVDD fallback path.
        self.clf_unsup = DeepSAD(n_features=self.n_features, epochs=5,
                                 hidden_neurons=[32, 16],
                                 contamination=self.contamination,
                                 preprocessing=False)
        self.clf_unsup.fit(self.X_train)

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        assert (hasattr(self.clf, '_mu') and
                self.clf._mu is not None)
        assert (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None)
        assert (hasattr(self.clf, 'model_') and
                self.clf.model_ is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

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

    def test_model_clone(self):
        clone(self.clf)
        clone(self.clf_unsup)

    def test_hidden_neurons_wider_than_input(self):
        # Deep SAD uses only an encoder, so hidden layers may be wider
        # than the input dimension; fitting must not raise.
        X = self.X_train[:, :8]
        clf = DeepSAD(n_features=8, epochs=2, hidden_neurons=[64, 32],
                      contamination=self.contamination)
        clf.fit(X)
        assert clf.decision_scores_ is not None

    def test_invalid_center_raises(self):
        # A center that does not match the representation dimension is
        # rejected rather than silently broadcasting.
        for bad_c in ([1.0, 2.0], np.ones(33), float('nan'),
                      np.array([float('inf')] * 32)):
            clf = DeepSAD(n_features=self.n_features, c=bad_c, epochs=1,
                          hidden_neurons=[64, 32], verbose=0)
            with assert_raises(ValueError):
                clf.fit(self.X_train)

    def test_all_zero_center_is_rejected(self):
        for bad_c in (0.0, np.zeros(32, dtype=np.float32)):
            clf = DeepSAD(n_features=self.n_features, c=bad_c, epochs=1,
                          hidden_neurons=[64, 32], verbose=0)
            with assert_raises(ValueError):
                clf.fit(self.X_train)

    def test_scalar_center_is_expanded(self):
        clf = DeepSAD(n_features=self.n_features, c=1.0, epochs=2,
                      hidden_neurons=[64, 32], verbose=0,
                      contamination=self.contamination)
        clf.fit(self.X_train)
        assert_equal(tuple(clf.c_.shape), (32,))
        assert torch.equal(clf.c_, torch.ones(32))
        assert clf.decision_scores_ is not None

    def test_valid_center_accepted(self):
        # A correctly sized, finite center is accepted.
        c = torch.ones(32)
        clf = DeepSAD(n_features=self.n_features, c=c, epochs=2,
                      hidden_neurons=[64, 32], verbose=0,
                      contamination=self.contamination)
        clf.fit(self.X_train)
        assert clf.decision_scores_ is not None
        before = clf.c_.clone()
        c[:] = 99.0
        assert torch.equal(before, clf.c_)

    def test_random_state_instance_is_accepted(self):
        X = self.X_train[:600, :8]
        scores = []
        for _ in range(2):
            clf = DeepSAD(n_features=8, epochs=2, hidden_neurons=[16, 8],
                          verbose=0,
                          random_state=np.random.RandomState(0))
            clf.fit(X)
            scores.append(clf.decision_scores_)
        assert np.array_equal(scores[0], scores[1])
        clone(clf)

    def test_all_listed_optimizers_train(self):
        assert 'lbfgs' not in optimizer_dict
        assert 'sparseadam' not in optimizer_dict
        X = self.X_train[:600, :8]
        for name in optimizer_dict:
            clf = DeepSAD(n_features=8, epochs=1, hidden_neurons=[16, 8],
                          optimizer=name, verbose=0)
            clf.fit(X)
            assert np.all(np.isfinite(clf.decision_scores_)), name

    def test_best_epoch_weights_are_restored(self):
        X = self.X_train[:600, :8]
        batch_size = 100
        steps_per_epoch = int(np.ceil(X.shape[0] / batch_size))

        class DivergeAfterFirstEpoch(torch.optim.SGD):
            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                self.n_steps = 0

            def step(self, closure=None):
                self.n_steps += 1
                if self.n_steps <= steps_per_epoch:
                    return super().step(closure)
                with torch.no_grad():
                    for group in self.param_groups:
                        for p in group['params']:
                            p.fill_(100.0)

        with mock.patch.dict(optimizer_dict,
                             {'sgd': DivergeAfterFirstEpoch}):
            clf = DeepSAD(n_features=8, epochs=3, hidden_neurons=[16, 8],
                          optimizer='sgd', batch_size=batch_size,
                          verbose=0, random_state=2021)
            clf.fit(X)

        live = clf.model_.state_dict()
        for key, snapshot in clf.best_model_dict.items():
            assert snapshot.data_ptr() != live[key].data_ptr(), key
            assert torch.equal(snapshot, live[key]), key
            assert not torch.any(snapshot == 100.0), key
        assert np.all(np.isfinite(clf.decision_scores_))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
