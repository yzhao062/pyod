# -*- coding: utf-8 -*-


import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.deep_sad import DeepSAD, InnerDeepSAD, optimizer_dict
from pyod.utils.data import generate_data


class TestDeepSAD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_train = 6000
        cls.n_test = 1000
        cls.n_features = 300
        cls.contamination = 0.1
        cls.roc_floor = 0.5
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_data(
            n_train=cls.n_train, n_test=cls.n_test,
            n_features=cls.n_features, contamination=cls.contamination,
            random_state=42)

        # Deep SAD is semi-supervised: reveal a subset of the training
        # anomalies as labeled (1) and leave everything else unlabeled (0).
        cls.semi_y = np.zeros(cls.n_train, dtype=int)
        anomaly_idx = np.where(cls.y_train == 1)[0]
        revealed = anomaly_idx[:len(anomaly_idx) // 2]
        cls.semi_y[revealed] = 1

        # The two fitted detectors are trained once and shared read-only
        # by the tests; tests that refit build their own small instance.
        cls.clf = DeepSAD(n_features=cls.n_features, epochs=10,
                          hidden_neurons=[64, 32],
                          contamination=cls.contamination,
                          random_state=2021)
        cls.clf.fit(cls.X_train, cls.semi_y)

        # a second detector trained fully unsupervised (y=None) exercises
        # the Deep SVDD fallback path.
        cls.clf_unsup = DeepSAD(n_features=cls.n_features, epochs=5,
                                hidden_neurons=[32, 16],
                                contamination=cls.contamination,
                                preprocessing=False)
        cls.clf_unsup.fit(cls.X_train)

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
        X, y = self.X_train[:600, :8], self.y_train[:600]
        clf = DeepSAD(n_features=8, epochs=2, hidden_neurons=[16, 8],
                      verbose=0, contamination=self.contamination)
        pred_labels = clf.fit_predict(X)
        assert_equal(pred_labels.shape, y.shape)

    def test_fit_predict_score(self):
        X, y = self.X_test[:, :8], self.y_test
        clf = DeepSAD(n_features=8, epochs=2, hidden_neurons=[16, 8],
                      verbose=0, contamination=self.contamination)
        clf.fit_predict_score(X, y)
        clf.fit_predict_score(X, y, scoring='roc_auc_score')
        clf.fit_predict_score(X, y, scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            clf.fit_predict_score(X, y, scoring='something')

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

    def test_eta_must_be_positive(self):
        for bad_eta in (-1, 0):
            with assert_raises(ValueError):
                DeepSAD(n_features=self.n_features, eta=bad_eta)

    def test_eps_must_be_positive_finite(self):
        for bad_eps in (0, -1e-6, float('nan'), float('inf')):
            with assert_raises(ValueError):
                DeepSAD(n_features=self.n_features, eps=bad_eps)

    def test_validation_size(self):
        X, y = self.X_train[:600, :8], self.semi_y[:600]
        original_init_c = InnerDeepSAD._init_c
        n_center = []

        def spy_init_c(model, X_norm, eps=0.1):
            n_center.append(X_norm.shape[0])
            return original_init_c(model, X_norm, eps)

        # 0.2 keeps 120 of the 600 samples out of the center initialization
        # and the training batches; 0 uses every sample.
        for validation_size in (0.2, 0):
            clf = DeepSAD(n_features=8, epochs=2, hidden_neurons=[16, 8],
                          validation_size=validation_size, verbose=0,
                          random_state=2021)
            with mock.patch.object(InnerDeepSAD, '_init_c', spy_init_c):
                clf.fit(X, y)
            assert_equal(len(clf.decision_scores_), X.shape[0])
            assert np.all(np.isfinite(clf.decision_scores_))
        assert_equal(n_center, [480, 600])
        for bad_size in (-0.1, 1.0):
            with assert_raises(ValueError):
                DeepSAD(n_features=8, validation_size=bad_size)

    def test_dropout_follows_hidden_activations(self):
        X = self.X_train[:600, :8]
        # (hidden_neurons, dropout_rate); None is the default [64, 32]
        for hidden_neurons, dropout_rate in ((None, 0.2),
                                             ([64, 32, 16], 0.5),
                                             (None, 0)):
            clf = DeepSAD(n_features=8, epochs=1,
                          hidden_neurons=hidden_neurons,
                          dropout_rate=dropout_rate, verbose=0)
            clf.fit(X)
            modules = list(clf.model_.model)
            dropouts = [m for m in modules if isinstance(m, nn.Dropout)]
            assert_equal(len(dropouts), len(clf.hidden_neurons) - 1)
            assert all(m.p == dropout_rate for m in dropouts)
            assert isinstance(modules[-1], nn.Linear)
            assert_equal(modules[-1].out_features, clf.hidden_neurons[-1])

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

        class DivergeAfterFirstEpoch(torch.optim.SGD):
            steps_per_epoch = None

            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                self.n_steps = 0

            def step(self, closure=None):
                self.n_steps += 1
                if self.n_steps <= self.steps_per_epoch:
                    return super().step(closure)
                with torch.no_grad():
                    for group in self.param_groups:
                        for p in group['params']:
                            p.fill_(100.0)

        # validation_size=0 selects the epoch by training loss, 0.5 by the
        # loss on the held-out half; both must restore the first epoch.
        for validation_size in (0, 0.5):
            n_fit = X.shape[0] - int(X.shape[0] * validation_size)
            DivergeAfterFirstEpoch.steps_per_epoch = int(
                np.ceil(n_fit / batch_size))
            with mock.patch.dict(optimizer_dict,
                                 {'sgd': DivergeAfterFirstEpoch}):
                clf = DeepSAD(n_features=8, epochs=3, hidden_neurons=[16, 8],
                              optimizer='sgd', batch_size=batch_size,
                              validation_size=validation_size, verbose=0,
                              random_state=2021)
                clf.fit(X)

            live = clf.model_.state_dict()
            for key, snapshot in clf.best_model_dict.items():
                assert snapshot.data_ptr() != live[key].data_ptr(), key
                assert torch.equal(snapshot, live[key]), key
                assert not torch.any(snapshot == 100.0), key
            assert np.all(np.isfinite(clf.decision_scores_))

    def test_validation_split_keeps_lone_labeled_anomaly(self):
        # a single labeled anomaly can never be held out, so whatever the
        # permutation it must reach the training targets; the 600 unlabeled
        # samples are still split in half.
        X = self.X_train[:601, :8]
        y = np.zeros(601, dtype=int)
        y[7] = 1
        for random_state in (0, 1, 2021):
            clf = DeepSAD(n_features=8, epochs=1, hidden_neurons=[16, 8],
                          validation_size=0.5, verbose=0,
                          random_state=random_state)
            with mock.patch('pyod.models.deep_sad.TensorDataset',
                            wraps=TensorDataset) as dataset:
                clf.fit(X, y)
            X_fit, semi_fit = dataset.call_args[0]
            assert_equal(X_fit.shape, (301, 8))
            assert_equal(int((semi_fit == -1).sum()), 1)
            assert_equal(len(clf.decision_scores_), 601)

    def test_validation_split_is_stratified(self):
        # 0.2 of the 10 labeled anomalies and 0.2 of the 590 unlabeled
        # samples are held out separately: 480 samples with 8 anomalies
        # train and 120 samples with 2 anomalies are held out.
        X = self.X_train[:600, :8]
        y = np.zeros(600, dtype=int)
        y[:10] = 1
        original_loss = DeepSAD._deep_sad_loss
        semi_val = []

        def spy_loss(clf, outputs, semi_targets):
            if not clf.model_.training:
                semi_val.append(semi_targets)
            return original_loss(clf, outputs, semi_targets)

        clf = DeepSAD(n_features=8, epochs=1, hidden_neurons=[16, 8],
                      validation_size=0.2, verbose=0, random_state=2021)
        with mock.patch('pyod.models.deep_sad.TensorDataset',
                        wraps=TensorDataset) as dataset, \
                mock.patch.object(DeepSAD, '_deep_sad_loss', spy_loss):
            clf.fit(X, y)
        X_fit, semi_fit = dataset.call_args[0]
        assert_equal(X_fit.shape, (480, 8))
        assert_equal(int((semi_fit == -1).sum()), 8)
        assert_equal(len(semi_val), 1)
        assert_equal(len(semi_val[0]), 120)
        assert_equal(int((semi_val[0] == -1).sum()), 2)
        assert_equal(len(clf.decision_scores_), 600)

    def test_scaler_is_fitted_on_training_split(self):
        X = self.X_train[:600, :8]
        original_split = DeepSAD._split_validation
        folds = []

        def spy_split(clf, semi_targets):
            fit_idx, val_idx = original_split(clf, semi_targets)
            folds.append((fit_idx, val_idx))
            return fit_idx, val_idx

        for validation_size, n_fit in ((0.2, 480), (0, 600)):
            clf = DeepSAD(n_features=8, epochs=1, hidden_neurons=[16, 8],
                          validation_size=validation_size, verbose=0,
                          random_state=2021)
            with mock.patch.object(DeepSAD, '_split_validation', spy_split):
                clf.fit(X)
            fit_idx, val_idx = folds[-1]
            assert_equal(len(fit_idx), n_fit)
            assert_equal(np.sort(np.concatenate([fit_idx, val_idx])),
                         np.arange(600))
            assert_equal(clf.scaler_.n_samples_seen_, n_fit)
            assert np.allclose(clf.scaler_.mean_, X[fit_idx].mean(axis=0))
            assert np.allclose(clf.scaler_.var_, X[fit_idx].var(axis=0))

    def test_epoch_loss_is_mean_per_sample(self):
        # 600 samples in batches of 256 leave an undersized last batch of
        # 88; the reported epoch loss weights every batch by its size.
        X = self.X_train[:600, :8]
        original_loss = DeepSAD._deep_sad_loss
        batches = []

        def spy_loss(clf, outputs, semi_targets):
            loss = original_loss(clf, outputs, semi_targets)
            batches.append((loss.item(), len(semi_targets)))
            return loss

        clf = DeepSAD(n_features=8, epochs=1, hidden_neurons=[16, 8],
                      batch_size=256, validation_size=0, verbose=1,
                      random_state=2021)
        with mock.patch.object(DeepSAD, '_deep_sad_loss', spy_loss), \
                mock.patch('builtins.print') as mock_print:
            clf.fit(X)
        assert_equal([n for _, n in batches], [256, 256, 88])
        expected = sum(loss * n for loss, n in batches) / 600
        message = mock_print.call_args[0][0]
        self.assertAlmostEqual(float(message.split('Loss: ')[1]), expected,
                               places=5)

    def test_single_hidden_layer(self):
        # hidden_neurons=[k] is one bias-free linear layer straight into
        # the representation space: no extra k -> k layer, activation or
        # dropout.
        X = self.X_train[:600, :8]
        clf = DeepSAD(n_features=8, epochs=2, hidden_neurons=[16], verbose=0,
                      contamination=self.contamination)
        clf.fit(X)
        modules = list(clf.model_.model)
        assert_equal(len(modules), 1)
        assert isinstance(modules[0], nn.Linear)
        assert_equal((modules[0].in_features, modules[0].out_features),
                     (8, 16))
        assert modules[0].bias is None
        assert_equal(tuple(clf.c_.shape), (16,))
        scores = clf.decision_function(self.X_test[:, :8])
        assert_equal(scores.shape[0], self.X_test.shape[0])
        assert np.all(np.isfinite(scores))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
