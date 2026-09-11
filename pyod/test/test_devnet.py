# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

import numpy as np
import torch
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.base import clone
from numpy.testing import assert_array_equal

from pyod.models.devnet import DevNet
from pyod.utils.data import generate_data


class TestDevNet(unittest.TestCase):
    def setUp(self):
        self.n_train = 3000
        self.n_test = 1500
        self.n_features = 2000
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = DevNet(epochs=3, contamination=self.contamination)
        self.clf.fit(self.X_train, self.y_train)

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
        assert (hasattr(self.clf, 'model') and
                self.clf.model is not None)

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
        pred_labels = self.clf.fit_predict(self.X_train, self.y_train)
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

    def test_random_seed_stored(self):
        clf = DevNet(random_seed=7)
        assert clf.random_seed == 7
        assert clf.get_params()['random_seed'] == 7

    def test_known_outliers_stored(self):
        clf = DevNet(known_outliers=10)
        assert clf.known_outliers == 10
        assert clf.get_params()['known_outliers'] == 10

    def test_data_format_stored(self):
        clf = DevNet(data_format=1)
        assert clf.data_format == 1
        assert clf.get_params()['data_format'] == 1

    def test_random_seed_clone(self):
        clf = DevNet(random_seed=99)
        cloned = clone(clf)
        assert cloned.random_seed == 99

    def test_random_seed_reproducibility(self):
        X_small, _, y_small, _ = generate_data(
            n_train=200, n_test=50, n_features=5,
            contamination=0.1, random_state=0)
        clf1 = DevNet(epochs=2, random_seed=0, contamination=0.1)
        clf2 = DevNet(epochs=2, random_seed=0, contamination=0.1)
        clf1.fit(X_small, y_small)
        clf2.fit(X_small, y_small)
        assert_array_equal(clf1.decision_scores_, clf2.decision_scores_)

    def test_known_outliers_cap(self):
        X_small, _, y_small, _ = generate_data(
            n_train=200, n_test=50, n_features=5,
            contamination=0.2, random_state=0)
        # 40 outliers in training set; cap to 5
        clf = DevNet(epochs=1, known_outliers=5, contamination=0.2)
        clf.fit(X_small, y_small)
        assert_equal(len(clf.decision_scores_), X_small.shape[0])

    def test_model_clone(self):
        cloned = clone(self.clf)
        assert cloned.random_seed == self.clf.random_seed
        assert cloned.known_outliers == self.clf.known_outliers
        assert cloned.data_format == self.clf.data_format

    def test_data_format_1_inference(self):
        X_small, _, y_small, _ = generate_data(
            n_train=200, n_test=50, n_features=5,
            contamination=0.1, random_state=0)
        clf = DevNet(epochs=1, data_format=1, contamination=0.1)
        clf.fit(X_small, y_small)
        scores = clf.decision_function(X_small)
        assert_equal(scores.shape[0], X_small.shape[0])

    def test_random_seed_none_does_not_crash(self):
        X_small, _, y_small, _ = generate_data(
            n_train=200, n_test=50, n_features=5,
            contamination=0.1, random_state=0)
        clf = DevNet(epochs=1, random_seed=None, contamination=0.1)
        clf.fit(X_small, y_small)
        assert_equal(len(clf.decision_scores_), X_small.shape[0])

    def test_deprecated_nb_batch_warns(self):
        with assert_raises(DeprecationWarning):
            # warnings are normally ignored; force them to raise
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                DevNet(nb_batch=20)

    def test_deprecated_cont_rate_warns(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with assert_raises(DeprecationWarning):
                DevNet(cont_rate=0.02)

    def test_legacy_positional_order_preserved(self):
        # Positional callers used the original order:
        # (network_depth, batch_size, epochs, nb_batch, known_outliers,
        #  cont_rate, data_format, random_seed, device, contamination)
        # nb_batch and cont_rate must stay at positions 4 and 6 so that
        # existing positional calls don't silently remap known_outliers,
        # random_seed, or contamination to wrong values.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            clf = DevNet(2, 512, 50, 20, 15, 0.02, 0, 7, None, 0.1)
        assert clf.network_depth == 2
        assert clf.batch_size == 512
        assert clf.epochs == 50
        assert clf.nb_batch == 20         # position 4 → nb_batch
        assert clf.known_outliers == 15   # position 5 → known_outliers
        assert clf.cont_rate == 0.02      # position 6 → cont_rate
        assert clf.data_format == 0       # position 7 → data_format
        assert clf.random_seed == 7       # position 8 → random_seed
        assert clf.device is not None      # position 9 → device (None→cpu in __init__)
        assert clf.contamination == 0.1   # position 10 → contamination

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
