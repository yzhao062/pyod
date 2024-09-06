# -*- coding: utf-8 -*-


import os
import sys
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.alad import ALAD
from pyod.utils.data import generate_data


class TestALAD(unittest.TestCase):
    def setUp(self):
        self.n_train = 500
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1
        self.roc_floor = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = ALAD(epochs=100, latent_dim=2,
                        learning_rate_disc=0.0001,
                        learning_rate_gen=0.0001,
                        dropout_rate=0.2,
                        add_recon_loss=False,
                        lambda_recon_loss=0.05,
                        add_disc_zz_loss=True,
                        dec_layers=[75, 100],
                        enc_layers=[100, 75],
                        disc_xx_layers=[100, 75],
                        disc_zz_layers=[25, 25],
                        disc_xz_layers=[100, 75],
                        spectral_normalization=False,
                        activation_hidden_disc='tanh',
                        activation_hidden_gen='tanh',
                        preprocessing=True, batch_size=200,
                        contamination=self.contamination)

        self.clf.fit(self.X_train)

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

    def test_prediction_scores_with_sigmoid(self):
        self.alad = ALAD(activation_hidden_gen='sigmoid',
                         activation_hidden_disc='sigmoid')
        self.alad.fit(self.X_train)

        pred_scores = self.alad.predict(self.X_test)

        roc_auc = roc_auc_score(self.y_test, pred_scores)
        print(f"ROC AUC Score with Sigmoid: {roc_auc}")

        self.assertGreaterEqual(roc_auc, 0)

    def test_prediction_scores_with_relu(self):
        self.alad = ALAD(activation_hidden_gen='relu',
                         activation_hidden_disc='relu')
        self.alad.fit(self.X_train)

        pred_scores = self.alad.predict(self.X_test)

        roc_auc = roc_auc_score(self.y_test, pred_scores)
        print(f"ROC AUC Score with ReLU: {roc_auc}")

        self.assertGreaterEqual(roc_auc, 0)

    def test_model_clone(self):
        # for deep models this may not apply
        clone_clf = clone(self.clf)

    def test_train_more(self):
        initial_scores = self.clf.decision_function(self.X_test)
        self.clf.train_more(self.X_train, epochs=50)
        new_scores = self.clf.decision_function(self.X_test)
        assert (roc_auc_score(self.y_test, new_scores) >= self.roc_floor)
        self.assertNotEqual(initial_scores.tolist(), new_scores.tolist(),
                            "Scores should change after training more")

    def test_plot_learning_curves(self):
        with patch('matplotlib.pyplot.show'):
            self.clf.plot_learning_curves()
        plt.close('all')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
