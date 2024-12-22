# -*- coding: utf-8 -*-


import os
import sys
import unittest

# noinspection PyProtectedMember
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
sys.path.append(os.path.abspath(os.path.dirname("__file__")))

from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data


class TestAutoEncoder(unittest.TestCase):
    def assertHasAttr(self, obj, intended_attr):
        self.assertTrue(hasattr(obj, intended_attr))

    def assertInRange(self, data, lower, upper):
        self.assertGreaterEqual(data.min(), lower)
        self.assertLessEqual(data.max(), upper)

    def setUp(self):
        self.n_train = 6000
        self.n_test = 1000
        self.n_features = 300
        self.contamination = 0.1
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = AutoEncoder(epoch_num=5, contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        self.assertHasAttr(self.clf, 'decision_scores_')
        self.assertIsNotNone(self.clf.decision_scores_)
        self.assertHasAttr(self.clf, 'labels_')
        self.assertIsNotNone(self.clf.labels_)
        self.assertHasAttr(self.clf, 'threshold_')
        self.assertIsNotNone(self.clf.threshold_)
        self.assertHasAttr(self.clf, '_mu')
        self.assertIsNotNone(self.clf._mu)
        self.assertHasAttr(self.clf, '_sigma')
        self.assertIsNotNone(self.clf._sigma)
        self.assertHasAttr(self.clf, 'model')
        self.assertIsNotNone(self.clf.model)

    def test_train_scores(self):
        self.assertEqual(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)
        self.assertEqual(pred_scores.shape[0], self.X_test.shape[0])
        self.assertGreaterEqual(roc_auc_score(self.y_test, pred_scores),
                                self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        self.assertEqual(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_parameter(self):
        self.assertRaises(ValueError, self.clf.predict_proba, self.X_test,
                          method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        self.assertEqual(pred_labels.shape, self.y_test.shape)
        self.assertEqual(confidence.shape, self.y_test.shape)
        self.assertInRange(confidence, 0, 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(self.X_test,
                                                        method='linear',
                                                        return_confidence=True)
        self.assertInRange(pred_proba, 0, 1)
        self.assertEqual(confidence.shape, self.y_test.shape)
        self.assertInRange(confidence, 0, 1)

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test,
                                                      return_stats=False)
        self.assertEqual(pred_labels.shape, self.y_test.shape)

    def test_prediction_with_rejection_stats(self):
        _, [expected_rejrate, ub_rejrate,
            ub_cost] = self.clf.predict_with_rejection(self.X_test,
                                                       return_stats=True)
        self.assertGreaterEqual(expected_rejrate, 0)
        self.assertLessEqual(expected_rejrate, 1)
        self.assertGreaterEqual(ub_rejrate, 0)
        self.assertLessEqual(ub_rejrate, 1)
        self.assertGreaterEqual(ub_cost, 0)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        self.assertEqual(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        self.assertRaises(NotImplementedError, self.clf.fit_predict_score,
                          self.X_test, self.y_test, scoring='something')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
