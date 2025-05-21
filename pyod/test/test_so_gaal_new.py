# -*- coding: utf-8 -*-


import os
import sys
import unittest

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
sys.path.append(os.path.abspath(os.path.dirname("__file__")))

from pyod.models.so_gaal_new import SO_GAAL
from pyod.utils.data import generate_data


class TestSO_GAAL(unittest.TestCase):
    """
    Notes: GAN may yield unstable results, so the test is design for running
    models only, without any performance check.
    """

    def assertHasAttr(self, obj, intended_attr):
        self.assertTrue(hasattr(obj, intended_attr))

    def assertInRange(self, data, lower, upper):
        self.assertGreaterEqual(data.min(), lower)
        self.assertLessEqual(data.max(), upper)

    def setUp(self):
        self.n_train = 1000
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)

        self.clf = SO_GAAL(contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        self.assertTrue(hasattr(self.clf, 'decision_scores_'))
        self.assertIsNotNone(self.clf.decision_scores_)
        self.assertTrue(hasattr(self.clf, 'labels_'))
        self.assertIsNotNone(self.clf.labels_)
        self.assertTrue(hasattr(self.clf, 'threshold_'))
        self.assertIsNotNone(self.clf.threshold_)
        self.assertTrue(hasattr(self.clf, '_mu'))
        self.assertIsNotNone(self.clf._mu)
        self.assertTrue(hasattr(self.clf, '_sigma'))
        self.assertIsNotNone(self.clf._sigma)
        self.assertTrue(hasattr(self.clf, 'discriminator'))
        self.assertIsNotNone(self.clf.discriminator)

    def test_train_scores(self):
        self.assertEqual(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        self.assertEqual(pred_scores.shape[0], self.X_test.shape[0])

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

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
