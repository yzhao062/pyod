# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
from os import path

import unittest
# noinspection PyProtectedMember
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import roc_auc_score

from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.mo_gaal import MO_GAAL
from pyod.utils.data import generate_data


class TestMO_GAAL(unittest.TestCase):
    def setUp(self):
        self.n_train = 5000
        self.n_test = 1000
        self.n_features = 20
        self.contamination = 0.1
        #self.roc_floor = 0.8
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)
        
                # Define data file and read X and y
        # Generate some data if the source data is missing
#        this_directory = path.abspath(path.dirname(__file__))
##        mat_file = 'pima.mat'
#        mat_file = 'cardio.mat'
#        try:
#            mat = loadmat(path.join(*[this_directory, 'data', mat_file]))
#
#        except TypeError:
#            print('{data_file} does not exist. Use generated data'.format(
#                data_file=mat_file))
#            X, y = generate_data(train_only=True)  # load data
#        except IOError:
#            print('{data_file} does not exist. Use generated data'.format(
#                data_file=mat_file))
#            X, y = generate_data(train_only=True)  # load data
#        else:
#            X = mat['X']
#            y = mat['y'].ravel()
#            X, y = check_X_y(X, y)
#        
#        self.X_train, self.X_test, self.y_train, self.y_test = \
#            train_test_split(X, y, test_size=0.4, random_state=42)

        self.clf = MO_GAAL(k = 1, stop_epochs = 2, contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_sklearn_estimator(self):
        # TODO: fix estimator check for AutoEncoder
        # check_estimator(self.clf)
        pass

    def test_parameters(self):
        assert_true(hasattr(self.clf, 'decision_scores_') and
                    self.clf.decision_scores_ is not None)
        assert_true(hasattr(self.clf, 'labels_') and
                    self.clf.labels_ is not None)
        assert_true(hasattr(self.clf, 'threshold_') and
                    self.clf.threshold_ is not None)
        assert_true(hasattr(self.clf, '_mu') and
                    self.clf._mu is not None)
        assert_true(hasattr(self.clf, '_sigma') and
                    self.clf._sigma is not None)
        assert_true(hasattr(self.clf, 'discriminator') and
                    self.clf.discriminator is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        #assert_greater(roc_auc_score(self.y_test, pred_scores), self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert_greater_equal(pred_proba.min(), 0)
        assert_less_equal(pred_proba.max(), 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        assert_greater_equal(pred_proba.min(), 0)
        assert_less_equal(pred_proba.max(), 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert_greater_equal(pred_proba.min(), 0)
        assert_less_equal(pred_proba.max(), 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

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

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
