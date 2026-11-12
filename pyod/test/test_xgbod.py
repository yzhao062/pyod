# -*- coding: utf-8 -*-


import os
import sys
import unittest
from os import path

import numpy as np
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.xgbod import XGBOD
from pyod.utils.data import generate_data


class TestXGBOD(unittest.TestCase):
    def setUp(self):
        # Define data file and read X and y
        # Generate some data if the source data is missing
        this_directory = path.abspath(path.dirname(__file__))
        csv_file = 'pima.csv'
        try:
            import numpy as np
            data = np.genfromtxt(
                path.join(*[this_directory, 'data', csv_file]),
                delimiter=',', skip_header=1)

        except IOError:
            print('{data_file} does not exist. Use generated data'.format(
                data_file=csv_file))
            X, y = generate_data(train_only=True)  # load data
        else:
            X = data[:, :-1]
            y = data[:, -1].astype(int)
            X, y = check_X_y(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4, random_state=42)

        self.clf = XGBOD(random_state=42)
        self.clf.fit(self.X_train, self.y_train)

        self.roc_floor = 0.75

    def test_parameters(self):
        assert (hasattr(self.clf, 'clf_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, '_scalar') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'n_detector_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'X_train_add_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)

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

    # def test_prediction_proba_linear(self):
    #     pred_proba = self.clf.predict_proba(self.X_test, method='linear')
    #     assert (pred_proba.min() >= 0)
    #     assert (pred_proba.max() <= 1)
    #
    # def test_prediction_proba_unify(self):
    #     pred_proba = self.clf.predict_proba(self.X_test, method='unify')
    #     assert (pred_proba.min() >= 0)
    #     assert (pred_proba.max() <= 1)
    #
    # def test_prediction_proba_parameter(self):
    #     with assert_raises(ValueError):
    #         self.clf.predict_proba(self.X_test, method='something')

    # def test_prediction_labels_confidence(self):
    #     pred_labels, confidence = self.clf.predict(self.X_test,
    #                                                return_confidence=True)
    #     assert_equal(pred_labels.shape, self.y_test.shape)
    #     assert_equal(confidence.shape, self.y_test.shape)
    #     assert (confidence.min() >= 0)
    #     assert (confidence.max() <= 1)
    #
    # def test_prediction_proba_linear_confidence(self):
    #     pred_proba, confidence = self.clf.predict_proba(self.X_test,
    #                                                     method='linear',
    #                                                     return_confidence=True)
    #     assert (pred_proba.min() >= 0)
    #     assert (pred_proba.max() <= 1)
    #
    #     assert_equal(confidence.shape, self.y_test.shape)
    #     assert (confidence.min() >= 0)
    #     assert (confidence.max() <= 1)

    # def test_prediction_with_rejection(self):
    #    pred_labels = self.clf.predict_with_rejection(self.X_test, return_stats = False)
    #    assert_equal(pred_labels.shape, self.y_test.shape)

    # def test_prediction_with_rejection_stats(self):
    #    _, [expected_rejrate, ub_rejrate, ub_cost] = self.clf.predict_with_rejection(self.X_test, return_stats = True)
    #    assert (expected_rejrate >= 0)
    #    assert (expected_rejrate <= 1)
    #    assert (ub_rejrate >= 0)
    #    assert (ub_rejrate <= 1)
    #    assert (ub_cost >= 0)

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

    def test_predict_rank(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)
        print(pred_ranks)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), rtol=4)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), rtol=4)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        assert_equal(clone_clf.get_params(), self.clf.get_params())

    def test_estimator_list_not_mutated_by_fit(self):
        clf = XGBOD(random_state=42)
        assert (clf.get_params(deep=False)['estimator_list'] is None)
        assert (clf.get_params(deep=False)['standardization_flag_list'] is
                None)

        clf.fit(self.X_train, self.y_train)

        # constructor arguments must stay untouched after fit
        assert (clf.get_params(deep=False)['estimator_list'] is None)
        assert (clf.get_params(deep=False)['standardization_flag_list'] is
                None)

        # the resolved values live on the trailing-underscore attributes
        assert (clf.estimator_list_ is not None)
        assert (clf.standardization_flag_list_ is not None)
        assert_equal(len(clf.estimator_list_),
                     len(clf.standardization_flag_list_))

        # a refit on a different subset should not reuse stale detectors
        clf2 = XGBOD(random_state=42)
        clf2.fit(self.X_test, self.y_test)
        assert_equal(len(clf2.estimator_list_), len(clf.estimator_list_))

    def test_refit_with_fewer_samples(self):
        # Initial fit on >= 51 rows instantiates detectors with n_neighbors up to 50
        X_large, y_large = generate_data(
            n_train=60, n_features=2, contamination=0.1, random_state=42,
            train_only=True)
        # Refit on < 51 rows (e.g. 35) must re-resolve valid detectors without error
        X_small, y_small = generate_data(
            n_train=35, n_features=2, contamination=0.1, random_state=42,
            train_only=True)

        clf = XGBOD(random_state=42)
        clf.fit(X_large, y_large)

        clf.fit(X_small, y_small)
        assert (clf.get_params(deep=False)['estimator_list'] is None)

        pred_scores = clf.decision_function(X_small)
        assert_equal(pred_scores.shape[0], X_small.shape[0])
        assert (np.isfinite(pred_scores).all())

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
