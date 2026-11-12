# -*- coding: utf-8 -*-

import os
import sys
import unittest

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.abod import ABOD
from pyod.models.abod import _calculate_wocs
from pyod.utils.data import generate_data


class TestFastABOD(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = ABOD(contamination=self.contamination)
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
        assert (hasattr(self.clf, 'tree_') and
                self.clf.tree_ is not None)

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
        clone_clf = clone(self.clf)

    def test_fast_mode_tree_and_neighbor_model_consistent(self):
        assert (hasattr(self.clf, 'neigh_') and self.clf.neigh_ is not None)
        assert (self.clf.tree_ is self.clf.neigh_)

    def test_fast_mode_neighbor_params_propagation(self):
        for algorithm in ['auto', 'kd_tree', 'brute']:
            clf = ABOD(contamination=self.contamination, n_neighbors=5,
                       method='fast', algorithm=algorithm, n_jobs=-1)
            clf.fit(self.X_train)
            assert_equal(clf.neigh_.algorithm, algorithm)
            assert_equal(clf.neigh_.n_jobs, -1)
            pred_scores = clf.decision_function(self.X_test)
            assert_equal(pred_scores.shape[0], self.X_test.shape[0])

    def test_fast_mode_train_scores_use_k_other_points(self):
        # each training point is scored on its n_neighbors nearest other
        # training points, like a test point in decision_function
        X = self.X_train[:40]
        clf = ABOD(n_neighbors=5, method='fast').fit(X)
        dist = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
        for i in range(X.shape[0]):
            others = [j for j in dist[i].argsort() if j != i][:5]
            assert_allclose(clf.decision_scores_[i],
                            -_calculate_wocs(X[i], X, others))

    def test_fast_mode_with_all_neighbors_matches_default(self):
        X = self.X_train[:30]
        fast = ABOD(n_neighbors=X.shape[0] - 1, method='fast').fit(X)
        default = ABOD(method='default').fit(X)
        assert_allclose(fast.decision_scores_, default.decision_scores_)

    def tearDown(self):
        pass


class TestABOD(unittest.TestCase):
    def setUp(self):
        self.n_train = 50
        self.n_test = 50
        self.contamination = 0.2
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = ABOD(contamination=self.contamination, method='default')
        self.clf.fit(self.X_train)

    def test_parameters(self):
        if not hasattr(self.clf,
                       'decision_scores_') or self.clf.decision_scores_ is None:
            self.assertRaises(AttributeError, 'decision_scores_ is not set')
        if not hasattr(self.clf, 'labels_') or self.clf.labels_ is None:
            self.assertRaises(AttributeError, 'labels_ is not set')
        if not hasattr(self.clf, 'threshold_') or self.clf.threshold_ is None:
            self.assertRaises(AttributeError, 'threshold_ is not set')

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

    # def test_score(self):
    #     self.clf.score(self.X_test, self.y_test)
    #     self.clf.score(self.X_test, self.y_test, scoring='roc_auc_score')
    #     self.clf.score(self.X_test, self.y_test, scoring='prc_n_score')
    #     with assert_raises(NotImplementedError):
    #         self.clf.score(self.X_test, self.y_test, scoring='something')

    def test_predict_rank(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=3.5)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=3.5)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def tearDown(self):
        pass


class TestABODKwargsRejection(unittest.TestCase):
    """Regression test for issue #685: ABOD must not forward arbitrary kwargs
    to sklearn NearestNeighbors. Before the fix, ABOD(random_state=42) crashed
    at fit time with "NearestNeighbors.__init__() got an unexpected keyword
    argument 'random_state'". After the fix, unknown kwargs are rejected
    cleanly at construction.
    """

    def test_random_state_rejected_cleanly(self):
        # Caller intent: "give me reproducibility". The detector is
        # deterministic, so the constructor rejects the kwarg rather than
        # silently forwarding it to a layer that does not accept it.
        with self.assertRaises(TypeError) as cm:
            ABOD(random_state=42)
        # Key invariant: the error must NOT leak NearestNeighbors (the
        # pre-fix shape pointed there). Python 3.10+ also prefixes the
        # class name (e.g., "ABOD.__init__()"), but we do not assert that
        # because Python 3.9 omits the class qualifier; checking the
        # kwarg name keeps the assertion meaningful across Python versions.
        msg = str(cm.exception)
        assert 'NearestNeighbors' not in msg, (
            "Error must not leak NearestNeighbors implementation detail; "
            "got: %s" % msg)
        assert 'random_state' in msg, (
            "Error must name the unexpected kwarg; got: %s" % msg)

    def test_unknown_kwarg_rejected_cleanly(self):
        with self.assertRaises(TypeError) as cm:
            ABOD(verbose=1)
        msg = str(cm.exception)
        assert 'NearestNeighbors' not in msg, msg
        assert 'verbose' in msg, msg

    def test_default_construction_works(self):
        ABOD()


def _numpy_abod_scores(model, queries=None):
    training = model.X_train_.astype(np.float64)
    points = training if queries is None else np.asarray(queries, dtype=float)
    if model.method == 'fast':
        # training points are not their own neighbors
        indices = model.neigh_.kneighbors(
            queries, n_neighbors=model.n_neighbors, return_distance=False)
    else:
        indices = [range(len(training))] * len(points)
    result = []
    for point, neighbors in zip(points, indices):
        angles = []
        for first, second in combinations(neighbors, 2):
            a, b = training[first] - point, training[second] - point
            if np.any(a) and np.any(b):
                angles.append(np.dot(a, b) / (np.dot(a, a) * np.dot(b, b)))
        result.append(-np.var(angles))
    return np.array(result)


@pytest.mark.parametrize('method', ['fast', 'default'])
@pytest.mark.parametrize('dtype', [np.bool_, np.uint8, np.int64, np.uint64,
                                   np.float16, np.float32, np.float64])
def test_abod_numeric_dtypes(method, dtype):
    matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1],
                       [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype)
    original = matrix.copy()
    model = ABOD(method=method, n_neighbors=3).fit(matrix)
    expected_dtype = dtype if dtype in (np.float32, np.float64) else np.float64
    assert model.X_train_.dtype == expected_dtype
    assert_allclose(model.decision_scores_, _numpy_abod_scores(model),
                    rtol=1e-5, atol=1e-10)
    queries = np.array([[0, 0, 1], [0, 0, 0]], dtype=dtype)
    assert_allclose(model.decision_function(queries),
                    _numpy_abod_scores(model, queries),
                    rtol=1e-5, atol=1e-10)
    assert_equal(matrix, original)


@pytest.mark.parametrize('method', ['fast', 'default'])
@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
def test_abod_half_prediction(method, train_dtype):
    training = np.array([[0, 0], [1, 3], [4, 1], [7, 8], [11, 2], [13, 17]],
                        dtype=train_dtype)
    model = ABOD(method=method, n_neighbors=3).fit(training)
    queries = np.array([[2, 1], [6, 5]], dtype=np.float16)
    expected = _numpy_abod_scores(model, queries)
    assert_allclose(model.decision_function(queries), expected,
                    rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize('method', ['fast', 'default'])
@pytest.mark.parametrize('input_type', [list, pd.DataFrame])
def test_abod_integer_input_forms(method, input_type):
    matrix = [[0, 0], [1, 3], [4, 1], [7, 8], [11, 2], [13, 17]]
    model = ABOD(method=method, n_neighbors=3).fit(input_type(matrix))
    assert model.X_train_.dtype == np.float64
    assert_allclose(model.decision_scores_, _numpy_abod_scores(model))


if __name__ == '__main__':
    unittest.main()
