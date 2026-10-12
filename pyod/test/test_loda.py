# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np
# noinspection PyProtectedMember
from numpy.testing import assert_allclose, assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.loda import LODA
from pyod.utils.data import generate_data


class TestLODA(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.75
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = LODA(contamination=self.contamination)
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
        assert (hasattr(self.clf, 'projections_') and
                self.clf.projections_ is not None)

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

    def tearDown(self):
        pass


class TestAutoLODA(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.75
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = LODA(contamination=self.contamination, n_bins='auto')
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
        assert (hasattr(self.clf, 'projections_') and
                self.clf.projections_ is not None)

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

    def tearDown(self):
        pass


class TestLODARandomState(unittest.TestCase):
    """Regression test for issue #469: LODA results are not reproducible
    because the constructor did not accept ``random_state`` and the inner
    ``np.random.randn`` / ``np.random.permutation`` calls fell back to
    numpy's module-level state. The fix adds ``random_state`` to
    ``LODA.__init__`` and threads it through both call sites via
    ``sklearn.utils.check_random_state``.
    """

    def setUp(self):
        import numpy as np
        self.X = np.random.RandomState(42).randn(200, 5)

    def test_same_seed_is_deterministic(self):
        import numpy as np
        results = []
        for _ in range(3):
            clf = LODA(random_state=42)
            clf.fit(self.X)
            results.append(clf.decision_scores_.copy())
        assert all(np.array_equal(results[0], r) for r in results[1:])

    def test_different_seeds_can_differ(self):
        import numpy as np
        c1 = LODA(random_state=1); c1.fit(self.X)
        c2 = LODA(random_state=2); c2.fit(self.X)
        # Two different seeds on the same data must not always collide.
        assert not np.array_equal(c1.decision_scores_, c2.decision_scores_)

    def test_no_seed_unchanged(self):
        # Sanity: LODA() without random_state still produces a usable fit;
        # determinism is not asserted to preserve v3.5.1 behavior.
        clf = LODA()
        clf.fit(self.X)
        assert clf.decision_scores_.shape == (self.X.shape[0],)


class TestLODAHistogramLookup(unittest.TestCase):
    def test_bin_interiors(self):
        X = np.array([0., .1, .2, 1.1, 2.1, 3.]).reshape(-1, 1)
        clf = LODA(n_bins=3, n_random_cuts=1, random_state=0).fit(X)
        probabilities = (np.array([3., 1., 2.]) + 1e-12) / (6 + 3e-12)
        expected = -np.log(probabilities[[0, 0, 0, 1, 2, 2]])
        assert_allclose(clf.decision_scores_, expected)
        assert_allclose(clf.decision_function(X), expected)
        assert_allclose(clf.decision_function([[.5], [1.5], [2.5]]),
                        -np.log(probabilities))

    def test_boundaries_and_outside_support(self):
        X = np.array([0., .1, .2, 1.1, 2.1, 3.]).reshape(-1, 1)
        clf = LODA(n_bins=3, n_random_cuts=1, random_state=0).fit(X)
        probabilities = (np.array([3., 1., 2.]) + 1e-12) / (6 + 3e-12)
        assert_allclose(clf.decision_function([[-1.], [0.], [1.],
                                              [2.], [3.], [4.]]),
                        -np.log(probabilities[[0, 0, 1, 2, 2, 2]]))

    def test_histogram_counts_for_multiple_cuts(self):
        rng = np.random.RandomState(7)
        X = rng.lognormal(size=(80, 4))
        queries = np.vstack([X, rng.lognormal(size=(20, 4))])
        for n_bins in (1, 5, 'auto'):
            with self.subTest(n_bins=n_bins):
                clf = LODA(n_bins=n_bins, n_random_cuts=7,
                           random_state=42).fit(X)
                expected = np.zeros(len(queries))
                for cut, projection in enumerate(clf.projections_):
                    edges = clf.limits_[cut]
                    training_values = projection.dot(X.T)
                    counts = []
                    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                        below_hi = (training_values <= hi if j == len(edges)-2
                                    else training_values < hi)
                        counts.append(np.count_nonzero(
                            (training_values >= lo) & below_hi))
                    probabilities = np.array(counts, dtype=float) + 1e-12
                    probabilities /= probabilities.sum()
                    for i, value in enumerate(projection.dot(queries.T)):
                        # Interior edges delimit left-closed bins; outer values
                        # retain the density of the nearest edge bin.
                        bin_index = sum(value >= edge for edge in edges[1:-1])
                        expected[i] -= (clf.weights[cut]
                                        * np.log(probabilities[bin_index]))
                expected /= clf.n_random_cuts
                assert_allclose(clf.decision_scores_, expected[:len(X)])
                assert_allclose(clf.decision_function(queries), expected)


if __name__ == '__main__':
    unittest.main()
