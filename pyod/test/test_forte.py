# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models import forte
from pyod.models.forte import Forte
from pyod.models.forte import per_point_prdc


def generate_shifted_data(n_train=200, n_test=100, n_features=16,
                          contamination=0.1, offset=3., random_state=42):
    """Clean training sample and a test sample with a shifted group."""
    rng = np.random.RandomState(random_state)
    n_outliers = int(n_test * contamination)
    X_train = rng.randn(n_train, n_features)
    X_test = rng.randn(n_test, n_features)
    X_test[n_test - n_outliers:] += offset
    y_test = np.zeros(n_test)
    y_test[n_test - n_outliers:] = 1
    return X_train, X_test, y_test


def naive_prdc(X_ref, X, k):
    """Direct per-point PRDC from full distance matrices."""
    dist = cdist(X_ref, X)
    ref_radii = np.sort(cdist(X_ref, X_ref), axis=1)[:, k]
    radii = np.sort(cdist(X, X), axis=1)[:, k]
    inside_ref = dist < ref_radii[:, np.newaxis]
    return np.column_stack([
        (dist < radii).sum(axis=0) / X_ref.shape[0],
        inside_ref.sum(axis=0) / (k * X_ref.shape[0]),
        inside_ref.any(axis=0),
        dist.min(axis=0) < radii])


class TestForte(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_test = generate_shifted_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = Forte(space_splits=[8], contamination=self.contamination,
                         random_state=42)
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
        assert (hasattr(self.clf, 'detector_') and
                self.clf.detector_ is not None)
        assert_equal(len(self.clf.reference_indices_), self.n_train // 2)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])
        assert np.isfinite(self.clf.decision_scores_).all()

    def test_train_scores_fit_half(self):
        # the fit half is scored against the reference half, which is
        # what decision_function does for a new sample
        mask = np.ones(self.n_train, dtype=bool)
        mask[self.clf.reference_indices_] = False
        assert_allclose(self.clf.decision_scores_[mask],
                        self.clf.decision_function(self.X_train[mask]))

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_scores_deterministic(self):
        assert_array_equal(self.clf.decision_function(self.X_test),
                           self.clf.decision_function(self.X_test))

    def test_density_estimators(self):
        for density_estimator in ['kde', 'ocsvm']:
            clf = Forte(density_estimator=density_estimator,
                        space_splits=[8], random_state=42)
            clf.fit(self.X_train)
            assert_equal(len(clf.decision_scores_), self.n_train)
            pred_scores = clf.decision_function(self.X_test)
            assert (roc_auc_score(self.y_test, pred_scores) >=
                    self.roc_floor)

    def test_single_space(self):
        clf = Forte(random_state=42)
        clf.fit(self.X_train)
        assert_equal(len(clf.decision_scores_), self.n_train)
        assert_equal(clf.decision_function(self.X_test).shape[0],
                     self.n_test)

    def test_space_splits(self):
        # each space contributes four PRDC statistics
        clf = Forte(space_splits=[4, 8], random_state=42)
        clf.fit(self.X_train)
        assert_equal(clf.detector_.means_.shape[1], 12)

        for space_splits in [[0], [16], [8, 8], [8, 4]]:
            with assert_raises(ValueError):
                Forte(space_splits=space_splits).fit(self.X_train)

    def test_per_point_prdc(self):
        rng = np.random.RandomState(0)
        X_ref = rng.randn(60, 5)
        X = rng.randn(45, 5) + 0.5
        expected = naive_prdc(X_ref, X, 5)
        assert_allclose(per_point_prdc(X_ref, X, n_neighbors=5), expected)

        # same result when distances are computed in small chunks
        chunk_entries = forte._CHUNK_ENTRIES
        forte._CHUNK_ENTRIES = 200
        try:
            assert_allclose(per_point_prdc(X_ref, X, n_neighbors=5),
                            expected)
        finally:
            forte._CHUNK_ENTRIES = chunk_entries

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

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test,
                                                      return_stats=False)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_predict_rank(self):
        pred_ranks = self.clf._predict_rank(self.X_test)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_invalid_parameters(self):
        with assert_raises(ValueError):
            Forte(density_estimator='something').fit(self.X_train)
        with assert_raises(ValueError):
            Forte(n_neighbors=0).fit(self.X_train)

    def test_small_samples(self):
        # every half of the training data needs more than n_neighbors
        with assert_raises(ValueError):
            Forte(n_neighbors=5).fit(self.X_train[:10])
        # so does a scored sample
        with assert_raises(ValueError):
            self.clf.decision_function(self.X_test[:5])

    def test_feature_mismatch(self):
        with assert_raises(ValueError):
            self.clf.decision_function(self.X_test[:, :8])

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def test_embedding_od(self):
        from pyod.models.embedding import EmbeddingOD

        def encoder(X):
            return np.asarray(X, dtype=np.float64)

        for detector in ['Forte', Forte(space_splits=[8], random_state=42)]:
            clf = EmbeddingOD(encoder=encoder, detector=detector,
                              standardize=False, random_state=42)
            clf.fit(self.X_train)
            assert_equal(len(clf.decision_scores_), self.n_train)
            assert_equal(clf.decision_function(self.X_test).shape[0],
                         self.n_test)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
