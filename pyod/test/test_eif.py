# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyod.models.eif import EIF
from pyod.models.eif import MAX_INT
from pyod.models.eif import _ExNode
from pyod.models.eif import _InNode
from pyod.utils.data import generate_data


def _leaf_sizes(node):
    if isinstance(node, _ExNode):
        return [node.size]
    return _leaf_sizes(node.left) + _leaf_sizes(node.right)


class TestEIF(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train,
            n_test=self.n_test,
            contamination=self.contamination,
            random_state=42,
        )

        self.clf = EIF(contamination=self.contamination, random_state=42)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        assert (
            hasattr(self.clf, "decision_scores_")
            and self.clf.decision_scores_ is not None
        )
        assert hasattr(self.clf, "labels_") and self.clf.labels_ is not None
        assert hasattr(self.clf, "threshold_") and self.clf.threshold_ is not None
        assert hasattr(self.clf, "_mu") and self.clf._mu is not None
        assert hasattr(self.clf, "_sigma") and self.clf._sigma is not None
        assert hasattr(self.clf, "max_samples_") and self.clf.max_samples_ is not None
        assert (
            hasattr(self.clf, "extension_level_")
            and self.clf.extension_level_ is not None
        )

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        pred_proba = self.clf.predict_proba(self.X_test)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_linear(self):
        pred_proba = self.clf.predict_proba(self.X_test, method="linear")
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_unify(self):
        pred_proba = self.clf.predict_proba(self.X_test, method="unify")
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method="something")

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test, return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.clf.predict_proba(
            self.X_test, method="linear", return_confidence=True
        )
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_prediction_with_rejection(self):
        pred_labels = self.clf.predict_with_rejection(self.X_test, return_stats=False)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_with_rejection_stats(self):
        _, [expected_rejrate, ub_rejrate, ub_cost] = self.clf.predict_with_rejection(
            self.X_test, return_stats=True
        )
        assert expected_rejrate >= 0
        assert expected_rejrate <= 1
        assert ub_rejrate >= 0
        assert ub_rejrate <= 1
        assert ub_cost >= 0

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test, scoring="roc_auc_score")
        self.clf.fit_predict_score(self.X_test, self.y_test, scoring="prc_n_score")
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test, scoring="something")

    def test_predict_rank(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=3)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=3)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_default_extension_level(self):
        # None defaults to the fully extended level (n_features - 1).
        assert_equal(self.clf.extension_level_, self.X_train.shape[1] - 1)

    def test_extension_level_zero_recovers_iforest(self):
        # extension_level=0 uses axis-parallel splits and should still detect
        # the injected outliers on this easy synthetic data.
        clf = EIF(extension_level=0, contamination=self.contamination, random_state=42)
        clf.fit(self.X_train)
        pred_scores = clf.decision_function(self.X_test)
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    def test_extension_level_invalid(self):
        with assert_raises(ValueError):
            EIF(extension_level=self.X_train.shape[1]).fit(self.X_train)
        with assert_raises(ValueError):
            EIF(extension_level=-1).fit(self.X_train)

    def test_max_samples_invalid(self):
        for max_samples in (0, -5, 0.0, -0.5, 1.5, "bogus"):
            with assert_raises(ValueError):
                EIF(max_samples=max_samples).fit(self.X_train)

    def test_max_samples_float_min_one(self):
        clf = EIF(max_samples=0.001, n_estimators=2, random_state=42)
        clf.fit(self.X_train)
        assert_equal(clf.max_samples_, 1)

    def test_max_samples_one_gives_neutral_scores(self):
        # A one-sample subsample grows single-leaf trees with no split, so
        # every sample gets the same neutral score as an unsplittable root,
        # whether max_samples is 1 or a fraction that rounds down to 1.
        for max_samples, X in ((1, self.X_train), (0.1, self.X_train[:5])):
            clf = EIF(max_samples=max_samples, n_estimators=5, random_state=42)
            clf.fit(X)
            assert_equal(clf.max_samples_, 1)
            for tree in clf._trees:
                assert isinstance(tree, _ExNode)
            assert_allclose(clf.decision_scores_, 0.5)
            assert_allclose(clf.decision_function(self.X_test), 0.5)

    def test_n_estimators_invalid(self):
        for n_estimators in (0, -1, 1.5, "10"):
            with assert_raises(ValueError):
                EIF(n_estimators=n_estimators).fit(self.X_train)

    def test_n_features_mismatch(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(64, 5))
        clf = EIF(n_estimators=5, random_state=42)
        clf.fit(X)
        assert_equal(clf.n_features_, 5)
        assert_equal(clf.decision_function(rng.normal(size=(8, 5))).shape, (8,))
        for n_features in (1, 6):
            with self.assertRaisesRegex(ValueError, "fitted with 5 features"):
                clf.decision_function(rng.normal(size=(8, n_features)))

    def test_identical_rows_are_leaves(self):
        X = np.ones((64, 3))
        clf = EIF(n_estimators=5, max_samples=64, random_state=42)
        clf.fit(X)
        for tree in clf._trees:
            assert isinstance(tree, _ExNode)
            assert_equal(tree.size, 64)
        assert_allclose(clf.decision_scores_, 0.5)

    def test_no_empty_splits_on_constant_features(self):
        rng = np.random.RandomState(0)
        X = np.hstack([rng.normal(size=(128, 2)), np.zeros((128, 1))])
        X[::2] = X[1::2]
        clf = EIF(
            n_estimators=20, max_samples=128, extension_level=0, random_state=42
        )
        clf.fit(X)
        assert any(isinstance(tree, _InNode) for tree in clf._trees)
        for tree in clf._trees:
            assert min(_leaf_sizes(tree)) >= 1

    def test_split_uses_varying_coordinates(self):
        # With one varying coordinate among many constant ones, the zeroed
        # coordinates are taken from the constant ones, so every tree splits
        # at the root instead of turning into a leaf after unlucky draws.
        rng = np.random.RandomState(0)
        X = np.zeros((64, 1000))
        X[:, 500] = rng.normal(size=64)
        X[0, 500] = 50.0
        for extension_level in (0, 500):
            clf = EIF(
                n_estimators=20, extension_level=extension_level, random_state=42
            )
            clf.fit(X)
            for tree in clf._trees:
                assert isinstance(tree, _InNode)
                assert tree.normal[500] != 0
                assert_equal(np.count_nonzero(tree.normal), extension_level + 1)
            assert clf.decision_scores_.max() > clf.decision_scores_.min()
            assert_equal(np.argmax(clf.decision_scores_), 0)

    def test_all_constant_features_are_leaves(self):
        X = np.full((32, 5), 3.0)
        clf = EIF(n_estimators=5, extension_level=0, random_state=42)
        clf.fit(X)
        for tree in clf._trees:
            assert isinstance(tree, _ExNode)
        assert np.isfinite(clf.decision_scores_).all()
        assert_allclose(clf.decision_scores_, clf.decision_scores_[0])
        pred_scores = clf.decision_function(X + 1.0)
        assert np.isfinite(pred_scores).all()
        assert_allclose(pred_scores, pred_scores[0])

    def test_split_draws_match_plain_sequence(self):
        # When every coordinate varies, the root hyperplane of each tree is
        # drawn exactly as intercept, normal, zeroed coordinates from the
        # tree's own random state, so seeded results stay unchanged.
        X = self.X_train
        n_features = X.shape[1]
        n_estimators = 5
        clf = EIF(
            n_estimators=n_estimators,
            max_samples=X.shape[0],
            extension_level=0,
            random_state=42,
        )
        clf.fit(X)
        seeds = np.random.RandomState(42).randint(MAX_INT, size=n_estimators)
        for seed, tree in zip(seeds, clf._trees):
            rng = np.random.RandomState(seed)
            intercept = rng.uniform(X.min(axis=0), X.max(axis=0))
            normal = rng.normal(0.0, 1.0, size=n_features)
            normal[rng.choice(n_features, n_features - 1, replace=False)] = 0.0
            assert isinstance(tree, _InNode)
            assert_equal(tree.intercept, intercept)
            assert_equal(tree.normal, normal)

    def test_model_clone(self):
        clone_clf = clone(self.clf)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
