"""Numerical and API regressions for FastLOF."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from sklearn.neighbors import LocalOutlierFactor

from pyod.models.fastlof import FastLOF


class TestFastLOFRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        self.X = np.r_[rng.normal(0, 0.2, (40, 3)),
                       rng.normal(3, 1.5, (40, 3))]
        self.Q = rng.normal(1, 2, (20, 3))

    def compare_reference(self, X, Q, **kwargs):
        params = dict(n_neighbors=7, metric='euclidean')
        params.update(kwargs)
        fast = FastLOF(threshold=0, random_state=42,
                       chunk_size=len(X), **params).fit(X)
        ref = LocalOutlierFactor(novelty=True, **params).fit(X)
        assert_allclose(fast.decision_scores_,
                        -ref.negative_outlier_factor_, rtol=2e-5, atol=2e-6)
        assert_allclose(fast.decision_function(Q), -ref.score_samples(Q),
                        rtol=2e-5, atol=2e-6)
        return fast

    def test_novelty_matches_exact_reference(self):
        self.compare_reference(self.X, self.Q)

    def test_multichunk_without_pruning_matches_reference(self):
        ref = LocalOutlierFactor(n_neighbors=7, novelty=True).fit(self.X)
        for seed in [0, 17, 42]:
            with self.subTest(seed=seed):
                fast = FastLOF(n_neighbors=7, threshold=0, chunk_size=12,
                               random_state=seed).fit(self.X)
                assert_allclose(fast.decision_scores_,
                                -ref.negative_outlier_factor_, rtol=2e-5)
                assert_allclose(fast.decision_function(self.Q),
                                -ref.score_samples(self.Q), rtol=2e-5)

    def test_later_diagonal_must_be_processed(self):
        # Offset one cannot improve these intra-chunk neighbors. Offset two can.
        X = np.array([[0.], [1.], [100.], [101.], [0.1], [0.15]])
        class IdentityRandomState(np.random.RandomState):
            def permutation(self, n):
                return np.arange(n)

        fast = FastLOF(n_neighbors=1, threshold=0, chunk_size=2,
                       random_state=IdentityRandomState(0)).fit(X)
        ref = LocalOutlierFactor(n_neighbors=1).fit(X)
        assert_allclose(fast.decision_scores_, -ref.negative_outlier_factor_,
                        rtol=2e-5)

    def test_refit_replaces_training_index(self):
        rng = np.random.RandomState(21)
        for shape in [(25, 3), (25, 2), (80, 3)]:
            with self.subTest(shape=shape):
                fast = FastLOF(n_neighbors=5, random_state=42).fit(self.X)
                fast.decision_function(self.Q)
                B = rng.normal(10, 1, shape)
                Q = rng.normal(10, 1, (9, shape[1]))
                fast.fit(B)
                fresh = FastLOF(n_neighbors=5, random_state=42).fit(B)
                assert_allclose(fast.decision_function(Q),
                                fresh.decision_function(Q), rtol=1e-12)

    def test_pruned_points_remain_candidates_in_both_directions(self):
        class IdentityRandomState(np.random.RandomState):
            def permutation(self, n):
                return np.arange(n)

        X = np.array([[0.], [1.], [2.], [100.], [101.], [102.],
                      [0.1], [1.1], [2.1]])
        for active_chunk in [0, 2]:
            with self.subTest(active_chunk=active_chunk):
                fast = FastLOF(n_neighbors=1, chunk_size=3,
                               random_state=IdentityRandomState(0))
                compute = fast._compute_lof
                calls = []

                def prune_after_first_interchunk_pass(*args):
                    lof, lrd, kdist = compute(*args)
                    calls.append(None)
                    if len(calls) == 2:
                        lof = np.zeros(9)
                        lof[active_chunk * 3:active_chunk * 3 + 3] = 2
                    return lof, lrd, kdist

                with patch.object(fast, '_compute_lof',
                                  side_effect=prune_after_first_interchunk_pass):
                    fast.fit(X)
                start = active_chunk * 3
                assert_allclose(fast.kdist_[start:start + 3], 0.1,
                                rtol=1e-12, atol=1e-12)

    def test_float64_large_offsets_preserve_neighborhoods(self):
        self.compare_reference(self.X + 1e8, self.Q + 1e8)

    def test_float32_inputs_match_reference(self):
        fast = self.compare_reference(self.X.astype(np.float32),
                                      self.Q.astype(np.float32))
        self.assertEqual(fast.X_train_.dtype, np.float32)

    def test_weighted_minkowski_matches_reference(self):
        self.compare_reference(self.X, self.Q, metric='minkowski', p=2,
                               metric_params={'w': np.array([1., 9., 0.1])})

    def test_metric_parameter_p_matches_reference(self):
        self.compare_reference(self.X, self.Q, metric='minkowski', p=2,
                               metric_params={'p': 3})

    def test_cosine_zero_vectors_match_reference(self):
        X = self.X.copy()
        X[0] = 0
        fast = FastLOF(n_neighbors=7, threshold=0, chunk_size=len(X),
                       metric='cosine', random_state=42).fit(X)
        ref = LocalOutlierFactor(n_neighbors=7, metric='cosine',
                                novelty=True).fit(X)
        # Every nonzero vector is tied at distance one from a zero vector.
        # Different valid tie choices can change the zero row's LOF.
        assert_allclose(fast.decision_scores_[1:],
                        -ref.negative_outlier_factor_[1:], rtol=2e-5)
        assert_allclose(fast.decision_function(self.Q),
                        -ref.score_samples(self.Q), rtol=2e-5)
        self.assertTrue(np.isfinite(fast.decision_scores_).all())
        self.assertTrue(np.isfinite(fast.decision_function([[0., 0., 0.]])).all())

    def test_metric_aliases_match_reference(self):
        for metric in ['l1', 'l2', 'cityblock', 'manhattan', 'minkowski', 'cosine']:
            with self.subTest(metric=metric):
                self.compare_reference(self.X, self.Q, metric=metric)

    def test_scaled_metrics_use_global_training_parameters(self):
        for metric, key, value in [
            ('seuclidean', 'V', np.var(self.X, axis=0, ddof=1)),
            ('mahalanobis', 'VI', np.linalg.inv(np.cov(self.X.T))),
        ]:
            with self.subTest(metric=metric):
                params = {key: value}
                ref = LocalOutlierFactor(
                    n_neighbors=7, metric=metric, metric_params=params,
                    algorithm='brute', novelty=True).fit(self.X)
                fast = FastLOF(n_neighbors=7, threshold=0, chunk_size=12,
                               metric=metric, algorithm='brute',
                               random_state=42).fit(self.X)
                assert_allclose(fast.decision_scores_,
                                -ref.negative_outlier_factor_, rtol=2e-5)
                assert_allclose(fast.decision_function(self.Q),
                                -ref.score_samples(self.Q), rtol=2e-5)

    def test_metric_params_not_mutated(self):
        params = {'p': 3}
        FastLOF(metric='minkowski', metric_params=params).fit(self.X)
        self.assertEqual(params, {'p': 3})

    def test_invalid_parameters_raise_value_error(self):
        for params in [dict(n_neighbors=0), dict(n_neighbors=-1),
                       dict(n_neighbors=2.5), dict(n_neighbors=True),
                       dict(chunk_size=0), dict(chunk_size=-1),
                       dict(chunk_size=2.5), dict(n_jobs=0),
                       dict(n_jobs=1.5), dict(threshold=np.nan),
                       dict(threshold=-1), dict(p=0.5)]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                FastLOF(**params).fit(self.X)

    def test_single_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            FastLOF().fit([[1., 2.]])

    def test_precomputed_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'precomputed'):
            FastLOF(metric='precomputed').fit(np.zeros((10, 10)))

    def test_duplicates_and_small_neighbor_counts(self):
        X = np.r_[self.X, self.X[:3]]
        for k in [1, 2, 300]:
            with self.subTest(k=k):
                self.compare_reference(X, self.Q, n_neighbors=k)

    def test_fit_builds_index_and_checks_query_features(self):
        fast = FastLOF().fit(self.X)
        self.assertTrue(hasattr(fast, 'nbrs_'))
        with self.assertRaises(ValueError):
            fast.decision_function(np.ones((3, 2)))

    def test_catalog_lists_and_builds_fastlof(self):
        from pyod.utils.knowledge import KnowledgeBase
        from pyod.utils._detector_factory import build_detector_from_plan

        kb = KnowledgeBase()
        names = [a['name'] for a in kb.list_by_data_type('tabular')]
        self.assertIn('FastLOF', names)
        fast = build_detector_from_plan(
            {'detector_name': 'FastLOF', 'params': {'n_neighbors': 7}},
            kb, random_state=42)
        self.assertIsInstance(fast, FastLOF)
        self.assertEqual(fast.random_state, 42)
        self.assertTrue(np.isfinite(fast.fit(self.X).decision_function(self.Q)).all())


if __name__ == '__main__':
    unittest.main()
