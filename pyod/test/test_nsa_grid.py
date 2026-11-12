# -*- coding: utf-8 -*-
"""Geometric and statistical checks for the grid NSA construction."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist

from pyod.models._nsa_grid import (
    _SelfGrid, _filter_candidates, estimate_negative_coverage,
    generate_deterministic_detectors, generate_grid_detectors,
)


class TestSelfGrid(unittest.TestCase):
    def test_nearest_matches_exhaustive_query(self):
        rng = np.random.RandomState(31)
        samples = rng.uniform(size=(120, 3))
        bounds = np.array([[0., 0., 0.], [1., 1., 1.]])
        grid = _SelfGrid(samples, bounds, max_depth=5, min_side=0.05)
        # Queries include domain boundaries and points outside the box.
        queries = np.vstack((rng.uniform(-1, 2, (100, 3)),
                             bounds, samples[:3]))
        actual = [grid.nearest_distance(point) for point in queries]
        expected = np.min(cdist(queries, samples), axis=1)
        assert_allclose(actual, expected, atol=1e-14)
        self.assertGreater(grid.pruned_nodes, 0)
        self.assertLess(grid.distance_evaluations, len(queries) * len(samples))

    def test_sparse_high_dimension_and_constant_features(self):
        samples = np.vstack((np.zeros(48), np.ones(48)))
        bounds = np.array([np.zeros(48), np.ones(48)])
        grid = _SelfGrid(samples, bounds, max_depth=3, min_side=0.)
        self.assertLessEqual(len(grid.nodes), 1 + 3 * len(samples))
        self.assertAlmostEqual(grid.nearest_distance(np.full(48, 0.25)),
                               np.sqrt(48) / 4)
        constant = _SelfGrid(np.ones((4, 2)), np.ones((2, 2)), 4, 0.)
        self.assertEqual(len(constant.nodes), 1)
        self.assertEqual(constant.nearest_distance(np.ones(2)), 0.)


class TestGridDetectors(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(10)
        self.samples = rng.uniform(0.35, 0.65, (30, 2))
        self.bounds = np.array([[0., 0.], [1., 1.]])

    def generate(self, seed=12, **kwargs):
        options = dict(n_detectors=30, self_radius=0.03,
                       max_candidates=1024, grid_depth=4)
        options.update(kwargs)
        return generate_grid_detectors(
            self.samples, self.bounds, np.random.RandomState(seed), **options)

    def test_excludes_every_self_ball_and_respects_budget(self):
        centers, radii, count, diagnostics = self.generate()
        self.assertGreater(len(centers), 0)
        self.assertLessEqual(len(centers), 30)
        self.assertLessEqual(count, 1024)
        expected = cdist(centers, self.samples).min(axis=1) - 0.03
        assert_allclose(radii, expected, atol=1e-14)
        self.assertTrue(np.all(radii > 0))
        self.assertTrue(np.all(centers >= self.bounds[0]))
        self.assertTrue(np.all(centers <= self.bounds[1]))
        self.assertGreater(diagnostics['grid_nodes'], 1)
        self.assertGreater(diagnostics['suppressed_centers'], 0)

    def test_reproducible_and_grid_depth_preserves_geometry(self):
        shallow = self.generate(grid_depth=1)
        deep = self.generate(grid_depth=8)
        repeat = self.generate(grid_depth=8)
        for index in (0, 1, 2):
            assert_array_equal(shallow[index], deep[index])
            assert_array_equal(deep[index], repeat[index])
        self.assertEqual(deep[3], repeat[3])

    def test_small_budget_flush_and_no_survivors(self):
        centers, radii, count, _ = self.generate(max_candidates=3)
        self.assertEqual(count, 3)
        self.assertGreater(len(centers), 0)
        centers, radii, count, _ = self.generate(
            self_radius=10, max_candidates=7)
        self.assertEqual(centers.shape, (0, 2))
        self.assertEqual(radii.shape, (0,))
        self.assertEqual(count, 7)

    def test_detector_limit(self):
        centers, _, count, _ = self.generate(n_detectors=1)
        self.assertEqual(len(centers), 1)
        self.assertEqual(count, 64)

    def test_invalid_depth(self):
        for depth in (True, 0, 21, 1.5, '4'):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.generate(grid_depth=depth)

    def test_overflowing_distance_is_not_an_infinite_detector(self):
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            generate_grid_detectors(
                np.array([[0., 0.]]), np.array([[0., 0.], [1e200, 1e200]]),
                np.random.RandomState(1), 10, 0.1, 10)

    def test_filter_retains_largest_and_preserves_old_coverage(self):
        centers, radii = [], []
        suppressed, removed = _filter_candidates(
            [(np.array([0.2]), 0.1), (np.array([0.]), 0.5)],
            centers, radii, 10)
        self.assertEqual(suppressed, 1)
        self.assertEqual(removed, 0)
        assert_array_equal(centers, [[0.]])
        centers, radii = [np.array([0.])], [0.2]
        _, removed = _filter_candidates([(np.array([0.4]), 0.7)],
                                        centers, radii, 10)
        self.assertEqual(removed, 1)
        assert_array_equal(centers, [[0.4]])
        # Center containment is insufficient: the old left edge -0.2 lies
        # outside the candidate [-0.1, 0.9], so both detectors must survive.
        centers, radii = [np.array([0.])], [0.2]
        _, removed = _filter_candidates([(np.array([0.4]), 0.5)],
                                        centers, radii, 10)
        self.assertEqual(removed, 0)
        self.assertEqual(len(centers), 2)


class TestCoverageDiagnostic(unittest.TestCase):
    def test_analytic_conditional_coverage(self):
        # Self [0.4, 0.6]; detector [0, 0.4]; conditional coverage = 1/2.
        result = estimate_negative_coverage(
            np.array([[0.2]]), np.array([0.2]), np.array([[0.5]]),
            np.array([[0.], [1.]]), 0.1, np.random.RandomState(9),
            n_samples=10000)
        self.assertEqual(result['n_samples'], 10000)
        self.assertTrue(7700 < result['n_nonself'] < 8300)
        self.assertAlmostEqual(result['estimate'], 0.5, delta=0.025)
        self.assertLess(result['lower_bound'], result['estimate'])

    def test_all_covered_exact_binomial_bound(self):
        result = estimate_negative_coverage(
            np.array([[0.5]]), np.array([2.]), np.array([[3.]]),
            np.array([[0.], [1.]]), 0., np.random.RandomState(1),
            n_samples=100, confidence=0.95)
        self.assertEqual(result['estimate'], 1.)
        self.assertAlmostEqual(result['lower_bound'], 0.05 ** (1 / 100))

    def test_no_detectors_and_no_negative_region(self):
        args = (np.empty((0, 1)), np.empty(0), np.array([[0.5]]),
                np.array([[0.], [1.]]))
        uncovered = estimate_negative_coverage(
            *args, 0.1, np.random.RandomState(1), n_samples=100)
        self.assertEqual(uncovered['estimate'], 0.)
        self.assertEqual(uncovered['lower_bound'], 0.)
        empty = estimate_negative_coverage(
            *args, 2., np.random.RandomState(1), n_samples=100)
        self.assertTrue(np.isnan(empty['estimate']))
        self.assertEqual(empty['lower_bound'], 0.)

    def test_invalid_statistical_parameters(self):
        args = (np.array([[0.]]), np.array([0.1]), np.array([[0.5]]),
                np.array([[0.], [1.]]), 0.1, np.random.RandomState(1))
        for options in ({'n_samples': 0}, {'n_samples': True},
                        {'confidence': 1.}, {'confidence': 0.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                estimate_negative_coverage(*args, **options)


class TestDeterministicDetectors(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[0., 0.], [1., 1.]])
        self.samples = np.array([[0.45, 0.4], [0.5, 0.5], [0.55, 0.6]])

    def generate(self, seed=1, **kwargs):
        options = dict(n_detectors=50, self_radius=0.03,
                       max_candidates=512, grid_depth=3)
        options.update(kwargs)
        return generate_deterministic_detectors(
            self.samples, self.bounds, np.random.RandomState(seed), **options)

    def test_lattice_radius_self_exclusion_and_actual_boundary_movement(self):
        centers, radii, count, diagnostics = self.generate()
        self.assertEqual(count, 64)
        assert_allclose(radii, np.sqrt(2) / 16)
        self.assertGreater(len(centers), 0)
        self.assertTrue(np.all(
            cdist(centers, self.samples) >= (radii[:, None] + 0.03) - 1e-14))
        self.assertGreater(diagnostics['boundary_candidates'], 0)
        self.assertGreater(diagnostics['boundary_movements'], 0)
        self.assertTrue(np.all(centers >= 0))
        self.assertTrue(np.all(centers <= 1))

    def test_seed_independence_and_population_limit(self):
        first = self.generate(seed=1, n_detectors=5)
        second = self.generate(seed=123, n_detectors=5)
        assert_array_equal(first[0], second[0])
        assert_array_equal(first[1], second[1])
        self.assertEqual(first[3], second[3])
        self.assertEqual(len(first[0]), 5)
        self.assertGreater(first[3]['thinned_detectors'], 0)
        self.assertTrue(first[3]['truncated'])

    def test_budget_checked_before_allocation(self):
        with self.assertRaisesRegex(ValueError, 'exceeds max_candidates'):
            self.generate(grid_depth=20)

    def test_unsafe_boundary_rejected_after_finite_moves(self):
        centers, radii, count, diagnostics = self.generate(self_radius=10.)
        self.assertEqual(centers.shape, (0, 2))
        self.assertEqual(radii.shape, (0,))
        self.assertEqual(count, 64)
        self.assertEqual(diagnostics['failed_boundary_candidates'], 16)
        self.assertEqual(diagnostics['boundary_movements'], 16 * 30)

    def test_exact_self_center_and_clipped_movement(self):
        # Every lattice site is self; normalized repulsion has zero norm.
        axis = np.array([0.25, 0.75])
        xx, yy = np.meshgrid(axis, axis)
        self.samples = np.column_stack((xx.ravel(), yy.ravel()))
        centers, _, _, diagnostics = self.generate(grid_depth=1)
        self.assertEqual(len(centers), 0)
        self.assertEqual(diagnostics['failed_boundary_candidates'], 4)
        self.assertEqual(diagnostics['boundary_movements'], 120)

    def test_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, 'two features'):
            generate_deterministic_detectors(
                np.zeros((3, 3)), np.array([np.zeros(3), np.ones(3)]),
                None, 10, 0.1, 512)
        with self.assertRaisesRegex(ValueError, 'positive sampling widths'):
            generate_deterministic_detectors(
                np.zeros((3, 2)), np.zeros((2, 2)), None, 10, 0.1, 512)
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            generate_deterministic_detectors(
                np.zeros((3, 2)), np.array([[0., 0.], [1e200, 1e200]]),
                None, 10, 0.1, 512)
        with self.assertRaisesRegex(ValueError, 'distances overflowed'):
            generate_deterministic_detectors(
                np.zeros((3, 2)), np.array([[0., 0.], [1e155, 1e155]]),
                None, 10, 0.1, 512, grid_depth=4)
        # Initial sites have finite distances, but a repelled corner crosses
        # the representable squared-distance limit during movement.
        with self.assertRaisesRegex(ValueError, 'distances overflowed'):
            generate_deterministic_detectors(
                np.zeros((3, 2)), np.array([[0., 0.], [1e154, 1e154]]),
                None, 10, 1.3e154, 512, grid_depth=3)
        for depth in (0, True, 21, 1.2):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.generate(grid_depth=depth)


if __name__ == '__main__':
    unittest.main()
