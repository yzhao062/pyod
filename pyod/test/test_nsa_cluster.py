# -*- coding: utf-8 -*-
"""Tests for hierarchical negative selection mechanisms and invariants."""
# License: BSD 2 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning

from pyod.models._nsa_cluster import (
    generate_dual_detectors, generate_hierarchical_detectors)


@pytest.fixture
def samples():
    rng = np.random.RandomState(8)
    return np.vstack([rng.normal(0.2, 0.025, (25, 2)),
                      rng.normal(0.8, 0.025, (25, 2))])


def generate(samples, **options):
    parameters = dict(bounds=np.array([[0., 0.], [1., 1.]]),
                      rng=np.random.RandomState(31), n_detectors=40,
                      self_radius=0.02, max_candidates=5000, n_levels=4)
    parameters.update(options)
    return generate_hierarchical_detectors(samples, **parameters)


def test_hierarchy_refines_self_covers_and_retains_all_samples(samples):
    centers, radii, count, diagnostic = generate(samples)
    assert centers.shape[1] == samples.shape[1]
    assert radii.shape == (len(centers),)
    assert 0 < len(centers) <= 40
    assert len(diagnostic['cluster_counts']) == 4
    assert (np.diff(diagnostic['cluster_counts']) >= 0).all()
    assert diagnostic['cluster_counts'][0] == 1
    assert diagnostic['cluster_counts'][-1] > 1
    assert count == sum(diagnostic['level_candidate_counts'])
    assert len(centers) == sum(diagnostic['level_detector_counts'])
    assert len(centers) <= count <= 5000
    for cluster_centers, cluster_radii in diagnostic['clusters']:
        distances = cdist(samples, cluster_centers)
        assert np.all(np.any(distances <= cluster_radii, axis=1))


@pytest.mark.parametrize('levels', [1, 2, 4, 10])
def test_every_retained_sphere_excludes_all_self_balls(levels, samples):
    centers, radii, _, _ = generate(samples, n_levels=levels)
    distances = cdist(centers, samples)
    assert np.all(distances.min(axis=1) >= radii + 0.02 - 1e-14)
    assert np.isfinite(centers).all()
    assert np.isfinite(radii).all()
    assert (radii > 0).all()
    # No later detector is centered inside a previously retained sphere.
    for index in range(1, len(centers)):
        previous = cdist(centers[index:index + 1], centers[:index])[0]
        assert np.all(previous > radii[:index])


def test_finer_levels_recover_gap_hidden_by_coarse_self_ball(samples):
    coarse_centers, coarse_radii, _, _ = generate(samples, n_levels=1)
    fine_centers, fine_radii, _, diagnostic = generate(samples, n_levels=4)
    gap = np.array([[0.5, 0.5]])
    coarse_score = (coarse_radii - cdist(gap, coarse_centers)).max()
    fine_score = (fine_radii - cdist(gap, fine_centers)).max()
    assert coarse_score < 0
    assert fine_score > 0
    assert sum(count > 0 for count in
               diagnostic['level_detector_counts']) > 1
    self_scores = (fine_radii - cdist(samples, fine_centers)).max(axis=1)
    assert np.all(self_scores < fine_score)


def test_candidates_use_parent_cluster_boxes_with_global_bounds(samples):
    centers, radii, _, diagnostic = generate(samples)
    offset = 0
    for level, count in enumerate(diagnostic['level_detector_counts']):
        boxes = diagnostic['sampling_boxes'][level]
        assert np.all(boxes[:, 0] >= 0)
        assert np.all(boxes[:, 1] <= 1)
        for center in centers[offset:offset + count]:
            inside = np.all(center >= boxes[:, 0], axis=1)
            inside &= np.all(center <= boxes[:, 1], axis=1)
            assert np.any(inside)
        if level:
            previous_centers, previous_radii = (
                diagnostic['clusters'][level - 1])
            assert len(boxes) == len(previous_centers)
            assert_allclose(boxes[:, 0],
                            np.maximum(0, previous_centers
                                       - previous_radii[:, None] - 0.02))
            assert_allclose(boxes[:, 1],
                            np.minimum(1, previous_centers
                                       + previous_radii[:, None] + 0.02))
        offset += count
    assert len(radii) == offset


def test_reproducibility_and_no_input_mutation(samples):
    original = samples.copy()
    first = generate(samples)
    second = generate(samples)
    assert_array_equal(first[0], second[0])
    assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert_array_equal(samples, original)


@pytest.mark.parametrize('budget', [1, 2, 3, 7, 20])
def test_small_budgets_and_quotas_never_overrun(samples, budget):
    centers, radii, count, diagnostic = generate(
        samples, max_candidates=budget, n_detectors=2)
    assert 0 <= len(centers) <= 2
    assert len(radii) == len(centers)
    assert count <= budget
    assert count == sum(diagnostic['level_candidate_counts'])
    assert len(diagnostic['cluster_counts']) == 4
    assert diagnostic['level_candidate_counts'][-1] > 0


@pytest.mark.parametrize('limit, skipped', [(1, 3), (2, 2), (3, 1), (5, 0)])
def test_small_detector_populations_reserve_slots_for_finer_levels(
        samples, limit, skipped):
    centers, radii, count, diagnostic = generate(samples, n_detectors=limit)
    assert len(centers) == limit
    assert len(diagnostic['cluster_counts']) == 4
    assert diagnostic['level_detector_counts'][-1] > 0
    assert diagnostic['level_detector_counts'][:skipped] == [0] * skipped
    assert diagnostic['level_candidate_counts'][:skipped] == [0] * skipped
    assert sum(diagnostic['level_detector_counts']) == limit
    assert sum(diagnostic['level_candidate_counts']) == count
    assert np.all(cdist(centers, samples).min(axis=1)
                  >= radii + 0.02 - 1e-14)


@pytest.mark.parametrize('limit', [1, 5])
def test_one_draw_is_reserved_for_a_fine_detector_inside_coarse_self_cover(
        samples, limit):
    class UpperCorner:
        def randint(self, size):
            return 0

        def uniform(self, low, high):
            return high.copy()

    centers, radii, count, diagnostic = generate(
        samples, n_detectors=limit, max_candidates=1, rng=UpperCorner())
    assert count == 1
    assert diagnostic['level_candidate_counts'] == [0, 0, 0, 1]
    assert diagnostic['level_detector_counts'] == [0, 0, 0, 1]
    assert len(centers) == 1
    # The single retained center lies in space the coarse ball would reject,
    # but finer self covers permit a sphere without weakening self exclusion.
    coarse_centers, coarse_radii = diagnostic['clusters'][0]
    assert cdist(centers, coarse_centers)[0, 0] < coarse_radii[0]
    assert np.all(cdist(centers, samples).min(axis=1) >= radii + 0.02)
    final_box = diagnostic['sampling_boxes'][-1][0]
    assert_array_equal(centers[0], final_box[1])


def test_slot_reservation_uses_actual_refinement_depth():
    X = np.array([[0., 0.], [1., 1.]])
    centers, _, _, diagnostic = generate(X, n_detectors=1, n_levels=100)
    assert diagnostic['cluster_counts'] == [1, 2]
    assert diagnostic['level_detector_counts'] == [0, 1]
    assert diagnostic['level_candidate_counts'][0] == 0
    assert len(centers) == 1


@pytest.mark.parametrize('X', [
    np.array([[0.5, 0.5]]), np.full((10, 2), 0.5),
])
def test_singletons_and_duplicate_self_points_stop_redundant_refinement(X):
    centers, radii, _, diagnostic = generate(X)
    assert diagnostic['cluster_counts'] == [1]
    assert len(centers) > 0
    assert np.all(cdist(centers, X).min(axis=1) >= radii + 0.02)


def test_distinct_singleton_children_stop_without_empty_extra_levels():
    X = np.array([[0., 0.], [1., 1.]])
    _, _, _, diagnostic = generate(X, n_levels=100, self_radius=0.)
    assert diagnostic['cluster_counts'] == [1, 2]
    centers, radii = diagnostic['clusters'][-1]
    assert_array_equal(centers, X)
    assert_allclose(radii, 0., atol=np.finfo(float).tiny)


def test_self_covering_domain_returns_empty_without_invalid_fallback(samples):
    centers, radii, count, diagnostic = generate(
        samples, self_radius=10., max_candidates=17)
    assert centers.shape == (0, 2)
    assert radii.shape == (0,)
    assert count == 17
    assert diagnostic['level_detector_counts'] == [0]


def test_constant_domain_does_not_produce_duplicate_detectors():
    X = np.zeros((4, 2))
    centers, radii, count, _ = generate(
        X, bounds=np.zeros((2, 2)), max_candidates=9)
    assert centers.shape == (0, 2)
    assert radii.size == 0
    assert count == 9


@pytest.mark.parametrize('levels', [0, -1, 2.5, True, 'four'])
def test_invalid_hierarchy_levels_raise(levels, samples):
    with pytest.raises(ValueError, match='n_levels'):
        generate(samples, n_levels=levels)


def test_overflowing_distances_raise_clear_error():
    with pytest.raises(ValueError, match='overflow'):
        generate(np.array([[0., 0.], [1e200, 1e200]]))
    with pytest.raises(ValueError, match='overflow'):
        generate(np.array([[0., 0.]]),
                 bounds=np.array([[-1e200, -1e200], [1e200, 1e200]]))


@pytest.fixture
def dual_samples():
    rng = np.random.RandomState(8)
    return np.vstack([rng.normal(0.2, 0.08, (25, 2)),
                      rng.normal(0.8, 0.08, (25, 2))])


def generate_dual(samples, **options):
    parameters = dict(bounds=np.array([[-0.2, -0.2], [1.2, 1.2]]),
                      rng=np.random.RandomState(31), n_detectors=40,
                      self_radius=0.02, max_candidates=5000, dual_clusters=3)
    parameters.update(options)
    return generate_dual_detectors(samples, **parameters)


@pytest.mark.parametrize('self_radius, direction', [(0.02, 1), (0.1, -1)])
def test_dual_adjusts_kmeans_count_using_published_radius_interval(
        dual_samples, self_radius, direction):
    _, _, _, diagnostic = generate_dual(dual_samples, self_radius=self_radius)
    counts = diagnostic['apc_cluster_counts']
    assert len(counts) > 1
    assert counts[1] - counts[0] == direction
    assert diagnostic['apc_radius_criterion_met']
    assert diagnostic['apc_termination_reason'] == 'radius_interval'
    raw_radii = diagnostic['apc_cluster_radii']
    assert np.all(raw_radii >= 2 * self_radius)
    assert np.all(raw_radii <= 5 * self_radius)
    assert np.all(diagnostic['apc_radii'] > raw_radii + self_radius)


def test_dual_populations_enclose_self_and_exclude_self_balls(dual_samples):
    centers, radii, count, diagnostic = generate_dual(dual_samples)
    assert len(centers) == 40
    assert len(centers) <= count <= 5000
    assert np.all(cdist(centers, dual_samples).min(axis=1)
                  >= radii + 0.02 - 1e-14)
    apc_centers = diagnostic['apc_centers']
    apc_radii = diagnostic['apc_radii']
    assert np.all(np.any(cdist(dual_samples, apc_centers)
                         <= apc_radii - 0.02, axis=1))
    assert np.all(np.any(cdist(centers, apc_centers) <= apc_radii, axis=1))
    assert np.all(centers >= -0.2)
    assert np.all(centers <= 1.2)
    for index in range(1, len(centers)):
        previous = cdist(centers[index:index + 1], centers[:index])[0]
        assert np.all(previous > radii[:index])
    assert diagnostic['candidate_matching_self'] > 0
    assert diagnostic['candidate_already_covered'] > 0


def test_dual_radius_expansion_protects_neighborhoods_not_only_self_centers(
        dual_samples):
    centers, radii, _, diagnostic = generate_dual(dual_samples)
    rng = np.random.RandomState(1)
    offsets = rng.normal(size=(500, 2))
    offsets *= 0.02 / np.linalg.norm(offsets, axis=1)[:, None]
    points = np.repeat(dual_samples, 10, axis=0) + offsets
    apc_margin = np.min(cdist(points, diagnostic['apc_centers'])
                        - diagnostic['apc_radii'], axis=1)
    negative_margin = np.max(radii - cdist(points, centers), axis=1)
    assert np.all(apc_margin <= 1e-14)
    assert np.all(negative_margin <= 1e-14)


def test_dual_sampling_rejects_points_outside_bounds(dual_samples):
    bounds = np.vstack([dual_samples.min(axis=0), dual_samples.max(axis=0)])
    centers, _, _, diagnostic = generate_dual(dual_samples, bounds=bounds)
    assert diagnostic['candidate_outside_bounds'] > 0
    assert np.all(centers >= bounds[0])
    assert np.all(centers <= bounds[1])


def test_dual_reproducibility_and_no_input_mutation(dual_samples):
    original = dual_samples.copy()
    first = generate_dual(dual_samples)
    second = generate_dual(dual_samples)
    assert_array_equal(first[0], second[0])
    assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    for name in ('apc_centers', 'apc_radii', 'apc_cluster_counts'):
        assert_array_equal(first[3][name], second[3][name])
    assert_array_equal(dual_samples, original)


@pytest.mark.parametrize('budget', [1, 2, 7])
def test_dual_candidate_budget_is_global_and_includes_rejections(
        dual_samples, budget):
    centers, radii, attempts, diagnostic = generate_dual(
        dual_samples, max_candidates=budget)
    assert attempts == budget
    rejections = sum(diagnostic[name] for name in (
        'candidate_outside_bounds', 'candidate_matching_self',
        'candidate_already_covered'))
    assert attempts == len(centers) + rejections
    assert len(radii) == len(centers)
    assert diagnostic['termination_reason'] == 'candidate_limit'


def test_dual_infeasible_radius_interval_stops_and_is_reported(dual_samples):
    _, _, _, diagnostic = generate_dual(dual_samples, self_radius=0.03)
    assert not diagnostic['apc_radius_criterion_met']
    assert diagnostic['apc_termination_reason'] == 'repeated_cluster_count'
    counts = diagnostic['apc_cluster_counts']
    assert len(counts) == len(set(counts))


def test_dual_adaptation_has_a_finite_clustering_budget():
    X = np.linspace(0, 1, 40)[:, None]
    _, _, attempts, diagnostic = generate_dual(
        X, bounds=np.array([[0.], [1.]]), self_radius=1e-5,
        dual_clusters=1, max_candidates=1)
    assert attempts == 1
    assert len(diagnostic['apc_cluster_counts']) == 32
    assert diagnostic['apc_termination_reason'] == 'clustering_limit'
    assert not diagnostic['apc_radius_criterion_met']


@pytest.mark.parametrize('X', [
    np.array([[0.5, 0.5]]), np.full((10, 2), 0.5),
])
def test_dual_constant_data_does_not_fabricate_negative_detectors(X):
    centers, radii, count, diagnostic = generate_dual(
        X, dual_clusters=100, max_candidates=13)
    assert centers.shape == (0, 2)
    assert radii.shape == (0,)
    assert count == 13
    assert diagnostic['apc_cluster_counts'] == [1]
    assert diagnostic['apc_termination_reason'] == 'cluster_count_boundary'
    assert not diagnostic['apc_radius_criterion_met']


def test_dual_near_distinct_points_keep_all_members_of_occupied_clusters():
    X = np.array([[0., 0.], [1e-17, 0.], [1., 1.]])
    with pytest.warns(ConvergenceWarning):
        centers, radii, _, diagnostic = generate_dual(
            X, rng=np.random.RandomState(0), self_radius=0.1)
    assert len(centers) > 0
    assert np.all(cdist(centers, X).min(axis=1) >= radii + 0.1 - 1e-14)
    distances = cdist(X, diagnostic['apc_centers'])
    assert np.all(np.any(distances <= diagnostic['apc_radii'] - 0.1,
                         axis=1))


@pytest.mark.parametrize('clusters', [0, -1, 1.5, True, 'three'])
def test_dual_invalid_cluster_count_raises(dual_samples, clusters):
    with pytest.raises(ValueError, match='dual_clusters'):
        generate_dual(dual_samples, dual_clusters=clusters)
