# -*- coding: utf-8 -*-
"""Analytical contracts for bounded Voronoi negative selection."""

import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from scipy.spatial import QhullError
from sklearn.exceptions import NotFittedError

from pyod.models import _nsa_optimization
from pyod.models._nsa_optimization import generate_annealed_detectors
from pyod.models._nsa_optimization import generate_suppressed_detectors
from pyod.models._nsa_optimization import generate_voronoi_detectors
from pyod.models.nsa import NSA


def generate(samples, n_detectors=100, self_radius=0.1, max_candidates=1000,
             bounds=None, min_radius=0.0):
    samples = np.asarray(samples, dtype=float)
    if bounds is None:
        bounds = np.array([np.zeros(samples.shape[1]),
                           np.ones(samples.shape[1])])
    return generate_voronoi_detectors(
        samples, bounds, np.random.RandomState(42), n_detectors,
        self_radius, max_candidates, min_radius=min_radius)


def sorted_rows(array):
    return np.asarray(sorted(map(tuple, array)))


def test_two_sites_have_only_boundary_shared_vertices():
    centers, radii, count, info = generate([[0.25, 0.5], [0.75, 0.5]])
    assert_allclose(sorted_rows(centers), [[0.5, 0], [0.5, 1]])
    assert_allclose(radii, np.sqrt(5) / 4 - 0.1)
    assert count == 6  # Includes four unshared corners.
    assert info['n_shared_vertices'] == 2
    assert info['n_type_one'] == 0
    assert info['n_type_two'] == 2
    assert not info['truncated']


def test_four_sites_have_an_interior_and_four_boundary_vertices():
    samples = [[0.25, 0.5], [0.5, 0.25], [0.5, 0.75], [0.75, 0.5]]
    centers, radii, count, info = generate(samples)
    assert_allclose(sorted_rows(centers),
                    [[0, 0], [0, 1], [0.5, 0.5], [1, 0], [1, 1]])
    assert_allclose(sorted(radii), [0.15] + [np.sqrt(5) / 4 - 0.1] * 4)
    assert count == 5
    assert info['n_type_one'] == 1
    assert info['n_type_two'] == 4


def test_two_sites_in_three_dimensions():
    samples = [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
    centers, radii, count, info = generate(samples)
    expected = [[0.5, y, z] for y, z in itertools.product((0, 1), repeat=2)]
    assert_allclose(sorted_rows(centers), expected)
    assert_allclose(radii, 0.65)
    assert count == 12
    assert info['n_type_two'] == 4


def test_cubical_sites_have_nineteen_shared_vertices():
    samples = list(itertools.product((0.25, 0.75), repeat=3))
    centers, radii, count, info = generate(samples)
    expected = [point for point in itertools.product((0, 0.5, 1), repeat=3)
                if 0.5 in point]
    assert_allclose(sorted_rows(centers), sorted_rows(expected))
    assert_allclose(radii, np.sqrt(3) / 4 - 0.1)
    assert count == 27
    assert info['n_type_one'] == 1
    assert info['n_type_two'] == 18


def test_duplicates_and_input_order_do_not_change_geometry():
    samples = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    original = generate(samples)
    repeated = generate(np.repeat(samples[::-1], [1, 2, 3, 4], axis=0))
    for index in range(3):
        assert_array_equal(original[index], repeated[index])
    assert repeated[3]['n_unique_sites'] == 4


@pytest.mark.parametrize('dimension', [2, 3])
def test_collinear_sites_are_supported(dimension):
    samples = np.ones((3, dimension)) * 0.5
    samples[:, 0] = [0, 0.5, 1]
    centers, radii, count, _ = generate(samples)
    assert len(centers) == 2 ** dimension
    assert_allclose(np.unique(centers[:, 0]), [0.25, 0.75])
    assert_allclose(radii, np.sqrt(0.25 ** 2 + (dimension - 1) / 4) - 0.1)
    assert count > len(centers)


def test_coplanar_sites_are_supported():
    samples = [[x, y, 0.5]
               for x, y in itertools.product((0.25, 0.75), repeat=2)]
    centers, radii, _, _ = generate(samples)
    assert len(centers) == 10
    distances = cdist(centers, samples)
    assert_allclose(radii, distances.min(axis=1) - 0.1)


def test_expanded_rectangular_domain():
    samples = [[0.25, 0.5], [0.75, 0.5]]
    centers, radii, _, _ = generate(samples, bounds=[[-1, -2], [2, 3]])
    assert_allclose(sorted_rows(centers), [[0.5, -2], [0.5, 3]])
    assert_allclose(radii, np.sqrt(0.25 ** 2 + 2.5 ** 2) - 0.1)


def test_minimum_radius_filters_interior_detector():
    samples = [[0.25, 0.5], [0.5, 0.25], [0.5, 0.75], [0.75, 0.5]]
    centers, radii, _, info = generate(samples, min_radius=0.2)
    assert len(centers) == 4
    assert np.all(radii > 0.2)
    assert info['n_type_one'] == 0


def test_truncation_selects_largest_radii_and_reports_full_count():
    samples = [[0.25, 0.5], [0.5, 0.25], [0.5, 0.75], [0.75, 0.5]]
    full = generate(samples)
    limited = generate(samples, n_detectors=2)
    assert_array_equal(limited[0], full[0][:2])
    assert_array_equal(limited[1], full[1][:2])
    assert limited[2] == full[2]
    assert limited[3]['n_eligible_detectors'] == 5
    assert limited[3]['truncated']


def test_budget_counts_unique_vertices_and_fails_without_partial_result():
    samples = [[0.25, 0.5], [0.75, 0.5]]
    assert generate(samples, max_candidates=6)[2] == 6
    with pytest.raises(ValueError, match='max_candidates'):
        generate(samples, max_candidates=5)


@pytest.mark.parametrize('samples,radius', [
    ([[0.5, 0.5]] * 3, 0.1), ([[0, 0], [1, 1]], 2)])
def test_no_valid_shared_detectors_returns_empty_arrays(samples, radius):
    centers, radii, _, info = generate(samples, self_radius=radius)
    assert centers.shape == (0, 2)
    assert radii.shape == (0,)
    assert info['n_eligible_detectors'] == 0


@pytest.mark.parametrize('dimension', [2, 3])
def test_random_geometry_excludes_self_spheres(dimension):
    samples = np.random.RandomState(7).uniform(size=(25, dimension))
    centers, radii, count, info = generate(samples, self_radius=0.02)
    distances = cdist(centers, samples)
    assert_allclose(radii, distances.min(axis=1) - 0.02)
    assert np.all(distances >= radii[:, None] + 0.02 - 1e-12)
    assert np.all((centers >= 0) & (centers <= 1))
    assert count == info['n_vertices']


@pytest.mark.parametrize('dimension', [1, 4, 10])
def test_dimensions_outside_two_and_three_are_rejected(dimension):
    with pytest.raises(ValueError, match='2 or 3 features'):
        generate(np.zeros((2, dimension)))


@pytest.mark.parametrize('bounds', [
    np.zeros((2, 2)), [[0, 0], [1, np.inf]],
    [[0], [1]], [[1, 1], [0, 0]]])
def test_invalid_domain(bounds):
    with pytest.raises(ValueError, match='positive finite widths'):
        generate([[0, 0], [1, 1]], bounds=bounds)


def test_self_outside_domain():
    with pytest.raises(ValueError, match='contain every self'):
        generate([[-0.1, 0], [1, 1]])


@pytest.mark.parametrize('radius', [-1, np.inf, np.nan])
def test_invalid_radius(radius):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        generate([[0, 0], [1, 1]], self_radius=radius)


@pytest.mark.parametrize('budget', [0, -1, 1.2, True])
def test_invalid_candidate_budget(budget):
    with pytest.raises(ValueError, match='limits must be positive'):
        generate([[0, 0], [1, 1]], max_candidates=budget)


@pytest.mark.parametrize('samples', [np.empty((0, 2)), [[np.nan, 0], [1, 1]]])
def test_invalid_self_sites(samples):
    with pytest.raises(ValueError, match='finite self samples'):
        generate(samples)


def test_nearby_sites_are_not_lost_when_coordinates_are_shifted():
    samples = [[0, 0], [1e-17, 0], [1, 1]]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # Either a resolved tessellation or a controlled numerical-degeneracy
        # error is acceptable; NaN normals and raw solver errors are not.
        try:
            centers, radii, _, _ = generate(samples)
        except ValueError as exc:
            assert 'Voronoi cell' in str(exc)
        else:
            assert np.isfinite(centers).all()
            assert np.isfinite(radii).all()


def test_linear_program_failure_does_not_leave_an_old_fitted_model(
        monkeypatch):
    samples = [[0, 0], [1, 1]]
    model = NSA(strategy='voronoi').fit(samples)
    monkeypatch.setattr(_nsa_optimization, 'linprog',
                        lambda *a, **kw: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match='bounded Voronoi cell'):
        model.fit(samples)
    assert not hasattr(model, 'detectors_')
    assert not hasattr(model, 'decision_scores_')
    with pytest.raises(NotFittedError):
        model.decision_function(samples)


def test_qhull_failure_has_a_controlled_error(monkeypatch):
    def fail(*args, **kwargs):
        raise QhullError('singular geometry')
    monkeypatch.setattr(_nsa_optimization, 'HalfspaceIntersection', fail)
    with pytest.raises(ValueError, match='numerically degenerate'):
        generate([[0, 0], [1, 1]])


def test_nonfinite_geometry_is_never_retained(monkeypatch):
    invalid = SimpleNamespace(intersections=np.array([[np.nan, 0]]))
    monkeypatch.setattr(
        _nsa_optimization, 'HalfspaceIntersection',
        lambda *a, **kw: invalid)
    with pytest.raises(ValueError, match='nonfinite data'):
        generate([[0, 0], [1, 1]])


class SequenceRng:
    """Supply geometric fixtures without depending on PRNG internals."""

    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def uniform(self, low, high):
        result = np.asarray(next(self.sequence))
        assert np.all((result >= low) & (result <= high))
        return result


def suppressed_score(samples, centers, radii, info):
    negative = np.max(radii - cdist(samples, centers), axis=1)
    positive = (cdist(samples, info['reverse_centers']).min(axis=1)
                - info['reverse_radius'])
    return np.minimum(negative, positive)


def test_suppression_preserves_an_outlier_inside_negative_sphere():
    samples = np.array([[0, 0], [0.02, 0], [0, 0.02], [0.02, 0.02], [1, 1]])
    centers, radii, draws, info = generate_suppressed_detectors(
        samples, np.array([[0, 0], [1, 1]]), SequenceRng([[0.7, 0.7]]),
        1, 0.1, 1, outlier_fraction=0.8, outlier_radius=0.1)
    assert_array_equal(info['outlier_mask'], [False] * 4 + [True])
    assert_allclose(info['outlier_distance_threshold'], 0.1 * np.sqrt(2))
    assert info['n_outlier_self'] == 1
    assert info['n_boundary_self'] == 1
    assert draws == 1
    assert_allclose(sorted_rows(info['reverse_centers']),
                    [[0.02, 0.02], [1, 1]])
    assert_allclose(radii, np.sqrt(2 * 0.68 ** 2))
    negative = radii - cdist(samples[-1:], centers)
    assert negative.item() > 0  # The negative sphere alone is unsafe.
    scores = suppressed_score(samples, centers, radii, info)
    assert np.all(scores <= 1e-12)
    assert_allclose(scores[-1], -0.1)
    assert suppressed_score([[0.7, 0.7]], centers, radii, info).item() > 0


def test_new_boundary_self_is_recorded_even_if_candidate_is_covered():
    samples = np.array([[0.25, 0.5], [0.75, 0.5]])
    sequence = [[0.5, 1], [0.8, 0.8], [0.7, 0.8]]
    centers, radii, draws, info = generate_suppressed_detectors(
        samples, np.array([[0, 0], [1, 1]]), SequenceRng(sequence),
        10, 0.05, 3, outlier_fraction=1.0)
    assert_array_equal(centers, sequence[:2])
    assert cdist(centers[:1], centers[1:]).item() < radii[0]
    assert draws == 3
    assert info['n_boundary_self'] == 2
    assert_allclose(sorted_rows(info['reverse_centers']), sorted_rows(samples))


@pytest.mark.parametrize('dimension', [2, 3, 10])
@pytest.mark.parametrize('self_radius', [0.0, 0.1])
def test_suppressed_random_geometry_protects_all_training_points(
        dimension, self_radius):
    rng = np.random.RandomState(19)
    samples = rng.uniform(0, 0.2, size=(29, dimension))
    samples = np.vstack((samples, np.ones(dimension)))
    centers, radii, draws, info = generate_suppressed_detectors(
        samples, np.array([np.zeros(dimension), np.ones(dimension)]),
        np.random.RandomState(2), 20, self_radius, 200,
        outlier_fraction=0.95, outlier_radius=0.2)
    assert info['n_outlier_self'] == 1
    assert 0 < len(centers) <= 20
    assert draws <= 200
    scores = suppressed_score(samples, centers, radii, info)
    assert np.all(scores <= 1e-12)
    assert_allclose(radii, cdist(centers, samples[:-1]).min(axis=1))


def test_suppressed_reproducibility():
    samples = [[0, 0], [0.01, 0], [0, 0.01], [0.01, 0.01], [1, 1]]
    results = [generate_suppressed_detectors(
        samples, np.array([[0, 0], [1, 1]]), np.random.RandomState(0),
        20, 0.1, 50, outlier_fraction=0.8) for _ in range(2)]
    for index in range(3):
        assert_array_equal(results[0][index], results[1][index])
    assert_array_equal(results[0][3]['reverse_centers'],
                       results[1][3]['reverse_centers'])


def test_suppressed_duplicate_candidates_consume_budget():
    centers, _, draws, _ = generate_suppressed_detectors(
        [[0, 0], [1, 1]], np.array([[0, 0], [1, 1]]),
        SequenceRng([[0.5, 0.5]] * 3), 10, 0.1, 3,
        outlier_fraction=1.0)
    assert len(centers) == 1
    assert draws == 3


def test_suppressed_self_candidates_consume_budget_and_return_empty():
    centers, radii, draws, info = generate_suppressed_detectors(
        [[0, 0], [1, 1]], np.array([[0, 0], [1, 1]]),
        SequenceRng([[0, 0], [1, 1]]), 10, 0.1, 2,
        outlier_fraction=1.0)
    assert centers.shape == (0, 2)
    assert radii.shape == (0,)
    assert draws == 2
    assert info['reverse_centers'].shape == (0, 2)


def test_suppressed_all_outliers_has_clear_failure():
    with pytest.raises(ValueError, match='every self sample'):
        generate_suppressed_detectors(
            [[0, 0], [1, 1]], [[0, 0], [1, 1]],
            np.random.RandomState(0), 10, 0.1, 20,
            outlier_fraction=0.5, outlier_radius=0.1)


@pytest.mark.parametrize('fraction', [0, -0.1, 1.1, np.inf, np.nan])
def test_suppressed_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='Outlier fraction'):
        generate_suppressed_detectors(
            [[0, 0], [1, 1]], [[0, 0], [1, 1]],
            np.random.RandomState(0), 10, 0.1, 20,
            outlier_fraction=fraction)


@pytest.mark.parametrize('radius', [0, -0.1, np.inf, np.nan])
def test_suppressed_invalid_distance_threshold(radius):
    with pytest.raises(ValueError, match='Outlier fraction'):
        generate_suppressed_detectors(
            [[0, 0], [1, 1]], [[0, 0], [1, 1]],
            np.random.RandomState(0), 10, 0.1, 20, outlier_radius=radius)


@pytest.mark.parametrize('samples', [np.empty((0, 2)), [[np.nan, 0], [1, 1]]])
def test_suppressed_invalid_self_data(samples):
    with pytest.raises(ValueError, match='finite nonempty self set'):
        generate_suppressed_detectors(
            samples, [[0, 0], [1, 1]], np.random.RandomState(0), 10, 0.1, 20)


@pytest.mark.parametrize('bounds', [[[1, 1], [0, 0]], [[0, 0], [1, np.inf]]])
def test_suppressed_invalid_bounds(bounds):
    with pytest.raises(ValueError, match='finite sampling bounds'):
        generate_suppressed_detectors(
            [[0, 0], [1, 1]], bounds, np.random.RandomState(0), 10, 0.1, 20)


@pytest.mark.parametrize('samples,bounds', [
    ([[1e308, 0], [-1e308, 0]], [[-1e308, -1], [1e308, 1]]),
    ([[0, 0], [1, 1]], [[1e308, 1e308], [1e308, 1e308]])])
def test_suppressed_distance_overflow_fails_clearly(samples, bounds):
    with pytest.raises(ValueError, match='distances must be finite'):
        generate_suppressed_detectors(
            samples, np.array(bounds), np.random.RandomState(0),
            10, 0.1, 20, outlier_fraction=1.0)


def annealed(samples=None, bounds=None, seed=42, **kwargs):
    if samples is None:
        samples = np.array([[0.4, 0.4], [0.6, 0.6]])
    samples = np.asarray(samples, dtype=float)
    if bounds is None:
        bounds = np.array([np.zeros(samples.shape[1]),
                           np.ones(samples.shape[1])])
    params = dict(n_detectors=10, self_radius=0.05, max_candidates=1000,
                  detector_radius=0.1, annealing_steps=20)
    params.update(kwargs)
    return generate_annealed_detectors(
        samples, bounds, np.random.RandomState(seed), **params)


class AnnealingSequenceRng(SequenceRng):
    """Separate uniform points, magnitudes and directions for exact moves."""

    def __init__(self, sequence, magnitudes=(), directions=(), indices=()):
        super().__init__(sequence)
        self.magnitudes = iter(magnitudes)
        self.directions = iter(directions)
        self.indices = iter(indices)

    def uniform(self, low=None, high=None):
        if low is None:
            return next(self.magnitudes)
        return super().uniform(low, high)

    def normal(self, size):
        result = np.asarray(next(self.directions), dtype=float)
        assert result.shape == (size,)
        return result

    def randint(self, high):
        result = next(self.indices)
        assert 0 <= result < high
        return result


def test_annealing_energy_counts_ordered_pairs_and_mean_radius_kernel():
    centers = np.array([[0.0], [0.5]])
    samples = np.array([[1.0]])
    radius, self_radius = 0.25, 0.75
    expected = 2 * np.exp(-4) + np.exp(-4) + np.exp(-1)
    assert_allclose(_nsa_optimization._annealing_energy(
        centers, samples, radius, self_radius), expected)
    moved = centers.copy()
    moved[0] = 0.2
    delta = (_nsa_optimization._annealing_local_cost(
        moved[0], centers[1:], samples, radius, self_radius)
        - _nsa_optimization._annealing_local_cost(
            centers[0], centers[1:], samples, radius, self_radius))
    assert_allclose(delta, _nsa_optimization._annealing_energy(
        moved, samples, radius, self_radius) - expected)


def test_annealing_underflowed_gaussian_width_has_finite_limit():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        affinity = _nsa_optimization._gaussian_affinity(
            np.array([0.0, 0.1]), 0.0)
        energy = _nsa_optimization._annealing_energy(
            np.array([[0.5]]), np.array([[0.0]]),
            np.nextafter(0.0, 1.0), 0.0)
    assert_array_equal(affinity, [1, 0])
    assert energy == 0


def test_annealing_monte_carlo_sizing_uses_inscribed_cube_volume():
    # Domain length 4, estimated nonself fraction 3/4, detector length 1.
    # Twelve post-probe trials retain three distinct valid centers; duplicates
    # consume the remaining budget so no perturbation is attempted.
    rng = AnnealingSequenceRng(
        [[0], [1], [2], [3], [1], [1], [1], [1], [1], [1], [1], [1],
         [1], [1], [2], [3]])
    centers, radii, draws, info = generate_annealed_detectors(
        [[0]], [[0], [4]], rng, 10, 0.1, 16,
        detector_radius=0.5)
    assert_allclose(centers, [[1], [2], [3]])
    assert_allclose(radii, 0.5)
    assert draws == 16
    assert info['volume_probe_count'] == 4
    assert info['nonself_fraction_estimate'] == 0.75
    assert_allclose(info['nonself_volume_estimate'], 3.0)
    assert info['target_detectors'] == 3
    assert not info['estimated_target_capped']
    assert info['target_population_reached']
    assert info['optimization_proposals'] == 0


def test_annealing_optimizes_a_real_population_reproducibly():
    centers, radii, draws, info = annealed()
    second = annealed()
    assert_array_equal(centers, second[0])
    assert_array_equal(radii, second[1])
    assert (draws, info) == second[2:]
    assert info['best_energy'] < 0.1 * info['initial_energy']
    assert info['accepted_moves'] > 0
    assert info['optimization_proposals'] >= info['accepted_moves']
    assert info['annealing_steps_completed'] == 20
    assert info['target_detectors'] == 10
    assert info['estimated_target_capped']
    assert np.all((centers >= 0) & (centers <= 1))
    assert np.all(cdist(centers, [[0.4, 0.4], [0.6, 0.6]])
                  > radii[:, None] + 0.05)
    assert draws <= 1000


@pytest.mark.parametrize('budget', [1, 2, 3, 4, 10, 37, 257])
def test_annealing_all_generation_stages_share_the_candidate_budget(budget):
    centers, radii, draws, info = annealed(max_candidates=budget)
    assert draws <= budget
    assert info['volume_probe_count'] == min(256, max(1, budget // 4))
    assert (info['volume_probe_count'] + len(centers)
            + info['optimization_proposals'] <= draws)
    assert centers.shape == (len(radii), 2)
    assert info['best_energy'] <= info['initial_energy'] + 1e-12


def test_annealing_no_nonself_volume_yields_no_population():
    centers, radii, draws, info = annealed(self_radius=10)
    assert centers.shape == (0, 2)
    assert radii.size == 0
    assert draws == 250
    assert info['target_detectors'] == 0
    assert info['nonself_fraction_estimate'] == 0
    assert info['target_population_reached']


def test_annealing_radius_can_make_population_estimate_zero():
    centers, _, _, info = annealed(detector_radius=10)
    assert len(centers) == 0
    assert info['target_detectors'] == 0
    assert info['nonself_fraction_estimate'] > 0


def test_annealing_strict_self_exclusion_can_exhaust_initialization():
    rng = AnnealingSequenceRng([[0.2], [0.2], [0.2]])
    centers, _, draws, info = generate_annealed_detectors(
        [[0]], [[0], [1]], rng, 1, 0.1, 3, detector_radius=0.2)
    assert len(centers) == 0
    assert draws == 3
    assert info['target_detectors'] == 1
    assert not info['target_population_reached']


@pytest.mark.parametrize('magnitude,direction,expected', [
    (0.5, [1], 0.7),       # downhill: moves away from self
    (0.5, [-1], 0.6),      # uphill accepted, but best initial retained
    (1.0, [-1], 0.6),      # enters a protected self sphere
    (1.0, [1], 0.6),       # outside the bounded domain
])
def test_annealing_moves_use_ball_radius_and_return_best_population(
        magnitude, direction, expected):
    # Perturbation radius .2, initial center .6, self .4. Domain upper .75.
    rng = AnnealingSequenceRng([[0.7], [0.6]],
                               magnitudes=[magnitude, 0],
                               directions=[direction], indices=[0])
    centers, _, draws, info = generate_annealed_detectors(
        [[0.4]], [[0], [0.75]], rng, 1, 0.05, 3,
        detector_radius=0.1, annealing_steps=1)
    assert_allclose(centers, [[expected]])
    assert draws == 3
    assert info['optimization_proposals'] == 1


def test_annealing_zero_direction_consumes_a_proposal():
    rng = AnnealingSequenceRng([[0.5], [0.6]],
                               directions=[[0]], indices=[0])
    centers, _, draws, info = generate_annealed_detectors(
        [[0]], [[0], [1]], rng, 1, 0.05, 3)
    assert_allclose(centers, [[0.6]])
    assert draws == 3
    assert info['optimization_proposals'] == 1
    assert info['accepted_moves'] == 0


def test_annealing_duplicate_proposal_is_not_accepted():
    rng = AnnealingSequenceRng([[0.8], [0.5], [0.75]], magnitudes=[0.5],
                               directions=[[1]], indices=[0])
    centers, _, _, info = generate_annealed_detectors(
        [[0]], [[0], [1]], rng, 2, 0, 4, detector_radius=0.25)
    assert_allclose(centers, [[0.5], [0.75]])
    assert info['optimization_proposals'] == 1
    assert info['accepted_moves'] == 0


@pytest.mark.parametrize('samples', [[], [[np.nan]], np.empty((2, 0))])
def test_annealing_invalid_self_data(samples):
    with pytest.raises(ValueError, match='finite nonempty self'):
        annealed(samples=samples, bounds=[[0], [1]])


@pytest.mark.parametrize('bounds', [
    [[0, 0]], [[0, 0], [0, 1]], [[0, 0], [1, np.inf]]])
def test_annealing_invalid_bounds(bounds):
    with pytest.raises(ValueError, match='positive finite widths'):
        annealed(bounds=bounds)


@pytest.mark.parametrize('param,value', [
    ('detector_radius', 0), ('detector_radius', np.inf),
    ('self_radius', -1), ('self_radius', np.nan),
    ('n_detectors', 0), ('max_candidates', 1.5),
    ('annealing_steps', True), ('annealing_steps', 0)])
def test_annealing_invalid_parameters(param, value):
    with pytest.raises(ValueError):
        annealed(**{param: value})


@pytest.mark.parametrize('bounds,radius,self_radius,match', [
    ([[-1e308], [1e308]], 0.1, 0, 'geometry overflows'),
    ([[0], [1]], 1e308, 0, 'geometry overflows'),
    ([[0], [1]], 8e307, 1e308, 'geometry overflows'),
    ([[0, 0], [1e200, 1e200]], 0.1, 0, 'volume overflows')])
def test_annealing_overflow_geometry_is_rejected_without_warnings(
        bounds, radius, self_radius, match):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(ValueError, match=match):
            annealed(samples=[np.zeros(len(bounds[0]))], bounds=bounds,
                     detector_radius=radius, self_radius=self_radius)


@pytest.mark.parametrize('points', [
    [[1.6e154]],                 # volume probe overflow
    [[5e153], [1.6e154]],        # initialization overflow
    [[5e153], [1.2e154]],        # proposal overflow
])
def test_annealing_distance_overflow_at_every_stage_fails_clearly(points):
    rng = AnnealingSequenceRng(points, magnitudes=[0.2],
                               directions=[[1]], indices=[0])
    with pytest.raises(ValueError, match='distances must be finite'):
        generate_annealed_detectors(
            [[0]], [[0], [2e154]], rng, 1, 0, 3,
            detector_radius=1e154, annealing_steps=1)


def test_annealing_pairwise_energy_overflow_fails_clearly():
    with pytest.raises(ValueError, match='distances must be finite'):
        _nsa_optimization._annealing_energy(
            np.array([[-1e154], [1e154]]), np.array([[0]]), 1e153, 0)
