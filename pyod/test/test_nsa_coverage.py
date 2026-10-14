# -*- coding: utf-8 -*-
"""Independent statistical and geometric checks for coverage generation."""
# License: BSD 2 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from scipy.stats import binom, binomtest

from pyod.models._nsa_coverage import (
    _coverage_lower_bound, generate_coverage_detectors)
from pyod.models.nsa import NSA


class PresetProbes:
    def __init__(self, values):
        self.values = iter(values)

    def uniform(self, low, high):
        return np.array([next(self.values)])


@pytest.mark.parametrize('total', [1, 8, 35, 256])
@pytest.mark.parametrize('fraction', [0, 0.25, 0.5, 1])
def test_lower_bound_matches_independent_scipy_binomial_test(total, fraction):
    count = int(total * fraction)
    expected = binomtest(count, total, alternative='greater').proportion_ci(
        confidence_level=0.975, method='exact').low
    assert_allclose(_coverage_lower_bound(count, total, 0.025),
                    expected, atol=1e-12)


@pytest.mark.parametrize('true_coverage', [0.1, 0.5, 0.9])
def test_exact_false_certificate_probability_is_at_most_alpha(true_coverage):
    # Sum over every possible outcome, rather than a stochastic pass/fail.
    counts = np.arange(41)
    lower = np.array([_coverage_lower_bound(k, 40, 0.025) for k in counts])
    probability = binom.pmf(counts, 40, true_coverage)[lower > true_coverage]
    assert probability.sum() <= 0.025 + 1e-12


def test_testing_freezes_detectors_and_excludes_self_probes():
    # The zero is self and cannot count as a failure in non-self coverage.
    # Round 1 must have zero hits even after seeing its first candidate.
    centers, radii, draws, info = generate_coverage_detectors(
        np.array([[0.0]]), np.array([[0.0], [1.0]]),
        PresetProbes([0, 0.2, 0.5, 0.8, 0.8, 0.8, 0.8]),
        20, 0.1, 7, target_coverage=0.2, coverage_samples=3)
    assert draws == 7
    assert_allclose(centers[:, 0], [0.2, 0.5])
    assert_allclose(radii, [0.1, 0.4])
    assert info['coverage_test_count'] == 2
    assert info['coverage_test_covered'] == 3
    assert info['coverage_reached']
    # Round 2 spends .05/(2*3), not the original .05 again.
    assert_allclose(info['coverage_lower_bound'], (0.05 / 6) ** (1 / 3))
    assert info['termination_reason'] == 'coverage'


def test_partial_test_never_certifies_coverage_but_keeps_safe_proposals():
    centers, radii, draws, info = generate_coverage_detectors(
        np.array([[0.0]]), np.array([[0.0], [1.0]]),
        PresetProbes([0.2, 0.8]), 20, 0.1, 2, coverage_samples=256)
    assert len(centers) == 2
    assert np.all(radii > 0)
    assert draws == 2
    assert info['coverage_test_count'] == 0
    assert info['coverage_lower_bound'] == 0
    assert not info['coverage_reached']


def test_detector_cap_is_not_reported_as_coverage_success():
    centers, _, _, info = generate_coverage_detectors(
        np.array([[0.0]]), np.array([[0.0], [1.0]]),
        PresetProbes([0.2, 0.8, 0.9, 0.9]), 1, 0.1, 4,
        target_coverage=0.9, coverage_samples=2)
    assert len(centers) == 1
    assert not info['coverage_reached']
    assert info['termination_reason'] == 'detector_limit'


def test_self_filled_domain_terminates_without_detector():
    centers, _, draws, info = generate_coverage_detectors(
        np.array([[0.0]]), np.array([[0.0], [0.05]]),
        np.random.RandomState(0), 10, 0.1, 17)
    assert centers.shape == (0, 1)
    assert draws == 17
    assert not info['coverage_reached']


def test_numerical_overflow_is_explicit():
    with pytest.raises(ValueError, match='overflowed'):
        generate_coverage_detectors(
            np.array([[-1e300]]), np.array([[0.0], [1.0]]),
            np.random.RandomState(0), 10, 0.1, 10)


def test_real_fit_certifies_and_independent_dense_domain_audit_agrees():
    X = np.c_[np.linspace(0, 1, 25), np.linspace(0, 1, 25)]
    model = NSA(strategy='coverage', n_detectors=200, random_state=7,
                target_coverage=0.85, max_candidates=12000).fit(X)
    assert model.coverage_reached_
    assert model.coverage_lower_bound_ >= 0.85
    assert np.all(cdist(model.detectors_, model.self_samples_)
                  >= model.detector_radii_[:, None] + model.self_radius
                  - 1e-12)
    axis = np.linspace(-0.5, 1.5, 251)
    mesh = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    outside_self = cdist(mesh, X).min(axis=1) > model.self_radius
    actual_fraction = np.mean(model.decision_function(mesh[outside_self]) >= 0)
    assert actual_fraction >= model.target_coverage


def test_fit_exposes_budget_failure_and_clears_coverage_state_on_refit():
    X = [[0., 0.], [1., 1.]]
    model = NSA(strategy='coverage', n_detectors=1, max_candidates=10,
                random_state=0)
    with pytest.warns(UserWarning, match='not certified'):
        model.fit(X)
    assert not model.coverage_reached_
    model.set_params(strategy='fixed', n_detectors=1, max_candidates=100)
    model.fit(X)
    assert not hasattr(model, 'coverage_reached_')
    assert not hasattr(model, 'generation_diagnostics_')


@pytest.mark.parametrize('name', ['target_coverage', 'coverage_confidence'])
@pytest.mark.parametrize('bad', [-1, 0, 1, np.inf, np.nan, True, '0.5'])
def test_probability_parameter_validation(name, bad):
    with pytest.raises(ValueError, match=name):
        NSA(**{name: bad}).fit([[0., 0.], [1., 1.]])


@pytest.mark.parametrize('name', ['coverage_samples', 'grid_depth',
                                  'hierarchy_levels', 'dual_clusters',
                                  'annealing_steps'])
@pytest.mark.parametrize('bad', [-1, 0, 1.5, np.inf, True])
def test_new_integer_parameter_validation(name, bad):
    with pytest.raises(ValueError, match=name):
        NSA(**{name: bad}).fit([[0., 0.], [1., 1.]])
