# -*- coding: utf-8 -*-
"""Variable-radius generation with independently checked domain coverage.

The generation/testing cycle follows the idea of Ji and Dasgupta (2009),
doi:10.1016/j.ins.2008.12.015. This adaptation uses an exact one-sided
binomial bound and spends alpha across rounds instead of repeatedly applying
an uncorrected normal approximation. It is not the original V-detector.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta


def _coverage_lower_bound(covered, total, alpha):
    """One-sided Clopper-Pearson lower bound for fixed Bernoulli trials."""
    if covered == 0:
        return 0.0
    return float(beta.ppf(alpha, covered, total - covered + 1))


def generate_coverage_detectors(self_samples, bounds, rng, n_detectors,
                                self_radius, max_candidates,
                                target_coverage=0.9, coverage_samples=256,
                                coverage_confidence=0.95):
    """Generate spheres, freezing the detector set during each coverage test.

    Inputs are validated by NSA. Random draws rejected as self still count
    toward max_candidates, but not toward the conditional non-self coverage
    test. Uncovered probes become proposals only after the test is finished.
    Each test uses fresh probes, independent of the frozen detector set.
    Round t spends alpha/(t*(t+1)), giving simultaneous confidence at least
    coverage_confidence by the union bound. This concerns geometric volume
    inside bounds outside the self balls, not real-world anomaly recall.
    """
    centers, radii = [], []
    draws, rounds = 0, 0
    lower_bound = 0.0
    last_total, last_covered = 0, 0
    reached = False
    reason = 'candidate_limit'
    while draws < max_candidates:
        proposals, proposed_radii = [], []
        covered, total = 0, 0
        while total < coverage_samples and draws < max_candidates:
            candidate = rng.uniform(*bounds)
            draws += 1
            distance = cdist(candidate[None, :], self_samples).min()
            if not np.isfinite(distance):
                raise ValueError('Candidate distances overflowed; reduce '
                                 'sampling_margin.')
            radius = distance - self_radius
            if radius <= 0:
                continue
            total += 1
            matches = (centers and np.any(
                cdist(candidate[None, :], centers)[0] <= radii))
            if matches:
                covered += 1
            else:
                proposals.append(candidate)
                proposed_radii.append(radius)

        if total == coverage_samples:
            rounds += 1
            alpha = ((1.0 - coverage_confidence)
                     / (rounds * (rounds + 1)))
            lower_bound = _coverage_lower_bound(covered, total, alpha)
            last_total, last_covered = total, covered
            if lower_bound >= target_coverage:
                reached = True
                reason = 'coverage'
                break
        # An incomplete final round is never used to assert coverage.
        # Any preceding bound remains valid after only adding detectors.
        if len(centers) == n_detectors:
            reason = 'detector_limit'
            break
        for candidate, radius in zip(proposals, proposed_radii):
            if centers and np.any(
                    cdist(candidate[None, :], centers)[0] <= radii):
                continue
            centers.append(candidate)
            radii.append(radius)
            if len(centers) == n_detectors:
                break
    diagnostics = {
        'coverage_reached': reached,
        'coverage_lower_bound': lower_bound,
        'coverage_test_count': rounds,
        'coverage_test_samples': last_total,
        'coverage_test_covered': last_covered,
        'termination_reason': reason,
    }
    return (np.asarray(centers).reshape(-1, self_samples.shape[1]),
            np.asarray(radii), draws, diagnostics)
