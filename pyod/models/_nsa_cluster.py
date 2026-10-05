# -*- coding: utf-8 -*-
"""Hierarchical sampling and dual self/non-self populations for NSA.

The coarse-to-fine self covers, restricted candidate boxes, and detector
clearance from cluster boundaries follow the ideas in Chen et al. (2011),
``A Negative Selection Algorithm Based on Hierarchical Clustering of Self
Set and its Application in Anomaly Detection``, DOI:
10.2991/ijcis.2011.4.4.1, Sections 3.1.2 and 3.2.

This is an NSA-inspired adaptation, not the published HC-RNSA or CB-RNSA.
It uses Euclidean enclosing balls and explicit finite sampling budgets.
It does not implement the paper's PCA preprocessing, fractional distance,
or hypothesis-test coverage stopping rule. A deterministic greedy cover
and balanced per-level budgets make those implementation choices explicit.

The dual-population helper follows the APC/T-cell mechanism of Zheng, Zhou
and Fang (2013), DOI:10.4304/jcp.8.8.1951-1959, with bounded clustering and
sampling, expanded APC self balls, and continuous PyOD scores.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def _ball_cover(samples, members, target_radius):
    """Partition one parent into covered subsets without losing self points."""
    uncovered = np.asarray(members, dtype=int)
    clusters = []
    while uncovered.size:
        anchor = samples[uncovered[0]]
        distances = cdist(anchor[None, :], samples[uncovered])[0]
        covered = distances <= target_radius
        child_members = uncovered[covered]
        center = samples[child_members].mean(axis=0)
        radius = cdist(center[None, :], samples[child_members]).max()
        # Round outward so numerical error never shrinks the enclosing ball.
        radius = np.nextafter(radius, np.inf)
        clusters.append((center, radius, child_members))
        uncovered = uncovered[~covered]
    return clusters


def _build_hierarchy(samples, self_radius, n_levels):
    """Refine nested memberships while halving the cover's target radius."""
    center = samples.mean(axis=0)
    radius = cdist(center[None, :], samples).max()
    if not np.isfinite(radius):
        raise ValueError('Self-cluster distances overflowed; rescale X.')
    root = (center, np.nextafter(radius, np.inf),
            np.arange(len(samples)))
    levels = [[root]]
    target_radius = radius
    for _ in range(1, n_levels):
        if target_radius <= self_radius:
            break
        target_radius *= 0.5
        next_level = []
        for _, _, members in levels[-1]:
            next_level.extend(_ball_cover(samples, members, target_radius))
        levels.append(next_level)
        if all(len(cluster[2]) == 1 for cluster in next_level):
            break
    return levels


def generate_hierarchical_detectors(self_samples, bounds, rng, n_detectors,
                                    self_radius, max_candidates, n_levels=4):
    """Generate censored spheres using progressively finer self-ball covers.

    Parameters
    ----------
    self_samples : ndarray of shape (n_samples, n_features)
        Finite scaled self observations. The caller validates input data.
    bounds : ndarray of shape (2, n_features)
        Finite lower and upper candidate bounds containing the self samples.
    rng : numpy.random.RandomState
        Random generator supplied by the parent estimator.
    n_detectors : int
        Positive upper bound on the number of retained detectors.
    self_radius : float
        Nonnegative self exclusion radius in scaled Euclidean space.
    max_candidates : int
        Positive global budget, including rejected candidates.
    n_levels : int, optional (default=4)
        Maximum number of hierarchy levels. Refinement stops earlier when
        the target cluster radius reaches the self radius or singleton
        clusters are reached.

    Returns
    -------
    centers : ndarray of shape (n_retained, n_features)
        Accepted negative detector centers; possibly empty.
    radii : ndarray of shape (n_retained,)
        Positive Euclidean radii excluding every training self ball.
    n_candidates : int
        Number of candidate draws consumed across all levels.
    diagnostics : dict
        Per-level self covers, candidate boxes, and sampling counts.

    Notes
    -----
    The first level samples the full supplied domain. Each finer level
    samples boxes around the preceding self clusters, clipped to that
    domain. A box is selected uniformly, then a point uniformly inside it;
    this is a proposal distribution, not uniform sampling of the box union.

    Candidate radius is the smallest distance to an enclosing cluster ball
    minus ``self_radius``. By the Euclidean triangle inequality, every
    resulting detector excludes all self balls. Coarse balls deliberately
    protect additional space; finer levels recover some of that space.
    Detector slots and candidate draws are divided across remaining levels,
    rounding down and carrying unused capacity to finer levels. A level with
    no detector slot consumes no draws. Thus even a single remaining slot
    or draw is reserved for the finest level instead of spent at a coarse
    level. Diagnostics include levels skipped by this allocation.
    """
    if (isinstance(n_levels, bool)
            or not isinstance(n_levels, (int, np.integer)) or n_levels < 1):
        raise ValueError('n_levels must be a positive integer')
    levels = _build_hierarchy(self_samples, self_radius, n_levels)
    centers, radii = [], []
    n_candidates = 0
    diagnostics = dict(cluster_counts=[], level_detector_counts=[],
                       level_candidate_counts=[], clusters=[],
                       sampling_boxes=[])
    for level_index, clusters in enumerate(levels):
        if n_candidates == max_candidates or len(centers) == n_detectors:
            break
        cluster_centers = np.asarray([item[0] for item in clusters])
        cluster_radii = np.asarray([item[1] for item in clusters])
        if level_index == 0:
            boxes = bounds[None, :, :]
        else:
            boxes = []
            for center, radius, _ in levels[level_index - 1]:
                outer_radius = radius + self_radius
                lower = np.maximum(bounds[0], center - outer_radius)
                upper = np.minimum(bounds[1], center + outer_radius)
                boxes.append(np.vstack([lower, upper]))
            boxes = np.asarray(boxes)
        levels_left = len(levels) - level_index
        detector_quota = (n_detectors - len(centers)) // levels_left
        candidate_budget = (max_candidates - n_candidates) // levels_left
        if detector_quota == 0:
            candidate_budget = 0
        level_candidates = 0
        level_detectors = 0
        for _ in range(candidate_budget):
            box = boxes[rng.randint(len(boxes))]
            candidate = rng.uniform(box[0], box[1])
            n_candidates += 1
            level_candidates += 1
            distance = cdist(candidate[None, :], cluster_centers)[0]
            radius = np.min(distance - cluster_radii) - self_radius
            if not np.isfinite(radius):
                raise ValueError('Candidate distances overflowed; '
                                 'reduce sampling_margin.')
            if radius <= 0:
                continue
            if centers:
                distances = cdist(candidate[None, :], np.asarray(centers))[0]
                if np.any(distances <= np.asarray(radii)):
                    continue
            centers.append(candidate)
            radii.append(radius)
            level_detectors += 1
            if level_detectors == detector_quota:
                break
        diagnostics['cluster_counts'].append(len(clusters))
        diagnostics['level_detector_counts'].append(level_detectors)
        diagnostics['level_candidate_counts'].append(level_candidates)
        diagnostics['clusters'].append((cluster_centers, cluster_radii))
        diagnostics['sampling_boxes'].append(boxes)
    shape = (len(centers), self_samples.shape[1])
    return (np.asarray(centers).reshape(shape), np.asarray(radii),
            n_candidates, diagnostics)


def _fit_apc_balls(samples, self_radius, rng, dual_clusters):
    """Adjust K-means cluster counts using the PRR paper's radius interval.

    Table I of Zheng et al. (2013), doi:10.4304/jcp.8.8.1951-1959, increases
    cluster count above radius 5*self_radius and decreases it below 2 times
    that radius. Mixed violations and impossible intervals are not resolved
    there. We prioritize oversized clusters and visit at most 32 distinct
    counts, retaining the fit with the fewest interval violations (then the
    smallest maximum and total violation). A repeated count stops the search.
    """
    n_unique = len(np.unique(samples, axis=0))
    count = min(dual_clusters, n_unique)
    visited, best = [], None
    reason = 'clustering_limit'
    for _ in range(min(n_unique, 32)):
        if count in visited:
            reason = 'repeated_cluster_count'
            break
        visited.append(count)
        model = KMeans(n_clusters=count, n_init=10, random_state=rng).fit(
            samples)
        # Near-distinct points can collapse numerically in K-means even
        # after exact unique-count clipping. Only occupied labels are balls.
        occupied = np.unique(model.labels_)
        cluster_centers = model.cluster_centers_[occupied]
        raw_radii = np.asarray([
            cdist(center[None, :], samples[model.labels_ == index]).max()
            for index, center in zip(occupied, cluster_centers)])
        if not np.all(np.isfinite(raw_radii)):
            raise ValueError('APC distances overflowed; rescale X.')
        violation = (np.maximum(2 * self_radius - raw_radii, 0)
                     + np.maximum(raw_radii - 5 * self_radius, 0))
        quality = (np.count_nonzero(violation), violation.max(),
                   violation.sum())
        if best is None or quality < best[0]:
            best = (quality, cluster_centers.copy(), raw_radii.copy())
        if not np.any(violation):
            reason = 'radius_interval'
            break
        next_count = count + (1 if np.any(raw_radii > 5 * self_radius)
                              else -1)
        if not 1 <= next_count <= n_unique:
            reason = 'cluster_count_boundary'
            break
        count = next_count
    _, cluster_centers, raw_radii = best
    # Unlike the paper's center-only APC cover, protect complete self balls.
    apc_radii = np.nextafter(raw_radii + self_radius, np.inf)
    diagnostic = {
        'apc_centers': cluster_centers,
        'apc_radii': apc_radii,
        'apc_cluster_radii': raw_radii,
        'apc_cluster_counts': visited,
        'apc_radius_criterion_met': bool(best[0][0] == 0),
        'apc_termination_reason': reason,
    }
    return cluster_centers, apc_radii, diagnostic


def generate_dual_detectors(self_samples, bounds, rng, n_detectors,
                            self_radius, max_candidates, dual_clusters=3):
    """Generate an APC self cover and a censored negative T-cell population.

    This is a bounded Euclidean adaptation of the three-stage mechanism in
    Zheng, Zhou and Fang (2013), doi:10.4304/jcp.8.8.1951-1959, Tables I-II
    and Section IV.B. The paper uses expected coverage for stopping; this
    helper instead uses explicit candidate and detector limits. Its APC
    radius adjustment has bounded, documented infeasibility handling.

    APC balls enclose K-means clusters and are expanded by self_radius to
    protect full self balls. Each proposal chooses an APC uniformly and then
    a point uniformly within that ball; points outside bounds are rejected.
    This mixture is not uniform over the union of overlapping APC balls.
    Candidate centers matching an existing negative detector are censored
    before evaluating distance to self. Retained negative radii equal the
    nearest-self distance minus self_radius.

    Return shape follows generate_hierarchical_detectors. The parent scores
    the union of negative balls and the complement of the APC union using
    max(max_j(r_j-dist(x,d_j)), min_k(dist(x,a_k)-apc_radius_k)).
    Unlike a negative-only finite population, the APC complement recognizes
    observations arbitrarily far outside the learned self envelopes.
    """
    if (isinstance(dual_clusters, bool)
            or not isinstance(dual_clusters, (int, np.integer))
            or dual_clusters < 1):
        raise ValueError('dual_clusters must be a positive integer')
    apc_centers, apc_radii, diagnostics = _fit_apc_balls(
        self_samples, self_radius, rng, dual_clusters)
    centers, radii = [], []
    rejected_bounds = rejected_self = rejected_covered = 0
    for candidate_index in range(max_candidates):
        index = rng.randint(len(apc_centers))
        direction = rng.normal(size=self_samples.shape[1])
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        magnitude = rng.uniform() ** (1.0 / self_samples.shape[1])
        candidate = (apc_centers[index]
                     + apc_radii[index] * magnitude * direction / norm)
        if np.any(candidate < bounds[0]) or np.any(candidate > bounds[1]):
            rejected_bounds += 1
            continue
        if centers and np.any(
                cdist(candidate[None, :], centers)[0] <= radii):
            rejected_covered += 1
            continue
        distance = cdist(candidate[None, :], self_samples).min()
        if not np.isfinite(distance):
            raise ValueError('Candidate distances overflowed; rescale X.')
        radius = distance - self_radius
        if radius <= 0:
            rejected_self += 1
            continue
        centers.append(candidate)
        radii.append(radius)
        if len(centers) == n_detectors:
            break
    diagnostics.update({
        'candidate_outside_bounds': rejected_bounds,
        'candidate_matching_self': rejected_self,
        'candidate_already_covered': rejected_covered,
        'termination_reason': ('detector_limit'
                               if len(centers) == n_detectors
                               else 'candidate_limit'),
    })
    return (np.asarray(centers).reshape(-1, self_samples.shape[1]),
            np.asarray(radii), candidate_index + 1, diagnostics)
