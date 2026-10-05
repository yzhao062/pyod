# -*- coding: utf-8 -*-
"""Geometric and immune-suppression detector construction for NSA.

The shared-vertex construction follows Algorithm 1 of Zhu et al. (2017),
"A Quick Negative Selection Algorithm for One-Class Classification in Big
Data Era", https://doi.org/10.1155/2017/3956415. This implementation permits
rectangular bounds instead of only the unit cube, caps the retained detector
count, and defaults the minimum detector radius to zero. It does not implement
the paper's Map/Reduce classifier or claim its construction complexity.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, QhullError
from scipy.spatial.distance import cdist


def generate_voronoi_detectors(self_samples, bounds, rng, n_detectors,
                               self_radius, max_candidates, min_radius=0.0):
    """Return detectors centered at shared bounded Voronoi vertices.

    Inputs are scaled real self samples and a (2, n_features) bounding box.
    Only two and three dimensions are supported. Exact duplicate sites are
    removed; collinear and coplanar sites need no random perturbation. ``rng``
    is accepted for the common generator interface but is unused.

    The return tuple is (centers, radii, candidate_count, diagnostics).
    ``candidate_count`` counts distinct cell vertices, including vertices
    belonging to only one cell. Exceeding ``max_candidates`` raises ValueError
    rather than returning an incomplete tessellation. Eligible shared vertices
    are sorted by decreasing radius, then lexicographic coordinates, before
    retaining at most ``n_detectors``. Diagnostics report the count before
    truncation. Empty valid geometry returns empty arrays for the caller to
    handle consistently with other NSA generators.

    Every cell is the intersection of nearest-site bisector halfspaces and
    the bounding box. A linear program finds an interior point for Qhull.
    Enumerating cells is more costly than specialized Voronoi algorithms;
    this routine is intended for modest training sets in low dimensions.
    """
    samples = np.asarray(self_samples, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if samples.ndim != 2 or samples.shape[1] not in (2, 3):
        raise ValueError("Voronoi generation requires 2 or 3 features.")
    dimension = samples.shape[1]
    if (bounds.shape != (2, dimension) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("Voronoi bounds must have positive finite widths.")
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError("Voronoi generation requires finite self samples.")
    if np.any(samples < bounds[0]) or np.any(samples > bounds[1]):
        raise ValueError("Voronoi bounds must contain every self sample.")
    if (not np.isfinite(self_radius) or self_radius < 0
            or not np.isfinite(min_radius) or min_radius < 0):
        raise ValueError("Voronoi radii must be finite and nonnegative.")
    if (isinstance(n_detectors, (bool, np.bool_))
            or not isinstance(n_detectors, (int, np.integer))
            or n_detectors < 1 or isinstance(max_candidates, (bool, np.bool_))
            or not isinstance(max_candidates, (int, np.integer))
            or max_candidates < 1):
        raise ValueError("Voronoi detector and candidate limits must be "
                         "positive.")

    sites = np.unique(samples, axis=0)
    box_normals = np.vstack((np.eye(dimension), -np.eye(dimension)))
    box_offsets = np.r_[-bounds[1], bounds[0]]
    # Shift coordinates to reduce cancellation in squared bisector offsets.
    origin = bounds[0] + (bounds[1] - bounds[0]) / 2
    box_offsets = box_offsets + box_normals @ origin
    vertices = {}
    scale = max(1.0, float(np.max(bounds[1] - bounds[0])))
    tolerance = 1e-9 * scale

    for index, site in enumerate(sites):
        others = np.delete(sites, index, axis=0)
        # Subtract before translation: shifting two nearby sites can make
        # their floating-point representations identical. Hypot also avoids
        # underflow when squaring a representable but very small separation.
        normals = others - site
        lengths = np.hypot.reduce(normals, axis=1)
        # Unit normals keep the Chebyshev-center slack in distance units.
        normals = normals / lengths[:, None]
        midpoints = others / 2 + site / 2 - origin
        offsets = -np.sum(normals * midpoints, axis=1)
        normals = np.vstack((normals, box_normals))
        offsets = np.r_[offsets, box_offsets]
        interior = linprog(
            np.r_[np.zeros(dimension), -1.0],
            A_ub=np.column_stack((normals, np.ones(len(normals)))),
            b_ub=-offsets, bounds=[(None, None)] * dimension + [(0, None)],
            method='highs')
        if not interior.success or interior.x[-1] <= 0:
            raise ValueError("Could not construct a bounded Voronoi cell.")
        try:
            cell = HalfspaceIntersection(
                np.column_stack((normals, offsets)),
                interior.x[:dimension]).intersections + origin
        except QhullError as exc:
            raise ValueError(
                "Voronoi cell is numerically degenerate; rescale the data."
            ) from exc
        for vertex in cell:
            if not np.isfinite(vertex).all():
                raise ValueError("Voronoi construction produced "
                                 "nonfinite data.")
            vertex = np.clip(vertex, bounds[0], bounds[1])
            # A common vertex is independently recovered from adjoining cells.
            # Round only the deduplication key, not the retained coordinates.
            key = tuple(np.round((vertex - bounds[0]) / scale, decimals=10))
            if key not in vertices:
                if len(vertices) >= max_candidates:
                    raise ValueError(
                        "Voronoi vertices exceed max_candidates; increase "
                        "the budget or use fewer self samples.")
                vertices[key] = vertex

    candidates = np.asarray(list(vertices.values()))
    centers, radii, types = [], [], []
    n_shared = 0
    for vertex in candidates:
        distances = cdist(vertex[None, :], sites)[0]
        nearest = float(np.min(distances))
        matches = int(np.sum(np.abs(distances - nearest) <= tolerance))
        if matches < 2:
            continue
        n_shared += 1
        radius = nearest - self_radius
        if radius > min_radius:
            centers.append(vertex)
            radii.append(radius)
            boundary = np.any(vertex <= bounds[0] + tolerance) or np.any(
                vertex >= bounds[1] - tolerance)
            types.append(2 if boundary else 1)

    centers = np.asarray(centers).reshape(-1, dimension)
    radii = np.asarray(radii)
    types = np.asarray(types, dtype=int)
    # np.lexsort takes the primary key last. Coordinates resolve radius ties.
    keys = tuple(centers[:, i] for i in reversed(range(dimension)))
    order = np.lexsort(keys + (-radii,))[:n_detectors]
    diagnostics = {
        'n_unique_sites': len(sites),
        'n_vertices': len(candidates),
        'n_shared_vertices': n_shared,
        'n_eligible_detectors': len(radii),
        'truncated': len(radii) > n_detectors,
        'n_type_one': int(np.sum(types[order] == 1)),
        'n_type_two': int(np.sum(types[order] == 2)),
    }
    return centers[order], radii[order], len(candidates), diagnostics


def generate_suppressed_detectors(self_samples, bounds, rng, n_detectors,
                                  self_radius, max_candidates,
                                  outlier_fraction=0.95, outlier_radius=0.5):
    """Construct negative and reverse detectors from an assumed-clean self set.

    Implements the outlier partition and immune-suppression mechanism in
    Li et al. (2010), "An Outlier Robust Negative Selection Algorithm Inspired
    by Immune Suppression", https://doi.org/10.4304/jcp.5.9.1348-1355,
    Section III. A self sample is a DB(p, D) outlier when at least fraction p
    of all self samples are farther than D from it. Here D is
    ``outlier_radius * sqrt(n_features)`` in scaled feature space. Original
    rows, including duplicates and the queried row, participate in this
    fraction. Unlike the paper's noisy-self alternative, every such outlier
    remains protected as a reverse detector (the clean-self alternative).

    Negative spheres are tangent to their nearest remaining self sample.
    That nearest self becomes a boundary reverse detector. An uncovered
    candidate, or one identifying a new boundary self, is retained. Reverse
    spheres have radius ``self_radius``. This replaces the paper's statistical
    coverage stopping with the supplied detector/draw caps, and uses the same
    self radius to reject candidate centers too close to remaining selves.

    Return (negative_centers, negative_radii, draws, diagnostics). The caller
    must score x as min(max_j(radius_j - distance(x, center_j)),
    min_k(distance(x, reverse_center_k)) - reverse_radius). Positive scores
    then require both negative recognition and absence of reverse recognition.
    All original training points have scores <= 0, even if an outlier lies
    inside a negative sphere. Negative spheres alone are not self-safe.
    """
    samples = np.asarray(self_samples, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if (samples.ndim != 2 or samples.shape[0] == 0
            or samples.shape[1] == 0 or not np.isfinite(samples).all()):
        raise ValueError("Suppression requires a finite nonempty self set.")
    dimension = samples.shape[1]
    if (bounds.shape != (2, dimension) or not np.isfinite(bounds).all()
            or np.any(bounds[1] < bounds[0])):
        raise ValueError("Suppression requires valid finite sampling bounds.")
    if (not np.isfinite(outlier_fraction)
            or not 0 < outlier_fraction <= 1
            or not np.isfinite(outlier_radius) or outlier_radius <= 0):
        raise ValueError("Outlier fraction must be in (0, 1] and outlier "
                         "radius must be positive and finite.")
    distance_threshold = outlier_radius * np.sqrt(dimension)
    outliers = np.empty(len(samples), dtype=bool)
    # Bound temporary pairwise storage while retaining exact DB(p, D) counts.
    batch_size = max(1, 1000000 // len(samples))
    for start in range(0, len(samples), batch_size):
        distances = cdist(samples[start:start + batch_size], samples)
        if not np.isfinite(distances).all():
            raise ValueError("Suppression distances must be finite.")
        outliers[start:start + batch_size] = (
            np.mean(distances > distance_threshold, axis=1)
            >= outlier_fraction)
    remaining = samples[~outliers]
    if len(remaining) == 0:
        raise ValueError("The distance rule marked every self sample as an "
                         "outlier; increase outlier_radius or "
                         "outlier_fraction.")

    centers, radii = [], []
    boundary = np.zeros(len(remaining), dtype=bool)
    draws = 0
    while len(centers) < n_detectors and draws < max_candidates:
        candidate = rng.uniform(bounds[0], bounds[1])
        draws += 1
        distances = cdist(candidate[None, :], remaining)[0]
        if not np.isfinite(distances).all():
            raise ValueError("Suppression distances must be finite.")
        nearest = int(np.argmin(distances))
        radius = float(distances[nearest])
        if radius <= self_radius:
            continue
        if centers:
            detector_distances = cdist(candidate[None, :], centers)[0]
            if np.any(detector_distances == 0):
                continue
            if (boundary[nearest]
                    and np.any(detector_distances < np.asarray(radii))):
                continue
        centers.append(candidate)
        radii.append(radius)
        boundary[nearest] = True

    reverse_centers = np.unique(
        np.vstack((samples[outliers], remaining[boundary])), axis=0)
    diagnostics = {
        'reverse_centers': reverse_centers,
        'reverse_radius': float(self_radius),
        'outlier_mask': outliers,
        'n_outlier_self': int(np.sum(outliers)),
        'n_boundary_self': int(np.sum(boundary)),
        'outlier_distance_threshold': float(distance_threshold),
    }
    return (np.asarray(centers).reshape(-1, dimension), np.asarray(radii),
            draws, diagnostics)


def _gaussian_affinity(distances, radius):
    """Evaluate exp(-distance**2 / radius**2) without ratio overflow."""
    if radius == 0:
        return np.asarray(distances == 0, dtype=float)
    with np.errstate(over='ignore', under='ignore'):
        ratios = distances / radius
        return np.exp(-ratios * ratios)


def _annealing_local_cost(center, others, samples, radius, self_radius):
    """Energy terms affected by replacing one center, including both orders."""
    detector_distances = cdist(center[None, :], others)[0]
    self_distances = cdist(center[None, :], samples)[0]
    if (not np.isfinite(detector_distances).all()
            or not np.isfinite(self_distances).all()):
        raise ValueError("Annealing distances must be finite.")
    # Eq. (9) sums ordered detector pairs, while Eq. (11) counts each self
    # and detector pair once. The penalty weight beta is fixed at one.
    return float(2 * _gaussian_affinity(detector_distances, radius).sum()
                 + _gaussian_affinity(
                     self_distances, radius / 2 + self_radius / 2).sum())


def _annealing_energy(centers, samples, radius, self_radius):
    """RRNS Gaussian overlap plus self-covering penalty, equations (9)-(11)."""
    value = 0.0
    for index, center in enumerate(centers):
        value += _annealing_local_cost(
            center, centers[index + 1:], samples, radius, self_radius)
    return value


def generate_annealed_detectors(self_samples, bounds, rng, n_detectors,
                                self_radius, max_candidates,
                                detector_radius=0.1, annealing_steps=20):
    """Estimate a fixed-radius population and optimize it with annealing.

    Based on Sections 2.1-2.2 and Figures 2-3 of Gonzalez, Dasgupta and Nino
    (2003), "A Randomized Real-Valued Negative Selection Algorithm",
    https://doi.org/10.1007/978-3-540-45192-1_25. The Gaussian energy is the
    paper's ordered-pair overlap plus self-covering penalty, with beta=1.

    Monte Carlo sampling estimates nonself volume in the supplied rectangle.
    The target is floor(volume / (2*detector_radius/sqrt(d))**d), capped by
    ``n_detectors``. This is a population heuristic, not certified coverage.
    The volume probe count is min(256, max(1, max_candidates//4)); this fixed
    budget replaces the paper's adaptive error stopping. All probes, initial
    candidate draws and perturbation attempts consume ``max_candidates``.

    Each annealing step accepts up to one move per retained detector or makes
    twice that many attempts. Proposals are uniform in the ball of initial
    radius 2*detector_radius, restricted to the bounding box. Temperature
    starts at 1, and temperature and perturbation radius decay by 0.9 per
    step. These fixed settings replace the unspecified initial-temperature
    subroutine and supplied schedule parameters of Figure 3. A hard
    self-sphere exclusion test at initialization and every move supplements
    its soft penalty. The best visited population is returned. No reheating
    or convergence guarantee is claimed.
    """
    samples = np.asarray(self_samples, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if (samples.ndim != 2 or samples.shape[0] == 0
            or samples.shape[1] == 0 or not np.isfinite(samples).all()):
        raise ValueError("Annealing requires a finite nonempty self set.")
    dimension = samples.shape[1]
    if (bounds.shape != (2, dimension) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("Annealing bounds must have positive finite widths.")
    if (not np.isfinite(detector_radius) or detector_radius <= 0
            or not np.isfinite(self_radius) or self_radius < 0):
        raise ValueError("Annealing requires a positive detector radius and "
                         "a nonnegative self radius, both finite.")
    for value in (n_detectors, max_candidates, annealing_steps):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError("Annealing counts must be positive integers.")
    with np.errstate(over='ignore'):
        widths = bounds[1] - bounds[0]
    if (not np.isfinite(widths).all()
            or detector_radius > np.finfo(float).max / 2
            or self_radius > np.finfo(float).max - detector_radius):
        raise ValueError("Annealing geometry overflows; reduce bounds/radii.")
    log_volume = float(np.log(widths).sum())
    if log_volume > np.log(np.finfo(float).max):
        raise ValueError("Annealing sampling volume overflows; reduce bounds.")
    n_probes = min(256, max(1, max_candidates // 4))
    self_hits = 0
    for _ in range(n_probes):
        probe = rng.uniform(bounds[0], bounds[1])
        distances = cdist(probe[None, :], samples)[0]
        if not np.isfinite(distances).all():
            raise ValueError("Annealing distances must be finite.")
        self_hits += np.min(distances) <= self_radius
    nonself_fraction = 1 - self_hits / n_probes
    target = 0
    target_capped = False
    if nonself_fraction > 0:
        log_count = (log_volume + np.log(nonself_fraction)
                     - dimension * (np.log(detector_radius) + np.log(2)
                                    - np.log(dimension) / 2))
        target_capped = log_count >= np.log(n_detectors)
        # The exp/log round trip can put an exact integer one ULP below
        # itself (for example, a volume estimate of three detector cells).
        target = (n_detectors if target_capped else int(np.floor(
            np.nextafter(np.exp(log_count), np.inf))))

    centers = []
    draws = n_probes
    separation = self_radius + detector_radius
    while len(centers) < target and draws < max_candidates:
        candidate = rng.uniform(bounds[0], bounds[1])
        draws += 1
        distances = cdist(candidate[None, :], samples)[0]
        if not np.isfinite(distances).all():
            raise ValueError("Annealing distances must be finite.")
        if np.min(distances) <= separation:
            continue
        if centers and np.any(cdist(candidate[None, :], centers) == 0):
            continue
        centers.append(candidate)
    centers = np.asarray(centers).reshape(-1, dimension)
    initial_energy = _annealing_energy(
        centers, samples, detector_radius, self_radius)
    current_energy = initial_energy
    best_energy = initial_energy
    best_centers = centers.copy()
    accepted_moves = 0
    attempts = 0
    completed_steps = 0
    temperature = 1.0
    perturbation_radius = 2 * detector_radius
    for _ in range(annealing_steps):
        if draws >= max_candidates or len(centers) == 0:
            break
        accepted_this_step = 0
        for _ in range(2 * len(centers)):
            if (draws >= max_candidates
                    or accepted_this_step >= len(centers)):
                break
            index = rng.randint(len(centers))
            direction = rng.normal(size=dimension)
            norm = float(np.hypot.reduce(direction))
            draws += 1
            attempts += 1
            if norm == 0:
                continue
            displacement = (direction / norm * perturbation_radius
                            * rng.uniform() ** (1 / dimension))
            with np.errstate(over='ignore'):
                candidate = centers[index] + displacement
            if not np.isfinite(candidate).all():
                raise ValueError("Annealing candidate geometry overflows.")
            if (np.any(candidate < bounds[0])
                    or np.any(candidate > bounds[1])):
                continue
            self_distances = cdist(candidate[None, :], samples)[0]
            if not np.isfinite(self_distances).all():
                raise ValueError("Annealing distances must be finite.")
            if np.min(self_distances) <= separation:
                continue
            others = np.delete(centers, index, axis=0)
            if np.any(cdist(candidate[None, :], others) == 0):
                continue
            delta = (_annealing_local_cost(
                candidate, others, samples, detector_radius, self_radius)
                - _annealing_local_cost(
                    centers[index], others, samples, detector_radius,
                    self_radius))
            with np.errstate(over='ignore', under='ignore'):
                acceptance = np.exp(-max(0.0, delta) / max(
                    temperature, np.finfo(float).tiny))
            if delta <= 0 or rng.uniform() < acceptance:
                centers[index] = candidate
                current_energy += delta
                accepted_moves += 1
                accepted_this_step += 1
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_centers = centers.copy()
        completed_steps += 1
        temperature *= 0.9
        perturbation_radius *= 0.9
    # Recompute the reported best energy to avoid accumulated delta roundoff.
    best_energy = _annealing_energy(
        best_centers, samples, detector_radius, self_radius)
    diagnostics = {
        'estimated_target_count': target,
        'target_detectors': target,
        'estimated_target_capped': bool(target_capped),
        'volume_probe_count': n_probes,
        'nonself_fraction_estimate': float(nonself_fraction),
        'nonself_volume_estimate': float(
            np.exp(log_volume) * nonself_fraction),
        'initial_energy': float(initial_energy),
        'best_energy': float(best_energy),
        'accepted_moves': accepted_moves,
        'annealing_attempts': attempts,
        'optimization_proposals': attempts,
        'annealing_steps_completed': completed_steps,
        'target_population_reached': len(best_centers) == target,
    }
    radii = np.full(len(best_centers), detector_radius)
    return best_centers, radii, draws, diagnostics
