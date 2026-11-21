# -*- coding: utf-8 -*-
"""Grid-indexed negative detector construction and coverage diagnostics.

The construction adapts the grid indexing and radius-ordered filtering of
Zhang, Li and Xiao (2013), DOI: 10.1155/2013/268639. It is not a complete
reproduction of GB-RNSA. Occupied orthants are represented sparsely, nearest
self queries are exact, and removal requires exact sphere containment.
Candidate-count stopping replaces the paper's sequential coverage test.

The two-dimensional deterministic construction adapts Barontini et al.
(2019), DOI: 10.1016/j.engstruct.2019.109444, Section 2.2 and Table 1.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import heapq
from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta


class _SelfGrid:
    """Sparse orthant tree with exact Euclidean nearest-self queries.

    Empty orthants need no storage because only self samples are queried.
    Unlike a search restricted to adjacent cells, bounding-box pruning cannot
    miss a closer self point in a nonadjacent occupied cell.
    """

    def __init__(self, samples, bounds, max_depth, min_side):
        self.samples = samples
        self.nodes = []
        self.distance_evaluations = 0
        self.pruned_nodes = 0
        self.n_leaves = 0
        self._build(np.arange(len(samples)), bounds[0], bounds[1],
                    max_depth, min_side)

    def _build(self, indices, lower, upper, depth, min_side):
        node_id = len(self.nodes)
        # Node: lower bound, upper bound, self indices, child node IDs.
        self.nodes.append((lower, upper, indices, []))
        widths = upper - lower
        if depth == 0 or np.max(widths) <= min_side:
            self.n_leaves += 1
            return node_id
        middle = lower + widths / 2
        active = (middle > lower) & (middle < upper)
        if not np.any(active):
            self.n_leaves += 1
            return node_id
        codes = self.samples[indices][:, active] >= middle[active]
        groups = {}
        for index, code in zip(indices, codes):
            groups.setdefault(tuple(code), []).append(index)
        children = []
        for code, group in groups.items():
            child_lower, child_upper = lower.copy(), upper.copy()
            right = np.zeros(len(lower), dtype=bool)
            right[active] = code
            child_lower[right] = middle[right]
            child_upper[active & ~right] = middle[active & ~right]
            children.append(self._build(
                np.asarray(group), child_lower, child_upper,
                depth - 1, min_side))
        self.nodes[node_id] = (lower, upper, None, children)
        return node_id

    @staticmethod
    def _lower_distance(point, lower, upper):
        delta = np.maximum(np.maximum(lower - point, point - upper), 0)
        return np.linalg.norm(delta)

    def nearest_distance(self, point):
        best = np.inf
        queue = [(0., 0)]
        while queue:
            lower_distance, node_id = heapq.heappop(queue)
            if lower_distance >= best:
                self.pruned_nodes += 1 + len(queue)
                break
            lower, upper, indices, children = self.nodes[node_id]
            if indices is not None:
                distances = cdist(point[None, :], self.samples[indices])[0]
                self.distance_evaluations += len(indices)
                best = min(best, np.min(distances))
            else:
                for child_id in children:
                    child = self.nodes[child_id]
                    bound = self._lower_distance(point, child[0], child[1])
                    if bound < best:
                        heapq.heappush(queue, (bound, child_id))
                    else:
                        self.pruned_nodes += 1
        return float(best)


def _filter_candidates(candidates, centers, radii, limit):
    """Retain largest candidates first, with coverage-preserving removal."""
    suppressed, removed = 0, 0
    for center, radius in sorted(candidates, key=lambda item: -item[1]):
        if centers:
            distances = cdist(center[None, :], np.asarray(centers))[0]
            mature_radii = np.asarray(radii)
            if np.any(distances <= mature_radii):
                suppressed += 1
                continue
            # A mature ball is redundant only if its entire ball is inside
            # the new one. Center containment alone would lose coverage.
            contained = distances + mature_radii <= radius
            removed += int(np.sum(contained))
            centers[:] = [c for c, drop in zip(centers, contained) if not drop]
            radii[:] = [r for r, drop in zip(radii, contained) if not drop]
        centers.append(center)
        radii.append(radius)
        if len(centers) >= limit:
            break
    return suppressed, removed


def generate_grid_detectors(self_samples, bounds, rng, n_detectors,
                            self_radius, max_candidates, grid_depth=4):
    """Construct variable-radius detectors using grid indexing and filtering.

    Parameters are in the already scaled feature space. ``bounds`` has shape
    (2, n_features); ``rng`` is a NumPy RandomState. The caller validates the
    common NSA parameters. ``grid_depth`` limits recursive occupied-cell
    subdivision, preventing the exponential allocation of empty cells.

    Each candidate radius is its exact nearest-self distance minus
    ``self_radius``. At most 64 draws form a buffer, filtered in descending
    radius order; centers already covered by a retained detector are rejected.
    A larger candidate may replace entirely contained mature balls. This is
    a geometric heuristic, not a guarantee of full negative-region coverage.

    Returns centers, radii, the number of draws, and construction diagnostics.
    Empty output arrays are possible when no candidate survives censoring.
    """
    if (isinstance(grid_depth, bool) or not isinstance(grid_depth, Integral)
            or not 1 <= grid_depth <= 20):
        raise ValueError('grid_depth must be an integer in [1, 20].')
    grid = _SelfGrid(self_samples, bounds, grid_depth, 4 * self_radius)
    centers, radii, candidates = [], [], []
    suppressed = removed = n_candidates = 0
    for n_candidates in range(1, max_candidates + 1):
        center = rng.uniform(bounds[0], bounds[1])
        radius = grid.nearest_distance(center) - self_radius
        if not np.isfinite(radius):
            raise ValueError('Grid distances overflowed; rescale the data.')
        if radius > 0:
            candidates.append((center, radius))
        if n_candidates % 64 == 0 or n_candidates == max_candidates:
            dropped, replaced = _filter_candidates(
                candidates, centers, radii, n_detectors)
            suppressed += dropped
            removed += replaced
            candidates = []
            if len(centers) >= n_detectors:
                break
    diagnostics = {
        'grid_nodes': len(grid.nodes),
        'grid_leaves': grid.n_leaves,
        'grid_pruned_nodes': grid.pruned_nodes,
        'self_distance_evaluations': grid.distance_evaluations,
        'suppressed_centers': suppressed,
        'removed_contained_detectors': removed,
    }
    return (np.asarray(centers).reshape(-1, self_samples.shape[1]),
            np.asarray(radii), n_candidates, diagnostics)


def estimate_negative_coverage(centers, radii, self_samples, bounds,
                               self_radius, rng, n_samples=2048,
                               confidence=0.95):
    """Estimate conditional nonself coverage on one independent holdout.

    Draw a fixed-size uniform sample from ``bounds``, discard points inside
    self balls, and count points strictly inside the union of detector balls.
    The reported lower bound is the one-sided exact Clopper-Pearson binomial
    bound, conditional on the number of retained nonself samples. Its
    interpretation requires a detector population fixed independently of
    these draws. Do not tune or stop generation using repeated calls and
    continue to claim this nominal confidence without multiplicity control.

    This measures coverage under the bounded uniform reference distribution,
    not recall for an unknown anomaly distribution. If no nonself point is
    drawn, the estimate is NaN and the uninformative lower bound is zero.
    """
    if (isinstance(n_samples, bool) or not isinstance(n_samples, Integral)
            or n_samples < 1):
        raise ValueError('n_samples must be a positive integer.')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1).')
    n_nonself = n_covered = 0
    # Bound temporary distance matrices for large training populations.
    for start in range(0, n_samples, 64):
        size = (min(64, n_samples - start), bounds.shape[1])
        points = rng.uniform(bounds[0], bounds[1], size=size)
        negative = np.min(cdist(points, self_samples), axis=1) > self_radius
        points = points[negative]
        n_nonself += len(points)
        if len(points) and len(centers):
            n_covered += int(np.count_nonzero(np.any(
                cdist(points, centers) < radii[None, :], axis=1)))
    estimate = n_covered / n_nonself if n_nonself else np.nan
    lower = (float(beta.ppf(1 - confidence, n_covered,
                            n_nonself - n_covered + 1))
             if n_covered else 0.)
    return {'n_samples': n_samples, 'n_nonself': n_nonself,
            'n_covered': n_covered, 'estimate': estimate,
            'lower_bound': lower, 'confidence': confidence}


def generate_deterministic_detectors(self_samples, bounds, rng, n_detectors,
                                     self_radius, max_candidates,
                                     grid_depth=4):
    """Generate a two-dimensional lattice and repel censored boundary sites.

    The initial lattice has ``2 ** grid_depth`` divisions per axis. Equal
    circular detectors circumscribe each cell. For each first-coordinate
    column, only the second-coordinate extremes among censored sites are
    eligible for movement. Their steps decay exponentially, as in the
    primary paper. The method is intentionally restricted to two dimensions.

    Adaptations: rectangular bounded domains, a strict self-ball clearance,
    clipped movements, deterministic tie handling, and rejection after 30
    unsuccessful moves. A population above ``n_detectors`` is thinned using
    evenly spaced indices in lexicographic order. Censoring and thinning
    remove the initial lattice's coverage guarantee. ``rng`` is unused.

    ``max_candidates`` bounds the number of initial lattice sites; movement
    evaluations have a separate fixed bound of 30 per boundary site. A
    lattice exceeding the budget raises before allocating it.
    """
    if self_samples.shape[1] != 2:
        raise ValueError('The deterministic strategy requires two features.')
    if (isinstance(grid_depth, bool) or not isinstance(grid_depth, Integral)
            or not 1 <= grid_depth <= 20):
        raise ValueError('grid_depth must be an integer in [1, 20].')
    divisions = 2 ** grid_depth
    n_candidates = divisions ** 2
    if n_candidates > max_candidates:
        raise ValueError(
            'The deterministic lattice exceeds max_candidates; reduce '
            'grid_depth or increase max_candidates.')
    widths = bounds[1] - bounds[0]
    if np.any(widths <= 0):
        raise ValueError(
            'The deterministic strategy requires positive sampling widths; '
            'increase sampling_margin for constant features.')
    cell_widths = widths / divisions
    radius = np.linalg.norm(cell_widths) / 2
    clearance = radius + self_radius
    if not np.isfinite(clearance):
        raise ValueError('Deterministic radii overflowed; rescale the data.')
    axes = [bounds[0, j] + (np.arange(divisions) + 0.5) * cell_widths[j]
            for j in range(2)]
    xx, yy = np.meshgrid(*axes, indexing='ij')
    lattice = np.column_stack((xx.ravel(), yy.ravel()))
    safe = np.empty(n_candidates, dtype=bool)
    for start in range(0, n_candidates, 64):
        batch = lattice[start:start + 64]
        distances = cdist(batch, self_samples)
        if not np.all(np.isfinite(distances)):
            raise ValueError(
                'Deterministic distances overflowed; rescale the data.')
        safe[start:start + 64] = np.min(distances, axis=1) >= clearance
    centers = [center.copy() for center in lattice[safe]]
    boundary = []
    for column in range(divisions):
        indices = np.arange(column * divisions, (column + 1) * divisions)
        censored = indices[~safe[indices]]
        if len(censored):
            boundary.append((censored[0], -1.))
            if len(censored) > 1:
                boundary.append((censored[-1], 1.))
    movements = failed = 0
    initial_step = 0.005 * np.linalg.norm(widths) / np.sqrt(2)
    for index, direction_sign in boundary:
        center = lattice[index].copy()
        accepted = False
        for age in range(31):
            distances = cdist(center[None, :], self_samples)[0]
            if not np.all(np.isfinite(distances)):
                raise ValueError(
                    'Deterministic distances overflowed; rescale the data.')
            nearest = int(np.argmin(distances))
            distance = distances[nearest]
            if distance >= clearance:
                accepted = True
                break
            if age == 30:
                break
            if distance == 0:
                # The paper's normalized direction is undefined at a self
                # center; use the boundary's outward vertical direction.
                direction = np.array([0., direction_sign])
            else:
                direction = (center - self_samples[nearest]) / distance
            step = initial_step * np.exp(-(age + 1) / 30)
            center = np.clip(center + step * direction, bounds[0], bounds[1])
            movements += 1
        if accepted:
            centers.append(center)
        else:
            failed += 1
    centers = np.asarray(centers).reshape(-1, 2)
    # Lexicographic order makes thinning independent of acceptance ordering;
    # duplicate sites can occur when movements reach a common box boundary.
    centers = np.unique(centers, axis=0)
    before_thinning = len(centers)
    if len(centers) > n_detectors:
        keep = np.linspace(0, len(centers) - 1, n_detectors, dtype=int)
        centers = centers[keep]
    radii = np.full(len(centers), radius)
    diagnostics = {
        'lattice_divisions': divisions,
        'initial_safe_detectors': int(np.sum(safe)),
        'boundary_candidates': len(boundary),
        'boundary_movements': movements,
        'failed_boundary_candidates': failed,
        'thinned_detectors': before_thinning - len(centers),
        'truncated': before_thinning > n_detectors,
    }
    return centers, radii, n_candidates, diagnostics
