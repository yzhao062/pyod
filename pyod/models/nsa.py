# -*- coding: utf-8 -*-
"""Negative-selection-inspired novelty detection.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


class NSA(BaseDetector):
    """Negative-selection-inspired detector with distinct generation methods.

    Negative selection retains candidate detectors when they do not match
    the training (self) samples. This estimator implements binary matching,
    fixed-radius hyperspheres, and variable-radius hyperspheres. The sources
    of these ideas are :cite:`forrest1994self,gonzalez2003anomaly,ji2004real`.
    These are NSA-inspired PyOD adaptations, not reproductions of every step
    of those papers. The three basic strategies use bounded random sampling
    and signed matching margins. In particular, ``fixed`` omits the original
    adaptive population and auxiliary classification stage.
    Additional real strategies use coverage testing, sparse grids,
    hierarchical self clusters, bounded Voronoi geometry, deterministic
    lattices, reverse-detector suppression, dual self envelopes, and
    annealing. Their sources are
    :cite:`ji2009vdetector,zhang2013grid,chen2011hierarchical`,
    :cite:`zhu2017quick,barontini2019deterministic,li2010suppression`, and
    :cite:`zheng2013dual,gonzalez2003randomized`.
    Their precise mechanisms and adaptations are described in the NSA
    example guide; paper acronyms are not exported as estimator aliases.

    Fit on representative normal samples for novelty detection. Every row
    passed to :meth:`fit` is treated as self, including any outliers. The
    ``contamination`` parameter calibrates PyOD's score threshold; it does
    not identify or remove contaminated training samples. The fitted score
    threshold is distinct from the geometric matching boundary at zero.

    Parameters
    ----------
    contamination : float in (0., 0.5], optional (default=0.1)
        Proportion used by PyOD to calibrate ``threshold_`` from training
        scores. Tied scores can yield fewer outlier labels than this fraction.

    strategy : str, optional (default='variable')
        Detector representation and generation strategy. ``fixed`` retains
        hyperspheres with ``detector_radius``; ``variable`` sets each radius
        to its center's nearest-self distance minus ``self_radius``.
        ``binary`` uses one median-threshold bit for every input feature.
        ``coverage`` adds exact statistical stopping to variable spheres.
        ``grid`` uses sparse spatial indexing and radius-ordered filtering.
        ``hierarchical`` samples around successively refined self clusters.
        ``voronoi`` generates centers from bounded Voronoi geometry in two
        or three dimensions. ``deterministic`` uses a regular lattice and
        boundary repulsion in two dimensions. These names describe
        implemented mechanisms.
        ``suppressed`` combines negative spheres with reverse self detectors.
        ``dual`` combines an enclosing cluster-ball population with negative
        detectors sampled inside that population's region.
        ``annealed`` estimates population size by Monte Carlo sampling and
        optimizes fixed-radius detector positions by simulated annealing.

    n_detectors : int, optional (default=100)
        Maximum number of distinct detectors to retain. Fewer may be found
        within ``max_candidates``; in that case fitting emits a UserWarning,
        unless a coverage target is certified or finite geometry is exhausted.

    self_radius : float, optional (default=0.1)
        Nonnegative radius around each self sample in scaled real space.
        Used by all real strategies.

    detector_radius : float, optional (default=0.1)
        Positive radius in scaled space for ``fixed`` and ``annealed``.
        Candidate spheres must be separated from all self spheres.

    max_candidates : int, optional (default=10000)
        Positive limit on random candidate draws, including rejected and
        duplicate candidates. If none is valid, fitting raises ValueError.
        For ``voronoi`` this limits distinct geometric vertices, with an
        error on incomplete enumeration. For ``deterministic`` it bounds
        initial lattice sites; at most 30 movements per boundary site are
        allowed separately. See ``generation_diagnostics_``.

    sampling_margin : float, optional (default=0.5)
        Nonnegative expansion of each scaled training feature's bounds for
        real candidate generation. Nonconstant features have training bounds
        [0, 1]; constant features have bounds [0, 0]. Each side is expanded
        by this value. This parameter has no effect on binary generation.

    binary_match : {'hamming', 'rcontiguous', 'rchunk'}, optional
        Binary matching rule (default='hamming'). ``hamming`` matches at most
        ``match_threshold`` differing bits. ``rcontiguous`` matches at least
        that many consecutive equal bits at corresponding positions.
        ``rchunk`` matches an entire chunk of that length at a fixed, randomly
        chosen position for each detector. Bits outside that chunk are ignored
        for that detector. The same rule censors and scores each detector.

    match_threshold : int, optional (default=1)
        Hamming mismatch limit (0 to n_features - 1), or required contiguous
        run/chunk length (1 to n_features). Only used for binary matching.

    target_coverage : float in (0, 1), optional (default=0.9)
        Desired covered fraction of the bounded non-self region, used only
        for ``coverage``. This is geometric coverage, not anomaly recall.

    coverage_samples : int, optional (default=256)
        Non-self probes per statistical test for ``coverage``. Rejected self
        probes count toward ``max_candidates`` but not the test sample size.

    coverage_confidence : float in (0, 1), optional (default=0.95)
        Simultaneous confidence for the coverage lower bounds. Fresh tests
        spend alpha/(t*(t+1)) in round t, with alpha=1-confidence.

    grid_depth : int, optional (default=4)
        Maximum sparse spatial-tree depth for ``grid`` (1 to 20). For
        ``deterministic``, use 2**grid_depth subdivisions per axis; the
        resulting 4**grid_depth lattice must fit ``max_candidates``.

    hierarchy_levels : int, optional (default=4)
        Number of coarse-to-fine cluster levels for ``hierarchical``.

    outlier_fraction : float in (0, 1), optional (default=0.95)
        For ``suppressed``, flag a self sample when at least this fraction
        of training points are farther away than the distance below. Flagged
        selves become reverse detectors and remain protected as normal.

    outlier_radius : float, optional (default=0.5)
        Positive distance multiplier for ``suppressed``. The outlier-distance
        cutoff is outlier_radius * sqrt(n_features) in scaled feature space.
        Reverse-detector radii use ``self_radius``.

    dual_clusters : int, optional (default=3)
        Initial number of self clusters for ``dual``. Bounded cluster-count
        adjustment seeks radii from two to five times ``self_radius``;
        diagnostics report when this target is infeasible or not reached.

    annealing_steps : int, optional (default=20)
        Maximum optimization sweeps for ``annealed``, also bounded by the
        shared candidate budget. The method returns its best encountered
        energy state. Temperature and neighborhood cooling are documented
        fixed implementation choices, not paper convergence guarantees.

    random_state : int, RandomState instance or None, optional (default=None)
        Controls candidate generation. An integer gives reproducible fits.

    Attributes
    ----------
    detectors_ : numpy array of shape (n_retained, n_features)
        Retained scaled real centers or binary detector vectors, with
        ``n_retained = n_detectors_``. For chunk
        matching, only each detector's selected chunk participates in scoring.

    detector_radii_ : numpy array of shape (n_retained,)
        Positive radii, available for real strategies.

    detector_starts_ : numpy array of shape (n_retained,)
        Chunk start positions, available for binary matching (zero for rules
        that use the complete detector vector).

    n_detectors_ : int
        Actual number of retained detectors.

    n_candidates_ : int
        Number of candidate draws made during fitting.

    n_features_in_ : int
        Number of input features.

    self_samples_ : numpy array of shape (n_samples, n_features)
        Transformed self samples used to censor candidate detectors.

    scaler_ : MinMaxScaler
        Fitted feature transform, available for real strategies.

    sampling_bounds_ : numpy array of shape (2, n_features)
        Lower and upper candidate bounds, available for real strategies.

    binary_thresholds_ : numpy array of shape (n_features,)
        Fitted feature medians, available for the binary strategy.

    generation_diagnostics_ : dict
        Strategy-specific generation statistics for the added real
        strategies, including budget termination and geometry diagnostics.

    coverage_reached_ : bool
        Whether ``coverage`` certified its target before reaching a budget.

    coverage_lower_bound_ : float
        Last complete exact-test lower bound for ``coverage``; zero if no
        complete test was possible. It remains valid after adding spheres.

    coverage_test_count_ : int
        Number of completed independent coverage tests for ``coverage``.

    reverse_detectors_ : numpy array of shape (n_reverse, n_features)
        Self detectors used to suppress negative matches for ``suppressed``.

    reverse_radius_ : float
        Radius of each reverse detector for ``suppressed``.

    apc_centers_ : numpy array of shape (n_clusters, n_features)
        Enclosing self-cluster centers for ``dual``.

    apc_radii_ : numpy array of shape (n_clusters,)
        Enclosing cluster radii expanded by self_radius for ``dual``.

    decision_scores_ : numpy array of shape (n_samples,)
        Training scores, identical to ``decision_function(X_train)``.
        Larger scores mean stronger evidence of non-self.

    threshold_ : float
        Score threshold computed by PyOD using ``contamination``.

    labels_ : numpy array of shape (n_samples,)
        Training labels: 0 denotes inliers and 1 denotes outliers.

    Notes
    -----
    Real scores are ``max_j(radius_j - distance(x, center_j))``. Positive
    values mean the sample is inside at least one negative detector. For
    ``suppressed``, take the minimum of that score and the distance to the
    nearest reverse detector minus its radius; a positive score then also
    requires being outside every reverse self ball.
    ``dual`` scores instead take the maximum with the signed distance
    outside all enclosing self-cluster balls. Signed
    scores retain ordering outside the detector union instead of collapsing
    all self scores to zero. Binary scores are matching margins with a half
    unit offset so that positive scores mean a match. For tied training
    scores, the standard deviation used by PyOD's probability conversion is
    floored at machine epsilon; raw scores and labels are unchanged.
    These scores are fixed
    at fit time and do not depend on other samples in an inference batch.

    A finite detector set can leave non-self regions uncovered, especially
    in high dimensions. Except for ``dual``, far outside the sampling domain
    real scores can decrease again; those strategies require a bounded
    feature domain. The ``dual`` strategy instead treats the entire
    complement of its learned self envelopes as anomalous, an explicit
    extrapolation assumption that may also produce false alarms.
    Binary quantization loses within-bin information, and contiguous/chunk
    rules depend on feature order.
    """

    def __init__(self, contamination=0.1, strategy='variable', n_detectors=100,
                 self_radius=0.1, detector_radius=0.1, max_candidates=10000,
                 sampling_margin=0.5, binary_match='hamming',
                 match_threshold=1, random_state=None, target_coverage=0.9,
                 coverage_samples=256, coverage_confidence=0.95,
                 grid_depth=4, hierarchy_levels=4, outlier_fraction=0.95,
                 outlier_radius=0.5, dual_clusters=3, annealing_steps=20):
        self.contamination = contamination
        self.strategy = strategy
        self.n_detectors = n_detectors
        self.self_radius = self_radius
        self.detector_radius = detector_radius
        self.max_candidates = max_candidates
        self.sampling_margin = sampling_margin
        self.binary_match = binary_match
        self.match_threshold = match_threshold
        self.random_state = random_state
        self.target_coverage = target_coverage
        self.coverage_samples = coverage_samples
        self.coverage_confidence = coverage_confidence
        self.grid_depth = grid_depth
        self.hierarchy_levels = hierarchy_levels
        self.outlier_fraction = outlier_fraction
        self.outlier_radius = outlier_radius
        self.dual_clusters = dual_clusters
        self.annealing_steps = annealing_steps

    def fit(self, X, y=None):
        """Generate detectors that exclude all supplied self samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite numeric normal/self samples.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._clear_fitted_state()
        try:
            return self._fit(X)
        except Exception:
            # Late failures (geometry, scoring, thresholding) are also
            # unfitted, including sklearn's attribute-free fitted check.
            self._clear_fitted_state()
            raise

    def _clear_fitted_state(self):
        for name in list(vars(self)):
            if name.endswith('_') or name in ('_mu', '_sigma', '_classes'):
                delattr(self, name)

    def _fit(self, X):
        BaseDetector.__init__(self, contamination=self.contamination)
        X = check_array(X, dtype=float)
        self._validate_parameters(X.shape[1])
        rng = check_random_state(self.random_state)
        self._set_n_classes(None)
        self.n_features_in_ = X.shape[1]
        self.strategy_ = self.strategy
        self.binary_match_ = self.binary_match
        self.match_threshold_ = self.match_threshold
        self.n_candidates_ = 0

        if self.strategy_ == 'binary':
            self.binary_thresholds_ = np.median(X, axis=0)
            if not np.all(np.isfinite(self.binary_thresholds_)):
                raise ValueError('Feature medians overflowed; rescale X.')
            self.self_samples_ = X > self.binary_thresholds_
            centers, extra = self._generate_binary(rng)
        else:
            self.scaler_ = MinMaxScaler()
            self.self_samples_ = check_array(self.scaler_.fit_transform(X))
            self.sampling_bounds_ = np.vstack((
                self.self_samples_.min(axis=0) - self.sampling_margin,
                self.self_samples_.max(axis=0) + self.sampling_margin))
            if self.strategy_ in ('fixed', 'variable'):
                centers, extra = self._generate_real(rng)
            else:
                centers, extra = self._generate_advanced(rng)

        if len(centers) == 0:
            raise ValueError(
                'No valid negative detectors were found within '
                'max_candidates; '
                'reduce the matching region or increase the sampling budget.')
        self.detectors_ = np.asarray(centers)
        self.n_detectors_ = len(centers)
        if self.strategy_ == 'binary':
            self.detector_starts_ = np.asarray(extra, dtype=int)
        else:
            self.detector_radii_ = np.asarray(extra)
        target = (self.generation_diagnostics_['target_detectors']
                  if self.strategy_ == 'annealed' else self.n_detectors)
        if (self.n_detectors_ < target
                and not getattr(self, 'coverage_reached_', False)
                and self.strategy_ not in ('voronoi', 'deterministic')):
            warnings.warn(
                'Only %d of %d negative detectors were found within '
                'max_candidates. Training samples remain treated as self.'
                % (self.n_detectors_, target),
                UserWarning, stacklevel=3)
        if self.strategy_ == 'coverage' and not self.coverage_reached_:
            warnings.warn(
                'Coverage target was not certified within the generation '
                'budgets; inspect coverage_lower_bound_ and '
                'generation_diagnostics_.', UserWarning, stacklevel=3)
        if (self.strategy_ in ('voronoi', 'deterministic')
                and self.generation_diagnostics_.get('truncated', False)):
            warnings.warn(
                'n_detectors truncated the finite geometric detector set; '
                'no complete domain coverage is claimed.',
                UserWarning, stacklevel=3)
        if (self.strategy_ == 'dual' and not self.generation_diagnostics_[
                'apc_radius_criterion_met']):
            warnings.warn(
                'APC radius interval was not achieved by the bounded '
                'cluster search; using the best enclosing self cover. '
                'Inspect generation_diagnostics_.',
                UserWarning, stacklevel=3)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        # Tied binary/self scores are legitimate. Keep the inherited 'unify'
        # probability conversion finite without changing scores or labels.
        if self._sigma == 0:
            self._sigma = np.finfo(float).eps
        return self

    def decision_function(self, X):
        """Return signed matching margins; higher values are more anomalous.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite numeric samples in the same feature domain used for fit.

        Returns
        -------
        scores : numpy array of shape (n_samples,)
            Batch-independent anomaly scores.
        """
        check_is_fitted(self, ['detectors_', 'n_features_in_'])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('X has %d features; NSA was fitted with %d.'
                             % (X.shape[1], self.n_features_in_))
        if self.strategy_ == 'binary':
            encoded = X > self.binary_thresholds_
            scores = np.full(X.shape[0], -np.inf)
            for detector, start in zip(self.detectors_, self.detector_starts_):
                scores = np.maximum(
                    scores, self._binary_margin(encoded, detector, start))
            return scores
        transformed = check_array(self.scaler_.transform(X))
        scores = np.empty(X.shape[0])
        # Bound the pairwise matrix to about 8 MiB independently of row count.
        block_size = max(1, 1048576 // self.n_detectors_)
        for start in range(0, X.shape[0], block_size):
            stop = min(start + block_size, X.shape[0])
            distances = cdist(transformed[start:stop], self.detectors_)
            scores[start:stop] = np.max(
                self.detector_radii_ - distances, axis=1)
            if not np.all(np.isfinite(scores[start:stop])):
                raise ValueError('Distances overflowed; rescale X.')
        if self.strategy_ == 'suppressed':
            block_size = max(1, 1048576 // len(self.reverse_detectors_))
            for start in range(0, X.shape[0], block_size):
                stop = min(start + block_size, X.shape[0])
                reverse_margin = np.min(cdist(
                    transformed[start:stop], self.reverse_detectors_),
                    axis=1) - self.reverse_radius_
                if not np.all(np.isfinite(reverse_margin)):
                    raise ValueError('Distances overflowed; rescale X.')
                scores[start:stop] = np.minimum(scores[start:stop],
                                                reverse_margin)
        if self.strategy_ == 'dual':
            block_size = max(1, 1048576 // len(self.apc_centers_))
            for start in range(0, X.shape[0], block_size):
                stop = min(start + block_size, X.shape[0])
                outside = np.min(cdist(transformed[start:stop],
                                       self.apc_centers_) - self.apc_radii_,
                                 axis=1)
                if not np.all(np.isfinite(outside)):
                    raise ValueError('Distances overflowed; rescale X.')
                scores[start:stop] = np.maximum(scores[start:stop], outside)
        return scores

    def _validate_parameters(self, n_features):
        if self.strategy not in ('binary', 'fixed', 'variable', 'coverage',
                                 'grid', 'hierarchical', 'voronoi',
                                 'deterministic', 'suppressed', 'dual',
                                 'annealed'):
            raise ValueError('Unknown strategy: %s' % self.strategy)
        for name in ('n_detectors', 'max_candidates', 'coverage_samples',
                     'grid_depth', 'hierarchy_levels', 'dual_clusters',
                     'annealing_steps'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or value < 1):
                raise ValueError('%s must be a positive integer' % name)
        if self.grid_depth > 20:
            raise ValueError('grid_depth must be an integer in [1, 20]')
        for name in ('target_coverage', 'coverage_confidence',
                     'outlier_fraction'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 < value < 1):
                raise ValueError('%s must be finite and in (0, 1)' % name)
        for name in ('self_radius', 'detector_radius', 'sampling_margin',
                     'outlier_radius'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0
                    or (name in ('detector_radius', 'outlier_radius')
                        and value == 0)):
                raise ValueError('%s must be finite and %s' % (
                    name, 'positive' if name in ('detector_radius',
                                                 'outlier_radius')
                    else 'nonnegative'))
        if self.binary_match not in ('hamming', 'rcontiguous', 'rchunk'):
            raise ValueError('binary_match must be hamming, rcontiguous, '
                             'or rchunk')
        if self.strategy == 'binary':
            value = self.match_threshold
            lower = 0 if self.binary_match == 'hamming' else 1
            upper = (n_features - 1 if self.binary_match == 'hamming'
                     else n_features)
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not lower <= value <= upper):
                raise ValueError('match_threshold must be an integer in '
                                 '[%d, %d] for this matching rule and feature '
                                 'count' % (lower, upper))

    def _generate_advanced(self, rng):
        args = (self.self_samples_, self.sampling_bounds_, rng,
                self.n_detectors, self.self_radius, self.max_candidates)
        if self.strategy_ == 'coverage':
            from ._nsa_coverage import generate_coverage_detectors
            result = generate_coverage_detectors(
                *args, target_coverage=self.target_coverage,
                coverage_samples=self.coverage_samples,
                coverage_confidence=self.coverage_confidence)
        elif self.strategy_ == 'grid':
            from ._nsa_grid import generate_grid_detectors
            result = generate_grid_detectors(*args, grid_depth=self.grid_depth)
        elif self.strategy_ == 'hierarchical':
            from ._nsa_cluster import generate_hierarchical_detectors
            result = generate_hierarchical_detectors(
                *args, n_levels=self.hierarchy_levels)
        elif self.strategy_ == 'deterministic':
            from ._nsa_grid import generate_deterministic_detectors
            result = generate_deterministic_detectors(
                *args, grid_depth=self.grid_depth)
        elif self.strategy_ == 'suppressed':
            from ._nsa_optimization import generate_suppressed_detectors
            result = generate_suppressed_detectors(
                *args, outlier_fraction=self.outlier_fraction,
                outlier_radius=self.outlier_radius)
        elif self.strategy_ == 'dual':
            from ._nsa_cluster import generate_dual_detectors
            result = generate_dual_detectors(
                *args, dual_clusters=self.dual_clusters)
        elif self.strategy_ == 'annealed':
            from ._nsa_optimization import generate_annealed_detectors
            result = generate_annealed_detectors(
                *args, detector_radius=self.detector_radius,
                annealing_steps=self.annealing_steps)
        else:
            from ._nsa_optimization import generate_voronoi_detectors
            result = generate_voronoi_detectors(*args)
        centers, radii, self.n_candidates_, diagnostics = result
        self.generation_diagnostics_ = diagnostics
        if self.strategy_ == 'coverage':
            self.coverage_reached_ = diagnostics['coverage_reached']
            self.coverage_lower_bound_ = diagnostics['coverage_lower_bound']
            self.coverage_test_count_ = diagnostics['coverage_test_count']
        if self.strategy_ == 'suppressed':
            self.reverse_detectors_ = diagnostics['reverse_centers']
            self.reverse_radius_ = diagnostics['reverse_radius']
        if self.strategy_ == 'dual':
            self.apc_centers_ = diagnostics['apc_centers']
            self.apc_radii_ = diagnostics['apc_radii']
        return centers, radii

    def _generate_real(self, rng):
        centers, radii = [], []
        for _ in range(self.max_candidates):
            self.n_candidates_ += 1
            candidate = rng.uniform(*self.sampling_bounds_)
            distance = cdist(candidate[None, :], self.self_samples_).min()
            if not np.isfinite(distance):
                raise ValueError('Candidate distances overflowed; reduce '
                                 'sampling_margin.')
            radius = (self.detector_radius if self.strategy_ == 'fixed'
                      else distance - self.self_radius)
            if radius <= 0:
                continue
            if (self.strategy_ == 'fixed'
                    and distance <= self.self_radius + radius):
                continue
            if centers:
                distances = cdist(candidate[None, :], np.asarray(centers))[0]
                if np.any(distances == 0):
                    continue
                # A variable detector centered in existing coverage adds
                # redundancy; skip it without weakening self exclusion.
                if (self.strategy_ == 'variable'
                        and np.any(distances <= np.asarray(radii))):
                    continue
            centers.append(candidate)
            radii.append(radius)
            if len(centers) == self.n_detectors:
                break
        return centers, radii

    def _generate_binary(self, rng):
        centers, starts, seen = [], [], set()
        for _ in range(self.max_candidates):
            self.n_candidates_ += 1
            candidate = rng.randint(2, size=self.n_features_in_).astype(bool)
            max_start = self.n_features_in_ - self.match_threshold_
            start = (rng.randint(max_start + 1)
                     if self.binary_match_ == 'rchunk' else 0)
            # Chunk identity includes its position, not ignored outside bits.
            key_bits = (candidate[start:start + self.match_threshold_]
                        if self.binary_match_ == 'rchunk' else candidate)
            key = (start, key_bits.tobytes())
            if key in seen:
                continue
            seen.add(key)
            if np.any(self._binary_margin(self.self_samples_, candidate,
                                          start) > 0):
                continue
            centers.append(candidate)
            starts.append(start)
            if len(centers) == self.n_detectors:
                break
        return centers, starts

    def _binary_margin(self, X, detector, start):
        if self.binary_match_ == 'hamming':
            return self.match_threshold_ + 0.5 - np.count_nonzero(
                X != detector, axis=1)
        if self.binary_match_ == 'rchunk':
            stop = start + self.match_threshold_
            return 0.5 - np.count_nonzero(
                X[:, start:stop] != detector[start:stop], axis=1)
        run = np.zeros(X.shape[0], dtype=int)
        longest = np.zeros_like(run)
        for column in range(self.n_features_in_):
            run = np.where(X[:, column] == detector[column], run + 1, 0)
            longest = np.maximum(longest, run)
        return longest - self.match_threshold_ + 0.5
