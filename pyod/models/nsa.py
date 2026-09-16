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
    """Negative-selection-inspired detector with three sampling strategies.

    Negative selection retains random detectors only when they do not match
    the training (self) samples. This estimator implements binary matching,
    fixed-radius hyperspheres, and variable-radius hyperspheres. The sources
    of these ideas are :cite:`forrest1994self,gonzalez2003anomaly,ji2004real`.
    These are NSA-inspired PyOD adaptations, not reproductions of every step
    of those papers. In particular, this implementation uses bounded random
    sampling and signed matching margins instead of a binary alarm, does not
    implement an adaptive population or an auxiliary classifier, and does
    not estimate a target coverage probability.

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

    strategy : {'binary', 'fixed', 'variable'}, optional (default='variable')
        Detector representation and generation strategy. ``fixed`` retains
        hyperspheres with ``detector_radius``; ``variable`` sets each radius
        to its center's nearest-self distance minus ``self_radius``.
        ``binary`` uses one median-threshold bit for every input feature.

    n_detectors : int, optional (default=100)
        Maximum number of distinct detectors to retain. Fewer may be found
        within ``max_candidates``; in that case fitting emits a UserWarning.

    self_radius : float, optional (default=0.1)
        Nonnegative radius around each self sample in scaled real space.
        Used by the ``fixed`` and ``variable`` strategies.

    detector_radius : float, optional (default=0.1)
        Positive detector radius in scaled space for ``strategy='fixed'``.
        Candidate spheres must be separated from all self spheres.

    max_candidates : int, optional (default=10000)
        Positive limit on random candidate draws, including rejected and
        duplicate candidates. If none is valid, fitting raises ValueError.

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
    values mean the sample is inside at least one negative detector. Signed
    scores retain ordering outside the detector union instead of collapsing
    all self scores to zero. Binary scores are matching margins with a half
    unit offset so that positive scores mean a match. For tied training
    scores, the standard deviation used by PyOD's probability conversion is
    floored at machine epsilon; raw scores and labels are unchanged.
    These scores are fixed
    at fit time and do not depend on other samples in an inference batch.

    A finite detector set can leave non-self regions uncovered, especially
    in high dimensions. Far outside the sampling domain, real scores can
    decrease again; this method is intended for a bounded feature domain.
    Binary quantization loses within-bin information, and contiguous/chunk
    rules depend on feature order. It is not a general extrapolation method.
    """

    def __init__(self, contamination=0.1, strategy='variable', n_detectors=100,
                 self_radius=0.1, detector_radius=0.1, max_candidates=10000,
                 sampling_margin=0.5, binary_match='hamming',
                 match_threshold=1, random_state=None):
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
        # A failed refit must not expose a detector set from an earlier fit.
        for name in ('detectors_', 'detector_radii_', 'detector_starts_',
                     'scaler_', 'sampling_bounds_', 'binary_thresholds_',
                     'decision_scores_', 'threshold_', 'labels_'):
            if hasattr(self, name):
                delattr(self, name)
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
            centers, extra = self._generate_real(rng)

        if not centers:
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
        if self.n_detectors_ < self.n_detectors:
            warnings.warn(
                'Only %d of %d negative detectors were found within '
                'max_candidates. The retained detectors still exclude self.'
                % (self.n_detectors_, self.n_detectors),
                UserWarning, stacklevel=2)
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
        return scores

    def _validate_parameters(self, n_features):
        if self.strategy not in ('binary', 'fixed', 'variable'):
            raise ValueError("strategy must be 'binary', 'fixed', "
                             "or 'variable'")
        for name in ('n_detectors', 'max_candidates'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or value < 1):
                raise ValueError('%s must be a positive integer' % name)
        for name in ('self_radius', 'detector_radius', 'sampling_margin'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0
                    or (name == 'detector_radius' and value == 0)):
                raise ValueError('%s must be finite and %s' % (
                    name, 'positive' if name == 'detector_radius'
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
