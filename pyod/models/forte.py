# -*- coding: utf-8 -*-
"""Forte: Finding Outliers with Representation Typicality Estimation.
"""
# Author: Debargha Ganguly <maildebargha@gmail.com>
# License: BSD 2 clause

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ..utils.utility import invert_order

# number of distance matrix entries held in memory at a time
_CHUNK_ENTRIES = 2 ** 25

_GMM_COMPONENTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_OCSVM_NU = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
_KDE_BANDWIDTHS = ('scott', 'silverman')


def _chunks(n_cols, n_rows):
    step = max(1, _CHUNK_ENTRIES // max(1, n_rows))
    for start in range(0, n_cols, step):
        yield slice(start, min(start + step, n_cols))


def _knn_radii(X, n_neighbors):
    """Distance of every sample to its k-th nearest neighbor within X.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.

    n_neighbors : int
        The neighbor rank k. The sample itself is not counted.

    Returns
    -------
    radii : numpy array of shape (n_samples,)
        The k-th nearest neighbor distances.
    """
    n_samples = X.shape[0]
    radii = np.empty(n_samples, dtype=np.float64)
    for sl in _chunks(n_samples, n_samples):
        dist = euclidean_distances(X[sl], X)
        dist[np.arange(sl.stop - sl.start), np.arange(sl.start, sl.stop)] = 0.
        radii[sl] = np.partition(dist, n_neighbors, axis=1)[:, n_neighbors]
    return radii


def per_point_prdc(X_ref, X, n_neighbors=5, ref_radii=None):
    """Per-point precision, recall, density and coverage of the sample X
    with respect to the reference sample X_ref.

    Recall and coverage use the k-th nearest neighbor distance of each
    point within X, so the statistics of a point depend on the sample
    it is evaluated with.

    Parameters
    ----------
    X_ref : numpy array of shape (n_ref_samples, n_features)
        The reference sample.

    X : numpy array of shape (n_samples, n_features)
        The evaluated sample.

    n_neighbors : int, optional (default=5)
        The neighbor rank k used for the nearest neighbor radii.

    ref_radii : numpy array of shape (n_ref_samples,), optional
        Precomputed k-th nearest neighbor distances within X_ref.

    Returns
    -------
    prdc : numpy array of shape (n_samples, 4)
        The columns are recall, density, precision and coverage.
    """
    X_ref = np.asarray(X_ref, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n_ref, n_samples = X_ref.shape[0], X.shape[0]

    if ref_radii is None:
        ref_radii = _knn_radii(X_ref, n_neighbors)
    radii = _knn_radii(X, n_neighbors)

    prdc = np.empty((n_samples, 4), dtype=np.float64)
    for sl in _chunks(n_samples, n_ref):
        dist = euclidean_distances(X_ref, X[sl])
        inside_ref = dist < ref_radii[:, np.newaxis]
        prdc[sl, 0] = (dist < radii[sl]).sum(axis=0) / n_ref
        prdc[sl, 1] = inside_ref.sum(axis=0) / (n_neighbors * n_ref)
        prdc[sl, 2] = inside_ref.any(axis=0)
        prdc[sl, 3] = dist.min(axis=0) < radii[sl]
    return prdc


class Forte(BaseDetector):
    """Forte detector for finding outliers with representation typicality
    estimation. See :cite:`ganguly2025forte` for details.

    Forte works on feature representations (embeddings) of the data. The
    training data is split in half into a reference sample and a fit
    sample. Per-point precision, recall, density and coverage (PRDC) of
    the fit sample are computed against the reference sample in each
    representation space, and a density model (GMM, KDE or one-class SVM)
    is fitted on these statistics. New data is scored by the density of
    its PRDC statistics against the same reference sample.

    Forte is a two-sample test: recall and coverage use the nearest
    neighbor distances within the evaluated sample, so
    ``decision_function(X)`` treats ``X`` as one sample and the score of a
    point depends on the other points it is scored with. Score the data
    of interest in one call, and do not compare scores across calls made
    on different samples.

    ``decision_scores_`` is obtained by a symmetric cross-fit. The density
    model is fitted on the PRDC statistics of the fit half against the
    reference half, which also gives the training scores of the fit half.
    The training scores of the reference half come from its PRDC
    statistics against the fit half, scored by the same density model.

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        The neighbor rank k used for the nearest neighbor radii in PRDC.

    density_estimator : {'gmm', 'kde', 'ocsvm'}, optional (default='gmm')
        The density model fitted on the PRDC statistics.

        - 'gmm': Gaussian mixture. The number of components is selected
          from (1, 2, 4, ..., 256) by the training log-likelihood.
        - 'kde': Gaussian kernel density estimate. The bandwidth rule
          ('scott' or 'silverman') is selected by the training
          log-likelihood. It fails when a PRDC statistic is constant
          on the fit sample.
        - 'ocsvm': one-class SVM with RBF kernel. ``nu`` is selected by
          the ratio of training inliers.

    space_splits : list of int, optional (default=None)
        Column indices at which ``X`` is split into representation spaces,
        as in ``numpy.split(X, space_splits, axis=1)``. For example, if
        ``X`` concatenates a 768-dimensional and a 384-dimensional
        embedding, use ``space_splits=[768]``. PRDC statistics are
        computed in each space and concatenated. If None, all columns
        form a single space.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting
        to define the threshold on the decision function.

    random_state : int, RandomState instance or None, optional (default=None)
        Controls the reference/fit split of the training data and the
        initialization of the Gaussian mixture.

    Attributes
    ----------
    detector_ : object
        The density model fitted on the PRDC statistics.

    reference_indices_ : numpy array of shape (n_samples // 2,)
        Indices of the training samples used as the reference sample.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_neighbors=5, density_estimator='gmm',
                 space_splits=None, contamination=0.1, random_state=None):
        super(Forte, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.density_estimator = density_estimator
        self.space_splits = space_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, dtype=np.float64)
        self._set_n_classes(y)

        if not isinstance(self.n_neighbors, (int, np.integer)) or \
                self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer, "
                             "got %s" % self.n_neighbors)
        if self.density_estimator not in ('gmm', 'kde', 'ocsvm'):
            raise ValueError("density_estimator must be 'gmm', 'kde' or "
                             "'ocsvm', got %s" % self.density_estimator)

        n_samples, self._n_features = X.shape
        self._slices = self._get_slices(self._n_features)

        ref_idx, fit_idx = train_test_split(
            np.arange(n_samples), test_size=0.5,
            random_state=self.random_state)
        if min(len(ref_idx), len(fit_idx)) <= self.n_neighbors:
            raise ValueError(
                "n_samples=%d is too small for n_neighbors=%d: each half "
                "of the training data must have more than n_neighbors "
                "samples" % (n_samples, self.n_neighbors))

        self.reference_indices_ = ref_idx
        self._ref = [X[ref_idx][:, sl] for sl in self._slices]
        self._ref_radii = [_knn_radii(ref, self.n_neighbors)
                           for ref in self._ref]

        prdc_fit = self._prdc(X[fit_idx])
        self.detector_ = self._fit_density(prdc_fit)

        # training scores of the reference half: swap the two halves
        prdc_ref = np.hstack([
            per_point_prdc(X[fit_idx][:, sl], ref, self.n_neighbors)
            for sl, ref in zip(self._slices, self._ref)])

        self.decision_scores_ = np.empty(n_samples, dtype=np.float64)
        self.decision_scores_[fit_idx] = self._score(prdc_fit)
        self.decision_scores_[ref_idx] = self._score(prdc_ref)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        X is treated as one evaluated sample: the score of a point
        depends on the other points in X. For consistency, outliers are
        assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The evaluated sample. It must have more than ``n_neighbors``
            samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self._n_features:
            raise ValueError("X has %d features, but Forte was fitted with "
                             "%d features" % (X.shape[1], self._n_features))
        if X.shape[0] <= self.n_neighbors:
            raise ValueError(
                "Forte scores X as one sample and needs more than "
                "n_neighbors=%d samples, got %d"
                % (self.n_neighbors, X.shape[0]))
        return self._score(self._prdc(X))

    def _get_slices(self, n_features):
        if self.space_splits is None:
            return [slice(0, n_features)]
        splits = [int(s) for s in self.space_splits]
        bounds = [0] + splits + [n_features]
        if any(lo >= hi for lo, hi in zip(bounds[:-1], bounds[1:])):
            raise ValueError(
                "space_splits must be strictly increasing column indices "
                "in (0, %d), got %s" % (n_features, self.space_splits))
        return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]

    def _prdc(self, X):
        """PRDC statistics of X against the reference sample, concatenated
        over the representation spaces."""
        return np.hstack([
            per_point_prdc(ref, X[:, sl], self.n_neighbors, ref_radii=radii)
            for sl, ref, radii in zip(self._slices, self._ref,
                                      self._ref_radii)])

    def _fit_density(self, prdc):
        if self.density_estimator == 'gmm':
            best_model, best_score = None, -np.inf
            for n_components in _GMM_COMPONENTS:
                if n_components >= prdc.shape[0]:
                    continue
                gmm = GaussianMixture(n_components=n_components,
                                      random_state=self.random_state)
                gmm.fit(prdc)
                score = gmm.score(prdc)
                if score > best_score:
                    best_model, best_score = gmm, score
            return best_model

        if self.density_estimator == 'kde':
            best_model, best_score = None, -np.inf
            for bw_method in _KDE_BANDWIDTHS:
                kde = gaussian_kde(prdc.T, bw_method=bw_method)
                score = kde.logpdf(prdc.T).mean()
                if best_model is None or score > best_score:
                    best_model, best_score = kde, score
            return best_model

        best_model, best_ratio = None, -np.inf
        for nu in _OCSVM_NU:
            ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
            ocsvm.fit(prdc)
            ratio = (ocsvm.predict(prdc) == 1).mean()
            if ratio > best_ratio:
                best_model, best_ratio = ocsvm, ratio
        return best_model

    def _score(self, prdc):
        if self.density_estimator == 'gmm':
            scores = self.detector_.score_samples(prdc)
        elif self.density_estimator == 'kde':
            scores = self.detector_.logpdf(prdc.T)
        else:
            scores = self.detector_.decision_function(prdc)
        # outliers come with higher outlier scores
        return invert_order(np.asarray(scores, dtype=np.float64).ravel())
