# -*- coding: utf-8 -*-
"""KShape: k-Shape clustering-based time series anomaly detection.

Adapts the k-Shape clustering algorithm (Paparrizos & Gravano, SIGMOD 2015)
for anomaly detection.  Sliding-window subsequences are clustered using
shape-based distance (SBD), and each subsequence is scored by its SBD to
the nearest centroid.  Subsequences far from all centroids are anomalous.

Reference:
    Paparrizos, J. and Gravano, L., 2015. k-Shape: Efficient and accurate
    clustering of time series. In *Proceedings of the 2015 ACM SIGMOD
    International Conference on Management of Data* (pp. 1855-1870).
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import (validate_ts_input, sliding_windows,
                         map_scores_to_timestamps,
                         aggregate_channel_scores)


def _znormalize(x):
    """Z-normalize a vector.  Returns zero vector if std is near zero."""
    s = np.std(x)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def _sbd(x, y):
    """Shape-based distance between two z-normalized sequences.

    SBD(x, y) = 1 - max_w CC_w(x, y) / (||x|| * ||y||)

    Parameters
    ----------
    x, y : np.ndarray of shape (m,)
        Z-normalized sequences of equal length.

    Returns
    -------
    dist : float
        Shape-based distance in [0, 2].  Lower is more similar.
    shift : int
        The optimal shift (lag) that maximizes cross-correlation.
    """
    m = len(x)
    fft_size = 2 * m - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size_padded = 1
    while fft_size_padded < fft_size:
        fft_size_padded *= 2

    # Cross-correlation via FFT
    fx = np.fft.fft(x, fft_size_padded)
    fy = np.fft.fft(y, fft_size_padded)
    cc = np.real(np.fft.ifft(fx * np.conj(fy)))

    # The meaningful cross-correlation values are at indices
    # 0..m-1 and fft_size_padded-m+1..fft_size_padded-1
    # Rearrange to get lags -(m-1)..0..+(m-1)
    cc_full = np.concatenate([cc[-(m - 1):], cc[:m]])

    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm < 1e-10:
        return 2.0, 0

    ncc = cc_full / norm
    idx = np.argmax(ncc)
    dist = 1.0 - ncc[idx]
    # Clamp to [0, 2] for numerical stability
    dist = max(0.0, min(2.0, dist))
    shift = idx - (m - 1)
    return dist, shift


def _shift_align(x, shift, m):
    """Shift-align x by the given lag, zero-padding where needed.

    Parameters
    ----------
    x : np.ndarray of shape (m,)
    shift : int
        Positive means x is shifted right (leading zeros).
    m : int
        Length of the output.

    Returns
    -------
    aligned : np.ndarray of shape (m,)
    """
    aligned = np.zeros(m)
    if shift >= 0:
        end = min(m, m - shift)
        if end > 0:
            aligned[shift:shift + end] = x[:end]
    else:
        start = -shift
        end = min(m, m + shift)
        if end > 0:
            aligned[:end] = x[start:start + end]
    return aligned


def _compute_centroid(members):
    """Compute the k-Shape centroid for a set of z-normalized, shift-aligned
    cluster members.

    The centroid maximizes total similarity.  It is the eigenvector
    corresponding to the smallest eigenvalue of M = I - Q/n, where
    Q = X^T X and X has members as rows.

    Parameters
    ----------
    members : np.ndarray of shape (n_members, m)
        Z-normalized and shift-aligned cluster members.

    Returns
    -------
    centroid : np.ndarray of shape (m,)
        Z-normalized centroid.
    """
    n, m = members.shape
    if n == 0:
        return np.zeros(m)
    if n == 1:
        return _znormalize(members[0])

    Q = members.T @ members  # (m, m)
    M = np.eye(m) - Q / n

    # The centroid maximises the Rayleigh quotient for similarity,
    # which corresponds to the *smallest* eigenvalue of M = I - Q/n.
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # eigh returns eigenvalues in ascending order; we want the smallest
    centroid = eigenvectors[:, 0]

    # Orient sign: pick the sign that maximizes aggregate similarity
    # with cluster members (eigenvectors are defined up to sign)
    if np.dot(members.mean(axis=0), centroid) < 0:
        centroid = -centroid

    return _znormalize(centroid)


def _kshape(subsequences, n_clusters, max_iter, random_state):
    """Run the k-Shape clustering algorithm.

    Parameters
    ----------
    subsequences : np.ndarray of shape (n_windows, window_size)
        Z-normalized sliding-window subsequences.
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum number of Lloyd's iterations.
    random_state : np.random.RandomState
        Random state for reproducible initialization.

    Returns
    -------
    centroids : np.ndarray of shape (n_clusters, window_size)
        Final cluster centroids (z-normalized).
    labels : np.ndarray of shape (n_windows,)
        Cluster assignments.
    distances : np.ndarray of shape (n_windows,)
        SBD to the nearest centroid.
    """
    n_windows, m = subsequences.shape

    # Initialize centroids by randomly selecting subsequences
    indices = random_state.choice(n_windows, size=n_clusters, replace=False)
    centroids = np.array([_znormalize(subsequences[i]) for i in indices])

    labels = np.full(n_windows, -1, dtype=int)
    distances = np.full(n_windows, np.inf)

    for iteration in range(max_iter):
        old_labels = labels.copy()

        # --- Assignment step ---
        for i in range(n_windows):
            best_dist = np.inf
            best_label = 0
            for k in range(n_clusters):
                d, _ = _sbd(subsequences[i], centroids[k])
                if d < best_dist:
                    best_dist = d
                    best_label = k
            labels[i] = best_label
            distances[i] = best_dist

        # Check convergence
        if np.array_equal(labels, old_labels):
            break

        # --- Update step ---
        for k in range(n_clusters):
            members_idx = np.where(labels == k)[0]
            if len(members_idx) == 0:
                # Re-initialize empty cluster with a random subsequence
                idx = random_state.randint(n_windows)
                centroids[k] = _znormalize(subsequences[idx])
                continue

            # Shift-align each member to the current centroid
            aligned = np.empty((len(members_idx), m))
            for j, idx in enumerate(members_idx):
                _, shift = _sbd(subsequences[idx], centroids[k])
                aligned[j] = _znormalize(
                    _shift_align(subsequences[idx], shift, m))

            centroids[k] = _compute_centroid(aligned)

    # Final distance computation with converged centroids
    for i in range(n_windows):
        best_dist = np.inf
        for k in range(n_clusters):
            d, _ = _sbd(subsequences[i], centroids[k])
            if d < best_dist:
                best_dist = d
        distances[i] = best_dist

    return centroids, labels, distances


class KShape(BaseDetector):
    """k-Shape clustering-based time series anomaly detector.

    Extracts sliding-window subsequences, clusters them using the
    k-Shape algorithm (shape-based distance + eigenvalue centroid
    update), and scores each subsequence by its SBD to the nearest
    centroid.  Subsequences far from all centroids are considered
    anomalous.

    This is an **inductive** detector: after fitting on training data,
    ``decision_function`` can score new time series.

    Parameters
    ----------
    n_clusters : int, optional (default=3)
        Number of clusters for k-Shape.

    window_size : int, optional (default=50)
        Size of the sliding window for extracting subsequences.

    max_iter : int, optional (default=100)
        Maximum number of Lloyd's iterations for k-Shape.

    contamination : float in (0., 0.5), optional (default=0.1)
        Expected proportion of outliers in the dataset.

    channel_aggregation : str, optional (default='max')
        How to aggregate per-channel scores for multivariate input.
        One of ``'max'`` or ``'mean'``.

    random_state : int or None, optional (default=42)
        Random seed for reproducible centroid initialization.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data. Higher is more abnormal.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels of training data (0: inlier, 1: outlier).

    centroids_ : list of numpy arrays
        Per-channel cluster centroids learned during fit.

    Examples
    --------
    >>> from pyod.models.ts_kshape import KShape
    >>> import numpy as np
    >>> X_train = np.random.randn(300)
    >>> clf = KShape(n_clusters=3, window_size=20, contamination=0.1)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(np.random.randn(200))

    References
    ----------
    .. [1] Paparrizos, J. and Gravano, L., 2015. k-Shape: Efficient and
       accurate clustering of time series. In *Proceedings of the 2015
       ACM SIGMOD International Conference on Management of Data*
       (pp. 1855-1870).
    """

    def __init__(self, n_clusters=3, window_size=50, max_iter=100,
                 contamination=0.1, channel_aggregation='max',
                 random_state=42):
        super(KShape, self).__init__(contamination=contamination)
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.max_iter = max_iter
        self.channel_aggregation = channel_aggregation
        self.random_state = random_state

    def _fit_channel(self, ts):
        """Fit k-Shape on a single channel and return per-timestamp scores.

        Parameters
        ----------
        ts : np.ndarray of shape (n_timestamps,)
            Single-channel time series.

        Returns
        -------
        scores : np.ndarray of shape (n_timestamps,)
            Per-timestamp anomaly scores.
        valid_mask : np.ndarray of shape (n_timestamps,), dtype=bool
            Mask indicating which timestamps have valid scores.
        centroids : np.ndarray of shape (n_clusters, window_size)
            Learned centroids.
        """
        n_timestamps = len(ts)
        # Extract sliding windows as a 2D array
        X_2d = ts.reshape(-1, 1)
        windows = sliding_windows(X_2d, self.window_size, step=1)

        # Z-normalize each window
        znorm_windows = np.array([_znormalize(w) for w in windows])

        rng = np.random.RandomState(self.random_state)
        centroids, _, distances = _kshape(
            znorm_windows, self.n_clusters, self.max_iter, rng)

        # Map window-level scores to timestamps
        scores, valid_mask = map_scores_to_timestamps(
            distances, self.window_size, step=1,
            n_timestamps=n_timestamps, aggregation='max')

        return scores, valid_mask, centroids

    def _score_channel(self, ts, centroids):
        """Score a single channel against pre-learned centroids.

        Parameters
        ----------
        ts : np.ndarray of shape (n_timestamps,)
            Single-channel time series.
        centroids : np.ndarray of shape (n_clusters, window_size)
            Learned centroids.

        Returns
        -------
        scores : np.ndarray of shape (n_timestamps,)
            Per-timestamp anomaly scores.
        valid_mask : np.ndarray of shape (n_timestamps,), dtype=bool
        """
        n_timestamps = len(ts)
        X_2d = ts.reshape(-1, 1)
        windows = sliding_windows(X_2d, self.window_size, step=1)
        znorm_windows = np.array([_znormalize(w) for w in windows])

        # Score each window by SBD to nearest centroid
        n_windows = len(znorm_windows)
        distances = np.empty(n_windows)
        for i in range(n_windows):
            best_dist = np.inf
            for k in range(len(centroids)):
                d, _ = _sbd(znorm_windows[i], centroids[k])
                if d < best_dist:
                    best_dist = d
            distances[i] = best_dist

        scores, valid_mask = map_scores_to_timestamps(
            distances, self.window_size, step=1,
            n_timestamps=n_timestamps, aggregation='max')

        return scores, valid_mask

    def fit(self, X, y=None):
        """Fit the k-Shape anomaly detector on time series data.

        Parameters
        ----------
        X : array-like of shape (n_timestamps,) or (n_timestamps, n_channels)
            Training time series data.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_ts_input(X)
        n_timestamps, n_channels = X.shape

        n_windows = n_timestamps - self.window_size + 1
        if n_windows < 1:
            raise ValueError(
                "Time series length %d is too short for window_size=%d. "
                "Need at least %d timestamps."
                % (n_timestamps, self.window_size, self.window_size + 1))
        if n_windows < self.n_clusters:
            raise ValueError(
                "Not enough subsequences (%d) for n_clusters=%d. "
                "Need a longer series or fewer clusters."
                % (n_windows, self.n_clusters))

        self._set_n_classes(y)

        # Fit k-Shape per channel
        self.centroids_ = []
        per_channel_results = []
        for ch in range(n_channels):
            scores, valid_mask, centroids = self._fit_channel(X[:, ch])
            self.centroids_.append(centroids)
            per_channel_results.append((scores, valid_mask))

        if n_channels == 1:
            scores, valid_mask = per_channel_results[0]
        else:
            # Aggregate across channels
            filled_scores = []
            combined_valid = per_channel_results[0][1].copy()
            for ch_scores, ch_valid in per_channel_results:
                filled = ch_scores.copy()
                filled[~ch_valid] = 0.0
                filled_scores.append(filled)
                combined_valid &= ch_valid
            scores = aggregate_channel_scores(
                filled_scores, method=self.channel_aggregation)
            valid_mask = combined_valid

        # Masked-score workflow: compute threshold on valid scores only
        valid_scores = scores[valid_mask]
        self.decision_scores_ = valid_scores
        self._process_decision_scores()

        # Reconstruct full-length arrays
        full_scores = scores.copy()
        full_scores[~valid_mask] = self.threshold_
        full_labels = (full_scores > self.threshold_).astype(int)
        self.decision_scores_ = full_scores
        self.labels_ = full_labels

        return self

    def decision_function(self, X):
        """Predict raw anomaly scores for time series X.

        Parameters
        ----------
        X : array-like of shape (n_timestamps,) or (n_timestamps, n_channels)
            Test time series data.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_timestamps,)
            Anomaly scores.  Higher is more abnormal.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_',
                                'centroids_'])
        X = validate_ts_input(X)
        n_timestamps, n_channels = X.shape

        if n_channels != len(self.centroids_):
            raise ValueError(
                "Number of channels in X (%d) does not match the number "
                "of channels seen during fit (%d)."
                % (n_channels, len(self.centroids_)))

        per_channel_results = []
        for ch in range(n_channels):
            scores, valid_mask = self._score_channel(
                X[:, ch], self.centroids_[ch])
            per_channel_results.append((scores, valid_mask))

        if n_channels == 1:
            scores, valid_mask = per_channel_results[0]
        else:
            filled_scores = []
            combined_valid = per_channel_results[0][1].copy()
            for ch_scores, ch_valid in per_channel_results:
                filled = ch_scores.copy()
                filled[~ch_valid] = 0.0
                filled_scores.append(filled)
                combined_valid &= ch_valid
            scores = aggregate_channel_scores(
                filled_scores, method=self.channel_aggregation)
            valid_mask = combined_valid

        # Fill invalid positions with threshold
        scores[~valid_mask] = self.threshold_
        return scores
