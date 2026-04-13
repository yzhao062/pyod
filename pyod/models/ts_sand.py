# -*- coding: utf-8 -*-
"""SAND: Streaming Anomaly detection with Normalization and Drift adaptation.

Simplified PyOD adaptation of:
Boniol, P., Paparrizos, J., Palpanas, T. and Franklin, M.J., 2021.
SAND: Streaming subsequence anomaly detection.
*Proceedings of the VLDB Endowment*, 14(10), pp. 1717-1729.

This implementation extracts z-normalized sliding-window subsequences,
initializes k centroids using k-Shape on the first batch, then scores
remaining subsequences by SBD (shape-based distance) to their nearest
centroid.  Centroids are updated every ``batch_size`` subsequences via
exponential moving average with parameter ``alpha``.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import (validate_ts_input, sliding_windows,
                         map_scores_to_timestamps, aggregate_channel_scores)
from .ts_kshape import _znormalize, _sbd, _kshape


class SAND(BaseDetector):
    """SAND streaming anomaly detector for time series.

    Extracts z-normalized sliding-window subsequences, initializes
    k centroids via k-Shape on the first batch, and scores each
    subsequence by its SBD to the nearest centroid.  Centroids are
    updated every ``batch_size`` subsequences using an exponential
    moving average, enabling drift adaptation.

    Parameters
    ----------
    window_size : int, optional (default=50)
        Subsequence length for the sliding window.

    n_clusters : int, optional (default=5)
        Number of centroids for k-Shape clustering.

    alpha : float, optional (default=0.5)
        Smoothing factor for centroid updates.
        ``new = alpha * batch_centroid + (1 - alpha) * old_centroid``.

    batch_size : int, optional (default=100)
        Number of subsequences between centroid updates.

    max_iter : int, optional (default=50)
        Maximum iterations for initial k-Shape clustering.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers.  Must be in (0, 0.5].

    channel_aggregation : str, optional (default='max')
        How to aggregate per-channel scores for multivariate input.
        One of ``'max'`` or ``'mean'``.

    random_state : int or None, optional (default=42)
        Random seed for reproducible centroid initialization.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data.  Higher is more abnormal.

    threshold_ : float
        Score threshold derived from ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels (0: inlier, 1: outlier).

    centroids_ : list of numpy arrays
        Per-channel centroids after processing all training data.

    Examples
    --------
    >>> from pyod.models.ts_sand import SAND
    >>> import numpy as np
    >>> X_train = np.random.randn(500)
    >>> clf = SAND(n_clusters=3, window_size=20, contamination=0.1)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(np.random.randn(200))

    References
    ----------
    .. [1] Boniol, P., Paparrizos, J., Palpanas, T. and Franklin, M.J.,
       2021. SAND: Streaming subsequence anomaly detection.
       *Proceedings of the VLDB Endowment*, 14(10), pp. 1717-1729.
    """

    def __init__(self, window_size=50, n_clusters=5, alpha=0.5,
                 batch_size=100, max_iter=50, contamination=0.1,
                 channel_aggregation='max', random_state=42):
        super(SAND, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.channel_aggregation = channel_aggregation
        self.random_state = random_state

    def _get_min_length(self):
        """Minimum time series length required."""
        return self.window_size + 1

    @staticmethod
    def _znorm_windows(windows):
        """Z-normalize each row of a window matrix."""
        return np.array([_znormalize(w) for w in windows])

    @staticmethod
    def _score_subsequences(subs, centroids):
        """Score each subsequence by SBD to nearest centroid.

        Parameters
        ----------
        subs : np.ndarray of shape (n_subs, length)
        centroids : np.ndarray of shape (n_clusters, length)

        Returns
        -------
        scores : np.ndarray of shape (n_subs,)
        """
        scores = np.empty(len(subs))
        for i, s in enumerate(subs):
            best_dist = np.inf
            for c in centroids:
                d, _ = _sbd(s, c)
                if d < best_dist:
                    best_dist = d
            scores[i] = best_dist
        return scores

    def _streaming_fit_channel(self, windows):
        """Fit on one channel: initialize centroids, stream with updates.

        Parameters
        ----------
        windows : np.ndarray of shape (n_windows, window_size)
            Z-normalized sliding windows for one channel.

        Returns
        -------
        scores : np.ndarray of shape (n_windows,)
        centroids : np.ndarray of shape (n_clusters, window_size)
        """
        n = len(windows)
        batch_size = min(self.batch_size, n)
        rng = np.random.RandomState(self.random_state)

        # Initialize centroids on the first batch using k-Shape
        init_batch = windows[:batch_size]
        centroids, _, _ = _kshape(init_batch, self.n_clusters,
                                  max_iter=self.max_iter,
                                  random_state=rng)

        # Score all subsequences in a streaming fashion
        scores = np.empty(n)

        # Process in batches for centroid updates
        pos = 0
        while pos < n:
            end = min(pos + batch_size, n)
            batch = windows[pos:end]

            # Score this batch against current centroids
            scores[pos:end] = self._score_subsequences(batch, centroids)

            # Update centroids if this is not the last batch
            if end < n and len(batch) >= self.n_clusters:
                batch_rng = np.random.RandomState(self.random_state)
                batch_centroids, _, _ = _kshape(
                    batch, self.n_clusters,
                    max_iter=self.max_iter,
                    random_state=batch_rng)
                for k in range(self.n_clusters):
                    updated = (self.alpha * batch_centroids[k]
                               + (1.0 - self.alpha) * centroids[k])
                    centroids[k] = _znormalize(updated)

            pos = end

        return scores, centroids

    def fit(self, X, y=None):
        """Fit the SAND detector on time series data.

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
        min_len = self._get_min_length()
        if n_timestamps < min_len:
            raise ValueError(
                "Time series length %d is shorter than minimum "
                "required length %d (window_size=%d)"
                % (n_timestamps, min_len, self.window_size))

        n_windows = n_timestamps - self.window_size + 1
        init_batch = min(self.batch_size, n_windows)
        if init_batch < self.n_clusters:
            raise ValueError(
                "Not enough subsequences in initial batch (%d) for "
                "n_clusters=%d. Need a longer series, larger batch_size, "
                "or fewer clusters." % (init_batch, self.n_clusters))

        self._set_n_classes(y)

        # Store channel count for validation in decision_function
        self.n_channels_ = n_channels

        # Process each channel independently
        self.centroids_ = []
        per_channel_results = []

        for ch in range(n_channels):
            channel = X[:, ch].reshape(-1, 1)
            windows = sliding_windows(channel, self.window_size, step=1)
            windows = self._znorm_windows(windows)

            ch_window_scores, ch_centroids = self._streaming_fit_channel(
                windows)
            self.centroids_.append(ch_centroids)

            # Map window scores back to timestamps
            ts_scores, valid_mask = map_scores_to_timestamps(
                ch_window_scores, self.window_size, step=1,
                n_timestamps=n_timestamps, aggregation='max')
            per_channel_results.append((ts_scores, valid_mask))

        # Aggregate channels
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

        # Masked-score workflow: threshold from valid subset
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
        """Predict raw anomaly scores for a test time series.

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

        if n_channels != self.n_channels_:
            raise ValueError(
                "Channel count mismatch: model fitted on %d channels, "
                "got %d" % (self.n_channels_, n_channels))

        per_channel_results = []
        for ch in range(n_channels):
            channel = X[:, ch].reshape(-1, 1)
            windows = sliding_windows(channel, self.window_size, step=1)
            windows = self._znorm_windows(windows)

            ch_window_scores = self._score_subsequences(
                windows,
                self.centroids_[ch])

            ts_scores, valid_mask = map_scores_to_timestamps(
                ch_window_scores, self.window_size, step=1,
                n_timestamps=n_timestamps, aggregation='max')
            per_channel_results.append((ts_scores, valid_mask))

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

        scores[~valid_mask] = self.threshold_
        return scores
