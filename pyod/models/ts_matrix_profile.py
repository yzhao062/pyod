# -*- coding: utf-8 -*-
"""MatrixProfile: Time series anomaly detection using the STOMP algorithm.

Computes the Matrix Profile via the STOMP (Scalable Time-series Ordered
Matrix Profile) algorithm, which identifies anomalous subsequences by
measuring z-normalized Euclidean distance to the nearest non-trivial
match.

See :cite:`yeh2016matrix` for details.

Reference:
    Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A.,
    Silva, D.F., Mueen, A. and Keogh, E., 2016. Matrix profile I:
    All pairs similarity joins for time series: a unifying view that
    includes motifs, discords and shapelets. In ICDM, pp. 1317-1322.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import (validate_ts_input, map_scores_to_timestamps,
                         aggregate_channel_scores)


def _compute_matrix_profile(T, m):
    """Compute the Matrix Profile of a 1-D time series using STOMP.

    Parameters
    ----------
    T : np.ndarray of shape (n,)
        Input time series (single channel).
    m : int
        Subsequence (window) length.

    Returns
    -------
    mp : np.ndarray of shape (n - m + 1,)
        Matrix Profile values (nearest-neighbor z-normalized distances).
    """
    n = len(T)
    n_subseq = n - m + 1
    exclusion_zone = m // 4

    # --- Precompute rolling mean and std using cumulative sums ---
    cumsum = np.cumsum(T)
    cumsum2 = np.cumsum(T ** 2)

    # sum of T[i:i+m] for each subsequence i
    subseq_sum = np.empty(n_subseq)
    subseq_sum[0] = cumsum[m - 1]
    subseq_sum[1:] = cumsum[m:] - cumsum[:n - m]

    subseq_sum2 = np.empty(n_subseq)
    subseq_sum2[0] = cumsum2[m - 1]
    subseq_sum2[1:] = cumsum2[m:] - cumsum2[:n - m]

    mu = subseq_sum / m
    sigma_sq = subseq_sum2 / m - mu ** 2
    sigma_sq = np.maximum(sigma_sq, 0.0)  # numerical stability
    sigma = np.sqrt(sigma_sq)

    # Mask for constant subsequences (std < 1e-10)
    const_mask = sigma < 1e-10

    # Initialize Matrix Profile with infinity
    mp = np.full(n_subseq, np.inf)

    # --- First column (j=0): compute distance profile using MASS (FFT) ---
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2

    T_fft = np.fft.rfft(T, n=fft_size)

    # First query subsequence (reversed, then padded)
    query = T[:m][::-1]
    Q_fft = np.fft.rfft(query, n=fft_size)

    # QT[i] = dot product of T[i:i+m] and T[0:m]
    QT_full = np.fft.irfft(T_fft * Q_fft, n=fft_size)
    QT = QT_full[m - 1:m - 1 + n_subseq].copy()

    # Compute distance for j=0
    _update_mp(mp, QT, mu, sigma, const_mask, m, 0, exclusion_zone, n_subseq)

    # Keep a copy of QT for incremental updates
    QT_prev = QT.copy()

    # --- STOMP: incremental updates for j=1..n_subseq-1 ---
    for j in range(1, n_subseq):
        # Incremental QT update:
        # QT_new[i] = QT_old[i-1] - T[i-1]*T[j-1] + T[i+m-1]*T[j+m-1]
        QT_new = np.empty(n_subseq)

        # QT_new[0] must be computed as a direct dot product
        QT_new[0] = np.dot(T[:m], T[j:j + m])

        # Vectorized incremental update for i=1..n_subseq-1
        i_indices = np.arange(1, n_subseq)
        QT_new[1:] = (QT_prev[:-1]
                       - T[i_indices - 1] * T[j - 1]
                       + T[i_indices + m - 1] * T[j + m - 1])

        _update_mp(mp, QT_new, mu, sigma, const_mask, m, j,
                   exclusion_zone, n_subseq)

        QT_prev = QT_new

    return mp


def _update_mp(mp, QT, mu, sigma, const_mask, m, j, exclusion_zone, n_subseq):
    """Update the Matrix Profile for column j.

    Converts QT values to z-normalized distances and updates the
    profile wherever the new distance is smaller.

    Parameters
    ----------
    mp : np.ndarray, modified in-place
    QT : np.ndarray of shape (n_subseq,)
    mu : np.ndarray of shape (n_subseq,)
    sigma : np.ndarray of shape (n_subseq,)
    const_mask : np.ndarray of bool
    m : int
    j : int, current column index
    exclusion_zone : int
    n_subseq : int
    """
    # Compute z-normalized distance:
    #   d = sqrt(2*m*(1 - (QT - m*mu_i*mu_j) / (m*sigma_i*sigma_j)))
    # where i ranges over all subsequences

    # Denominator: m * sigma_i * sigma_j
    denom = m * sigma * sigma[j]

    # Numerator inside the (1 - ...) term
    numerator = QT - m * mu * mu[j]

    # Compute the argument of sqrt
    # Avoid division by zero: where denom is 0 (constant subsequences),
    # distance is inf
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0, numerator / denom, 0.0)

    dist_sq = 2 * m * (1 - corr)

    # Clip for numerical stability
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)

    # Set distance to inf for constant subsequences
    dist[const_mask] = np.inf
    if const_mask[j]:
        dist[:] = np.inf

    # Apply exclusion zone: ignore indices where |i - j| <= exclusion_zone
    ez_start = max(0, j - exclusion_zone)
    ez_end = min(n_subseq, j + exclusion_zone + 1)
    dist[ez_start:ez_end] = np.inf

    # Update Matrix Profile where distance is smaller
    mask = dist < mp
    mp[mask] = dist[mask]


class MatrixProfile(BaseDetector):
    """Matrix Profile time series anomaly detector using STOMP.

    Computes the Matrix Profile via the STOMP algorithm, identifying
    anomalous subsequences by measuring z-normalized Euclidean distance
    to the nearest non-trivial neighbor. Subsequences with high
    Matrix Profile values (discords) are anomalous.

    This is a **transductive** detector: it only scores the training
    data. ``decision_function``, ``predict``, ``predict_proba``, and
    ``predict_confidence`` raise ``NotImplementedError`` when called
    with new data.

    Parameters
    ----------
    window_size : int, optional (default=50)
        Subsequence length for the Matrix Profile computation.

    contamination : float in (0., 0.5), optional (default=0.1)
        Expected proportion of outliers in the dataset.

    channel_aggregation : str, optional (default='max')
        How to aggregate per-channel scores for multivariate data.
        One of 'max' or 'mean'.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data. Higher is more abnormal.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels of training data (0: inlier, 1: outlier).

    References
    ----------
    .. bibliography::
       Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A.,
       Silva, D.F., Mueen, A. and Keogh, E., 2016. Matrix profile I:
       All pairs similarity joins for time series. In ICDM, pp. 1317-1322.

    Examples
    --------
    >>> from pyod.models.ts_matrix_profile import MatrixProfile
    >>> import numpy as np
    >>> X_train = np.random.randn(300)
    >>> clf = MatrixProfile(window_size=20, contamination=0.1)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_scores_
    """

    def __init__(self, window_size=50, contamination=0.1,
                 channel_aggregation='max'):
        super(MatrixProfile, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.channel_aggregation = channel_aggregation

    def fit(self, X, y=None):
        """Fit the Matrix Profile detector on time series data.

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
        m = self.window_size

        if n_timestamps < m + 1:
            raise ValueError(
                "Time series length %d is too short for window_size=%d. "
                "Need at least %d timestamps." % (n_timestamps, m, m + 1))

        self._set_n_classes(y)

        # Compute Matrix Profile per channel
        per_channel_ts_scores = []
        for ch in range(n_channels):
            ts = X[:, ch]
            mp = _compute_matrix_profile(ts, m)

            # Map subsequence-level MP scores back to timestamps
            # step=1 since MP computes all subsequences
            scores, valid_mask = map_scores_to_timestamps(
                mp, m, step=1, n_timestamps=n_timestamps,
                aggregation=self.channel_aggregation)
            per_channel_ts_scores.append((scores, valid_mask))

        if n_channels == 1:
            scores, valid_mask = per_channel_ts_scores[0]
        else:
            # Aggregate across channels
            # First, fill NaN positions in each channel with 0 for aggregation
            filled_scores = []
            combined_valid = per_channel_ts_scores[0][1].copy()
            for ch_scores, ch_valid in per_channel_ts_scores:
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
        """Not supported (transductive detector).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "MatrixProfile is a transductive detector and does not support "
            "decision_function on new data. Access decision_scores_ after "
            "fit() for training scores.")

    def predict(self, X, return_confidence=False):
        """Not supported (transductive detector).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "MatrixProfile is a transductive detector and does not support "
            "predict on new data. Access labels_ after fit() for training "
            "labels.")

    def predict_proba(self, X, method='linear', return_confidence=False):
        """Not supported (transductive detector).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "MatrixProfile is a transductive detector and does not support "
            "predict_proba on new data.")

    def predict_confidence(self, X):
        """Not supported (transductive detector).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "MatrixProfile is a transductive detector and does not support "
            "predict_confidence on new data.")
