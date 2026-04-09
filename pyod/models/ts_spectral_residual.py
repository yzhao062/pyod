# -*- coding: utf-8 -*-
"""SpectralResidual: FFT-based saliency for time series anomaly detection.

Implements the spectral residual (SR) saliency computation from
Ren et al., "Time-Series Anomaly Detection Service at Microsoft", KDD 2019.

Only the SR saliency step is implemented (not the full service pipeline
with CNN post-processing).
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import validate_ts_input, aggregate_channel_scores


class SpectralResidual(BaseDetector):
    """Spectral Residual anomaly detector for time series.

    Computes a saliency map via the spectral residual of the
    log-amplitude spectrum.  This is a **dense** method that produces
    one anomaly score per timestamp with no gaps.

    Parameters
    ----------
    score_window : int, optional (default=3)
        Size of the uniform averaging filter applied to the
        log-amplitude spectrum.  Must be >= 1.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers.  Must be in (0, 0.5].

    channel_aggregation : str, optional (default='max')
        How to aggregate per-channel saliency scores for multivariate
        input.  One of ``'max'`` or ``'mean'``.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Saliency-based outlier scores of the training data.
        Higher is more abnormal.

    threshold_ : float
        Score threshold derived from ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels (0: inlier, 1: outlier).

    Examples
    --------
    >>> from pyod.models.ts_spectral_residual import SpectralResidual
    >>> import numpy as np
    >>> X_train = np.random.randn(500)
    >>> clf = SpectralResidual(contamination=0.1)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(np.random.randn(200))

    References
    ----------
    .. [1] Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X.,
       Xing, T., Yang, M., Tong, J. and Zhang, Q., 2019.
       Time-series anomaly detection service at Microsoft.
       In *Proceedings of the 25th ACM SIGKDD International Conference
       on Knowledge Discovery & Data Mining* (pp. 3009-3017).
    """

    def __init__(self, score_window=3, contamination=0.1,
                 channel_aggregation='max'):
        super(SpectralResidual, self).__init__(contamination=contamination)
        self.score_window = score_window
        self.channel_aggregation = channel_aggregation

    def _get_min_length(self):
        """Return the minimum time series length required.

        Returns
        -------
        min_length : int
        """
        return max(self.score_window, 2)

    @staticmethod
    def _spectral_residual(x, score_window):
        """Compute spectral residual saliency for a 1-D signal.

        Parameters
        ----------
        x : np.ndarray of shape (n_timestamps,)
            Univariate time series.

        score_window : int
            Width of the uniform averaging kernel.

        Returns
        -------
        saliency : np.ndarray of shape (n_timestamps,)
            Non-negative saliency values.
        """
        # Step 1 -- FFT
        F = np.fft.fft(x)

        # Step 2 -- log amplitude spectrum
        A = np.log(np.abs(F) + 1e-10)

        # Step 3 -- phase spectrum
        P = np.angle(F)

        # Step 4 -- smooth log amplitude with uniform averaging filter
        q = score_window
        kernel = np.ones(q) / q
        A_smooth = np.convolve(A, kernel, mode='same')

        # Step 5 -- spectral residual
        R = A - A_smooth

        # Step 6 -- reconstruct with residual amplitude + original phase
        S = np.exp(R + 1j * P)

        # Step 7 -- inverse FFT => saliency map
        saliency = np.abs(np.fft.ifft(S))

        return saliency

    def _compute_scores(self, X):
        """Compute per-timestamp anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_timestamps, n_channels)
            Validated time series data (2-D).

        Returns
        -------
        scores : np.ndarray of shape (n_timestamps,)
        """
        n_channels = X.shape[1]

        if n_channels == 1:
            return self._spectral_residual(X[:, 0], self.score_window)

        # Multivariate: per-channel SR, then aggregate
        per_channel = []
        for c in range(n_channels):
            per_channel.append(
                self._spectral_residual(X[:, c], self.score_window))
        return aggregate_channel_scores(per_channel,
                                        method=self.channel_aggregation)

    def fit(self, X, y=None):
        """Fit detector on time series data.

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
        n_timestamps = X.shape[0]
        min_len = self._get_min_length()
        if n_timestamps < min_len:
            raise ValueError(
                "Time series length %d is shorter than minimum "
                "required length %d" % (n_timestamps, min_len))

        self._set_n_classes(y)

        # Compute saliency scores -- dense, so valid_mask is all True
        scores = self._compute_scores(X)

        self.decision_scores_ = scores
        self._process_decision_scores()
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
            Saliency-based anomaly scores.  Higher is more abnormal.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        X = validate_ts_input(X)
        return self._compute_scores(X)
