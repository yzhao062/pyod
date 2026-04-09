# -*- coding: utf-8 -*-
"""TimeSeriesOD: Windowed bridge that wraps any PyOD detector for
time series anomaly detection.

Creates sliding windows from a time series, runs any PyOD detector on
the resulting window matrix, and maps anomaly scores back to individual
timestamps.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import validate_ts_input, sliding_windows, map_scores_to_timestamps
from .embedding import _DETECTOR_SHORTCUTS, resolve_detector


class TimeSeriesOD(BaseDetector):
    """Windowed bridge that wraps any PyOD detector for time series
    anomaly detection.

    Takes a time series, creates sliding windows, runs any PyOD detector
    on the window matrix, and maps scores back to timestamps.

    Parameters
    ----------
    detector : str or BaseDetector, optional (default='IForest')
        Any PyOD detector. String resolves to a default-configured
        instance via the shortcut registry. If a BaseDetector instance
        is passed, it will be cloned.

    window_size : int, optional (default=50)
        Size of the sliding window.

    step : int, optional (default=1)
        Step size between consecutive windows.

    score_aggregation : str, optional (default='max')
        How to aggregate window-level scores to timestamp-level scores.
        One of 'max' or 'mean'.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers in the dataset.
        Must be in (0, 0.5].

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data. Higher is more abnormal.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels of training data (0: inlier, 1: outlier).

    detector_ : BaseDetector
        The resolved and fitted inner detector instance.

    Examples
    --------
    >>> from pyod.models.ts_od import TimeSeriesOD
    >>> import numpy as np
    >>> X_train = np.random.randn(500)
    >>> clf = TimeSeriesOD(detector='IForest', window_size=20)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(np.random.randn(200))
    """

    def __init__(self, detector='IForest', window_size=50, step=1,
                 score_aggregation='max', contamination=0.1):
        super(TimeSeriesOD, self).__init__(contamination=contamination)
        self.detector = detector
        self.window_size = window_size
        self.step = step
        self.score_aggregation = score_aggregation

    def _get_min_length(self):
        """Return the minimum time series length required.

        Returns
        -------
        min_length : int
        """
        return self.window_size

    def fit(self, X, y=None):
        """Fit detector on time series data.

        Validates the input, creates sliding windows, fits the inner
        detector on the window matrix, and maps scores back to
        timestamps using the masked-score workflow.

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
                "required length %d (window_size=%d)"
                % (n_timestamps, min_len, self.window_size))

        self._set_n_classes(y)

        # Resolve the inner detector
        self.detector_ = resolve_detector(self.detector, self.contamination)

        # Create sliding windows and fit inner detector
        windows = sliding_windows(X, self.window_size, self.step)
        self.detector_.fit(windows)
        window_scores = self.detector_.decision_scores_

        # Map window scores back to timestamps
        scores, valid_mask = map_scores_to_timestamps(
            window_scores, self.window_size, self.step,
            n_timestamps, aggregation=self.score_aggregation)

        # Process on valid subset to compute threshold
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
            Anomaly scores. Higher is more abnormal.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        X = validate_ts_input(X)
        n_timestamps = X.shape[0]

        # Create sliding windows and score with fitted detector
        windows = sliding_windows(X, self.window_size, self.step)
        window_scores = self.detector_.decision_function(windows)

        # Map back to timestamps
        scores, valid_mask = map_scores_to_timestamps(
            window_scores, self.window_size, self.step,
            n_timestamps, aggregation=self.score_aggregation)

        # Fill invalid positions with threshold
        scores[~valid_mask] = self.threshold_
        return scores
