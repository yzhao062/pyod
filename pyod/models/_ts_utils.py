# pyod/models/_ts_utils.py
# -*- coding: utf-8 -*-
"""Shared utilities for time series anomaly detection models."""

import numpy as np


def validate_ts_input(X):
    """Validate and reshape time series input.

    Parameters
    ----------
    X : array-like
        Time series data. 1D (n_timestamps,) or 2D (n_timestamps, n_channels).

    Returns
    -------
    X : np.ndarray of shape (n_timestamps, n_channels)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("Expected 1D or 2D input, got %dD" % X.ndim)
    return X


def sliding_windows(X, window_size, step=1):
    """Extract sliding windows from a time series.

    Parameters
    ----------
    X : np.ndarray of shape (n_timestamps, n_channels)
    window_size : int
    step : int

    Returns
    -------
    windows : np.ndarray of shape (n_windows, window_size * n_channels)
    """
    n_timestamps, n_channels = X.shape
    n_windows = max(0, (n_timestamps - window_size) // step + 1)
    windows = np.empty((n_windows, window_size * n_channels))
    for i in range(n_windows):
        start = i * step
        windows[i] = X[start:start + window_size].ravel()
    return windows


def map_scores_to_timestamps(window_scores, window_size, step,
                              n_timestamps, aggregation='max'):
    """Map window-level scores back to per-timestamp scores.

    Parameters
    ----------
    window_scores : np.ndarray of shape (n_windows,)
    window_size : int
    step : int
    n_timestamps : int
    aggregation : str, 'max' or 'mean'

    Returns
    -------
    scores : np.ndarray of shape (n_timestamps,)
    valid_mask : np.ndarray of shape (n_timestamps,), dtype=bool
    """
    scores = np.full(n_timestamps, np.nan)
    counts = np.zeros(n_timestamps)

    for i, score in enumerate(window_scores):
        start = i * step
        end = min(start + window_size, n_timestamps)
        if aggregation == 'max':
            for j in range(start, end):
                if np.isnan(scores[j]) or score > scores[j]:
                    scores[j] = score
        else:  # mean
            for j in range(start, end):
                if np.isnan(scores[j]):
                    scores[j] = score
                else:
                    scores[j] += score
                counts[j] += 1

    if aggregation == 'mean':
        mask = counts > 0
        scores[mask] /= counts[mask]

    valid_mask = ~np.isnan(scores)
    return scores, valid_mask


def aggregate_channel_scores(per_channel_scores, method='max'):
    """Aggregate per-channel anomaly scores.

    Z-normalizes each channel before aggregation to prevent
    high-variance channels from dominating.

    Parameters
    ----------
    per_channel_scores : list of np.ndarray, each shape (n_timestamps,)
    method : str, 'max' or 'mean'

    Returns
    -------
    scores : np.ndarray of shape (n_timestamps,)
    """
    normalized = []
    for ch_scores in per_channel_scores:
        mu = np.mean(ch_scores)
        sigma = np.std(ch_scores)
        if sigma > 0:
            normalized.append((ch_scores - mu) / sigma)
        else:
            normalized.append(ch_scores - mu)

    stacked = np.column_stack(normalized)
    if method == 'max':
        return np.max(stacked, axis=1)
    else:
        return np.mean(stacked, axis=1)
