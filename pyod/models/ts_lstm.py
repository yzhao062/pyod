# -*- coding: utf-8 -*-
"""LSTMAD: LSTM-based time series anomaly detection using prediction
error with Mahalanobis distance scoring.

Simplified PyOD adaptation of Malhotra et al., ESANN 2015.
Single-step prediction (horizon=1). Error vector per timestamp has
n_channels dimensions. Anomaly score = Mahalanobis distance of
prediction errors from a fitted multivariate Gaussian.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import validate_ts_input


class LSTMAD(BaseDetector):
    """LSTM-based anomaly detector for time series.

    Trains a stacked LSTM to predict the next timestep, then scores
    each timestamp by the Mahalanobis distance of its prediction error
    from a multivariate Gaussian fitted on training errors.

    Parameters
    ----------
    window_size : int, optional (default=50)
        Number of past timesteps used as input context for prediction.

    hidden_size : int, optional (default=64)
        Number of hidden units in each LSTM layer.

    n_layers : int, optional (default=2)
        Number of stacked LSTM layers.

    epochs : int, optional (default=50)
        Number of training epochs.

    lr : float, optional (default=1e-3)
        Learning rate for Adam optimizer.

    batch_size : int, optional (default=32)
        Mini-batch size for training.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers.  Must be in (0, 0.5].

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data. Higher is more abnormal.
        First ``window_size`` timestamps are filled with ``threshold_``
        (no lookback available).

    threshold_ : float
        Score threshold derived from ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels (0: inlier, 1: outlier).

    Examples
    --------
    >>> from pyod.models.ts_lstm import LSTMAD
    >>> import numpy as np
    >>> X_train = np.random.randn(500)
    >>> clf = LSTMAD(window_size=20, epochs=5)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(np.random.randn(200))

    References
    ----------
    .. [1] Malhotra, P., Vig, L., Shroff, G. and Agarwal, P., 2015.
       Long short term memory networks for anomaly detection in time
       series. In *Proceedings of the European Symposium on Artificial
       Neural Networks (ESANN)* (p. 89).
    """

    def __init__(self, window_size=50, hidden_size=64, n_layers=2,
                 epochs=50, lr=1e-3, batch_size=32, contamination=0.1):
        super(LSTMAD, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def _get_min_length(self):
        """Return the minimum time series length required.

        Need enough timestamps beyond the window to produce several
        error vectors for a stable covariance estimate.

        Returns
        -------
        min_length : int
        """
        return self.window_size + 10

    @staticmethod
    def _build_pairs(X, window_size):
        """Create (input, target) sliding-window pairs.

        Parameters
        ----------
        X : np.ndarray of shape (n_timestamps, n_channels)
            Time series data (2-D, already validated).

        window_size : int
            Lookback window length.

        Returns
        -------
        inputs : np.ndarray of shape (n_pairs, window_size, n_channels)
        targets : np.ndarray of shape (n_pairs, n_channels)
        """
        n_timestamps, n_channels = X.shape
        n_pairs = n_timestamps - window_size
        inputs = np.empty((n_pairs, window_size, n_channels))
        targets = np.empty((n_pairs, n_channels))
        for i in range(n_pairs):
            inputs[i] = X[i:i + window_size]
            targets[i] = X[i + window_size]
        return inputs, targets

    def _train_model(self, inputs, targets, n_channels):
        """Build and train the LSTM model using PyTorch.

        Parameters
        ----------
        inputs : np.ndarray of shape (n_pairs, window_size, n_channels)
        targets : np.ndarray of shape (n_pairs, n_channels)
        n_channels : int

        Returns
        -------
        model : _LSTMModel
            Trained PyTorch model (on CPU, eval mode).
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        model = _LSTMModel(n_channels, self.hidden_size, self.n_layers)
        device = torch.device('cpu')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(inputs, dtype=torch.float32)
        y_tensor = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        return model

    def _predict_model(self, model, inputs):
        """Run the trained LSTM model on input windows.

        Parameters
        ----------
        model : _LSTMModel
        inputs : np.ndarray of shape (n_pairs, window_size, n_channels)

        Returns
        -------
        predictions : np.ndarray of shape (n_pairs, n_channels)
        """
        import torch

        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(inputs, dtype=torch.float32)
            preds = model(X_tensor).cpu().numpy()
        return preds

    @staticmethod
    def _mahalanobis_scores(errors):
        """Compute Mahalanobis distance scores from prediction errors.

        Parameters
        ----------
        errors : np.ndarray of shape (n_valid, n_channels)
            Prediction errors (actual - predicted).

        Returns
        -------
        scores : np.ndarray of shape (n_valid,)
        mu : np.ndarray of shape (n_channels,)
        cov_inv : np.ndarray of shape (n_channels, n_channels)
        """
        n_channels = errors.shape[1]
        mu = np.mean(errors, axis=0)

        if n_channels == 1:
            # Univariate case: np.cov returns a scalar (0-D array)
            var = np.var(errors, ddof=1) + 1e-6
            cov_inv = np.array([[1.0 / var]])
        else:
            cov = np.cov(errors.T) + 1e-6 * np.eye(n_channels)
            cov_inv = np.linalg.inv(cov)

        diff = errors - mu
        scores = np.sum(diff @ cov_inv * diff, axis=1)
        return scores, mu, cov_inv

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
        n_timestamps, n_channels = X.shape
        min_len = self._get_min_length()
        if n_timestamps < min_len:
            raise ValueError(
                "Time series length %d is shorter than minimum "
                "required length %d (window_size=%d)"
                % (n_timestamps, min_len, self.window_size))

        self._set_n_classes(y)

        # Store n_channels for decision_function
        self.n_channels_ = n_channels

        # Step 1: Create sliding (input, target) pairs
        inputs, targets = self._build_pairs(X, self.window_size)

        # Step 2: Build and train LSTM
        self.model_ = self._train_model(inputs, targets, n_channels)

        # Step 3: Compute prediction errors on training data
        predictions = self._predict_model(self.model_, inputs)
        errors = targets - predictions  # shape (n_valid, n_channels)

        # Step 4-5: Fit Gaussian and compute Mahalanobis scores
        valid_scores, self.error_mu_, self.error_cov_inv_ = (
            self._mahalanobis_scores(errors))

        # Step 6-7: Causal boundary -- first window_size timestamps have
        # no lookback, so they are invalid.  Use masked-score workflow:
        # compute threshold on valid scores, then fill invalids.
        self.decision_scores_ = valid_scores
        self._process_decision_scores()

        # Reconstruct full-length score array
        full_scores = np.empty(n_timestamps)
        full_scores[:self.window_size] = self.threshold_
        full_scores[self.window_size:] = valid_scores

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
            Mahalanobis-distance anomaly scores. Higher is more abnormal.
            First ``window_size`` timestamps are filled with ``threshold_``.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        X = validate_ts_input(X)
        n_timestamps = X.shape[0]

        # Create pairs and predict
        inputs, targets = self._build_pairs(X, self.window_size)
        predictions = self._predict_model(self.model_, inputs)
        errors = targets - predictions

        # Mahalanobis distance using training Gaussian parameters
        diff = errors - self.error_mu_
        valid_scores = np.sum(diff @ self.error_cov_inv_ * diff, axis=1)

        # Fill causal boundary
        full_scores = np.empty(n_timestamps)
        full_scores[:self.window_size] = self.threshold_
        full_scores[self.window_size:] = valid_scores

        return full_scores


class _LSTMModel:
    """Thin wrapper around a PyTorch LSTM + Linear model.

    Imports torch lazily so PyTorch remains an optional dependency.
    """

    def __init__(self, n_features, hidden_size, n_layers):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden_size, n_layers,
                                    batch_first=True)
                self.linear = nn.Linear(hidden_size, n_features)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.linear(out[:, -1, :])

        self._net = _Net()

    # Delegate common methods to the inner nn.Module
    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)

    def parameters(self):
        return self._net.parameters()

    def to(self, device):
        self._net = self._net.to(device)
        return self

    def train(self):
        self._net.train()

    def eval(self):
        self._net.eval()
