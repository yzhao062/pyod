# -*- coding: utf-8 -*-
"""AnomalyTransformer: Transformer-based time series anomaly detector
with association discrepancy.

Implements the Anomaly Transformer (Xu et al., ICLR 2022) which uses
anomaly-attention with series-association and prior-association to detect
anomalies in time series via a minimax optimization strategy.

Reference:
    Xu, J., Wu, H., Wang, J., & Long, M. (2022).
    Anomaly Transformer: Time Series Anomaly Detection with Association
    Discrepancy. In *International Conference on Learning Representations*.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import math
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._ts_utils import validate_ts_input

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class _AnomalyAttention(nn.Module):
    """Single anomaly-attention mechanism.

    Computes two associations:
    - Series-association: standard softmax attention softmax(QK^T / sqrt(d_k))
    - Prior-association: Gaussian kernel softmax(-|i-j|^2 / (2*sigma^2))

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    n_heads : int
        Number of attention heads.
    window_size : int
        Length of the input sequence (for pre-computing distance matrix).
    """

    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable sigma per head (one scalar each), initialised to 1.0
        self.sigma = nn.Parameter(torch.ones(n_heads))

        # Pre-compute squared-distance matrix for the prior: |i - j|^2
        positions = torch.arange(window_size, dtype=torch.float32)
        # Shape: (window_size, window_size)
        dist_sq = (positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2
        self.register_buffer('dist_sq', dist_sq)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, d_model)

        Returns
        -------
        out : Tensor, shape (batch, seq_len, d_model)
        series_assoc : Tensor, shape (batch, n_heads, seq_len, seq_len)
        prior_assoc : Tensor, shape (batch, n_heads, seq_len, seq_len)
        """
        B, L, _ = x.shape

        # Q, K, V projections -> (B, n_heads, L, d_k)
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Series-association: softmax(QK^T / sqrt(d_k))
        scale = math.sqrt(self.d_k)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale
        series_assoc = torch.softmax(attn_logits, dim=-1)  # (B, H, L, L)

        # Prior-association: Gaussian kernel per head
        # sigma clamped to avoid degenerate values
        sigma = torch.clamp(self.sigma, min=1e-4)  # (n_heads,)
        # dist_sq: (L, L) -> expand to (1, n_heads, L, L)
        dist = self.dist_sq[:L, :L].unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        sigma_sq = (sigma ** 2).view(1, self.n_heads, 1, 1)  # (1,H,1,1)
        prior_logits = -dist / (2.0 * sigma_sq)
        prior_assoc = torch.softmax(prior_logits, dim=-1)  # (1, H, L, L)
        prior_assoc = prior_assoc.expand(B, -1, -1, -1)  # (B, H, L, L)

        # Attention output using series-association
        out = torch.matmul(series_assoc, V)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)

        return out, series_assoc, prior_assoc


class _AnomalyTransformerLayer(nn.Module):
    """One encoder layer with anomaly-attention + feedforward + layer norm.

    Parameters
    ----------
    d_model : int
    n_heads : int
    window_size : int
    d_ff : int
        Inner dimensionality of the feedforward network. Defaults to 4*d_model.
    dropout : float
    """

    def __init__(self, d_model, n_heads, window_size, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = _AnomalyAttention(d_model, n_heads, window_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Returns
        -------
        x : Tensor (B, L, d_model)
        series_assoc : Tensor (B, H, L, L)
        prior_assoc : Tensor (B, H, L, L)
        """
        attn_out, series_assoc, prior_assoc = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x, series_assoc, prior_assoc


class _AnomalyTransformerModel(nn.Module):
    """Full Anomaly Transformer model.

    Architecture:
    - Input embedding: 1D convolution (kernel=3, pad=1) projecting
      n_channels -> d_model
    - Learnable positional encoding
    - Stack of anomaly-attention encoder layers
    - Output projection: linear d_model -> n_channels

    Parameters
    ----------
    n_channels : int
    window_size : int
    d_model : int
    n_heads : int
    n_layers : int
    dropout : float
    """

    def __init__(self, n_channels, window_size, d_model=512,
                 n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        # Input embedding via 1D convolution
        self.input_embed = nn.Conv1d(
            in_channels=n_channels, out_channels=d_model,
            kernel_size=3, padding=1
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, window_size, d_model) * 0.02
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            _AnomalyTransformerLayer(d_model, n_heads, window_size,
                                     dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, n_channels)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, window_size, n_channels)

        Returns
        -------
        reconstruction : Tensor, shape (batch, window_size, n_channels)
        associations : list of (series_assoc, prior_assoc) per layer
        """
        B, L, C = x.shape

        # Embed: (B, L, C) -> conv expects (B, C, L)
        h = self.input_embed(x.transpose(1, 2)).transpose(1, 2)  # (B, L, d)

        # Add positional encoding
        h = h + self.pos_encoding[:, :L, :]

        # Encoder layers
        associations = []
        for layer in self.layers:
            h, s_assoc, p_assoc = layer(h)
            associations.append((s_assoc, p_assoc))

        # Output
        reconstruction = self.output_proj(h)  # (B, L, C)
        return reconstruction, associations


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_association_discrepancy(associations, detach_prior=False,
                                     detach_series=False):
    """Compute association discrepancy across layers and heads.

    AssDis = sum over layers of [KL(P||S) + KL(S||P)] / 2
    averaged over heads.

    Parameters
    ----------
    associations : list of (series_assoc, prior_assoc)
    detach_prior : bool
        If True, stop gradient on prior (minimize phase).
    detach_series : bool
        If True, stop gradient on series (maximize phase).

    Returns
    -------
    ass_dis : Tensor, shape (batch, seq_len)
        Association discrepancy per timestamp, summed across layers.
    """
    total = None
    eps = 1e-8

    for s_assoc, p_assoc in associations:
        S = s_assoc  # (B, H, L, L)
        P = p_assoc  # (B, H, L, L)

        if detach_prior:
            P = P.detach()
        if detach_series:
            S = S.detach()

        # KL(P || S) = sum_j P(j) * log(P(j) / S(j))
        kl_ps = (P * (torch.log(P + eps) - torch.log(S + eps))).sum(dim=-1)
        # KL(S || P) = sum_j S(j) * log(S(j) / P(j))
        kl_sp = (S * (torch.log(S + eps) - torch.log(P + eps))).sum(dim=-1)
        # Symmetric KL per head per position: (B, H, L)
        sym_kl = (kl_ps + kl_sp) / 2.0
        # Average over heads: (B, L)
        layer_dis = sym_kl.mean(dim=1)

        if total is None:
            total = layer_dis
        else:
            total = total + layer_dis

    return total


def _create_windows_3d(X, window_size, step=1):
    """Create overlapping windows preserving the channel dimension.

    Parameters
    ----------
    X : np.ndarray, shape (n_timestamps, n_channels)
    window_size : int
    step : int

    Returns
    -------
    windows : np.ndarray, shape (n_windows, window_size, n_channels)
    """
    n_timestamps, n_channels = X.shape
    n_windows = max(0, (n_timestamps - window_size) // step + 1)
    windows = np.empty((n_windows, window_size, n_channels), dtype=np.float64)
    for i in range(n_windows):
        start = i * step
        windows[i] = X[start:start + window_size]
    return windows


def _map_window_scores_to_timestamps(window_scores, window_size, step,
                                     n_timestamps):
    """Map per-window per-timestamp scores to full time series scores.

    For each window, per-position scores are assigned to the corresponding
    global timestamp. Overlapping positions are averaged.

    Parameters
    ----------
    window_scores : np.ndarray, shape (n_windows, window_size)
    window_size : int
    step : int
    n_timestamps : int

    Returns
    -------
    scores : np.ndarray, shape (n_timestamps,)
    valid_mask : np.ndarray, shape (n_timestamps,), dtype=bool
    """
    accum = np.zeros(n_timestamps, dtype=np.float64)
    counts = np.zeros(n_timestamps, dtype=np.float64)

    for i in range(window_scores.shape[0]):
        start = i * step
        end = min(start + window_size, n_timestamps)
        length = end - start
        accum[start:end] += window_scores[i, :length]
        counts[start:end] += 1.0

    valid_mask = counts > 0
    scores = np.zeros(n_timestamps, dtype=np.float64)
    scores[valid_mask] = accum[valid_mask] / counts[valid_mask]
    return scores, valid_mask


def _score_windows(model, windows_tensor, device, window_size):
    """Compute anomaly scores for a batch of windows.

    Score(t) = softmax(-AssDis(t)) * ||x_t - x_hat_t||^2

    Parameters
    ----------
    model : _AnomalyTransformerModel
    windows_tensor : Tensor, shape (n_windows, window_size, n_channels)
    device : torch.device
    window_size : int

    Returns
    -------
    scores : np.ndarray, shape (n_windows, window_size)
    """
    model.eval()
    all_scores = []

    batch_size = 64
    n_windows = windows_tensor.shape[0]

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            batch = windows_tensor[start:start + batch_size].to(device)
            recon, associations = model(batch)

            # Reconstruction error per timestamp: (B, L)
            recon_err = ((batch - recon) ** 2).mean(dim=-1)

            # Association discrepancy: (B, L)
            ass_dis = _compute_association_discrepancy(associations)

            # Anomaly score: softmax(-AssDis) * recon_err
            weights = torch.softmax(-ass_dis, dim=-1)  # (B, L)
            score = weights * recon_err  # (B, L)

            all_scores.append(score.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class AnomalyTransformer(BaseDetector):
    """Anomaly Transformer for time series anomaly detection.

    Implements the Anomaly Transformer (Xu et al., ICLR 2022), a
    Transformer-based architecture with anomaly-attention that computes
    series-association and prior-association to detect anomalies via
    association discrepancy and minimax training.

    Parameters
    ----------
    window_size : int, optional (default=100)
        Size of the sliding window.

    d_model : int, optional (default=512)
        Dimensionality of the Transformer model.

    n_heads : int, optional (default=8)
        Number of attention heads. Must divide d_model evenly.

    n_layers : int, optional (default=3)
        Number of Transformer encoder layers.

    epochs : int, optional (default=10)
        Number of training epochs.

    lr : float, optional (default=1e-4)
        Learning rate.

    batch_size : int, optional (default=32)
        Training batch size.

    lambda_ : float, optional (default=3.0)
        Weight for the association discrepancy term in the loss.

    contamination : float, optional (default=0.1)
        Expected proportion of outliers. Must be in (0, 0.5].

    step : int, optional (default=1)
        Step size between consecutive windows.

    dropout : float, optional (default=0.1)
        Dropout rate in the Transformer.

    device : str, optional (default='auto')
        Device to use ('cpu', 'cuda', or 'auto').

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
        Outlier scores of the training data. Higher is more abnormal.

    threshold_ : float
        Score threshold based on ``contamination``.

    labels_ : numpy array of shape (n_timestamps,)
        Binary labels of training data (0: inlier, 1: outlier).

    Examples
    --------
    >>> from pyod.models.ts_anomaly_transformer import AnomalyTransformer
    >>> import numpy as np
    >>> X_train = np.random.randn(500)
    >>> clf = AnomalyTransformer(window_size=50, d_model=32, n_heads=2,
    ...                           n_layers=1, epochs=2)
    >>> clf.fit(X_train)
    >>> scores = clf.decision_function(np.random.randn(200))

    References
    ----------
    .. [1] Xu, J., Wu, H., Wang, J., & Long, M. (2022).
       Anomaly Transformer: Time Series Anomaly Detection with Association
       Discrepancy. ICLR 2022.
    """

    def __init__(self, window_size=100, d_model=512, n_heads=8, n_layers=3,
                 epochs=10, lr=1e-4, batch_size=32, lambda_=3.0,
                 contamination=0.1, step=1, dropout=0.1, device='auto'):
        super(AnomalyTransformer, self).__init__(contamination=contamination)

        if not _TORCH_AVAILABLE:
            raise ImportError(
                "AnomalyTransformer requires PyTorch. "
                "Install it with: pip install torch"
            )

        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.step = step
        self.dropout = dropout
        self.device = device

    def _resolve_device(self):
        """Resolve the device string to a torch.device."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def fit(self, X, y=None):
        """Fit the Anomaly Transformer on time series data.

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

        if n_timestamps < self.window_size:
            raise ValueError(
                "Time series length %d is shorter than window_size %d"
                % (n_timestamps, self.window_size))

        self._set_n_classes(y)

        device = self._resolve_device()

        # Create overlapping windows: (n_windows, window_size, n_channels)
        windows = _create_windows_3d(X, self.window_size, self.step)

        # Build model
        self.model_ = _AnomalyTransformerModel(
            n_channels=n_channels,
            window_size=self.window_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(device)

        # Training
        self._train(windows, device)

        # Compute anomaly scores on training data
        windows_tensor = torch.tensor(windows, dtype=torch.float32)
        window_scores = _score_windows(
            self.model_, windows_tensor, device, self.window_size
        )

        # Map window-level per-timestamp scores back to the full time series
        scores, valid_mask = _map_window_scores_to_timestamps(
            window_scores, self.window_size, self.step, n_timestamps
        )

        # Process using masked-score workflow
        valid_scores = scores[valid_mask]
        self.decision_scores_ = valid_scores
        self._process_decision_scores()

        # Reconstruct full-length arrays
        full_scores = scores.copy()
        full_scores[~valid_mask] = self.threshold_
        full_labels = (full_scores > self.threshold_).astype(int)
        self.decision_scores_ = full_scores
        self.labels_ = full_labels

        # Store metadata for decision_function
        self.n_channels_ = n_channels
        return self

    def _train(self, windows, device):
        """Run minimax training loop.

        Parameters
        ----------
        windows : np.ndarray, shape (n_windows, window_size, n_channels)
        device : torch.device
        """
        model = self.model_
        model.train()

        dataset = TensorDataset(
            torch.tensor(windows, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for (batch,) in loader:
                batch = batch.to(device)

                # ---- Minimize phase ----
                # Minimize: L = recon_loss - lambda * AssDis
                # Detach prior (stop gradient on P)
                optimizer.zero_grad()
                recon, associations = model(batch)
                recon_loss = F.mse_loss(recon, batch)
                ass_dis_min = _compute_association_discrepancy(
                    associations, detach_prior=True
                )
                loss_min = recon_loss - self.lambda_ * ass_dis_min.mean()
                loss_min.backward()
                optimizer.step()

                # ---- Maximize phase ----
                # Maximize AssDis only. Detach series (stop gradient on S).
                optimizer.zero_grad()
                recon, associations = model(batch)
                ass_dis_max = _compute_association_discrepancy(
                    associations, detach_series=True
                )
                loss_max = -ass_dis_max.mean()
                loss_max.backward()
                optimizer.step()

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
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_',
                                'model_', 'n_channels_'])

        X = validate_ts_input(X)
        n_timestamps, n_channels = X.shape

        if n_channels != self.n_channels_:
            raise ValueError(
                "Expected %d channels, got %d" %
                (self.n_channels_, n_channels))

        if n_timestamps < self.window_size:
            raise ValueError(
                "Time series length %d is shorter than window_size %d"
                % (n_timestamps, self.window_size))

        device = self._resolve_device()

        # Create windows and score
        windows = _create_windows_3d(X, self.window_size, self.step)
        windows_tensor = torch.tensor(windows, dtype=torch.float32)
        window_scores = _score_windows(
            self.model_, windows_tensor, device, self.window_size
        )

        # Map back to timestamps
        scores, valid_mask = _map_window_scores_to_timestamps(
            window_scores, self.window_size, self.step, n_timestamps
        )

        # Fill invalid positions with training threshold
        scores[~valid_mask] = self.threshold_
        return scores
