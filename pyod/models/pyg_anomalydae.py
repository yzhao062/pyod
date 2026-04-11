# -*- coding: utf-8 -*-
"""AnomalyDAE: Dual Autoencoder for Anomaly Detection.

Attention-based structure encoder (GAT) and MLP attribute encoder,
with separate decoders for each modality. Per-node anomaly score
is the weighted sum of structure and attribute reconstruction error.

See :cite:`fan2020anomalydae` for details.

Reference:
    Fan, H., Zhang, F. and Li, Z., 2020. AnomalyDAE: Dual Autoencoder
    for Anomaly Detection on Attributed Networks. In CIKM, pp. 747-756.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input


class AnomalyDAE(BaseDetector):
    """AnomalyDAE: Dual Autoencoder for Anomaly Detection.

    Uses GATConv for structure encoding and an MLP for attribute
    encoding. Reconstructs adjacency via inner product and
    attributes via an MLP decoder.

    This detector is **transductive**.

    Parameters
    ----------
    embed_dim : int, default=64
        Embedding dimension.

    num_heads : int, default=4
        Number of attention heads in GAT.

    alpha : float, default=0.5
        Weight for structure loss.

    dropout : float, default=0.3
        Dropout rate.

    epochs : int, default=100
        Number of training epochs.

    lr : float, default=5e-3
        Learning rate.

    contamination : float, default=0.1
        Expected proportion of anomalies.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_nodes,)
    labels_ : numpy array of shape (n_nodes,)
    threshold_ : float
    """

    def __init__(self, embed_dim=64, num_heads=4, alpha=0.5,
                 dropout=0.3, epochs=100, lr=5e-3,
                 contamination=0.1):
        super(AnomalyDAE, self).__init__(contamination=contamination)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y=None, edge_index=None):
        """Fit the detector on graph data.

        Parameters
        ----------
        X : Data or array-like
        y : ignored
        edge_index : array-like or None

        Returns
        -------
        self
        """
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GATConv

        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        if data.x is None:
            raise ValueError(
                "AnomalyDAE requires node features (data.x).")

        in_dim = data.x.shape[1]

        model = _AnomalyDAEModel(
            in_dim, self.embed_dim, self.num_heads, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        x = data.x
        ei = data.edge_index

        adj = torch.zeros(n_nodes, n_nodes)
        adj[ei[0], ei[1]] = 1.0

        model.train()
        for epoch in range(self.epochs):
            a_hat, x_hat = model(x, ei)
            struct_loss = torch.sum((adj - a_hat) ** 2, dim=1)
            attr_loss = torch.sum((x - x_hat) ** 2, dim=1)
            loss = torch.mean(
                self.alpha * struct_loss
                + (1 - self.alpha) * attr_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            a_hat, x_hat = model(x, ei)
            struct_err = torch.sum((adj - a_hat) ** 2, dim=1)
            attr_err = torch.sum((x - x_hat) ** 2, dim=1)
            scores = (self.alpha * struct_err
                      + (1 - self.alpha) * attr_err)

        self.decision_scores_ = scores.cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "AnomalyDAE is a transductive detector. Use "
            "decision_scores_ after fit().")

    def predict(self, X, return_confidence=False):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "AnomalyDAE is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X, method="linear", return_confidence=False):
        """Not supported (transductive detector)."""
        raise NotImplementedError("AnomalyDAE is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError("AnomalyDAE is a transductive detector.")


def _AnomalyDAEModel(in_dim, embed_dim, num_heads, dropout):
    """Factory: returns a torch.nn.Module for the AnomalyDAE."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Structure encoder: GAT
            self.gat = GATConv(
                in_dim, embed_dim, heads=num_heads, dropout=dropout,
                concat=False)
            # Attribute encoder: MLP
            self.attr_encoder = nn.Sequential(
                nn.Linear(in_dim, embed_dim),
                nn.ReLU(),
            )
            # Attribute decoder
            self.attr_decoder = nn.Linear(embed_dim, in_dim)

        def forward(self, x, edge_index):
            # Structure embedding (n_nodes, embed_dim)
            z_struct = self.gat(x, edge_index)
            # Attribute embedding
            z_attr = self.attr_encoder(x)
            # Combined
            z = (z_struct + z_attr) / 2.0
            # Reconstruct
            a_hat = torch.sigmoid(z @ z.t())
            x_hat = self.attr_decoder(z)
            return a_hat, x_hat

    return _Model()
