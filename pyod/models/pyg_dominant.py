# -*- coding: utf-8 -*-
"""DOMINANT: Deep Anomaly Detection on Attributed Networks.

GCN autoencoder that jointly reconstructs the adjacency matrix and
node attributes. Per-node anomaly score = weighted sum of structure
and attribute reconstruction error.

See :cite:`ding2019dominant` for details.

Reference:
    Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019. Deep Anomaly
    Detection on Attributed Networks. In SDM, pp. 594-602.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input


class DOMINANT(BaseDetector):
    """DOMINANT: Deep Anomaly Detection on Attributed Networks.

    GCN encoder maps nodes to embeddings. Structure decoder
    reconstructs adjacency via inner product. Attribute decoder
    reconstructs features via linear layer. Anomaly score per node
    = ``alpha * struct_err + (1 - alpha) * attr_err``.

    This detector is **transductive**.

    Parameters
    ----------
    hidden_dim : int, default=64
        Hidden dimension of GCN layers.

    num_layers : int, default=2
        Number of GCN encoder layers.

    dropout : float, default=0.3
        Dropout rate during training.

    alpha : float, default=0.5
        Weight for structure loss (1 - alpha for attribute loss).

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

    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.3,
                 alpha=0.5, epochs=100, lr=5e-3, contamination=0.1):
        super(DOMINANT, self).__init__(contamination=contamination)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y=None, edge_index=None):
        """Fit the detector on graph data."""
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv

        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        if data.x is None:
            raise ValueError("DOMINANT requires node features (data.x).")

        in_dim = data.x.shape[1]

        model = _DOMINANTModel(
            in_dim, self.hidden_dim, self.num_layers, self.dropout)
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
                self.alpha * struct_loss + (1 - self.alpha) * attr_loss)

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
        raise NotImplementedError(
            "DOMINANT is a transductive detector. Use "
            "decision_scores_ after fit().")

    def predict(self, X):
        raise NotImplementedError(
            "DOMINANT is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X):
        raise NotImplementedError(
            "DOMINANT is a transductive detector.")

    def predict_confidence(self, X):
        raise NotImplementedError(
            "DOMINANT is a transductive detector.")


def _DOMINANTModel(in_dim, hid_dim, num_layers, dropout):
    """Factory: returns a torch.nn.Module for the DOMINANT AE."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_dim, hid_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hid_dim, hid_dim))
            self.attr_decoder = nn.Linear(hid_dim, in_dim)
            self.dropout = dropout

        def forward(self, x, edge_index):
            z = x
            for i, conv in enumerate(self.convs):
                z = conv(z, edge_index)
                if i < len(self.convs) - 1:
                    z = torch.relu(z)
                    z = torch.dropout(
                        z, p=self.dropout, train=self.training)

            a_hat = torch.sigmoid(z @ z.t())
            x_hat = self.attr_decoder(z)
            return a_hat, x_hat

    return _Model()
