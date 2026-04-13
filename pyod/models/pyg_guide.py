# -*- coding: utf-8 -*-
"""GUIDE: Higher-order Structure Based Anomaly Detection.

Dual GCN autoencoders: one on the original adjacency, one on a
motif (triangle) adjacency. Anomaly score = weighted sum of
reconstruction errors from both views.

See :cite:`yuan2021guide` for details.

Reference:
    Yuan, X., Zhou, N., Yu, S., Huang, H., Chen, Z. and Xia, F.,
    2021. Higher-order Structure Based Anomaly Detection on Attributed
    Networks. In IEEE BigData, pp. 2691-2700.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input, to_sparse_adj


class GUIDE(BaseDetector):
    """GUIDE: Higher-order Structure Based Anomaly Detection.

    Constructs a motif adjacency from triangle participation
    (binarized in v1: edges in at least one triangle) and runs
    two GCN autoencoders in parallel. Score = ``alpha * err_orig
    + (1 - alpha) * err_motif``.

    This detector is **transductive**.

    Parameters
    ----------
    hidden_dim : int, default=64
        Hidden dimension of GCN layers.

    num_layers : int, default=2
        Number of GCN encoder layers.

    alpha : float, default=0.5
        Weight for original-graph reconstruction error.

    dropout : float, default=0.3
        Dropout rate.

    epochs : int, default=100
        Training epochs.

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

    def __init__(self, hidden_dim=64, num_layers=2, alpha=0.5,
                 dropout=0.3, epochs=100, lr=5e-3,
                 contamination=0.1):
        super(GUIDE, self).__init__(contamination=contamination)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        from torch_geometric.nn import GCNConv

        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        if data.x is None:
            raise ValueError("GUIDE requires node features (data.x).")

        in_dim = data.x.shape[1]

        ei = data.edge_index
        ei_np = ei.cpu().numpy()

        # Build motif adjacency (triangle counts per edge)
        adj_sp = to_sparse_adj(ei_np, n_nodes)
        motif_adj = adj_sp.dot(adj_sp).multiply(adj_sp)
        motif_coo = motif_adj.tocoo()
        if motif_coo.nnz == 0:
            raise ValueError(
                "GUIDE requires higher-order structures (triangles) "
                "in the graph. This graph has no triangles. Use "
                "DOMINANT or CoLA instead.")
        ei_motif = torch.LongTensor(
            np.array([motif_coo.row, motif_coo.col]))

        x = data.x

        # Dense adjacencies for loss
        adj_dense = torch.zeros(n_nodes, n_nodes)
        adj_dense[ei[0], ei[1]] = 1.0

        motif_dense = torch.zeros(n_nodes, n_nodes)
        motif_dense[ei_motif[0], ei_motif[1]] = 1.0

        model = _GUIDEModel(
            in_dim, self.hidden_dim, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            x_hat_o, x_hat_m, a_hat_o, a_hat_m = model(
                x, ei, ei_motif)

            # Original graph errors
            s_err_o = torch.sum(
                (adj_dense - a_hat_o) ** 2, dim=1)
            a_err_o = torch.sum((x - x_hat_o) ** 2, dim=1)
            err_orig = s_err_o + a_err_o

            # Motif graph errors
            s_err_m = torch.sum(
                (motif_dense - a_hat_m) ** 2, dim=1)
            a_err_m = torch.sum((x - x_hat_m) ** 2, dim=1)
            err_motif = s_err_m + a_err_m

            loss = torch.mean(
                self.alpha * err_orig
                + (1 - self.alpha) * err_motif)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_hat_o, x_hat_m, a_hat_o, a_hat_m = model(
                x, ei, ei_motif)

            s_err_o = torch.sum(
                (adj_dense - a_hat_o) ** 2, dim=1)
            a_err_o = torch.sum((x - x_hat_o) ** 2, dim=1)
            err_orig = s_err_o + a_err_o

            s_err_m = torch.sum(
                (motif_dense - a_hat_m) ** 2, dim=1)
            a_err_m = torch.sum((x - x_hat_m) ** 2, dim=1)
            err_motif = s_err_m + a_err_m

            scores = (self.alpha * err_orig
                      + (1 - self.alpha) * err_motif)

        self.decision_scores_ = scores.cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "GUIDE is a transductive detector. Use decision_scores_ "
            "after fit().")

    def predict(self, X, return_confidence=False):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "GUIDE is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X, method="linear", return_confidence=False):
        """Not supported (transductive detector)."""
        raise NotImplementedError("GUIDE is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError("GUIDE is a transductive detector.")


def _GUIDEModel(in_dim, hid_dim, num_layers, dropout):
    """Factory: returns torch.nn.Module for GUIDE dual AE."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Original-graph encoder
            self.enc_orig = nn.ModuleList()
            self.enc_orig.append(GCNConv(in_dim, hid_dim))
            for _ in range(num_layers - 1):
                self.enc_orig.append(GCNConv(hid_dim, hid_dim))

            # Motif-graph encoder
            self.enc_motif = nn.ModuleList()
            self.enc_motif.append(GCNConv(in_dim, hid_dim))
            for _ in range(num_layers - 1):
                self.enc_motif.append(GCNConv(hid_dim, hid_dim))

            self.dec_attr_orig = nn.Linear(hid_dim, in_dim)
            self.dec_attr_motif = nn.Linear(hid_dim, in_dim)
            self._dropout = dropout

        def _encode(self, x, edge_index, encoder):
            z = x
            for i, conv in enumerate(encoder):
                z = conv(z, edge_index)
                if i < len(encoder) - 1:
                    z = torch.relu(z)
                    z = torch.dropout(
                        z, p=self._dropout, train=self.training)
            return z

        def forward(self, x, ei_orig, ei_motif):
            z_o = self._encode(x, ei_orig, self.enc_orig)
            z_m = self._encode(x, ei_motif, self.enc_motif)

            x_hat_o = self.dec_attr_orig(z_o)
            x_hat_m = self.dec_attr_motif(z_m)

            a_hat_o = torch.sigmoid(z_o @ z_o.t())
            a_hat_m = torch.sigmoid(z_m @ z_m.t())

            return x_hat_o, x_hat_m, a_hat_o, a_hat_m

    return _Model()
