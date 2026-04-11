# -*- coding: utf-8 -*-
"""CoLA: Contrastive Self-Supervised Learning for Anomaly Detection.

Contrasts each node's embedding against its local neighbor context
(mean of neighbors' embeddings). Nodes whose embeddings are
indistinguishable from shuffled-feature embeddings are anomalous.
Multi-round scoring for robustness.

See :cite:`liu2022cola` for details.

Reference:
    Liu, Y., Li, Z., Pan, S., Gool, T., Xiang, T. and Gong, B., 2022.
    Anomaly Detection on Attributed Networks via Contrastive
    Self-Supervised Learning. In WWW, pp. 2137-2147.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input


class CoLA(BaseDetector):
    """CoLA: Contrastive Anomaly Detection on Attributed Networks.

    GCN encoder maps nodes to embeddings. A bilinear discriminator
    scores how well a node's embedding matches its local neighbor
    context (mean of neighbors' embeddings).
    Nodes with low discriminator scores are anomalous.

    This detector is **transductive**.

    Parameters
    ----------
    hidden_dim : int, default=64
        Hidden dimension of GCN.

    num_layers : int, default=2
        Number of GCN layers.

    epochs : int, default=100
        Training epochs.

    lr : float, default=1e-3
        Learning rate.

    contamination : float, default=0.1
        Expected proportion of anomalies.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_nodes,)
    labels_ : numpy array of shape (n_nodes,)
    threshold_ : float
    """

    def __init__(self, hidden_dim=64, num_layers=2, epochs=100,
                 lr=1e-3, contamination=0.1):
        super(CoLA, self).__init__(contamination=contamination)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        if data.x is None:
            raise ValueError("CoLA requires node features (data.x).")

        in_dim = data.x.shape[1]

        model = _CoLAModel(in_dim, self.hidden_dim, self.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        x = data.x
        ei = data.edge_index

        # Row-normalized adjacency for local context
        from torch_geometric.utils import degree
        row_deg = degree(ei[0], num_nodes=n_nodes)
        row_deg = row_deg.clamp(min=1)
        edge_weight = 1.0 / row_deg[ei[0]]
        adj_norm = torch.sparse_coo_tensor(
            ei, edge_weight, (n_nodes, n_nodes)).to_dense()

        model.train()
        for epoch in range(self.epochs):
            z = model.encode(x, ei)

            # Local context: mean of neighbors' embeddings
            local_ctx = adj_norm @ z  # (n, hid)

            # Positive: (node, local_context) pairs
            pos_scores = model.discriminate(z, local_ctx)

            # Negative: shuffle features, re-encode
            perm = torch.randperm(n_nodes)
            z_neg = model.encode(x[perm], ei)
            neg_scores = model.discriminate(z_neg, local_ctx)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones(n_nodes))
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros(n_nodes))
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Multi-round scoring for robustness
        model.eval()
        all_scores = []
        for _ in range(5):
            with torch.no_grad():
                z = model.encode(x, ei)
                local_ctx = adj_norm @ z
                s = -model.discriminate(z, local_ctx)
                all_scores.append(s.cpu().numpy())
        scores = torch.FloatTensor(np.mean(all_scores, axis=0))

        self.decision_scores_ = scores.cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "CoLA is a transductive detector. Use decision_scores_ "
            "after fit().")

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "CoLA is a transductive detector. Use labels_ after fit().")

    def predict_proba(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError("CoLA is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError("CoLA is a transductive detector.")


def _CoLAModel(in_dim, hid_dim, num_layers):
    """Factory: returns a torch.nn.Module for CoLA.

    Uses local-context contrastive learning: a GCN encoder
    produces node embeddings, and a bilinear discriminator
    scores (node, local_neighbor_context) pairs.
    """
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
            self.disc = nn.Bilinear(hid_dim, hid_dim, 1)

        def encode(self, x, edge_index):
            z = x
            for i, conv in enumerate(self.convs):
                z = conv(z, edge_index)
                if i < len(self.convs) - 1:
                    z = torch.relu(z)
            return z

        def discriminate(self, z, local_ctx):
            """Score (node_embedding, local_context) pairs."""
            return self.disc(z, local_ctx).squeeze(-1)

    return _Model()
