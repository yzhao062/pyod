# -*- coding: utf-8 -*-
"""CONAD: Contrastive Attributed Network Anomaly Detection.

Constructs an anomalous view by injecting synthetic anomalies
(attribute swapping + random edge injection), then contrasts
with the original view. Dual reconstruction (structure +
attributes) from the original view. Nodes with high contrastive
distance and high reconstruction error are anomalous.

See :cite:`xu2022conad` for details.

Reference:
    Xu, Z., Huang, X., Zhao, Y., Dong, Y., and Li, J., 2022.
    Contrastive Attributed Network Anomaly Detection with Data
    Augmentation. In PAKDD, pp. 444-457.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input


class CONAD(BaseDetector):
    """CONAD: Contrastive + Reconstruction Anomaly Detection.

    Constructs an anomalous view via attribute swapping and
    random edge injection, encodes both views with a shared
    GCN, and scores nodes by contrastive distance + dual
    (structure + attribute) reconstruction error.

    This detector is **transductive**.

    Parameters
    ----------
    hidden_dim : int, default=64
        Hidden dimension.

    num_layers : int, default=2
        Number of GCN layers.

    aug_ratio : float, default=0.2
        Fraction of edges/attributes to drop/mask.

    alpha : float, default=0.5
        Weight for reconstruction loss (vs contrastive).

    dropout : float, default=0.3
        Dropout rate.

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

    def __init__(self, hidden_dim=64, num_layers=2, aug_ratio=0.2,
                 alpha=0.5, dropout=0.3, epochs=100, lr=1e-3,
                 contamination=0.1):
        super(CONAD, self).__init__(contamination=contamination)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aug_ratio = aug_ratio
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
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        if data.x is None:
            raise ValueError("CONAD requires node features (data.x).")

        in_dim = data.x.shape[1]

        model = _CONADModel(
            in_dim, self.hidden_dim, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        x = data.x
        ei = data.edge_index

        # Dense adjacency for structure reconstruction
        adj = torch.zeros(n_nodes, n_nodes)
        adj[ei[0], ei[1]] = 1.0

        model.train()
        for epoch in range(self.epochs):
            # Create anomalous view (inject synthetic anomalies)
            x_aug, ei_aug = _create_anomalous_view(
                x, ei, self.aug_ratio)

            z_orig, z_aug, x_hat, a_hat = model(
                x, ei, x_aug, ei_aug)

            # Contrastive loss between original and anomalous views
            z_o = F.normalize(z_orig, dim=1)
            z_a = F.normalize(z_aug, dim=1)
            cos_sim = (z_o * z_a).sum(dim=1)
            contrastive_loss = -cos_sim.mean()

            # Dual reconstruction: structure + attributes
            struct_loss = torch.mean(
                torch.sum((adj - a_hat) ** 2, dim=1))
            attr_loss = torch.mean(
                torch.sum((x - x_hat) ** 2, dim=1))
            recon_loss = struct_loss + attr_loss

            loss = contrastive_loss + self.alpha * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Score: contrastive distance + dual reconstruction error
        model.eval()
        with torch.no_grad():
            x_aug, ei_aug = _create_anomalous_view(
                x, ei, self.aug_ratio)
            z_orig, z_aug, x_hat, a_hat = model(
                x, ei, x_aug, ei_aug)

            z_o = F.normalize(z_orig, dim=1)
            z_a = F.normalize(z_aug, dim=1)
            cos_dist = 1.0 - (z_o * z_a).sum(dim=1)
            struct_err = torch.sum((adj - a_hat) ** 2, dim=1)
            attr_err = torch.sum((x - x_hat) ** 2, dim=1)
            scores = cos_dist + self.alpha * (struct_err + attr_err)

        self.decision_scores_ = scores.cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "CONAD is a transductive detector. Use decision_scores_ "
            "after fit().")

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "CONAD is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError("CONAD is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError("CONAD is a transductive detector.")


def _create_anomalous_view(x, edge_index, ratio):
    """Create anomalous view by injecting synthetic anomalies.

    Following the CONAD paper: perturb a subset of node attributes
    by swapping with random other nodes, and add random edges.

    Parameters
    ----------
    x : torch.Tensor of shape (n, d)
    edge_index : torch.LongTensor of shape (2, m)
    ratio : float

    Returns
    -------
    x_aug : torch.Tensor
    edge_index_aug : torch.LongTensor
    """
    import torch

    n = x.shape[0]
    n_perturb = max(1, int(n * ratio))

    # Attribute perturbation: swap attributes with random nodes
    x_aug = x.clone()
    perturb_idx = torch.randperm(n)[:n_perturb]
    swap_idx = torch.randperm(n)[:n_perturb]
    x_aug[perturb_idx] = x[swap_idx]

    # Edge perturbation: add random edges
    n_edges = edge_index.shape[1]
    n_add = max(1, int(n_edges * ratio))
    new_src = torch.randint(0, n, (n_add,))
    new_dst = torch.randint(0, n, (n_add,))
    ei_aug = torch.cat(
        [edge_index, torch.stack([new_src, new_dst])], dim=1)

    return x_aug, ei_aug


def _CONADModel(in_dim, hid_dim, num_layers, dropout):
    """Factory: returns torch.nn.Module for CONAD."""
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
            self._dropout = dropout

        def _encode(self, x, edge_index):
            z = x
            for i, conv in enumerate(self.convs):
                z = conv(z, edge_index)
                if i < len(self.convs) - 1:
                    z = torch.relu(z)
                    z = torch.dropout(
                        z, p=self._dropout, train=self.training)
            return z

        def forward(self, x, ei, x_aug, ei_aug):
            z_orig = self._encode(x, ei)
            z_aug = self._encode(x_aug, ei_aug)
            # Dual reconstruction from original view
            x_hat = self.attr_decoder(z_orig)
            a_hat = torch.sigmoid(z_orig @ z_orig.t())
            return z_orig, z_aug, x_hat, a_hat

    return _Model()
