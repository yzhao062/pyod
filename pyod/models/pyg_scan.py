# -*- coding: utf-8 -*-
"""SCAN: Structural Clustering Algorithm for Networks.

Detects anomalous nodes based on structural similarity to neighbors.
Nodes with low average structural similarity to their neighbors are
scored as anomalous. This is a structure-only method — node features
are not used.

See :cite:`xu2007scan` for details.

Reference:
    Xu, X., Yuruk, N., Feng, Z. and Schweiger, T.A.J., 2007. SCAN:
    A Structural Clustering Algorithm for Networks. In KDD, pp. 824-833.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input


class SCAN(BaseDetector):
    """SCAN: Structural Clustering Algorithm for Networks.

    Implements the full SCAN procedure: computes structural
    similarity ``sigma(u,v) = |N[u] ∩ N[v]| / sqrt(|N[u]| *
    |N[v]|)`` between neighbors, identifies cores (nodes with
    at least ``mu`` epsilon-neighbors), clusters cores via
    BFS reachability on epsilon-edges, and classifies remaining
    nodes as hubs (adjacent to 2+ clusters) or outliers.

    Continuous anomaly score is derived from this classification:
    cores receive low scores (inversely proportional to their
    epsilon-neighbor count), hubs receive medium scores, and
    outliers receive the highest scores.

    This detector is **transductive**.

    Parameters
    ----------
    epsilon : float, default=0.5
        Structural similarity threshold. An edge (u,v) is an
        epsilon-edge if ``sigma(u,v) >= epsilon``.

    mu : int, default=2
        Minimum number of epsilon-neighbors for a node to be
        a core.

    contamination : float, default=0.1
        Expected proportion of anomalies in the dataset.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_nodes,)
        Anomaly scores. Higher = more anomalous.

    labels_ : numpy array of shape (n_nodes,)
        Binary labels (0=inlier, 1=outlier).

    threshold_ : float
        Score threshold derived from contamination.

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> data = Data(edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
    ...             num_nodes=3)
    >>> clf = SCAN(contamination=0.3)
    >>> clf.fit(data)  # doctest: +SKIP
    >>> clf.decision_scores_  # doctest: +SKIP
    """

    def __init__(self, epsilon=0.5, mu=2, contamination=0.1):
        super(SCAN, self).__init__(contamination=contamination)
        self.epsilon = epsilon
        self.mu = mu

    def fit(self, X, y=None, edge_index=None):
        """Fit the detector on graph data.

        Parameters
        ----------
        X : Data or array-like
            PyG Data object, or node features (n_nodes, n_features).
            For SCAN, node features are ignored — only structure is
            used. When passing a structure-only graph as Data, set
            ``data.num_nodes`` explicitly.

        y : ignored

        edge_index : array-like or None
            COO edge list. Required when X is numpy.

        Returns
        -------
        self
        """
        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        ei = data.edge_index.cpu().numpy()
        scores = self._compute_scores(ei, n_nodes)

        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def _compute_scores(self, edge_index, num_nodes):
        """Full SCAN procedure: similarity, cores, clustering, scoring.

        Parameters
        ----------
        edge_index : np.ndarray of shape (2, n_edges)
        num_nodes : int

        Returns
        -------
        scores : np.ndarray of shape (num_nodes,)
        """
        from scipy.sparse import csr_matrix, eye as sp_eye, diags
        from scipy.sparse.csgraph import connected_components

        if edge_index.shape[1] == 0:
            return np.ones(num_nodes)

        row, col = edge_index[0], edge_index[1]
        adj = csr_matrix(
            (np.ones(len(row), dtype=np.float64), (row, col)),
            shape=(num_nodes, num_nodes))

        # --- Structural similarity ---
        adj_self = adj + sp_eye(num_nodes, dtype=np.float64)
        adj_self = (adj_self > 0).astype(np.float64)
        deg_self = np.asarray(adj_self.sum(axis=1)).ravel()
        intersection = adj_self.dot(adj_self.T)
        inv_sqrt = np.zeros_like(deg_self)
        nz = deg_self > 0
        inv_sqrt[nz] = 1.0 / np.sqrt(deg_self[nz])
        sim_matrix = diags(inv_sqrt).dot(intersection).dot(
            diags(inv_sqrt))

        # --- Epsilon-neighborhood ---
        sim_edges = sim_matrix.multiply(adj)
        eps_adj = sim_edges.copy()
        eps_adj.data[eps_adj.data < self.epsilon] = 0
        eps_adj.eliminate_zeros()
        n_eps = np.asarray(
            (eps_adj > 0).sum(axis=1)).ravel().astype(int)

        # --- Core detection ---
        is_core = n_eps >= self.mu

        # --- Cluster cores via connected components ---
        core_mask = is_core.astype(np.float64)
        core_graph = diags(core_mask).dot(eps_adj).dot(
            diags(core_mask))
        _, comp_labels = connected_components(
            core_graph, directed=False)
        cluster_of = np.where(is_core, comp_labels, -1)

        # --- Classify non-cores: hub vs outlier ---
        node_deg = np.asarray(adj.sum(axis=1)).ravel()
        sim_sum = np.asarray(sim_edges.sum(axis=1)).ravel()
        mean_sim = np.where(
            node_deg > 0, sim_sum / node_deg, 0.0)

        scores = np.zeros(num_nodes)
        for u in range(num_nodes):
            if is_core[u]:
                # Core: low score, inversely proportional to
                # excess epsilon-neighbors
                scores[u] = max(
                    0.0, 1.0 - n_eps[u] / max(self.mu * 2, 1))
            else:
                # Count adjacent clusters
                nbrs = adj[u].indices
                adj_clusters = set()
                for v in nbrs:
                    if cluster_of[v] >= 0:
                        adj_clusters.add(cluster_of[v])
                if len(adj_clusters) >= 2:
                    # Hub: medium score
                    scores[u] = 0.5 + 0.5 * (1.0 - mean_sim[u])
                else:
                    # Outlier: high score
                    scores[u] = 1.0 + (1.0 - mean_sim[u])

        return scores

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "SCAN is a transductive detector. Use decision_scores_ "
            "after fit().")

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "SCAN is a transductive detector. Use labels_ after fit().")

    def predict_proba(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "SCAN is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "SCAN is a transductive detector.")
