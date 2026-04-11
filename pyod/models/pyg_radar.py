# -*- coding: utf-8 -*-
"""Radar: Residual Analysis for Anomaly Detection in Attributed Networks.

Models node attributes as X ≈ WX + R, where W is a learned n x n
weight matrix that predicts each node's attributes from other nodes'
attributes, and R is the residual. The adjacency enters only through
the graph Laplacian regularizer on R. Anomaly scores are the L2
norms of residual rows (nodes with large residuals are anomalous).

See :cite:`li2017radar` for details.

Reference:
    Li, J., Dani, H., Hu, X. and Liu, H., 2017. Radar: Residual Analysis
    for Anomaly Detection in Attributed Networks. In IJCAI, pp. 2152-2158.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input, to_sparse_adj


class Radar(BaseDetector):
    """Radar: Residual Analysis for Anomaly Detection.

    Solves ``min_{W,R} ||X - WX - R||_F^2 + alpha * ||W||_F^2
    + gamma * (||R||_{2,1} + tr(R^T L R))`` via alternating
    optimization. W is a learned n x n weight matrix (not the
    adjacency). The graph Laplacian L smooths the residual R.
    Anomaly score per node = L2 norm of residual row.

    This detector is **transductive**.

    Parameters
    ----------
    alpha : float, default=1.0
        Frobenius norm penalty on W (regularization).

    gamma : float, default=0.01
        Residual regularization strength: controls both L21
        row-sparsity and graph Laplacian smoothing on R.
        Scale-sensitive; typical range 0.001-0.1.

    max_iter : int, default=100
        Number of alternating optimization iterations.

    contamination : float, default=0.1
        Expected proportion of anomalies.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_nodes,)
    labels_ : numpy array of shape (n_nodes,)
    threshold_ : float
    """

    def __init__(self, alpha=1.0, gamma=0.01, max_iter=100,
                 contamination=0.1):
        super(Radar, self).__init__(contamination=contamination)
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter

    def fit(self, X, y=None, edge_index=None):
        """Fit the detector on graph data.

        Parameters
        ----------
        X : Data or array-like
            PyG Data or node features (n_nodes, n_features).
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

        if data.x is None:
            raise ValueError("Radar requires node features (data.x).")

        features = data.x.cpu().numpy().astype(np.float64)
        ei = data.edge_index.cpu().numpy()
        adj = to_sparse_adj(ei, n_nodes)

        scores = self._factorize(adj, features)

        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def _factorize(self, A, X):
        """Alternating optimization for Radar.

        Objective: min_{W,R} ||X - WX - R||^2 + alpha * ||W||_F^2
                   + gamma * ||R||_{2,1}
        W is n x n (learned, initialized to identity).
        A enters via Laplacian regularizer on R.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix of shape (n, n)
        X : np.ndarray of shape (n, d)

        Returns
        -------
        scores : np.ndarray of shape (n,)
        """
        n, d = X.shape

        # Graph Laplacian L = D - A (for residual smoothness)
        A_dense = A.toarray()
        degree = np.asarray(A.sum(axis=1)).ravel()
        L = np.diag(degree) - A_dense

        # Initialize W (n x n), predict each node from others
        W = np.eye(n, dtype=np.float64)
        R = np.zeros((n, d), dtype=np.float64)

        XXT = X @ X.T  # (n x n), precompute

        for _ in range(self.max_iter):
            # Update W: dL/dW = -2(X-WX-R)X^T + 2*alpha*W = 0
            # W(XX^T + alpha*I) = (X - R)X^T
            rhs = (X - R) @ X.T
            W = rhs @ np.linalg.inv(
                XXT + self.alpha * np.eye(n))

            # Update R: Laplacian smoothing + L21 proximal
            P = X - W @ X
            # Graph-aware smoothing: (I + gamma*L)^{-1} P
            P_smooth = np.linalg.solve(
                np.eye(n) + self.gamma * L, P)
            # L21 shrinkage for row-sparsity
            row_norms = np.linalg.norm(
                P_smooth, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-10)
            shrink = np.maximum(
                0.0, 1.0 - self.gamma / (2.0 * row_norms))
            R = shrink * P_smooth

        scores = np.linalg.norm(R, axis=1)
        return scores

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "Radar is a transductive detector. Use decision_scores_ "
            "after fit().")

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "Radar is a transductive detector. Use labels_ after fit().")

    def predict_proba(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "Radar is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "Radar is a transductive detector.")
