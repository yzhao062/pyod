# -*- coding: utf-8 -*-
"""ANOMALOUS: A Joint Modeling Approach for Anomaly Detection.

CUR-decomposition-based method that models attributes as
``X ≈ XWX + R``, where W (d x n) jointly selects representative
attributes and nodes. The residual R captures anomalies via L21
sparsity, regularized by the graph Laplacian for network coherence.

See :cite:`peng2018anomalous` for details.

Reference:
    Peng, Z., Luo, M., Li, J., Liu, H. and Zheng, Q., 2018.
    ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on
    Attributed Networks. In IJCAI, pp. 3529-3535.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from ._pyg_utils import validate_graph_input, to_sparse_adj


class ANOMALOUS(BaseDetector):
    """ANOMALOUS: Joint Modeling for Anomaly Detection.

    CUR-style decomposition: ``min_{W,R} ||X - XWX - R||_F^2
    + alpha * ||W||_F^2 + lambda_r * ||R||_{2,1} + gamma *
    tr(R^T L R)`` where W is d x n, L is the graph Laplacian.
    Anomaly score = L2 norm of each residual row.

    This detector is **transductive**.

    Parameters
    ----------
    alpha : float, default=1.0
        Frobenius norm penalty on W (regularization).

    gamma : float, default=1.0
        Graph Laplacian smoothing strength on R.

    lambda_r : float, default=0.01
        L21-norm penalty on residual R (row-sparsity).
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

    def __init__(self, alpha=1.0, gamma=1.0, lambda_r=0.01,
                 max_iter=100, contamination=0.1):
        super(ANOMALOUS, self).__init__(contamination=contamination)
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_r = lambda_r
        self.max_iter = max_iter

    def fit(self, X, y=None, edge_index=None):
        """Fit the detector on graph data.

        Parameters
        ----------
        X : Data or array-like
            PyG Data or node features.
        y : ignored
        edge_index : array-like or None

        Returns
        -------
        self
        """
        data = validate_graph_input(X, edge_index)
        n_nodes = data.num_nodes
        self._set_n_classes(y)

        if data.x is None:
            raise ValueError(
                "ANOMALOUS requires node features (data.x).")

        features = data.x.cpu().numpy().astype(np.float64)
        ei = data.edge_index.cpu().numpy()
        adj = to_sparse_adj(ei, n_nodes)

        scores = self._factorize(adj, features)

        self.decision_scores_ = scores
        self._process_decision_scores()
        return self

    def _factorize(self, A, X):
        """Alternating optimization for ANOMALOUS.

        Objective: min_{W,R} ||X - XWX - R||^2 + alpha * ||W||_F^2
                   + lambda_r * ||R||_{2,1} + gamma * tr(R^T L R)
        W is d x n (CUR-style). X selects columns, W mixes,
        X selects rows.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix of shape (n, n)
        X : np.ndarray of shape (n, d)

        Returns
        -------
        scores : np.ndarray of shape (n,)
        """
        n, d = X.shape
        A_dense = A.toarray()

        # Graph Laplacian: L = D - A
        degree = np.asarray(A.sum(axis=1)).ravel()
        L = np.diag(degree) - A_dense

        # Normalize features for numerical stability
        feat_norms = np.linalg.norm(X, axis=0, keepdims=True)
        feat_norms = np.maximum(feat_norms, 1e-10)
        X_norm = X / feat_norms

        # Initialize W (d x n) and R (n x d)
        W = np.zeros((d, n), dtype=np.float64)
        R = np.zeros((n, d), dtype=np.float64)
        lr = 0.001

        for _ in range(self.max_iter):
            # Forward: XWX is (n,d)@(d,n)@(n,d) = (n,d)
            XWX = X_norm @ W @ X_norm
            residual = X_norm - XWX - R

            # Update W via gradient descent with clipping
            grad_W = (-2.0 * X_norm.T @ residual @ X_norm.T
                       + 2.0 * self.alpha * W)
            grad_norm = np.linalg.norm(grad_W)
            if grad_norm > 1.0:
                grad_W = grad_W / grad_norm
            W -= lr * grad_W

            # Update R: Laplacian smoothing + L21 proximal
            P = X_norm - X_norm @ W @ X_norm
            # Graph-aware smoothing: (I + gamma*L)^{-1} P
            P_smooth = np.linalg.solve(
                np.eye(n) + self.gamma * L, P)
            # L21 shrinkage for row-sparsity
            row_norms = np.linalg.norm(
                P_smooth, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-10)
            shrink = np.maximum(
                0.0,
                1.0 - self.lambda_r / (2.0 * row_norms))
            R = shrink * P_smooth

        scores = np.linalg.norm(R, axis=1)
        return scores

    def decision_function(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector. Use "
            "decision_scores_ after fit().")

    def predict(self, X, return_confidence=False):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X, method="linear", return_confidence=False):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector.")
