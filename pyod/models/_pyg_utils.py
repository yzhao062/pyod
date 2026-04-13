# -*- coding: utf-8 -*-
"""Shared utilities for graph anomaly detection models."""

import numpy as np


def validate_graph_input(X, edge_index=None):
    """Convert input to PyG Data object.

    Accepts:
    - PyG ``Data`` object (returned as-is)
    - numpy ``X`` (n_nodes, n_features) + numpy ``edge_index`` (2, n_edges)

    Parameters
    ----------
    X : Data or array-like
        Node features or a PyG Data object.
    edge_index : array-like or None
        COO edge list. Required when X is not a Data object.

    Returns
    -------
    data : torch_geometric.data.Data
    """
    import torch
    from torch_geometric.data import Data

    if isinstance(X, Data):
        return X

    x = torch.FloatTensor(np.asarray(X, dtype=np.float32))
    if edge_index is None:
        raise ValueError(
            "edge_index required when X is not a PyG Data object")
    ei = torch.LongTensor(np.asarray(edge_index, dtype=np.int64))
    return Data(x=x, edge_index=ei)


def to_dense_adj_numpy(edge_index, num_nodes):
    """Convert edge_index to dense adjacency matrix (numpy).

    Parameters
    ----------
    edge_index : np.ndarray of shape (2, n_edges)
        COO edge list.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    adj : np.ndarray of shape (num_nodes, num_nodes)
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    row, col = edge_index[0], edge_index[1]
    adj[row, col] = 1.0
    return adj


def to_sparse_adj(edge_index, num_nodes):
    """Convert edge_index to scipy sparse CSR adjacency.

    Parameters
    ----------
    edge_index : np.ndarray of shape (2, n_edges)
        COO edge list.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    adj : scipy.sparse.csr_matrix
    """
    from scipy.sparse import csr_matrix
    row, col = edge_index[0], edge_index[1]
    data = np.ones(len(row), dtype=np.float64)
    return csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
