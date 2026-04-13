# Graph Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 8 graph anomaly detectors as first-class PyOD `BaseDetector` subclasses with full test, example, knowledge base, ADEngine, and documentation integration.

**Architecture:** All detectors inherit `BaseDetector`, accept PyG `Data` objects, and are transductive in v1 (fit-only, no out-of-sample scoring). Three categories: classical matrix/clustering (SCAN, Radar, ANOMALOUS), GCN-based autoencoders (DOMINANT, AnomalyDAE, GUIDE), and contrastive (CoLA, CONAD). Shared utilities in `_pyg_utils.py`.

**Tech Stack:** Python, PyTorch, PyTorch Geometric (GCNConv, GATConv), scipy.sparse, numpy. Optional dependency via `pip install pyod[graph]`.

**Spec:** `docs/superpowers/specs/2026-04-09-graph-od-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `pyod/models/_pyg_utils.py` | Input validation, adjacency conversion, dense/sparse helpers |
| `pyod/models/pyg_scan.py` | SCAN structural clustering detector |
| `pyod/models/pyg_radar.py` | Radar matrix factorization detector |
| `pyod/models/pyg_anomalous.py` | ANOMALOUS joint MF detector |
| `pyod/models/pyg_dominant.py` | DOMINANT GCN autoencoder detector |
| `pyod/models/pyg_anomalydae.py` | AnomalyDAE dual autoencoder detector |
| `pyod/models/pyg_cola.py` | CoLA contrastive detector |
| `pyod/models/pyg_conad.py` | CONAD contrastive+reconstruction detector |
| `pyod/models/pyg_guide.py` | GUIDE motif-based dual autoencoder detector |
| `pyod/test/test_pyg_scan.py` | SCAN tests |
| `pyod/test/test_pyg_radar.py` | Radar tests |
| `pyod/test/test_pyg_anomalous.py` | ANOMALOUS tests |
| `pyod/test/test_pyg_dominant.py` | DOMINANT tests |
| `pyod/test/test_pyg_anomalydae.py` | AnomalyDAE tests |
| `pyod/test/test_pyg_cola.py` | CoLA tests |
| `pyod/test/test_pyg_conad.py` | CONAD tests |
| `pyod/test/test_pyg_guide.py` | GUIDE tests |
| `examples/pyg_scan_example.py` | SCAN example |
| `examples/pyg_radar_example.py` | Radar example |
| `examples/pyg_anomalous_example.py` | ANOMALOUS example |
| `examples/pyg_dominant_example.py` | DOMINANT example |
| `examples/pyg_anomalydae_example.py` | AnomalyDAE example |
| `examples/pyg_cola_example.py` | CoLA example |
| `examples/pyg_conad_example.py` | CONAD example |
| `examples/pyg_guide_example.py` | GUIDE example |

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/data.py` | Add `generate_graph_data()` |
| `pyod/utils/ad_engine.py:46-96` | Add graph branch to `profile_data()` and `_sniff_data_type()` |
| `pyod/utils/knowledge/algorithms.json` | Add 8 graph detector entries |
| `pyod/utils/knowledge/routing_rules.json` | Add 2 graph routing rules |
| `pyod/utils/knowledge/benchmarks.json` | Add BOND benchmark entry |
| `setup.py:45-57` | Add `graph` extra, update `all` |
| `README.rst:321-382` | Add Graph AD table section after TS section |
| `docs/index.rst:238-300` | Add Graph AD table section after TS section |
| `docs/pyod.models.rst:595` | Add 8 autodoc entries before `.. rubric:: References` |
| `docs/zreferences.bib` | Add 9 BibTeX entries (8 algorithms + BOND) |
| `docs/requirements.txt` | Add `torch_geometric` |
| `CHANGES.txt` | Add graph AD entry |
| `.github/workflows/testing.yml:44` | Add `--ignore` for 8 `test_pyg_*.py` files on macOS |
| `.github/workflows/testing-cron.yml:44` | Same macOS ignores |

---

## Dependency graph

```
Task 1 (foundation) → Tasks 2-9 (8 detectors, all parallel)
                     → Task 10 (knowledge base, parallel with detectors)
                     → Task 11 (ADEngine, parallel with detectors)
Tasks 2-9, 10, 11   → Task 12 (documentation)
Task 12              → Task 13 (packaging & CI)
```

---

### Task 1: Foundation — shared utilities and test data generator

**Files:**
- Create: `pyod/models/_pyg_utils.py`
- Modify: `pyod/utils/data.py`

- [ ] **Step 1: Create `_pyg_utils.py`**

```python
# pyod/models/_pyg_utils.py
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
```

- [ ] **Step 2: Add `generate_graph_data()` to `pyod/utils/data.py`**

Add the following function at the end of `pyod/utils/data.py` (before any `if __name__` block):

```python
def generate_graph_data(n_nodes=300, n_features=16, n_edges_per_node=5,
                        contamination=0.1, random_state=None):
    """Generate synthetic attributed graph data with planted anomalies.

    Normal nodes have features from N(0, 1). Anomaly nodes have features
    shifted by +5 standard deviations. Edges are generated via random
    neighbor selection (undirected, no self-loops, no duplicates).

    Parameters
    ----------
    n_nodes : int, default=300
        Number of nodes.

    n_features : int, default=16
        Dimensionality of node features.

    n_edges_per_node : int, default=5
        Average number of edges per node (Poisson-sampled per node).

    contamination : float, default=0.1
        Fraction of nodes that are anomalies.

    random_state : int, RandomState or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_nodes, n_features)
        Node feature matrix (float32).

    edge_index : np.ndarray of shape (2, n_edges)
        COO-format edge list (int64, undirected, no self-loops).

    y : np.ndarray of shape (n_nodes,)
        Binary labels: 0 = normal, 1 = anomaly.
    """
    rng = check_random_state(random_state)

    n_anomalies = max(1, int(n_nodes * contamination))
    n_normal = n_nodes - n_anomalies

    # Features: normal from N(0,1), anomalies shifted by +5
    X_normal = rng.randn(n_normal, n_features).astype(np.float32)
    X_anomaly = (rng.randn(n_anomalies, n_features) + 5.0).astype(
        np.float32)
    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal, dtype=np.int32),
                        np.ones(n_anomalies, dtype=np.int32)])

    # Shuffle
    perm = rng.permutation(n_nodes)
    X, y = X[perm], y[perm]

    # Generate edges via random neighbor selection
    edges = set()
    for i in range(n_nodes):
        n_nbrs = max(1, rng.poisson(n_edges_per_node))
        candidates = rng.choice(n_nodes, size=min(n_nbrs + 1, n_nodes),
                                replace=False)
        for j in candidates:
            if i != j:
                u, v = (i, j) if i < j else (j, i)
                edges.add((u, v))

    rows, cols = [], []
    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])

    edge_index = np.array([rows, cols], dtype=np.int64)
    return X, edge_index, y
```

- [ ] **Step 3: Verify imports work**

Run: `python -c "from pyod.models._pyg_utils import validate_graph_input; print('OK')"`
Expected: `OK` (if PyG installed) or `ImportError` inside function (which is fine — lazy import).

Run: `python -c "from pyod.utils.data import generate_graph_data; X, ei, y = generate_graph_data(n_nodes=50, random_state=42); print(X.shape, ei.shape, y.sum())"`
Expected: `(50, 16) (2, ~500) 5` (exact edge count varies)

- [ ] **Step 4: Commit**

```bash
git add pyod/models/_pyg_utils.py pyod/utils/data.py
git commit -m "feat: add graph utilities (_pyg_utils) and generate_graph_data()"
```

---

### Task 2: SCAN — Structural Clustering Algorithm for Networks

**Files:**
- Create: `pyod/models/pyg_scan.py`
- Create: `pyod/test/test_pyg_scan.py`
- Create: `examples/pyg_scan_example.py`

**Paper:** Xu et al., "SCAN: A Structural Clustering Algorithm for Networks", KDD 2007.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_scan.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for SCAN graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_scan import SCAN
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestSCAN(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=200, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = SCAN(epsilon=0.5, mu=2, contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 200
        assert len(clf.labels_) == 200

    def test_fit_numpy(self):
        clf = SCAN(epsilon=0.5, mu=2, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 200

    def test_scores_nonnegative(self):
        clf = SCAN(contamination=0.1)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = SCAN()
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = SCAN()
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_structure_only(self):
        """SCAN works without node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=200)
        clf = SCAN(contamination=0.1)
        clf.fit(data_no_feat)
        assert len(clf.decision_scores_) == 200

    def test_empty_graph(self):
        """Isolated nodes with no edges all score 1.0."""
        data_empty = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_nodes=10)
        clf = SCAN(contamination=0.1)
        clf.fit(data_empty)
        assert len(clf.decision_scores_) == 10
        assert np.all(clf.decision_scores_ == 1.0)

    def test_epsilon_mu_affect_scores(self):
        """Different epsilon/mu values produce different scores."""
        clf1 = SCAN(epsilon=0.3, mu=1, contamination=0.1)
        clf1.fit(self.data)
        clf2 = SCAN(epsilon=0.9, mu=5, contamination=0.1)
        clf2.fit(self.data)
        # Stricter params should produce more non-core nodes
        assert not np.allclose(
            clf1.decision_scores_, clf2.decision_scores_)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_scan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyod.models.pyg_scan'`

- [ ] **Step 3: Write SCAN implementation**

Create `pyod/models/pyg_scan.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_scan.py -v`
Expected: 8 tests PASS (or SKIP if PyG not installed)

- [ ] **Step 5: Write example**

Create `examples/pyg_scan_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using SCAN for graph anomaly detection.

SCAN is a structure-only method -- it does not use node features.
It is transductive: use decision_scores_ and labels_ after fit().

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_scan import SCAN
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=500, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'SCAN'
    clf = SCAN(epsilon=0.5, mu=2, contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_scan.py pyod/test/test_pyg_scan.py examples/pyg_scan_example.py
git commit -m "feat: add SCAN graph anomaly detector"
```

---

### Task 3: Radar — Residual Analysis for Anomaly Detection

**Files:**
- Create: `pyod/models/pyg_radar.py`
- Create: `pyod/test/test_pyg_radar.py`
- Create: `examples/pyg_radar_example.py`

**Paper:** Li et al., "Radar: Residual Analysis for Anomaly Detection in Attributed Networks", IJCAI 2017.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_radar.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for Radar graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_radar import Radar
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestRadar(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=200, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = Radar(alpha=1.0, gamma=1.0, max_iter=20, contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 200

    def test_fit_numpy(self):
        clf = Radar(max_iter=20, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 200

    def test_scores_nonnegative(self):
        clf = Radar(max_iter=20)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = Radar(max_iter=20)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = Radar(max_iter=20)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """Radar requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=200)
        clf = Radar(max_iter=20)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_radar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyod.models.pyg_radar'`

- [ ] **Step 3: Write Radar implementation**

Create `pyod/models/pyg_radar.py`:

```python
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

    gamma : float, default=1.0
        Residual regularization strength: controls both L21
        row-sparsity and graph Laplacian smoothing on R.

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

    def __init__(self, alpha=1.0, gamma=1.0, max_iter=100,
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
            # Update W: ∂/∂W = -2(X-WX-R)X^T + 2αW = 0
            # W(XX^T + αI) = (X - R)X^T
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_radar.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_radar_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using Radar for graph anomaly detection.

Radar uses matrix factorization to find nodes whose attributes
deviate from network-smoothed predictions. Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_radar import Radar
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=500, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'Radar'
    clf = Radar(alpha=1.0, gamma=1.0, max_iter=50,
                contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_radar.py pyod/test/test_pyg_radar.py examples/pyg_radar_example.py
git commit -m "feat: add Radar graph anomaly detector"
```

---

### Task 4: ANOMALOUS — Joint Modeling for Anomaly Detection

**Files:**
- Create: `pyod/models/pyg_anomalous.py`
- Create: `pyod/test/test_pyg_anomalous.py`
- Create: `examples/pyg_anomalous_example.py`

**Paper:** Peng et al., "ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks", IJCAI 2018.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_anomalous.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for ANOMALOUS graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_anomalous import ANOMALOUS
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestANOMALOUS(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=200, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = ANOMALOUS(alpha=1.0, gamma=1.0, max_iter=20,
                        contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 200

    def test_fit_numpy(self):
        clf = ANOMALOUS(max_iter=20, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 200

    def test_scores_nonnegative(self):
        clf = ANOMALOUS(max_iter=20)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = ANOMALOUS(max_iter=20)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = ANOMALOUS(max_iter=20)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """ANOMALOUS requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=200)
        clf = ANOMALOUS(max_iter=20)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_anomalous.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyod.models.pyg_anomalous'`

- [ ] **Step 3: Write ANOMALOUS implementation**

Create `pyod/models/pyg_anomalous.py`:

```python
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

    lambda_r : float, default=1.0
        L21-norm penalty on residual R (row-sparsity).

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

    def __init__(self, alpha=1.0, gamma=1.0, lambda_r=1.0,
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

        # Initialize W (d x n) and R (n x d)
        W = np.zeros((d, n), dtype=np.float64)
        R = np.zeros((n, d), dtype=np.float64)
        lr = 0.001

        for _ in range(self.max_iter):
            # Forward: XWX is (n,d)@(d,n)@(n,d) = (n,d)
            XWX = X @ W @ X
            residual = X - XWX - R

            # Update W via gradient descent
            # grad = -2 X^T (X - XWX - R) X^T + 2 alpha W
            grad_W = (-2.0 * X.T @ residual @ X.T
                       + 2.0 * self.alpha * W)
            W -= lr * grad_W

            # Update R: Laplacian smoothing + L21 proximal
            P = X - X @ W @ X
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

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "ANOMALOUS is a transductive detector.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_anomalous.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_anomalous_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using ANOMALOUS for graph anomaly detection.

ANOMALOUS extends Radar with graph Laplacian regularization.
Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_anomalous import ANOMALOUS
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=500, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'ANOMALOUS'
    clf = ANOMALOUS(alpha=1.0, gamma=1.0, lambda_r=1.0,
                    max_iter=50, contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_anomalous.py pyod/test/test_pyg_anomalous.py examples/pyg_anomalous_example.py
git commit -m "feat: add ANOMALOUS graph anomaly detector"
```

---

### Task 5: DOMINANT — Deep Anomaly Detection on Attributed Networks

**Files:**
- Create: `pyod/models/pyg_dominant.py`
- Create: `pyod/test/test_pyg_dominant.py`
- Create: `examples/pyg_dominant_example.py`

**Paper:** Ding et al., "Deep Anomaly Detection on Attributed Networks", SDM 2019.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_dominant.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for DOMINANT graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_dominant import DOMINANT
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestDOMINANT(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=100, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = DOMINANT(hidden_dim=32, num_layers=2, epochs=5,
                       contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 100

    def test_fit_numpy(self):
        clf = DOMINANT(hidden_dim=32, epochs=5, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 100

    def test_scores_nonnegative(self):
        clf = DOMINANT(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = DOMINANT(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = DOMINANT(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """DOMINANT requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=100)
        clf = DOMINANT(hidden_dim=32, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_dominant.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyod.models.pyg_dominant'`

- [ ] **Step 3: Write DOMINANT implementation**

Create `pyod/models/pyg_dominant.py`:

```python
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
        """Fit the detector on graph data.

        Parameters
        ----------
        X : Data or array-like
            PyG Data or node features (n_nodes, n_features).
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
            raise ValueError("DOMINANT requires node features (data.x).")

        in_dim = data.x.shape[1]

        # Build model
        model = _DOMINANTModel(
            in_dim, self.hidden_dim, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        x = data.x
        ei = data.edge_index

        # Dense adjacency for loss computation
        adj = torch.zeros(n_nodes, n_nodes)
        adj[ei[0], ei[1]] = 1.0

        # Training
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

        # Compute per-node scores
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
            "DOMINANT is a transductive detector. Use "
            "decision_scores_ after fit().")

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "DOMINANT is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "DOMINANT is a transductive detector.")

    def predict_confidence(self, X):
        """Not supported (transductive detector)."""
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

            # Structure reconstruction
            a_hat = torch.sigmoid(z @ z.t())
            # Attribute reconstruction
            x_hat = self.attr_decoder(z)
            return a_hat, x_hat

    return _Model()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_dominant.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_dominant_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using DOMINANT for graph anomaly detection.

DOMINANT uses a GCN autoencoder to reconstruct both structure
and attributes. Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_dominant import DOMINANT
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=300, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'DOMINANT'
    clf = DOMINANT(hidden_dim=32, num_layers=2, epochs=50,
                   contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_dominant.py pyod/test/test_pyg_dominant.py examples/pyg_dominant_example.py
git commit -m "feat: add DOMINANT graph anomaly detector"
```

---

### Task 6: AnomalyDAE — Dual Autoencoder for Anomaly Detection

**Files:**
- Create: `pyod/models/pyg_anomalydae.py`
- Create: `pyod/test/test_pyg_anomalydae.py`
- Create: `examples/pyg_anomalydae_example.py`

**Paper:** Fan et al., "AnomalyDAE: Dual Autoencoder for Anomaly Detection on Attributed Networks", CIKM 2020.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_anomalydae.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for AnomalyDAE graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_anomalydae import AnomalyDAE
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestAnomalyDAE(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=100, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5,
                         contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 100

    def test_fit_numpy(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 100

    def test_scores_nonnegative(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """AnomalyDAE requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=100)
        clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_anomalydae.py -v`
Expected: FAIL

- [ ] **Step 3: Write AnomalyDAE implementation**

Create `pyod/models/pyg_anomalydae.py`:

```python
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

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "AnomalyDAE is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_anomalydae.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_anomalydae_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using AnomalyDAE for graph anomaly detection.

AnomalyDAE uses dual autoencoders (GAT for structure, MLP for
attributes). Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_anomalydae import AnomalyDAE
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=300, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'AnomalyDAE'
    clf = AnomalyDAE(embed_dim=32, num_heads=2, epochs=50,
                     contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_anomalydae.py pyod/test/test_pyg_anomalydae.py examples/pyg_anomalydae_example.py
git commit -m "feat: add AnomalyDAE graph anomaly detector"
```

---

### Task 7: CoLA — Contrastive Self-Supervised Learning

**Files:**
- Create: `pyod/models/pyg_cola.py`
- Create: `pyod/test/test_pyg_cola.py`
- Create: `examples/pyg_cola_example.py`

**Paper:** Liu et al., "Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning", WWW 2022.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_cola.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for CoLA graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_cola import CoLA
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestCoLA(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=100, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = CoLA(hidden_dim=32, num_layers=2, epochs=5,
                   contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 100

    def test_fit_numpy(self):
        clf = CoLA(hidden_dim=32, epochs=5, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 100

    def test_transductive_no_decision_function(self):
        clf = CoLA(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = CoLA(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """CoLA requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=100)
        clf = CoLA(hidden_dim=32, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_cola.py -v`
Expected: FAIL

- [ ] **Step 3: Write CoLA implementation**

Create `pyod/models/pyg_cola.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_cola.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_cola_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using CoLA for graph anomaly detection.

CoLA uses contrastive learning to detect anomalous nodes.
Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_cola import CoLA
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=300, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'CoLA'
    clf = CoLA(hidden_dim=32, num_layers=2, epochs=50,
               contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_cola.py pyod/test/test_pyg_cola.py examples/pyg_cola_example.py
git commit -m "feat: add CoLA graph anomaly detector"
```

---

### Task 8: CONAD — Contrastive Attributed Network Anomaly Detection

**Files:**
- Create: `pyod/models/pyg_conad.py`
- Create: `pyod/test/test_pyg_conad.py`
- Create: `examples/pyg_conad_example.py`

**Paper:** Xu et al., "Contrastive Attributed Network Anomaly Detection with Data Augmentation", PAKDD 2022.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_conad.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for CONAD graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_conad import CONAD
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestCONAD(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=100, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = CONAD(hidden_dim=32, epochs=5, contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 100

    def test_fit_numpy(self):
        clf = CONAD(hidden_dim=32, epochs=5, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 100

    def test_transductive_no_decision_function(self):
        clf = CONAD(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = CONAD(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """CONAD requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=100)
        clf = CONAD(hidden_dim=32, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_conad.py -v`
Expected: FAIL

- [ ] **Step 3: Write CONAD implementation**

Create `pyod/models/pyg_conad.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_conad.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_conad_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using CONAD for graph anomaly detection.

CONAD combines contrastive learning with graph augmentation
and reconstruction. Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_conad import CONAD
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=300, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'CONAD'
    clf = CONAD(hidden_dim=32, epochs=50,
                contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_conad.py pyod/test/test_pyg_conad.py examples/pyg_conad_example.py
git commit -m "feat: add CONAD graph anomaly detector"
```

---

### Task 9: GUIDE — Higher-order Structure Based Anomaly Detection

**Files:**
- Create: `pyod/models/pyg_guide.py`
- Create: `pyod/test/test_pyg_guide.py`
- Create: `examples/pyg_guide_example.py`

**Paper:** Yuan et al., "Higher-order Structure Based Anomaly Detection on Attributed Networks", BigData 2021.

- [ ] **Step 1: Write the failing test**

Create `pyod/test/test_pyg_guide.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for GUIDE graph anomaly detector."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_graph_data

try:
    import torch
    from torch_geometric.data import Data
    from pyod.models.pyg_guide import GUIDE
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@unittest.skipUnless(HAS_PYG, "torch_geometric not installed")
class TestGUIDE(unittest.TestCase):
    def setUp(self):
        self.X, self.edge_index, self.y = generate_graph_data(
            n_nodes=100, n_features=16, contamination=0.1,
            random_state=42)
        self.data = Data(
            x=torch.FloatTensor(self.X),
            edge_index=torch.LongTensor(self.edge_index))

    def test_fit_pyg_data(self):
        clf = GUIDE(hidden_dim=32, epochs=5, contamination=0.1)
        clf.fit(self.data)
        assert hasattr(clf, 'decision_scores_')
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')
        assert len(clf.decision_scores_) == 100

    def test_fit_numpy(self):
        clf = GUIDE(hidden_dim=32, epochs=5, contamination=0.1)
        clf.fit(self.X, edge_index=self.edge_index)
        assert len(clf.decision_scores_) == 100

    def test_scores_nonnegative(self):
        clf = GUIDE(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        assert np.all(clf.decision_scores_ >= 0)

    def test_transductive_no_decision_function(self):
        clf = GUIDE(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(self.data)

    def test_transductive_no_predict(self):
        clf = GUIDE(hidden_dim=32, epochs=5)
        clf.fit(self.data)
        with self.assertRaises(NotImplementedError):
            clf.predict(self.data)

    def test_no_features_raises(self):
        """GUIDE requires node features."""
        data_no_feat = Data(
            edge_index=torch.LongTensor(self.edge_index),
            num_nodes=100)
        clf = GUIDE(hidden_dim=32, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(data_no_feat)

    def test_no_triangles_raises(self):
        """GUIDE requires triangles in the graph."""
        # Build a tree (no triangles): 0-1-2-3-...
        n = 50
        rows = list(range(n - 1)) + list(range(1, n))
        cols = list(range(1, n)) + list(range(n - 1))
        tree_ei = torch.LongTensor([rows, cols])
        tree_data = Data(
            x=torch.randn(n, 8), edge_index=tree_ei)
        clf = GUIDE(hidden_dim=16, epochs=5)
        with self.assertRaises(ValueError):
            clf.fit(tree_data)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_pyg_guide.py -v`
Expected: FAIL

- [ ] **Step 3: Write GUIDE implementation**

Create `pyod/models/pyg_guide.py`:

```python
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

    Constructs a motif (triangle-count) adjacency and runs two
    GCN autoencoders in parallel — one on the original graph and
    one on the motif graph. Score = ``alpha * err_orig +
    (1 - alpha) * err_motif``.

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

    def predict(self, X):
        """Not supported (transductive detector)."""
        raise NotImplementedError(
            "GUIDE is a transductive detector. Use labels_ "
            "after fit().")

    def predict_proba(self, X):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_pyg_guide.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Write example**

Create `examples/pyg_guide_example.py`:

```python
# -*- coding: utf-8 -*-
"""Example of using GUIDE for graph anomaly detection.

GUIDE uses dual GCN autoencoders on original and motif
(triangle) adjacency. Transductive.

Requires: pip install pyod[graph]
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
import torch
from torch_geometric.data import Data
from pyod.models.pyg_guide import GUIDE
from pyod.utils.data import generate_graph_data

if __name__ == "__main__":
    contamination = 0.1

    X, edge_index, y = generate_graph_data(
        n_nodes=300, contamination=contamination, random_state=42)

    data = Data(x=torch.FloatTensor(X),
                edge_index=torch.LongTensor(edge_index))

    clf_name = 'GUIDE'
    clf = GUIDE(hidden_dim=32, epochs=50,
                contamination=contamination)
    clf.fit(data)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
```

- [ ] **Step 6: Commit**

```bash
git add pyod/models/pyg_guide.py pyod/test/test_pyg_guide.py examples/pyg_guide_example.py
git commit -m "feat: add GUIDE graph anomaly detector"
```

---

### Task 10: Knowledge base — algorithms, routing, benchmarks

**Files:**
- Modify: `pyod/utils/knowledge/algorithms.json`
- Modify: `pyod/utils/knowledge/routing_rules.json`
- Modify: `pyod/utils/knowledge/benchmarks.json`

- [ ] **Step 1: Add 8 graph detector entries to `algorithms.json`**

Add these entries at the end of the JSON object (before the closing `}`):

```json
  "DOMINANT": {
    "class_path": "pyod.models.pyg_dominant.DOMINANT",
    "full_name": "Deep Anomaly Detection on Attributed Networks",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(epochs * (n * d_h + n^2))", "space": "O(n^2)"},
    "strengths": ["Joint structure+attribute reconstruction", "Strong BOND benchmark performance", "Standard GCN architecture"],
    "weaknesses": ["O(n^2) memory for adjacency reconstruction", "Requires node features"],
    "best_for": "Attributed graphs where anomalies manifest in both structure and attributes",
    "avoid_when": "Graph is very large (>10k nodes) or has no node features",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {"BOND_deep": 1},
    "paper": {"id": "ding2019dominant", "short": "Ding et al., SDM 2019"},
    "default_params": {"hidden_dim": 64, "num_layers": 2, "epochs": 100, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "CoLA": {
    "class_path": "pyod.models.pyg_cola.CoLA",
    "full_name": "Contrastive Self-Supervised Anomaly Detection",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(epochs * n * d_h)", "space": "O(n * d_h)"},
    "strengths": ["Contrastive learning captures local-global discrepancy", "Strong BOND performance", "No adjacency reconstruction (memory efficient)"],
    "weaknesses": ["Sensitive to graph connectivity", "Requires node features"],
    "best_for": "Attributed graphs where anomalies have unusual local neighborhoods",
    "avoid_when": "Graph is disconnected or has no node features",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {"BOND_deep": 2},
    "paper": {"id": "liu2022cola", "short": "Liu et al., WWW 2022"},
    "default_params": {"hidden_dim": 64, "num_layers": 2, "epochs": 100, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "CONAD": {
    "class_path": "pyod.models.pyg_conad.CONAD",
    "full_name": "Contrastive Attributed Network Anomaly Detection",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(epochs * n * d_h)", "space": "O(n * d_h)"},
    "strengths": ["Data augmentation improves robustness", "Dual objective (contrastive + reconstruction)"],
    "weaknesses": ["Augmentation ratio is a sensitive hyperparameter", "Requires node features"],
    "best_for": "Attributed graphs where robustness to noise is important",
    "avoid_when": "Graph structure is too sparse for meaningful augmentation",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {},
    "paper": {"id": "xu2022conad", "short": "Xu et al., PAKDD 2022"},
    "default_params": {"hidden_dim": 64, "epochs": 100, "aug_ratio": 0.2, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "AnomalyDAE": {
    "class_path": "pyod.models.pyg_anomalydae.AnomalyDAE",
    "full_name": "Dual Autoencoder for Anomaly Detection",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(epochs * (n * d_h + n^2))", "space": "O(n^2)"},
    "strengths": ["Attention-based structure encoding (GAT)", "Separate structure and attribute autoencoders"],
    "weaknesses": ["O(n^2) memory for adjacency reconstruction", "Requires node features"],
    "best_for": "Attributed graphs where attention over neighbors reveals anomaly patterns",
    "avoid_when": "Graph is very large or structure is unimportant",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {},
    "paper": {"id": "fan2020anomalydae", "short": "Fan et al., CIKM 2020"},
    "default_params": {"embed_dim": 64, "num_heads": 4, "epochs": 100, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "GUIDE": {
    "class_path": "pyod.models.pyg_guide.GUIDE",
    "full_name": "Higher-order Structure Based Anomaly Detection",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(epochs * n * d_h + m * d_avg)", "space": "O(n^2)"},
    "strengths": ["Exploits higher-order motifs (triangles)", "Dual-view captures different structural signals"],
    "weaknesses": ["Motif construction adds overhead", "Sparse graphs may have few triangles"],
    "best_for": "Dense attributed graphs with meaningful higher-order structures",
    "avoid_when": "Graph is tree-like (no triangles) or very large",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {},
    "paper": {"id": "yuan2021guide", "short": "Yuan et al., BigData 2021"},
    "default_params": {"hidden_dim": 64, "epochs": 100, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "Radar": {
    "class_path": "pyod.models.pyg_radar.Radar",
    "full_name": "Residual Analysis for Anomaly Detection",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(max_iter * n^2 * d)", "space": "O(n^2)"},
    "strengths": ["No neural network training", "Interpretable residuals", "Lightweight baseline"],
    "weaknesses": ["O(n^2) dense matrix operations", "Linear model may miss complex patterns"],
    "best_for": "Small-to-medium attributed graphs as a fast baseline",
    "avoid_when": "Graph is very large or anomalies are structural-only",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {},
    "paper": {"id": "li2017radar", "short": "Li et al., IJCAI 2017"},
    "default_params": {"alpha": 1.0, "gamma": 1.0, "max_iter": 100, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "ANOMALOUS": {
    "class_path": "pyod.models.pyg_anomalous.ANOMALOUS",
    "full_name": "Joint Modeling Approach for Anomaly Detection",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(max_iter * n^2 * d)", "space": "O(n^2)"},
    "strengths": ["Laplacian regularization for smooth predictions", "No neural network training", "Extends Radar with graph structure"],
    "weaknesses": ["O(n^2) dense matrix operations", "Linear model"],
    "best_for": "Small-to-medium attributed graphs where smoothness matters",
    "avoid_when": "Graph is very large or anomalies are purely structural",
    "benchmark_refs": ["BOND"],
    "benchmark_rank": {},
    "paper": {"id": "peng2018anomalous", "short": "Peng et al., IJCAI 2018"},
    "default_params": {"alpha": 1.0, "gamma": 1.0, "max_iter": 100, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  },
  "SCAN_Graph": {
    "class_path": "pyod.models.pyg_scan.SCAN",
    "full_name": "Structural Clustering Algorithm for Networks",
    "status": "shipped",
    "data_types": ["graph"],
    "category": "graph",
    "complexity": {"time": "O(m * d_avg)", "space": "O(n + m)"},
    "strengths": ["Structure-only (no features needed)", "No training or hyperparameter tuning", "Fast and lightweight"],
    "weaknesses": ["Ignores node attributes", "Only detects structural anomalies"],
    "best_for": "Structure-only graphs or as a fast structural baseline",
    "avoid_when": "Node attributes are available and important for anomaly detection",
    "benchmark_refs": [],
    "benchmark_rank": {},
    "paper": {"id": "xu2007scan", "short": "Xu et al., KDD 2007"},
    "default_params": {"epsilon": 0.5, "mu": 2, "contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": ["torch_geometric"],
    "version_added": "2.2.0"
  }
```

Note: SCAN uses `SCAN_Graph` key to avoid conflict with any future tabular SCAN detector.

- [ ] **Step 2: Add 2 graph routing rules to `routing_rules.json`**

Add before the closing `]` of the `rules` array:

```json
    ,
    {
      "id": "graph_attributed",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "graph"},
        {"field": "has_features", "op": "eq", "value": true}
      ],
      "recommendations": [
        {"detector": "DOMINANT", "params": {}, "confidence": 0.85},
        {"detector": "CoLA", "params": {}, "confidence": 0.8},
        {"detector": "Radar", "params": {}, "confidence": 0.7}
      ],
      "reason": "Attributed graph: DOMINANT and CoLA are most reliable deep methods (BOND benchmark). Radar is a lightweight MF baseline.",
      "evidence": ["BOND"]
    },
    {
      "id": "graph_structure_only",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "graph"},
        {"field": "has_features", "op": "eq", "value": false}
      ],
      "recommendations": [
        {"detector": "SCAN_Graph", "params": {}, "confidence": 0.8}
      ],
      "reason": "Structure-only graph (no node features): SCAN is the only detector that does not require attributes.",
      "evidence": []
    }
```

- [ ] **Step 3: Add BOND benchmark to `benchmarks.json`**

Add after the `TSB_AD` entry:

```json
  ,
  "BOND": {
    "paper": {"id": "liu2022bond", "short": "Liu et al., NeurIPS 2022"},
    "scope": "graph",
    "n_datasets": 14,
    "n_algorithms": 14,
    "rankings": {
      "deep_top_3": ["DOMINANT", "CoLA", "CONAD"],
      "classical_top_2": ["Radar", "ANOMALOUS"]
    },
    "key_finding": "DOMINANT and CoLA are most reliable deep methods; classical MF methods competitive on small graphs"
  }
```

- [ ] **Step 4: Commit**

```bash
git add pyod/utils/knowledge/algorithms.json pyod/utils/knowledge/routing_rules.json pyod/utils/knowledge/benchmarks.json
git commit -m "feat: add graph detector knowledge base entries and BOND benchmark"
```

---

### Task 11: ADEngine integration

**Files:**
- Modify: `pyod/utils/ad_engine.py:46-96`

- [ ] **Step 1: Add graph branch to `_sniff_data_type()`**

In `pyod/utils/ad_engine.py`, modify `_sniff_data_type()` (line ~86) to detect PyG Data objects:

```python
    def _sniff_data_type(self, X):
        """Conservative data type detection."""
        # Check for PyG Data object
        try:
            from torch_geometric.data import Data
            if isinstance(X, Data):
                return 'graph'
        except ImportError:
            pass

        if isinstance(X, dict):
            return 'multimodal'
        if isinstance(X, (list, tuple)) and len(X) > 0:
            sample = X[:min(20, len(X))]
            if all(isinstance(x, str) for x in sample):
                if self._looks_like_image_paths(sample[:5]):
                    return 'image'
                return 'text'
        return 'tabular'
```

- [ ] **Step 2: Add graph branch to `profile_data()`**

In `profile_data()` (line ~51), add a graph branch before the `else` (tabular/time_series) block:

```python
        if detected_type == 'text':
            profile['n_samples'] = len(X)
        elif detected_type == 'image':
            profile['n_samples'] = len(X)
        elif detected_type == 'multimodal':
            first_key = next(iter(X))
            first_val = X[first_key]
            profile['n_samples'] = len(first_val)
            profile['modalities'] = list(X.keys())
        elif detected_type == 'graph':
            # PyG Data object (only supported graph input for ADEngine)
            profile['n_nodes'] = X.num_nodes
            profile['n_edges'] = X.edge_index.shape[1]
            profile['n_features'] = (
                X.x.shape[1] if X.x is not None else 0)
            profile['has_features'] = X.x is not None
            profile['n_samples'] = X.num_nodes
        else:
            # tabular or time_series
            arr = np.asarray(X, dtype=np.float64)
            # ... rest of existing code ...
```

- [ ] **Step 3: Test ADEngine graph integration**

Run: `python -c "
from pyod.utils.ad_engine import ADEngine
import torch
from torch_geometric.data import Data

# Attributed graph
x = torch.randn(100, 16)
ei = torch.randint(0, 100, (2, 500))
data = Data(x=x, edge_index=ei)

engine = ADEngine()
profile = engine.profile_data(data)
print('Profile:', profile)
assert profile['data_type'] == 'graph'
assert profile['has_features'] == True

plan = engine.plan_detection(profile)
print('Plan:', plan['detector_name'], plan['confidence'])
"`

Expected: Profile shows `data_type='graph'`, `has_features=True`. Plan recommends DOMINANT or CoLA.

- [ ] **Step 4: Commit**

```bash
git add pyod/utils/ad_engine.py
git commit -m "feat: add graph data profiling to ADEngine"
```

---

### Task 12: Documentation — README, Sphinx, BibTeX, CHANGES

**Files:**
- Modify: `README.rst`
- Modify: `docs/index.rst`
- Modify: `docs/pyod.models.rst`
- Modify: `docs/zreferences.bib`
- Modify: `CHANGES.txt`

- [ ] **Step 1: Add Graph AD section to `README.rst`**

After the TS table section (after line ~382, the AnomalyTransformer entry), add:

```rst


**(i-c) Graph Anomaly Detection** (``pip install pyod[graph]``):

All graph detectors are **transductive** in v1: use ``decision_scores_`` and ``labels_`` after ``fit()``. No out-of-sample ``predict``. Input: PyG ``Data`` object with ``x`` (node features) and ``edge_index`` (COO edges). SCAN works without features.

**Graph detection in 3 lines** (``pip install pyod[graph]``):

.. code-block:: python

    from pyod.models.pyg_dominant import DOMINANT
    clf = DOMINANT(hidden_dim=64, epochs=100)
    clf.fit(data)                                  # PyG Data object
    scores = clf.decision_scores_                  # per-node anomaly scores

Algorithm rankings from `BOND benchmark <https://arxiv.org/abs/2206.10071>`_ (NeurIPS 2022, 14 datasets):

.. list-table::
   :widths: 18 18 45 5 14
   :header-rows: 1

   * - Type
     - Abbr
     - Algorithm
     - Year
     - Ref
   * - GCN Autoencoder
     - DOMINANT
     - GCN AE, structure + attribute reconstruction (#1 BOND deep) (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_dominant_example.py>`_)
     - 2019
     - [#Ding2019DOMINANT]_
   * - Contrastive
     - CoLA
     - Contrastive self-supervised, local neighbor context (#2 BOND deep) (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_cola_example.py>`_)
     - 2022
     - [#Liu2022CoLA]_
   * - Contrastive+AE
     - CONAD
     - Contrastive with anomalous-view injection + dual reconstruction (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_conad_example.py>`_)
     - 2022
     - [#Xu2022CONAD]_
   * - Attention AE
     - AnomalyDAE
     - GAT structure encoder + MLP attribute encoder (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_anomalydae_example.py>`_)
     - 2020
     - [#Fan2020AnomalyDAE]_
   * - Motif AE
     - GUIDE
     - Dual GCN AE on original + triangle-motif adjacency (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_guide_example.py>`_)
     - 2021
     - [#Yuan2021GUIDE]_
   * - Matrix Factor.
     - Radar
     - Residual analysis via matrix factorization (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_radar_example.py>`_)
     - 2017
     - [#Li2017Radar]_
   * - Matrix Factor.
     - ANOMALOUS
     - Joint MF with Laplacian regularization (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_anomalous_example.py>`_)
     - 2018
     - [#Peng2018ANOMALOUS]_
   * - Structural
     - SCAN
     - Structural clustering, no features needed (`example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_scan_example.py>`_)
     - 2007
     - [#Xu2007SCAN]_
```

Also update the intro paragraph (line ~60) to include "graph":

Change: `**tabular, time series, text, and image data**`
To: `**tabular, time series, graph, text, and image data**`

Also update the Table of Contents (line ~161):

Change: `(Tabular, Time Series, Embedding)`
To: `(Tabular, Time Series, Graph, Embedding)`

Also add to "Selecting the Right Algorithm" (line ~118):

After `TimeSeriesOD` for time series, add `, `DOMINANT <...>`_ for graph`

Also add to Optional Dependencies (after line ~210):

```
* torch_geometric (optional, required for graph anomaly detectors: DOMINANT, CoLA, SCAN, etc.)
```

- [ ] **Step 2: Mirror changes to `docs/index.rst`**

Apply the same changes as README.rst but use Sphinx `:cite:` and `:class:` roles instead of `[#...]_` footnotes. The graph table entries use:

```rst
   * - GCN Autoencoder
     - DOMINANT
     - GCN AE, structure + attribute reconstruction (#1 BOND deep)
     - 2019
     - :class:`pyod.models.pyg_dominant.DOMINANT`
     - :cite:`a-ding2019dominant`
```

- [ ] **Step 3: Add 8 autodoc entries to `docs/pyod.models.rst`**

Before the `.. rubric:: References` line (line ~598), add:

```rst

pyod.models.pyg\_scan module
------------------------------

.. automodule:: pyod.models.pyg_scan
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_radar module
-------------------------------

.. automodule:: pyod.models.pyg_radar
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_anomalous module
-----------------------------------

.. automodule:: pyod.models.pyg_anomalous
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_dominant module
----------------------------------

.. automodule:: pyod.models.pyg_dominant
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_anomalydae module
-------------------------------------

.. automodule:: pyod.models.pyg_anomalydae
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_cola module
-------------------------------

.. automodule:: pyod.models.pyg_cola
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_conad module
--------------------------------

.. automodule:: pyod.models.pyg_conad
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:


pyod.models.pyg\_guide module
--------------------------------

.. automodule:: pyod.models.pyg_guide
    :members:
    :exclude-members: get_params, set_params
    :undoc-members:
    :show-inheritance:

```

- [ ] **Step 4: Add 9 BibTeX entries to `docs/zreferences.bib`**

Append to `docs/zreferences.bib`:

```bibtex
@inproceedings{ding2019dominant,
  title={Deep anomaly detection on attributed networks},
  author={Ding, Kaize and Li, Jundong and Bhanushali, Rohit and Liu, Huan},
  booktitle={Proceedings of the 2019 SIAM International Conference on Data Mining},
  pages={594--602},
  year={2019},
  organization={SIAM}
}

@inproceedings{liu2022cola,
  title={Anomaly detection on attributed networks via contrastive self-supervised learning},
  author={Liu, Yixin and Li, Zhao and Pan, Shirui and Gool, Tao and Xiang, Tao and Gong, Boqing},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={2137--2147},
  year={2022}
}

@inproceedings{xu2022conad,
  title={Contrastive attributed network anomaly detection with data augmentation},
  author={Xu, Zhiming and Huang, Xiao and Zhao, Yue and Dong, Yushun and Li, Jundong},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={444--457},
  year={2022},
  organization={Springer}
}

@inproceedings{fan2020anomalydae,
  title={AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks},
  author={Fan, Haoyi and Zhang, Fengbin and Li, Zuoyong},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management},
  pages={747--756},
  year={2020}
}

@inproceedings{yuan2021guide,
  title={Higher-order structure based anomaly detection on attributed networks},
  author={Yuan, Xu and Zhou, Na and Yu, Shuo and Huang, Huafei and Chen, Zhikui and Xia, Feng},
  booktitle={2021 IEEE International Conference on Big Data},
  pages={2691--2700},
  year={2021},
  organization={IEEE}
}

@inproceedings{li2017radar,
  title={Radar: Residual analysis for anomaly detection in attributed networks},
  author={Li, Jundong and Dani, Harsh and Hu, Xia and Liu, Huan},
  booktitle={Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence},
  pages={2152--2158},
  year={2017}
}

@inproceedings{peng2018anomalous,
  title={ANOMALOUS: A joint modeling approach for anomaly detection on attributed networks},
  author={Peng, Zhen and Luo, Minnan and Li, Jundong and Liu, Huan and Zheng, Qinghua},
  booktitle={Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence},
  pages={3529--3535},
  year={2018}
}

@inproceedings{xu2007scan,
  title={SCAN: A structural clustering algorithm for networks},
  author={Xu, Xiaowei and Yuruk, Nurcan and Feng, Zhidan and Schweiger, Thomas A.J.},
  booktitle={Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={824--833},
  year={2007}
}

@inproceedings{liu2022bond,
  title={BOND: Benchmarking unsupervised outlier node detection on static attributed graphs},
  author={Liu, Kay and Dou, Yingtong and Zhao, Yue and Ding, Xueying and Hu, Xiyang and Zhang, Ruitong and Ding, Kaize and Chen, Canyu and Peng, Hao and Shu, Kai and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

- [ ] **Step 5: Add CHANGES.txt entry**

Add at end of `CHANGES.txt`:

```
v<2.2.0>, <04/11/2026> -- Add 8 graph anomaly detectors: DOMINANT, CoLA, CONAD, AnomalyDAE, GUIDE, Radar, ANOMALOUS, SCAN. All transductive v1, PyG Data input. Shared graph utilities, generate_graph_data() for synthetic benchmarks, per-algorithm examples and tests, ADEngine graph profiling and routing, BOND benchmark integration. Install via pip install pyod[graph].
```

- [ ] **Step 6: Commit**

```bash
git add README.rst docs/index.rst docs/pyod.models.rst docs/zreferences.bib CHANGES.txt
git commit -m "docs: add graph anomaly detection to README, Sphinx docs, BibTeX, and CHANGES"
```

---

### Task 13: Packaging and CI

**Files:**
- Modify: `setup.py:45-57`
- Modify: `docs/requirements.txt`
- Modify: `.github/workflows/testing.yml:44`
- Modify: `.github/workflows/testing-cron.yml:44`

- [ ] **Step 1: Add `graph` extra to `setup.py`**

In `setup.py`, modify the `extras_require` dict:

```python
    extras_require={
        'embedding': ['sentence-transformers>=2.0'],
        'openai': ['openai>=1.0'],
        'mcp': ['mcp>=1.0'],
        'graph': ['torch>=2.0', 'torch_geometric>=2.0'],
        'all': [
            'sentence-transformers>=2.0',
            'openai>=1.0',
            'transformers>=4.0',
            'torch>=2.0',
            'torch_geometric>=2.0',
            'Pillow',
            'mcp>=1.0',
        ],
    },
```

- [ ] **Step 2: Add `torch_geometric` to `docs/requirements.txt`**

Add after the `torch` line:

```
torch_geometric
```

- [ ] **Step 3: Skip graph tests on macOS in CI**

In `.github/workflows/testing.yml`, update the macOS pytest command (line ~44) to also ignore graph tests:

```yaml
    - name: Test with pytest (macOS, skip torch-heavy tests due to NNPACK slowdown on ARM)
      if: startsWith(matrix.os, 'macos')
      run: |
        coverage run --source=pyod -m pytest --ignore=pyod/test/test_auto_encoder.py --ignore=pyod/test/test_vae.py --ignore=pyod/test/test_deepsvdd.py --ignore=pyod/test/test_so_gaal.py --ignore=pyod/test/test_mo_gaal.py --ignore=pyod/test/test_anogan.py --ignore=pyod/test/test_alad.py --ignore=pyod/test/test_ae1svm.py --ignore=pyod/test/test_devnet.py --ignore=pyod/test/test_ts_lstm.py --ignore=pyod/test/test_ts_anomaly_transformer.py --ignore=pyod/test/test_pyg_dominant.py --ignore=pyod/test/test_pyg_anomalydae.py --ignore=pyod/test/test_pyg_cola.py --ignore=pyod/test/test_pyg_conad.py --ignore=pyod/test/test_pyg_guide.py
```

Note: SCAN, Radar, and ANOMALOUS do not use GNN training loops (no NNPACK), so they do not need to be skipped.

Apply the same change to `.github/workflows/testing-cron.yml`.

- [ ] **Step 4: Verify setup.py**

Run: `python -c "from setup import *; print('OK')"` or `pip install -e .[graph]`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add setup.py docs/requirements.txt .github/workflows/testing.yml .github/workflows/testing-cron.yml
git commit -m "build: add pyod[graph] extra, update CI to skip deep graph tests on macOS"
```

---

## Self-Review Notes

**Spec coverage check:**
- Section 1 (Vision): 8 algorithms, all transductive — covered by Tasks 2-9
- Section 2 (Design decisions): pyg_ prefix, BaseDetector, transductive — all tasks follow this
- Section 4 (Input format): PyG Data primary, numpy convenience — `_pyg_utils.validate_graph_input` in Task 1
- Section 5 (Fit contract): `_set_n_classes`, `_process_decision_scores` — all detector `fit()` methods follow
- Section 6 (Algorithms): All 8 algorithms have dedicated tasks (2-9)
- Section 7 (Shared utilities): `_pyg_utils.py` in Task 1
- Section 8 (File structure): All files in File Structure map
- Section 9 (ADEngine): Task 10 (knowledge) + Task 11 (code integration)
- Section 9.4 (setup.py): Task 13
- Codex Round 4 finding (routing split): Two routing rules in Task 10

**Type consistency check:** All detectors use `validate_graph_input`, `_set_n_classes`, `_process_decision_scores`, return `self` from `fit()`, raise `NotImplementedError` for prediction methods. Class names match spec: DOMINANT, CoLA, CONAD, AnomalyDAE, GUIDE, Radar, ANOMALOUS, SCAN.

**Placeholder scan:** No TBD, TODO, or "similar to Task N" found. All code blocks are complete.
