# Graph Anomaly Detection for PyOD

**Date:** 2026-04-09
**Status:** Draft (v2 -- Round 1 review fixes)
**Version:** 2

---

## 1. Vision

Add graph anomaly detection as first-class PyOD functionality. 8 algorithms covering GNN-based, contrastive, matrix factorization, and clustering approaches. All inherit `BaseDetector`, accept PyG `Data` objects as input, and output per-node anomaly scores. All detectors are **transductive** in v1 (trained and scored on the same graph).

**No breaking changes.** Everything is additive.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target user | Researchers + practitioners who have graph data | Graph AD is a specialized domain |
| Dependencies | `torch_geometric` as optional extra (`pip install pyod[graph]`). ALL graph detectors require this. | PyG is the universal standard in graph OD (2024-2026 papers all use it) |
| File convention | `pyg_` prefix, flat in `pyod/models/` | Signals PyG dependency, consistent with `ts_` |
| Class naming | Original paper names (DOMINANT, CoLA, etc.) | Well-known in literature |
| Input format | Primary: PyG `Data` objects. Convenience: numpy features + edge_index for direct `fit()` calls only. Inherited methods (`predict`, `predict_proba`, ADEngine) require PyG `Data`. | PyG `Data` is the standard; numpy is a convenience layer |
| Output format | `decision_scores_` of shape `(n_nodes,)` | One score per node |
| Base class | `BaseDetector` | Full PyOD compatibility |
| Transductive (v1) | ALL 8 detectors are transductive. `decision_function`/`predict`/`predict_proba`/`predict_confidence` raise `NotImplementedError`. Users consume `decision_scores_` and `labels_` after `fit()`. | Graph OD methods are inherently single-graph; out-of-sample scoring is not well-defined without specifying graph context. Same pattern as MatrixProfile. |
| Implementation | Fresh from papers, using PyG layers. PyGOD as reference only, not a dependency | Independence from PyGOD |
| GNN layers | Use PyG's built-in layers (`GCNConv`, `GATConv`, etc.) directly | No custom GNN infrastructure needed |

---

## 3. Paper-Reading Requirement

Before implementing each algorithm, the implementer MUST read the original paper. Implementations must be faithful to the core algorithm. Each detector's docstring must cite the paper.

**BibTeX entries to add to `docs/zreferences.bib`:**

| Algorithm | Paper | Venue |
|-----------|-------|-------|
| DOMINANT | Ding et al., "Deep Anomaly Detection on Attributed Networks" | SDM 2019 |
| CoLA | Liu et al., "Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning" | WWW 2022 |
| CONAD | Xu et al., "Contrastive Attributed Network Anomaly Detection with Data Augmentation" | PAKDD 2022 |
| AnomalyDAE | Fan et al., "AnomalyDAE: Dual Autoencoder for Anomaly Detection on Attributed Networks" | CIKM 2020 |
| GUIDE | Yuan et al., "Higher-order Structure Based Anomaly Detection on Attributed Networks" | BigData 2021 |
| Radar | Li et al., "Radar: Residual Analysis for Anomaly Detection in Attributed Networks" | IJCAI 2017 |
| ANOMALOUS | Peng et al., "ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks" | ICDM 2018 |
| SCAN | Xu et al., "SCAN: A Structural Clustering Algorithm for Networks" | KDD 2007 |
| BOND benchmark | Liu et al., "BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs" | NeurIPS 2022 |

---

## 4. Input Format

### Primary input: PyG `Data` object

```python
from torch_geometric.data import Data

data = Data(x=node_features, edge_index=edge_index)
clf.fit(data)
scores = clf.decision_scores_  # per-node anomaly scores
labels = clf.labels_            # binary labels
```

This is the only format supported through inherited BaseDetector methods and ADEngine. `data.x` (node features) is required for all detectors except SCAN. **SCAN structure-only contract:** when `data.x is None`, SCAN computes degree features internally. In this case `data.num_nodes` MUST be explicitly set on the `Data` object (PyG cannot infer it reliably from `edge_index` alone when isolated nodes exist).

### Convenience: numpy + edge_index (direct `fit()` only)

```python
clf.fit(X, edge_index=edge_index)
# X: numpy array (n_nodes, n_features) -- node attributes
# edge_index: numpy array (2, n_edges) -- COO edge list
```

This is converted to PyG `Data` internally. NOT usable through `predict()`, `predict_proba()`, or ADEngine -- those paths require PyG `Data`. This is a convenience for users who have numpy data and want a quick fit.

### Validation utility (`_pyg_utils.py`)

```python
def validate_graph_input(X, edge_index=None):
    """Convert input to PyG Data object.

    Accepts:
    - PyG Data object (returned as-is)
    - numpy X + numpy edge_index (converted to Data)

    Returns
    -------
    data : torch_geometric.data.Data
    """
    import torch
    from torch_geometric.data import Data

    if isinstance(X, Data):
        return X

    # numpy input
    x = torch.FloatTensor(np.asarray(X, dtype=np.float32))
    if edge_index is None:
        raise ValueError("edge_index required when X is not a PyG Data object")
    ei = torch.LongTensor(np.asarray(edge_index, dtype=np.int64))
    return Data(x=x, edge_index=ei)
```

---

## 5. BaseDetector Fit Contract

All graph detectors follow this pattern:

```python
def fit(self, X, y=None, edge_index=None):
    # 1. Validate and convert input
    data = validate_graph_input(X, edge_index)
    self._n_nodes = data.num_nodes

    # 2. Set n_classes
    self._set_n_classes(y)

    # 3. Validate feature requirements (per-detector)
    #    Most detectors require data.x; SCAN can use degree features.
    self._validate_features(data)

    # 4. Train model and compute anomaly scores (algorithm-specific)
    scores = self._train_and_score(data)

    # 5. Set decision scores and process
    self.decision_scores_ = scores  # shape (n_nodes,)
    self._process_decision_scores()

    return self
```

No masked-score workflow needed (unlike TS) -- graph detectors produce one score per node with no gaps.

### Transductive (all 8 detectors in v1)

**(Addressing Codex finding #2.)** All graph detectors are transductive in v1. The following methods raise `NotImplementedError`:

- `decision_function(X)`
- `predict(X)`
- `predict_proba(X)`
- `predict_confidence(X)`

Users consume `decision_scores_` and `labels_` after `fit()`, same as MatrixProfile in the TS module.

**Rationale:** Graph anomaly detection methods score nodes within the context of a specific graph. "Out-of-sample" prediction requires defining what a test graph means (new nodes added? entirely new graph? same structure, different attributes?). None of the 8 papers define this clearly. Rather than inventing an untested adaptation, we follow the MatrixProfile precedent and make transductivity explicit.

**ADEngine integration:** Graph detectors participate in routing for `data_type='graph'`. Since all are transductive, `ADEngine.run_detection()` calls `clf.fit(X_train)` normally but skips `X_test` scoring (catches `NotImplementedError`, already implemented for MatrixProfile). `detect(X_train)` works end-to-end; `detect(X_train, X_test=...)` returns `scores_test=None, labels_test=None`.

**ADEngine `profile_data()` patch:** Add a dedicated graph branch before the `np.asarray` coercion:

```python
def _sniff_data_type(self, X):
    # Check for PyG Data object
    try:
        from torch_geometric.data import Data
        if isinstance(X, Data):
            return 'graph'
    except ImportError:
        pass
    # ... existing logic ...
```

**Graph profile schema:** When `data_type='graph'`:
```python
profile = {
    'data_type': 'graph',
    'n_nodes': data.num_nodes,
    'n_edges': data.edge_index.shape[1],  # raw COO count (includes reciprocal edges for undirected graphs)
    'n_features': data.x.shape[1] if data.x is not None else 0,
    'has_features': data.x is not None,
}
```

---

## 6. Algorithms

### 6.1 Overview

All detectors require `pyod[graph]` (`torch_geometric`). All are transductive in v1.

| File | Class | Category | Uses GNN layers? |
|------|-------|----------|-----------------|
| `pyg_dominant.py` | `DOMINANT` | GNN+AE (reconstruction) | Yes (`GCNConv`) |
| `pyg_cola.py` | `CoLA` | GNN+contrastive | Yes (`GCNConv`/`GINConv`) |
| `pyg_conad.py` | `CONAD` | GNN+contrastive+AE | Yes (`GCNConv`) |
| `pyg_anomalydae.py` | `AnomalyDAE` | Attention+AE | Yes (`GATConv`) |
| `pyg_guide.py` | `GUIDE` | GNN+AE (dual motif) | Yes (`GCNConv`) |
| `pyg_radar.py` | `Radar` | Matrix factorization | No (numpy/scipy ops on PyG Data) |
| `pyg_anomalous.py` | `ANOMALOUS` | Matrix factorization | No (numpy/scipy ops on PyG Data) |
| `pyg_scan.py` | `SCAN` | Structural clustering | No (numpy/scipy ops on PyG Data) |

### 6.2 DOMINANT

**Original method:** Ding et al. (SDM 2019). GCN autoencoder that jointly reconstructs adjacency matrix and node attributes. Anomaly score = weighted sum of structure reconstruction error and attribute reconstruction error.

**Architecture:**
- Encoder: multi-layer GCN (`GCNConv` from PyG)
- Structure decoder: inner product of embeddings → reconstructed adjacency
- Attribute decoder: linear/GCN layer → reconstructed attributes
- Loss: `alpha * structure_loss + (1-alpha) * attribute_loss`
- Anomaly score per node: same loss computed per-node after training

**Parameters:** `hidden_dim` (64), `num_layers` (2), `dropout` (0.3), `alpha` (0.5), `epochs` (100), `lr` (5e-3), `contamination` (0.1).

### 6.3 CoLA

**Original method:** Liu et al. (WWW 2022). Contrastive self-supervised learning. Contrasts target node's local subgraph patch against randomly sampled patches. Nodes whose patches are hard to distinguish from random are anomalous.

**Architecture:**
- Subgraph sampler: extracts k-hop subgraph around each node
- GNN encoder: encodes subgraphs
- Readout: global summary vector (mean pooling)
- Discriminator: bilinear function scoring (patch, summary) pairs
- Anomaly score: negative discriminator score (anomalies score low)

**Parameters:** `hidden_dim` (64), `num_layers` (2), `subgraph_size` (4), `epochs` (100), `lr` (1e-3), `contamination` (0.1).

### 6.4 CONAD

**Original method:** Xu et al. (PAKDD 2022). Extends contrastive detection with graph augmentation (edge/attribute perturbation) and dual objectives (contrastive + reconstruction).

**Architecture:**
- Augmentation: random edge drop + attribute masking
- GCN encoder shared across views
- Contrastive loss between augmented views
- Reconstruction loss for attributes
- Score: combined contrastive + reconstruction error

**Parameters:** `hidden_dim` (64), `num_layers` (2), `aug_ratio` (0.2), `alpha` (0.5), `epochs` (100), `lr` (1e-3), `contamination` (0.1).

### 6.5 AnomalyDAE

**Original method:** Fan et al. (CIKM 2020). Dual autoencoder with attention-based structure encoder and MLP attribute encoder. Separate decoders for each modality.

**Architecture:**
- Structure encoder: attention-based (`GATConv` from PyG)
- Attribute encoder: MLP
- Structure decoder: inner product
- Attribute decoder: MLP
- Score: `alpha * structure_error + (1-alpha) * attribute_error`

**Parameters:** `embed_dim` (64), `num_heads` (4), `alpha` (0.5), `dropout` (0.3), `epochs` (100), `lr` (5e-3), `contamination` (0.1).

### 6.6 GUIDE

**Original method:** Yuan et al. (BigData 2021). Exploits higher-order structures (motifs) alongside attributes. Constructs motif-based adjacency and runs dual GCN autoencoders.

**Architecture:**
- Construct motif adjacency (e.g., triangle-based)
- GCN AE on original adjacency → attribute reconstruction
- GCN AE on motif adjacency → motif reconstruction
- Score: combined reconstruction errors

**Parameters:** `hidden_dim` (64), `num_layers` (2), `alpha` (0.5), `epochs` (100), `lr` (5e-3), `contamination` (0.1).

### 6.7 Radar

**Original method:** Li et al. (IJCAI 2017). Matrix factorization approach. Models node attributes as `X = AW + R` where A is adjacency, W is weight matrix, R is residual. Nodes with large residuals are anomalous.

**PyOD adaptation:** No GNN layers needed. Uses scipy sparse operations on the adjacency and numpy for the factorization. Accepts PyG `Data` but extracts adjacency and features internally.

**Transductive:** `decision_function` raises `NotImplementedError`.

**Parameters:** `alpha` (1.0), `gamma` (1.0), `max_iter` (100), `contamination` (0.1).

### 6.8 ANOMALOUS

**Original method:** Peng et al. (ICDM 2018). Extends Radar with joint CUR decomposition that captures both structural and attribute anomalies. Alternating optimization.

**Transductive:** `decision_function` raises `NotImplementedError`.

**Parameters:** `alpha` (1.0), `gamma` (1.0), `max_iter` (100), `contamination` (0.1).

### 6.9 SCAN

**Original method:** Xu et al. (KDD 2007). Structural clustering. Computes structural similarity between connected nodes (ratio of shared neighbors). Density-based clustering with epsilon/mu thresholds. Unclustered nodes = outliers.

**PyOD adaptation:** No GNN needed. Classical graph algorithm. Uses scipy sparse for adjacency operations. Outputs anomaly score based on structural similarity to neighbors (low similarity = anomalous).

**Transductive:** `decision_function` raises `NotImplementedError`.

**Parameters:** `epsilon` (0.5), `mu` (2), `contamination` (0.1).

---

## 7. Shared Utilities (`pyod/models/_pyg_utils.py`)

Thin utility -- only input validation and conversion:

```python
def validate_graph_input(X, edge_index=None):
    """Convert input to PyG Data. Accepts PyG Data or numpy arrays."""

def to_dense_adj_numpy(edge_index, num_nodes):
    """Convert edge_index to dense adjacency matrix (numpy)."""

def to_sparse_adj(edge_index, num_nodes):
    """Convert edge_index to scipy sparse adjacency."""
```

No custom GNN layers. PyG provides all needed layers.

---

## 8. File Structure

### New files

| File | Responsibility |
|------|---------------|
| `pyod/models/_pyg_utils.py` | Input validation + format conversion |
| `pyod/models/pyg_dominant.py` | DOMINANT |
| `pyod/models/pyg_cola.py` | CoLA |
| `pyod/models/pyg_conad.py` | CONAD |
| `pyod/models/pyg_anomalydae.py` | AnomalyDAE |
| `pyod/models/pyg_guide.py` | GUIDE |
| `pyod/models/pyg_radar.py` | Radar |
| `pyod/models/pyg_anomalous.py` | ANOMALOUS |
| `pyod/models/pyg_scan.py` | SCAN |
| `pyod/test/test_pyg_*.py` | Per-algorithm tests (skip if PyG not installed) |
| `examples/pyg_*_example.py` | Per-algorithm examples |

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/knowledge/algorithms.json` | Add 8 entries with BOND benchmark rankings |
| `pyod/utils/knowledge/routing_rules.json` | Add graph routing rules |
| `setup.py` | Add `graph` extra: `['torch>=2.0', 'torch_geometric>=2.0']`. Add both to `all` extra. Note: `torch` is already an implicit dependency for deep learning models; making it explicit in the `graph` extra ensures `pip install pyod[graph]` is a complete one-step install. |
| `README.rst` | Add Graph AD table section (separate, like TS). Note all transductive. |
| `docs/index.rst` | Add Graph AD table section |
| `docs/pyod.models.rst` | Add 8 autodoc entries |
| `docs/zreferences.bib` | Add 9 BibTeX entries |
| `docs/requirements.txt` | Add `torch_geometric` so Sphinx autodoc can import graph modules |
| `docs/conf.py` | Add `pyod.models.pyg_*` modules to `autodoc_mock_imports` if needed as fallback |
| `CHANGES.txt` | Add graph AD entry |

---

## 9. ADEngine Integration

1. **algorithms.json:** Add 8 entries with `data_types: ["graph"]`, BOND rankings.
2. **routing_rules.json:** Add two graph routing rules:
   ```json
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
     "reason": "Attributed graph: DOMINANT and CoLA are most reliable deep methods (BOND benchmark on attributed graphs). Radar is a lightweight MF baseline.",
     "evidence": ["BOND"]
   },
   {
     "id": "graph_structure_only",
     "conditions": [
       {"field": "data_type", "op": "eq", "value": "graph"},
       {"field": "has_features", "op": "eq", "value": false}
     ],
     "recommendations": [
       {"detector": "SCAN", "params": {}, "confidence": 0.8}
     ],
     "reason": "Structure-only graph (no node features): SCAN is the only detector that does not require attributes.",
     "evidence": []
   }
   ```
3. **ADEngine code:** Add graph branch to `profile_data()` and `_sniff_data_type()` as specified in Section 5 (detect PyG `Data`, return graph profile with `n_nodes`, `n_edges`, `n_features`, `has_features`). `run_detection()` already handles transductive detectors via `NotImplementedError` catch.
4. **setup.py:** Add `'graph': ['torch>=2.0', 'torch_geometric>=2.0']` to extras_require. Add both to `'all'` extra.

---

## 10. Implementation Feasibility

| Algorithm | Complexity | Confidence | Notes |
|-----------|-----------|------------|-------|
| DOMINANT | Medium (~200 lines) | High | Well-documented, standard GCN AE |
| CoLA | Medium-high (~250 lines) | High | Subgraph sampling adds complexity |
| CONAD | Medium (~200 lines) | High | Similar to DOMINANT + augmentation |
| AnomalyDAE | Medium (~200 lines) | High | GAT + MLP, straightforward |
| GUIDE | Medium-high (~250 lines) | Medium | Motif construction is nontrivial |
| Radar | Low (~100 lines) | High | Matrix ops only, no GNN |
| ANOMALOUS | Low (~100 lines) | High | Extends Radar, similar structure |
| SCAN | Low (~80 lines) | High | Classical algorithm, no training |

---

## 11. Implementation Order

| Order | What | Depends on |
|-------|------|-----------|
| 0 | Read papers | Nothing |
| 1 | `_pyg_utils.py` + BibTeX entries | Papers |
| 2 | `pyg_scan.py` (simplest, no training) | `_pyg_utils.py` |
| 3 | `pyg_radar.py` | `_pyg_utils.py` |
| 4 | `pyg_anomalous.py` | `_pyg_utils.py` |
| 5 | `pyg_dominant.py` | `_pyg_utils.py` |
| 6 | `pyg_anomalydae.py` | `_pyg_utils.py` |
| 7 | `pyg_cola.py` | `_pyg_utils.py` |
| 8 | `pyg_conad.py` | `_pyg_utils.py` |
| 9 | `pyg_guide.py` | `_pyg_utils.py` |
| 10 | Tests, examples, knowledge base, docs | All above |

Tasks 2-9 are independent and can be parallelized. Classical methods (2-4) are simplest. Deep methods (5-9) require PyTorch.

---

## 12. Reference Implementations

| Algorithm | Primary paper | Reference code |
|-----------|--------------|----------------|
| DOMINANT | Ding et al., SDM 2019 | PyGOD `pygod/nn/dominant.py` |
| CoLA | Liu et al., WWW 2022 | PyGOD `pygod/nn/cola.py` |
| CONAD | Xu et al., PAKDD 2022 | PyGOD `pygod/nn/conad.py` |
| AnomalyDAE | Fan et al., CIKM 2020 | PyGOD `pygod/nn/anomalydae.py` |
| GUIDE | Yuan et al., BigData 2021 | PyGOD `pygod/nn/guide.py` |
| Radar | Li et al., IJCAI 2017 | PyGOD `pygod/nn/radar.py` |
| ANOMALOUS | Peng et al., ICDM 2018 | PyGOD `pygod/nn/anomalous.py` |
| SCAN | Xu et al., KDD 2007 | PyGOD `pygod/nn/scan.py` |

All reference implementations in PyGOD are BSD-2-Clause licensed, same as PyOD. Fresh implementations with attribution.

---

## 13. Codex Review Resolution (Round 2)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | SCAN no-feature case breaks shared fit (`data.x.shape[0]`) | **Resolved** | Use `data.num_nodes`; per-detector `_validate_features()`; SCAN requires explicit `num_nodes` when `x is None` |
| 2 | ADEngine routing contradicts "build-only" | **Resolved** | Graph detectors in routing; `X_test` ignored for transductive. Full graph profile schema + `_sniff_data_type` patch specified. Section 9 synced. |
| 3 | `pyod[graph]` missing `torch` | **Resolved** | Added `torch>=2.0` to graph extra for one-step install. Section 9 synced. |

## 16. Codex Review Resolution (Round 4)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Routing recommends feature-requiring detectors for featureless graphs | **Resolved** | Split into `graph_attributed` (requires `has_features=true`, recommends DOMINANT/CoLA/Radar) and `graph_structure_only` (requires `has_features=false`, recommends SCAN only) |

## 14. Codex Review Resolution (Round 3)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| New 1 | SCAN structure-only path needs explicit `num_nodes` | **Resolved** | Documented requirement in input format section |
| New 2 | `n_edges` assumes undirected duplicate edges | **Resolved** | Changed to raw COO count with comment |
| Reopened 1 | Section 9 stale instructions + duplicate CHANGES.txt + numbering | **Resolved** | Section 9 synced with Sections 5/8; duplicate removed; numbering fixed |

## 15. Codex Review Resolution (Round 1)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Dual input format incompatible with BaseDetector/ADEngine API | **Resolved** | PyG `Data` is primary input. numpy+edge_index is convenience for direct `fit()` only. Documented clearly. |
| 2 | Deep methods labeled inductive but papers are transductive | **Resolved** | All 8 detectors transductive in v1. `decision_function`/`predict`/etc. raise `NotImplementedError`. Same pattern as MatrixProfile. |
| 3 | Classical methods misleadingly labeled numpy/scipy but require PyG | **Resolved** | All detectors require `pyod[graph]`. Table simplified. Packaging plan extended with `all` extra, docs requirements, conf.py fallback. |
