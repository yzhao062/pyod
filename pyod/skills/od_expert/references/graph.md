# Graph anomaly detection reference

PyOD ships 8 graph detectors. The agent loads this file when the master decision tree (in SKILL.md) routes to graph.

> **All detectors in this reference require `pip install pyod[graph]`.** The agent must check `importlib.util.find_spec("torch_geometric")` before recommending any of these detectors. If the extra is missing, escalate (Trigger 7) with the install command and either wait for the user to install or fall back to running a tabular detector on the node features only (which loses edge information and changes the result interpretation).

## Decision table by graph type (expert heuristics)

These are rules of thumb drawn from BOND (Liu et al. 2022) and general GAD literature. They are **not** predictions of exact `engine.plan` output. On a representative probe (10k-node attributed graph), the current planner returned `['DOMINANT', 'CoLA', 'Radar']`; use `state.plans` at runtime for the live selection and this table to check whether the plan is plausible.

| Graph type | Heuristic starters | Why |
|---|---|---|
| Static homogeneous, with node features | `DOMINANT`, `CoLA`, `AnomalyDAE` / `Radar` | Reconstruction-based GAD on attributed graphs |
| Static homogeneous, no node features | `Radar`, `ANOMALOUS` | Structure-only GAD methods |
| Heterogeneous (node types differ) | `GUIDE`, `CONAD` | Heterogeneity-aware GAD |
| With anomaly labels (semi-supervised) | `CONAD`, `GUIDE` | Contrastive / supervised-augmented GAD |
| Very small graph (< 100 nodes) | (consider tabular fallback) | Deep GAD overfits on tiny graphs |
| Need edge-level or graph-level AD | (gap — node-level only in v3.2.0) | Edge/graph-level GAD planned for v3.3.0 |

## Detectors available in PyOD (KB-derived)

<!-- BEGIN KB-DERIVED: graph-detector-list -->
- **ANOMALOUS** (Joint Modeling Approach for Anomaly Detection) — complexity: time O(max_iter * n^2 * d), space O(n^2); best for: Small-to-medium attributed graphs where smoothness matters; avoid when: Graph is very large or anomalies are purely structural; requires: pyod[graph]; paper: Peng et al., IJCAI 2018
- **AnomalyDAE** (Dual Autoencoder for Anomaly Detection) — complexity: time O(epochs * (n * d_h + n^2)), space O(n^2); best for: Attributed graphs where attention over neighbors reveals anomaly patterns; avoid when: Graph is very large or structure is unimportant; requires: pyod[graph]; paper: Fan et al., CIKM 2020
- **CONAD** (Contrastive Attributed Network Anomaly Detection) — complexity: time O(epochs * n * d_h), space O(n * d_h); best for: Attributed graphs where robustness to noise is important; avoid when: Graph structure is too sparse for meaningful augmentation; requires: pyod[graph]; paper: Xu et al., PAKDD 2022
- **CoLA** (Contrastive Self-Supervised Anomaly Detection) — complexity: time O(epochs * n * d_h), space O(n * d_h + m); best for: Attributed graphs where anomalies have unusual local neighborhoods; avoid when: Graph is disconnected or has no node features; requires: pyod[graph]; paper: Liu et al., WWW 2022
- **DOMINANT** (Deep Anomaly Detection on Attributed Networks) — complexity: time O(epochs * (n * d_h + n^2)), space O(n^2); best for: Attributed graphs where anomalies manifest in both structure and attributes; avoid when: Graph is very large (>10k nodes) or has no node features; requires: pyod[graph]; paper: Ding et al., SDM 2019
- **GUIDE** (Higher-order Structure Based Anomaly Detection) — complexity: time O(epochs * n * d_h + m * d_avg), space O(n^2); best for: Dense attributed graphs with meaningful higher-order structures; avoid when: Graph is tree-like (no triangles) or very large; requires: pyod[graph]; paper: Yuan et al., BigData 2021
- **Radar** (Residual Analysis for Anomaly Detection) — complexity: time O(max_iter * n^2 * d), space O(n^2); best for: Small-to-medium attributed graphs as a fast baseline; avoid when: Graph is very large or anomalies are structural-only; requires: pyod[graph]; paper: Li et al., IJCAI 2017
- **SCAN_Graph** (Structural Clustering Algorithm for Networks) — complexity: time O(m * d_avg), space O(n + m); best for: Structure-only graphs or as a fast structural baseline; avoid when: Node attributes are available and important for anomaly detection; requires: pyod[graph]; paper: Xu et al., KDD 2007
<!-- END KB-DERIVED: graph-detector-list -->

## PyG installation check

```python
import importlib.util

if importlib.util.find_spec("torch_geometric") is None:
    print("Graph detection requires `pip install pyod[graph]`.")
    # Escalate Trigger 7 — fall back or wait for user
```

The PyG dependency is heavy (PyTorch + torch_geometric + their CUDA extensions). On systems without CUDA, install the CPU-only variant:

```bash
pip install pyod[graph] --extra-index-url https://download.pytorch.org/whl/cpu
```

## Graph data format expected by ADEngine

```python
from pyod.utils.ad_engine import ADEngine

# Either a torch_geometric Data object:
import torch
from torch_geometric.data import Data

data = Data(
    x=torch.tensor(node_features, dtype=torch.float),  # [num_nodes, num_features]
    edge_index=torch.tensor(edges, dtype=torch.long),  # [2, num_edges]
)

engine = ADEngine()
state = engine.start(data)
# state.profile['data_type'] == 'graph'
```

If the user provides a NetworkX graph, an adjacency matrix, or a CSV of edges, ADEngine attempts to infer and convert. If the conversion is ambiguous, it returns a `confirm_with_user` state with the proposed conversion.

## Worked example: small social-network anomaly

### Setup

A social platform shares a node-attributed graph: 10,000 users (nodes), 200,000 follow relationships (edges), 64 user features per node (account age, post count, etc.). They want to find suspicious accounts.

### Agent flow

```python
import torch
from torch_geometric.data import Data
from pyod.utils.ad_engine import ADEngine

# Load into a Data object (assume node_features and edges are already arrays)
data = Data(
    x=torch.tensor(node_features, dtype=torch.float),
    edge_index=torch.tensor(edges.T, dtype=torch.long),
)

engine = ADEngine()
state = engine.start(data)
# state.profile: {'data_type': 'graph', 'n_nodes': 10000,
#                 'n_edges': 200000, 'n_features': 64, ...}
# Modality: graph ✓
# Has node features → DOMINANT/CoLA/AnomalyDAE eligible

# Trigger 7 check
import importlib.util
assert importlib.util.find_spec("torch_geometric"), "pyod[graph] missing"

state = engine.plan(state)
# [p['detector_name'] for p in state.plans[:3]] == ['DOMINANT', 'CoLA', 'Radar']
# (Captured from a live probe on a 10k-node attributed graph, 2026-04-14.
# The planner may evolve; always read state.plans at runtime.)

state = engine.run(state)
state = engine.analyze(state)
# state.quality: {'agreement': 0.55, 'separation': 0.22, 'stability': 0.61,
#                 'overall': ..., 'verdict': 'medium'}
# state.analysis['best_detector']: 'DOMINANT'
```

Note: agreement is below the 0.4 floor would NOT trigger here (0.55 > 0.4). Separation 0.22 is just above the 0.1 critical floor — moderate confidence. Report with a hedge.

### What the agent reports

> "Found 320 suspicious accounts of 10,000 (3.2%). The three graph detectors I ran (`DOMINANT`, `CoLA`, `Radar`) agreed moderately (consensus 0.55). `DOMINANT` performed best.
>
> Top-5 most suspicious accounts: nodes [42, 891, 2103, 5566, 7891]. The dominant pattern was unusual follower-to-following ratios combined with rare account-age values.
>
> Confidence: medium. Separation is 0.22 (above the 0.1 floor, but not strong), so I recommend manually reviewing the top-20 against your trust-and-safety database before acting on these flags.
>
> **What I assumed**:
> - Graph, node-attributed, homogeneous
> - Detectors: `DOMINANT`, `CoLA`, `Radar` — the triple returned by the current planner on a 10k-node attributed graph
> - Best detector: `DOMINANT`"

## Graph-specific pitfalls

### G1. Disconnected components

A graph with many disconnected components is not really one graph. GNN-based methods average information across the graph and produce useless embeddings on disconnected pieces. Mitigation: detect disconnected components first, run GAD per component, combine results.

### G2. Isolated nodes

Nodes with degree 0 cannot benefit from neighbor information. GNN methods will return their raw features as the embedding. Mitigation: exclude isolated nodes from the GAD pipeline, flag them via a tabular detector on node features only.

### G3. Leakage from neighbors

GNN-based GAD propagates information across edges. If the graph contains "ground truth" labels as edges to a "label node" or similar, the detector can trivially identify anomalies via this leakage. Mitigation: scrub label-related edges before training.

### G4. Heterogeneous graph treated as homogeneous

If the graph has multiple node types (users, posts, hashtags) but is loaded as a single homogeneous graph, the detector loses semantic meaning. Mitigation: use `GUIDE` or `CONAD` which handle heterogeneity.

### G5. Edge-level anomaly with node-level detector

Asking a node-level GAD detector to find anomalous edges always returns garbage. Mitigation: PyOD does not yet ship edge-level GAD (v3.3.0 backlog item). For now, if edge anomalies are needed, escalate and recommend a different library or a hand-rolled approach.

## Graph-specific escalation triggers

In addition to the global triggers in SKILL.md, watch for these graph-specific cases:

- **PyG missing**: `pip install pyod[graph]` not run → Trigger 7. Escalate with the install command.
- **No node features**: only edge structure → recommend `Radar` or `ANOMALOUS`, do not use feature-dependent methods.
- **No edges**: a graph with zero edges is just tabular → escalate, ask if the data should be re-routed to tabular.
- **Edge-level AD requested**: PyOD does not yet support → escalate, point to v3.3.0 backlog.
- **Graph-level AD requested**: same — not yet supported.

## See also

- `pitfalls.md` — extended pitfalls library (preprocessing → detection → analysis → reporting)
- `workflow.md` — the autonomous loop pattern
- SKILL.md — top-10 critical pitfalls and master decision tree
