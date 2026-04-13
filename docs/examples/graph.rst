Layer 1: Graph Anomaly Detection
==================================

PyOD has 8 graph anomaly detectors built on PyTorch Geometric (PyG). Install with ``pip install pyod[graph]``. All 8 are transductive in v1 -- use ``decision_scores_`` after ``fit()``.

.. code-block:: python

    from pyod.models.pyg_dominant import DOMINANT
    import torch
    from torch_geometric.data import Data

    data = Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(edge_index))
    clf = DOMINANT(hidden_dim=64, epochs=100)
    clf.fit(data)
    scores = clf.decision_scores_                 # per-node anomaly scores

----

Detectors
---------

Rankings from `BOND benchmark <https://arxiv.org/abs/2206.10071>`_ (NeurIPS 2022):

.. list-table::
   :widths: 18 52 8 12
   :header-rows: 1

   * - Type
     - Detector
     - Year
     - Venue
   * - GCN Autoencoder
     - `DOMINANT <https://github.com/yzhao062/pyod/blob/development/examples/pyg_dominant_example.py>`__: GCN AE, structure + attribute
     - 2019
     - SDM
   * - Contrastive
     - `CoLA <https://github.com/yzhao062/pyod/blob/development/examples/pyg_cola_example.py>`__: local neighbor context
     - 2022
     - WWW
   * - Contrastive+AE
     - `CONAD <https://github.com/yzhao062/pyod/blob/development/examples/pyg_conad_example.py>`__: anomalous-view injection
     - 2022
     - PAKDD
   * - Attention AE
     - `AnomalyDAE <https://github.com/yzhao062/pyod/blob/development/examples/pyg_anomalydae_example.py>`__: GAT + MLP dual AE
     - 2020
     - CIKM
   * - Motif AE
     - `GUIDE <https://github.com/yzhao062/pyod/blob/development/examples/pyg_guide_example.py>`__: triangle-motif dual AE
     - 2021
     - BigData
   * - Matrix Factor.
     - `Radar <https://github.com/yzhao062/pyod/blob/development/examples/pyg_radar_example.py>`__: residual analysis
     - 2017
     - IJCAI
   * - Matrix Factor.
     - `ANOMALOUS <https://github.com/yzhao062/pyod/blob/development/examples/pyg_anomalous_example.py>`__: CUR-style, Laplacian
     - 2018
     - IJCAI
   * - Structural
     - `SCAN <https://github.com/yzhao062/pyod/blob/development/examples/pyg_scan_example.py>`__: structural clustering
     - 2007
     - KDD

----

Input Format
------------

**Primary**: PyG ``Data`` object with ``x`` (node features) and ``edge_index`` (COO edges).

**Convenience**: numpy arrays for direct ``fit()`` calls:

.. code-block:: python

    clf.fit(X, edge_index=edge_index)
    # X: numpy (n_nodes, n_features)
    # edge_index: numpy (2, n_edges)

``SCAN`` is the only structure-only detector -- it works without node features. All others require ``data.x``.
