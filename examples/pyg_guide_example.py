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
