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
