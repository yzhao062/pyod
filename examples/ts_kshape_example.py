# -*- coding: utf-8 -*-
"""Example of using KShape for time series anomaly detection.

KShape is experimental.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from pyod.models.ts_kshape import KShape
from pyod.utils.data import generate_ts_data

if __name__ == "__main__":
    contamination = 0.1

    # Generate synthetic time series with anomalies
    X_train, X_test, y_train, y_test = generate_ts_data(
        n_train=500, n_test=200, contamination=0.05, random_state=42)

    clf_name = 'KShape'
    clf = KShape(n_clusters=3, window_size=20, contamination=contamination)
    clf.fit(X_train)

    print("Detector: %s" % clf_name)
    print("Number of anomalies: %d" % clf.labels_.sum())
    print("Top 5 anomaly scores:", np.sort(clf.decision_scores_)[-5:])
