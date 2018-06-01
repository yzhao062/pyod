# -*- coding: utf-8 -*-
# %%
"""
Example of using Feature Bagging for outlier detection
"""
import os, sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from pyod.models.feat_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.base import clone
from pyod.utils.load_data import generate_data
from pyod.utils.utility import precision_n_scores
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor
import numpy as np

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 100
    n_test = 50

    X_train, y_train, c_train, X_test, y_test, c_test = generate_data(
        n_train=n_train, n_test=n_test, contamination=contamination)



# TODO: place holder only
