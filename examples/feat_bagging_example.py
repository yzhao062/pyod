# -*- coding: utf-8 -*-
# %%
"""
Example of using Feature Bagging for outlier detection
"""
from __future__ import division
from __future__ import print_function

import os
import sys

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
from pyod.models.hbos import HBOS
from pyod.models.base import clone
from pyod.utils.data import generate_data
from pyod.utils.utility import precision_n_scores
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from scipy.io import loadmat
from pyod.models.combination import average
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import LocalOutlierFactor
import numpy as np

if __name__ == "__main__":
    # contamination = 0.1  # percentage of outliers
    # n_train = 100
    # n_test = 50
    #
    # X_train, y_train, X_test, y_test = generate_data(
    #     n_train=n_train, n_test=n_test, contamination=contamination)
    mat_file = 'cardio.mat'

    try:
        mat = loadmat(os.path.join('example_data', mat_file))

    except TypeError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    except IOError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    else:
        X = mat['X']
        y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4,
                                                        random_state=2)
    
    # contamination = 0.1  # percentage of outliers
    # n_train = 1000
    # n_test = 500
    #
    # X_train, y_train, X_test, y_test = generate_data(
    #     n_train=n_train, n_test=n_test, contamination=contamination)
    clf_name = 'FBagging'
    clf = FeatureBagging(base_estimator=KNN(), random_state=2)
    clf.fit(X_train)
