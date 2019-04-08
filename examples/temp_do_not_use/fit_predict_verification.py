# -*- coding: utf-8 -*-
"""Example of using LOF for outlier detection
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
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from numpy import allclose

if __name__ == "__main__":
    # contamination = 0.1  # percentage of outliers
    # n_train = 200  # number of training points
    # n_test = 100  # number of testing points
    #
    # # Generate sample data
    # X_train, y_train, X_test, y_test = \
    #     generate_data(n_train=n_train,
    #                   n_test=n_test,
    #                   n_features=2,
    #                   contamination=contamination,
    #                   random_state=42)

    rng = np.random.RandomState(42)
    X = rng.rand(1000, 3)
    # train LOF detector
    clf_name = 'LOF'
    # clf = IForest(random_state=42)
    # clf = OCSVM()
    clf = KNN(n_neighbors=50)
    # clf.fit(X)

    # get the prediction labels and outlier scores of the training data
    # y_train_pred = clf.predict(X)  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores

    y_train_pred_fp = clf.fit_predict(X)
    y_train_pred = clf.fit(X).predict(X)
    y_train_scores = clf.fit(X).labels_
    # print(allclose(y_train_pred, y_train_pred_fp))
    # print(allclose(y_train_pred, y_train_scores))
    # print(allclose(y_train_pred_fp, y_train_scores))

    # calculate the difference
    print(np.sum(np.abs(y_train_pred-y_train_pred_fp)))
    print(np.sum(np.abs(y_train_pred-y_train_scores)))
    print(np.sum(np.abs(y_train_pred_fp-y_train_scores)))

    # the ones below can be commented out
    print(y_train_pred)
    print(y_train_pred_fp)
    print(y_train_scores)
