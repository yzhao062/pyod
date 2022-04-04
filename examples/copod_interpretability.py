# -*- coding: utf-8 -*-
"""Example of using Copula Based Outlier Detector (COPOD) for outlier detection
Sample wise interpretation is provided here.
"""
# Author: Winston Li <jk_zhengli@hotmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from pyod.models.copod import COPOD
from pyod.utils.utility import standardizer

if __name__ == "__main__":
    # Define data file and read X and y
    # Generate some data if the source data is missing
    mat_file = 'cardio.mat'

    mat = loadmat(os.path.join('data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)

    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    # train COPOD detector
    clf_name = 'COPOD'
    clf = COPOD()

    # you could try parallel version as well.
    # clf = COPOD(n_jobs=2)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    print('The first sample is an outlier', y_train[0])
    clf.explain_outlier(0)

    # we could see feature 7, 16, and 20 is above the 0.99 cutoff
    # and play a more important role in deciding it is an outlier.
