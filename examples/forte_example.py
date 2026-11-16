# -*- coding: utf-8 -*-
"""Example of using Forte for outlier detection
"""
# Author: Debargha Ganguly <maildebargha@gmail.com>
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

from pyod.models.forte import Forte
from pyod.utils.data import evaluate_print

if __name__ == "__main__":
    n_train = 1000  # number of training points
    n_test = 500  # number of testing points
    n_outliers = 100  # number of shifted points in the test sample

    # Forte works on feature representations. Here two 16-dimensional
    # random spaces stand in for the embeddings of two encoders,
    # concatenated column-wise. The training data is clean and the test
    # sample contains a group of points from a shifted distribution.
    rng = np.random.RandomState(42)
    X_train = rng.randn(n_train, 32)
    X_test = rng.randn(n_test, 32)
    X_test[-n_outliers:] += 1.5
    y_test = np.zeros(n_test)
    y_test[-n_outliers:] = 1

    # train Forte detector, columns 0-15 and 16-31 are separate spaces
    clf_name = 'Forte'
    clf = Forte(space_splits=[16], contamination=0.1, random_state=42)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data. X_test is scored as one sample:
    # the score of a point depends on the other points passed with it
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
