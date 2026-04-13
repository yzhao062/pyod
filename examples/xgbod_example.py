# -*- coding: utf-8 -*-
"""Example of using XGBOD for outlier detection
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
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

from pyod.models.xgbod import XGBOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

if __name__ == "__main__":
    # Define data file and read X and y
    # Generate some data if the source data is missing
    csv_file = 'cardio.csv'
    try:
        data = np.genfromtxt(os.path.join('data', csv_file), delimiter=',',
                             skip_header=1)

    except IOError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=csv_file))
        X, y = generate_data(train_only=True)  # load data
    else:
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        X, y = check_X_y(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=42)

    # train XGBOD detector
    clf_name = 'XGBOD'
    clf = XGBOD(random_state=42)
    clf.fit(X_train, y_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
