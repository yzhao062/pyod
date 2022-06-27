# -*- coding: utf-8 -*-
"""Example of using DeepSVDD for outlier detection
"""
# Author: Rafal Bodziony <bodziony.rafal@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.deep_svdd import DeepSVDD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 20000  # number of training points
    n_test = 2000  # number of testing points
    n_features = 300  # number of features
    use_ae = False  # hyperparameter for use ae architecture instead of simple NN
    random_state = 10  # if C is set to None use random_state

    # Generate sample data
    X_train, X_test, y_train, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=n_features,
                      contamination=contamination,
                      random_state=42)

    # train DeepSVDD detector (Without-AE)
    clf_name = 'DeepSVDD'
    clf = DeepSVDD(use_ae=use_ae, epochs=5, contamination=contamination,
                   random_state=random_state)
    clf.fit(X_train)

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
