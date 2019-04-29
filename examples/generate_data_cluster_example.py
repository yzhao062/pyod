# -*- coding: utf-8 -*-
"""Example of using and visualizing ``generate_data_clusters`` function.
Also fit a LOF detector.
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.lof import LOF
from pyod.utils.data import generate_data_clusters
from pyod.utils.example import data_visualize
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers

    # Generate sample data in clusters
    X, y = generate_data_clusters(n_train=450,
                                  n_test=50,
                                  n_clusters=3,
                                  n_features=2,
                                  contamination=contamination,
                                  size='different',
                                  density='different',
                                  dist=0.2,
                                  random_state=42,
                                  return_in_clusters=True)

    # visualize the results
    data_visualize(X, y, show_figure=True, save_figure=False)

    # test on the generated datasets

    # Generate sample data in clusters
    X_train, X_test, y_train, y_test = generate_data_clusters(
        n_train=450,
        n_test=50,
        n_clusters=3,
        n_features=2,
        contamination=contamination,
        size='different',
        density='different',
        dist=0.2,
        random_state=42,
        return_in_clusters=False)

    # train LOF detector
    clf_name = 'LOF'
    clf = LOF()
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

    # visualize the results
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
