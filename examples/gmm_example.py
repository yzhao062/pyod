# -*- coding: utf-8 -*-
"""Example of using GMM for outlier detection
"""

# Author: Akira Tamamori
# License: BSD 2 clause

from __future__ import division, print_function

import os
import sys

from pyod.models.gmm import GMM
from pyod.utils.data import evaluate_print, generate_data_clusters
from pyod.utils.example import visualize

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test = generate_data_clusters(
        n_train=n_train,
        n_test=n_test,
        n_features=2,
        n_clusters=4,
        contamination=contamination,
        random_state=42,
    )

    # train kNN detector
    clf_name = "GMM"
    clf = GMM(n_components=4)
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
