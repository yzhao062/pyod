# -*- coding: utf-8 -*-
"""Example for Anomaly Detection with Generative Adversarial Networks  (AnoGAN)
 Paper: https://arxiv.org/pdf/1703.05921.pdf
 Note, that this is another implementation of AnoGAN as the one from https://github.com/fuchami/ANOGAN
"""
# Author: Michiel Bongaerts (but not author of the AnoGAN method)
# License: BSD 2 clause


from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.anogan import AnoGAN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=2,
        contamination=contamination,
        behaviour="new",
        random_state=42,
    )

    # train AutoEncoder detector
    clf_name = 'AnoGAN'
    clf = AnoGAN( G_layers = [10,20], D_layers = [20,2], 
                  preprocessing = True, index_D_layer_for_recon_error = 1,
                  epochs = 200, contamination = contamination, verbose = 1 )
    
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
              y_test_pred, show_figure=True, save_figure=True)


    clf.plot_learning_curves(window_smoothening=30)

