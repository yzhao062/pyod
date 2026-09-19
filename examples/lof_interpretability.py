# -*- coding: utf-8 -*-
"""Example of using LOF for outlier detection with interpretability
Sample wise interpretation is provided here.
"""
# Author: [Your Name/Contribution]
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

from pyod.models.lof import LOF
from pyod.utils.utility import standardizer

if __name__ == "__main__":
    # Define data file and read X and y
    mat_file = 'cardio.mat'

    mat = loadmat(os.path.join('data', mat_file))
    X = mat['X']
    y = mat['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)

    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    # train LOF detector
    clf_name = 'LOF'
    clf = LOF(n_neighbors=10, novelty=False)
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    print('Training data has %d samples' % X_train.shape[0])
    print('Training data has %d features' % X_train.shape[1])
    
    # Explain the first sample
    print('\nExplaining sample 0:')
    print('True label:', 'Outlier' if y_train[0] == 1 else 'Inlier')
    print('Predicted label:', 'Outlier' if y_train_pred[0] == 1 else 'Inlier')
    print('Outlier score: %.4f' % y_train_scores[0])
    print('\nGenerating dimensional outlier explanation...')
    
    clf.explain_outlier(0)
    
    # The horizontal bar chart shows per-feature LOF scores (1D approximation).
    # Each bar represents the LOF score computed on that dimension independently.
    # Features with bars exceeding the cutoff lines (dashed/dash-dot vertical lines)
    # exhibit unusual local density patterns and are primary contributors to
    # the sample being flagged as an outlier.
    
    # Example with custom cutoffs
    print('\n' + '='*60)
    print('Example with custom cutoff bands (80th and 95th percentiles):')
    print('='*60)
    
    clf.explain_outlier(0, cutoffs=[0.80, 0.95])

