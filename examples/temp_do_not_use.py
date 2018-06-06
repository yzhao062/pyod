# -*- coding: utf-8 -*-
"""
Example of combining multiple base outlier scores. Four combination
frameworks are demonstrated:

1. Average: take the average of all base detectors
2. maximization : take the maximum score across all detectors as the score
3. Average of Maximum (AOM)
4. Maximum of Average (MOA)

"""
from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.pca import PCA
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

if __name__ == "__main__":

    n_clf = 20  # number of base detectors
    ite = 10  # number of iterations

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

    for t in range(ite):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.4)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        # initialize 20 base detectors for combination

        clf = PCA()
        clf.fit(X_train_norm)

        train_scores = clf.decision_scores_
        test_scores = clf.decision_function(X_test_norm)
        
        print()
        evaluate_print('PCA Train', y_train, train_scores)
        evaluate_print('PCA Test', y_test, test_scores)

