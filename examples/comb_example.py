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

from pyod.models.knn import KNN
from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data

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

    # lists for storing roc information
    roc_average = []
    roc_maximization = []
    roc_aom = []
    roc_moa = []

    prn_average = []
    prn_maximization = []
    prn_aom = []
    prn_moa = []

    print('Combining {n_clf} kNN detectors'.format(n_clf=n_clf))
    for t in range(ite):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.4)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        # initialize 20 base detectors for combination
        k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                  150, 160, 170, 180, 190, 200]

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        for i in range(n_clf):
            k = k_list[i]

            clf = KNN(n_neighbors=k, method='largest')
            clf.fit(X_train_norm)

            train_scores[:, i] = clf.decision_scores_
            test_scores[:, i] = clf.decision_function(X_test_norm)

        # decision scores have to be normalized before combination
        train_scores_norm, test_scores_norm = standardizer(train_scores,
                                                           test_scores)

        # combination by average
        comb_by_average = average(test_scores_norm)
        roc_average.append(roc_auc_score(y_test, comb_by_average))
        prn_average.append(precision_n_scores(y_test, comb_by_average))
        print('ite', t + 1, 'comb by average,',
              'ROC:', roc_auc_score(y_test, comb_by_average),
              'precision@n_train:',
              precision_n_scores(y_test, comb_by_average))

        # combination by max
        comb_by_maximization = maximization(test_scores_norm)
        roc_maximization.append(roc_auc_score(y_test, comb_by_maximization))
        prn_maximization.append(
            precision_n_scores(y_test, comb_by_maximization))
        print('ite', t + 1, 'comb by max,', 'ROC:',
              roc_auc_score(y_test, comb_by_maximization),
              'precision@n_train:',
              precision_n_scores(y_test, comb_by_maximization))

        # combination by aom
        comb_by_aom = aom(test_scores_norm, 5)
        roc_aom.append(roc_auc_score(y_test, comb_by_aom))
        prn_aom.append(precision_n_scores(y_test, comb_by_aom))
        print('ite', t + 1, 'comb by aom,', 'ROC:',
              roc_auc_score(y_test, comb_by_aom),
              'precision@n_train:', precision_n_scores(y_test, comb_by_aom))

        # combination by moa
        comb_by_moa = moa(test_scores_norm, 5)
        roc_moa.append(roc_auc_score(y_test, comb_by_moa))
        prn_moa.append(precision_n_scores(y_test, comb_by_moa))
        print('ite', t + 1, 'comb by moa,', 'ROC:',
              roc_auc_score(y_test, comb_by_moa),
              'precision@n_train:', precision_n_scores(y_test, comb_by_moa))

        print()

    ##########################################################################
    print('summary of {ite} iterations'.format(ite=ite))
    print('comb by average, ROC: {roc}, precision@n_train: {prn}'.format(
        roc=np.average(roc_average), prn=np.average(prn_average)))
    print('comb by max, ROC: {roc}, precision@n_train: {prn}'.format(
        roc=np.average(roc_maximization), prn=np.average(prn_maximization)))
    print('comb by aom, ROC: {roc}, precision@n_train: {prn}'.format(
        roc=np.average(roc_aom), prn=np.average(prn_aom)))
    print('comb by moa, ROC: {roc}, precision@n_train: {prn}'.format(
        roc=np.average(roc_moa), prn=np.average(prn_moa)))
