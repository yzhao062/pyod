# %%
from itertools import combinations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.exceptions import NotFittedError
from scipy.stats import scoreatpercentile
from scipy.stats import rankdata
from scipy.special import erf

import os
import sys

sys.path.append("..")
import pathlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from data.load_data import generate_data
from pyod.models import Knn
from pyod.utils import precision_n_scores

X = np.array(
    [[2, 0], [3, 0], [3, 1], [5, 0], [5, 1], [5, 2], [3, 4], [1, 2], [3, 3],
     [10, 2], [-6, -6]])

plt.scatter(X[:, 0], X[:, 1])
plt.show()


# %%

class ABOD:
    '''
    ABOD class for outlier detection
    support original ABOD and fast ABOD
    '''

    def __init__(self, contamination=0.1, fast_method=False):
        self.contamination = contamination
        self.fast_method = fast_method

    def fit(self, X_train):
        self.X_train = np.asarray(X_train)
        self._isfitted = True
        self.n = X_train.shape[0]
        self.decision_scores = np.zeros([self.n, 1])

        for i in range(self.n):
            curr_pt = X_train[i, :]
            wcos_list = []

            # get the index pairs of the neighbors
            ind = list(range(0, self.n))
            ind.remove(i)
            curr_pair_inds = list(combinations(ind, 2))

            for j, (a_ind, b_ind) in enumerate(curr_pair_inds):
                a = X_train[a_ind, :]
                b = X_train[b_ind, :]

                a_curr = a - curr_pt
                b_curr = b - curr_pt

                # wcos = (<a_curr, b_curr>/((|a_curr|*|b_curr|)^2)
                wcos = np.dot(a_curr, b_curr) / (
                        np.linalg.norm(a_curr, 2) ** 2) / (
                               np.linalg.norm(b_curr, 2) ** 2)
                wcos_list.append(wcos)

            # calculate the variance of the wcos
            self.decision_scores[i, 0] = np.var(wcos_list)

        self.decision_scores = self.decision_scores.ravel()
        self.threshold = scoreatpercentile(self.decision_scores,
                                           100 * self.contamination)
        self.y_pred = (self.decision_scores < self.threshold).astype('int')

    def decision_function(self, X_test):

        if not self._isfitted:
            NotFittedError('ABOD is not fitted yet')

        # initialize the output score
        pred_score = np.zeros([X_test.shape[0], 1])

        for i in range(X_test.shape[0]):
            curr_pt = X_test[i, :]
            wcos_list = []

            # get the index pairs of the neighbors
            ind = list(range(0, self.n))
            curr_pair_inds = list(combinations(ind, 2))

            for j, (a_ind, b_ind) in enumerate(curr_pair_inds):
                a = self.X_train[a_ind, :]
                b = self.X_train[b_ind, :]

                a_curr = a - curr_pt
                b_curr = b - curr_pt

                # wcos = (<a_curr, b_curr>/((|a_curr|*|b_curr|)^2)
                wcos = np.dot(a_curr, b_curr) / (
                        np.linalg.norm(a_curr, 2) ** 2) / (
                               np.linalg.norm(b_curr, 2) ** 2)
                wcos_list.append(wcos)

            # record the current item
            pred_score[i, :] = np.var(wcos_list)

        return pred_score

    def predict(self, X_test):
        pred_score = self.decision_function(X_test)
        return (pred_score < self.threshold).astype('int').ravel()

    # def predict_proba(self, X_test, method='linear'):
    #     test_scores = self.decision_function(X_test)
    #     train_scores = self.decision_scores
    #
    #     if method == 'linear':
    #         scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
    #         proba = scaler.transform(test_scores.reshape(-1, 1))
    #         return proba.clip(0, 1)
    #     else:
    #         # turn output into probability
    #         pre_erf_score = (test_scores - self.mu) / (
    #                 self.sigma * np.sqrt(2))
    #         erf_score = erf(pre_erf_score)
    #         proba = erf_score.clip(0)
    #
    #         # TODO: move to testing code
    #         assert (proba.min() >= 0)
    #         assert (proba.max() <= 1)
    #         return proba
    #
    # def predict_rank(self, X_test):
    #     test_scores = self.decision_function(X_test)
    #     train_scores = self.decision_scores
    #
    #     ranks = np.zeros([X_test.shape[0], 1])
    #
    #     for i in range(test_scores.shape[0]):
    #         train_scores_i = np.append(train_scores.reshape(-1, 1),
    #                                    test_scores[i])
    #
    #         ranks[i] = rankdata(train_scores_i)[-1]
    #
    #     # return normalized ranks
    #     ranks_norm = ranks / ranks.max()
    #     return ranks_norm


contamination = 0.1  # percentage of outliers
n_train = 200
n_test = 100

X_train, y_train, c_train, X_test, y_test, c_test = generate_data(
    n=n_train,
    contamination=contamination,
    n_test=n_test)


#def abod(X):
#    n = X.shape[0]
#    abod_list = []
#    for i in range(n):
#        curr_pt = X[i, :]
#
#        # get the index pairs of the neighbors
#        ind = list(range(0, n))
#        ind.remove(i)
#        curr_pair_inds = list(combinations(ind, 2))
#        wcos_list = []
#        for j, (a_ind, b_ind) in enumerate(curr_pair_inds):
#            a = X[a_ind, :]
#            b = X[b_ind, :]
#
#            a_curr = a - curr_pt
#            b_curr = b - curr_pt
#
#            wcos = np.dot(a_curr, b_curr) / (
#                    np.linalg.norm(a_curr, 2) ** 2) / (
#                           np.linalg.norm(b_curr, 2) ** 2)
#            wcos_list.append(wcos)
#
#        abod_list.append(np.var(wcos_list))

#    return abod_list


#abod = abod(X_train)
ab = ABOD(contamination=0.1)
ab.fit(X_train)
y_train_score = ab.decision_scores
ypred = ab.y_pred

y_test_score = ab.decision_function(X_test)
#test_y = ab.predict(X_test)

print('Train ROC:{roc}, precision@n:{prn}'.format(
    roc=roc_auc_score(y_train, y_train_score*-1),
    prn=precision_n_scores(y_train, y_train_score*-1)))

print('Test ROC:{roc}, precision@n:{prn}'.format(
    roc=roc_auc_score(y_test, y_test_score*-1),
    prn=precision_n_scores(y_test, y_test_score*-1)))
