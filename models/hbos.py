import numpy as np
import math
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import scoreatpercentile
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from utility.utility import get_precn
from models.knn import Knn


class Hbos(object):

    def __init__(self, bins=10, alpha=0.3, beta=0.5, contamination=0.05):

        self.bins = bins
        self.alpha = alpha
        self.beta = beta
        self.contamination = contamination

    def fit(self, X):

        self.n, self.d = X.shape[0], X.shape[1]
        out_scores = np.zeros([self.n, self.d])

        hist = np.zeros([self.bins, self.d])
        bin_edges = np.zeros([self.bins + 1, self.d])

        for i in range(self.d):
            hist[:, i], bin_edges[:, i] = np.histogram(X[:, i], bins=self.bins,
                                                       density=True)
            # check the integrity
            assert (
                math.isclose(np.sum(hist[:, i] * np.diff(bin_edges[:, i])), 1))

        # calculate the threshold
        for i in range(self.d):
            # find histogram assignments of data points
            bin_ind = np.digitize(X[:, i], bin_edges[:, i], right=False)

            # very important to do scaling. Not necessary to use min max
            density_norm = MinMaxScaler().fit_transform(
                hist[:, i].reshape(-1, 1))
            out_score = np.log(1 / (density_norm + self.alpha))

            for j in range(self.n):
                # out sample left
                if bin_ind[j] == 0:
                    dist = np.abs(X[j, i] - bin_edges[0, i])
                    bin_width = bin_edges[1, i] - bin_edges[0, i]
                    # assign it to bin 0
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j]]
                    else:
                        out_scores[j, i] = np.max(out_score)

                # out sample right
                elif bin_ind[j] == bin_edges.shape[0]:
                    dist = np.abs(X[j, i] - bin_edges[-1, i])
                    bin_width = bin_edges[-1, i] - bin_edges[-2, i]
                    # assign it to bin k
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j] - 2]
                    else:
                        out_scores[j, i] = np.max(out_score)
                else:
                    out_scores[j, i] = out_score[bin_ind[j] - 1]

        out_scores_sum = np.sum(out_scores, axis=1)
        self.threshold = scoreatpercentile(out_scores_sum,
                                           100 * (1 - self.contamination))
        self.hist = hist
        self.bin_edges = bin_edges
        self.decision_scores = out_scores_sum
        self.y_pred = (self.decision_scores > self.threshold).astype('int')

    def decision_function(self, X_test):

        n_test = X_test.shape[0]
        out_scores = np.zeros([n_test, self.d])

        for i in range(self.d):
            # find histogram assignments of data points
            bin_ind = np.digitize(X_test[:, i], self.bin_edges[:, i],
                                  right=False)

            # very important to do scaling. Not necessary to use minmax
            density_norm = MinMaxScaler().fit_transform(
                self.hist[:, i].reshape(-1, 1))

            out_score = np.log(1 / (density_norm + self.alpha))

            for j in range(n_test):
                # out sample left
                if bin_ind[j] == 0:
                    dist = np.abs(X_test[j, i] - self.bin_edges[0, i])
                    bin_width = self.bin_edges[1, i] - self.bin_edges[0, i]
                    # assign it to bin 0
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j]]
                    else:
                        out_scores[j, i] = np.max(out_score)

                # out sample right
                elif bin_ind[j] == self.bin_edges.shape[0]:
                    dist = np.abs(X_test[j, i] - self.bin_edges[-1, i])
                    bin_width = self.bin_edges[-1, i] - self.bin_edges[-2, i]
                    # assign it to bin k
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j] - 2]
                    else:
                        out_scores[j, i] = np.max(out_score)
                else:
                    out_scores[j, i] = out_score[bin_ind[j] - 1]

        out_scores_sum = np.sum(out_scores, axis=1)
        return out_scores_sum

    def predict(self, X_test):
        pred_score = self.decision_function(X_test)
        return (pred_score > self.threshold).astype('int')

##############################################################################

# roc_result_hbos = []
# roc_result_knn = []
#
# prec_result_hbos = []
# prec_result_knn = []
#
# for t in range(10):
#     n_inliers = 1000
#     n_outliers = 100
#
#     n_inliers_test = 500
#     n_outliers_test = 50
#
#     offset = 2
#     contamination = n_outliers / (n_inliers + n_outliers)
#
#     # generate normal data
#     X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
#     X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
#     X = np.r_[X1, X2]
#     # generate outliers
#     X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]
#     y = np.zeros([X.shape[0], 1])
#     color = np.full([X.shape[0], 1], 'b', dtype=str)
#     y[n_inliers:] = 1
#     color[n_inliers:] = 'r'
#
#     # generate normal data
#     X1_test = 0.3 * np.random.randn(n_inliers_test // 2, 2) - offset
#     X2_test = 0.3 * np.random.randn(n_inliers_test // 2, 2) + offset
#     X_test = np.r_[X1_test, X2_test]
#     # generate outliers
#     X_test = np.r_[
#         X_test, np.random.uniform(low=-8, high=8, size=(n_outliers_test, 2))]
#     y_test = np.zeros([X_test.shape[0], 1])
#     color_test = np.full([X_test.shape[0], 1], 'b', dtype=str)
#     y_test[n_inliers_test:] = 1
#     color_test[n_inliers_test:] = 'r'
#
#     clf = Hbos(contamination=contamination, alpha=0.2, beta=0.5, bins=5)
#     clf.fit(X)
#     pred_score_hbos = clf.decision_function(X_test)
#     y_pred = clf.predict(X_test)
#
#     roc_result_hbos.append(roc_auc_score(y_test, pred_score_hbos))
#     prec_result_hbos.append(get_precn(y_test, pred_score_hbos))
#
#     clf_knn = Knn(n_neighbors=10, contamination=contamination, method='mean')
#     clf_knn.fit(X)
#     pred_score_knn = clf_knn.sample_scores(X_test)
#     roc_result_knn.append(roc_auc_score(y_test, pred_score_knn))
#     prec_result_knn.append(get_precn(y_test, pred_score_knn))
#     # print(get_precn(y_test, clf.decision_function(X_test)))
#     # print(roc_auc_score(y_test, clf.decision_function(X_test)))
#
# print(np.mean(roc_result_hbos), np.mean(prec_result_hbos))
# print(np.mean(roc_result_knn), np.mean(prec_result_knn))
#
# plt.figure(figsize=(9, 7))
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
# plt.show()
