import math

import numpy as np
from scipy.stats import scoreatpercentile
from scipy.stats import rankdata
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array


class HBOS(object):

    def __init__(self, bins=10, beta=0.5, contamination=0.05):

        self.bins = bins
        self.beta = beta
        self.contamination = contamination

    def fit(self, X_train):

        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % self.contamination)

        X_train = check_array(X_train)

        self.n, self.d = X_train.shape[0], X_train.shape[1]
        out_scores = np.zeros([self.n, self.d])

        hist = np.zeros([self.bins, self.d])
        bin_edges = np.zeros([self.bins + 1, self.d])

        # build the bins
        for i in range(self.d):
            hist[:, i], bin_edges[:, i] = np.histogram(X_train[:, i],
                                                       bins=self.bins,
                                                       density=True)
            # check the integrity
            assert (
                math.isclose(np.sum(hist[:, i] * np.diff(bin_edges[:, i])), 1))

        # calculate the threshold
        for i in range(self.d):
            # find histogram assignments of data points
            bin_ind = np.digitize(X_train[:, i], bin_edges[:, i], right=False)

            # very important to do scaling. Not necessary to use min max
            out_score = np.max(hist[:, i]) - hist[:, i]
            out_score = MinMaxScaler().fit_transform(out_score.reshape(-1, 1))

            for j in range(self.n):
                # out sample left
                if bin_ind[j] == 0:
                    dist = np.abs(X_train[j, i] - bin_edges[0, i])
                    bin_width = bin_edges[1, i] - bin_edges[0, i]
                    # assign it to bin 0
                    if dist < bin_width * self.beta:
                        out_scores[j, i] = out_score[bin_ind[j]]
                    else:
                        out_scores[j, i] = np.max(out_score)

                # out sample right
                elif bin_ind[j] == bin_edges.shape[0]:
                    dist = np.abs(X_train[j, i] - bin_edges[-1, i])
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
        self.mu = np.mean(self.decision_scores)
        self.sigma = np.std(self.decision_scores)

    def decision_function(self, X_test):

        X_test = check_array(X_test)
        n_test = X_test.shape[0]
        out_scores = np.zeros([n_test, self.d])

        for i in range(self.d):
            # find histogram assignments of data points
            bin_ind = np.digitize(X_test[:, i], self.bin_edges[:, i],
                                  right=False)

            # very important to do scaling. Not necessary to use minmax
            out_score = np.max(self.hist[:, i]) - self.hist[:, i]
            out_score = MinMaxScaler().fit_transform(out_score.reshape(-1, 1))

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

    def predict_proba(self, X_test, method='linear'):
        train_scores = self.decision_scores
        test_scores = self.decision_function(X_test)

        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            proba = scaler.transform(test_scores.reshape(-1, 1))
            return proba.clip(0, 1)
        else:
            #        # turn output into probability
            pre_erf_score = (test_scores - self.mu) / (self.sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            proba = erf_score.clip(0)

            # TODO: move to testing code
            assert (proba.min() >= 0)
            assert (proba.max() <= 1)
            return proba

    def predict_rank(self, X_test):
        test_scores = self.decision_function(X_test)
        train_scores = self.decision_scores

        ranks = np.zeros([X_test.shape[0], 1])

        for i in range(test_scores.shape[0]):
            train_scores_i = np.append(train_scores.reshape(-1, 1),
                                       test_scores[i])

            ranks[i] = rankdata(train_scores_i)[-1]

        # return normalized ranks
        ranks_norm = ranks / ranks.max()
        return ranks_norm

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
