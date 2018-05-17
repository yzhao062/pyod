import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.exceptions import NotFittedError
from scipy.stats import scoreatpercentile
from scipy.stats import rankdata
from scipy.special import erf


class Knn(object):
    '''
    Knn class for outlier detection
    support original knn, average knn, and median knn
    '''

    def __init__(self, n_neighbors=1, contamination=0.05, method='largest'):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.method = method

    def fit(self, X_train):
        self.X_train = X_train
        self._isfitted = True
        self.tree = KDTree(X_train)

        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        neigh.fit(self.X_train)

        result = neigh.kneighbors(n_neighbors=self.n_neighbors,
                                  return_distance=True)
        dist_arr = result[0]

        if self.method == 'largest':
            dist = dist_arr[:, -1]
        elif self.method == 'mean':
            dist = np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            dist = np.median(dist_arr, axis=1)

        self.threshold = scoreatpercentile(dist,
                                           100 * (1 - self.contamination))
        self.decision_scores = dist.ravel()
        self.y_pred = (self.decision_scores > self.threshold).astype('int')

        self.mu = np.mean(self.decision_scores)
        self.sigma = np.std(self.decision_scores)

    def decision_function(self, X_test):

        if not self._isfitted:
            NotFittedError('Knn is not fitted yet')

        # initialize the output score
        pred_score = np.zeros([X_test.shape[0], 1])

        for i in range(X_test.shape[0]):
            x_i = X_test[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, ind_arr = self.tree.query(x_i, k=self.n_neighbors)

            if self.method == 'largest':
                dist = dist_arr[:, -1]
            elif self.method == 'mean':
                dist = np.mean(dist_arr, axis=1)
            elif self.method == 'median':
                dist = np.median(dist_arr, axis=1)

            pred_score_i = dist[-1]

            # record the current item
            pred_score[i, :] = pred_score_i

        return pred_score

    def predict(self, X_test):
        pred_score = self.decision_function(X_test)
        return (pred_score > self.threshold).astype('int').ravel()

    def predict_proba(self, X_test, method='linear'):
        test_scores = self.decision_function(X_test)
        train_scores = self.decision_scores

        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            proba = scaler.transform(test_scores.reshape(-1, 1))
            return proba.clip(0, 1)
        else:
            # turn output into probability
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
# samples = [[-1, 0], [0., 0.], [1., 1], [2., 5.], [3, 1]]
#
# clf = Knn()
# clf.fit(samples)
#
# scores = clf.decision_function(np.asarray([[2, 3], [6, 8]])).ravel()
# assert (scores[0] == [2])
# assert (scores[1] == [5])
# #
# labels = clf.predict(np.asarray([[2, 3], [6, 8]])).ravel()
# assert (labels[0] == [0])
# assert (labels[1] == [1])
