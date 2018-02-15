import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.exceptions import NotFittedError
from scipy.stats import scoreatpercentile


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

        neigh = NearestNeighbors()
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

        threshold = scoreatpercentile(dist, 100 * (1 - self.contamination))

        self.threshold = threshold
        self.decision_scores = dist.ravel()
        self.y_pred = (self.decision_scores > self.threshold).astype('int')

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
        return (pred_score > self.threshold).astype('int')


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
