import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError

from .base import BaseDetector


class KNN(BaseDetector):
    """
    Knn class for outlier detection
    support original knn, average knn, and median knn
    """

    def __init__(self, n_neighbors=1, contamination=0.1, method='largest'):
        """

        :param n_neighbors:
        :param contamination:
        :param method: {'largest', 'mean', 'median'}
        """
        super().__init__(contamination=contamination)
        self.n_neighbors_ = n_neighbors
        self.method = method

    def fit(self, X_train):

        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % self.contamination)

        X_train = check_array(X_train)
        self._isfitted = True
        self.tree = KDTree(X_train)

        neigh = NearestNeighbors(n_neighbors=self.n_neighbors_)
        neigh.fit(X_train)

        result = neigh.kneighbors(n_neighbors=self.n_neighbors_,
                                  return_distance=True)
        dist_arr = result[0]

        if self.method == 'largest':
            dist = dist_arr[:, -1]
        elif self.method == 'mean':
            dist = np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            dist = np.median(dist_arr, axis=1)

        self.threshold_ = scoreatpercentile(dist,
                                            100 * (1 - self.contamination))
        self.decision_scores = dist.ravel()
        self.y_pred = (self.decision_scores > self.threshold_).astype('int')

        self.mu = np.mean(self.decision_scores)
        self.sigma = np.std(self.decision_scores)

        return self

    def decision_function(self, X):

        if not self._isfitted:
            NotFittedError('Model is not fitted yet')

        X = check_array(X)

        # initialize the output score
        pred_score = np.zeros([X.shape[0], 1])

        for i in range(X.shape[0]):
            x_i = X[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, ind_arr = self.tree.query(x_i, k=self.n_neighbors_)

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
