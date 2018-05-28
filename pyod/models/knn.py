import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


class KNN(BaseDetector):
    """
    kNN class for outlier detection.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. More to see the references below.

    Three kNN detectors are supported:
    largest: use the distance to the kth neighbor as the outlier score
    mean: use the average of all k neighbors as the outlier score
    median: use the median of the distance to k neighbors as the outlier score

    .. [1] Ramaswamy, S., Rastogi, R. and Shim, K., 2000, May.
           Efficient algorithms for mining outliers from large data sets. In
           ACM Sigmod Record (Vol. 29, No. 2, pp. 427-438). ACM.

    .. [2] Angiulli, F. and Pizzuti, C., 2002, August. Fast outlier detection
           in high dimensional spaces. In European Conference on Principles of
           Data Mining and Knowledge Discovery,pp. 15-27.

    """

    def __init__(self, n_neighbors=5, contamination=0.1, method='largest'):
        """
        :param n_neighbors: Number of neighbors to use by default
            for kneighbors queries.
        :type n_neighbors: int, optional (default=5)
        :param contamination: he amount of contamination of the data set, i.e.
            the proportion of outliers in the data set. Used when fitting to
            define the threshold on the decision function.
        :type contamination: float in (0, 0.5], optional (default=0.1)
        :param method: {'largest', 'mean', 'median'},
            largest: use the distance to the kth neighbor as the outlier score
            mean: use the average of all k neighbors as the outlier score
            median: use the median of the distance to k neighbors as the outlier score
        :type method: str, optional (default='largest')
        """
        super().__init__(contamination=contamination)
        self.n_neighbors_ = n_neighbors
        self.method = method

    def fit(self, X):

        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % self.contamination)

        X = check_array(X)
        self.tree_ = KDTree(X)

        neigh = NearestNeighbors(n_neighbors=self.n_neighbors_)
        neigh.fit(X)

        result = neigh.kneighbors(n_neighbors=self.n_neighbors_,
                                  return_distance=True)
        dist_arr = result[0]

        if self.method == 'largest':
            dist = dist_arr[:, -1]
        elif self.method == 'mean':
            dist = np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            dist = np.median(dist_arr, axis=1)

        self.decision_scores = dist.ravel()
        self._process_decision_scores()
        return self

    def decision_function(self, X):

        check_is_fitted(self,
                        ['tree_', 'decision_scores', 'threshold_', 'y_pred'])

        X = check_array(X)

        # initialize the output score
        pred_score = np.zeros([X.shape[0], 1])

        for i in range(X.shape[0]):
            x_i = X[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, ind_arr = self.tree_.query(x_i, k=self.n_neighbors_)

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
# decision_scores = clf.decision_function(np.asarray([[2, 3], [6, 8]])).ravel()
# assert (decision_scores[0] == [2])
# assert (decision_scores[1] == [5])
# #
# labels = clf.predict(np.asarray([[2, 3], [6, 8]])).ravel()
# assert (labels[0] == [0])
# assert (labels[1] == [1])
