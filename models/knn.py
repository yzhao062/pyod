import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from scipy.stats import scoreatpercentile
from utility import get_precn

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

        self.distance = self.decision_scores()
        self.tree = KDTree(X_train)

    def decision_scores(self):

        if not self._isfitted:
            NotFittedError('Knn is not fitted yet')

        neigh = NearestNeighbors()
        neigh.fit(self.X_train)

        result = neigh.kneighbors(n_neighbors=self.n_neighbors,
                                  return_distance=True)

        dist_arr = result[0]
        ind_array = result[1]

        if self.method == 'largest':
            dist = dist_arr[:, -1]
        elif self.method == 'mean':
            dist = np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            dist = np.median(dist_arr, axis=1)

        threshold = scoreatpercentile(dist, 100 * (1 - self.contamination))
        self.threshold = threshold

        return dist.ravel()

    def sample_scores(self, X_test):

        # initialize the output score
        pred_score = np.zeros([X_test.shape[0], 1])
        # initialize the output label
        pred_label = np.zeros([X_test.shape[0], 1])

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
            pred_label_i = (pred_score_i > self.threshold).astype('int')

            # record the current item
            pred_score[i, :] = pred_score_i
            pred_label[i, :] = pred_label_i

        return pred_score, pred_label

    def evaluate(self, X_test, y_test):
        pred_score, _ = self.sample_scores(X_test)
        roc = np.round(roc_auc_score(y_test, pred_score), decimals=4)
        prec_n = np.round(get_precn(y_test, pred_score), decimals=4)

        print("precision@n", prec_n)
        print("roc", roc)

    def predict(self, X_test):

        _, pred_label = self.sample_scores(X_test)
        return pred_label

#############################################################################
# samples = [[-1, 0], [0., 0.], [1., 1], [2., 5.], [3, 1]]
#
# clf = Knn()
# clf.fit(samples)
# print(clf.sample_scores(np.asarray([[2, 3], [6, 8]])))
# clf.evaluate(np.asarray([[2, 3], [6, 8]]), np.asarray([[0],[1]]))