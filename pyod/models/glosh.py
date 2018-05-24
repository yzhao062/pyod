import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.utils import check_array

# TODO: fix broken model here
import hdbscan
from .base import BaseDetector


class Glosh(BaseDetector):
    def __init__(self, min_cluster_size=5, contamination=0.1):
        super().__init__(contamination=contamination)
        self.min_cluster_size = min_cluster_size

    def fit(self, X_train):
        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % self.contamination)

        X_train = check_array(X_train)
        self.X_train = X_train
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(self.X_train)

        self.scores = clusterer.outlier_scores_
        self.threshold = scoreatpercentile(self.scores,
                                           100 * (1 - self.contamination))

    def decision_function(self, X_test):

        X_test = check_array(X_test)
        # initialize the outputs
        pred_score = np.zeros([X_test.shape[0], 1])

        for i in range(X_test.shape[0]):
            x_i = X_test[i, :]

            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])
            x_comb = np.concatenate((self.X_train, x_i), axis=0)

            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(x_comb)

            # record the current item
            pred_score[i, :] = clusterer.outlier_scores_[-1]
        return pred_score
