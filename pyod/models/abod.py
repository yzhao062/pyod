from itertools import combinations

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector


class ABOD(BaseDetector):
    """
    ABOD class for outlier detection
    support original ABOD and fast ABOD
    """

    def __init__(self, contamination=0.1, fast_method=False):
        super().__init__(contamination=contamination)
        self.fast_method = fast_method
        self.X_train = None

        # TODO: verify if n_train is needed
        self.n_train = None

    def fit(self, X_train):
        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % self.contamination)

        X_train = check_array(X_train)
        self.X_train = X_train
        self.n_train = X_train.shape[0]
        self.decision_scores = np.zeros([self.n_train, 1])

        for i in range(self.n_train):
            curr_pt = X_train[i, :]

            # get the index pairs of the neighbors
            ind = list(range(0, self.n_train))
            ind.remove(i)

            wcos_list = []
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

        self.decision_scores = self.decision_scores.ravel() * -1
        self._process_decision_scores()
        return self

    def decision_function(self, X):

        check_is_fitted(self,
                        ['X_train', 'decision_scores', 'threshold_', 'y_pred'])

        X = check_array(X)
        # initialize the output score
        pred_score = np.zeros([X.shape[0], 1])

        for i in range(X.shape[0]):
            curr_pt = X[i, :]

            # get the index pairs of the neighbors
            ind = list(range(0, self.n_train))

            wcos_list = []
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

        # outliers have higher decision_scores
        return pred_score * -1
