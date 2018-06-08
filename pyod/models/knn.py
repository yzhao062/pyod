# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


class KNN(BaseDetector):
    """kNN class for outlier detection.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.

    Three kNN detectors are supported:
    largest: use the distance to the kth neighbor as the outlier score
    mean: use the average of all k neighbors as the outlier score
    median: use the median of the distance to k neighbors as the outlier score

    :param contamination: the amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    :type contamination: float in (0, 0.5], optional (default=0.1)

    :param n_neighbors: Number of neighbors to use by default
        for k neighbors queries.
    :type n_neighbors: int, optional (default=5)

    :param method: {'largest', 'mean', 'median'}

        - largest: use the distance to the kth neighbor as the outlier score
        - mean: use the average of all k neighbors as the outlier score
        - median: use the median of the distance to k neighbors as the outlier
          score
    :type method: str, optional (default='largest')
    """

    def __init__(self, contamination=0.1, n_neighbors=5, method='largest'):
        super(KNN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.method = method

    def fit(self, X, y=None):

        # Validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        self.tree_ = KDTree(X)

        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        neigh.fit(X)

        dist_arr, _ = neigh.kneighbors(n_neighbors=self.n_neighbors,
                                       return_distance=True)

        if self.method == 'largest':
            dist = dist_arr[:, -1]
        elif self.method == 'mean':
            dist = np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            dist = np.median(dist_arr, axis=1)

        self.decision_scores_ = dist.ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X):

        check_is_fitted(self,
                        ['tree_', 'decision_scores_', 'threshold_', 'labels_'])

        X = check_array(X)

        # initialize the output score
        pred_scores = np.zeros([X.shape[0], 1])

        for i in range(X.shape[0]):
            x_i = X[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, _ = self.tree_.query(x_i, k=self.n_neighbors)

            if self.method == 'largest':
                dist = dist_arr[:, -1]
            elif self.method == 'mean':
                dist = np.mean(dist_arr, axis=1)
            elif self.method == 'median':
                dist = np.median(dist_arr, axis=1)

            pred_score_i = dist[-1]

            # record the current item
            pred_scores[i, :] = pred_score_i

        return pred_scores.ravel()
