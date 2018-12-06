# -*- coding: utf-8 -*-
"""XGBOD: Improving Supervised Outlier Detection with Unsupervised
Representation Learning
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
from .knn import KNN
from .lof import LOF
from .iforest import IForest
from .hbos import HBOS
from .ocsvm import OCSVM
from ..utils.utility import check_detector


class XGBOD(BaseDetector):
    """XGBOD class for outlier detection.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.


    Parameters
    ----------
    estimator_list
    standardization_flag_list
    random_state

    """

    def __init__(self, estimator_list=None, standardization_flag_list=None,
                 random_state=None):
        super(XGBOD, self).__init__()
        self.estimator_list = estimator_list
        self.standardization_flag_list = standardization_flag_list
        self.random_state = random_state

        if self.standardization_flag_list is None:
            if len(self.estimator_list) != len(self.standardization_flag_list):
                raise ValueError(
                    "estimator_list length ({0}) is not equal "
                    "to standardization_flag_list length ({1})".format(
                        len(self.estimator_list),
                        len(self.standardization_flag_list)))

    def _init_detectors(self, X):
        estimator_list = []
        standardization_flag_list = []

        # predefined range of k
        k_range = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50,
                   60, 70, 80, 90, 100, 150, 200, 250]
        # validate the value of k
        k_range = [k for k in k_range if k < X.shape[0]]

        for k in k_range:
            estimator_list.append(KNN(n_neighbors=k, method='largest'))
            estimator_list.append(KNN(n_neighbors=k, method='mean'))
            estimator_list.append(LOF(n_neighbors=k))
            standardization_flag_list.append(True)
            standardization_flag_list.append(True)
            standardization_flag_list.append(True)

        n_bins_range = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50]
        for n_bins in n_bins_range:
            estimator_list.append(HBOS(n_bins=n_bins))
            standardization_flag_list.append(False)

        # predefined range of nu for one-class svm
        nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        for nu in nu_range:
            estimator_list.append(OCSVM(nu=nu))
            standardization_flag_list.append(True)

        # predefined range for number of estimators in isolation forests
        n_range = [10, 20, 50, 70, 100, 150, 200, 250]
        for n in n_range:
            estimator_list.append(IForest(n_estimators=n))
            standardization_flag_list.append(False)

        return estimator_list, standardization_flag_list

    def fit(self, X, y=None):

        # Validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        if self.estimator_list is None:
            self.estimator_list, \
            self.standardization_flag_list = self._init_detectors(X)

        if self.standardization_flag_list is None:
            self.standardization_flag_list = [True] * len(self.estimator_list)

        self.tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        self.neigh_.fit(X)

        dist_arr, _ = self.neigh_.kneighbors(n_neighbors=self.n_neighbors,
                                             return_distance=True)

        dist = np.zeros(shape=(X.shape[0], 1))
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
