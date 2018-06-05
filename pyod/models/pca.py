# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array

from .base import BaseDetector


# TODO: placeholder, do not use
class PCA(BaseDetector):

    def __init__(self, n_components=None, contamination=0.1, copy=True,
                 whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None):
        super(PCA, self).__init__(contamination=contamination)
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        """
        Fit the model using X as training data.

        :param X: Training data. If array or matrix,
            shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        :type X: {array-like, sparse matrix, BallTree, KDTree}

        :return: self
        :rtype: object
        """
        # Validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        self.detector_ = sklearn_PCA(n_components=self.n_components,
                                     copy=self.copy,
                                     whiten=self.whiten,
                                     svd_solver=self.svd_solver,
                                     tol=self.tol,
                                     iterated_power=self.iterated_power,
                                     random_state=self.random_state)
        self.detector_.fit(X=X, y=y)
        # self.decision_scores_ =
        # self._process_decision_scores()
        return self

    def decision_function(self, X):
        pass
