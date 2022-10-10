# -*- coding: utf-8 -*-
"""k-Nearest Neighbors Detector (kNN)
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
from __future__ import division
from __future__ import print_function

from warnings import warn

import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


# TODO: algorithm parameter is deprecated and will be removed in 0.7.6.
# Warning has been turned on.
# TODO: since Ball_tree is used by default, may introduce its parameters.

class KNN(BaseDetector):
    # noinspection PyPep8
    """kNN class for outlier detection.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.

    Three kNN detectors are supported:
    largest: use the distance to the kth neighbor as the outlier score
    mean: use the average of all k neighbors as the outlier score
    median: use the median of the distance to k neighbors as the outlier score

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.

    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}

        - 'largest': use the distance to the kth neighbor as the outlier score
        - 'mean': use the average of all k neighbors as the outlier score
        - 'median': use the median of the distance to k neighbors as the
          outlier score

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for `radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

        .. deprecated:: 0.74
           ``algorithm`` is deprecated in PyOD 0.7.4 and will not be
           possible in 0.7.6. It has to use BallTree for consistency.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, contamination=0.1, n_neighbors=5, method='largest',
                 radius=1.0, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None, n_jobs=1,
                 **kwargs):
        super(KNN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        if self.algorithm != 'auto' and self.algorithm != 'ball_tree':
            warn('algorithm parameter is deprecated and will be removed '
                 'in version 0.7.6. By default, ball_tree will be used.',
                 FutureWarning)

        self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                       radius=self.radius,
                                       algorithm=self.algorithm,
                                       leaf_size=self.leaf_size,
                                       metric=self.metric,
                                       p=self.p,
                                       metric_params=self.metric_params,
                                       n_jobs=self.n_jobs,
                                       **kwargs)

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        self.neigh_.fit(X)

        # In certain cases, _tree does not exist for NearestNeighbors
        # See Issue #158 (https://github.com/yzhao062/pyod/issues/158)
        # n_neighbors = 100
        if self.neigh_._tree is not None:
            self.tree_ = self.neigh_._tree

        else:
            if self.metric_params is not None:
                self.tree_ = BallTree(X, leaf_size=self.leaf_size,
                                      metric=self.metric,
                                      **self.metric_params)
            else:
                self.tree_ = BallTree(X, leaf_size=self.leaf_size,
                                      metric=self.metric)

        dist_arr, _ = self.neigh_.kneighbors(n_neighbors=self.n_neighbors,
                                             return_distance=True)
        dist = self._get_dist_by_method(dist_arr)

        self.decision_scores_ = dist.ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['tree_', 'decision_scores_',
                               'threshold_', 'labels_'])

        X = check_array(X)

        # initialize the output score
        pred_scores = np.zeros([X.shape[0], 1])

        for i in range(X.shape[0]):
            x_i = X[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, _ = self.tree_.query(x_i, k=self.n_neighbors)
            dist = self._get_dist_by_method(dist_arr)
            pred_score_i = dist[-1]

            # record the current item
            pred_scores[i, :] = pred_score_i

        return pred_scores.ravel()

    def _get_dist_by_method(self, dist_arr):
        """Internal function to decide how to process passed in distance array

        Parameters
        ----------
        dist_arr : numpy array of shape (n_samples, n_neighbors)
            Distance matrix.

        Returns
        -------
        dist : numpy array of shape (n_samples,)
            The outlier scores by distance.
        """

        if self.method == 'largest':
            return dist_arr[:, -1]
        elif self.method == 'mean':
            return np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            return np.median(dist_arr, axis=1)
