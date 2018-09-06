# -*- coding: utf-8 -*-
"""k-Nearest Neighbors Detector (kNN)
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

    :param contamination: the amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    :type contamination: float in (0, 0.5], optional (default=0.1)

    :param n_neighbors: Number of neighbors to use by default
        for k neighbors queries.
    :type n_neighbors: int, optional (default=5)

    :param method: {'largest', 'mean', 'median'}

            - largest: use the distance to the kth neighbor as the outlier
              score
            - mean: use the average of all k neighbors as the outlier score
            - median: use the median of the distance to k neighbors as the
              outlier score
    :type method: str, optional (default='largest')

    :param radius: Range of parameter space to use by default for
        radius_neighbors queries. Not applicable
    :type radius: float, optional (default = 1.0)

    :param algorithm: Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use BallTree
            - 'kd_tree' will use KDTree
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    :type algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional

    :param leaf_size: Leaf size passed to BallTree or KDTree. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    :type leaf_size: int, optional (default=30)

    :param metric: metric used for the distance computation. Any metric from
        scikit-learn or scipy.spatial.distance can be used.

        If 'precomputed', the training input X is expected to be a distance
        matrix.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1',
          'l2','manhattan']
        - from scipy.spatial.distance: ['braycurtis', 'canberra',
          'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
          'kulsinski', 'mahalanobis', 'matching', 'minkowski',
          'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
          'sokalsneath', 'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    :type metric: str or callable, default 'minkowski'

    :param p: Parameter for the Minkowski metric for sklearn.metrics.pairwise.
        pairwise_distances.
        When p = 1, this is equivalent to using manhattan_distance (l1), and
        euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance
        (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
    :type p: int, optional (default=2)

    :param metric_params: Additional keyword arguments for the metric function.
    :type metric_params: dict, optional (default=None)

    :param n_jobs: The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.
    :type n_jobs: int, optional (default=1)

    :var decision_scores\_: The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    :vartype decision_scores\_: numpy array of shape (n_samples,)

    :var threshold\_: The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    :vartype threshold\_: float

    :var labels\_: The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    :vartype labels\_: int, either 0 or 1
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

        # Validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

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
