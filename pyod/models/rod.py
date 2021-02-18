# -*- coding: utf-8 -*-
"""Rotation-based Outlier Detector (ROD)
"""
# Author: Yahya Almardeny <almardeny@gmail.com>
# License: BSD 2 clause
from __future__ import division
from __future__ import print_function

from itertools import combinations as com
from multiprocessing import Pool
import multiprocessing
import numba
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import check_array

from .base import BaseDetector


@numba.njit
def mad(costs):
    """Apply the robust median absolute deviation (MAD)
    to measure the inconsistency/variability of the
    rotation costs.

    Parameters
    ----------
    costs :

    Returns
    -------
    z : float
        the modified z scores
    """
    costs_ = np.reshape(costs, (-1, 1))
    median = np.nanmedian(costs_)
    diff = np.abs(costs_ - median)
    return np.ravel(0.6745 * diff / np.median(diff))

def angle(v1, v2):
    """find the angle between two 3D vectors
    """
    return np.arccos(np.dot(v1, v2) /
                     (np.linalg.norm(v1) * np.linalg.norm(v2)))


def geometric_median(x, eps=1e-5):
    """
    Find the multivariate geometric L1-median by applying
    Vardi and Zhang algorithm.
    :param x: the data points in array-like structure
    :param eps: float, a threshold to indicate when to stop
    :return: Geometric L1-median
    """
    points = np.unique(x, axis=0)
    gm_ = np.mean(points, 0)  # initialize geometric median
    while True:
        D = euclidean(points, gm_, c=True)
        non_zeros = (D != 0)[:, 0]
        Dinv = 1 / D[non_zeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * points[non_zeros], 0)
        num_zeros = len(points) - np.sum(non_zeros)
        if num_zeros == 0:
            gm1 = T
        elif num_zeros == len(points):
            return gm_
        else:
            R = (T - gm_) * Dinvs
            r = np.linalg.norm(R)
            r_inv = 0 if r == 0 else num_zeros / r
            gm1 = max(0, 1 - r_inv) * T + min(1, r_inv) * gm_

        if euclidean(gm_, gm1) < eps:
            return gm1

        gm_ = gm1


def scale_angles(gammas):
    """
    Scale all angles in which angles <= 90
    degree will be scaled within [0 - 54.7] and
    angles > 90 will be scaled within [90 - 126]
    :param gammas: list of angles
    :return: scaled angles
    """
    first, second = [], []
    first_ind, second_ind = [], []
    q1 = np.pi / 2.
    for i, g in enumerate(gammas):
        if g <= q1:
            first.append(g)
            first_ind.append(i)
        else:
            second.append(g)
            second_ind.append(i)
    first = MinMaxScaler((0.001, 0.955)).fit_transform(
        np.array(first).reshape(-1, 1)).reshape(-1) if first else []
    second = MinMaxScaler((q1 + 0.001, 2.186)).fit_transform(
        np.array(second).reshape(-1, 1)).reshape(-1) if second else []
    # restore original order
    return np.concatenate([first, second])[np.argsort(first_ind + second_ind)]


def euclidean(v1, v2, c=False):
    """
    Find the euclidean distance between two vectors
    or between a vector and a collection of vectors.
    :param v1: list, first 3D vector or collection of vectors
    :param v2: list, second 3D vector
    :param c: bool (default=False), if True, it means the
              v1 is a list of vectors.
    :return: list of list of euclidean distances if c==True.
             Otherwise float: the euclidean distance
    """
    if c:
        res = []
        for _v in v1:
            res.append([np.sqrt((_v[0] - v2[0]) ** 2 +
                                (_v[1] - v2[1]) ** 2 +
                                (_v[2] - v2[2]) ** 2)])
        return np.array(res, copy=False)
    return np.sqrt((v1[0] - v2[0]) ** 2 +
                   (v1[1] - v2[1]) ** 2 +
                   (v1[2] - v2[2]) ** 2)


def rod_3D(x):
    """
    Find ROD scores for 3D Data.
    """
    # find the geometric median
    gm = geometric_median(x)
    # find its norm and center data around it
    norm_ = np.linalg.norm(gm)
    _x = x - gm
    # calculate the scaled angles between the geometric median
    # and each data point vector
    v_norm = np.linalg.norm(_x, axis=1)
    gammas = scale_angles(np.arccos(np.clip(np.dot(_x, gm) / (v_norm * norm_), -1, 1)))
    # apply the ROD main equation to find the rotation costs
    costs = np.power(v_norm, 3) * np.cos(gammas) * np.square(np.sin(gammas))
    # apply MAD to calculate the decision scores
    return mad(costs)


@numba.njit
def sigmoid(xx):
    """
    Implementation of Sigmoid function
    """
    return 1 / (1 + np.exp(-xx))


def process_sub(subspace):
    """
    Apply ROD on a 3D subSpace
    """
    mad_subspace = np.nan_to_num(np.array(rod_3D(subspace)))
    return sigmoid(mad_subspace)


def rod_nD(X, parallel):
    """
    Find ROD overall scores when n>3 Data:
      # scale dataset using Robust Scaler
      # decompose the full space into a combinations of 3D subspaces,
      # Apply ROD on each combination,
      # squish scores per subspace, so we compare apples to apples,
      # return the average of ROD scores of all subspaces per observation.
    """
    X = RobustScaler().fit_transform(X)
    dim = X.shape[1]
    all_subspaces = [X[:, _com] for _com in com(range(dim), 3)]
    if parallel:
        p = Pool(multiprocessing.cpu_count())
        subspaces_scores = p.map(process_sub, all_subspaces)
        return np.average(np.array(subspaces_scores).T, axis=1).reshape(-1)
    subspaces_scores = []
    for subspace in all_subspaces:
        subspaces_scores.append(process_sub(subspace))
    return np.average(np.array(subspaces_scores).T, axis=1).reshape(-1)


class ROD(BaseDetector):
    """Rotation-based Outlier Detection (ROD), is a robust and parameter-free
    algorithm that requires no statistical distribution assumptions and
    works intuitively in three-dimensional space, where the 3D-vectors,
    representing the data points, are rotated about the geometric median
    two times counterclockwise using Rodrigues rotation formula.
    The results of the rotation are parallelepipeds where their volumes are
    mathematically analyzed as cost functions and used to calculate the
    Median Absolute Deviations to obtain the outlying score.
    For high dimensions > 3, the overall score is calculated by taking the
    average of the overall 3D-subspaces scores, that were resulted from
    decomposing the original data space.
    See :cite:`Y. Almardeny, N. Boujnah and F. Cleary,
    "A Novel Outlier Detection Method for Multivariate Data,"
    in IEEE Transactions on Knowledge and Data Engineering,
    doi: 10.1109/TKDE.2020.3036524.` for details.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    parallel_execution: bool, optional (default=False).
        If set to True, it makes the algorithm run in parallel,
        for a better execution time. It is recommended to set
        this parameter to True ONLY for high dimensional data > 10,
        if a proper hardware is available.


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

    def __init__(self, contamination=0.1, parallel_execution=False):
        super(ROD, self).__init__(contamination=contamination)
        if not isinstance(parallel_execution, bool):
            raise TypeError("parallel_execution should be bool. "
                            "Got {}".format(type(parallel_execution)))
        self.parallel = parallel_execution

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
        X = check_array(X)
        self._set_n_classes(y)
        self.decision_scores_ = self.decision_function(X)
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
        X = check_array(X)
        if X.shape[1] < 3:
            X = np.hstack((X, np.zeros(shape=(X.shape[0], 3 - X.shape[1]))))

        if X.shape[1] == 3:
            return rod_3D(X)

        return rod_nD(X, self.parallel)
