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
def mad(costs, median=None):
    """Apply the robust median absolute deviation (MAD)
    to measure the inconsistency/variability of the
    rotation costs.

    Parameters
    ----------
    costs : list of rotation costs
    median: float (default=None), MAD median

    Returns
    -------
    z : float
        the modified z scores
    """
    costs_ = np.reshape(costs, (-1, 1))
    median = np.nanmedian(costs_) if median is None else median
    diff = np.abs(costs_ - median)
    return np.ravel(0.6745 * diff / np.median(diff)), median


def angle(v1, v2):
    """find the angle between two 3D vectors.

    Parameters
    ----------
    v1 : list, first vector
    v2 : list, second vector

    Returns
    -------
    angle : float, the angle
    """
    return np.arccos(np.dot(v1, v2) /
                     (np.linalg.norm(v1) * np.linalg.norm(v2)))


def geometric_median(x, eps=1e-5):
    """
    Find the multivariate geometric L1-median by applying
    Vardi and Zhang algorithm.

    Parameters
    ----------
    x : array-like, the data points
    eps: float (default=1e-5), a threshold to indicate when to stop

    Returns
    -------
    gm : array, Geometric L1-median
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


def scale_angles(gammas, scaler1=None, scaler2=None):
    """
    Scale all angles in which angles <= 90
    degree will be scaled within [0 - 54.7] and
    angles > 90 will be scaled within [90 - 126]

    Parameters
    ----------
    gammas : list, angles
    scaler1: obj (default=None), MinMaxScaler of Angles group 1
    scaler2: obj (default=None), MinMaxScaler of Angles group 2

    Returns
    -------
    scaled angles, scaler1, scaler2
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
    if scaler1 is None:  # this indicates the `fit()`
        min_f, max_f = 0.001, 0.955
        scaler1 = MinMaxScaler(feature_range=(min_f, max_f))
        # min_f and max_f are required to be fit by scaler for consistency between train and test sets
        scaler1.fit(np.array(first + [min_f, max_f]).reshape(-1, 1))
        first = scaler1.transform(np.array(first).reshape(-1, 1)).reshape(-1) if first else []
    else:
        first = scaler1.transform(np.array(first).reshape(-1, 1)).reshape(-1) if first else []
    if scaler2 is None:  # this indicates the `fit()`
        min_s, max_s = q1 + 0.001, 2.186
        scaler2 = MinMaxScaler(feature_range=(min_s, max_s))
        # min_s and max_s are required to be fit by scaler for consistency between train and test sets
        scaler2.fit(np.array(second + [min_s, max_s]).reshape(-1, 1))
        second = scaler2.transform(np.array(second).reshape(-1, 1)).reshape(-1) if second else []
    else:
        second = scaler2.transform(np.array(second).reshape(-1, 1)).reshape(-1) if second else []
    # restore original order
    return np.concatenate([first, second])[np.argsort(first_ind + second_ind)], scaler1, scaler2


def euclidean(v1, v2, c=False):
    """
    Find the euclidean distance between two vectors
    or between a vector and a collection of vectors.

    Parameters
    ----------
    v1 : list, first 3D vector or collection of vectors
    v2 : list, second 3D vector
    c : bool (default=False), if True, it means the v1 is a list of vectors.

    Returns
    -------
    list of list of euclidean distances if c==True.
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


def rod_3D(x, gm=None, median=None, scaler1=None, scaler2=None):
    """
    Find ROD scores for 3D Data.
    note that gm, scaler1 and scaler2 will be returned "as they are"
    and without being changed if the model has been fit already

    Parameters
    ----------
    x : array-like, 3D data points.
    gm: list (default=None), the geometric median
    median: float (default=None), MAD median
    scaler1: obj (default=None), MinMaxScaler of Angles group 1
    scaler2: obj (default=None), MinMaxScaler of Angles group 2

    Returns
    -------
    decision_scores, gm, scaler1, scaler2
    """
    # find the geometric median if it is not already fit
    gm = geometric_median(x) if gm is None else gm
    # find its norm and center data around it
    norm_ = np.linalg.norm(gm)
    _x = x - gm
    # calculate the scaled angles between the geometric median and each data point vector
    v_norm = np.linalg.norm(_x, axis=1)
    gammas, scaler1, scaler2 = scale_angles(np.arccos(np.clip(np.dot(_x, gm) / (v_norm * norm_), -1, 1)),
                                            scaler1=scaler1, scaler2=scaler2)
    # apply the ROD main equation to find the rotation costs
    costs = np.power(v_norm, 3) * np.cos(gammas) * np.square(np.sin(gammas))
    # apply MAD to calculate the decision scores
    decision_scores, median = mad(costs, median=median)
    return decision_scores, list(gm), median, scaler1, scaler2


@numba.njit
def sigmoid(x):
    """
    Implementation of Sigmoid function

    Parameters
    ----------
    x : array-like, decision scores

    Returns
    -------
    array-like, x after applying sigmoid
    """
    return 1 / (1 + np.exp(-x))


def process_sub(subspace, gm, median, scaler1, scaler2):
    """
    Apply ROD on a 3D subSpace then process it with sigmoid
    to compare apples to apples

    Parameters
    ----------
    subspace : array-like, 3D subspace of the data
    gm: list, the geometric median
    median: float, MAD median
    scaler1: obj, MinMaxScaler of Angles group 1
    scaler2: obj, MinMaxScaler of Angles group 2

    Returns
    -------
    ROD decision scores with sigmoid applied, gm, scaler1, scaler2
    """
    mad_subspace, gm, median, scaler1, scaler2 = rod_3D(subspace, gm=gm,
                                                        median=median,
                                                        scaler1=scaler1,
                                                        scaler2=scaler2)
    return sigmoid(np.nan_to_num(np.array(mad_subspace))), gm, median, scaler1, scaler2


def rod_nD(X, parallel, gm=None, median=None, data_scaler=None, angles_scalers1=None, angles_scalers2=None):
    """
    Find ROD overall scores when Data is higher than 3D:
      # scale dataset using Robust Scaler
      # decompose the full space into a combinations of 3D subspaces,
      # Apply ROD on each combination,
      # squish scores per subspace, so we compare apples to apples,
      # calculate average of ROD scores of all subspaces per observation.
    Note that if gm, data_scaler, angles_scalers1, angles_scalers2 are None,
    that means it is a `fit()` process and they will be calculated and returned
    to the class to be saved for future prediction. Otherwise, if they are not None,
    then it is a prediction process.

    Parameters
    ----------
    X : array-like, data points
    parallel: bool, True runs the algorithm in parallel
    gm: list (default=None), the geometric median
    median: list (default=None), MAD medians
    data_scaler: obj (default=None), RobustScaler of data
    angles_scalers1: list (default=None), MinMaxScalers of Angles group 1
    angles_scalers2: list (default=None), MinMaxScalers of Angles group 2

    Returns
    -------
    ROD decision scores, gm, median, data_scaler, angles_scalers1, angles_scalers2
    """
    if data_scaler is None:  # for fitting
        data_scaler = RobustScaler()
        X = data_scaler.fit_transform(X)
    else:  # for prediction
        X = data_scaler.transform(X)
    dim = X.shape[1]
    all_subspaces = [X[:, _com] for _com in com(range(dim), 3)]
    all_gms = [None] * len(all_subspaces) if gm is None else gm
    all_meds = [None] * len(all_subspaces) if median is None else median
    all_angles_scalers1 = [None] * len(all_subspaces) if angles_scalers1 is None else angles_scalers1
    all_angles_scalers2 = [None] * len(all_subspaces) if angles_scalers2 is None else angles_scalers2
    if parallel:
        p = Pool(multiprocessing.cpu_count())
        args = [[a, b, c, d, e] for a, b, c, d, e in zip(all_subspaces, all_gms, all_meds,
                                                         all_angles_scalers1, all_angles_scalers2)]
        results = p.starmap(process_sub, args)
        subspaces_scores, gm, median, angles_scalers1, angles_scalers2 = [], [], [], [], []
        for res in results:
            subspaces_scores.append(list(res[0]))
            gm.append(res[1])
            median.append(res[2])
            angles_scalers1.append(res[3])
            angles_scalers2.append(res[4])
        scores = np.average(np.array(subspaces_scores).T, axis=1).reshape(-1)
        p.close()
        p.join()
        return scores, gm, median, data_scaler, angles_scalers1, angles_scalers2
    subspaces_scores, gm, median, angles_scalers1, angles_scalers2 = [], [], [], [], []
    for subspace, _gm, med, ang_s1, ang_s2 in zip(all_subspaces, all_gms, all_meds, all_angles_scalers1,
                                                  all_angles_scalers2):
        scores_, gm_, med_, ang_s1_, ang_s2_ = process_sub(subspace=subspace, gm=_gm, median=med,
                                                           scaler1=ang_s1, scaler2=ang_s2)
        subspaces_scores.append(scores_)
        gm.append(gm_)
        median.append(med_)
        angles_scalers1.append(ang_s1_)
        angles_scalers2.append(ang_s2_)
    scores = np.average(np.array(subspaces_scores).T, axis=1).reshape(-1)
    return scores, gm, median, data_scaler, angles_scalers1, angles_scalers2


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
    See :cite:`almardeny2020novel` for details.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    parallel_execution: bool, optional (default=False).
        If set to True, the algorithm will run in parallel,
        for a better execution time. It is recommended to set
        this parameter to True ONLY for high dimensional data > 10,
        and if a proper hardware is available.

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
        self.gm = None  # geometric median(s)
        self.median = None  # MAD median(s)
        self.data_scaler = None  # data scaler (in case of d>3)
        self.angles_scaler1 = None  # scaler(s) of Angles Group 1
        self.angles_scaler2 = None  # scaler(s) of Angles Group 2

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
        # reset learning parameters after each fit
        self.gm = None
        self.median = None
        self.data_scaler = None
        self.angles_scaler1 = None
        self.angles_scaler2 = None
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
            scores, self.gm, self.median, self.angles_scaler1, self.angles_scaler2 = rod_3D(x=X, gm=self.gm,
                                                                                            median=self.median,
                                                                                            scaler1=self.angles_scaler1,
                                                                                            scaler2=self.angles_scaler2)
            return scores

        scores, self.gm, self.median, self.data_scaler, \
            self.angles_scaler1, self.angles_scaler2 = rod_nD(X=X,
                                                              parallel=self.parallel,
                                                              gm=self.gm,
                                                              median=self.median,
                                                              data_scaler=self.data_scaler,
                                                              angles_scalers1=self.angles_scaler1,
                                                              angles_scalers2=self.angles_scaler2)
        return scores
