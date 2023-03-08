# -*- coding: utf-8 -*-
""" A collection of statistical models
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from numba import njit
from scipy.stats import pearsonr
from sklearn.utils.validation import check_array
# noinspection PyProtectedMember
from sklearn.utils.validation import check_consistent_length


# TODO: disable p value calculation due to python 2.7 break
# from scipy.special import betainc


def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.

    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_helper(X, Y)


@njit
def _pairwise_distances_no_broadcast_helper(X, Y):  # pragma: no cover
    """Internal function for calculating the distance with numba. Do not use.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Intermediate results. Do not use.

    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def wpearsonr(x, y, w=None):
    """Utility function to calculate the weighted Pearson correlation of two
    samples.

    See https://stats.stackexchange.com/questions/221246/such-thing-as-a-weighted-correlation
    for more information

    Parameters
    ----------
    x : array, shape (n,)
        Input x.

    y : array, shape (n,)
        Input y.

    w : array, shape (n,)
        Weights w.

    Returns
    -------
    scores : float in range of [-1,1]
        Weighted Pearson Correlation between x and y.

    """

    # unweighted version
    # note the return is different
    # TODO: fix output differences
    if w is None:
        return pearsonr(x, y)

    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)

    check_consistent_length([x, y, w])
    # n = len(x)

    w_sum = w.sum()
    mx = np.sum(x * w) / w_sum
    my = np.sum(y * w) / w_sum

    xm, ym = (x - mx), (y - my)

    r_num = np.sum(xm * ym * w) / w_sum

    xm2 = np.sum(xm * xm * w) / w_sum
    ym2 = np.sum(ym * ym * w) / w_sum

    r_den = np.sqrt(xm2 * ym2)
    r = r_num / r_den

    r = max(min(r, 1.0), -1.0)

    # TODO: disable p value calculation due to python 2.7 break
    #    df = n_train_ - 2
    #
    #    if abs(r) == 1.0:
    #        prob = 0.0
    #    else:
    #        t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))
    #        prob = _betai(0.5 * df, 0.5, df / (df + t_squared))
    return r  # , prob


#####################################
#      PROBABILITY CALCULATIONS     #
#####################################

# TODO: disable p value calculation due to python 2.7 break
# def _betai(a, b, x):
#     x = np.asarray(x)
#     x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
#     return betainc(a, b, x)


def pearsonr_mat(mat, w=None):
    """Utility function to calculate pearson matrix (row-wise).

    Parameters
    ----------
    mat : numpy array of shape (n_samples, n_features)
        Input matrix.

    w : numpy array of shape (n_features,)
        Weights.

    Returns
    -------
    pear_mat : numpy array of shape (n_samples, n_samples)
        Row-wise pearson score matrix.

    """
    mat = check_array(mat)
    n_row = mat.shape[0]
    n_col = mat.shape[1]
    pear_mat = np.full([n_row, n_row], 1).astype(float)

    if w is not None:
        for cx in range(n_row):
            for cy in range(cx + 1, n_row):
                curr_pear = wpearsonr(mat[cx, :], mat[cy, :], w)
                pear_mat[cx, cy] = curr_pear
                pear_mat[cy, cx] = curr_pear
    else:
        for cx in range(n_col):
            for cy in range(cx + 1, n_row):
                curr_pear = pearsonr(mat[cx, :], mat[cy, :])[0]
                pear_mat[cx, cy] = curr_pear
                pear_mat[cy, cx] = curr_pear

    return pear_mat


def column_ecdf(matrix: np.ndarray) -> np.ndarray:
    """
    Utility function to compute the column wise empirical cumulative distribution of a 2D feature matrix,
    where the rows are samples and the columns are features per sample. The accumulation is done in the positive
    direction of the sample axis.

    E.G.
    p(1) = 0.2, p(0) = 0.3, p(2) = 0.1, p(6) = 0.4
    ECDF E(5) = p(x <= 5)
    ECDF E would be E(-1) = 0, E(0) = 0.3, E(1) = 0.5, E(2) = 0.6, E(3) = 0.6, E(4) = 0.6, E(5) = 0.6, E(6) = 1

    Similar to and tested against:
    https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html

    Returns
    -------

    """
    # check the matrix dimensions
    assert len(matrix.shape) == 2, 'Matrix needs to be two dimensional for the ECDF computation.'

    # create a probability array the same shape as the feature matrix which we will reorder to build
    # the ecdf
    probabilities = np.linspace(np.ones(matrix.shape[1]) / matrix.shape[0], np.ones(matrix.shape[1]), matrix.shape[0])

    # get the sorting indices for a numpy array
    sort_idx = np.argsort(matrix, axis=0)

    # sort the numpy array, as we need to look for duplicates in the feature values (that would have different
    # probabilities if we would just resort the probabilities array)
    matrix = np.take_along_axis(matrix, sort_idx, axis=0)

    # deal with equal values
    ecdf_terminate_equals_inplace(matrix, probabilities)

    # return the resorted accumulated probabilities (by reverting the sorting of the input matrix)
    # looks a little complicated but is faster this way
    reordered_probabilities = np.ones_like(probabilities)
    np.put_along_axis(reordered_probabilities, sort_idx, probabilities, axis=0)
    return reordered_probabilities


@njit
def ecdf_terminate_equals_inplace(matrix: np.ndarray, probabilities: np.ndarray):
    """
    This is a helper function for computing the ecdf of an array. It has been outsourced from the original
    function in order to be able to use the njit compiler of numpy for increased speeds, as it unfortunately
    needs a loop over all rows and columns of a matrix. It acts in place on the probabilities' matrix.

    Parameters
    ----------
    matrix : a feature matrix where the rows are samples and each column is a feature !(expected to be sorted)!

    probabilities : a probability matrix that will be used building the ecdf. It has values between 0 and 1 and
                    is also sorted.

    Returns
    -------

    """
    for cx in range(probabilities.shape[1]):
        for rx in range(probabilities.shape[0] - 2, -1, -1):
            if matrix[rx, cx] == matrix[rx + 1, cx]:
                probabilities[rx, cx] = probabilities[rx + 1, cx]
