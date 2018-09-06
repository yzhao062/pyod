# -*- coding: utf-8 -*-
""" A collection of statistical models
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np

from scipy.stats import pearsonr
from sklearn.utils.validation import check_array
from sklearn.utils.testing import assert_allclose
from sklearn.utils.validation import check_consistent_length


# TODO: disable p value calculation due to python 2.7 break
# from scipy.special import betainc

def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.

    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4)

    :param X: First input samples
    :param X: array of shape (n_samples, n_features)

    :param Y: Second input samples
    :type Y: array of shape (n_samples, n_features)

    :return: Row-wise euclidean distnace of X and Y
    :rtype: array of shape (n_samples,)
    """
    X = check_array(X)
    Y = check_array(Y)
    assert_allclose(X.shape, Y.shape)
    euclidean_sq = np.square(Y - X)

    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def wpearsonr(x, y, w=None):
    # noinspection PyPep8
    """Utility function to calculate the weighted Pearson correlation of two
    samples.

    See https://stats.stackexchange.com/questions/221246/such-thing-as-a-weighted-correlation
    for more information

    :param x: Input x
    :type x: array, shape (n,)

    :param y: Input y
    :type y: array, shape (n,)

    :param w: Weights
    :type w: array, shape (n,)

    :return: Weighted Pearson Correlation between x and y
    :rtype: float [-1,1]
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
