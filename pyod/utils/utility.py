# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
import numbers

import sklearn
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.testing import assert_equal
from scipy.stats import scoreatpercentile

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input parameter is with in the range low and high bounds.

    :param param: The input parameter to check
    :type param: int, float

    :param low: The lower bound of the range
    :type low: int, float

    :param high: The higher bound of the range
    :type high: int, float

    :param param_name: The name of the parameter
    :type param_name: str, optional (default='')

    :param include_left: Whether includes the lower bound (lower bound <=)
    :type include_left: bool, optional (default=False)

    :param include_right: Whether includes the higher bound (<= higher bound )
    :type include_right: bool, optional (default=False)

    :return: Whether the parameter is within the range of (low, high)
    :rtype: bool or raise errors
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, np.float)):
        raise TypeError(
            '{param_name} is set to {param}. '
            'Not numerical'.format(param=param,
                                   param_name=param_name))

    if not isinstance(low, (numbers.Integral, np.integer, np.float)):
        raise TypeError(
            'low is set to {low}. ''Not numerical'.format(low=low))
    if not isinstance(high, (numbers.Integral, np.integer, np.float)):
        raise TypeError(
            'high is set to {high}. ''Not numerical'.format(high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True


def standardizer(X, X_t=None):
    """Conduct Z-normalization on data to turn input samples become zero-mean
    and unit variance.

    :param X: The training samples
    :type X: numpy array of shape (n_samples, n_features)

    :param X_t: The data to be converted
    :type X_t: numpy array of shape (n_samples_new, n_features)

    :return: X and X_t after the Z-score normalization
    :rtype: ndarray (n_samples, n_features),
        ndarray (n_samples_new, n_features)
    """
    X = check_array(X)
    if X_t is None:
        return StandardScaler().fit_transform(X)

    X_t = check_array(X_t)
    assert_equal(X.shape[1], X_t.shape[1])
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler.transform(X_t)


def score_to_label(pred_scores, outliers_fraction=0.1):
    """Turn raw outlier outlier scores to binary labels (0 or 1)

    :param pred_scores: raw outlier scores
    :param outliers_fraction: percentage of outliers
    :return: binary labels (1 stands for outlier)
    :rtype: int
    """
    threshold = scoreatpercentile(pred_scores, 100 * (1 - outliers_fraction))
    pred_labels = (pred_scores > threshold).astype('int')
    return pred_labels


def precision_n_scores(y, y_pred, n=None):
    """
    Utility function to calculate precision@ rank

    :param y: ground truth
    :param y_pred: predicted outlier scores as returned by fitted model (not rounded off)
    :param n: number of outliers, if not defined, infer using ground truth
    :return: precision at rank n score
    :rtype: float
    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """
    Function to turn raw outlier scores into binary labels by assign 1 to top
    n outlier scores.

    Example y: [0,1,1,0,0,0]
            y_pred: [0.1, 0.5, 0.3, 0.2, 0.7]
            return [0, 1, 0, 0, 1]

    :param y: ground truth
    :type y: list or numpy array of shape (n_samples,)

    :param y_pred: raw outlier scores
    :type y_pred: list or numpy array of shape (n_samples,)

    :param n: number of outliers, if not defined, infer using ground truth
    :type n: int

    :return: binary labels 0: normal points and 1: outliers
    """

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = scoreatpercentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred


def argmaxn(value_list, n, order='desc'):
    """
    Return the index of top n elements in the list if order is set to 'desc',
    otherwise return the index of n smallest elements

    :param value_list: a list containing all values
    :type value_list: list, array
    :param n: the number of the elements to select
    :type n: int
    :param order: the order to sort {'desc', 'asc'}
    :type order: str, optional (default='desc')
    :return: the index of the top n elements
    :rtype: list
    """
    value_list = column_or_1d(value_list)
    length = len(value_list)

    # validate the choice of n
    check_parameter(n, 1, length, include_left=True, include_right=True,
                    param_name='n')

    # for the smallest n, flip the value
    if order != 'desc':
        n = length - n

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if order == 'desc':
        return np.where(np.greater_equal(value_list, threshold))[0]
    else:  # return the index of n smallest elements
        return np.where(np.less(value_list, threshold))[0]


def invert_order(scores, method='multiplication'):
    """ Invert the order of a list of values. The smallest value becomes
    the largest in the inverted list. This is useful while combining
    multiple detectors since their score order could be different.

    Examples:
        >>>scores1 = [0.1, 0.3, 0.5, 0.7, 0.2, 0.1]
        >>>invert_order(scores1)
        array[-0.1, -0.3, -0.5, -0.7, -0.2, -0.1]
        >>>invert_order(scores1, method='subtraction')
        array[0.6, 0.4, 0.2, 0, 0.5, 0.6]

    :param scores: The list of values to be inverted
    :type scores: list, array or numpy array with shape (n_samples,)

    :param method: {'multiplication', 'subtraction'}:

            - 'multiplication': multiply by -1
            - 'subtraction': max(scores) - scores
    :type method: str, optional (default='multiplication')

    :return: The inverted list
    :rtype: numpy array of shape (n_samples,)
    """

    scores = column_or_1d(scores)

    if method == 'multiplication':
        return scores.ravel() * -1

    if method == 'subtraction':
        return (scores.max() - scores).ravel()

def _sklearn_version_20():
    """ Utility function to decide the version of sklearn
    In sklearn 20.0, LOF is changed. Specifically, _decision_function
    is replaced by _score_samples

    :return: True if sklearn.__version__ is newer than 0.20.0
    """
    sklearn_version = str(sklearn.__version__)
    if int(sklearn_version.split(".")[1]) > 19:
        return True
    else:
        return False

