# -*- coding: utf-8 -*-
"""
A set of utility functions to support outlier detection
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import numbers
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.testing import assert_equal

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """
    Check if an input parameter is with in the range low and high bounds.

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


def standardizer(X_train, X_test):
    """
    normalization function wrapper, z- normalization function

    :param X_train:
    :param X_test:
    :return: X_train_ and X_test after the Z-score normalization
    :rtype: tuple(ndarray, ndarray)
    """
    X_train = check_array(X_train)
    X_test = check_array(X_test)
    assert_equal(X_train.shape[1], X_test.shape[1])
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train), scaler.transform(X_test))


def score_to_label(pred_scores, outlier_perc=0.1):
    """
    turn raw outlier decision_scores_ to binary labels (0 or 1)

    :param pred_scores: raw outlier decision_scores_
    :param outlier_perc: percentage of outliers
    :return: binary labels (1 stands for outlier)
    :rtype: int
    """
    threshold = scoreatpercentile(pred_scores, 100 * (1 - outlier_perc))
    pred_labels = (pred_scores > threshold).astype('int')
    return pred_labels


def precision_n_scores(y, y_pred, n=None):
    """
    Utility function to calculate precision@ rank

    :param y: ground truth
    :param y_pred: number of outliers
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
            labels_: [0.1, 0.5, 0.3, 0.2, 0.7]
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
        outlier_perc = n / y_len
    else:
        outlier_perc = np.count_nonzero(y) / y_len

    threshold = scoreatpercentile(y_pred, 100 * (1 - outlier_perc))
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
