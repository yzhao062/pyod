# -*- coding: utf-8 -*-
"""
A set of utility functions to support outlier detection
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d


def check_parameter_range(param, low=None, high=None):
    """
    check if input parameter is with in the range low and high

    :param param: the input parameter to check
    :type param: int, float

    :param low: lower bound of the range
    :type low: int, float

    :param high: higher bound of the range
    :type high: int, float

    :return: whether the parameter is within the range of (low, high)
    :rtype: bool
    """
    if low is None or high is None:
        raise ValueError('either low or high bounds is undefined')

    if low is not None and high is not None:
        if low >= high:
            raise ValueError('low is equal or larger than high')

    if not isinstance(param, int) and not isinstance(param, float):
        raise TypeError('{param} is not numerical'.format(param=param))

    if param < low or param > high:
        raise ValueError(
            '{param} is not in the range of {low} and {high}'.format(
                param=param, low=low, high=high))
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
    Utlity function to calculate precision@ rank

    :param y: ground truth
    :param y_pred: number of outliers
    :param n: number of outliers, if not defined, infer using ground truth
    :return: precison at rank n score
    :rtype: float
    """

    # turn prediction decision_scores_ into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """
    Function to turn decision_scores_ into binary labels by assign 1 to top
    n_train_ decision_scores_.

    Example y: [0,1,1,0,0,0]
            labels_: [0.1, 0.5, 0.3, 0.2, 0.7]
            return [0, 1, 0, 0, 1]

    :param y: ground truth
    :param y_pred: number of outliers
    :param n: number of outliers, if not defined, infer using ground truth
    :return: binary labels 0: normal points and 1: outliers
    """

    # enforce formats of imputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    y_len = len(y)  # the length of targets

    if y.shape != y_pred.shape:
        ValueError(
            'ground truth y and prediction labels_ shape does not match')

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
    check_parameter_range(n, 1, length)

    # for the smallest n, flip the value
    if order != 'desc':
        n = length - n

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if order == 'desc':
        return np.where(np.greater_equal(value_list, threshold))[0]
    else:  # return the index of n smallest elements
        return np.where(np.less(value_list, threshold))[0]
