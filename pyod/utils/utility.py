"""
A set of utility functions to support outlier detection
"""
import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d


def check_parameter_range(para, low=None, high=None):
    """
    check if input parameter is with in the range low and high

    :param para: the input parameter to check
    :type para: int, float

    :param low: lower bound of the range
    :type low: int, float

    :param high: higher bound of the range
    :type high: int, float

    :return: whether the parameter is within the range of (low, high)
    :rtype: bool
    """
    if low is None and high is None:
        raise ValueError('both low and high bounds are undefined')

    if low is not None and high is not None:
        if low >= high:
            raise ValueError('low is equal or larger than high')

    if not isinstance(para, int) and not isinstance(para, float):
        raise TypeError('{para} is not numerical'.format(para=para))

    if para < low or para > high:
        return False
    else:
        return True


def standardizer(X_train, X_test):
    """
    normalization function wrapper, z- normalization function

    :param X_train:
    :param X_test:
    :return: X_train and X_test after the Z-score normalization
    :rtype: tuple(ndarray, ndarray)
    """
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train), scaler.transform(X_test))


def scores_to_lables(pred_scores, outlier_perc=0.1):
    """
    turn raw outlier decision_scores to binary labels (0 or 1)

    :param pred_scores: raw outlier decision_scores
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

    # turn prediction decision_scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and y_pred
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """
    Function to turn decision_scores into binary labels by assign 1 to top n_train decision_scores
    Example y: [0,1,1,0,0,0]
            y_pred: [0.1, 0.5, 0.3, 0.2, 0.7]
            return [0, 1, 0, 0, 1]
    :param y: ground truth
    :param y_pred: number of outliers
    :param n: number of outliers, if not defined, infer using ground truth
    :return: binary labels 0: normal points and 1: outliers
    """
    # enforce formats of imputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    if y.shape != y_pred.shape:
        ValueError('ground truth y and prediction y_pred shape does not match')

    # calculate the percentage of outliers
    if n is not None:
        outlier_perc = n / y.shape[0]
    else:
        outlier_perc = np.count_nonzero(y) / y.shape[0]

    threshold = scoreatpercentile(y_pred, 100 * (1 - outlier_perc))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred


def get_top_n(value_list, n, top=True):
    """
    return the index of top n_train elements in the list
    :param value_list: a list
    :param n:
    :param top:
    :return:
    """
    value_list = np.asarray(value_list)
    length = value_list.shape[0]

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if top:
        return np.where(np.greater_equal(value_list, threshold))
    else:
        return np.where(np.less(value_list, threshold))


def argmaxp(a, p):
    """
    Utlity function to return the index of top p values in a
    :param a: list variable
    :param p: number of elements to select
    :return: index of top p elements in a
    """

    a = np.asarray(a).ravel()
    length = a.shape[0]
    pth = np.argpartition(a, length - p)
    return pth[length - p:]
