import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def get_precn(y, y_pred):
    '''
    Utlity function to calculate precision@n
    :param y: ground truth
    :param y_pred: number of outliers
    :return: score
    '''
    # calculate the percentage of outliers
    out_perc = np.count_nonzero(y) / len(y)

    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return precision_score(y, y_pred)


def get_top_n(roc_list, n, top=True):
    '''
    for use of Accurate Selection only
    :param roc_list: a li
    :param n:
    :param top:
    :return:
    '''
    roc_list = np.asarray(roc_list)
    length = roc_list.shape[0]

    roc_sorted = np.partition(roc_list, length - n)
    threshold = roc_sorted[int(length - n)]

    if top:
        return np.where(np.greater_equal(roc_list, threshold))
    else:
        return np.where(np.less(roc_list, threshold))