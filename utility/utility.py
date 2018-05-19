import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler


def standardizer(X_train, X_test):
    '''
    normalization function wrapper
    :param X_train:
    :param X_test:
    :return: X_train and X_test after the Z-score normalization
    '''
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def scores_to_lables(pred_scores, outlier_perc=0.05):
    '''
    turn raw outlier scores to binary labels (0 or 1)
    :param pred_scores: raw outlier scores
    :param outlier_perc: percentage of outliers
    :return: binary labels (1 stands for outlier)
    '''
    threshold = scoreatpercentile(pred_scores, 100 * (1 - outlier_perc))
    pred_labels = (pred_scores > threshold).astype('int')
    return pred_labels


def precision_n_scores(y, y_pred):
    '''
    Utlity function to calculate precision@n
    :param y: ground truth
    :param y_pred: number of outliers
    :return: score
    '''
    # calculate the percentage of outliers
    outlier_perc = np.count_nonzero(y) / len(y)

    threshold = scoreatpercentile(y_pred, 100 * (1 - outlier_perc))
    y_pred = (y_pred > threshold).astype('int')
    return precision_score(y, y_pred)


def get_top_n(value_list, n, top=True):
    '''
    return the index of top n elements in the list
    :param value_list: a list
    :param n:
    :param top:
    :return:
    '''
    value_list = np.asarray(value_list)
    length = value_list.shape[0]

    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if top:
        return np.where(np.greater_equal(value_list, threshold))
    else:
        return np.where(np.less(value_list, threshold))


def get_label_n(y, y_pred):
    '''
    function to turn scores into binary labels by assign 1 to top n scores
    Example y: [0,1,1,0,0,0]
            y_pred: [0.1, 0.5, 0.3, 0.2, 0.7]
            return [0, 1, 0, 0, 1]
    :param y:
    :param y_pred:
    :return:
    '''
    # calculate the percentage of outlier scores
    out_perc = np.count_nonzero(y) / len(y)
    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return y_pred


def argmaxp(a, p):
    '''
    Utlity function to return the index of top p values in a
    :param a: list variable
    :param p: number of elements to select
    :return: index of top p elements in a
    '''

    a = np.asarray(a).ravel()
    length = a.shape[0]
    pth = np.argpartition(a, length - p)
    return pth[length - p:]
