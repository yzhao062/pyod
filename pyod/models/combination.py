import numpy as np
from sklearn.utils import check_array


def aom(scores, n_buckets, n_estimators, standard=True):
    '''
    Average of Maximum - An ensemble method for outlier detection

    Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms
    for outlier ensembles. ACM SIGKDD Explorations Newsletter, 17(1), pp.24-47.

    :param scores:
    :param n_buckets:
    :param n_estimators:
    :param standard:
    :return:
    '''
    scores = check_array(scores)
    if scores.shape[1] != n_estimators:
        raise ValueError('score matrix should be n_samples by n_estimaters')

    scores_aom = np.zeros([scores.shape[0], n_buckets])

    # standardized scores
    # TODO: implement standardization check here to make sure the score
    # if actually normalized before combination

    # TODO: replace random sampling methods
    # TODO: check number of estimatoes and scores match

    n_estimators_per_bucket = int(n_estimators / n_buckets)
    if n_estimators % n_buckets != 0:
        Warning('n_estimators / n_buckets leads to a remainder')

    # shuffle the estimator order
    estimators_list = list(range(0, n_estimators, 1))
    np.random.shuffle(estimators_list)

    head = 0
    for i in range(0, n_estimators, n_estimators_per_bucket):
        tail = i + n_estimators_per_bucket
        batch_ind = int(i / n_estimators_per_bucket)

        scores_aom[:, batch_ind] = np.max(
            scores[:, estimators_list[head:tail]], axis=1)

        head = head + n_estimators_per_bucket
        tail = tail + n_estimators_per_bucket

    return np.mean(scores_aom, axis=1)


def moa(scores, n_buckets, n_estimators):
    '''
    Maximum of Average - An ensemble method for outlier detection

    Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms
    for outlier ensembles. ACM SIGKDD Explorations Newsletter, 17(1), pp.24-47.

    :param scores:
    :param n_buckets:
    :param n_estimators:
    :param standard:
    :return:
    '''
    scores = check_array(scores)
    if scores.shape[1] != n_estimators:
        raise ValueError('score matrix should be n_samples by n_estimaters')

    scores_moa = np.zeros([scores.shape[0], n_buckets])

    # standardized scores
    # TODO: implement standardization check here to make sure the score
    # if actually normalized before combination

    n_estimators_per_bucket = int(n_estimators / n_buckets)
    if n_estimators % n_buckets != 0:
        Warning('n_estimators / n_buckets leads to a remainder')

    # shuffle the estimator order
    estimators_list = list(range(0, n_estimators, 1))
    np.random.shuffle(estimators_list)

    head = 0
    for i in range(0, n_estimators, n_estimators_per_bucket):
        tail = i + n_estimators_per_bucket
        batch_ind = int(i / n_estimators_per_bucket)

        scores_moa[:, batch_ind] = np.mean(
            scores[:, estimators_list[head:tail]], axis=1)

        head = head + n_estimators_per_bucket
        tail = tail + n_estimators_per_bucket

    return np.max(scores_moa, axis=1)
