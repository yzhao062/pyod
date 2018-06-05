# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_array
from sklearn.utils import column_or_1d
# noinspection PyProtectedMember
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.testing import assert_equal
from ..utils.utility import check_parameter


def aom(scores, n_buckets, method='static', replace=False, random_state=None):
    """
    Average of Maximum - An ensemble method for combining multiple detectors

    First dividing detectors into subgroups, take the maximum score as the
    subgroup score.

    Finally, take the average of all subgroup decision_scores_.

    :param scores: a score matrix from different detectors
    :type scores:

    :param n_buckets: number of subgroups
    :type n_buckets:

    :param method: static or dynamic, default: static
    :type method

    :param replace:
    :type replace:

    :param random_state:
    :type random_state:

    :return:
    :rtype:

    .. [1] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and
           algorithms for outlier ensembles. ACM SIGKDD Explorations
           Newsletter, 17(1), pp.24-47.
    """

    # TODO: add one more parameter for max number of estimators
    # use random_state instead
    # for now it is fixed at n_estimators/2
    scores = check_array(scores)
    n_estimators = scores.shape[1]
    check_parameter(n_buckets, 2, n_estimators, param_name='n_buckets')

    scores_aom = np.zeros([scores.shape[0], n_buckets])

    if method == 'static':

        n_estimators_per_bucket = int(n_estimators / n_buckets)
        if n_estimators % n_buckets != 0:
            Warning('n_estimators / n_buckets has a remainder')

        if not replace:
            # shuffle the estimator order
            shuffled_list = shuffle(list(range(0, n_estimators, 1)),
                                    random_state=random_state)

            head = 0
            for i in range(0, n_estimators, n_estimators_per_bucket):
                tail = i + n_estimators_per_bucket
                batch_ind = int(i / n_estimators_per_bucket)

                scores_aom[:, batch_ind] = np.max(
                    scores[:, shuffled_list[head:tail]], axis=1)

                # increment indexes
                head = head + n_estimators_per_bucket
                # noinspection PyUnusedLocal
                tail = tail + n_estimators_per_bucket
        else:
            for i in range(n_buckets):
                ind = sample_without_replacement(n_estimators,
                                                 n_estimators_per_bucket,
                                                 random_state=random_state)
                scores_aom[:, i] = np.max(scores[:, ind], axis=1)


    elif method == 'dynamic':  # random bucket size
        for i in range(n_buckets):
            # the number of estimators in a bucket should be 2 - n/2
            max_estimator_per_bucket = RandomState(seed=random_state).randint(
                2, int(n_estimators / 2))
            ind = sample_without_replacement(n_estimators,
                                             max_estimator_per_bucket,
                                             random_state=random_state)
            scores_aom[:, i] = np.max(scores[:, ind], axis=1)

    else:
        raise NotImplementedError(
            '{method} is not implemented'.format(method=method))

    return np.mean(scores_aom, axis=1)


def moa(scores, n_buckets, method='static', replace=False, random_state=None):
    """
    Maximization of Average - An ensemble method for combining multiple
    detectors

    First dividing detectors into subgroups, take the average score as the
    subgroup score.

    Finally, take the maximization of all subgroup decision_scores_.

    .. [1] Aggarwal, C.C. and Sathe, S., 2015. Theoretical
           foundations and algorithms for outlier ensembles.
           ACM SIGKDD Explorations Newsletter, 17(1), pp.24-47.

    :param scores: a score matrix from different detectors
    :type scores:

    :param n_buckets: number of subgroups
    :type n_buckets:

    :param method: static or dynamic, default: static
    :type method

    :param replace:
    :type replace:

    :param random_state:
    :type random_state:

    :return:
    :rtype:
    """

    # TODO: add one more parameter for max number of estimators
    #       for now it is fixed to n_estimators/2
    scores = check_array(scores)
    n_estimators = scores.shape[1]
    check_parameter(n_buckets, 2, n_estimators, param_name='n_buckets')

    scores_aom = np.zeros([scores.shape[0], n_buckets])

    if method == 'static':

        n_estimators_per_bucket = int(n_estimators / n_buckets)
        if n_estimators % n_buckets != 0:
            Warning('n_estimators / n_buckets has a remainder')

        if not replace:
            # shuffle the estimator order
            shuffled_list = shuffle(list(range(0, n_estimators, 1)),
                                    random_state=random_state)

            head = 0
            for i in range(0, n_estimators, n_estimators_per_bucket):
                tail = i + n_estimators_per_bucket
                batch_ind = int(i / n_estimators_per_bucket)

                scores_aom[:, batch_ind] = np.mean(
                    scores[:, shuffled_list[head:tail]], axis=1)

                # increment index
                head = head + n_estimators_per_bucket
                tail = tail + n_estimators_per_bucket
        else:
            for i in range(n_buckets):
                ind = sample_without_replacement(n_estimators,
                                                 n_estimators_per_bucket,
                                                 random_state=random_state)
                scores_aom[:, i] = np.mean(scores[:, ind], axis=1)


    elif method == 'dynamic':  # random bucket size
        for i in range(n_buckets):
            # the number of estimators in a bucket should be 2 - n/2
            max_estimator_per_bucket = RandomState(seed=random_state).randint(
                2, int(n_estimators / 2))
            ind = sample_without_replacement(n_estimators,
                                             max_estimator_per_bucket,
                                             random_state=random_state)
            scores_aom[:, i] = np.mean(scores[:, ind], axis=1)

    else:
        raise NotImplementedError(
            '{method} is not implemented'.format(method=method))

    return np.max(scores_aom, axis=1)


def average(scores, detector_weight=None):
    """
    Combine the outlier scores from multiple detectors by averaging

    :param scores: score matrix from multiple detectors on the same samples
    :type scores: numpy array of shape (n_samples, n_detectors)

    :param detector_weight: if specified, using weighted average
    :type detector_weight: list of shape (1, n_detectors)

    :return: the combined outlier scores
    :rtype: numpy array of shape (n_samples, )
    """
    scores = check_array(scores)

    if detector_weight is not None:
        detector_weight = column_or_1d(detector_weight).reshape(1, -1)
        assert_equal(scores.shape[1], detector_weight.shape[1])

        # (d1*w1 + d2*w2 + ...+ dn*wn)/(w1+w2+...+wn)
        # generated weighted scores
        scores = np.sum(np.multiply(scores, detector_weight), axis=1) / np.sum(
            detector_weight)
        return scores.ravel()

    else:
        return np.mean(scores, axis=1).ravel()


def maximization(scores):
    """
    Combine the outlier scores from multiple detectors by taking the maximum

    :param scores: score matrix from multiple detectors on the same samples
    :type scores: numpy array of shape (n_samples, n_detectors)

    :return: the combined outlier scores
    :rtype: numpy array of shape (n_samples, )
    """
    scores = check_array(scores)
    return np.max(scores, axis=1).ravel()
