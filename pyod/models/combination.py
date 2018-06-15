# -*- coding: utf-8 -*-
"""A collection of model combination functionalities.
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

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


def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False,
        random_state=None):
    """Average of Maximum - An ensemble method for combining multiple
    estimators. See :cite:`aggarwal2015theoretical` for details.

    First dividing estimators into subgroups, take the maximum score as the
    subgroup score. Finally, take the average of all subgroup outlier scores.

    :param scores: The score matrix outputted from various estimators
    :type scores: numpy array of shape (n_samples, n_estimators)

    :param n_buckets: The number of subgroups to build
    :type n_buckets: int, optional (default=5)

    :param method: {'static', 'dynamic'}, if 'dynamic', build subgroups
        randomly with dynamic bucket size.
    :type method: str, optional (default='static')

    :param bootstrap_estimators: Whether estimators are drawn with replacement.
    :type bootstrap_estimators: bool, optional (default=False)

    :param random_state: If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.
    :type random_state: int, RandomState instance or None,
        optional (default=None)

    :return: The combined outlier scores.
    :rtype: Numpy array of shape (n_samples,)
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
            raise ValueError('n_estimators / n_buckets has a remainder. Not '
                             'allowed in static mode.')

        if not bootstrap_estimators:
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


def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False,
        random_state=None):
    """Maximization of Average - An ensemble method for combining multiple
    estimators. See :cite:`aggarwal2015theoretical` for details.

    First dividing estimators into subgroups, take the average score as the
    subgroup score. Finally, take the maximization of all subgroup outlier
    scores.

    :param scores: The score matrix outputted from various estimators
    :type scores: numpy array of shape (n_samples, n_estimators)

    :param n_buckets: The number of subgroups to build
    :type n_buckets: int, optional (default=5)

    :param method: {'static', 'dynamic'}, if 'dynamic', build subgroups
        randomly with dynamic bucket size.
    :type method: str, optional (default='static')

    :param bootstrap_estimators: Whether estimators are drawn with replacement.
    :type bootstrap_estimators: bool, optional (default=False)

    :param random_state: If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.
    :type random_state: int, RandomState instance or None,
        optional (default=None)

    :return: The combined outlier scores.
    :rtype: Numpy array of shape (n_samples,)
    """

    # TODO: add one more parameter for max number of estimators
    #       for now it is fixed to n_estimators/2
    scores = check_array(scores)
    n_estimators = scores.shape[1]
    check_parameter(n_buckets, 2, n_estimators, param_name='n_buckets')

    scores_moa = np.zeros([scores.shape[0], n_buckets])

    if method == 'static':

        n_estimators_per_bucket = int(n_estimators / n_buckets)
        if n_estimators % n_buckets != 0:
            raise ValueError('n_estimators / n_buckets has a remainder. Not '
                             'allowed in static mode.')

        if not bootstrap_estimators:
            # shuffle the estimator order
            shuffled_list = shuffle(list(range(0, n_estimators, 1)),
                                    random_state=random_state)

            head = 0
            for i in range(0, n_estimators, n_estimators_per_bucket):
                tail = i + n_estimators_per_bucket
                batch_ind = int(i / n_estimators_per_bucket)

                scores_moa[:, batch_ind] = np.mean(
                    scores[:, shuffled_list[head:tail]], axis=1)

                # increment index
                head = head + n_estimators_per_bucket
                # noinspection PyUnusedLocal
                tail = tail + n_estimators_per_bucket
        else:
            for i in range(n_buckets):
                ind = sample_without_replacement(n_estimators,
                                                 n_estimators_per_bucket,
                                                 random_state=random_state)
                scores_moa[:, i] = np.mean(scores[:, ind], axis=1)

    elif method == 'dynamic':  # random bucket size
        for i in range(n_buckets):
            # the number of estimators in a bucket should be 2 - n/2
            max_estimator_per_bucket = RandomState(seed=random_state).randint(
                2, int(n_estimators / 2))
            ind = sample_without_replacement(n_estimators,
                                             max_estimator_per_bucket,
                                             random_state=random_state)
            scores_moa[:, i] = np.mean(scores[:, ind], axis=1)

    else:
        raise NotImplementedError(
            '{method} is not implemented'.format(method=method))

    return np.max(scores_moa, axis=1)


def average(scores, estimator_weight=None):
    """
    Combine the outlier scores from multiple estimators by averaging

    :param scores: score matrix from multiple estimators on the same samples
    :type scores: numpy array of shape (n_samples, n_estimators)

    :param estimator_weight: if specified, using weighted average
    :type estimator_weight: list of shape (1, n_estimators)

    :return: the combined outlier scores
    :rtype: numpy array of shape (n_samples, )
    """
    scores = check_array(scores)

    if estimator_weight is not None:
        estimator_weight = column_or_1d(estimator_weight).reshape(1, -1)
        assert_equal(scores.shape[1], estimator_weight.shape[1])

        # (d1*w1 + d2*w2 + ...+ dn*wn)/(w1+w2+...+wn)
        # generated weighted scores
        scores = np.sum(np.multiply(scores, estimator_weight),
                        axis=1) / np.sum(
            estimator_weight)
        return scores.ravel()

    else:
        return np.mean(scores, axis=1).ravel()


def maximization(scores):
    """
    Combine the outlier scores from multiple estimators by taking the maximum

    :param scores: score matrix from multiple estimators on the same samples
    :type scores: numpy array of shape (n_samples, n_estimators)

    :return: the combined outlier scores
    :rtype: numpy array of shape (n_samples, )
    """
    scores = check_array(scores)
    return np.max(scores, axis=1).ravel()
