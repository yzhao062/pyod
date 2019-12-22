# -*- coding: utf-8 -*-
"""A collection of model combination functionalities.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

from combo.models.score_comb import aom as combo_aom
from combo.models.score_comb import moa as combo_moa
from combo.models.score_comb import average as combo_average
from combo.models.score_comb import maximization as combo_maximization
from combo.models.score_comb import majority_vote as combo_majority_vote
from combo.models.score_comb import median as combo_median


def aom(scores, n_buckets=5, method='static', bootstrap_estimators=False,
        random_state=None):
    """Average of Maximum - An ensemble method for combining multiple
    estimators. See :cite:`aggarwal2015theoretical` for details.

    First dividing estimators into subgroups, take the maximum score as the
    subgroup score. Finally, take the average of all subgroup outlier scores.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        The score matrix outputted from various estimators

    n_buckets : int, optional (default=5)
        The number of subgroups to build

    method : str, optional (default='static')
        {'static', 'dynamic'}, if 'dynamic', build subgroups
        randomly with dynamic bucket size.

    bootstrap_estimators : bool, optional (default=False)
        Whether estimators are drawn with replacement.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.

    Returns
    -------
    combined_scores : Numpy array of shape (n_samples,)
        The combined outlier scores.

    """

    return combo_aom(scores, n_buckets, method, bootstrap_estimators,
                     random_state)


def moa(scores, n_buckets=5, method='static', bootstrap_estimators=False,
        random_state=None):
    """Maximization of Average - An ensemble method for combining multiple
    estimators. See :cite:`aggarwal2015theoretical` for details.

    First dividing estimators into subgroups, take the average score as the
    subgroup score. Finally, take the maximization of all subgroup outlier
    scores.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        The score matrix outputted from various estimators

    n_buckets : int, optional (default=5)
        The number of subgroups to build

    method : str, optional (default='static')
        {'static', 'dynamic'}, if 'dynamic', build subgroups
        randomly with dynamic bucket size.

    bootstrap_estimators : bool, optional (default=False)
        Whether estimators are drawn with replacement.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.

    Returns
    -------
    combined_scores : Numpy array of shape (n_samples,)
        The combined outlier scores.

    """
    return combo_moa(scores, n_buckets, method, bootstrap_estimators,
                     random_state)


def average(scores, estimator_weights=None):
    """Combination method to merge the outlier scores from multiple estimators
    by taking the average.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    estimator_weights : list of shape (1, n_estimators)
        If specified, using weighted average

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined outlier scores.

    """
    return combo_average(scores, estimator_weights)


def maximization(scores):
    """Combination method to merge the outlier scores from multiple estimators
    by taking the maximum.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined outlier scores.

    """
    return combo_maximization(scores)


def majority_vote(scores, weights=None):
    """Combination method to merge the scores from multiple estimators
    by majority vote.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.


    weights : numpy array of shape (1, n_estimators)
        If specified, using weighted majority weight.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.

    """
    return combo_majority_vote(scores, n_classes=2, weights=weights)


def median(scores):
    """Combination method to merge the scores from multiple estimators
    by taking the median.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.

    """
    return combo_median(scores)
