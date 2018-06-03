# -*- coding: utf-8 -*-
# %%

from __future__ import division
from __future__ import print_function

from sklearn.ensemble import BaggingClassifier

from sklearn.utils.estimator_checks import check_estimator
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.utils import check_array
from sklearn.utils.random import sample_without_replacement
from .base import BaseDetector

MAX_INT = np.iinfo(np.int32).max


def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """
    Draw randomly sampled indices.

    See sklearn/ensemble/bagging.py
    """
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def _generate_bagging_indices(random_state, bootstrap_features, n_features,
                              min_features, max_features):
    """
    Randomly draw feature indices.

    Modified from sklearn/ensemble/bagging.py
    """
    # Get valid random state
    random_state = check_random_state(random_state)

    # decide number of features to draw
    random_n_features = random_state.randint(min_features, max_features)

    # Draw indices
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, random_n_features)

    return feature_indices


# TODO: place holder only
class FeatureBagging(BaseDetector):
    """
    place holder only

    """

    def __init__(self, base_estimator, n_estimators=10, contamination=0.1,
                 min_features=0.5, max_features=1,
                 bootstrap_features=False, random_state=None):
        super(FeatureBagging, self).__init__(contamination=contamination)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.min_features = min_features
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)

        X = check_array(X)
        self.n_features_ = X.shape[1]

        # TODO add a check for min_features, e.g. d<=3 & max_features as well
        # at least 0.5 of total
        self.min_features_ = int(self.n_features_ * self.min_features)
        self.max_features_ = int(self.n_features_ * self.max_features)

        self.estimators_ = []
        self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        for i in range(self.n_estimators):
            random_state = np.random.RandomState(seeds[i])

            features = _generate_bagging_indices(random_state,
                                                 self.bootstrap_features,
                                                 self.n_features_,
                                                 self.min_features_,
                                                 self.max_features_)

            self.estimators_features_.append(features)

    def decision_function(self, X):
        pass
