# -*- coding: utf-8 -*-
# %%
from sklearn.ensemble import BaggingClassifier

from sklearn.utils.estimator_checks import check_estimator
from pyod.models.knn import KNN
from pyod.utils.load_data import generate_data
from sklearn.base import BaseEstimator

from .base import BaseDetector


# TODO: place holder only
class FeatureBagging(BaseDetector):
    def __init__(self, base_estimator, n_estimators=10, contamination=0.1,
                 min_features=0.5):
        super().__init__(contamination=contamination)
        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.min_features_ = min_features

    def fit(self, X, y=None):
        pass

    def decision_function(self, X):
        pass
