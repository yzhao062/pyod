"""Outlier detection for small reference sets using Mahalanobis distance
with Ledoit-Wolf shrinkage covariance.

Standard covariance-based detectors assume enough samples to
estimate covariance robustly, usually well more than the
number of features. Ledoit-Wolf shrinkage instead adapts its
regularisation to the ratio of sample count to dimensionality, staying
stable down to n=3, by pulling the covariance estimate toward a scaled
identity matrix rather than requiring a full-rank sample covariance.
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector


class SmallN(BaseDetector):
    """Mahalanobis distance outlier detector for small reference sets."""

    def __init__(self, contamination=0.1):
        super(SmallN, self).__init__(contamination=contamination)

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        self.centroid_ = np.mean(X, axis=0)
        n = X.shape[0]
        if n >= 2:
            lw = LedoitWolf()
            lw.fit(X)
            self.inv_cov_ = np.linalg.pinv(lw.covariance_)
        else:
            self.inv_cov_ = np.eye(X.shape[1])

        self.decision_scores_ = self._mahalanobis_batch(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['centroid_', 'inv_cov_'])
        X = check_array(X)
        return self._mahalanobis_batch(X)

    def _mahalanobis_batch(self, X):
        diffs = X - self.centroid_
        mahalanobis_sq = np.einsum(
            'ij,jk,ik->i', diffs, self.inv_cov_, diffs)
        return np.sqrt(np.maximum(mahalanobis_sq, 0))
