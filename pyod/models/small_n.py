"""Outlier detection for small reference sets using Mahalanobis distance
with Ledoit-Wolf shrinkage covariance.

Standard covariance-based detectors, including PyOD's own MCD, assume
enough samples to estimate covariance robustly, usually well more than
the number of features. Ledoit-Wolf shrinkage instead adapts its
regularisation to the ratio of sample count to dimensionality, staying
stable down to n=3, by pulling the covariance estimate toward a scaled
identity matrix rather than requiring a full-rank sample covariance.
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector

_RIDGE = 1e-6


class SmallN(BaseDetector):
    """Mahalanobis distance outlier detector for small reference sets."""

    def __init__(self, contamination=0.1):
        super(SmallN, self).__init__(contamination=contamination)

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        self.n_features_ = X.shape[1]
        self.centroid_ = np.mean(X, axis=0)
        n = X.shape[0]
        if n >= 2:
            lw = LedoitWolf()
            lw.fit(X)
            cov = lw.covariance_
        else:
            cov = np.eye(self.n_features_)

        # A degenerate (all-zero) covariance happens when every reference
        # row is identical, its pseudo-inverse is also zero, which would
        # silently score every future point as 0 regardless of how far
        # it is from the centroid. A small ridge keeps the metric usable.
        cov = cov + _RIDGE * np.eye(self.n_features_)
        self.inv_cov_ = np.linalg.pinv(cov)

        self.decision_scores_ = self._mahalanobis_batch(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['centroid_', 'inv_cov_', 'n_features_'])
        X = check_array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                'X has {} features, but SmallN was fitted with {} '
                'features.'.format(X.shape[1], self.n_features_))
        return self._mahalanobis_batch(X)

    def _mahalanobis_batch(self, X):
        diffs = X - self.centroid_
        mahalanobis_sq = np.einsum(
            'ij,jk,ik->i', diffs, self.inv_cov_, diffs)
        return np.sqrt(np.maximum(mahalanobis_sq, 0))

    def compute_rejection_stats(self, T=32, delta=0.1, c_fp=1, c_fn=1,
                                c_r=-1, verbose=False):
        """Overrides BaseDetector.compute_rejection_stats to fail with a
        clear message instead of an opaque scipy error.

        The inherited implementation computes
        int(n * contamination) - 1, which goes negative for small n and
        typical contamination values (e.g. n=3, contamination=0.1 gives
        -1), and then hands that negative value to a binomial CDF root
        solver with no valid bracket, raising an unrelated-looking
        ValueError from inside scipy. This detector is specifically for
        small n, so that combination is expected to come up, and it
        deserves a message that actually explains what to change.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        n = len(self.decision_scores_)
        if int(n * self.contamination) - 1 < 0:
            raise ValueError(
                'compute_rejection_stats is not supported for this '
                'combination of n={} and contamination={}, since '
                'int(n * contamination) - 1 is negative. Use a larger '
                'reference set or a higher contamination value if '
                'rejection statistics are needed.'.format(
                    n, self.contamination))
        return super(SmallN, self).compute_rejection_stats(
            T=T, delta=delta, c_fp=c_fp, c_fn=c_fn, c_r=c_r,
            verbose=verbose)
