# -*- coding: utf-8 -*-
"""Extended Isolation Forest (EIF) for outlier detection."""
# Author: Jayesh Suryavanshi <jayeshsuryavanshi808@gmail.com>
# License: BSD 2 clause
#
# Extended Isolation Forest replaces the axis-parallel splits of the standard
# Isolation Forest with random hyperplanes (a random slope ``n`` and a random
# intercept ``p``), which removes the axis-aligned bias in the anomaly score
# maps. The algorithm and the ``c_factor`` normalization follow Hariri,
# Carrasco Kind and Brunner, "Extended Isolation Forest", IEEE TKDE 2021, and
# the authors' reference implementation (https://github.com/sahandha/eif). The
# code below is a clean, numpy-only reimplementation that follows PyOD
# conventions and mirrors ``pyod/models/iforest.py``; it adds no new
# dependency.

from __future__ import division
from __future__ import print_function

import numbers
from warnings import warn

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

from .base import BaseDetector

__all__ = ["EIF"]

MAX_INT = np.iinfo(np.int32).max
EULER_GAMMA = 0.5772156649015329


def _c_factor(n):
    """Average path length of an unsuccessful search in a binary search tree
    of ``n`` points, used to normalize the anomaly score.

    ``c(n) = 2 H(n - 1) - 2 (n - 1) / n``, with ``H`` the harmonic number
    approximated by ``ln(n - 1) + gamma`` as in the Isolation Forest
    literature. Returns 0 for ``n <= 1``.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


class _ExNode:
    """External (leaf) node holding the number of training samples that
    reached it."""

    __slots__ = ["size"]

    def __init__(self, size):
        self.size = size


class _InNode:
    """Internal node storing the random hyperplane (normal vector ``n`` and
    intercept ``p``) and the two children."""

    __slots__ = ["left", "right", "normal", "intercept"]

    def __init__(self, left, right, normal, intercept):
        self.left = left
        self.right = right
        self.normal = normal
        self.intercept = intercept


class EIF(BaseDetector):
    """Extended Isolation Forest.

    The Extended Isolation Forest is a variant of the Isolation Forest that
    isolates observations by slicing the feature space with randomly oriented
    hyperplanes instead of axis-parallel cuts. At every node a random normal
    vector ``n`` and a random intercept point ``p`` (drawn uniformly from the
    bounding box of the samples in that node) define the split
    ``(x - p) . n <= 0``. The number of splits required to isolate a sample,
    averaged over the forest and normalized by the expected path length, is
    used as the outlier score; anomalies are isolated with fewer splits and
    therefore receive higher scores.

    See :cite:`hariri2021extended` for details.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators (trees) in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.

        - If int, then draw ``max_samples`` samples.
        - If float, then draw ``max_samples * X.shape[0]`` samples.
        - If "auto", then ``max_samples = min(256, n_samples)``.

    extension_level : int, optional (default=None)
        The extension level of the hyperplanes, between 0 and
        ``n_features - 1``. ``0`` recovers the standard axis-parallel Isolation
        Forest, while ``n_features - 1`` (the default when None) uses fully
        random hyperplanes.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used when fitting to define the threshold on
        the decision function.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.

    Attributes
    ----------
    max_samples_ : int
        The actual number of samples used to train each base estimator.

    extension_level_ : int
        The actual extension level used for the hyperplanes.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data. The higher, the more
        abnormal. Outliers tend to have higher scores.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers and 1 for
        outliers/anomalies. It is generated by applying ``threshold_`` on
        ``decision_scores_``.
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        extension_level=None,
        contamination=0.1,
        random_state=None,
    ):
        super(EIF, self).__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, accept_sparse=False)
        self._set_n_classes(y)

        n_samples, n_features = X.shape

        # Resolve max_samples following the Isolation Forest convention.
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(256, n_samples)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported. Valid choices are: "
                    '"auto", int or float' % self.max_samples
                )
        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the total number of "
                    "samples (%s). max_samples will be set to n_samples for "
                    "estimation." % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples
                )
            max_samples = int(self.max_samples * n_samples)
        max_samples = max(1, max_samples)
        self.max_samples_ = max_samples

        # Resolve and validate the extension level.
        if self.extension_level is None:
            extension_level = n_features - 1
        else:
            if not isinstance(self.extension_level, numbers.Integral):
                raise ValueError(
                    "extension_level must be an int, got %r" % self.extension_level
                )
            if not 0 <= self.extension_level <= n_features - 1:
                raise ValueError(
                    "extension_level must be between 0 and n_features - 1 "
                    "(%d), got %d" % (n_features - 1, self.extension_level)
                )
            extension_level = self.extension_level
        self.extension_level_ = extension_level

        # Height limit for each tree, as in the Isolation Forest.
        self._height_limit = int(np.ceil(np.log2(max(2, max_samples))))

        rng = check_random_state(self.random_state)
        seeds = rng.randint(MAX_INT, size=self.n_estimators)

        self._trees = []
        for i in range(self.n_estimators):
            tree_rng = check_random_state(seeds[i])
            if max_samples < n_samples:
                sample_idx = tree_rng.choice(n_samples, max_samples, replace=False)
                X_sub = X[sample_idx]
            else:
                X_sub = X
            self._trees.append(
                self._build_tree(X_sub, 0, extension_level, n_features, tree_rng)
            )

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def _build_tree(self, X, current_height, extension_level, n_features, rng):
        """Recursively grow a single extended isolation tree."""
        n = X.shape[0]
        if current_height >= self._height_limit or n <= 1:
            return _ExNode(n)

        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        # Random intercept point inside the bounding box of the node.
        intercept = rng.uniform(mins, maxs)
        # Random normal vector; zero out coordinates not used at this
        # extension level (extension_level == n_features - 1 keeps all).
        normal = rng.normal(0.0, 1.0, size=n_features)
        n_zero = n_features - extension_level - 1
        if n_zero > 0:
            zero_idx = rng.choice(n_features, n_zero, replace=False)
            normal[zero_idx] = 0.0

        projection = (X - intercept) @ normal
        left_mask = projection <= 0
        X_left = X[left_mask]
        X_right = X[~left_mask]

        return _InNode(
            self._build_tree(
                X_left, current_height + 1, extension_level, n_features, rng
            ),
            self._build_tree(
                X_right, current_height + 1, extension_level, n_features, rng
            ),
            normal,
            intercept,
        )

    @staticmethod
    def _path_length(x, node, current_height):
        """Path length of a single point ``x`` down one tree."""
        while isinstance(node, _InNode):
            if (x - node.intercept) @ node.normal <= 0:
                node = node.left
            else:
                node = node.right
            current_height += 1
        return current_height + _c_factor(node.size)

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is the normalized, forest-averaged
        isolation path length. For consistency, outliers are assigned larger
        anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ["_trees", "max_samples_"])
        X = check_array(X, accept_sparse=False)

        c_norm = _c_factor(self.max_samples_)
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            path_lengths = [self._path_length(X[i], tree, 0) for tree in self._trees]
            mean_path = np.mean(path_lengths)
            if c_norm > 0:
                scores[i] = 2 ** (-mean_path / c_norm)
            else:
                # A single-sample subsample cannot isolate anything.
                scores[i] = 2 ** (-mean_path)
        return scores
