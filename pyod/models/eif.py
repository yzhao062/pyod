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
# dependency. One deliberate departure from the reference implementation: the
# coordinates zeroed out of the normal vector at a node are taken from the
# coordinates that are constant within that node first, so the active
# coordinates always include one that varies; a node whose coordinates are all
# constant becomes a leaf at once, and a hyperplane that still sends every
# point of a node to the same side is redrawn up to ``MAX_SPLIT_ATTEMPTS``
# times before the node becomes a leaf, so constant features and duplicate
# rows do not inflate path lengths.

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
MAX_SPLIT_ATTEMPTS = 10


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

    n_features_ : int
        The number of features seen during the fit.

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
        self.n_features_ = n_features

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
            if self.max_samples < 1:
                raise ValueError(
                    "max_samples must be at least 1, got %r" % self.max_samples
                )
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
            max_samples = max(1, int(self.max_samples * n_samples))
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

        if (
            not isinstance(self.n_estimators, numbers.Integral)
            or self.n_estimators < 1
        ):
            raise ValueError(
                "n_estimators must be a positive integer, got %r"
                % self.n_estimators
            )

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
        varying = maxs > mins
        if not varying.any():
            return _ExNode(n)
        varying_idx = np.flatnonzero(varying)
        const_idx = np.flatnonzero(~varying)
        n_const = const_idx.shape[0]
        n_zero = n_features - extension_level - 1

        for _ in range(MAX_SPLIT_ATTEMPTS):
            # Random intercept point inside the bounding box of the node.
            intercept = rng.uniform(mins, maxs)
            # Random normal vector; zero out coordinates not used at this
            # extension level (extension_level == n_features - 1 keeps all),
            # taking the coordinates that are constant within the node first
            # so that the active coordinates always include one that varies.
            normal = rng.normal(0.0, 1.0, size=n_features)
            if n_zero > 0:
                if n_const > n_zero:
                    zero_idx = rng.choice(const_idx, n_zero, replace=False)
                elif n_const < n_zero:
                    extra_idx = rng.choice(varying_idx, n_zero - n_const, replace=False)
                    zero_idx = np.concatenate((const_idx, extra_idx))
                else:
                    zero_idx = const_idx
                normal[zero_idx] = 0.0

            left_mask = (X - intercept) @ normal <= 0
            n_left = np.count_nonzero(left_mask)
            if 0 < n_left < n:
                return _InNode(
                    self._build_tree(
                        X[left_mask],
                        current_height + 1,
                        extension_level,
                        n_features,
                        rng,
                    ),
                    self._build_tree(
                        X[~left_mask],
                        current_height + 1,
                        extension_level,
                        n_features,
                        rng,
                    ),
                    normal,
                    intercept,
                )

        return _ExNode(n)

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
        check_is_fitted(self, ["_trees", "max_samples_", "n_features_"])
        X = check_array(X, accept_sparse=False)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "X has %d features, but EIF was fitted with %d features"
                % (X.shape[1], self.n_features_)
            )

        c_norm = _c_factor(self.max_samples_)
        if c_norm <= 0:
            # A single-sample subsample grows trees without a split, which
            # carry no anomaly information: give every sample the neutral
            # score of an unsplittable root.
            return np.full(X.shape[0], 0.5)
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            path_lengths = [self._path_length(X[i], tree, 0) for tree in self._trees]
            scores[i] = 2 ** (-np.mean(path_lengths) / c_norm)
        return scores
