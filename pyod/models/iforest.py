# -*- coding: utf-8 -*-
"""IsolationForest Outlier Detector. Implemented on scikit-learn library.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from joblib import Parallel
from joblib.parallel import delayed
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector
# noinspection PyProtectedMember
from ..utils.utility import _get_sklearn_version
from ..utils.utility import invert_order


# TODO: behavior of Isolation Forest will change in sklearn 0.22. See below.
# in 0.22, scikit learn will start adjust decision_function values by
# offset to make the values below zero as outliers. In other words, it is
# an absolute shift, which SHOULD NOT affect the result of PyOD at all as
# the order is still preserved.

# Behaviour of the decision_function which can be either ‘old’ or ‘new’.
# Passing behaviour='new' makes the decision_function change to match other
# anomaly detection algorithm API which will be the default behaviour in the
# future. As explained in details in the offset_ attribute documentation,
# the decision_function becomes dependent on the contamination parameter,
# in such a way that 0 becomes its natural threshold to detect outliers.

# offset_ : float
# Offset used to define the decision function from the raw scores.
# We have the relation: decision_function = score_samples - offset_.
# Assuming behaviour == ‘new’, offset_ is defined as follows.
# When the contamination parameter is set to “auto”,
# the offset is equal to -0.5 as the scores of inliers are close to 0 and the
# scores of outliers are close to -1. When a contamination parameter different
# than “auto” is provided, the offset is defined in such a way we obtain the
# expected number of outliers (samples with decision function < 0) in training.
# Assuming the behaviour parameter is set to ‘old’,
# we always have offset_ = -0.5, making the decision function independent from
# the contamination parameter.

# check https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html for more information


class IForest(BaseDetector):
    """Wrapper of scikit-learn Isolation Forest with more functionalities.

    The IsolationForest 'isolates' observations by randomly selecting a
    feature and then randomly selecting a split value between the maximum and
    minimum values of the selected feature.
    See :cite:`liu2008isolation,liu2012isolation` for details.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    behaviour : str, default='old'
        Behaviour of the ``decision_function`` which can be either 'old' or
        'new'. Passing ``behaviour='new'`` makes the ``decision_function``
        change to match other anomaly detection algorithm API which will be
        the default behaviour in the future. As explained in details in the
        ``offset_`` attribute documentation, the ``decision_function`` becomes
        dependent on the contamination parameter, in such a way that 0 becomes
        its natural threshold to detect outliers.

        .. versionadded:: 0.7.0
           ``behaviour`` is added in 0.7.0 for back-compatibility purpose.

        .. deprecated:: 0.20
           ``behaviour='old'`` is deprecated in sklearn 0.20 and will not be
           possible in 0.22.

        .. deprecated:: 0.22
           ``behaviour`` parameter will be deprecated in sklearn 0.22 and
           removed in 0.24.

        .. warning::
            Only applicable for sklearn 0.20 above.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 behaviour='old',
                 random_state=None,
                 verbose=0):
        super(IForest, self).__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.random_state = random_state
        self.verbose = verbose

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
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # In sklearn 0.20+ new behaviour is added (arg behaviour={'new','old'})
        # to IsolationForest that shifts the location of the anomaly scores
        # noinspection PyProtectedMember
        sklearn_version = _get_sklearn_version()
        if sklearn_version == 21:
            self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                             max_samples=self.max_samples,
                                             contamination=self.contamination,
                                             max_features=self.max_features,
                                             bootstrap=self.bootstrap,
                                             n_jobs=self.n_jobs,
                                             behaviour=self.behaviour,
                                             random_state=self.random_state,
                                             verbose=self.verbose)

        # Do not pass behaviour argument when sklearn version is < 0.20 or >0.21
        else:  # pragma: no cover
            self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                             max_samples=self.max_samples,
                                             contamination=self.contamination,
                                             max_features=self.max_features,
                                             bootstrap=self.bootstrap,
                                             n_jobs=self.n_jobs,
                                             random_state=self.random_state,
                                             verbose=self.verbose)

        self.detector_.fit(X=X, y=None, sample_weight=None)

        # invert decision_scores_. Outliers comes with higher outlier scores.
        self.decision_scores_ = invert_order(
            self.detector_.decision_function(X))
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        # invert outlier scores. Outliers comes with higher outlier scores
        return invert_order(self.detector_.decision_function(X))

    @property
    def estimators_(self):
        """The collection of fitted sub-estimators.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.estimators_

    @property
    def estimators_samples_(self):
        """The subset of drawn samples (i.e., the in-bag samples) for
        each base estimator.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.estimators_samples_

    @property
    def max_samples_(self):
        """The actual number of samples.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.detector_.max_samples_

    @property
    def feature_importances_(self):
        """The impurity-based feature importance. The higher, the more
        important the feature. The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        .. warning::
        impurity-based feature importance can be misleading for
        high cardinality features (many unique values). See
        https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
        as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)
        all_importances = Parallel(
            n_jobs=self.n_jobs)(
            delayed(getattr)(tree, "feature_importances_")
            for tree in self.detector_.estimators_
            if tree.tree_.node_count > 1
        )

        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)
