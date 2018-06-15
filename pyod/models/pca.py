# -*- coding: utf-8 -*-
"""Principal Component Analysis (PCA) Outlier Detector
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_array

from .base import BaseDetector
from ..utils.utility import check_parameter


class PCA(BaseDetector):
    # noinspection PyPep8
    """
    Principal component analysis (PCA) can be used in detecting outliers. PCA
    is a linear dimensionality reduction using Singular Value Decomposition
    of the data to project it to a lower dimensional space.

    In this procedure, covariance matrix of the data can be decomposed to
    orthogonal vectors, called eigenvectors, associated with eigenvalues. The
    eigenvectors with high eigenvalues capture most of the variance in the
    data.

    Therefore, a low dimensional hyperplane constructed by k eigenvectors can
    capture most of the variance in the data. However, outliers are different
    from normal data points, which is more obvious on the hyperplane
    constructed by the eigenvectors with small eigenvalues.

    Therefore, outlier scores can be obtained as the sum of the projected
    distance of a sample on all eigenvectors.
    See :cite:`shyu2003novel,aggarwal2015outlier` for details.

    Score(X) = Sum of weighted euclidean distance between each sample to the
    hyperplane constructed by the selected eigenvectors

    :param n_components: Number of principal components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.
    :type n_components: int, float, None or str

    :param n_selected_components: Number of selected principal components
        for calculating the outlier scores. It is not necessarily equal to
        the total number of the principal components. If not set, use
        all principal components.
    :type n_selected_components: int, optional (default=None)

    :param contamination: The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    :type contamination: float in (0., 0.5), optional (default=0.1)

    :param copy: If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.
    :type copy: bool (default True)

    :param whiten: When True (False by default) the `components_` vectors are
        multiplied by the square root of n_samples and then divided by the
        singular values to ensure uncorrelated outputs with unit
        component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.
    :type whiten: bool, optional (default False)

    :param svd_solver:
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.
    :type svd_solver: str {'auto', 'full', 'arpack', 'randomized'}

    :param tol: Tolerance for singular values computed by
        svd_solver == 'arpack'.
    :type tol: float >= 0, optional (default .0)

    :param iterated_power: Number of iterations for the power method computed
        by svd_solver == 'randomized'.
    :type iterated_power: int >= 0, or 'auto', (default 'auto')

    :param random_state: If int, random_state is the seed used by the random
        number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.
    :type random_state: int, RandomState instance or None,
        optional (default None)

    :param weighted: If True, the eigenvalues are used in score computation.
        The eigenvectors with samll eigenvalues comes with more importance
        in outlier score calculation.
    :type weighted: bool, optional (default=True)

    :param standardization: If True, perform standardization first to convert
        data to zero mean and unit variance.
        See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
    :type standardization: bool, optional (default=True)

    :var components\_: Components with maximum variance.
    :vartype components\_: array, shape (n_components, n_features)

    :var explained_variance_ratio\_: Percentage of variance explained by each
        of the selected components. If k is not set then all components are
        stored and the sum of explained variances is equal to 1.0.
    :vartype explained_variance_ratio\_: array, shape (n_components,)

    :var singular_values\_: The singular values corresponding to each of the
        selected components. The singular values are equal to the 2-norms of
        the ``n_components`` variables in the lower-dimensional space.
    :vartype singular_values\_: array, shape (n_components,)

    :var mean\_: Per-feature empirical mean, estimated from the training set.
    :vartype mean\_: array, shape (n_features,)

    :var decision_scores\_: The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    :vartype decision_scores\_: numpy array of shape (n_samples,)

    :var threshold\_: The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    :vartype threshold\_: float

    :var labels\_: The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    :vartype labels\_: int, either 0 or 1
    """

    def __init__(self, n_components=None, n_selected_components=None,
                 contamination=0.1, copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=None,
                 weighted=True, standardization=True):

        super(PCA, self).__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted
        self.standardization = standardization

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        # Validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # PCA is recommended to use on the standardized data (zero mean and
        # unit variance).
        if self.standardization:
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)

        self.detector_ = sklearn_PCA(n_components=self.n_components,
                                     copy=self.copy,
                                     whiten=self.whiten,
                                     svd_solver=self.svd_solver,
                                     tol=self.tol,
                                     iterated_power=self.iterated_power,
                                     random_state=self.random_state)
        self.detector_.fit(X=X, y=y)

        # copy the attributes from the sklearn PCA object
        self.n_components_ = self.detector_.n_components_
        self.components_ = self.detector_.components_

        # validate the number of components to be used for outlier detection
        if self.n_selected_components is None:
            self.n_selected_components_ = self.n_components_
        else:
            self.n_selected_components_ = self.n_selected_components
        check_parameter(self.n_selected_components_, 1, self.n_components_,
                        include_left=True, include_right=True,
                        param_name='n_selected_components_')

        # use eigenvalues as the weights of eigenvectors
        self.w_components_ = np.ones([self.n_components_, ])
        if self.weighted:
            self.w_components_ = self.detector_.explained_variance_ratio_

        # outlier scores is the sum of the weighted distances between each
        # sample to the eigenvectors. The eigenvectors with smaller
        # eigenvalues have more influence
        # Not all eigenvectors are used, only n_selected_components_ smallest
        # are used since they better reflect the variance change

        self.selected_components_ = self.components_[
                                    -1 * self.n_selected_components_:, :]
        self.selected_w_components_ = self.w_components_[
                                      -1 * self.n_selected_components_:]

        self.decision_scores_ = np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_,
            axis=1).ravel()

        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['components_', 'w_components_'])

        X = check_array(X)
        if self.standardization:
            X = self.scaler_.transform(X)

        return np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_,
            axis=1).ravel()

    @property
    def explained_variance_ratio_(self):
        """Percentage of variance explained by each of the selected components.
        If k is not set then all components are stored and the sum of explained
        variances is equal to 1.0.
        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.explained_variance_ratio_

    @property
    def singular_values_(self):
        """The singular values corresponding to each of the selected
        components. The singular values are equal to the 2-norms of the
        ``n_components`` variables in the lower-dimensional space.
        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.singular_values_

    @property
    def mean_(self):
        """Per-feature empirical mean, estimated from the training set.
        Decorator for scikit-learn PCA attributes.
        """
        return self.detector_.mean_
