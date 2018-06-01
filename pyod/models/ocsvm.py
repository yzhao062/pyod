# -*- coding: utf-8 -*-
import warnings

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .base import BaseDetector


class OCSVM(BaseDetector):
    """
    Wrapper of scikit-learn one-class SVM Class with more functionalities.
    Unsupervised Outlier Detection.

    Estimate the support of a high-dimensional distribution.

    The implementation is based on libsvm.
    Read more: http://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection

    :param kernel: Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to precompute the kernel matrix.
    :type kernel: str, optional (default='rbf')

    :param degree: Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    :type degree: int, optional (default=3)

    :param gamma:  Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.
    :type gamma: float, optional (default='auto')

    :param coef0:  Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    :type coef0: float, optional (default=0.0)

    :param tol: Tolerance for stopping criterion.
    :type tol: float, optional

    :param nu: An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.
    :type nu: float, optional

    :param shrinking: Whether to use the shrinking heuristic.
    :type shrinking: bool, optional

    :param cache_size: Specify the size of the kernel cache (in MB).
    :type cache_size: float, optional

    :param verbose: Enable verbose output. Note that this setting takes
        advantage of a per-process runtime setting in libsvm that, if enabled,
        may not work properly in a multithreaded context.
    :type verbose: bool, default: False

    :param max_iter: Hard limit on iterations within solver, or -1 for no limit.
    :type max_iter: int, optional (default=-1)

    :param random_state: The seed of the pseudo random number generator to use
        when shuffling the data.
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    :type random_state: int, RandomState instance or None, optional
        (default=None)

    :var support_(array): Indices of support vectors.

    :var support_vectors_(array): Support vectors.

    :var dual_coef_(array): Coefficients of the support vectors in the
        decision function.

    :var coef_(array): Weights assigned to the features (coefficients
        in the primal problem). This is only available in the case of
        a linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`

    :var intercept_(array): Constant in the decision function.
    """

    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, random_state=None):
        super().__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None, **params):
        self.classes_ = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            print(np.unique(y, return_counts=True))
            self.classes_ = len(np.unique(y))
            warnings.warn(
                "y should not be presented in unsupervised learning.")

        self.detector_ = OneClassSVM(kernel=self.kernel,
                                     degree=self.degree,
                                     gamma=self.gamma,
                                     coef0=self.coef0,
                                     tol=self.tol,
                                     nu=self.nu,
                                     shrinking=self.shrinking,
                                     cache_size=self.cache_size,
                                     verbose=self.verbose,
                                     max_iter=self.max_iter,
                                     random_state=self.random_state)
        self.detector_.fit(X=X, y=y, sample_weight=sample_weight,
                           **params)

        # invert decision_scores_. Outliers comes with higher decision_scores_
        self.decision_scores_ = self.detector_.decision_function(X) * -1
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        # invert decision_scores_. Outliers comes with higher decision_scores_
        return self.detector_.decision_function(X) * -1

    @property
    def support_(self):
        """
        Indices of support vectors.

        Decorator for scikit-learn One class SVM attributes.

        :return: Indices of support vectors.
        :rtype: array-like, shape = [n_SV]
        """
        return self.detector_.support_

    @property
    def support_vectors_(self):
        """
        Support vectors.
        Decorator for scikit-learn One class SVM attributes.

        :return: Support vectors.
        :rtype: array-like, shape = [nSV, n_features]
        """
        return self.detector_.support_vectors_

    @property
    def dual_coef_(self):
        """
        Coefficients of the support vectors in the decision function.

        Decorator for scikit-learn One class SVM attributes.

        :return: Coefficients of the support vectors in the decision function.
        :rtype: array, shape = [1, n_SV]
        """
        return self.detector_.dual_coef_

    @property
    def coef_(self):
        """
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`

        Decorator for scikit-learn One class SVM attributes.

        :return: Weights assigned to the features
        :rtype: array, shape = [1, n_features]
        """
        return self.detector_.coef_

    @property
    def intercept_(self):
        """

        Decorator for scikit-learn One class SVM attributes.

        :return: Constant in the decision function.
        :rtype: array, shape = [1,]
        """
        return self.detector_.intercept_
