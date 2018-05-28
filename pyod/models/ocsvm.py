from sklearn.svm import OneClassSVM
from scipy.stats import scoreatpercentile
from sklearn.exceptions import NotFittedError
from .base import BaseDetector


class OCSVM(BaseDetector):
    """

    Wrapper of Sklearn one-class SVM Class with more functionalities.
    Unsupervised Outlier Detection.

    Estimate the support of a high-dimensional distribution.

    The implementation is based on libsvm.
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

        self.detector_ = OneClassSVM(kernel=kernel,
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

    def fit(self, X_train, y=None, sample_weight=None, **params):
        self._isfitted = True
        self.detector_.fit(X=X_train, y=y, sample_weight=sample_weight,
                           **params)
        # invert decision_scores. Outliers comes with higher decision_scores
        self.decision_scores = self.detector_.decision_function(X_train) * -1
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        if not self._isfitted:
            NotFittedError('Model is not fitted yet')
        # invert decision_scores. Outliers comes with higher decision_scores
        return self.detector_.decision_function(X) * -1

    @property
    def support_(self):
        """
        decorator for sklearn Oneclass SVM attributes
        :return:
        """
        return self.detector_.support_

    @property
    def support_vectors_(self):
        """
        decorator for sklearn Oneclass SVM attributes
        :return:
        """
        return self.detector_.support_vectors_

    @property
    def dual_coef_(self):
        """
        decorator for sklearn Oneclass SVM attributes
        :return:
        """
        return self.detector_.dual_coef_

    @property
    def coef_(self):
        """
        decorator for sklearn Oneclass SVM attributes
        :return:
        """
        return self.detector_.coef_

    @property
    def intercept_(self):
        """
        decorator for sklearn Oneclass SVM attributes
        :return:
        """
        return self.detector_.intercept_
