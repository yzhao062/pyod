"""
Abstract base class for outlier detector models
"""

from abc import ABC, abstractmethod
import numpy as np

from scipy.stats import rankdata
from scipy.special import erf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import NotFittedError
from ..utils.utility import precision_n_scores


class BaseDetector(ABC):

    @abstractmethod
    def __init__(self, contamination=0.1):
        """
        :param contamination: percentage of outliers, range in (0, 0.5]
        :type contamination: float
        """
        self.contamination = contamination
        self.threshold_ = None
        self.decision_scores = None
        self.y_pred = None
        self._isfitted = False

    @abstractmethod
    def decision_function(self, X):
        """
        Anomaly score of X of the base classifiers.
        The anomaly score of an input sample is computed based on different detector algorithms.
        For consistency, outliers have larger anomaly scores.

        :param X: The training input samples. Sparse matrices are accepted only if they are supported by the base estimator.
        :type X: {array-like, sparse matrix}
        :return: scores: The anomaly score of the input samples. The lower, the more abnormal.
        :rtype: array of shape (n_samples,)
        """
        pass

    @abstractmethod
    def fit(self):
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.y_pred

    def predict(self, X_test):
        if not self._isfitted:
            NotFittedError('Model is not fitted yet')

        pred_score = self.decision_function(X_test)
        return (pred_score > self.threshold_).astype('int').ravel()

    def predict_proba(self, X_test, method='linear'):
        if not self._isfitted:
            NotFittedError('Model is not fitted yet')

        test_scores = self.decision_function(X_test)
        train_scores = self.decision_scores

        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            proba = scaler.transform(test_scores.reshape(-1, 1))
            return proba.clip(0, 1)
        else:
            # turn output into probability
            pre_erf_score = (test_scores - self.mu) / (self.sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            proba = erf_score.clip(0)

            return proba

    def predict_rank(self, X_test):
        if not self._isfitted:
            NotFittedError('Model is not fitted yet')

        test_scores = self.decision_function(X_test)
        train_scores = self.decision_scores

        ranks = np.zeros([X_test.shape[0], 1])

        for i in range(test_scores.shape[0]):
            train_scores_i = np.append(train_scores.reshape(-1, 1),
                                       test_scores[i])

            ranks[i] = rankdata(train_scores_i)[-1]

        # return normalized ranks
        ranks_norm = ranks / ranks.max()
        return ranks_norm

    def evaluate(self, X_test, y_test):
        pred_score = self.decision_function(X_test)
        prec_n = (precision_n_scores(y_test, pred_score))
        roc = roc_auc_score(y_test, pred_score)
        print("roc", prec_n)
        print("precision@n_train", prec_n)
