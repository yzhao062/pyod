from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector


class LOF(BaseDetector):
    """
    Wrapper of Sklearn LOF Class with more functionalities.
    Unsupervised Outlier Detection using Local Outlier Factor (LOF)

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.
    """

    def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.n_jobs = n_jobs

        self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                            algorithm=self.algorithm,
                                            leaf_size=self.leaf_size,
                                            metric=self.metric,
                                            p=self.p,
                                            metric_params=self.metric_params,
                                            contamination=self.contamination,
                                            n_jobs=self.n_jobs)

    def fit(self, X_train, y=None):
        self.detector_.fit(X=X_train, y=y)
        self.decision_scores = self.detector_.negative_outlier_factor_ * -1
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores', 'threshold_', 'y_pred'])

        # invert decision_scores. Outliers comes with higher decision_scores
        return self.detector_._decision_function(X) * -1

    @property
    def negative_outlier_factor_(self):
        """
        decorator for sklearn LOF attributes
        :return:
        """
        return self.detector_.negative_outlier_factor_

    @property
    def n_neighbors_(self):
        """
        decorator for sklearn LOF attributes
        :return:
        """
        return self.detector_.n_neighbors_
