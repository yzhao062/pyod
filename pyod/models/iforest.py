from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector


class IForest(BaseDetector):
    """
    Wrapper of Sklearn Isolation Forest Class with more functionalities.

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree_ structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.
    """

    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        """

        :param n_estimators: int, optional (default=100).
            The number of base estimators in the ensemble.
        :param max_samples: int or float, optional (default="auto")
        :param contamination: float in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. Used when fitting to define the
            threshold_ on the decision function.
        :param max_features: int or float, optional (default=1.0)
            The number of features to draw from X to train each base estimator.
        :param bootstrap:
        :param n_jobs:
        :param random_state:
        :param verbose:
        """
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                         max_samples=self.max_samples,
                                         contamination=self.contamination,
                                         max_features=self.max_features,
                                         bootstrap=self.bootstrap,
                                         n_jobs=self.n_jobs,
                                         random_state=self.random_state,
                                         verbose=self.verbose)

    def fit(self, X_train, y=None, sample_weight=None):
        self._isfitted = True
        self.detector_.fit(X_train,
                           y=None,
                           sample_weight=None)

        # invert decision_scores. Outliers comes with higher decision_scores
        self.decision_scores = self.detector_.decision_function(X_train) * -1
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores', 'threshold_', 'y_pred'])
        # invert decision_scores. Outliers comes with higher decision_scores
        return self.detector_.decision_function(X) * -1

    @property
    def estimators_(self):
        """
        decorator for sklearn Isolation Forest attributes
        :return:
        """
        return self.detector_.estimators_

    @property
    def estimators_samples_(self):
        """
        decorator for sklearn Isolation Forest attributes
        :return:
        """
        return self.detector_.estimators_samples_

    @property
    def max_samples_(self):
        """
        decorator for sklearn Isolation Forest attributes
        :return:
        """
        return self.detector_.max_samples_
