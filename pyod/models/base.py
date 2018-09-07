# -*- coding: utf-8 -*-
"""Base class for all outlier detector models
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import warnings
from collections import defaultdict

import abc
from sklearn.externals.funcsigs import signature
from sklearn.externals import six
from sklearn.externals.joblib import cpu_count

import numpy as np
from scipy import sparse
from scipy.special import erf
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from ..utils.utility import precision_n_scores


def _first_and_last_element(arr):
    """Returns first and last element of numpy array or sparse matrix.
    See sklearn/base.py for more information.
    """

    if isinstance(arr, np.ndarray) or hasattr(arr, 'data'):
        # numpy array or sparse matrix with .data attribute
        data = arr.data if sparse.issparse(arr) else arr
        return data.flat[0], data.flat[-1]
    else:
        # Sparse matrices without .data attribute. Only dok_matrix at
        # the time of writing, in this case indexing is fast
        return arr[0, 0], arr[-1, -1]

def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.
    See sklearn/utils/__init__.py for more information.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    Examples
    --------
    >>> from sklearn.utils import _get_n_jobs
    >>> _get_n_jobs(4)
    4
    >>> jobs = _get_n_jobs(-2)
    >>> assert jobs == max(cpu_count() - 1, 1)
    >>> _get_n_jobs(0)
    Traceback (most recent call last):
    ...
    ValueError: Parameter n_jobs == 0 has no meaning.
    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs

def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs.
    See sklearn/ensemble/base.py for more information.
    """
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()

def _pprint(params, offset=0, printer=repr):
    # noinspection PyPep8
    """Pretty print the dictionary 'params'

    See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    and sklearn/base.py for more information.

    :param params: The dictionary to pretty print
    :type params: dict

    :param offset: The offset in characters to add at the begin of each line.
    :type offset: int

    :param printer: The function to convert entries to strings, typically
        the builtin str or repr
    :type printer: callable

    :return: None
    """

    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


@six.add_metaclass(abc.ABCMeta)
class BaseDetector(object):
    """Abstract class for all outlier detection algorithms.

    :param contamination: The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    :type contamination: float in (0., 0.5), optional (default=0.1)

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

    @abc.abstractmethod
    def __init__(self, contamination=0.1):

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination

    @abc.abstractmethod
    def decision_function(self, X):
        """Predict anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        :param X: The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        :type X: numpy array of shape (n_samples, n_features)

        :return: The anomaly score of the input samples.
        :rtype: array, shape (n_samples,)
        """
        pass

    # noinspection PyIncorrectDocstring
    @abc.abstractmethod
    def fit(self, X, y=None):
        """Fit detector.

        :param X: The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        :type X: numpy array of shape (n_samples, n_features)

        :return: return self
        :rtype: object
        """
        pass

    # noinspection PyIncorrectDocstring
    def fit_predict(self, X, y=None):
        """Fit detector and predict if a particular sample is an outlier or
        not.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :return: For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        :rtype: array, shape (n_samples,)
        """

        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :return: For each observation, tells whether or not
            it should be considered as an outlier according to the fitted
            model. 0 stands for inliers and 1 for outliers.
        :rtype: array, shape (n_samples,)
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score = self.decision_function(X)
        return (pred_score > self.threshold_).astype('int').ravel()

    def predict_proba(self, X, method='linear'):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :param method: probability conversion method. It must be one of
            'linear' or 'unify'.
        :type method: str, optional (default='linear')

        :return: For each observation, return the outlier probability, ranging
            in [0,1]
        :rtype: array, shape (n_samples,)

        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

    def _predict_rank(self, X, normalized=False):
        """Predict the outlyingness rank of a sample in a fitted model. The
        method is specifically for combining various outlier detectors.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :param normalized: If set to True, all ranks are normalized to [0,1]
        :type normalized: bool, optional (default=False)

        :return: outlying rank of a sample according to the training data
        :rtype: array, shape (n_samples,)
        """

        check_is_fitted(self, ['decision_scores_'])

        test_scores = self.decision_function(X)
        train_scores = self.decision_scores_

        sorted_train_scores = np.sort(train_scores)
        ranks = np.searchsorted(sorted_train_scores, test_scores)

        if normalized:
            # return normalized ranks
            ranks = ranks / ranks.max()
        return ranks

    def fit_predict_score(self, X, y, scoring='roc_auc_score'):
        """Fit the detector, predict on samples, and evaluate the model by
        ROC and Precision @ rank n

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :param y: Outlier labels of the input samples
        :type y: array, shape (n_samples,)

        :param scoring: Evaluation metric

                -' roc_auc_score': ROC score
                - 'prc_n_score': Precision @ rank n score
        :type scoring: str, optional (default='roc_auc_score')

        :return: Evaluation score
        :rtype: float
        """

        self.fit(X)

        if scoring == 'roc_auc_score':
            score = roc_auc_score(y, self.decision_scores_)
        elif scoring == 'prc_n_score':
            score = precision_n_scores(y, self.decision_scores_)
        else:
            raise NotImplementedError('PyOD built-in scoring only supports '
                                      'ROC and Precision @ rank n')

        print("{metric}: {score}".format(metric=scoring, score=score))

        return score

    # def score(self, X, y, scoring='roc_auc_score'):
    #     """Returns the evaluation resulted on the given test data and labels.
    #     ROC is chosen as the default evaluation metric
    #
    #     :param X: The input samples
    #     :type X: numpy array of shape (n_samples, n_features)
    #
    #     :param y: Outlier labels of the input samples
    #     :type y: array, shape (n_samples,)
    #
    #     :param scoring: Evaluation metric
    #
    #             -' roc_auc_score': ROC score
    #             - 'prc_n_score': Precision @ rank n score
    #     :type scoring: str, optional (default='roc_auc_score')
    #
    #     :return: Evaluation score
    #     :rtype: float
    #     """
    #     check_is_fitted(self, ['decision_scores_'])
    #     if scoring == 'roc_auc_score':
    #         score = roc_auc_score(y, self.decision_function(X))
    #     elif scoring == 'prc_n_score':
    #         score = precision_n_scores(y, self.decision_function(X))
    #     else:
    #         raise NotImplementedError('PyOD built-in scoring only supports '
    #                                   'ROC and Precision @ rank n')
    #
    #     print("{metric}: {score}".format(metric=scoring, score=score))
    #
    #     return score

    def _set_n_classes(self, y):
        """Set the number of classes if y is presented, which is not expected.
        It could be useful for multi-class outlier detection.

        :param y: Ground truth
        :type y: numpy array of shape (n_samples,)
        """

        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))
            warnings.warn(
                "y should not be presented in unsupervised learning.")

    def _process_decision_scores(self):
        """Internal function to calculate key attributes:

        - threshold: used to decide the binary label
        - labels_: binary labels of training data

        :return: self
        :rtype: object
        """

        self.threshold_ = scoreatpercentile(self.decision_scores_,
                                            100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        # noinspection PyPep8
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        :param deep: If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        :return: mapping of string to any
            Parameter names mapped to their values.
        :rtype: str
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        :return: self
        :rtype: object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        # noinspection PyPep8
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name), ),)
