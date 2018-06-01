# -*- coding: utf-8 -*-
"""
Base class for all outlier detector models
"""

import copy
import warnings
from inspect import signature
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import six
from scipy import sparse
from scipy.stats import rankdata
from scipy.special import erf
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from ..utils.utility import precision_n_scores


def _first_and_last_element(arr):
    """
    Returns first and last element of numpy array or sparse matrix.

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


def clone(estimator, safe=True):
    """
    Constructs a new estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    See sklearn/base.py for more information.

    Parameters
    ----------
    estimator : estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned

    safe : boolean, optional
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.

    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in six.iteritems(new_object_params):
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is param2:
            # this should always happen
            continue
        if isinstance(param1, np.ndarray):
            # For most ndarrays, we do not test for complete equality
            if not isinstance(param2, type(param1)):
                equality_test = False
            elif (param1.ndim > 0
                  and param1.shape[0] > 0
                  and isinstance(param2, np.ndarray)
                  and param2.ndim > 0
                  and param2.shape[0] > 0):
                equality_test = (
                        param1.shape == param2.shape
                        and param1.dtype == param2.dtype
                        and (_first_and_last_element(param1) ==
                             _first_and_last_element(param2))
                )
            else:
                equality_test = np.all(param1 == param2)
        elif sparse.issparse(param1):
            # For sparse matrices equality doesn't work
            if not sparse.issparse(param2):
                equality_test = False
            elif param1.size == 0 or param2.size == 0:
                equality_test = (
                        param1.__class__ == param2.__class__
                        and param1.size == 0
                        and param2.size == 0
                )
            else:
                equality_test = (
                        param1.__class__ == param2.__class__
                        and (_first_and_last_element(param1) ==
                             _first_and_last_element(param2))
                        and param1.nnz == param2.nnz
                        and param1.shape == param2.shape
                )
        else:
            # fall back on standard equality
            equality_test = param1 == param2
        if equality_test:
            warnings.warn("Estimator %s modifies parameters in __init__."
                          " This behavior is deprecated as of 0.18 and "
                          "support for this behavior will be removed in 0.20."
                          % type(estimator).__name__, DeprecationWarning)
        else:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'does not seem to set parameter %s' %
                               (estimator, name))

    return new_object


def _pprint(params, offset=0, printer=repr):
    """
    Pretty print the dictionary 'params'

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
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
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


class BaseDetector(ABC):
    """
    Abstract class for all outlier detection algorithms.

    :param contamination: The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    :type contamination: float in (0., 0.5), optional (default=0.1)
    """

    @abstractmethod
    def __init__(self, contamination=0.1):

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination

    @abstractmethod
    def decision_function(self, X):
        """
        Predict Anomaly score of X of the base classifiers. The anomaly score
        of an input sample is computed based on different detector algorithms.
        For consistency, outliers have larger anomaly decision_scores_.

        :param X: The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        :type X: numpy array of shape (n_samples, n_features)

        :return: decision_scores_: The anomaly score of the input samples.
        :rtype: array, shape (n_samples,)
        """
        pass

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit detector.

        :param X: The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        :type X: numpy array of shape (n_samples, n_features)

        :param y: ground truth
        :type y: numpy array of shape (n_samples,), optional

        :return: return self
        :rtype: object
        """
        pass

    def fit_predict(self, X, y=None):
        """
        Fit detector and predict if a particular sample is an outlier or not.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :param y: ground truth
        :type y: numpy array of shape (n_samples,), optional

        :return: For each observation, tells whether or not
            it should be considered as an outlier according to the fitted model.
            0 stands for inliers and 1 for outliers.
        :rtype: array, shape (n_samples,)
        """

        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.

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
        """
        Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           decision_scores_ into the range of [0,1]. The model must be fitted first.
        2. use unifying decision_scores_, see reference [1] below.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :param method: probability conversion method. It must be one of
            'linear' or 'unify'.
        :type method: str, optional (default='linear')

        :return: For each observation, return the outlier probability, ranging
            in [0,1]
        :rtype: array, shape (n_samples,)


        .. [1] Kriegel, H.P., Kroger, P., Schubert, E. and Zimek, A., 2011,
               April. Interpreting and unifying outlier decision_scores_.
               In Proc' SIAM, 2011.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], int(self.classes_)])
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

    def predict_rank(self, X):
        """
        Predict the outlyingness rank of a sample in a fitted model. The
        method is specifically for combining various outlier detectors.

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :return: outlying rank of a sample according to the training data
        :rtype: array, shape (n_samples,)
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        test_scores = self.decision_function(X)
        train_scores = self.decision_scores_

        ranks = np.zeros([X.shape[0], 1])

        for i in range(test_scores.shape[0]):
            train_scores_i = np.append(train_scores.reshape(-1, 1),
                                       test_scores[i])

            ranks[i] = rankdata(train_scores_i)[-1]

        # return normalized ranks
        ranks_norm = ranks / ranks.max()
        return ranks_norm

    def fit_predict_evaluate(self, X, y):
        """
        Fit the detector, predict on samples, and evaluate the model

        :param X: The input samples
        :type X: numpy array of shape (n_samples, n_features)

        :param y: Outlier labels of the input samples
        :type y: array, shape (n_samples,)

        :return: roc score and precision @ rank n score
        :rtype:  tuple (float, float)
        """

        self.fit(X)
        roc = roc_auc_score(y, self.decision_scores_)
        prec_n = precision_n_scores(y, self.decision_scores_)

        print("roc score:", roc)
        print("precision @ rank n:", prec_n)

        return roc, prec_n

    def _process_decision_scores(self):
        """
        Internal function to calculate key attributes:
        threshold: used to decide the binary label
        labels_: binary labels of training data

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

    def _get_param_names(cls):
        """
        Get parameter names for the estimator

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
        """
        Get parameters for this estimator.

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
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name), ),)
