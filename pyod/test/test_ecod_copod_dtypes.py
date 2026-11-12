# -*- coding: utf-8 -*-
"""ECOD and COPOD must score integer, boolean and nullable inputs as floats.

Both detectors take the right-tail ECDF of -X. Negating unsigned integers
wraps them, booleans reject unary minus, and pandas nullable dtypes reach
numba as object arrays (#764, #765, #497).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD

MODELS = [ECOD, COPOD]
N_JOBS = [1, 2]

# The last row is the largest in both columns, so its right-tail
# probability is 1/4 and it must receive the highest score.
X_SMALL = np.array([[0, 0], [1, 2], [2, 3], [4, 9]])


def _scores(model, n_jobs, X_train, X_test=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model(n_jobs=n_jobs).fit(X_train)
        if X_test is None:
            return clf.decision_scores_
        return clf.decision_function(X_test)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('n_jobs', N_JOBS)
@pytest.mark.parametrize('dtype', [np.uint8, np.uint64, np.int8, np.int64,
                                   np.float32])
def test_numeric_dtypes_match_float64(model, n_jobs, dtype):
    X = X_SMALL.astype(dtype)
    expected = _scores(model, n_jobs, X.astype(np.float64))
    assert_allclose(_scores(model, n_jobs, X), expected, rtol=1e-6)
    # decision_function scores new rows against the training rows plus
    # the new rows, so compare it with the float64 path, not with fit.
    assert_allclose(_scores(model, n_jobs, X, X[:2]),
                    _scores(model, n_jobs, X.astype(np.float64),
                            X[:2].astype(np.float64)), rtol=1e-6)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('n_jobs', N_JOBS)
def test_unsigned_right_tail(model, n_jobs):
    scores = _scores(model, n_jobs, X_SMALL.astype(np.uint64))
    assert scores[3] == scores.max()


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('n_jobs', N_JOBS)
def test_signed_minimum(model, n_jobs):
    info = np.iinfo(np.int64)
    X = np.array([[info.min, 0], [-1, 2], [0, 3], [info.max, 9]],
                 dtype=np.int64)
    expected = _scores(model, n_jobs, X.astype(np.float64))
    assert_allclose(_scores(model, n_jobs, X), expected)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('n_jobs', N_JOBS)
def test_boolean(model, n_jobs):
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [0, 0]],
                 dtype=bool)
    expected = _scores(model, n_jobs, X.astype(np.float64))
    assert_allclose(_scores(model, n_jobs, X), expected)


@pytest.mark.parametrize('model', MODELS)
def test_pandas_nullable_dtype(model):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randint(0, 10, size=(30, 3))).astype('Int64')
    X = df.to_numpy(dtype=np.float64)
    assert_allclose(_scores(model, 1, df), _scores(model, 1, X))
    assert_allclose(_scores(model, 1, df, df.iloc[:5]),
                    _scores(model, 1, X, X[:5]))
