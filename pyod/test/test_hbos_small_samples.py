# License: BSD 2 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyod.models.hbos import HBOS
from pyod.utils.utility import get_optimal_n_bins


@pytest.mark.parametrize('n_samples', [1, 2, 3])
def test_optimal_bins_small_samples(n_samples):
    values = np.arange(n_samples, dtype=float)
    assert get_optimal_n_bins(values) == 1


@pytest.mark.parametrize('n_samples', [1, 2, 3])
def test_hbos_auto_small_training(n_samples):
    values = np.arange(n_samples, dtype=float)
    X = np.column_stack((values, np.zeros(n_samples)))
    automatic = HBOS(n_bins='auto').fit(X)
    # With these sample counts the only candidate histogram has one bin.
    for feature in range(X.shape[1]):
        histogram, edges = np.histogram(X[:, feature], bins=1, density=True)
        assert_allclose(automatic.hist_[feature], histogram)
        assert_allclose(automatic.bin_edges_[feature], edges)
    assert np.isfinite(automatic.decision_scores_).all()
    assert_allclose(automatic.decision_function(X), automatic.decision_scores_)
    assert automatic.predict(X).shape == (n_samples,)
    assert np.isfinite(automatic.decision_function(X[:1] + 10)).all()


def test_optimal_bins_empty_samples():
    with pytest.raises(ValueError):
        get_optimal_n_bins(np.array([]))


@pytest.mark.parametrize('n_samples', [4, 9, 16])
def test_optimal_bins_larger_samples(n_samples):
    values = np.arange(n_samples, dtype=float)
    assert get_optimal_n_bins(values) == 1


@pytest.mark.parametrize('upper_bound', [2, 3, 10])
def test_optimal_bins_explicit_bound(upper_bound):
    assert get_optimal_n_bins(np.arange(25, dtype=float), upper_bound) == 1
