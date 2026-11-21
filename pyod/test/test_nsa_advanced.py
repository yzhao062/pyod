# -*- coding: utf-8 -*-
"""Public PyOD contracts shared by the researched NSA mechanisms."""
# License: BSD 2 clause

import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from pyod.models.nsa import NSA


ADVANCED = ['coverage', 'grid', 'hierarchical', 'voronoi', 'deterministic',
            'suppressed', 'dual', 'annealed']


def fitted(strategy, **overrides):
    params = dict(strategy=strategy, n_detectors=100, max_candidates=12000,
                  random_state=42, self_radius=0.04)
    params.update(overrides)
    X = np.repeat(np.linspace(0, 1, 21)[:, None], 2, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return NSA(**params).fit(X), X


@pytest.mark.parametrize('strategy', ADVANCED)
def test_public_scoring_polarity_and_heldout_discrimination(strategy):
    model, X = fitted(strategy)
    normal = np.repeat(np.linspace(0.15, 0.85, 25)[:, None], 2, axis=1)
    abnormal = np.tile([0.2, 0.8], (25, 1))
    abnormal += np.random.RandomState(9).uniform(-0.03, 0.03, abnormal.shape)
    query = np.vstack([normal, abnormal])
    y = np.repeat([0, 1], 25)
    scores = model.decision_function(query)
    assert scores.shape == (50,)
    assert np.isfinite(scores).all()
    assert roc_auc_score(y, scores) >= 0.95
    assert scores[y == 1].mean() > scores[y == 0].mean()
    assert_allclose(model.decision_scores_, model.decision_function(X))
    assert_array_equal(model.labels_, model.predict(X))
    assert np.isfinite(model.predict_proba(query)).all()
    assert np.isfinite(model.predict_proba(query, method='unify')).all()
    labels, confidence = model.predict(query, return_confidence=True)
    assert labels.shape == confidence.shape == (50,)
    assert np.all((confidence >= 0) & (confidence <= 1))
    # Independent explicit geometry oracle for every returned score.
    oracle = np.max(model.detector_radii_[None, :] - cdist(
        model.scaler_.transform(query), model.detectors_), axis=1)
    if strategy == 'suppressed':
        reverse = cdist(model.scaler_.transform(query),
                        model.reverse_detectors_).min(axis=1)
        oracle = np.minimum(oracle, reverse - model.reverse_radius_)
    if strategy == 'dual':
        outside = (cdist(model.scaler_.transform(query), model.apc_centers_)
                   - model.apc_radii_).min(axis=1)
        oracle = np.maximum(oracle, outside)
    assert_allclose(scores, oracle)
    assert np.max(model.decision_scores_) < 0
    assert 0 < model.n_detectors_ <= model.n_detectors
    assert model.n_detectors_ <= model.n_candidates_ <= model.max_candidates
    if strategy != 'suppressed':
        assert np.all(cdist(model.detectors_, model.self_samples_)
                      >= model.detector_radii_[:, None] + model.self_radius
                      - 1e-10)


@pytest.mark.parametrize('strategy', ADVANCED)
def test_clone_pickle_reproducibility_and_batch_independence(strategy):
    model, X = fitted(strategy)
    copied, _ = fitted(strategy)
    assert_allclose(model.detectors_, copied.detectors_)
    assert_allclose(model.detector_radii_, copied.detector_radii_)
    assert clone(model).get_params() == model.get_params()
    restored = pickle.loads(pickle.dumps(model))
    query = np.random.RandomState(17).uniform(-0.4, 1.4, (31, 2))
    expected = model.decision_function(query)
    assert_allclose(restored.decision_function(query), expected)
    assert_allclose(np.concatenate([
        model.decision_function(part) for part in np.array_split(query, 7)]),
        expected)
    assert_allclose(model.decision_function(query[::-1]), expected[::-1])
    assert_array_equal(X[:, 0], np.linspace(0, 1, 21))


@pytest.mark.parametrize('strategy', ADVANCED)
def test_y_is_ignored_and_failed_refit_is_not_predictable(strategy):
    model, X = fitted(strategy)
    original = model.decision_scores_.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model.fit(X, np.arange(len(X)) % 3)
    assert_allclose(model.decision_scores_, original)
    assert model.predict_proba(X).shape == (len(X), 2)
    with pytest.raises(ValueError):
        model.fit([[np.nan, 0]])
    with pytest.raises(NotFittedError):
        model.decision_function(X)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not hasattr(model, 'generation_diagnostics_')


@pytest.mark.parametrize('strategy', ['voronoi', 'deterministic'])
def test_geometric_strategy_rejects_unsupported_dimensions(strategy):
    with pytest.raises(ValueError, match='dimension|feature'):
        NSA(strategy=strategy).fit(np.ones((5, 5)))


def test_grid_depth_upper_limit_is_explicit():
    with pytest.raises(ValueError, match='grid_depth'):
        NSA(strategy='grid', grid_depth=21).fit([[0, 0], [1, 1]])


def test_reverse_suppression_is_active_through_public_scoring():
    X = np.vstack([np.zeros((19, 2)), [1., 1.]])
    with pytest.warns(UserWarning, match='Only'):
        model = NSA(strategy='suppressed', outlier_radius=0.2,
                    n_detectors=40, random_state=42).fit(X)
    negative_only = np.max(model.detector_radii_ - cdist(
        [[1., 1.]], model.detectors_)[0])
    assert negative_only > 0
    assert_allclose(model.decision_function([[1., 1.]]), [-0.1])
    assert np.all(model.decision_scores_ <= 0)
    model.set_params(strategy='fixed', n_detectors=1).fit(X)
    assert not hasattr(model, 'reverse_detectors_')


@pytest.mark.parametrize('name,bad', [
    ('outlier_fraction', 0), ('outlier_fraction', 1),
    ('outlier_fraction', np.nan), ('outlier_fraction', True),
    ('outlier_radius', 0), ('outlier_radius', -1),
    ('outlier_radius', np.inf), ('outlier_radius', True)])
def test_suppression_parameters_validated(name, bad):
    with pytest.raises(ValueError, match=name):
        NSA(strategy='suppressed', **{name: bad}).fit([[0, 0], [1, 1]])


@pytest.mark.parametrize('stage', ['generation', 'scoring', 'thresholding'])
def test_late_fit_failures_clear_all_learned_state(monkeypatch, stage):
    model, X = fitted('fixed')
    check_is_fitted(model)
    before = model.get_params().copy()

    def fail(*args, **kwargs):
        raise ValueError('simulated late failure')

    method = {'generation': '_generate_real',
              'scoring': 'decision_function',
              'thresholding': '_process_decision_scores'}[stage]
    monkeypatch.setattr(model, method, fail)
    with pytest.raises(ValueError, match='simulated late failure'):
        model.fit(X)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not any(name.endswith('_') for name in vars(model))
    assert not any(name in vars(model)
                   for name in ['_mu', '_sigma', '_classes'])
    assert model.get_params() == before


def test_fresh_generation_failure_has_no_fitted_markers():
    model = NSA(strategy='fixed', sampling_margin=0, self_radius=10,
                max_candidates=2)
    with pytest.raises(ValueError, match='No valid'):
        model.fit([[0, 0], [1, 1]])
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    assert not any(name.endswith('_') for name in vars(model))
