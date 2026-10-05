# -*- coding: utf-8 -*-
"""Behavioral and geometric contracts for negative selection strategies."""
# License: BSD 2 clause

import itertools
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from pyod.models.base import BaseDetector
from pyod.models.nsa import NSA


STRATEGIES = [
    pytest.param({'strategy': 'fixed', 'detector_radius': 0.25}, id='fixed'),
    pytest.param({'strategy': 'variable'}, id='variable'),
    pytest.param({'strategy': 'binary', 'binary_match': 'hamming',
                  'match_threshold': 1}, id='binary-hamming'),
    pytest.param({'strategy': 'binary', 'binary_match': 'rcontiguous',
                  'match_threshold': 3}, id='binary-rcontiguous'),
    pytest.param({'strategy': 'binary', 'binary_match': 'rchunk',
                  'match_threshold': 3}, id='binary-rchunk'),
]


@pytest.fixture
def self_data():
    """A thin normal manifold with two correlated binary self patterns."""
    return np.repeat(np.linspace(0., 1., 61)[:, None], 8, axis=1)


@pytest.fixture
def query_data():
    # Both populations lie inside the fitted domain; off-manifold points
    # break the correlation without relying on an arbitrary score sign.
    normal = np.repeat(np.linspace(0.15, 0.85, 24)[:, None], 8, axis=1)
    abnormal = np.tile([0.15, 0.85], (24, 4))
    abnormal += np.random.RandomState(21).uniform(-0.03, 0.03, abnormal.shape)
    return np.vstack([normal, abnormal]), np.repeat([0, 1], 24)


def make_detector(options, **overrides):
    parameters = dict(n_detectors=24, max_candidates=10000,
                      self_radius=0.08, random_state=42)
    parameters.update(options)
    parameters.update(overrides)
    return NSA(**parameters)


@pytest.mark.parametrize('options', STRATEGIES)
def test_each_strategy_has_correct_polarity_and_discrimination(
        options, self_data, query_data):
    detector = make_detector(options)
    X, y = query_data
    # Real hypersphere coverage is tested in a low-dimensional bounded
    # domain. Binary matching retains eight dimensions to permit valid
    # three-bit contiguous/chunk rules and multiple nonself patterns.
    if options['strategy'] != 'binary':
        self_data, X = self_data[:, :2], X[:, :2]
    assert detector.fit(self_data) is detector
    scores = detector.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert scores[y == 1].mean() > scores[y == 0].mean()
    assert roc_auc_score(y, scores) >= 0.95
    assert detector.decision_scores_.shape == (len(self_data),)
    assert detector.labels_.shape == (len(self_data),)
    assert np.isfinite(detector.threshold_)
    assert detector.n_features_in_ == self_data.shape[1]
    assert detector.n_detectors_ == detector.n_detectors
    assert detector.n_detectors_ <= detector.n_candidates_
    assert detector.n_candidates_ <= detector.max_candidates
    assert_allclose(detector.decision_scores_,
                    detector.decision_function(self_data))
    assert_array_equal(detector.labels_, detector.predict(self_data))


@pytest.mark.parametrize('options', STRATEGIES)
def test_inference_does_not_depend_on_batch_or_mutate_inputs(
        options, self_data, query_data):
    train_copy = self_data.copy()
    detector = make_detector(options).fit(self_data)
    X, _ = query_data
    query_copy = X.copy()
    before = detector.detectors_.copy()
    full = detector.decision_function(X)
    chunks = np.concatenate([detector.decision_function(chunk)
                             for chunk in np.array_split(X, 7)])
    individually = np.array([detector.decision_function(row[None, :])[0]
                             for row in X])
    assert_allclose(full, chunks)
    assert_allclose(full, individually)
    assert_allclose(detector.decision_function(X[::-1]), full[::-1])
    assert_array_equal(self_data, train_copy)
    assert_array_equal(X, query_copy)
    assert_array_equal(detector.detectors_, before)


@pytest.mark.parametrize('options', STRATEGIES)
@pytest.mark.parametrize('use_random_state', [False, True])
def test_reproducibility_and_pickle(options, use_random_state, self_data,
                                    query_data):
    seeds = ([np.random.RandomState(9), np.random.RandomState(9)]
             if use_random_state else [9, 9])
    first = make_detector(options, random_state=seeds[0]).fit(self_data)
    second = make_detector(options, random_state=seeds[1]).fit(self_data)
    X, _ = query_data
    assert_array_equal(first.detectors_, second.detectors_)
    assert_allclose(first.decision_function(X), second.decision_function(X))
    restored = pickle.loads(pickle.dumps(first))
    assert_allclose(restored.decision_function(X), first.decision_function(X))
    assert_array_equal(restored.predict(X), first.predict(X))


@pytest.mark.parametrize('options', STRATEGIES)
def test_constructor_and_clone_only_store_parameters(options, self_data):
    detector = make_detector(options)
    assert isinstance(detector, BaseDetector)
    assert not any(name.endswith('_') for name in vars(detector))
    parameters = detector.get_params()
    copied = clone(detector)
    assert copied.get_params() == parameters
    detector.fit(self_data)
    assert detector.get_params() == parameters
    assert not any(name.endswith('_') for name in vars(clone(detector)))
    assert not hasattr(detector, 'partial_fit')


@pytest.mark.parametrize('options', STRATEGIES)
def test_predict_and_linear_probability_contract(
        options, self_data, query_data):
    detector = make_detector(options).fit(self_data)
    X, _ = query_data
    scores = detector.decision_function(X)
    labels, confidence = detector.predict(X, return_confidence=True)
    assert_array_equal(labels, (scores > detector.threshold_).astype(int))
    assert labels.shape == confidence.shape == (len(X),)
    assert np.isfinite(confidence).all()
    assert ((confidence >= 0.) & (confidence <= 1.)).all()
    probability, probability_confidence = detector.predict_proba(
        X, method='linear', return_confidence=True)
    assert probability.shape == (len(X), 2)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.) & (probability <= 1.)).all()
    assert_allclose(probability.sum(axis=1), 1.)
    assert_allclose(probability_confidence, confidence)


@pytest.mark.parametrize('options', STRATEGIES)
@pytest.mark.parametrize('constant_training', [False, True])
def test_unified_probabilities_are_finite_even_with_tied_training_scores(
        options, constant_training, self_data, query_data):
    training = np.ones((10, 8)) if constant_training else self_data
    detector = make_detector(options, n_detectors=12).fit(training)
    queries, _ = query_data
    X = np.vstack([training, queries])
    probabilities, confidence = detector.predict_proba(
        X, method='unify', return_confidence=True)
    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(confidence).all()
    assert ((probabilities >= 0.) & (probabilities <= 1.)).all()
    assert ((confidence >= 0.) & (confidence <= 1.)).all()
    assert_allclose(probabilities.sum(axis=1), 1.)
    score_order = np.argsort(detector.decision_function(X))
    assert (np.diff(probabilities[score_order, 1]) >= 0.).all()


@pytest.mark.parametrize('strategy', ['fixed', 'variable'])
def test_real_detectors_preserve_self_exclusion_and_geometry(
        strategy, self_data):
    detector = make_detector({'strategy': strategy}).fit(self_data)
    distances = cdist(detector.detectors_,
                      detector.scaler_.transform(self_data))
    nearest = distances.min(axis=1)
    # Every accepted detector ball is outside every self ball.
    assert np.all(nearest >= detector.detector_radii_ + detector.self_radius)
    assert (detector.detector_radii_ > 0).all()
    assert detector.detectors_.shape == (detector.n_detectors_, 8)
    assert np.all(detector.detectors_ >= -detector.sampling_margin)
    assert np.all(detector.detectors_ <= 1 + detector.sampling_margin)
    if strategy == 'fixed':
        assert_allclose(detector.detector_radii_, detector.detector_radius)
    else:
        assert_allclose(detector.detector_radii_,
                        nearest - detector.self_radius)
    assert (detector.decision_scores_ <= -detector.self_radius + 1e-12).all()
    # A detector's own center must be recognized as nonself.
    centers = detector.scaler_.inverse_transform(detector.detectors_)
    assert np.all(detector.decision_function(centers) > 0.)


def longest_matching_run(left, right):
    """Small independent oracle for the contiguous-bit matching definition."""
    runs = itertools.groupby(left == right)
    return max((sum(1 for _ in run) for equal, run in runs if equal),
               default=0)


@pytest.mark.parametrize('rule,threshold', [
    ('hamming', 1), ('rcontiguous', 3), ('rchunk', 3),
])
def test_binary_matching_rules_and_self_exclusion(rule, threshold, self_data):
    detector = make_detector({'strategy': 'binary', 'binary_match': rule,
                              'match_threshold': threshold}).fit(self_data)
    assert detector.detectors_.shape == (detector.n_detectors_, 8)
    assert np.isin(detector.detectors_, [0, 1]).all()
    assert detector.binary_thresholds_.shape == (8,)
    # Enumerate all bit strings, rather than testing only detector centers.
    queries = np.asarray(list(itertools.product([0., 1.], repeat=8)))
    expected = []
    for query in queries:
        if rule == 'hamming':
            mismatch = np.count_nonzero(detector.detectors_ != query, axis=1)
            expected.append(threshold + 0.5 - mismatch.min())
        elif rule == 'rcontiguous':
            longest = max(longest_matching_run(pattern, query)
                          for pattern in detector.detectors_)
            expected.append(longest - threshold + 0.5)
        else:
            mismatches = [np.count_nonzero(pattern[start:start + threshold]
                                           != query[start:start + threshold])
                          for pattern, start in zip(detector.detectors_,
                                                    detector.detector_starts_)]
            expected.append(0.5 - min(mismatches))
    assert_allclose(detector.decision_function(queries), expected)
    assert np.all(detector.decision_scores_ < 0.)
    assert np.all(detector.decision_function(detector.detectors_) > 0.)
    if rule == 'rchunk':
        assert detector.detector_starts_.shape == (detector.n_detectors_,)
        assert (detector.detector_starts_ >= 0).all()
        assert (detector.detector_starts_ <= 8 - threshold).all()


def test_binary_uses_features_beyond_machine_word_length():
    X = np.repeat([[0.], [1.]], 72, axis=1)
    detector = NSA(strategy='binary', n_detectors=1,
                   match_threshold=0, random_state=5).fit(X)
    pattern = detector.detectors_[0].astype(float)
    changed = pattern.copy()
    changed[-1] = 1 - changed[-1]
    scores = detector.decision_function(np.vstack([pattern, changed]))
    assert_allclose(scores, [0.5, -0.5])


@pytest.mark.parametrize('options', STRATEGIES)
@pytest.mark.parametrize('X', [
    np.ones((10, 8)), np.ones((1, 8)), np.tile([0., 1.], (12, 4)),
])
def test_constant_duplicate_and_single_sample_data(options, X):
    detector = make_detector(options, n_detectors=12).fit(X)
    assert np.isfinite(detector.decision_scores_).all()
    assert_allclose(detector.decision_function(X), detector.decision_scores_)
    assert np.all(detector.decision_scores_ < 0.)


@pytest.mark.parametrize('strategy', ['fixed', 'variable'])
def test_one_feature_and_array_like_input(strategy):
    detector = NSA(strategy=strategy, n_detectors=4, random_state=3)
    detector.fit([[0.], [1.]])
    assert detector.n_features_in_ == 1
    assert detector.decision_function([[0.5]]).shape == (1,)


@pytest.mark.parametrize('options', STRATEGIES)
def test_inference_validates_fit_and_feature_count(options, self_data):
    detector = make_detector(options)
    with pytest.raises(NotFittedError):
        detector.decision_function(self_data)
    detector.fit(self_data)
    for X in [np.ones((2, 7)), np.ones((2, 9)), np.ones(8),
              np.full((2, 8), np.nan), np.full((2, 8), np.inf),
              np.empty((0, 8))]:
        with pytest.raises(ValueError):
            detector.decision_function(X)


@pytest.mark.parametrize('X', [
    np.ones(8), np.empty((0, 8)), np.empty((3, 0)),
    np.full((3, 8), np.nan), np.full((3, 8), np.inf),
])
def test_fit_rejects_invalid_data(X):
    with pytest.raises(ValueError):
        NSA().fit(X)


@pytest.mark.parametrize('parameter,value', [
    ('strategy', 'unimplemented'),
    ('binary_match', 'not-a-rule'),
    ('n_detectors', 0), ('n_detectors', -1), ('n_detectors', 2.5),
    ('n_detectors', True),
    ('max_candidates', 0), ('max_candidates', 1.5),
    ('max_candidates', False),
    ('self_radius', -0.1), ('self_radius', np.nan),
    ('self_radius', np.inf),
    ('detector_radius', 0.), ('detector_radius', -0.1),
    ('detector_radius', np.nan), ('detector_radius', np.inf),
    ('sampling_margin', -0.1), ('sampling_margin', np.nan),
    ('sampling_margin', np.inf),
    ('match_threshold', -1), ('match_threshold', 1.5),
    ('match_threshold', True),
    ('random_state', 'invalid'),
])
def test_invalid_hyperparameters(parameter, value, self_data):
    parameters = {parameter: value}
    if parameter == 'match_threshold':
        parameters['strategy'] = 'binary'
    detector = NSA(**parameters)
    with pytest.raises((TypeError, ValueError)):
        detector.fit(self_data)


@pytest.mark.parametrize('rule,threshold', [
    ('hamming', 8), ('hamming', 9), ('rcontiguous', 0),
    ('rcontiguous', 9), ('rchunk', 0), ('rchunk', 9),
])
def test_binary_threshold_bounds(rule, threshold, self_data):
    with pytest.raises(ValueError):
        NSA(strategy='binary', binary_match=rule,
            match_threshold=threshold).fit(self_data)


@pytest.mark.parametrize('options,X', [
    ({'strategy': 'fixed', 'self_radius': 10.}, [[0.], [1.]]),
    ({'strategy': 'variable', 'self_radius': 10.}, [[0.], [1.]]),
    ({'strategy': 'binary', 'match_threshold': 0}, [[0.], [1.]]),
])
def test_no_valid_candidate_raises_instead_of_inserting_fallback(options, X):
    detector = NSA(max_candidates=10, n_detectors=5,
                   random_state=42, **options)
    with pytest.raises(ValueError):
        detector.fit(X)
    with pytest.raises(NotFittedError):
        detector.decision_function(X)


@pytest.mark.parametrize('options', STRATEGIES)
def test_partial_population_honors_budget_and_warns(options, self_data):
    detector = make_detector(options, n_detectors=20, max_candidates=5,
                             random_state=0)
    with pytest.warns(UserWarning):
        detector.fit(self_data)
    assert 0 < detector.n_detectors_ <= 5
    assert detector.n_candidates_ == 5
    assert detector.detectors_.shape[0] == detector.n_detectors_
    assert (detector.decision_scores_ < 0).all()


@pytest.mark.parametrize('strategy', ['fixed', 'variable'])
def test_real_scores_use_fitted_scaling(strategy, self_data, query_data):
    X, _ = query_data
    first = make_detector({'strategy': strategy}).fit(self_data)
    shift, scale = np.arange(8) * 10., np.arange(1., 9.)
    second = make_detector({'strategy': strategy}).fit(
        self_data * scale + shift)
    assert_allclose(first.decision_function(X),
                    second.decision_function(X * scale + shift), atol=1e-12)


def test_refit_replaces_old_training_state(self_data):
    detector = NSA(n_detectors=10, random_state=8).fit(self_data)
    new_data = self_data[:19, :3]
    detector.fit(new_data)
    assert detector.n_features_in_ == 3
    assert detector.detectors_.shape[1] == 3
    assert detector.decision_scores_.shape == (19,)
    assert_allclose(detector.decision_scores_,
                    detector.decision_function(new_data))


def test_failed_refit_does_not_leave_a_usable_old_model(self_data):
    detector = NSA(n_detectors=10, random_state=8).fit(self_data)
    detector.set_params(self_radius=10., max_candidates=3)
    with pytest.raises(ValueError):
        detector.fit(self_data)
    with pytest.raises(NotFittedError):
        detector.decision_function(self_data)
    with pytest.raises(NotFittedError):
        detector.predict(self_data)


def test_refit_switches_representation_and_discards_stale_attributes(
        self_data):
    detector = NSA(n_detectors=12, random_state=8).fit(self_data)
    detector.set_params(strategy='binary')
    detector.fit(self_data)
    assert not hasattr(detector, 'scaler_')
    assert not hasattr(detector, 'detector_radii_')
    assert hasattr(detector, 'binary_thresholds_')
    detector.set_params(strategy='fixed')
    detector.fit(self_data)
    assert not hasattr(detector, 'binary_thresholds_')
    assert not hasattr(detector, 'detector_starts_')
    assert hasattr(detector, 'scaler_')
    assert_allclose(detector.decision_scores_,
                    detector.decision_function(self_data))


@pytest.mark.parametrize('options', STRATEGIES)
def test_integer_seed_repeated_fit_is_reproducible(options, self_data):
    detector = make_detector(options).fit(self_data)
    centers = detector.detectors_.copy()
    scores = detector.decision_scores_.copy()
    detector.fit(self_data)
    assert_array_equal(detector.detectors_, centers)
    assert_allclose(detector.decision_scores_, scores)


@pytest.mark.parametrize('options', STRATEGIES)
@pytest.mark.parametrize('label_kind', ['all-zero', 'all-one', 'multiclass'])
def test_fit_ignores_labels_and_preserves_two_probability_columns(
        options, label_kind, self_data, query_data):
    if label_kind == 'all-zero':
        labels = np.zeros(len(self_data), dtype=int)
    elif label_kind == 'all-one':
        labels = np.ones(len(self_data), dtype=int)
    else:
        labels = np.arange(len(self_data)) % 3
    baseline = make_detector(options).fit(self_data)
    supplied = make_detector(options).fit(self_data, y=labels)
    assert supplied._classes == 2
    assert_array_equal(supplied.detectors_, baseline.detectors_)
    assert_allclose(supplied.decision_scores_, baseline.decision_scores_)
    assert_array_equal(supplied.labels_, baseline.labels_)
    queries, _ = query_data
    assert_allclose(supplied.decision_function(queries),
                    baseline.decision_function(queries))
    for method in ['linear', 'unify']:
        probabilities = supplied.predict_proba(queries, method=method)
        assert probabilities.shape == (len(queries), 2)
        assert np.isfinite(probabilities).all()
        assert_allclose(probabilities.sum(axis=1), 1.)
        assert_allclose(probabilities,
                        baseline.predict_proba(queries, method=method))


@pytest.mark.parametrize('contamination', [0., -0.1, 0.51, np.nan])
def test_invalid_contamination_is_checked_at_fit(contamination, self_data):
    detector = NSA(contamination=contamination)
    with pytest.raises(ValueError):
        detector.fit(self_data)


def test_overflowing_binary_quantization_raises_clear_error():
    # Finite input can still overflow in median's mean of the middle pair.
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='overflow'):
            NSA(strategy='binary').fit(np.full((2, 3), 1e308))


def test_overflowing_candidate_distances_raise_clear_error(self_data):
    with pytest.raises(ValueError, match='overflow'):
        NSA(sampling_margin=1e200).fit(self_data)


@pytest.mark.parametrize('strategy', ['fixed', 'variable'])
def test_overflowing_inference_distances_raise_clear_error(
        strategy, self_data):
    detector = make_detector({'strategy': strategy}).fit(self_data)
    with pytest.raises(ValueError, match='overflow'):
        detector.decision_function(np.full((2, 8), 1e200))
