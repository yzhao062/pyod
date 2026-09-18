# -*- coding: utf-8 -*-
"""Tests for pyod.models.nsa."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from pyod.models.nsa import (
    NegativeSelection,
    BinaryNSA,
    RNSA,
    RRNSA,
    VDetector,
    GridNSA,
    GFRNSA,
    MatrixNSA,
    ANSA,
    EvoSeedRNSA,
    ORNSA,
    OptimizedNSA,
    FtNSA,
    IVRNSA,
    CBNSA,
    PRR2NSA,
    DENSA,
    AntigenNSA,
    NSADE,
    NSAPSO,
    IORNSA,
    BIORVNSA,
    HNSAIDSA,
    NSAII,
    OALFBNSA,
    FBNSA,
    MNSA,
    NSNAD,
    RENNSA,
    AINSA,
    ODNSA,
    CNSA,
    VORNSA,
)


def _dataset(seed=7):
    rng = np.random.default_rng(seed)
    X_in, _ = make_blobs(
        n_samples=450,
        centers=[[-1.0, -1.0], [1.0, 1.0]],
        cluster_std=[0.25, 0.30],
        random_state=seed,
    )
    X_out = rng.uniform(low=-3.0, high=3.0, size=(90, 2))
    X_out = X_out[np.linalg.norm(X_out, axis=1) > 2.0][:40]
    idx = rng.permutation(len(X_in))
    X_train = X_in[idx[:300]]
    X_test = np.vstack([X_in[idx[300:]], X_out])
    y_test = np.r_[np.zeros(len(X_in) - 300, dtype=int), np.ones(len(X_out), dtype=int)]
    contamination = len(X_out) / len(X_test)
    return X_train, X_test, y_test, contamination


def _models(contamination):
    return [
        NegativeSelection(contamination=contamination, variant="vdetector", n_detectors=32, random_state=1),
        BinaryNSA(contamination=contamination, n_detectors=32, random_state=1),
        RNSA(contamination=contamination, n_detectors=32, random_state=1),
        RRNSA(contamination=contamination, n_detectors=32, random_state=1),
        VDetector(contamination=contamination, n_detectors=32, random_state=1),
        GridNSA(contamination=contamination, n_detectors=32, random_state=1),
        GFRNSA(contamination=contamination, n_detectors=32, random_state=1),
        MatrixNSA(contamination=contamination, n_detectors=32, random_state=1),
        ANSA(contamination=contamination, n_detectors=32, random_state=1),
        EvoSeedRNSA(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        ORNSA(contamination=contamination, n_detectors=32, random_state=1),
        OptimizedNSA(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        FtNSA(contamination=contamination, n_detectors=32, random_state=1),
        IVRNSA(contamination=contamination, n_detectors=32, random_state=1),
        CBNSA(contamination=contamination, n_detectors=32, n_clusters=6, random_state=1),
        PRR2NSA(contamination=contamination, n_detectors=32, n_clusters=6, random_state=1),
        DENSA(contamination=contamination, n_detectors=32, random_state=1),
        AntigenNSA(contamination=contamination, n_detectors=32, random_state=1),
        NSADE(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        NSAPSO(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        IORNSA(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        BIORVNSA(contamination=contamination, n_detectors=32, random_state=1),
        HNSAIDSA(contamination=contamination, n_detectors=32, random_state=1),
        NSAII(contamination=contamination, n_detectors=32, random_state=1),
        OALFBNSA(contamination=contamination, n_detectors=32, random_state=1),
        FBNSA(contamination=contamination, n_detectors=32, random_state=1),
        MNSA(contamination=contamination, n_detectors=32, n_estimators=3, random_state=1),
        NSNAD(contamination=contamination, n_detectors=32, feature_subsample=1.0, random_state=1),
        RENNSA(contamination=contamination, n_detectors=32, random_state=1),
        AINSA(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        ODNSA(contamination=contamination, n_detectors=32, optimization_iter=3, random_state=1),
        CNSA(contamination=contamination, n_detectors=32, n_clusters=6, optimization_iter=3, random_state=1),
        VORNSA(contamination=contamination, n_detectors=32, random_state=1),
    ]


def test_all_nsa_variants_fit_score_predict():
    X_train, X_test, y_test, contamination = _dataset()
    for model in _models(contamination):
        model.fit(X_train)
        scores = model.decision_function(X_test)
        labels = model.predict(X_test)
        assert scores.shape == (X_test.shape[0],)
        assert labels.shape == (X_test.shape[0],)
        assert np.all(np.isfinite(scores))
        assert set(np.unique(labels)).issubset({0, 1})
        # NSA variants should rank the synthetic remote anomalies above inliers.
        assert roc_auc_score(y_test, scores) >= 0.60


def test_online_feedback_partial_fit():
    X_train, X_test, _, contamination = _dataset()
    model = OALFBNSA(contamination=contamination, n_detectors=32, random_state=1)
    model.fit(X_train[:200])
    before = model.decision_function(X_test[:10])
    model.partial_fit(X_train[200:260])
    after = model.decision_function(X_test[:10])
    assert before.shape == after.shape
    assert np.all(np.isfinite(after))
