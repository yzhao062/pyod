# -*- coding: utf-8 -*-
"""Benchmark the NSA family on a synthetic anomaly-detection dataset."""

import json
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from pyod.models.nsa import VDetector, GridNSA, MNSA, BinaryNSA, DENSA

rng = np.random.default_rng(7)
X_in, _ = make_blobs(n_samples=600, centers=[[-1.0, -1.0], [1.0, 1.0]], cluster_std=[0.25, 0.30], random_state=7)
X_out = rng.uniform(low=-3.0, high=3.0, size=(100, 2))
X_out = X_out[np.linalg.norm(X_out, axis=1) > 2.0][:60]
idx = rng.permutation(len(X_in))
X_train = X_in[idx[:400]]
X_test = np.vstack([X_in[idx[400:]], X_out])
y_test = np.r_[np.zeros(len(X_in)-400, dtype=int), np.ones(len(X_out), dtype=int)]
contamination = len(X_out) / len(X_test)

models = [
    VDetector(contamination=contamination, n_detectors=64, random_state=42),
    GridNSA(contamination=contamination, n_detectors=64, n_grid=10, random_state=42),
    MNSA(contamination=contamination, n_detectors=64, n_estimators=5, random_state=42),
    BinaryNSA(contamination=contamination, n_detectors=64, random_state=42),
    DENSA(contamination=contamination, n_detectors=64, random_state=42),
]

for model in models:
    model.fit(X_train)
    scores = model.decision_function(X_test)
    pred = model.predict(X_test)
    print(json.dumps({
        "model": model.__class__.__name__,
        "roc_auc": round(roc_auc_score(y_test, scores), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
    }))
