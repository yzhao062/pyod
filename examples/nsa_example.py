# -*- coding: utf-8 -*-
"""Compare negative selection strategies with IForest on a real dataset.

Run ``python examples/nsa_example.py`` from an installed PyOD checkout.
The Wisconsin breast cancer dataset is bundled with scikit-learn, so this
example needs no download. It is used as a reproducible *novelty-detection
demonstration*: fit on the benign class only, then score held-out samples
from both classes. The class labels define this example's novelty task;
the scores are not medical predictions or a general benchmark ranking.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

from time import perf_counter

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pyod.models.iforest import IForest
from pyod.models.nsa import NSA


def main():
    """Fit all supported strategies and report held-out ranking metrics."""
    X, target = load_breast_cancer(return_X_y=True)
    # sklearn encodes benign as 1; PyOD uses 1 for the anomalous class.
    y = (target == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42)
    X_self = X_train[y_train == 0]
    common = dict(n_detectors=200, max_candidates=20000, random_state=42)
    models = [
        ('NSA fixed', NSA(strategy='fixed', **common)),
        ('NSA variable', NSA(strategy='variable', **common)),
        ('NSA binary hamming', NSA(strategy='binary', **common)),
        ('NSA binary contiguous', NSA(strategy='binary',
                                      binary_match='rcontiguous',
                                      match_threshold=12, **common)),
        ('NSA binary chunk', NSA(strategy='binary', binary_match='rchunk',
                                 match_threshold=12, **common)),
        ('IForest', IForest(random_state=42)),
    ]
    print('Wisconsin breast cancer: held-out novelty demonstration')
    print('Seed: 42; training uses only the normal class.')
    print('Training: {} normal samples; test: {} normal, {} novel.'.format(
        len(X_self), int((y_test == 0).sum()), int(y_test.sum())))
    print('Preprocessing is fitted on training data only.')
    print('AUROC/AP evaluate raw scores; contamination sets the label cutoff.')
    print('{:25s} {:>8s} {:>8s} {:>9s} {:>9s}'.format(
        'Detector', 'AUROC', 'AP', 'fit (s)', 'score (s)'))
    for name, model in models:
        start = perf_counter()
        model.fit(X_self)
        fitted = perf_counter()
        scores = model.decision_function(X_test)
        scored = perf_counter()
        print('{:25s} {:8.4f} {:8.4f} {:9.3f} {:9.3f}'.format(
            name, roc_auc_score(y_test, scores),
            average_precision_score(y_test, scores),
            fitted - start, scored - fitted))


if __name__ == '__main__':
    main()
