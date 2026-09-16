# -*- coding: utf-8 -*-
"""Compare real NSA mechanisms on a fixed train-only two-feature projection.

This complements nsa_example.py; it does not replace its raw-feature results.
Two PCA components are chosen before evaluation to accommodate the geometric
strategies' dimensional limits. This is a novelty-detection demonstration,
not a clinical model or a general benchmark ranking. No held-out tuning.
"""
# Author: Kishor Datta Gupta
# License: BSD 2 clause

from time import perf_counter

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyod.models.iforest import IForest
from pyod.models.nsa import NSA


def main():
    X, target = load_breast_cancer(return_X_y=True)
    y = (target == 0).astype(int)
    train, test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42)
    transform = make_pipeline(StandardScaler(), PCA(n_components=2,
                                                    svd_solver='full'))
    normal = transform.fit_transform(train[y_train == 0])
    query = transform.transform(test)
    strategies = ['fixed', 'variable', 'coverage', 'grid', 'hierarchical',
                  'voronoi', 'deterministic', 'suppressed', 'dual', 'annealed']
    models = [(name, NSA(strategy=name, n_detectors=200,
                         max_candidates=20000, random_state=42))
              for name in strategies]
    models.append(('IForest', IForest(random_state=42)))
    print('Wisconsin breast cancer: fixed train-only StandardScaler + PCA(2)')
    print('Seed 42; train {} normal; test {} normal, {} novel.'.format(
        len(normal), int((y_test == 0).sum()), int(y_test.sum())))
    print('Separate representation from nsa_example.py; no test tuning.')
    print('{:15s} {:>8s} {:>8s} {:>9s} {:>9s}'.format(
        'Detector', 'AUROC', 'AP', 'fit (s)', 'score (s)'))
    for name, model in models:
        start = perf_counter()
        model.fit(normal)
        fitted = perf_counter()
        scores = model.decision_function(query)
        scored = perf_counter()
        print('{:15s} {:8.4f} {:8.4f} {:9.3f} {:9.3f}'.format(
            name, roc_auc_score(y_test, scores),
            average_precision_score(y_test, scores),
            fitted - start, scored - fitted))
        if name == 'coverage':
            print('  coverage certified={}, lower bound={:.4f}, tests={}'
                  .format(model.coverage_reached_,
                          model.coverage_lower_bound_,
                          model.coverage_test_count_))


if __name__ == '__main__':
    main()
