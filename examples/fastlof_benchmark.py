# -*- coding: utf-8 -*-
"""Measure FastLOF quality, memory, and fit plus novelty-query time.

Run from the repository root. A baseline module path is optional and must
point to an earlier fastlof.py source. Memory sampling requires psutil.
"""
# License: BSD 2 clause

import argparse
import importlib.util
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numba
import numpy as np
import scipy
import sklearn
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyod.models.fastlof import FastLOF


def quality(y, scores):
    return {'auc': roc_auc_score(y, scores),
            'ap': average_precision_score(y, scores)}


def measure(factory, train, query, y_train, y_query):
    """Measure one estimator; sample process RSS every 5 milliseconds."""
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    stop = threading.Event()
    peak = [process.memory_info().rss if process is not None else 0]

    def sample():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True) if process else None
    if sampler:
        sampler.start()
    try:
        clf = factory()
        start = time.perf_counter()
        clf.fit(train)
        fit_seconds = time.perf_counter() - start
        is_reference = isinstance(clf, LocalOutlierFactor)
        score = ((lambda X: -clf.score_samples(X)) if is_reference
                 else clf.decision_function)
        start = time.perf_counter()
        scores = score(query)
        query_seconds = time.perf_counter() - start
        start = time.perf_counter()
        score(query)
        repeated_seconds = time.perf_counter() - start
        train_scores = (-clf.negative_outlier_factor_ if is_reference
                        else clf.decision_scores_)
        return dict(fit_seconds=fit_seconds, first_query_seconds=query_seconds,
                    repeated_query_seconds=repeated_seconds,
                    total_seconds=fit_seconds + query_seconds,
                    peak_process_rss_mib=peak[0] / 2**20 if process else None,
                    train=quality(y_train, train_scores),
                    query=quality(y_query, scores))
    finally:
        stop.set()
        if sampler:
            sampler.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('notebooks/data'))
    parser.add_argument('--output', type=Path, default=Path('fastlof_benchmark.json'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    implementations = [('fixed', FastLOF)]
    if args.baseline:
        spec = importlib.util.spec_from_file_location(
            'pyod.models._fastlof_baseline', args.baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        implementations.insert(0, ('original', module.FastLOF))
    result = dict(environment=dict(python=platform.python_version(),
                                   platform=platform.platform(),
                                   numpy=np.__version__, scipy=scipy.__version__,
                                   sklearn=sklearn.__version__, numba=numba.__version__,
                                   threads=args.threads),
                  first_use_seconds={}, runs=[])
    rng = np.random.RandomState(0)
    warm = rng.normal(size=(100, 9))
    for label, cls in implementations:
        start = time.perf_counter()
        cls(random_state=42).fit(warm).decision_function(warm[:10])
        result['first_use_seconds'][label] = time.perf_counter() - start
    # Initialize sklearn's brute query backend before measured runs too.
    LocalOutlierFactor(novelty=True, algorithm='brute').fit(warm).score_samples(
        warm[:10])
    for dataset in ['pima', 'cardio', 'shuttle']:
        data = loadmat(args.data_dir / (dataset + '.mat'))
        X = np.asarray(data['X'], dtype=np.float64)
        y = data['y'].ravel().astype(int)
        train, query, y_train, y_query = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        scaler = StandardScaler().fit(train)
        train, query = scaler.transform(train), scaler.transform(query)
        for seed in [0, 17, 42]:
            factories = [(label, lambda cls=cls: cls(
                n_neighbors=20, threshold=1.1, random_state=seed,
                algorithm='auto', n_jobs=args.threads))
                         for label, cls in implementations]
            factories += [(f'exact_{algorithm}', lambda algorithm=algorithm:
                           LocalOutlierFactor(n_neighbors=20, novelty=True,
                                              algorithm=algorithm,
                                              n_jobs=args.threads))
                          for algorithm in ['auto', 'brute']]
            for label, factory in factories:
                row = dict(dataset=dataset, seed=seed, implementation=label,
                           train_shape=train.shape, query_shape=query.shape,
                           **measure(factory, train, query, y_train, y_query))
                result['runs'].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
