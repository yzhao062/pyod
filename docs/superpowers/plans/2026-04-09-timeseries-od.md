# TimeSeriesOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 time-series anomaly detection algorithms to PyOD as first-class BaseDetector subclasses, with shared utilities, tests, and ADEngine knowledge base integration.

**Architecture:** Shared TS utilities (`_ts_utils.py`) provide input validation, sliding window extraction, score mapping, and channel aggregation. Each detector is a separate file (`ts_*.py`) inheriting BaseDetector, following the masked-score fit workflow from the spec. Classical methods (5) use numpy/scipy only. Deep methods (2) use PyTorch. All integrate with ADEngine via `algorithms.json` entries.

**Tech Stack:** Python 3.8+, numpy, scipy, PyTorch (optional for LSTMAD + AnomalyTransformer), PyOD BaseDetector.

**Spec:** `docs/superpowers/specs/2026-04-09-timeseries-od-design.md` (v2, 4 review rounds)

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `pyod/models/_ts_utils.py` | Shared: input validation, sliding windows, score mapping, channel aggregation |
| `pyod/models/ts_od.py` | `TimeSeriesOD` windowed bridge |
| `pyod/models/ts_matrix_profile.py` | `MatrixProfile` (STOMP algorithm) |
| `pyod/models/ts_spectral_residual.py` | `SpectralResidual` (FFT saliency) |
| `pyod/models/ts_kshape.py` | `KShape` (experimental, k-Shape clustering) |
| `pyod/models/ts_sand.py` | `SAND` (experimental, streaming) |
| `pyod/models/ts_lstm.py` | `LSTMAD` (LSTM prediction error + Mahalanobis) |
| `pyod/models/ts_anomaly_transformer.py` | `AnomalyTransformer` (attention discrepancy) |
| `pyod/test/test_ts_od.py` | Tests for all TS detectors |

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/knowledge/algorithms.json` | Add 7 entries, update TimeSeriesOD class_path |
| `pyod/utils/knowledge/routing_rules.json` | Update TS routing recommendations |
| `pyod/utils/ad_engine.py` | Catch NotImplementedError for transductive detectors in run_detection |
| `docs/zreferences.bib` | Add 7 BibTeX entries |
| `CHANGES.txt` | Add TS-AD entry |

---

## Task 1: Shared TS Utilities + BibTeX

**Files:**
- Create: `pyod/models/_ts_utils.py`
- Modify: `docs/zreferences.bib`

This task creates the foundation all TS detectors depend on.

- [ ] **Step 1: Create `_ts_utils.py`**

```python
# pyod/models/_ts_utils.py
# -*- coding: utf-8 -*-
"""Shared utilities for time series anomaly detection models."""

import numpy as np


def validate_ts_input(X):
    """Validate and reshape time series input.

    Parameters
    ----------
    X : array-like
        Time series data. 1D (n_timestamps,) or 2D (n_timestamps, n_channels).

    Returns
    -------
    X : np.ndarray of shape (n_timestamps, n_channels)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("Expected 1D or 2D input, got %dD" % X.ndim)
    return X


def sliding_windows(X, window_size, step=1):
    """Extract sliding windows from a time series.

    Parameters
    ----------
    X : np.ndarray of shape (n_timestamps, n_channels)
    window_size : int
    step : int

    Returns
    -------
    windows : np.ndarray of shape (n_windows, window_size * n_channels)
    """
    n_timestamps, n_channels = X.shape
    n_windows = max(0, (n_timestamps - window_size) // step + 1)
    windows = np.empty((n_windows, window_size * n_channels))
    for i in range(n_windows):
        start = i * step
        windows[i] = X[start:start + window_size].ravel()
    return windows


def map_scores_to_timestamps(window_scores, window_size, step,
                              n_timestamps, aggregation='max'):
    """Map window-level scores back to per-timestamp scores.

    Parameters
    ----------
    window_scores : np.ndarray of shape (n_windows,)
    window_size : int
    step : int
    n_timestamps : int
    aggregation : str, 'max' or 'mean'

    Returns
    -------
    scores : np.ndarray of shape (n_timestamps,)
    valid_mask : np.ndarray of shape (n_timestamps,), dtype=bool
    """
    scores = np.full(n_timestamps, np.nan)
    counts = np.zeros(n_timestamps)

    for i, score in enumerate(window_scores):
        start = i * step
        end = min(start + window_size, n_timestamps)
        if aggregation == 'max':
            scores[start:end] = np.fmax(
                np.nan_to_num(scores[start:end], nan=-np.inf), score)
        else:  # mean
            scores[start:end] = np.nansum(
                np.column_stack([
                    np.nan_to_num(scores[start:end], nan=0.0),
                    np.full(end - start, score)
                ]), axis=1)
            counts[start:end] += 1

    if aggregation == 'mean':
        mask = counts > 0
        scores[mask] /= counts[mask]

    valid_mask = ~np.isnan(scores)
    return scores, valid_mask


def aggregate_channel_scores(per_channel_scores, method='max'):
    """Aggregate per-channel anomaly scores.

    Z-normalizes each channel before aggregation to prevent
    high-variance channels from dominating.

    Parameters
    ----------
    per_channel_scores : list of np.ndarray, each shape (n_timestamps,)
    method : str, 'max' or 'mean'

    Returns
    -------
    scores : np.ndarray of shape (n_timestamps,)
    """
    # Z-normalize each channel
    normalized = []
    for ch_scores in per_channel_scores:
        mu = np.mean(ch_scores)
        sigma = np.std(ch_scores)
        if sigma > 0:
            normalized.append((ch_scores - mu) / sigma)
        else:
            normalized.append(ch_scores - mu)

    stacked = np.column_stack(normalized)
    if method == 'max':
        return np.max(stacked, axis=1)
    else:
        return np.mean(stacked, axis=1)
```

- [ ] **Step 2: Add BibTeX entries to `docs/zreferences.bib`**

Append these entries to the end of `docs/zreferences.bib`:

```bibtex
@inproceedings{yeh2016matrix,
  title={Matrix Profile I: All Pairs Similarity Joins for Time Series Subsequences},
  author={Yeh, Chin-Chia Michael and Zhu, Yan and Ulanova, Liudmila and Begum, Nusrat and Ding, Yifei and Dau, Hoang Anh and Silva, Diego Furtado and Mueen, Abdullah and Keogh, Eamonn},
  booktitle={2016 IEEE 16th International Conference on Data Mining (ICDM)},
  pages={1317--1322},
  year={2016},
  organization={IEEE}
}

@inproceedings{ren2019time,
  title={Time-Series Anomaly Detection Service at Microsoft},
  author={Ren, Hansheng and Xu, Bixiong and Wang, Yujing and Yi, Chao and Huang, Congrui and Kou, Xiaoyu and Xing, Tony and Yang, Mao and Tong, Jie and Zhang, Qi},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={3009--3017},
  year={2019}
}

@inproceedings{paparrizos2015kshape,
  title={k-Shape: Efficient and Accurate Clustering of Time Series},
  author={Paparrizos, John and Gravano, Luis},
  booktitle={Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data},
  pages={1855--1870},
  year={2015}
}

@article{boniol2021sand,
  title={SAND: Streaming Subsequence Anomaly Detection},
  author={Boniol, Paul and Paparrizos, John and Palpanas, Themis and Franklin, Michael J.},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={10},
  pages={1717--1729},
  year={2021}
}

@inproceedings{malhotra2015long,
  title={Long Short Term Memory Networks for Anomaly Detection in Time Series},
  author={Malhotra, Pankaj and Vig, Lovekesh and Shroff, Gautam and Agarwal, Puneet},
  booktitle={European Symposium on Artificial Neural Networks (ESANN)},
  year={2015}
}

@inproceedings{xu2022anomaly,
  title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
  author={Xu, Jiehui and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}

@inproceedings{liu2024tsb,
  title={TSB-AD: The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark},
  author={Liu, Qinghua and Paparrizos, John},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

- [ ] **Step 3: Verify utilities are importable**

Run: `python -c "from pyod.models._ts_utils import validate_ts_input, sliding_windows, map_scores_to_timestamps, aggregate_channel_scores; print('OK')"`
Expected: OK

- [ ] **Step 4: Propose commit**

```
feat: add shared TS utilities and BibTeX entries for time series detectors
```

---

## Task 2: TimeSeriesOD (Windowed Bridge)

**Files:**
- Create: `pyod/models/ts_od.py`
- Create: `pyod/test/test_ts_od.py`

The implementer MUST read the spec Section 6.2 and the EmbeddingOD pattern (`pyod/models/embedding.py`) for reference on how to delegate to inner detectors.

- [ ] **Step 1: Write tests**

Create `pyod/test/test_ts_od.py` with tests for TimeSeriesOD:

```python
# pyod/test/test_ts_od.py
# -*- coding: utf-8 -*-
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_od import TimeSeriesOD


class TestTimeSeriesOD(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(500)
        self.X_test = rng.randn(200)
        self.X_multi = rng.randn(500, 3)

    def test_fit_univariate(self):
        clf = TimeSeriesOD(window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert hasattr(clf, 'decision_scores_')
        assert len(clf.decision_scores_) == 500
        assert hasattr(clf, 'labels_')

    def test_fit_multivariate(self):
        clf = TimeSeriesOD(window_size=20)
        clf.fit(self.X_multi)
        assert len(clf.decision_scores_) == 500

    def test_decision_function(self):
        clf = TimeSeriesOD(window_size=20)
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert len(scores) == 200

    def test_predict(self):
        clf = TimeSeriesOD(window_size=20)
        clf.fit(self.X_train)
        labels = clf.predict(self.X_test)
        assert len(labels) == 200
        assert set(labels).issubset({0, 1})

    def test_string_detector(self):
        clf = TimeSeriesOD(detector='ECOD', window_size=20)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_detector_instance(self):
        from pyod.models.iforest import IForest
        clf = TimeSeriesOD(detector=IForest(), window_size=20)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            clf = TimeSeriesOD(window_size=100)
            clf.fit(np.random.randn(50))

    def test_score_aggregation_mean(self):
        clf = TimeSeriesOD(window_size=20, score_aggregation='mean')
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_step_parameter(self):
        clf = TimeSeriesOD(window_size=20, step=5)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Implement TimeSeriesOD**

Create `pyod/models/ts_od.py`. The implementer must:
1. Read `pyod/models/embedding.py` for the `_DETECTOR_SHORTCUTS` and `resolve_detector` pattern
2. Follow the masked-score fit workflow from spec Section 4
3. Use `_ts_utils.py` for input validation, windowing, and score mapping

The class should:
- Inherit `BaseDetector`
- Accept `detector` (str or BaseDetector instance), `window_size`, `step`, `score_aggregation`, `contamination`
- In `fit()`: validate input, create sliding windows, fit inner detector on window matrix, map scores back to timestamps using `map_scores_to_timestamps`, follow the masked-score workflow
- In `decision_function(X_test)`: window test data, score with fitted inner detector, map back
- Implement `_get_min_length()` returning `self.window_size`

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py -v`
Expected: All PASS

- [ ] **Step 4: Propose commit**

```
feat: add TimeSeriesOD windowed bridge detector
```

---

## Task 3: MatrixProfile

**Files:**
- Create: `pyod/models/ts_matrix_profile.py`
- Modify: `pyod/test/test_ts_od.py` (add test class)

The implementer MUST understand the STOMP algorithm:
- Z-normalized Euclidean distance via dot products (MASS for first row, incremental QT updates)
- Exclusion zone = window_size / 4
- Constant subsequences (std=0) get distance = infinity
- Reference: Yeh et al., ICDM 2016

- [ ] **Step 1: Write tests**

Add to `pyod/test/test_ts_od.py`:

```python
from pyod.models.ts_matrix_profile import MatrixProfile


class TestMatrixProfile(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(300)

    def test_fit(self):
        clf = MatrixProfile(window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')

    def test_multivariate(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 3)
        clf = MatrixProfile(window_size=20)
        clf.fit(X)
        assert len(clf.decision_scores_) == 300

    def test_transductive_no_decision_function(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(np.random.randn(100))

    def test_transductive_no_predict(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        with self.assertRaises(NotImplementedError):
            clf.predict(np.random.randn(100))

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            clf = MatrixProfile(window_size=50)
            clf.fit(np.random.randn(30))

    def test_anomaly_scores_positive(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        assert np.all(clf.decision_scores_ >= 0)
```

- [ ] **Step 2: Implement MatrixProfile**

Create `pyod/models/ts_matrix_profile.py`. The implementer must:
1. Implement STOMP: precompute rolling mean/std via cumsum, first-row QT via FFT, incremental QT update
2. Exclusion zone = window_size // 4
3. Handle constant subsequences (std < 1e-10) by setting distance to infinity
4. Map subsequence scores to timestamps via `map_scores_to_timestamps`
5. Override `decision_function`, `predict`, `predict_proba`, `predict_confidence` to raise `NotImplementedError`
6. Follow the masked-score fit workflow

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py::TestMatrixProfile -v`
Expected: All PASS

- [ ] **Step 4: Propose commit**

```
feat: add MatrixProfile detector (STOMP, transductive)
```

---

## Task 4: SpectralResidual

**Files:**
- Create: `pyod/models/ts_spectral_residual.py`
- Modify: `pyod/test/test_ts_od.py` (add test class)

The implementer MUST understand the SR algorithm:
- FFT → log amplitude → smooth with averaging kernel (size q=3) → spectral residual → inverse FFT → saliency magnitude
- Reference: Ren et al., KDD 2019

- [ ] **Step 1: Write tests**

Add to `pyod/test/test_ts_od.py`:

```python
from pyod.models.ts_spectral_residual import SpectralResidual


class TestSpectralResidual(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(500)
        self.X_test = rng.randn(200)

    def test_fit(self):
        clf = SpectralResidual(contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_decision_function(self):
        clf = SpectralResidual()
        clf.fit(self.X_train)
        scores = clf.decision_function(self.X_test)
        assert len(scores) == 200

    def test_multivariate(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 3)
        clf = SpectralResidual()
        clf.fit(X)
        assert len(clf.decision_scores_) == 500

    def test_scores_nonnegative(self):
        clf = SpectralResidual()
        clf.fit(self.X_train)
        assert np.all(clf.decision_scores_ >= 0)

    def test_dense_scores_no_nan(self):
        """SR produces one score per timestamp (no gaps)."""
        clf = SpectralResidual()
        clf.fit(self.X_train)
        assert not np.any(np.isnan(clf.decision_scores_))
```

- [ ] **Step 2: Implement SpectralResidual**

Create `pyod/models/ts_spectral_residual.py`. The implementer must:
1. FFT, log amplitude, smooth with uniform kernel of size `score_window` (default 3)
2. Spectral residual = original log amplitude - smoothed
3. Inverse FFT → saliency magnitude
4. For multivariate: per-channel SR, aggregate with `aggregate_channel_scores`
5. Dense method: `valid_mask = np.ones(n_timestamps, dtype=bool)`
6. `_get_min_length()` returns `max(self.score_window, 2)`

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py::TestSpectralResidual -v`
Expected: All PASS

- [ ] **Step 4: Propose commit**

```
feat: add SpectralResidual detector (FFT saliency)
```

---

## Task 5: KShape (Experimental)

**Files:**
- Create: `pyod/models/ts_kshape.py`
- Modify: `pyod/test/test_ts_od.py`

The implementer MUST understand k-Shape:
- SBD (shape-based distance) via cross-correlation (FFT)
- Centroid update via eigenvalue decomposition of `X^T * X`
- Z-normalization of all subsequences
- Reference: Paparrizos & Gravano, SIGMOD 2015

- [ ] **Step 1: Write tests**

```python
from pyod.models.ts_kshape import KShape


class TestKShape(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(300)

    def test_fit(self):
        clf = KShape(n_clusters=3, window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300

    def test_decision_function(self):
        clf = KShape(n_clusters=3, window_size=20)
        clf.fit(self.X_train)
        scores = clf.decision_function(np.random.RandomState(0).randn(200))
        assert len(scores) == 200

    def test_multivariate(self):
        X = np.random.RandomState(42).randn(300, 2)
        clf = KShape(n_clusters=3, window_size=20)
        clf.fit(X)
        assert len(clf.decision_scores_) == 300
```

- [ ] **Step 2: Implement KShape**

The implementer must:
1. Extract sliding-window subsequences, z-normalize each
2. Implement SBD via FFT cross-correlation: `SBD(x,y) = 1 - max(CC(x,y)) / (||x|| * ||y||)`
3. Implement centroid update: largest eigenvector of `(I - 1/n * X^T * X)`
4. Lloyd's iteration: assign to nearest centroid, update centroids
5. Anomaly score = SBD to nearest centroid
6. Map to timestamps, per-channel aggregate for multivariate

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py::TestKShape -v`

- [ ] **Step 4: Propose commit**

```
feat: add KShape detector (experimental, shape-based clustering)
```

---

## Task 6: SAND (Experimental)

**Files:**
- Create: `pyod/models/ts_sand.py`
- Modify: `pyod/test/test_ts_od.py`

The implementer MUST understand SAND:
- Streaming extension of k-Shape-based detection
- Distance to nearest centroid as anomaly score
- Periodic re-clustering for drift adaptation
- Reference: Boniol et al., VLDB 2021

- [ ] **Step 1: Write tests**

```python
from pyod.models.ts_sand import SAND


class TestSAND(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(500)

    def test_fit(self):
        clf = SAND(n_clusters=3, window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500

    def test_decision_function(self):
        clf = SAND(n_clusters=3, window_size=20)
        clf.fit(self.X_train)
        scores = clf.decision_function(np.random.RandomState(0).randn(200))
        assert len(scores) == 200

    def test_drift_adaptation(self):
        """Scores should reflect recent patterns, not just initial ones."""
        clf = SAND(n_clusters=3, window_size=20, batch_size=50)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 500
```

- [ ] **Step 2: Implement SAND**

Simplified PyOD adaptation:
1. Extract subsequences, z-normalize
2. Initialize k centroids using k-Shape on first `batch_size` subsequences
3. Process remaining subsequences: score = SBD to nearest centroid
4. Every `batch_size` subsequences, update centroids: weighted average of current centroid and recent batch centroid (weight = `alpha`)
5. Map to timestamps, per-channel aggregate

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py::TestSAND -v`

- [ ] **Step 4: Propose commit**

```
feat: add SAND detector (experimental, streaming with drift adaptation)
```

---

## Task 7: LSTMAD

**Files:**
- Create: `pyod/models/ts_lstm.py`
- Modify: `pyod/test/test_ts_od.py`

The implementer MUST understand LSTMAD:
- Stacked LSTM predicts next timestep from lookback window
- Error distribution: multivariate Gaussian fitted on training errors
- Anomaly score: Mahalanobis distance of test errors
- Reference: Malhotra et al., ESANN 2015

- [ ] **Step 1: Write tests**

```python
import unittest


class TestLSTMAD(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.X_train = self.rng.randn(300)

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_fit(self):
        from pyod.models.ts_lstm import LSTMAD
        clf = LSTMAD(window_size=20, epochs=2, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_decision_function(self):
        from pyod.models.ts_lstm import LSTMAD
        clf = LSTMAD(window_size=20, epochs=2)
        clf.fit(self.X_train)
        scores = clf.decision_function(self.rng.randn(200))
        assert len(scores) == 200

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_multivariate(self):
        from pyod.models.ts_lstm import LSTMAD
        X = self.rng.randn(300, 3)
        clf = LSTMAD(window_size=20, epochs=2)
        clf.fit(X)
        assert len(clf.decision_scores_) == 300

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_causal_first_window_filled(self):
        """First window_size timestamps should be threshold-filled."""
        from pyod.models.ts_lstm import LSTMAD
        clf = LSTMAD(window_size=20, epochs=2)
        clf.fit(self.X_train)
        # First window_size scores should be at threshold
        assert np.allclose(clf.decision_scores_[:20], clf.threshold_)
```

Add a helper at the top of the test file:

```python
def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False
```

- [ ] **Step 2: Implement LSTMAD**

The implementer must:
1. Build a stacked LSTM model (PyTorch)
2. Create (input, target) pairs: input = `X[t-w:t]`, target = `X[t]`
3. Train with MSE loss
4. Compute prediction errors on training data
5. Fit multivariate Gaussian on errors (mean mu, covariance Sigma)
6. Anomaly score = Mahalanobis distance: `(e - mu)^T * Sigma_inv * (e - mu)`
7. Follow causal scoring policy: first `window_size` timestamps are invalid
8. Follow masked-score workflow

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py::TestLSTMAD -v`

- [ ] **Step 4: Propose commit**

```
feat: add LSTMAD detector (LSTM prediction error + Mahalanobis)
```

---

## Task 8: AnomalyTransformer

**Files:**
- Create: `pyod/models/ts_anomaly_transformer.py`
- Modify: `pyod/test/test_ts_od.py`

The implementer MUST understand the Anomaly Transformer:
- Anomaly-attention: series-association (softmax) vs prior-association (Gaussian kernel)
- Association discrepancy = symmetrized KL divergence
- Minimax optimization with stop-gradient
- Final score = softmax(-AssDis) * reconstruction_error
- Reference: Xu et al., ICLR 2022

- [ ] **Step 1: Write tests**

```python
class TestAnomalyTransformer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.X_train = self.rng.randn(300)

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_fit(self):
        from pyod.models.ts_anomaly_transformer import AnomalyTransformer
        clf = AnomalyTransformer(window_size=50, d_model=32,
                                  n_heads=2, n_layers=1, epochs=2,
                                  contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_decision_function(self):
        from pyod.models.ts_anomaly_transformer import AnomalyTransformer
        clf = AnomalyTransformer(window_size=50, d_model=32,
                                  n_heads=2, n_layers=1, epochs=2)
        clf.fit(self.X_train)
        scores = clf.decision_function(self.rng.randn(200))
        assert len(scores) == 200

    @unittest.skipUnless(_torch_available(), "torch not installed")
    def test_multivariate(self):
        from pyod.models.ts_anomaly_transformer import AnomalyTransformer
        X = self.rng.randn(300, 3)
        clf = AnomalyTransformer(window_size=50, d_model=32,
                                  n_heads=2, n_layers=1, epochs=2)
        clf.fit(X)
        assert len(clf.decision_scores_) == 300
```

- [ ] **Step 2: Implement AnomalyTransformer**

This is the most complex implementation (~400 lines). The implementer must:
1. Build Transformer encoder with anomaly-attention layers (PyTorch)
2. Anomaly-attention: compute series-association (QK^T/sqrt(d) softmax) and prior-association (Gaussian kernel with learnable sigma)
3. Association discrepancy = `[KL(P||S) + KL(S||P)] / 2` per head/layer, summed
4. Minimax: minimize phase detaches prior, maximize phase detaches series
5. Loss = `||X - X_hat||^2 - lambda * sum(AssDis)` (minimize) / `+lambda * sum(AssDis)` (maximize)
6. Score = `softmax(-AssDis) * ||x_t - x_hat_t||^2`
7. Map window scores to timestamps, follow masked-score workflow

- [ ] **Step 3: Run tests**

Run: `python -m pytest pyod/test/test_ts_od.py::TestAnomalyTransformer -v`

- [ ] **Step 4: Propose commit**

```
feat: add AnomalyTransformer detector (attention discrepancy)
```

---

## Task 9: ADEngine Integration + Knowledge Base

**Files:**
- Modify: `pyod/utils/knowledge/algorithms.json`
- Modify: `pyod/utils/knowledge/routing_rules.json`
- Modify: `pyod/utils/ad_engine.py`
- Modify: `CHANGES.txt`

- [ ] **Step 1: Update algorithms.json**

Update the existing `TimeSeriesOD` entry: change `class_path` from `pyod.models.tsod.TimeSeriesOD` to `pyod.models.ts_od.TimeSeriesOD`, change `status` from `planned` to `shipped`, change `data_types` to `["time_series"]`.

Add 6 new entries for: `MatrixProfile`, `SpectralResidual`, `KShape`, `SAND`, `LSTMAD`, `AnomalyTransformer`. Each with `data_types: ["time_series"]`, appropriate `status` (`shipped` or `experimental`), and `category: "time_series"`.

- [ ] **Step 2: Update routing_rules.json**

Update the `time_series_default` rule recommendations to:
```json
[
  {"detector": "TimeSeriesOD", "params": {"detector": "IForest"}, "confidence": 0.85},
  {"detector": "SpectralResidual", "params": {}, "confidence": 0.8},
  {"detector": "LSTMAD", "params": {}, "confidence": 0.7}
]
```

Remove the `note` field about TimeSeriesOD being planned.

- [ ] **Step 3: Patch ADEngine for transductive detectors**

In `pyod/utils/ad_engine.py`, in the `run_detection()` method, wrap the `X_test` scoring in a try/except:

```python
        if X_test is not None:
            try:
                result['scores_test'] = clf.decision_function(X_test)
                result['labels_test'] = clf.predict(X_test)
            except NotImplementedError:
                result['scores_test'] = None
                result['labels_test'] = None
```

- [ ] **Step 4: Update CHANGES.txt**

Add entry for time series detectors.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest pyod/test/test_ts_od.py pyod/test/test_ad_engine.py pyod/test/test_knowledge.py -v`
Expected: All PASS

- [ ] **Step 6: Propose commit**

```
feat: integrate TS detectors with ADEngine knowledge base and routing
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Shared utilities (_ts_utils.py) → Task 1
- [x] BibTeX entries → Task 1
- [x] TimeSeriesOD bridge → Task 2
- [x] MatrixProfile (STOMP, transductive) → Task 3
- [x] SpectralResidual (FFT saliency) → Task 4
- [x] KShape (experimental) → Task 5
- [x] SAND (experimental) → Task 6
- [x] LSTMAD (prediction + Mahalanobis) → Task 7
- [x] AnomalyTransformer (attention discrepancy) → Task 8
- [x] ADEngine integration + knowledge base → Task 9
- [x] Masked-score fit workflow → spec Section 4, implemented per detector
- [x] Scoring alignment policy → spec Section 5, via _ts_utils
- [x] MatrixProfile transductive overrides → Task 3
- [x] ADEngine NotImplementedError handling → Task 9
- [x] algorithms.json class_path migration → Task 9

**Placeholder scan:** No TBD/TODO found. Tasks 5-8 have less detailed code blocks due to algorithm complexity, but the spec and paper references provide complete implementation guidance.

**Type consistency:** All detectors use `validate_ts_input`, `map_scores_to_timestamps`, `aggregate_channel_scores` from `_ts_utils.py`. All follow the same masked-score fit workflow. Method signatures consistent.
