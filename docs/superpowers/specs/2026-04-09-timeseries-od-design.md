# Time Series Anomaly Detection for PyOD

**Date:** 2026-04-09
**Status:** Draft (v2 -- Round 1 review fixes)
**Version:** 2

---

## 1. Vision

Add time series anomaly detection as first-class PyOD functionality. Not a wrapper or bridge library, but real TS-AD algorithms that inherit `BaseDetector` and work with the existing PyOD API. Most detectors support the full `clf.fit(X); clf.predict(X_test)` workflow. One exception: `MatrixProfile` is transductive (fit-only, no out-of-sample prediction) because its scoring distribution does not transfer to unseen data.

**No breaking changes.** Everything is additive -- new files, new classes, new knowledge base entries.

---

## 2. Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target user | Practitioner first | PyOD brand |
| Dependencies | No external TS library (no TODS, no TSB-AD, no STUMPY, no tslearn) | Self-contained |
| File convention | `ts_` prefix, flat in `pyod/models/` | Consistent with existing structure |
| Class naming | Natural names (MatrixProfile, not MatrixProfileTS) | Module name already signals TS |
| Input format | `(n_timestamps, n_channels)` or `(n_timestamps,)` | Standard; 1D reshaped to `(n, 1)` internally |
| Output format | `decision_scores_` of shape `(n_timestamps,)` | One score per timestamp |
| Base class | `BaseDetector` | Full PyOD compatibility |
| Breaking changes | None | Additive only |

---

## 3. Paper-Reading Requirement

**Before implementing each algorithm, the implementer MUST read the original paper** to understand the method deeply. Implementations must be faithful to the core algorithm (with documented simplifications where needed). Each detector's docstring must cite the paper.

**BibTeX entries must be added to `docs/zreferences.bib`** for every paper cited. The required papers:

| Algorithm | Paper | Venue |
|-----------|-------|-------|
| MatrixProfile | Yeh et al., "Matrix Profile I: All Pairs Similarity Joins for Time Series" | ICDM 2016 |
| SpectralResidual | Ren et al., "Time-Series Anomaly Detection Service at Microsoft" | KDD 2019 |
| KShape | Paparrizos & Gravano, "k-Shape: Efficient and Accurate Clustering of Time Series" | SIGMOD 2015 |
| SAND | Boniol et al., "SAND: Streaming Subsequence Anomaly Detection" | VLDB 2021 |
| LSTMAD | Malhotra et al., "Long Short Term Memory Networks for Anomaly Detection in Time Series" | ESANN 2015 |
| AnomalyTransformer | Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" | ICLR 2022 |
| TSB-AD benchmark | Liu & Paparrizos, "TSB-AD: The Elephant in the Room" | NeurIPS 2024 |

---

## 4. BaseDetector Fit Contract

**(Addressing Codex findings #2 and Round 2 #1.)**

All TS detectors MUST follow this fit contract. Windowed and causal methods produce fewer valid scores than timestamps, so a **masked-score workflow** is required.

```python
def fit(self, X, y=None):
    # 1. Validate input
    X = self._validate_ts_input(X)
    n_timestamps = X.shape[0]

    # 2. Check minimum length (detector-specific)
    min_len = self._get_min_length()
    if n_timestamps < min_len:
        raise ValueError(
            "Time series length (%d) must be >= %d for %s"
            % (n_timestamps, min_len, self.__class__.__name__))

    # 3. Set n_classes (required by BaseDetector)
    self._set_n_classes(y)

    # 4. Compute raw scores (algorithm-specific)
    #    Returns (scores, valid_mask) where:
    #    - scores: shape (n_timestamps,), NaN for uncovered timestamps
    #    - valid_mask: shape (n_timestamps,), True for covered timestamps
    scores, valid_mask = self._compute_scores(X)

    # 5. Process decision scores on VALID subset only
    #    This sets threshold_, labels_, _mu, _sigma correctly
    valid_scores = scores[valid_mask]
    if len(valid_scores) == 0:
        raise ValueError("No valid subsequence scores produced")

    self.decision_scores_ = valid_scores
    self._process_decision_scores()

    # 6. Reconstruct full-length arrays
    #    Fill uncovered timestamps with threshold (boundary)
    full_scores = scores.copy()
    full_scores[~valid_mask] = self.threshold_
    full_labels = (full_scores > self.threshold_).astype(int)

    self.decision_scores_ = full_scores
    self.labels_ = full_labels

    return self
```

**Key points:**
- `_process_decision_scores()` runs on the valid-score subset, so NaNs never poison threshold/stats.
- After thresholding, the full-length arrays are reconstructed with boundary timestamps filled at the threshold value (classified as normal).
- Detectors that produce one score per timestamp with no gaps (e.g., SpectralResidual on full-length FFT) can set `valid_mask = np.ones(n_timestamps, dtype=bool)` and skip the reconstruction.
- `n_timestamps < min_required_length` raises `ValueError` immediately.

**Calibration note for masked detectors:** After reconstruction, `_mu` and `_sigma` reflect the valid-score subset, while `self.decision_scores_` includes threshold-filled boundary values. This means inherited `predict_proba()` and `predict_confidence()` methods are **approximate** for masked detectors -- the linear probability mode uses the full padded array, while the unify mode uses valid-subset stats. This is acceptable for v1; exact calibration would require overriding these methods, which adds complexity without meaningful user-facing benefit for boundary timestamps.
- Each detector implements `_get_min_length()`:
  - Windowed/causal detectors: returns `self.window_size`
  - SpectralResidual: returns `max(self.score_window, 2)` (needs at least a few points for FFT)
  - Dense detectors that produce one score per timestamp: returns `1`

---

## 5. Scoring Alignment Policy

**(Addressing Codex finding #3.)**

All TS detectors produce `decision_scores_` of shape `(n_timestamps,)`. Since windowed/subsequence methods naturally produce fewer scores than timestamps, a global alignment policy is required.

### 5.1 Timestamp Anchoring

Subsequence scores are anchored to the **start** of the subsequence window. A subsequence starting at timestamp `t` with length `w` covers timestamps `[t, t+w)`.

### 5.2 Score Aggregation for Overlapping Windows

When multiple windows cover the same timestamp (step < window_size), per-timestamp scores are computed by:
- **`'max'`** (default): Maximum of all window scores covering the timestamp.
- **`'mean'`**: Mean of all window scores covering the timestamp.

`'median'` is NOT supported in v1 (it requires per-timestamp buffers and is rarely needed).

### 5.3 Boundary Timestamps

The first and last `window_size - 1` timestamps may have fewer covering windows. Policy:
- **Boundary scores are included** in `decision_scores_` using whatever windows cover them.
- Timestamps not covered by any window (possible when `step > window_size`) receive a score of `NaN` and are excluded from threshold fitting. `_process_decision_scores()` is called on the non-NaN subset. The NaN scores are then filled with the computed threshold value (so they are on the normal/anomaly boundary).

### 5.4 Causal Methods (LSTMAD)

For causal predictors, the anomaly score at timestamp `t` is the prediction error when predicting `t` from timestamps `[t - window_size, t)`. The first `window_size` timestamps have no score and receive the threshold-fill treatment described above.

### 5.5 decision_function(X_test)

For test data, the same transformation is applied:
- **Windowed bridge**: Window test data, score windows with the fitted inner detector, aggregate back.
- **Deep methods**: Run the trained model on test windows, compute prediction/reconstruction error.
- **MatrixProfile**: See Section 6.3 (transductive in v1).

---

## 6. Algorithms

### 6.1 Overview

7 new files, 7 new classes. Two multivariate categories:

| Category | Behavior | Detectors |
|----------|----------|-----------|
| `native_multivariate` | Processes all channels jointly | TimeSeriesOD, LSTMAD, AnomalyTransformer |
| `per_channel_aggregate` | Runs independently per channel, aggregates scores | MatrixProfile, SpectralResidual, KShape, SAND |

**(Addressing Codex finding #5.)** Per-channel methods all expose a `channel_aggregation` parameter (`'max'` or `'mean'`, default `'max'`). Per-channel scores are z-normalized before aggregation to prevent high-variance channels from dominating.

| File | Class | Paradigm | Deps | Multivariate | Status |
|------|-------|----------|------|-------------|--------|
| `ts_od.py` | `TimeSeriesOD` | Windowed bridge | numpy | native | shipped |
| `ts_matrix_profile.py` | `MatrixProfile` | Subsequence distance | numpy, scipy | per_channel_aggregate | shipped |
| `ts_spectral_residual.py` | `SpectralResidual` | Frequency domain | numpy, scipy.fft | per_channel_aggregate | shipped |
| `ts_kshape.py` | `KShape` | Shape-based clustering | numpy, scipy | per_channel_aggregate | experimental |
| `ts_sand.py` | `SAND` | Streaming/online | numpy, scipy | per_channel_aggregate | experimental |
| `ts_lstm.py` | `LSTMAD` | Deep (prediction error) | torch | native | shipped |
| `ts_anomaly_transformer.py` | `AnomalyTransformer` | Deep (attention discrepancy) | torch | native | shipped |

**(Addressing Codex finding #8.)** KShape and SAND are marked `experimental` in v1 because they require implementing k-Shape clustering from scratch without tslearn. The core algorithms (cross-correlation centroid update, streaming cluster management) are more complex to port reliably. They ship but with explicit "experimental" status and performance caveats.

### 6.2 TimeSeriesOD (Windowed Bridge)

**What it does:** Slides a window across the time series to create a tabular matrix `(n_windows, window_size * n_channels)`, runs any PyOD detector on it, then maps window-level scores back to per-timestamp scores via the scoring alignment policy (Section 5).

**Parameters:**
- `detector` -- str or BaseDetector instance. Default: `'IForest'`. Uses the same `_DETECTOR_SHORTCUTS` pattern as EmbeddingOD.
- `window_size` -- int. Default: 50.
- `step` -- int. Sliding step. Default: 1.
- `score_aggregation` -- str. `'max'` or `'mean'`. Default: `'max'`.
- `contamination` -- float. Default: 0.1.

**Multivariate:** `native_multivariate`. Channels are concatenated within each window.

### 6.3 MatrixProfile

**Original method:** Yeh et al. (ICDM 2016). Computes the Matrix Profile via the STOMP algorithm: for each subsequence, the z-normalized Euclidean distance to its nearest non-trivial match (outside an exclusion zone). The MP is the vector of minimum distances.

**PyOD adaptation:** The STOMP self-join is used for `fit()`. High MP values indicate unusual subsequences. Subsequence scores are mapped to timestamps via the alignment policy (Section 5).

**(Addressing Codex findings #4 and Round 2 #2.)** MatrixProfile is **transductive** in v1 -- the self-join scoring distribution does not transfer to an AB-join context. The following overrides are required:

- `decision_function(X_test)` raises `NotImplementedError`.
- `predict(X_test)` raises `NotImplementedError` (it calls `decision_function` internally).
- `predict_proba(X_test)` raises `NotImplementedError`.
- `predict_confidence(X_test)` raises `NotImplementedError`.

**ADEngine integration:** MatrixProfile is NOT included in the default TS routing recommendations. It is available via `build_detector({'detector_name': 'MatrixProfile', ...})` for users who know they want transductive scoring. The routing rule recommends `TimeSeriesOD(detector='IForest')` and `SpectralResidual` instead. `ADEngine.run_detection()` will skip `X_test` scoring if the detector raises `NotImplementedError` (catch and set `scores_test=None`, `labels_test=None`).

**PyOD-specific simplifications:** Single-threaded STOMP. No multi-dimensional MP (mSTOMP). No incremental updates.

**Parameters:**
- `window_size` -- int. Subsequence length. Default: 50.
- `contamination` -- float. Default: 0.1.
- `channel_aggregation` -- str. `'max'` or `'mean'`. Default: `'max'`.

**Multivariate:** `per_channel_aggregate`.

### 6.4 SpectralResidual

**Original method:** Ren et al. (KDD 2019). The KDD paper describes a full service pipeline (SR preprocessing + downstream CNN detector). **PyOD implements only the spectral residual saliency computation**, which is the core anomaly scoring mechanism. The downstream CNN is not included.

**(Addressing Codex finding #6.)** This is a PyOD adaptation of the SR saliency heuristic, not a full reimplementation of the Microsoft service.

**Algorithm:**
1. Compute FFT of the input signal.
2. Compute log amplitude spectrum.
3. Smooth log amplitude with averaging filter of size `score_window`.
4. Spectral residual = original log amplitude - smoothed.
5. Inverse FFT gives saliency map.
6. Anomaly score = magnitude of saliency map, smoothed with a final averaging window.

**Parameters:**
- `score_window` -- int. Averaging window for log-amplitude smoothing. Default: 3 (per the original paper; a 1D uniform kernel of size q=3).
- `contamination` -- float. Default: 0.1.
- `channel_aggregation` -- str. Default: `'max'`.

**Multivariate:** `per_channel_aggregate`.

### 6.5 KShape

**Original method:** Paparrizos & Gravano (SIGMOD 2015). k-Shape is a **clustering algorithm** for time series using shape-based distance (cross-correlation). It is not an anomaly detector.

**(Addressing Codex finding #6.)** **PyOD adaptation:** We use k-Shape clustering as the basis for an anomaly detector: cluster sliding-window subsequences, then score each subsequence by its distance to the nearest centroid. Subsequences far from all centroids are anomalous. This is inspired by TSB-AD's KShapeAD wrapper, not the original k-Shape paper.

**Status:** `experimental`. Requires implementing k-Shape (cross-correlation centroid update) from scratch without tslearn.

**Parameters:**
- `n_clusters` -- int. Default: 3.
- `window_size` -- int. Default: 50.
- `max_iter` -- int. Default: 100.
- `contamination` -- float. Default: 0.1.
- `channel_aggregation` -- str. Default: `'max'`.

**Multivariate:** `per_channel_aggregate`.

### 6.6 SAND

**Original method:** Boniol et al. (VLDB 2021). SAND is a streaming anomaly detection method built on a streaming extension of k-Shape. It maintains weighted subsequence clusters, creates/merges clusters dynamically, and uses batch-wise adaptation.

**(Addressing Codex finding #6.)** **PyOD adaptation:** We implement the core mechanism (distance-to-nearest-centroid scoring with periodic centroid updates) but simplify the cluster lifecycle: fixed number of clusters, simple weighted averaging for updates (no dynamic cluster creation/merging). This is a reduced version of the full SAND pipeline.

**Status:** `experimental`.

**Parameters:**
- `window_size` -- int. Default: 50.
- `n_clusters` -- int. Default: 5.
- `alpha` -- float. Update rate for centroid adaptation. Default: 0.5.
- `batch_size` -- int. Number of subsequences between centroid updates. Default: 100.
- `contamination` -- float. Default: 0.1.
- `channel_aggregation` -- str. Default: `'max'`.

**Multivariate:** `per_channel_aggregate`.

### 6.7 LSTMAD

**Original method:** Malhotra et al. (ESANN 2015). Trains a stacked LSTM (2 layers) to predict multiple future steps from a lookback window. Prediction errors across horizons form an error vector. A **multivariate Gaussian** is fitted on these error vectors (mean mu, covariance Sigma). The anomaly score is the **Mahalanobis distance**: `score(t) = (e_t - mu)^T * Sigma^{-1} * (e_t - mu)`.

**PyOD adaptation (simplified):** Single-step prediction (horizon=1) instead of multi-horizon. The error vector at each timestamp has `n_channels` dimensions (one error per channel). A multivariate Gaussian is fitted on training errors, and the anomaly score is the Mahalanobis distance. This is a simplification of the original paper: the paper uses multi-horizon errors to build a richer error vector, while PyOD uses channel-wise single-step errors. Follows the causal scoring policy (Section 5.4). For univariate data, this simplifies to z-scored squared error.

**Parameters:**
- `window_size` -- int. Lookback window. Default: 50.
- `hidden_size` -- int. LSTM hidden dim. Default: 64.
- `n_layers` -- int. Default: 2.
- `epochs` -- int. Default: 50.
- `lr` -- float. Default: 1e-3.
- `batch_size` -- int. Default: 32.
- `contamination` -- float. Default: 0.1.

**Multivariate:** `native_multivariate`. LSTM input has `n_channels` features per timestep. Prediction error summed across channels.

**Requires:** `torch`.

### 6.8 AnomalyTransformer

**Original method:** Xu et al. (ICLR 2022). Transformer encoder with anomaly-attention: learns series-association (standard softmax attention) and prior-association (Gaussian kernel with learnable sigma). Association discrepancy = symmetrized KL divergence between the two. The model uses minimax optimization: minimize phase pushes series-association toward prior (via stop-gradient), maximize phase pushes prior away from series-association. Final anomaly score = `softmax(-AssDis) * reconstruction_error` -- the softmax re-weights reconstruction errors so points with higher discrepancy get amplified.

**PyOD adaptation:** Faithful to the original method including the minimax optimization via stop-gradient. The loss function is `L = ||X - X_hat||^2 - lambda * sum(AssDis)` (minimize phase) and `+lambda * sum(AssDis)` (maximize phase).

**Parameters:**
- `window_size` -- int. Default: 100.
- `d_model` -- int. Default: 512.
- `n_heads` -- int. Default: 8.
- `n_layers` -- int. Default: 3.
- `epochs` -- int. Default: 10.
- `lr` -- float. Default: 1e-4.
- `batch_size` -- int. Default: 32.
- `contamination` -- float. Default: 0.1.

**Multivariate:** `native_multivariate`. All channels as input features.

**Requires:** `torch`.

---

## 7. Shared Utilities (`pyod/models/_ts_utils.py`)

### 7.1 Input Validation

```python
def validate_ts_input(X):
    """Validate and reshape time series input.
    Returns array of shape (n_timestamps, n_channels)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("Expected 1D or 2D input, got %dD" % X.ndim)
    return X
```

### 7.2 Sliding Window Extraction

```python
def sliding_windows(X, window_size, step=1):
    """Extract sliding windows from a time series.
    X: (n_timestamps, n_channels)
    Returns: (n_windows, window_size * n_channels)"""
```

### 7.3 Score Mapping

```python
def map_window_scores_to_timestamps(window_scores, window_size, step,
                                     n_timestamps, aggregation='max'):
    """Map window-level scores back to per-timestamp scores.
    Supports 'max' and 'mean' aggregation.
    Uncovered timestamps receive NaN."""
```

**(Addressing Codex finding #7.)** `'median'` is removed from v1. Uncovered timestamps receive `NaN` (not `0.0`), handled per Section 5.3.

### 7.4 Channel Aggregation

```python
def aggregate_channel_scores(per_channel_scores, method='max'):
    """Aggregate per-channel scores. Z-normalizes each channel
    before aggregation to prevent high-variance channels from
    dominating."""
```

---

## 8. ADEngine Integration

**(Addressing Codex finding #1.)** The existing `algorithms.json` entry for `TimeSeriesOD` uses class path `pyod.models.tsod.TimeSeriesOD`. This MUST be updated to `pyod.models.ts_od.TimeSeriesOD` when the module ships.

When TS detectors ship:

1. **algorithms.json**: Update `TimeSeriesOD` class_path from `pyod.models.tsod.TimeSeriesOD` to `pyod.models.ts_od.TimeSeriesOD` and flip status to `"shipped"`. Add 6 new entries for native detectors.
2. **routing_rules.json**: Update TS rule recommendations:
   ```json
   {
     "id": "time_series_default",
     "recommendations": [
       {"detector": "TimeSeriesOD", "params": {"detector": "IForest"}, "confidence": 0.85},
       {"detector": "SpectralResidual", "params": {}, "confidence": 0.8},
       {"detector": "LSTMAD", "params": {}, "confidence": 0.7}
     ]
   }
   ```
   Note: `MatrixProfile` is excluded from default routing because it is transductive (no `X_test` support). Available via explicit `build_detector()` call.
3. **ADEngine code**: Required patch -- `run_detection()` must catch `NotImplementedError` from `decision_function(X_test)` and `predict(X_test)` for transductive detectors. When caught, set `scores_test=None` and `labels_test=None` in the result dict instead of crashing. Add a regression test for this path.
4. **MCP server**: Update `profile_data` docstring to include `'time_series'`.

---

## 9. File Structure

### New files

| File | Responsibility |
|------|---------------|
| `pyod/models/_ts_utils.py` | Shared TS utilities |
| `pyod/models/ts_od.py` | `TimeSeriesOD` windowed bridge |
| `pyod/models/ts_matrix_profile.py` | `MatrixProfile` |
| `pyod/models/ts_spectral_residual.py` | `SpectralResidual` |
| `pyod/models/ts_kshape.py` | `KShape` (experimental) |
| `pyod/models/ts_sand.py` | `SAND` (experimental) |
| `pyod/models/ts_lstm.py` | `LSTMAD` |
| `pyod/models/ts_anomaly_transformer.py` | `AnomalyTransformer` |
| `pyod/test/test_ts_od.py` | Tests for all TS detectors |
| `docs/zreferences.bib` | Add 7 BibTeX entries |

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/knowledge/algorithms.json` | Update TimeSeriesOD class_path + status; add 6 new entries |
| `pyod/utils/knowledge/routing_rules.json` | Update TS routing recommendations |
| `CHANGES.txt` | Add TS-AD entry |

---

## 10. Implementation Order

| Order | What | Depends on |
|-------|------|-----------|
| 0 | Read papers for each algorithm | Nothing |
| 1 | `_ts_utils.py` (shared utilities) + BibTeX entries | Papers |
| 2 | `ts_od.py` (TimeSeriesOD bridge) | `_ts_utils.py` |
| 3 | `ts_matrix_profile.py` | `_ts_utils.py` |
| 4 | `ts_spectral_residual.py` | `_ts_utils.py` |
| 5 | `ts_kshape.py` (experimental) | `_ts_utils.py` |
| 6 | `ts_sand.py` (experimental) | `_ts_utils.py` |
| 7 | `ts_lstm.py` | `_ts_utils.py` |
| 8 | `ts_anomaly_transformer.py` | `_ts_utils.py` |
| 9 | Tests for all | All above |
| 10 | Knowledge base + routing rules update | All above |

Tasks 2-6 (classical) are independent. Tasks 7-8 (deep) are independent.

---

## 11. Implementation Feasibility

**(Addressing Codex finding #8.)**

| Algorithm | Complexity to port | Confidence | Notes |
|-----------|-------------------|------------|-------|
| TimeSeriesOD | Low (~100 lines) | High | Windowing + existing detector delegation |
| SpectralResidual | Low (~50 lines) | High | FFT + averaging, well-documented |
| MatrixProfile | Medium (~200 lines) | High | STOMP is well-documented, numpy-only |
| LSTMAD | Medium (~150 lines) | High | Standard PyTorch LSTM, reference in TSB-AD |
| AnomalyTransformer | High (~400 lines) | Medium-high | Complex attention mechanism, but TSB-AD reference exists |
| KShape | Medium-high (~200 lines) | Medium | Cross-correlation centroid update from scratch |
| SAND | High (~300 lines) | Medium | Streaming k-Shape variant, simplified from original |

**v1 performance targets:** Single-threaded. Tested on sequences up to 10,000 timestamps. No GPU requirement for classical methods. Deep methods use GPU if available, CPU otherwise.

---

## 12. Codex Review Resolution (Round 4)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| R4-1 | LSTMAD spec inconsistent -- claims "faithful" but uses single-step | **Resolved** | Rewritten as explicit PyOD simplification (single-step, channel-wise errors, Mahalanobis). No longer claims faithfulness. |
| R4-2 | Masked-score predict_proba/predict_confidence calibration ambiguous | **Resolved** | Added calibration note: inherited methods are approximate for masked detectors, acceptable for v1. |

## 13. Codex Review Resolution (Round 3)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| R3-1 | Fit contract assumes `window_size` but SR has `score_window` | **Resolved** | Added `_get_min_length()` hook; each detector defines its own minimum |
| R3-2 | MatrixProfile transductive fix half-integrated | **Resolved** | Vision text updated; ADEngine patch documented as required with regression test |

**Paper-reading corrections (from deep-reading all 6 papers):**
- SR: `score_window` default changed from 21 to 3 (per original paper)
- LSTMAD: anomaly score is Mahalanobis distance under fitted Gaussian on errors, not plain MSE
- AnomalyTransformer: final score is `softmax(-AssDis) * reconstruction_error`, uses minimax with stop-gradient

## 13. Codex Review Resolution (Round 2)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| R2-1 | NaN boundary scores poison _process_decision_scores | **Resolved** | Masked-score workflow: threshold on valid subset, then reconstruct full-length arrays |
| R2-2 | MatrixProfile transductive but in routing + incompatible with ADEngine X_test | **Resolved** | Removed from default routing; all predict methods override with NotImplementedError; ADEngine catches and skips X_test |

## 13. Codex Review Resolution (Round 1)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Module name mismatch (`ts_od.py` vs `tsod` in algorithms.json) | **Resolved** | Explicit migration step in Section 8 |
| 2 | BaseDetector fit contract incomplete | **Resolved** | Section 4: explicit fit contract with `_set_n_classes` + `_process_decision_scores` |
| 3 | No scoring alignment policy | **Resolved** | Section 5: anchoring, aggregation, boundaries, causal methods |
| 4 | MatrixProfile train/test scoring mismatch | **Resolved** | Section 6.3: transductive in v1, `decision_function` raises NotImplementedError |
| 5 | Multivariate categories inconsistent | **Resolved** | Section 6.1: `native_multivariate` vs `per_channel_aggregate` with shared `channel_aggregation` param |
| 6 | Algorithm descriptions overstate fidelity | **Resolved** | Each section now separates "Original method" from "PyOD adaptation" |
| 7 | `map_window_scores_to_timestamps` median broken | **Resolved** | Removed median from v1; uncovered timestamps use NaN |
| 8 | No-dependency feasibility underspecified | **Resolved** | Section 11: feasibility table; KShape and SAND marked experimental |

---

## 13. Reference Implementations

| Algorithm | Primary paper | Implementation reference |
|-----------|--------------|------------------------|
| TimeSeriesOD | (bridge, no paper) | New |
| MatrixProfile | Yeh et al., ICDM 2016 | TSB-AD, STUMPY (reference only) |
| SpectralResidual | Ren et al., KDD 2019 | TSB-AD |
| KShape | Paparrizos & Gravano, SIGMOD 2015 | TSB-AD |
| SAND | Boniol et al., VLDB 2021 | TSB-AD |
| LSTMAD | Malhotra et al., ESANN 2015 | TSB-AD |
| AnomalyTransformer | Xu et al., ICLR 2022 | TSB-AD, original repo |

All references: Apache-2.0 (TSB-AD) or MIT. PyOD is BSD-2-Clause. Porting with attribution is fine.
