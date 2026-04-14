# Time series anomaly detection reference

PyOD ships 7 time-series detectors plus the `TimeSeriesOD` bridge (which wraps any tabular detector for windowed time-series use). The agent loads this file when the master decision tree (in SKILL.md) routes to time series.

## Decision table by sequence type

| Sequence type | Recommended starters (top-3) | Why |
|---|---|---|
| Univariate, regular sampling | `SpectralResidual`, `MatrixProfile`, `TimeSeriesOD` over `ECOD` | Simple statistical methods dominate; TSB-AD NeurIPS 2024 |
| Multivariate, regular sampling | `LSTMAD`, `KShape`, `TimeSeriesOD` over `IForest` | Multivariate sequence modeling + windowed bridge |
| Long sequences (> 100k points) | `SpectralResidual`, `SAND` | Streaming / non-stationary specialized |
| Short sequences (< 1k points) | `MatrixProfile`, `TimeSeriesOD` over `HBOS` | Avoid deep methods on small data |
| Subsequence anomalies | `MatrixProfile`, `KShape` | Designed for shape-based subsequence detection |
| Need foundation-model embeddings | (gap — see backlog) | TSFM encoder wrappers planned for v3.3.0 |

## Detectors available in PyOD (KB-derived)

<!-- BEGIN KB-DERIVED: time-series-detector-list -->
- **AnomalyTransformer** (Anomaly Transformer) — complexity: time O(n * L * d^2), space O(model_params); best for: Research use only; simpler methods outperform on standard benchmarks; avoid when: Accuracy matters -- underperforms simpler methods like MatrixProfile and IForest on all TSB-AD scenarios; requires: pyod[torch]; paper: Xu et al., ICLR 2022
- **KShape** (k-Shape Clustering Anomaly Detection) — complexity: time O(n * k * max_iter), space O(n * m); best for: Detecting shape-based anomalies in short-to-medium time series subsequences; avoid when: Time series is very long (performance degrades); paper: Paparrizos & Gravano, SIGMOD 2015
- **LSTMAD** (LSTM-based Anomaly Detection) — complexity: time O(n * epochs), space O(model_params); best for: Multivariate and long time series with complex temporal patterns; avoid when: Fast inference is critical or data is very short; requires: pyod[torch]; paper: Malhotra et al., ESANN 2015
- **MatrixProfile** (Matrix Profile (STOMP)) — complexity: time O(n^2), space O(n); best for: Subsequence anomaly detection where exact distances matter; avoid when: Out-of-sample prediction is needed; paper: Yeh et al., ICDM 2016
- **SAND** (Streaming Anomaly Detection) — complexity: time O(n * k), space O(k * m); best for: Non-stationary time series with evolving normal patterns, especially short series; avoid when: Production reliability is critical; paper: Boniol et al., VLDB 2021
- **SpectralResidual** (Spectral Residual Anomaly Detection) — complexity: time O(n log n), space O(n); best for: Fast detection of point/spike anomalies in periodic or seasonal time series; avoid when: Data has no frequency structure; paper: Ren et al., KDD 2019
- **TimeSeriesOD** (Time Series Outlier Detection) — complexity: time varies by detector, space varies by detector; best for: General-purpose time series anomaly detection with any PyOD detector; avoid when: Specialized temporal methods (LSTM, Transformer) are more appropriate; paper: Zhao et al., 2024
<!-- END KB-DERIVED: time-series-detector-list -->

## TSB-AD context

The TSB-AD benchmark (NeurIPS 2024, Liu/Paparrizos et al.) is the current state-of-the-art comparison set for time-series anomaly detection: 1,070 series across 40 datasets, 40 algorithms, deliberately uses VUS-PR as the scoring metric instead of point-F1 (which inflates F1 from 0.32 to 0.85 on random scores under the classical Point Adjustment protocol).

**Two takeaways from TSB-AD that shape PyOD's recommendations:**

1. **Simple statistical methods often beat SOTA transformers.** `SpectralResidual` and `MatrixProfile` reliably outperform `LSTMAD` and `AnomalyTransformer` on most TSB-AD tasks. Always try the simple methods first.
2. **Foundation-model TSAD ties with one-line baselines.** Recent TSFM-based detectors (the published ones, not yet in PyOD) frequently tie with moving-window variance baselines. The wrapper pattern (TSFM encoder + classical OD) is a v3.3.0 backlog item.

## TimeSeriesOD bridge usage

The `TimeSeriesOD` class wraps any tabular detector for windowed time-series use:

```python
from pyod.utils.ad_engine import ADEngine

engine = ADEngine()
state = engine.start(time_series_data)  # 1D or 2D numpy array
# state.profile['data_type'] == 'time_series'
# state.profile['n_samples'] == series length

state = engine.plan(state)
# For univariate regular sampling, state.plan['detectors'] typically includes
# 'SpectralResidual', 'MatrixProfile', and a TimeSeriesOD wrapper over a
# tabular base detector (e.g. ECOD).
```

The bridge handles windowing, stride, and per-window scoring automatically. Do not manually slice the series and feed to a tabular detector — the bridge handles edge cases (overlap, padding) correctly.

## Worked example: industrial sensor fault detection

### Setup

A factory sensor logs temperature every minute for one week (~10,000 points). The user wants to know if any periods look anomalous. No labels.

### Agent flow

```python
import pandas as pd
from pyod.utils.ad_engine import ADEngine

df = pd.read_csv("sensor.csv", parse_dates=["timestamp"])
y = df["temperature"].values  # 1D array

engine = ADEngine()
state = engine.start(y)
# state.profile: {data_type: 'time_series', n_samples: 10080, sampling: regular}
# Modality: time_series ✓
# n in 1k-100k range
# No timestamps as features (already collapsed) → no Trigger 1

state = engine.plan(state)
# state.plan['detectors'] = ['SpectralResidual', 'MatrixProfile', 'TimeSeriesOD']

state = engine.run(state)
state = engine.analyze(state)
# state.quality: {agreement: 0.62, separation: 0.31, stability: 0.74}
# best_detector: SpectralResidual

# No triggers fire. Proceed to report.
report = engine.report(state)
```

### What the agent reports

> "Found 47 anomalous time intervals across 10,080 minutes (0.5%). The three detectors I ran (`SpectralResidual`, `MatrixProfile`, `TimeSeriesOD`) agreed reasonably well (consensus 0.62). `SpectralResidual` performed best.
>
> Top-5 most anomalous intervals: minutes [842, 1503, 4221, 5917, 8330]. The dominant pattern in 4 of 5 was a sharp temperature spike followed by a slow recovery — characteristic of a sensor glitch or cooling failure.
>
> Confidence: medium-high. Separation is 0.31 (above the 0.2 floor), agreement is 0.62 (above the 0.4 floor), stability is 0.74.
>
> **What I assumed**:
> - Time series, univariate, regular sampling
> - Detectors: `SpectralResidual`, `MatrixProfile`, `TimeSeriesOD` — standard univariate triple
> - Best detector: `SpectralResidual`"

## Time-series-specific pitfalls

### TS1. Windowing strategy mismatch

A one-week sensor time series has different periodicities (hour, day, week). Using a window size of 60 minutes catches per-hour anomalies but misses day-of-week patterns. Mitigation: when uncertain, run multiple window sizes and combine.

### TS2. Leakage from look-ahead in evaluation

Time series anomaly evaluation must use train/test split by time, never random. A random split leaks future information into the training set. Mitigation: when the user asks "how good is this detector?", use a temporal split, not k-fold.

### TS3. Seasonality and trend confound point-wise detectors

`SpectralResidual` is robust to seasonality, but `MatrixProfile` can flag every Sunday as anomalous if Sundays differ from weekdays. Mitigation: detrend / deseasonalize first, or use `KShape` which is shape-aware.

### TS4. Long sequences crash distance-based methods

`KNN` and `LOF` are O(n²) — running them on a 1M-point series exhausts memory. Mitigation: use `SpectralResidual`, `SAND`, or windowed `TimeSeriesOD` over `HBOS`/`COPOD`.

### TS5. Point Adjustment inflates F1

The classical Point Adjustment protocol (PA) flags an entire anomalous segment correctly if any point in it is detected. This inflates F1 from 0.32 to 0.85 on random scores. Use VUS-PR or Balanced Point Adjustment instead. PyOD does not yet ship VUS-PR; this is a v3.3.0 backlog item.

## Time-series-specific escalation triggers

In addition to the global triggers in SKILL.md, watch for these time-series-specific cases:

- **Irregular sampling**: gaps between observations vary → escalate, ask whether to resample to regular grid
- **Very long sequences (> 1M points)**: distance-based methods will OOM → escalate, recommend `SpectralResidual` / `SAND`
- **Missing windows**: large gaps in the series → escalate, ask whether to interpolate or split into segments

## See also

- `pitfalls.md` — extended pitfalls library (preprocessing → detection → analysis → reporting)
- `workflow.md` — the autonomous loop pattern
- SKILL.md — top-10 critical pitfalls and master decision tree
