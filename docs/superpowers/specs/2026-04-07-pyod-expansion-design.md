# PyOD Expansion: Time Series, Graph, and Agent-Based Anomaly Detection

**Date:** 2026-04-07
**Status:** Brainstorm (in progress)
**Version:** 1

---

## 1. Vision

Make PyOD **the single go-to library for outlier detection across all data types** -- tabular, text, image (shipped in v2.1.0), time series, and graph -- with an LLM-powered agent layer that recommends the right pipeline automatically.

Three expansion tracks:

| Track | Priority | Status | Target |
|-------|----------|--------|--------|
| 1. Time Series AD | **High** (next) | Brainstorming | v2.2.0 |
| 2. Graph AD | Medium | Idea stage | v2.3.0+ |
| 3. Agent / LLM skill | Medium | Idea stage | v2.3.0+ |

---

## 2. Track 1: Time Series Anomaly Detection

### 2.1 Motivation

- **TODS is dead** (no updates since Sep 2023). No maintained, lightweight TS-AD library exists.
- **Merlion** (Salesforce, 4.5k stars) is actively maintained but is a heavy full-stack ecosystem -- overkill for most users.
- **TSB-AD** (NeurIPS 2024, 244 stars) is an excellent *benchmark* (40 algorithms, 1070 datasets) but not a user-facing library. API is functional (`run_Unsupervise_AD('IForest', data)`), not object-oriented.
- TSB-AD already wraps PyOD for many of its statistical detectors (LOF, KNN, IForest, HBOS, COPOD, etc.).
- **The gap**: a practitioner-friendly TS-AD tool with PyOD's API (`clf.fit(X); clf.predict(X)`).

### 2.2 Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target user | **Practitioner first** (option C: practical tool now, benchmark later) | PyOD brand is "practical first, comprehensive second" |
| Dependencies | **No external TS library dependency** (no TODS, no TSB-AD) | Both libraries are dead or benchmark-only; keep PyOD self-contained |
| Algorithm source | Port curated algorithms natively using numpy/scipy/sklearn (+ optional PyTorch for deep methods) | Consistent with existing PyOD dependency philosophy |
| Relationship to TSB-AD | Use their benchmark results to pick which algorithms to port; cite their paper | Complementary: "TSB-AD tells you what works, PyOD lets you deploy it" |

### 2.3 Proposed API (under discussion)

**Option C (recommended):** Both a generic bridge class and dedicated TS detectors.

#### Generic bridge: `TimeSeriesOD`

Follows the EmbeddingOD pattern -- handles windowing/shingling, then pipes into any existing PyOD detector. Maps scores back to per-timestamp granularity.

```python
from pyod.models.tsod import TimeSeriesOD

clf = TimeSeriesOD(detector='IForest', window_size=50)
clf.fit(ts_train)   # shape (n_timestamps,) or (n_timestamps, n_channels)
scores = clf.decision_scores_  # per-timestamp anomaly scores
```

#### Native TS detectors

For methods that don't reduce to "window + tabular OD":

- **MatrixProfile** -- distance-profile based, no windowing abstraction needed
- **Series2Graph** -- graph-of-subsequences approach
- **SAND** -- online streaming detection

Each would inherit `BaseDetector` with TS-specific parameters.

#### Foundation model bridge for TS

Extends EmbeddingOD pattern with TS foundation models:

- **Chronos** (Amazon) -- zero-shot TS forecasting, anomaly = large residual
- **TimesFM** (Google) -- similar approach
- **MOMENT** (CMU) -- TS foundation model with anomaly detection head

```python
clf = TimeSeriesOD(encoder='chronos', detector='KNN')
```

### 2.4 Key Technical Questions (to resolve)

1. **Windowing strategy**: Fixed-size sliding window? Multi-scale? Learned?
2. **Score aggregation**: How to map window-level scores back to point-level? (max, mean, weighted?)
3. **Univariate vs multivariate**: Same class or separate?
4. **Streaming/online**: Support incremental updates or batch-only for v1?
5. **Evaluation**: Ship TS-specific metrics (e.g., point-adjusted F1, range-based metrics) or defer?
6. **Which TSB-AD algorithms to port first**: Need to review their benchmark results for top performers.

---

## 3. Track 2: Graph Anomaly Detection

### 3.1 Motivation

- **PyGOD** (1.5k stars, same creator) has 18 algorithms but is slow-moving (last commit Nov 2024).
- Most PyGOD algorithms require PyTorch Geometric -- heavy dependency.
- Non-GNN methods (SCAN, Radar, ANOMALOUS) could work with scipy sparse matrices alone.
- The EmbeddingOD bridge approach could work here too: graph encoder (node2vec, GNN embeddings) → PyOD detector.

### 3.2 Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Priority | **After time series** | TS has a clearer gap; PyGOD still exists |
| Dependencies | **No PyGOD dependency** | Library is slow-moving |

### 3.3 Open Questions

- Absorb PyGOD algorithms or build a new graph encoder bridge?
- Which algorithms are worth porting without PyG?
- Should PyGOD be deprecated in favor of PyOD's graph module?

---

## 4. Track 3: Agent / LLM-Powered Anomaly Detection

### 4.1 Motivation

- **AD-AGENT** (USC-FORTIS, 97 stars, WIP) already orchestrates PyOD + PyGOD + TSLib with LLM agents -- this is from Yue Zhao's own group.
- A dedicated "anomaly detection agent skill" that knows all PyOD algorithms, benchmarks (ADBench, NLP-ADBench, TSB-AD), and papers could be a killer feature.
- Use cases:
  - "I have server CPU metrics, what detector should I use?" → agent recommends pipeline
  - "Run anomaly detection on this dataset" → agent selects encoder, detector, hyperparameters
  - Summarize all group's works (PyOD, PyGOD, ADBench, AD-LLM, etc.) into a knowledge base

### 4.2 Open Questions

- Should this live in PyOD or as a separate tool/skill?
- Integration with AD-AGENT or independent?
- What knowledge base to build? (algorithm cards, benchmark results, decision trees)

---

## 5. Relationship to Existing Work

| Project | Role | Status |
|---------|------|--------|
| **PyOD** (v2.1.0) | Core library -- tabular + embedding (text/image) | Active, just released |
| **PyGOD** (v1.1.0) | Graph AD | Slow-moving |
| **TODS** | Time series AD | Dead |
| **ADBench** | Tabular benchmark (57 datasets, 30 algorithms) | Complete |
| **NLP-ADBench** | Text AD benchmark | Complete |
| **TSB-AD** | TS benchmark (40 algorithms, 1070 datasets) | Active (external) |
| **AD-AGENT** | Multi-agent AD framework | WIP |
| **AD-LLM** | LLM zero-shot AD | Research |
| **MetaOD** | Data-driven model selection | Complete |

---

## 6. Release Roadmap (tentative)

| Version | Content | Timeline |
|---------|---------|----------|
| v2.1.0 | EmbeddingOD, MultiModalOD (text/image) | **Shipped** (Apr 2026) |
| v2.2.0 | TimeSeriesOD bridge + 3-5 native TS detectors | TBD |
| v2.3.0 | Graph AD (bridge + native detectors) | TBD |
| v3.0.0 | Agent layer, LLMAD, unified API across all modalities | TBD |

---

## Appendix: TSB-AD Algorithm Reference

40 algorithms across 3 categories (from NeurIPS 2024 paper):

**Statistical (18):** MCD, OCSVM, LOF, KNN, KMeansAD, CBLOF, POLY, IForest, HBOS, KShapeAD, MatrixProfile, PCA, RobustPCA, EIF, SR, COPOD, Series2Graph, SAND

**Neural Network (12):** AutoEncoder, LSTMAD, xLSTMAD, Donut, CNN, OmniAnomaly, USAD, AnomalyTransformer, TranAD, TimesNet, FITS, M2N2

**Foundation Model (5):** OFA, Lag-Llama, Chronos, TimesFM, MOMENT
