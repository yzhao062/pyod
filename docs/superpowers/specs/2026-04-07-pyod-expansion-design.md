# PyOD Expansion: The Unified Intelligent Anomaly Detection Platform

**Date:** 2026-04-07
**Status:** Brainstorm (approved direction, details under review)
**Version:** 2

---

## 1. Vision

Make PyOD **the single go-to library for outlier detection across all data types** -- tabular, text, image (shipped in v2.1.0), time series, and graph -- with an intelligent agent layer that recommends the right pipeline automatically and speaks to any LLM.

**"One ring rules all for OD."**

No library does this today. The closest competitors cover single modalities (sklearn for tabular, Anomalib for images, Merlion for TS). PyOD would be the first unified, intelligent anomaly detection platform.

### Priority Order (revised)

| Track | Priority | Status | Target |
|-------|----------|--------|--------|
| 1. **Agent Intelligence Layer** | **Highest** (next) | Design complete | v2.2.0 |
| 2. Time Series AD | High | Brainstorming | v2.2.0 or v2.3.0 |
| 3. Graph AD | Medium | Idea stage | v2.3.0+ |

The agent layer ships first because (a) it's low engineering effort / high impact, (b) it works with today's PyOD immediately, and (c) it becomes more powerful as TS/graph modules are added.

---

## 2. Track 1: Agent Intelligence Layer

### 2.1 Architecture Overview

Three layers, each builds on the previous:

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: Agent Interfaces                          │
│  MCP Server │ Claude Code Skill │ OpenAI Schema     │
│  (any LLM can talk to PyOD)                         │
├─────────────────────────────────────────────────────┤
│  Layer 2: Smart Router (Python)                     │
│  Advisor class: data sniffing, algorithm selection, │
│  hyperparameter defaults, benchmark-backed reasons  │
│  (works WITHOUT an LLM -- pure Python)              │
├─────────────────────────────────────────────────────┤
│  Layer 1: Knowledge Base (structured data)          │
│  Algorithm registry, benchmark results,             │
│  decision trees, paper citations                    │
│  (single source of truth)                           │
└─────────────────────────────────────────────────────┘
```

Key design principle: **the intelligence lives in Python (Layer 2), not in prompts.** LLM interfaces (Layer 3) are thin wrappers. This means PyOD's recommendations work offline, cost nothing, and are deterministic. LLMs enhance the experience but aren't required.

### 2.2 Layer 1: Knowledge Base

A structured, machine-readable registry of every detector in PyOD, enriched with benchmark evidence.

#### 2.2.1 Algorithm Registry (`algorithms.json`)

```json
{
  "ECOD": {
    "class": "pyod.models.ecod.ECOD",
    "full_name": "Empirical Cumulative Distribution Based Outlier Detection",
    "data_types": ["tabular"],
    "category": "probabilistic",
    "complexity": {"time": "O(n*d)", "space": "O(n*d)"},
    "strengths": [
      "Parameter-free",
      "Fast on high-dimensional data",
      "Interpretable (per-feature scores)"
    ],
    "weaknesses": [
      "Assumes feature independence",
      "Struggles with complex feature interactions"
    ],
    "best_for": "High-dimensional tabular data where speed matters",
    "avoid_when": "Strong feature correlations exist",
    "benchmark_rank": {
      "ADBench_overall": 5,
      "ADBench_high_dim": 2
    },
    "paper": "Li et al., TKDE 2022",
    "default_params": {"contamination": 0.1},
    "requires": []
  },
  "EmbeddingOD": {
    "class": "pyod.models.embedding.EmbeddingOD",
    "full_name": "Embedding-based Outlier Detection",
    "data_types": ["text", "image", "multimodal"],
    "category": "embedding",
    "complexity": {"time": "encoder-dependent", "space": "encoder-dependent"},
    "strengths": [
      "Works on any data with an embedding model",
      "Leverages foundation model knowledge",
      "Two-step approach beats end-to-end (NLP-ADBench)"
    ],
    "weaknesses": [
      "Embedding quality is decisive",
      "Requires embedding model download or API key"
    ],
    "best_for": "Text or image anomaly detection",
    "avoid_when": "Data is already tabular/numeric",
    "benchmark_rank": {
      "NLP_ADBench_overall": 1
    },
    "paper": "NLP-ADBench, EMNLP 2025",
    "default_params": {"detector": "KNN"},
    "requires": ["sentence-transformers OR openai OR transformers+torch"]
  }
}
```

Every PyOD model gets an entry. Current count: ~45 detectors + EmbeddingOD + MultiModalOD. Future: TimeSeriesOD, GraphOD.

#### 2.2.2 Benchmark Results (`benchmarks.json`)

Structured summaries of major benchmarks:

```json
{
  "ADBench": {
    "paper": "Han et al., NeurIPS 2022",
    "scope": "tabular",
    "datasets": 57,
    "algorithms": 30,
    "top_5": ["ECOD", "IForest", "KNN", "COPOD", "HBOS"],
    "key_finding": "No single algorithm dominates; ensemble of top-5 is robust"
  },
  "NLP_ADBench": {
    "paper": "Li et al., EMNLP 2025",
    "scope": "text",
    "datasets": 8,
    "algorithms": 19,
    "top_5": ["OpenAI+LUNAR", "OpenAI+LOF", "OpenAI+AE", "MiniLM+KNN", "BERT+LOF"],
    "key_finding": "Embedding quality >> detector choice; two-step beats end-to-end"
  },
  "TSB_AD": {
    "paper": "Liu & Paparrizos, NeurIPS 2024",
    "scope": "time_series",
    "datasets": 1070,
    "algorithms": 40,
    "top_5": ["IForest", "LOF", "POLY", "KNN", "KShapeAD"],
    "key_finding": "Classical methods competitive with deep; MatrixProfile strong on subsequence anomalies"
  }
}
```

#### 2.2.3 Decision Tree (`decision_tree.json`)

Machine-readable routing logic:

```json
{
  "rules": [
    {
      "condition": {"data_type": "tabular", "n_features": ">100", "priority": "speed"},
      "recommend": ["ECOD", "HBOS", "IForest"],
      "reason": "High-dim tabular + speed priority → parameter-free fast methods"
    },
    {
      "condition": {"data_type": "tabular", "n_features": "<20", "n_samples": "<5000"},
      "recommend": ["KNN", "LOF", "CBLOF"],
      "reason": "Low-dim small dataset → proximity-based methods excel"
    },
    {
      "condition": {"data_type": "text"},
      "recommend": ["EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN')"],
      "reason": "Text → EmbeddingOD with sentence-transformers (NLP-ADBench top performer)"
    },
    {
      "condition": {"data_type": "image"},
      "recommend": ["EmbeddingOD(encoder='dinov2-small', detector='KNN')"],
      "reason": "Image → EmbeddingOD with DINOv2 vision encoder"
    },
    {
      "condition": {"data_type": "time_series"},
      "recommend": ["TimeSeriesOD(detector='IForest')", "MatrixProfileOD"],
      "reason": "TS → windowed IForest (TSB-AD top performer) or MatrixProfile for subsequence"
    },
    {
      "condition": {"data_type": "multimodal"},
      "recommend": ["MultiModalOD"],
      "reason": "Mixed modalities → score fusion across separate detectors"
    }
  ]
}
```

#### 2.2.4 Papers & Citations (`papers.json`)

```json
{
  "pyod": {
    "title": "PyOD: A Python Toolbox for Scalable Outlier Detection",
    "authors": "Zhao, Nasrullah, Li",
    "venue": "JMLR 2019",
    "bibtex": "..."
  },
  "nlp_adbench": { ... },
  "adbench": { ... },
  "ad_llm": { ... }
}
```

### 2.3 Layer 2: Smart Router (`Advisor` class)

Pure Python, no LLM dependency. Rule-based now, meta-learned later.

```python
# pyod/utils/advisor.py

class Advisor:
    """Intelligent anomaly detection advisor.

    Recommends detectors, encoders, and hyperparameters based on
    dataset characteristics and benchmark evidence.

    Works without an LLM -- pure Python rule-based engine backed
    by structured knowledge (algorithm registry, benchmark results,
    decision trees).

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge base directory. If None, uses the bundled
        knowledge base shipped with PyOD.

    Examples
    --------
    >>> from pyod.utils.advisor import Advisor
    >>> advisor = Advisor()

    # Auto-detect data type and recommend
    >>> result = advisor.recommend(X_train)
    >>> print(result['detector'], result['reason'])
    'ECOD' 'Tabular, 50 features, 10k samples → ECOD (ADBench rank #2 for high-dim)'

    # Explicit data type
    >>> result = advisor.recommend(texts, data_type='text')
    >>> print(result['pipeline'])
    "EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN')"

    # Get runnable detector instance
    >>> clf = advisor.build(X_train)
    >>> clf.fit(X_train)

    # Compare options
    >>> options = advisor.compare(X_train, top_k=3)
    >>> for opt in options:
    ...     print(opt['detector'], opt['reason'], opt['benchmark_rank'])
    """

    def __init__(self, knowledge_dir=None):
        ...

    def sniff_data_type(self, X):
        """Auto-detect data type from input.

        Returns
        -------
        data_type : str
            One of 'tabular', 'text', 'image', 'time_series',
            'multimodal', 'graph'.
        metadata : dict
            Detected characteristics (n_samples, n_features, etc.)
        """
        ...

    def recommend(self, X=None, data_type=None, n_samples=None,
                  n_features=None, priority='balanced', **kwargs):
        """Recommend a detector pipeline.

        Parameters
        ----------
        X : array-like, list, or dict, optional
            Input data. If provided, data_type is auto-detected.
        data_type : str, optional
            Explicit data type override.
        priority : str, optional (default='balanced')
            One of 'speed', 'accuracy', 'balanced'.

        Returns
        -------
        recommendation : dict
            Keys: 'detector', 'pipeline', 'params', 'reason',
            'benchmark_evidence', 'confidence'.
        """
        ...

    def build(self, X=None, data_type=None, **kwargs):
        """Build and return a configured detector instance.

        Same parameters as recommend(), but returns a ready-to-fit
        detector instead of a recommendation dict.
        """
        ...

    def compare(self, X=None, data_type=None, top_k=3, **kwargs):
        """Compare top-k detector options with trade-offs.

        Returns
        -------
        options : list of dict
            Each with 'detector', 'reason', 'pros', 'cons',
            'benchmark_rank', 'estimated_speed'.
        """
        ...

    def explain(self, detector_name):
        """Explain an algorithm: how it works, when to use it.

        Returns
        -------
        explanation : dict
            Keys: 'name', 'full_name', 'description', 'strengths',
            'weaknesses', 'best_for', 'avoid_when', 'paper',
            'benchmark_results'.
        """
        ...

    def list_detectors(self, data_type=None):
        """List available detectors, optionally filtered by data type.

        Returns
        -------
        detectors : list of dict
        """
        ...
```

#### Data Type Sniffing Logic

```python
def sniff_data_type(self, X):
    if isinstance(X, dict):
        return 'multimodal', {...}
    if isinstance(X, list):
        if all(isinstance(x, str) for x in X[:10]):
            return 'text', {'n_samples': len(X)}
        if _looks_like_image_paths(X[:5]):
            return 'image', {'n_samples': len(X)}
    X = np.asarray(X)
    if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
        return 'time_series', {'n_timestamps': X.shape[0], 'channels': 1}
    if X.ndim == 2 and X.shape[0] > X.shape[1] * 10:
        # Very long relative to features → likely time series
        return 'time_series', {'n_timestamps': X.shape[0],
                               'channels': X.shape[1]}
    return 'tabular', {'n_samples': X.shape[0],
                       'n_features': X.shape[1]}
```

### 2.4 Layer 3: Agent Interfaces

Thin wrappers around the Advisor. Multiple formats, same intelligence.

#### 2.4.1 MCP Server (`pyod/mcp_server.py`)

Works with Claude Code, VS Code Copilot, Cursor, Gemini, any MCP client.

```python
from mcp.server.fastmcp import FastMCP
from pyod.utils.advisor import Advisor

mcp = FastMCP("pyod-advisor")
advisor = Advisor()

@mcp.tool()
def recommend_detector(
    data_type: str = "auto",
    n_samples: int = None,
    n_features: int = None,
    priority: str = "balanced",
    description: str = ""
) -> str:
    """Recommend a PyOD anomaly detector for the given dataset.

    Args:
        data_type: One of 'tabular', 'text', 'image', 'time_series',
                   'multimodal', 'graph', or 'auto'.
        n_samples: Number of samples/data points.
        n_features: Number of features (for tabular data).
        priority: One of 'speed', 'accuracy', 'balanced'.
        description: Free-text description of the dataset.
    """
    result = advisor.recommend(
        data_type=data_type, n_samples=n_samples,
        n_features=n_features, priority=priority)
    return format_recommendation(result)

@mcp.tool()
def explain_algorithm(name: str) -> str:
    """Explain how a PyOD algorithm works, with strengths and weaknesses."""
    return format_explanation(advisor.explain(name))

@mcp.tool()
def compare_detectors(
    names: list[str] = None,
    data_type: str = "tabular",
    top_k: int = 3
) -> str:
    """Compare detectors for a given data type."""
    if names:
        return format_comparison([advisor.explain(n) for n in names])
    return format_comparison(advisor.compare(data_type=data_type, top_k=top_k))

@mcp.tool()
def list_available_detectors(data_type: str = None) -> str:
    """List all PyOD detectors, optionally filtered by data type."""
    return format_list(advisor.list_detectors(data_type=data_type))

@mcp.tool()
def get_benchmark_results(benchmark: str = "all") -> str:
    """Get benchmark results (ADBench, NLP-ADBench, TSB-AD)."""
    return format_benchmarks(advisor.get_benchmarks(benchmark))

@mcp.tool()
def generate_code(
    data_type: str,
    detector: str = None,
    task: str = "fit_predict"
) -> str:
    """Generate ready-to-run PyOD code for a given scenario."""
    return advisor.generate_code(data_type=data_type,
                                 detector=detector, task=task)
```

#### 2.4.2 Claude Code Skill (`skills/od-expert/SKILL.md`)

Loaded when users work in the PyOD repo or ask about anomaly detection:

```markdown
---
name: od-expert
description: Anomaly detection expert. Recommends PyOD detectors,
  explains algorithms, and generates code for any data type.
---

You are an anomaly detection expert powered by PyOD's knowledge base.

## When to activate
- User asks "which detector should I use?"
- User has data and wants anomaly detection
- User asks about PyOD algorithms
- User asks to compare methods

## Decision Logic
[embedded decision tree from decision_tree.json]

## Algorithm Quick Reference
[embedded from algorithms.json]

## Benchmark Evidence
[embedded from benchmarks.json]
```

#### 2.4.3 OpenAI Function Schema Export

For integration with AD-AGENT and other GPT-based tools:

```python
# pyod/utils/advisor.py
class Advisor:
    ...
    def to_openai_tools(self):
        """Export advisor methods as OpenAI function-calling tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "recommend_detector",
                    "description": "Recommend a PyOD detector...",
                    "parameters": { ... }
                }
            },
            ...
        ]
```

### 2.5 Evolution Toward AD-AGENT

The Advisor maps to AD-AGENT's four agents:

| AD-AGENT Agent | PyOD Advisor Method | Phase |
|----------------|-------------------|-------|
| **Processor** (data prep) | `sniff_data_type()` + auto windowing/encoding | Phase 1 |
| **Detection** (run AD) | `build()` → returns configured detector | Phase 1 |
| **Explanation** (interpret) | `explain()` + benchmark context | Phase 1 |
| **Adaptation** (iterate) | MetaOD integration, hyperparameter search | Phase 3+ |

Key difference from AD-AGENT: PyOD's advisor works **without an LLM** at the core. AD-AGENT requires OpenAI API ($0.02-0.05/call). PyOD's advisor is free, offline, deterministic. LLMs enhance it through MCP/skills but aren't required.

### 2.6 File Structure

```
pyod/
├── pyod/
│   ├── utils/
│   │   ├── advisor.py              # Layer 2: Advisor class
│   │   └── knowledge/              # Layer 1: Knowledge base
│   │       ├── algorithms.json     # Algorithm registry (45+ entries)
│   │       ├── benchmarks.json     # ADBench, NLP-ADBench, TSB-AD results
│   │       ├── decision_tree.json  # Routing rules
│   │       └── papers.json         # Citations
│   └── mcp_server.py              # Layer 3a: MCP server
├── skills/
│   └── od-expert/
│       └── SKILL.md               # Layer 3b: Claude Code skill
└── examples/
    └── advisor_example.py         # Usage examples
```

### 2.7 Implementation Phases

| Phase | What | Effort | Delivers |
|-------|------|--------|----------|
| **Phase 1a** | Knowledge base: `algorithms.json` for all 45+ current detectors | 1-2 days | Foundation for everything |
| **Phase 1b** | Knowledge base: `benchmarks.json`, `decision_tree.json`, `papers.json` | 1 day | Complete knowledge layer |
| **Phase 2a** | `Advisor` class: `sniff_data_type()`, `recommend()`, `explain()` | 2-3 days | Pure Python advisor |
| **Phase 2b** | `Advisor` class: `build()`, `compare()`, `list_detectors()` | 1-2 days | Full advisor API |
| **Phase 3a** | MCP server | 1 day | Any LLM can talk to PyOD |
| **Phase 3b** | Claude Code skill | 1 day | Works in PyOD dev workflow |
| **Phase 3c** | OpenAI function schema export | 0.5 day | AD-AGENT integration |
| **Phase 4** | Tests + examples + documentation | 1-2 days | Ship-ready |
| **Phase 5** | MetaOD integration (data-driven model selection) | Longer term | Adaptive recommendations |

**Total for v1 (Phases 1-4): ~8-12 days**

---

## 3. Track 2: Time Series Anomaly Detection

### 3.1 Motivation

- **TODS is dead** (no updates since Sep 2023). No maintained, lightweight TS-AD library exists.
- **Merlion** (Salesforce, 4.5k stars) is actively maintained but is a heavy full-stack ecosystem -- overkill for most users.
- **TSB-AD** (NeurIPS 2024, 244 stars) is an excellent *benchmark* (40 algorithms, 1070 datasets) but not a user-facing library. API is functional (`run_Unsupervise_AD('IForest', data)`), not object-oriented.
- TSB-AD already wraps PyOD for many of its statistical detectors (LOF, KNN, IForest, HBOS, COPOD, etc.).
- **The gap**: a practitioner-friendly TS-AD tool with PyOD's API (`clf.fit(X); clf.predict(X)`).

### 3.2 Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target user | **Practitioner first** (practical tool now, benchmark later) | PyOD brand is "practical first, comprehensive second" |
| Dependencies | **No external TS library dependency** (no TODS, no TSB-AD) | Both libraries are dead or benchmark-only; keep PyOD self-contained |
| Algorithm source | Port curated algorithms natively using numpy/scipy/sklearn (+ optional PyTorch for deep methods) | Consistent with existing PyOD dependency philosophy |
| Relationship to TSB-AD | Use their benchmark results to pick which algorithms to port; cite their paper | Complementary: "TSB-AD tells you what works, PyOD lets you deploy it" |

### 3.3 Proposed API

**Both a generic bridge class and dedicated TS detectors** (same pattern as EmbeddingOD).

#### Generic bridge: `TimeSeriesOD`

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

#### Foundation model bridge for TS

```python
clf = TimeSeriesOD(encoder='chronos', detector='KNN')
```

### 3.4 Key Technical Questions (to resolve)

1. **Windowing strategy**: Fixed-size sliding window for v1 (recommended)
2. **Score aggregation**: How to map window-level scores to point-level? (max, mean, weighted?)
3. **Univariate vs multivariate**: Same class or separate?
4. **Streaming/online**: Batch-only for v1 (recommended)
5. **Evaluation**: Defer TS-specific metrics to v2
6. **Which TSB-AD algorithms to port first**: Review benchmark results for top performers

---

## 4. Track 3: Graph Anomaly Detection

### 4.1 Motivation

- **PyGOD** (1.5k stars, same creator) has 18 algorithms but is slow-moving (last commit Nov 2024).
- Most PyGOD algorithms require PyTorch Geometric -- heavy dependency.
- Non-GNN methods (SCAN, Radar, ANOMALOUS) could work with scipy sparse matrices alone.
- The EmbeddingOD bridge approach could work here too: graph encoder (node2vec, GNN embeddings) → PyOD detector.

### 4.2 Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Priority | **After time series and agent layer** | TS has a clearer gap; PyGOD still exists |
| Dependencies | **No PyGOD dependency** | Library is slow-moving |

### 4.3 Open Questions

- Absorb PyGOD algorithms or build a new graph encoder bridge?
- Which algorithms are worth porting without PyG?
- Should PyGOD be deprecated in favor of PyOD's graph module?

---

## 5. Relationship to Existing Work

| Project | Role | Relationship to PyOD |
|---------|------|---------------------|
| **PyOD** (v2.1.0) | Core library -- tabular + embedding (text/image) | This project |
| **PyGOD** (v1.1.0) | Graph AD | Same creator; may absorb later |
| **TODS** | Time series AD | Dead; PyOD fills the gap |
| **AD-AGENT** | Multi-agent AD framework | Same group; Advisor feeds into it |
| **ADBench** | Tabular benchmark (57 datasets, 30 algorithms) | Knowledge base source |
| **NLP-ADBench** | Text AD benchmark | Knowledge base source |
| **TSB-AD** | TS benchmark (40 algorithms, 1070 datasets) | Knowledge base source (external) |
| **AD-LLM** | LLM zero-shot AD | Future LLMAD integration |
| **MetaOD** | Data-driven model selection | Future Advisor enhancement |
| **Anomalib** | Image AD (Intel) | Complementary; EmbeddingOD covers this |
| **Merlion** | Time series (Salesforce) | Competitor; too heavy for most users |

---

## 6. Release Roadmap

| Version | Content | Effort |
|---------|---------|--------|
| v2.1.0 | EmbeddingOD, MultiModalOD (text/image) | **Shipped** (Apr 2026) |
| v2.2.0 | **Agent Intelligence Layer** (Advisor + MCP + Skill) | ~8-12 days |
| v2.3.0 | TimeSeriesOD bridge + native TS detectors | TBD |
| v2.4.0 | Graph AD (bridge + native detectors) | TBD |
| v3.0.0 | LLMAD, MetaOD integration, full multi-modal unified API | TBD |

---

## Appendix A: TSB-AD Algorithm Reference

40 algorithms across 3 categories (from NeurIPS 2024 paper):

**Statistical (18):** MCD, OCSVM, LOF, KNN, KMeansAD, CBLOF, POLY, IForest, HBOS, KShapeAD, MatrixProfile, PCA, RobustPCA, EIF, SR, COPOD, Series2Graph, SAND

**Neural Network (12):** AutoEncoder, LSTMAD, xLSTMAD, Donut, CNN, OmniAnomaly, USAD, AnomalyTransformer, TranAD, TimesNet, FITS, M2N2

**Foundation Model (5):** OFA, Lag-Llama, Chronos, TimesFM, MOMENT

## Appendix B: Current PyOD Detector Inventory (v2.1.0)

**Probabilistic (10):** ECOD, ABOD, COPOD, MAD, SOS, QMCD, KDE, Sampling, GMM, FastABOD
**Linear (6):** PCA, KPCA, MCD, CD, OCSVM, LMDD
**Proximity-based (9):** LOF, COF, CBLOF, LOCI, HBOS, HDBSCAN, KNN, SOD, ROD
**Ensemble (8):** IForest, INNE, DIF, FeatureBagging, LSCP, LODA, SUOD, XGBOD
**Deep Learning (9):** AutoEncoder, VAE, SO_GAAL, MO_GAAL, DeepSVDD, AnoGAN, ALAD, AE1SVM, DevNet
**Graph-based (2):** RGraph, LUNAR
**Embedding/Foundation Model (2):** EmbeddingOD, MultiModalOD

**Total: ~46 detectors**
