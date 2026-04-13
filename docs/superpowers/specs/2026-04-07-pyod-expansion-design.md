# PyOD 3.0: The Intelligent Anomaly Detection Platform

**Date:** 2026-04-07
**Status:** Design complete (4 design review rounds + Python-first revision)
**Version:** 6

---

## 1. Vision

**The future of anomaly detection is agent interaction.**

Today, anomaly detection is a manual, code-heavy process: pick a detector, tune parameters, interpret results, iterate. PyOD 3.0 changes this. Any AI agent that can execute Python -- Claude Code, Cursor, Copilot, AD-AGENT, custom agents -- can use PyOD's `ADEngine` to handle the **entire anomaly detection lifecycle**: understand the data, plan the pipeline, execute detection, analyze results, iterate, and report. The agent writes Python; ADEngine does the heavy lifting.

PyOD becomes not just a library of detectors, but **the execution engine for intelligent anomaly detection** across all data types -- tabular, text, image, time series, graph, and multi-modal. The primary interface is Python (`ADEngine`), with an optional MCP server for knowledge queries and planning.

**"One ring rules all for OD."**

### What This Looks Like

A user opens any agent CLI and says:

> "I have server CPU metrics from the last 30 days. Find the anomalies."

The agent:
1. Calls `profile_data()` → "univariate time series, 43,200 points, 1-minute resolution"
2. Calls `plan_detection()` → "TimeSeriesOD with IForest, window=60 (TSB-AD top performer for TS)"
3. Calls `run_detection()` → scores, labels, runtime stats
4. Calls `analyze_results()` → "37 anomalies found, clustered around 03:00-04:00 daily"
5. User: "These look like normal maintenance windows. Exclude 03:00-04:00 and rerun."
6. Calls `run_detection()` with exclusion → "8 anomalies remaining, 3 are severity-high"
   *(Note: exclusion-window support is a future enhancement. In Tier B v1,
   iteration is via `suggest_next_step()` returning a `new_plan` for a plain rerun.)*
7. Calls `explain_findings()` → "Sample #12,847: CPU spike to 98% with no corresponding scheduled job"
8. Calls `generate_report()` → summary with scores, timestamps, explanations

No PyOD code written by the user. No detector selection. No parameter tuning. The agent writes Python calling `ADEngine` methods and handles it all.

### Priority Order

| Track | Priority | Status | Target |
|-------|----------|--------|--------|
| 1. **Agent Intelligence Layer** | **Highest** (next) | Design v3 | v2.2.0 |
| 2. Time Series AD | High | Brainstorming | v2.3.0 |
| 3. Graph AD | Medium | Idea stage | v2.4.0+ |

---

## 2. Track 1: Agent Intelligence Layer

### 2.1 Architecture

Three layers. Each layer is independently useful but they compose into the full lifecycle.

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: Agent Interfaces                                   │
│  MCP Server (knowledge/planning) │ Claude Code Skill         │
│  (Python agents: full lifecycle │ MCP-only: Tier A queries)  │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Lifecycle Engine (Python)                          │
│  ADEngine class: profile → plan → run → analyze → iterate   │
│  (works WITHOUT an LLM -- pure Python, deterministic)        │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Knowledge Base (structured data)                   │
│  Algorithm registry, benchmark results, routing rules,       │
│  paper citations (single source of truth for all layers)     │
└──────────────────────────────────────────────────────────────┘
```

**Key principle:** The intelligence lives in Python (Layer 2), not in prompts. LLM interfaces (Layer 3) are thin wrappers. PyOD's lifecycle engine works offline, costs nothing, and is deterministic. LLMs orchestrate multi-step workflows but the execution is always PyOD.

**Key difference from AD-AGENT:** AD-AGENT requires OpenAI API ($0.02-0.05/call), is CLI-only, and has a fixed 4-agent architecture. PyOD's approach is **library-centric**: any agent writes Python calling `ADEngine` methods. MCP provides optional read-only access for knowledge queries and planning, but execution always runs through Python. The intelligence is in the engine, not in the orchestration layer.

### 2.1.1 Python-First Lifecycle (revised)

**Key decision (2026-04-08): Tier B is Python-first, not MCP-first.**

On Windows with antivirus (e.g., Bitdefender), MCP server subprocesses are routinely blocked or require approval prompts. Since most PyOD users are on Windows, building the stateful lifecycle around MCP would create a painful default experience.

**New approach:** The full lifecycle lives in `ADEngine` as Python methods. Agents (Claude Code, Copilot, Cursor) call ADEngine directly by writing and executing Python code -- no MCP subprocess needed. The agent holds state in its conversation context (variables in the Python session it's running).

```
Agent (any LLM)                    Python session
  │                                     │
  │── writes: engine = ADEngine() ────►│
  │── writes: profile = engine.       ►│  returns profile dict
  │     profile_data(X_train)           │
  │── writes: plan = engine.          ►│  returns plan dict
  │     plan_detection(profile)         │
  │── writes: result = engine.        ►│  fits detector, returns result dict
  │     run_detection(X_train, plan)    │  with scores, labels, fitted detector
  │── writes: analysis = engine.      ►│  returns analysis dict
  │     analyze_results(result)         │
  │── writes: suggestion = engine.    ►│  returns suggestion with new_plan
  │     suggest_next_step(result,       │
  │       analysis, "too many FPs")     │
  │── (agent adjusts plan, reruns) ───►│  iterate
```

**No `RunSession`, no `run_id`, no server-side state.** The Python objects (fitted detector, score arrays, result dicts) live in the Python session the agent is driving. This is simpler, faster, and works everywhere.

**Execution model:** The lifecycle must run inside a single persistent Python session (script, notebook, or REPL). All modern AI agent CLIs (Claude Code, Cursor, Copilot) maintain a persistent Python interpreter during a session, so this is not a practical limitation. For batch or offline use, `ADEngine.detect()` is a one-shot shortcut that runs the entire lifecycle in one call. For agents that need to hand off state between separate Python invocations, `result` dicts are serializable via `joblib.dump()` / `joblib.load()` (the fitted detector and numpy arrays are all pickle-safe).

**MCP stays as-is (Tier A only):** The existing MCP server provides knowledge queries and planning tools. It does NOT need stateful lifecycle tools. Agents that can run Python (which is all of them in practice) use ADEngine directly for execution.

### 2.2 Layer 1: Knowledge Base

A structured, machine-readable registry serving as the **single source of truth** for all layers. No other layer embeds or duplicates this data.

Location: `pyod/utils/knowledge/`

#### 2.2.1 Algorithm Registry (`algorithms.json`)

Every PyOD detector gets an entry with structured metadata.

```json
{
  "ECOD": {
    "class_path": "pyod.models.ecod.ECOD",
    "full_name": "Empirical Cumulative Distribution Based Outlier Detection",
    "status": "shipped",
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
    "benchmark_refs": ["ADBench"],
    "benchmark_rank": {
      "ADBench_overall": 5,
      "ADBench_high_dim": 2
    },
    "paper": {"id": "ecod", "short": "Li et al., TKDE 2022"},
    "default_params": {"contamination": 0.1},
    "requires": [],
    "version_added": "0.6.0"
  },
  "EmbeddingOD": {
    "class_path": "pyod.models.embedding.EmbeddingOD",
    "full_name": "Embedding-based Outlier Detection",
    "status": "shipped",
    "data_types": ["text", "image"],
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
    "presets": {
      "for_text": "EmbeddingOD.for_text()",
      "for_image": "EmbeddingOD.for_image()"
    },
    "benchmark_refs": ["NLP_ADBench"],
    "benchmark_rank": {
      "NLP_ADBench_overall": 1
    },
    "paper": {"id": "nlp_adbench", "short": "Li et al., EMNLP 2025"},
    "default_params": {"detector": "LUNAR"},
    "preprocessing_mode": "internal",
    "preprocessing_note": "EmbeddingOD manages standardize and reduce_dim internally. Use presets (for_text, for_image) instead of top-level plan preprocessing steps.",
    "requires": ["sentence-transformers OR openai OR transformers+torch"],
    "version_added": "2.1.0"
  },
  "TimeSeriesOD": {
    "class_path": "pyod.models.tsod.TimeSeriesOD",
    "full_name": "Time Series Outlier Detection Bridge",
    "status": "planned",
    "data_types": ["time_series"],
    "category": "bridge",
    "note": "Planned for v2.3.0. Not yet available."
  }
}
```

**Design choices (addressing Codex finding #1, #5, #6):**
- `status` field: `"shipped"` | `"experimental"` | `"planned"`. Layer 2 filters by status -- `plan_detection()` and `build_detector()` only return shipped detectors unless caller passes `include_planned=True`. `list_detectors()` defaults to `status='shipped'` but accepts `status='all'`.
- `default_params` match current library defaults (e.g., EmbeddingOD defaults to `detector='LUNAR'`, not `'KNN'`). The knowledge base describes reality, not aspirations.
- `presets` field links to existing factory methods like `EmbeddingOD.for_text()`.

#### 2.2.2 Benchmark Results (`benchmarks.json`)

```json
{
  "ADBench": {
    "paper": {"id": "adbench", "short": "Han et al., NeurIPS 2022"},
    "scope": "tabular",
    "n_datasets": 57,
    "n_algorithms": 30,
    "rankings": {
      "overall_top_5": ["ECOD", "IForest", "KNN", "COPOD", "HBOS"],
      "high_dim_top_3": ["ECOD", "COPOD", "IForest"],
      "low_dim_top_3": ["KNN", "LOF", "CBLOF"]
    },
    "key_finding": "No single algorithm dominates; ensemble of top-5 is robust"
  },
  "NLP_ADBench": {
    "paper": {"id": "nlp_adbench", "short": "Li et al., EMNLP 2025"},
    "scope": "text",
    "n_datasets": 8,
    "n_algorithms": 19,
    "rankings": {
      "overall_top_5": ["OpenAI+LUNAR", "OpenAI+LOF", "OpenAI+AE", "MiniLM+KNN", "BERT+LOF"]
    },
    "key_finding": "Embedding quality >> detector choice; two-step beats end-to-end"
  },
  "TSB_AD": {
    "paper": {"id": "tsb_ad", "short": "Liu & Paparrizos, NeurIPS 2024"},
    "scope": "time_series",
    "n_datasets": 1070,
    "n_algorithms": 40,
    "rankings": {
      "overall_top_5": ["IForest", "LOF", "POLY", "KNN", "KShapeAD"],
      "subsequence_top_3": ["MatrixProfile", "SAND", "Series2Graph"]
    },
    "key_finding": "Classical methods competitive with deep; MatrixProfile strong on subsequence anomalies"
  }
}
```

#### 2.2.3 Routing Rules (`routing_rules.json`)

**Addressing Codex finding #3:** Structured predicates, not stringly-typed.

```json
{
  "version": 1,
  "rules": [
    {
      "id": "tabular_high_dim_fast",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "tabular"},
        {"field": "n_features", "op": "gte", "value": 100},
        {"field": "priority", "op": "eq", "value": "speed"}
      ],
      "recommendations": [
        {
          "detector": "ECOD",
          "params": {},
          "confidence": 0.9
        },
        {
          "detector": "HBOS",
          "params": {},
          "confidence": 0.85
        },
        {
          "detector": "IForest",
          "params": {},
          "confidence": 0.8
        }
      ],
      "reason": "High-dimensional tabular + speed priority: parameter-free fast methods",
      "evidence": ["ADBench"]
    },
    {
      "id": "tabular_low_dim_small",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "tabular"},
        {"field": "n_features", "op": "lt", "value": 20},
        {"field": "n_samples", "op": "lt", "value": 5000}
      ],
      "recommendations": [
        {"detector": "KNN", "params": {}, "confidence": 0.85},
        {"detector": "LOF", "params": {}, "confidence": 0.8},
        {"detector": "CBLOF", "params": {}, "confidence": 0.75}
      ],
      "reason": "Low-dim small dataset: proximity-based methods excel",
      "evidence": ["ADBench"]
    },
    {
      "id": "text_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "text"}
      ],
      "recommendations": [
        {
          "detector": "EmbeddingOD",
          "params": {},
          "preset": "for_text",
          "confidence": 0.9
        }
      ],
      "reason": "Text data: EmbeddingOD.for_text() with benchmark-informed defaults (NLP-ADBench top performer)",
      "evidence": ["NLP_ADBench"]
    },
    {
      "id": "image_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "image"}
      ],
      "recommendations": [
        {
          "detector": "EmbeddingOD",
          "params": {},
          "preset": "for_image",
          "confidence": 0.85
        }
      ],
      "reason": "Image data: EmbeddingOD.for_image() with DINOv2 vision encoder",
      "evidence": []
    },
    {
      "id": "time_series_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "time_series"}
      ],
      "recommendations": [
        {
          "detector": "TimeSeriesOD",
          "params": {"detector": "IForest"},
          "status_required": "shipped",
          "confidence": 0.85
        }
      ],
      "reason": "Time series: windowed IForest (TSB-AD top performer)",
      "evidence": ["TSB_AD"]
    },
    {
      "id": "multimodal_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "multimodal"}
      ],
      "recommendations": [
        {
          "detector": "MultiModalOD",
          "params": {},
          "confidence": 0.8
        }
      ],
      "reason": "Multi-modal data: score fusion across per-modality detectors",
      "evidence": []
    }
  ]
}
```

Rules reference detectors by name; Layer 2 checks the `status` field in `algorithms.json` before returning any recommendation. If a recommended detector has `status: "planned"`, Layer 2 either skips it or downgrades it with a note ("TimeSeriesOD is planned for v2.3.0; for now, try windowing your data manually and using IForest").

#### 2.2.4 Papers & Citations (`papers.json`)

```json
{
  "pyod": {
    "title": "PyOD: A Python Toolbox for Scalable Outlier Detection",
    "authors": "Zhao, Nasrullah, Li",
    "venue": "JMLR 2019",
    "url": "https://jmlr.org/papers/v20/19-011.html"
  },
  "adbench": {
    "title": "ADBench: Anomaly Detection Benchmark",
    "authors": "Han, Hu, Huang, Jiang, Zhao",
    "venue": "NeurIPS 2022"
  },
  "nlp_adbench": {
    "title": "NLP-ADBench: NLP Anomaly Detection Benchmark",
    "authors": "Li, Li, Xiao, Yang, Nian, Hu, Zhao",
    "venue": "EMNLP Findings 2025"
  },
  "tsb_ad": {
    "title": "TSB-AD: The Elephant in the Room",
    "authors": "Liu, Paparrizos",
    "venue": "NeurIPS 2024"
  },
  "ad_agent": {
    "title": "AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection",
    "authors": "Yang et al.",
    "venue": "arXiv 2025"
  },
  "ecod": {
    "title": "ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions",
    "authors": "Li, Zhao, Hu, Botta",
    "venue": "TKDE 2022"
  }
}
```

### 2.3 Layer 2: Lifecycle Engine (`ADEngine`)

The core Python class that drives the **entire anomaly detection lifecycle**. Renamed from `Advisor` to `ADEngine` to reflect that it doesn't just advise -- it executes.

```python
# pyod/utils/ad_engine.py

class ADEngine:
    """Anomaly detection lifecycle engine.

    Handles the complete anomaly detection workflow: data profiling,
    pipeline planning, detection execution, result analysis, and
    iterative refinement. Works as a standalone Python API (no LLM
    required) or as the backend for MCP/agent interfaces.

    Supersedes ``AutoModelSelector`` (deprecated in v2.2.0).

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge base directory. If None, uses the bundled
        knowledge base shipped with PyOD.

    Examples
    --------
    >>> from pyod.utils.ad_engine import ADEngine
    >>> engine = ADEngine()

    # Full lifecycle in Python (no LLM needed)
    >>> profile = engine.profile_data(X_train)
    >>> plan = engine.plan_detection(profile)
    >>> result = engine.run_detection(X_train, plan)
    >>> analysis = engine.analyze_results(result)
    >>> print(analysis['summary'])

    # One-shot shortcut
    >>> result = engine.detect(X_train)
    """
```

#### 2.3.1 Lifecycle Methods

**Phase 1: Understand**

```python
def profile_data(self, X, data_type=None):
    """Profile the input data.

    Auto-detects data type (unless overridden) and computes
    characteristics relevant to detector selection.

    Parameters
    ----------
    X : array-like, list, dict, or str (file path)
        The input data. Accepted formats:
        - numpy array / pandas DataFrame → tabular or time_series
        - list of str → text
        - list of file paths (images) → image
        - dict of {modality: data} → multimodal

    data_type : str or None
        Explicit override. One of 'tabular', 'text', 'image',
        'time_series', 'multimodal', 'graph'.
        If None, auto-detected with conservative defaults.

    Returns
    -------
    profile : dict
        Keys: 'data_type', 'n_samples', 'n_features',
        'dtype', 'has_nan', 'sparsity', 'dimensionality_class'
        ('low'/'medium'/'high'), 'estimated_memory',
        and type-specific fields.
    """
```

**Data type sniffing (addressing Codex finding #2):**

Conservative defaults. Numeric arrays are **always tabular** unless the caller explicitly says `data_type='time_series'` or provides sequence metadata. No heuristic guessing based on aspect ratios.

```python
def _sniff_data_type(self, X):
    # Unambiguous cases only
    if isinstance(X, dict):
        return 'multimodal'
    if isinstance(X, list) and len(X) > 0:
        if all(isinstance(x, str) for x in X[:20]):
            if _looks_like_image_paths(X[:5]):
                return 'image'
            return 'text'
    # All numeric arrays → tabular (conservative default)
    # User must explicitly pass data_type='time_series'
    return 'tabular'
```

**Phase 2: Plan**

```python
def plan_detection(self, profile, priority='balanced',
                   constraints=None):
    """Plan a detection pipeline based on data profile.

    Parameters
    ----------
    profile : dict
        Output of profile_data().
    priority : str
        'speed', 'accuracy', or 'balanced'.
    constraints : dict or None
        Optional constraints: {'max_runtime_seconds': 60,
        'exclude_detectors': ['DeepSVDD'],
        'require_interpretable': True}

    Returns
    -------
    plan : DetectionPlan (dict with closed schema)
        Allowlisted fields only -- see Plan Schema below.
    """
```

**Plan Schema (addressing Codex Round 2 finding #3):**

Plans use a **closed, declarative schema**. Agents cannot inject arbitrary class paths or preprocessing graphs. All detector names are validated against `algorithms.json`. Unknown fields are rejected with structured validation errors.

```python
# DetectionPlan schema (validated by ADEngine)
{
    "detector_name": str,        # must exist in algorithms.json with status='shipped'
    "preset": str or None,       # e.g., 'for_text', 'for_image' (validated against registry)
    "params": dict,              # detector __init__ kwargs (validated against known params)
    "preprocessing": [           # ordered list of allowlisted steps
        {"step": "standardize"},           # StandardScaler
        {"step": "reduce_dim", "n_components": 50},  # PCA
        {"step": "clip", "floor": -10, "ceil": 10},
        {"step": "nan_to_num"},
    ],
    "threshold_strategy": str,   # 'contamination' (default), 'percentile', 'manual'
    "threshold_value": float,    # interpretation depends on strategy
    "reason": str,               # human-readable (from routing rules)
    "evidence": [str],           # benchmark IDs
    "confidence": float,         # 0.0-1.0
    "alternatives": [...]        # list of runner-up plans (same schema)
}
```

Allowlisted preprocessing steps: `standardize`, `reduce_dim`, `clip`, `nan_to_num`, `normalize`. No arbitrary code execution.

**Preprocessing ownership (addressing Codex Round 3 finding #3):** Each detector in `algorithms.json` has a `preprocessing_mode` field:
- `"external"` (default for most detectors): preprocessing is applied by `ADEngine` before passing data to the detector. Top-level plan preprocessing steps are used.
- `"internal"` (e.g., `EmbeddingOD`): the detector manages its own preprocessing (standardize, PCA, etc.) internally. `ADEngine` rejects overlapping external preprocessing steps and routes through presets instead.
- `"mixed"`: some preprocessing is external, some internal. The `preprocessing_note` field documents which steps are owned by the detector.

This prevents double-applying scaling/PCA and preserves existing library behavior.

**Phase 3: Execute**

```python
def run_detection(self, X_train, plan, X_test=None):
    """Execute a detection plan.

    Parameters
    ----------
    X_train : array-like
        Training data.
    plan : dict (DetectionPlan)
        Output of plan_detection(). Must conform to the closed
        plan schema. Validated before execution -- unknown fields
        or unregistered detectors raise ValueError.
    X_test : array-like or None
        Optional test data. If provided, scores are computed
        for test data after fitting on training data.

    Returns
    -------
    result : dict
        Keys: 'scores_train', 'scores_test' (if X_test),
        'labels_train', 'labels_test' (if X_test),
        'threshold', 'detector' (fitted detector instance),
        'runtime_seconds', 'plan' (the plan used).
    """
```

**Phase 4: Analyze**

```python
def analyze_results(self, result, X=None):
    """Analyze detection results.

    Parameters
    ----------
    result : dict
        Output of run_detection().
    X : array-like or None
        Original data, for computing feature-level explanations.

    Returns
    -------
    analysis : dict
        Keys: 'n_anomalies', 'anomaly_ratio',
        'score_distribution' (mean, std, min, max, quartiles),
        'top_anomalies' (indices + scores of top-k),
        'clustering' (if anomalies cluster in feature/time space),
        'summary' (human-readable paragraph),
        'feature_importance' (if X is tabular, per-feature
        contribution to anomaly scores where supported).
    """
```

**Phase 5: Iterate**

```python
def suggest_next_step(self, result, analysis, feedback=None):
    """Suggest what to try next based on results and user feedback.

    Parameters
    ----------
    result : dict
        Output of run_detection().
    analysis : dict
        Output of analyze_results().
    feedback : str or None
        User feedback, e.g., "too many false positives",
        "missed the anomalies at index 500-510",
        "try a different detector".

    Returns
    -------
    suggestion : dict
        Keys: 'action' ('adjust_threshold', 'try_alternative',
        'adjust_params', 'ensemble', 'done'),
        'new_plan' (if action involves rerunning),
        'reason' (why this suggestion).
    """

def explain_findings(self, result, indices=None, top_k=5):
    """Explain why specific samples were flagged as anomalies.

    Parameters
    ----------
    result : dict
        Output of run_detection().
    indices : list of int or None
        Specific sample indices to explain. If None, explains
        the top-k most anomalous samples.

    Returns
    -------
    explanations : list of dict
        Each with 'index', 'score', 'percentile',
        'contributing_features' (where supported),
        'narrative' (human-readable sentence).
    """
```

**Phase 6: Report**

```python
def generate_report(self, result, analysis, format='text'):
    """Generate a summary report of the detection run.

    Parameters
    ----------
    format : str
        'text' (markdown), 'json', or 'html'.

    Returns
    -------
    report : str
        Complete report with: data profile, detector used,
        parameter settings, result summary, top anomalies,
        recommendations for next steps.
    """
```

**Convenience shortcuts:**

```python
def detect(self, X_train, X_test=None, data_type=None,
           priority='balanced'):
    """One-shot anomaly detection: profile → plan → run → analyze.

    Returns
    -------
    result : dict
        Combined output of run_detection() and analyze_results().
    """
    profile = self.profile_data(X_train, data_type=data_type)
    plan = self.plan_detection(profile, priority=priority)
    result = self.run_detection(X_train, plan, X_test=X_test)
    result['analysis'] = self.analyze_results(result, X=X_train)
    return result
```

**Factory method (addressing Codex Round 2 finding #4):**

```python
def build_detector(self, plan):
    """Build and return an UNFITTED detector instance from a plan.

    For users who want detector selection but not automatic
    execution -- they integrate the detector into their own
    code path. This is the direct replacement for
    AutoModelSelector.get_top_clf().

    Parameters
    ----------
    plan : dict (DetectionPlan)
        Output of plan_detection().

    Returns
    -------
    detector : BaseDetector
        Configured but unfitted detector instance.
    """
```

**Knowledge query methods:**

```python
def list_detectors(self, data_type=None, status='shipped'):
    """List available detectors, filtered by data type and status."""

def explain_detector(self, name):
    """Explain a detector: how it works, strengths, weaknesses."""

def compare_detectors(self, names=None, data_type=None, top_k=3):
    """Compare detectors for a given data type."""

def get_benchmarks(self, benchmark='all'):
    """Get benchmark results."""
```

#### 2.3.2 Migration from AutoModelSelector

**Addressing Codex finding #6.**

`AutoModelSelector` currently uses GPT-4o to tag datasets and select from 10 deep-learning models. The migration path:

| `AutoModelSelector` feature | `ADEngine` equivalent |
|---|---|
| `model_auto_select()` (GPT-4o call) | `plan_detection()` (rule-based, no LLM, covers all 46+ detectors) |
| `get_top_clf()` (returns unfitted detector) | `build_detector()` (returns unfitted detector from plan) |
| `load_model_analyses_labels_only()` | `list_detectors()` (reads from `algorithms.json`) |
| `_MODEL_REGISTRY` (10 models) | `algorithms.json` (46+ models, all categories) |
| `model_analysis_jsons/*.json` (10 files) | `knowledge/algorithms.json` (single unified file) |

**Migration plan:**
- v2.2.0: Ship `ADEngine`. Reimplement `AutoModelSelector` as a thin compatibility shim on top of `ADEngine` (internally calls `profile_data` → `plan_detection` → `build_detector`). Add deprecation warning on import.
- v2.2.0: Migrate `auto_model_selection_example/` notebook to use `ADEngine`.
- v2.3.0: Remove `AutoModelSelector` shim and `model_analysis_jsons/`.
- `model_analysis_jsons/` data (strengths/weaknesses labels) is folded into `algorithms.json` entries.

### 2.4 Layer 3: Agent Interfaces

Thin wrappers around `ADEngine`. The skill file (Layer 3b) does NOT embed knowledge -- it points agents to call `ADEngine` methods or read the JSON files. **(Addressing Codex finding #4.)**

#### 2.4.1 MCP Server (`pyod/mcp_server.py`)

An **optional** agent interface for knowledge queries and planning. Works with Claude Code, VS Code Copilot, Cursor, Gemini, any MCP client. **Not required** -- agents can (and should) call ADEngine directly in Python for the full lifecycle.

**Scope:** MCP provides Tier A tools only (knowledge + planning). Tier B execution methods are Python-only. On Windows with antivirus (Bitdefender), MCP may be blocked -- agents fall back to direct Python import.

**v2.2.0 data support:** tabular, text, and image (shipped modalities). The Python API (`ADEngine`) has no such restriction.

```python
from mcp.server.fastmcp import FastMCP
from pyod.utils.ad_engine import ADEngine

mcp = FastMCP("pyod")
engine = ADEngine()

# ============================================================
# Tier A tools: knowledge queries + stateless planning
# (shipped in v2.2.0)
# ============================================================

@mcp.tool()
def profile_data(
    data_path: str,
    data_type: str = "auto"
) -> str:
    """Profile a dataset for anomaly detection.

    Loads data, detects the data type and characteristics,
    returns a structured profile for use in plan_detection().

    Args:
        data_path: Path to data file (CSV, NPY, JSON, image dir).
        data_type: Override. One of 'tabular', 'text', 'image',
            or 'auto'. (time_series and graph support added in
            later releases when those detectors ship.)
    """
    ...

@mcp.tool()
def plan_detection(
    data_profile: str,
    priority: str = "balanced",
    constraints: str = ""
) -> str:
    """Plan an anomaly detection pipeline.

    Returns a DetectionPlan (closed schema) with detector name,
    params, preprocessing, reason, and benchmark evidence.

    Args:
        data_profile: JSON from profile_data().
        priority: 'speed', 'accuracy', or 'balanced'.
        constraints: Optional JSON, e.g. '{"require_interpretable": true}'.
    """
    ...

@mcp.tool()
def list_detectors(data_type: str = "", status: str = "shipped") -> str:
    """List available PyOD detectors, filtered by data type and status."""
    ...

@mcp.tool()
def explain_detector(name: str) -> str:
    """Explain a PyOD detector: how it works, strengths, weaknesses,
    benchmark performance, and recommended use cases."""
    ...

@mcp.tool()
def compare_detectors(
    names: str = "",
    data_type: str = "tabular",
    top_k: int = 3
) -> str:
    """Compare detectors for a given data type. If names is empty,
    compares the top-k detectors from benchmarks."""
    ...

@mcp.tool()
def get_benchmarks(benchmark: str = "all") -> str:
    """Get benchmark results (ADBench, NLP-ADBench, TSB-AD).
    Returns rankings, key findings, and top performers."""
    ...

@mcp.tool()
def build_detector(plan: str) -> str:
    """Get constructor metadata for a detector from a plan.

    Returns the import path, class name, and validated params
    needed to instantiate the detector. Does NOT return a live
    object (impossible over MCP). The agent uses this to
    generate correct instantiation code.

    Args:
        plan: JSON from plan_detection() (closed schema).

    Returns:
        JSON with 'import_path', 'class_name', 'params',
        'preprocessing_steps', 'code_snippet' (ready-to-run
        Python code for instantiation).
    """
    ...

# ============================================================
# Note: Tier B lifecycle (run, analyze, iterate, report) is
# Python-only via ADEngine. No MCP tools for execution.
# Agents write and run Python code directly.
# See Section 2.1.1 for rationale (Windows/Bitdefender).
# ============================================================
```

#### 2.4.2 Claude Code Skill (`skills/od-expert/SKILL.md`)

**Addressing Codex finding #4:** The skill is a pointer, not a copy.

```markdown
---
name: od-expert
description: Anomaly detection expert. Drives the full PyOD lifecycle
  through the ADEngine API -- profiling, planning, execution, analysis,
  and iterative refinement.
---

You are an anomaly detection expert backed by PyOD's ADEngine.

## When to activate
- User has data and wants anomaly detection
- User asks "which detector should I use?"
- User asks about PyOD algorithms or benchmarks
- User asks to compare detection methods

## How to work
Do NOT embed detection knowledge. Instead, import and call ADEngine
directly in Python:

```python
from pyod.utils.ad_engine import ADEngine
engine = ADEngine()

# Full lifecycle
profile = engine.profile_data(X_train)
plan = engine.plan_detection(profile)
result = engine.run_detection(X_train, plan)
analysis = engine.analyze_results(result)
explanations = engine.explain_findings(result, top_k=5)
# iterate with engine.suggest_next_step(result, analysis, feedback)
report = engine.generate_report(result, analysis)
```

For knowledge queries and planning only (no execution), MCP tools
are also available if the MCP server is running: profile_data,
plan_detection, build_detector, list_detectors, explain_detector,
compare_detectors, get_benchmarks.

## Lifecycle flow
profile_data → plan_detection → run_detection → analyze_results
→ explain_findings → (iterate with suggest_next_step) → generate_report

All lifecycle methods are pure Python on ADEngine. MCP is optional
and limited to knowledge/planning queries.
```

#### 2.4.3 OpenAI Function Schema Export

For integration with AD-AGENT and GPT-based tools. **Limited to Tier A stateless methods only** (knowledge queries + planning). Tier B lifecycle methods are Python-only and should not be exposed as JSON tool-calling surfaces.

```python
class ADEngine:
    def to_openai_tools(self):
        """Export Tier A methods as OpenAI function-calling tools.

        Returns a list of tool dicts compatible with OpenAI's
        chat completions API. Includes: profile_data, plan_detection,
        build_detector, list_detectors, explain_detector,
        compare_detectors, get_benchmarks.

        Does NOT include Tier B execution methods (run_detection,
        analyze_results, etc.) -- those require a persistent Python
        session and should be called directly, not via tool-calling.
        """
        return _generate_openai_schema(self)
```

### 2.5 Relationship to AD-AGENT

AD-AGENT's 4-agent architecture maps to ADEngine's lifecycle:

| AD-AGENT Agent | ADEngine Phase | Key Difference |
|----------------|---------------|----------------|
| **Processor** | `profile_data()` + preprocessing in `run_detection()` | ADEngine: deterministic, no LLM. AD-AGENT: LLM generates preprocessing code. |
| **Detection** | `plan_detection()` + `run_detection()` | ADEngine: rule-based selection from 46+ detectors. AD-AGENT: LLM selects from 10. |
| **Explanation** | `analyze_results()` + `explain_findings()` | ADEngine: structured analysis. AD-AGENT: LLM narration. |
| **Adaptation** | `suggest_next_step()` | ADEngine: rule-based suggestions. AD-AGENT: LLM-driven iteration. |

**The convergence path:** AD-AGENT could use ADEngine as its execution backend instead of generating PyOD code from scratch. The LLM orchestrates the conversation; ADEngine handles the detection. This would make AD-AGENT more reliable (no code generation errors), cheaper (fewer LLM calls), and broader (46+ detectors instead of 10).

### 2.6 File Structure

```
pyod/
├── pyod/
│   ├── utils/
│   │   ├── ad_engine.py             # Layer 2: lifecycle engine
│   │   ├── auto_model_selector.py   # DEPRECATED in v2.2.0 (points to ADEngine)
│   │   ├── knowledge/               # Layer 1: knowledge base
│   │   │   ├── __init__.py
│   │   │   ├── algorithms.json      # 46+ detector entries with status field
│   │   │   ├── benchmarks.json      # ADBench, NLP-ADBench, TSB-AD
│   │   │   ├── routing_rules.json   # structured predicates
│   │   │   └── papers.json          # citations
│   │   └── model_analysis_jsons/    # DEPRECATED (data migrated to algorithms.json)
│   │       └── *.json
│   └── mcp_server.py               # Layer 3a: MCP server
├── skills/
│   └── od-expert/
│       └── SKILL.md                 # Layer 3b: pointer skill (no embedded knowledge)
├── examples/
│   ├── ad_engine_example.py         # Python lifecycle demo
│   └── mcp_usage_example.md         # Agent interaction transcript
└── setup.py                         # Updated extras_require
```

### 2.7 Packaging

**Addressing Codex finding #7.**

**New `package_data`:**
```python
package_data={
    'pyod.utils': ['model_analysis_jsons/*.json'],      # deprecated, keep for one release
    'pyod.utils.knowledge': ['*.json'],                  # new knowledge base
},
```

**New extras:**
```python
extras_require={
    'embedding': ['sentence-transformers>=2.0'],
    'openai': ['openai>=1.0'],
    'mcp': ['mcp>=1.0'],                                # new: MCP server
    'all': [
        'sentence-transformers>=2.0',
        'openai>=1.0',
        'transformers>=4.0',
        'torch>=2.0',
        'Pillow',
        'mcp>=1.0',
    ],
},
```

**MCP server launch:**
```bash
# Option 1: module entry point
python -m pyod.mcp_server

# Option 2: console script (if we add entry_points)
pyod-mcp
```

**Python version:** MCP server uses `list[str]` (3.9+ syntax). The MCP extra requires Python >= 3.9. Core PyOD remains compatible with 3.8+. The MCP server file uses `from __future__ import annotations` for 3.8 compatibility, or we narrow support -- decision deferred to implementation.

### 2.8 Implementation Phases

Two scope tiers. Tier A is the minimum viable release; Tier B completes the full lifecycle vision.

#### Tier A: Knowledge + Core Engine + Read-Only MCP (v2.2.0-alpha)

| Phase | What | Effort | Delivers |
|-------|------|--------|----------|
| **A1** | Knowledge base: `algorithms.json` for all 46+ shipped detectors | 1-2 days | Foundation for everything |
| **A2** | Knowledge base: `benchmarks.json`, `routing_rules.json`, `papers.json` | 1 day | Complete knowledge layer |
| **A3** | `ADEngine` core: `profile_data()`, `plan_detection()`, `build_detector()`, `detect()` shortcut | 2-3 days | Core engine (stateless) |
| **A4** | `ADEngine` knowledge queries: `list_detectors()`, `explain_detector()`, `compare_detectors()`, `get_benchmarks()` | 1 day | Knowledge access |
| **A5** | `AutoModelSelector` compatibility shim + deprecation warning | 0.5 day | Clean migration |
| **A6** | MCP server: knowledge tools + `profile_data` + `plan_detection` + `build_detector` (returns constructor metadata, not live objects) | 1-2 days | Agents can query + plan + get instantiation code |
| **A7** | Claude Code skill (pointer) + OpenAI schema export | 1 day | Multi-agent support |
| **A8** | Tests + examples + docs for Tier A | 2-3 days | Ship-ready |

**Tier A total: ~10-13 days**

#### Tier B: Full Python Lifecycle (v2.2.0 or v2.3.0)

**Revised (2026-04-08): Python-first, no MCP session state.**

All Tier B methods are pure Python on `ADEngine`. Agents call them by writing Python code -- no MCP subprocess, no RunSession, no run_id. This works on all platforms including Windows with Bitdefender.

| Phase | What | Effort | Delivers |
|-------|------|--------|----------|
| **B1** | `ADEngine.run_detection(X_train, plan, X_test)` -- full execution, returns result dict with scores, labels, fitted detector | 2-3 days | Core execution |
| **B2** | `ADEngine.analyze_results(result, X)` -- score distribution, top anomalies, clustering, feature importance, narrative summary | 2-3 days | Analysis |
| **B3** | `ADEngine.explain_findings(result, indices, top_k)` -- per-sample explanations with contributing features | 1-2 days | Explainability |
| **B4** | `ADEngine.suggest_next_step(result, analysis, feedback)` -- rule-based iteration suggestions (adjust threshold, try alternative, ensemble) | 1-2 days | Iteration |
| **B5** | `ADEngine.generate_report(result, analysis, format)` -- markdown/json/html summary report | 1 day | Reporting |
| **B6** | Update `od-expert` skill to guide agents through the full Python lifecycle | 1 day | Agent experience |
| **B7** | Tests + examples + docs for Tier B | 2-3 days | Ship-ready |

**Tier B total: ~10-15 days**

**Combined total: ~20-28 days**

Tier A is independently useful -- agents can query PyOD's knowledge, get recommendations, and receive configured detector instances. Tier B adds the analysis, explanation, iteration, and reporting that makes the vision example work end-to-end. All through Python -- no MCP complexity.

---

## 3. Track 2: Time Series Anomaly Detection

### 3.1 Motivation

- **TODS is dead** (no updates since Sep 2023). No maintained, lightweight TS-AD library exists.
- **Merlion** (Salesforce, 4.5k stars) is actively maintained but is a heavy full-stack ecosystem -- overkill for most users.
- **TSB-AD** (NeurIPS 2024, 244 stars) is an excellent *benchmark* (40 algorithms, 1070 datasets) but not a user-facing library.
- TSB-AD already wraps PyOD for many of its statistical detectors.
- **The gap**: a practitioner-friendly TS-AD tool with PyOD's API.

### 3.2 Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target user | **Practitioner first** | PyOD brand is "practical first, comprehensive second" |
| Dependencies | **No external TS library dependency** | Keep PyOD self-contained |
| Algorithm source | Port curated algorithms natively (numpy/scipy/sklearn, optional PyTorch) | Consistent with existing PyOD dependency philosophy |
| Relationship to TSB-AD | Use benchmark results to pick algorithms; cite paper | "TSB-AD tells you what works, PyOD lets you deploy it" |

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

- **MatrixProfile** -- distance-profile based
- **Series2Graph** -- graph-of-subsequences
- **SAND** -- online streaming detection

#### Foundation model bridge for TS

```python
clf = TimeSeriesOD(encoder='chronos', detector='KNN')
```

### 3.4 Open Technical Questions

1. **Windowing strategy**: Fixed sliding window for v1
2. **Score aggregation**: Window → point mapping (max, mean, weighted?)
3. **Univariate vs multivariate**: Same class or separate?
4. **Streaming/online**: Batch-only for v1
5. **Evaluation**: Defer TS-specific metrics
6. **Which TSB-AD algorithms to port**: Review benchmark for top performers

### 3.5 Integration with ADEngine

When TimeSeriesOD ships, the agent layer gets TS support with minimal changes:
- `algorithms.json`: flip `TimeSeriesOD` status from `"planned"` to `"shipped"`
- `routing_rules.json`: TS rules already exist, now pass the status check
- `ADEngine.profile_data()`: TS profiling (already handles explicit `data_type='time_series'`)
- MCP: update `profile_data` docstring to include `'time_series'` as accepted override; no architectural changes needed
- Skill: no changes needed (points to ADEngine, which picks up TS automatically)

---

## 4. Track 3: Graph Anomaly Detection

### 4.1 Motivation

- **PyGOD** (1.5k stars, same creator) has 18 algorithms but is slow-moving.
- Most PyGOD algorithms require PyTorch Geometric -- heavy dependency.
- Non-GNN methods (SCAN, Radar, ANOMALOUS) could work with scipy sparse matrices.
- EmbeddingOD bridge approach: graph encoder (node2vec, GNN) → PyOD detector.

### 4.2 Design Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Priority | **After time series and agent layer** | TS has a clearer gap; PyGOD still exists |
| Dependencies | **No PyGOD dependency** | Library is slow-moving |

### 4.3 Open Questions

- Absorb PyGOD algorithms or build a graph encoder bridge?
- Which algorithms worth porting without PyG?
- Should PyGOD be deprecated in favor of PyOD's graph module?

---

## 5. Relationship to Existing Work

| Project | Role | Relationship |
|---------|------|-------------|
| **PyOD** (v2.1.0) | Core detection library | This project |
| **AutoModelSelector** | GPT-4o model selection (10 models) | **Superseded by ADEngine** in v2.2.0 |
| **PyGOD** (v1.1.0) | Graph AD | Same creator; may absorb later |
| **TODS** | Time series AD | Dead; PyOD fills the gap |
| **AD-AGENT** | Multi-agent framework | Same group; ADEngine can serve as backend |
| **ADBench** | Tabular benchmark | Knowledge base source |
| **NLP-ADBench** | Text benchmark | Knowledge base source |
| **TSB-AD** | TS benchmark | Knowledge base source (external) |
| **AD-LLM** | LLM zero-shot AD | Future LLMAD integration |
| **MetaOD** | Data-driven model selection | Future ADEngine enhancement |
| **Anomalib** | Image AD (Intel) | Complementary; EmbeddingOD covers this |
| **Merlion** | Time series (Salesforce) | Too heavy for most users |

---

## 6. Release Roadmap

| Version | Content | Theme |
|---------|---------|-------|
| v2.1.0 | EmbeddingOD, MultiModalOD | **Multi-modal foundation** (shipped) |
| v2.2.0 | ADEngine + MCP server + knowledge base | **Intelligent agent layer** |
| v2.3.0 | TimeSeriesOD + native TS detectors | **Time series support** |
| v2.4.0 | Graph AD (bridge + native detectors) | **Graph support** |
| v3.0.0 | LLMAD, MetaOD integration, full platform | **PyOD 3.0: The Platform** |

---

## 7. Codex Review Resolution (Round 1)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Agent layer recommends unshipped detectors | **Resolved** | Added `status` field to algorithm registry; Layer 2 filters by status |
| 2 | `sniff_data_type()` misclassifies tabular as TS | **Resolved** | Conservative default: all numeric arrays → tabular unless explicit override |
| 3 | Knowledge base schema is stringly-typed | **Resolved** | Replaced with structured `{field, op, value}` predicates in `routing_rules.json` |
| 4 | Claude skill duplicates Layer 1 | **Resolved** | Skill is now a pointer that instructs agents to call ADEngine or read JSON files |
| 5 | API inconsistencies + conflicts with library defaults | **Resolved** | Unified API (`ADEngine`); defaults match current library; added `get_benchmarks()` to formal API |
| 6 | Ignores existing `AutoModelSelector` | **Resolved** | Added migration plan: deprecate in v2.2.0, remove in v2.3.0, fold data into `algorithms.json` |
| 7 | Packaging story incomplete | **Resolved** | Added packaging section: `package_data`, `mcp` extra, launch command, Python version notes |

## 8. Codex Review Resolution (Round 2)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| New 1 | MCP can't carry fitted detector state as JSON | **Resolved** | Added `run_id` + server-side `RunSession` store; MCP returns compact summaries, full state stays server-side |
| New 2 | MCP input model doesn't cover multimodal/graph/exclusion | **Resolved** | Scoped MCP v2.2.0 to tabular/text/image only; multimodal/graph manifest deferred to Tier B (Phase B6) |
| New 3 | Plan schema too open-ended for agent-authored plans | **Resolved** | Closed `DetectionPlan` schema with allowlisted fields; detector names validated against `algorithms.json`; preprocessing limited to allowlist |
| New 4 | AutoModelSelector migration needs `build_detector()` | **Resolved** | Added `build_detector(plan)` returning unfitted instance; migration table updated; compatibility shim for one release |
| New 5 | 12-16 day estimate optimistic | **Resolved** | Split into Tier A (10-13 days, knowledge + core + read-only MCP) and Tier B (10-16 days, stateful lifecycle); each tier independently shippable |
| Reopened 1 | Stale `recommend()`/`build()` references from v2 | **Resolved** | Updated to v3 method names (`plan_detection()`, `build_detector()`, `list_detectors()`) |

---

## 9. Codex Review Resolution (Round 3)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| New 1 | RunSession lifecycle underspecified | **Resolved** | Formal `RunSession` schema; TTL with refresh; scavenging; memory cap (10 sessions); structured `run_not_found` error; `data_path` stored instead of raw X_train; analysis caching policy |
| New 2 | Tier A/B boundary leaks | **Resolved** | MCP `build_detector` returns constructor metadata + code snippet (not live objects); `profile_data` docstring aligned to shipped modalities; phase A6 description updated |
| New 3 | Preprocessing ownership ambiguity | **Resolved** | Added `preprocessing_mode` field (`external`/`internal`/`mixed`) to algorithm registry; ADEngine rejects overlapping external preprocessing for `internal` detectors; routes through presets instead |

---

## 10. Codex Review Resolution (Round 4)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| New 1 | Analysis caching contradiction in RunSession vs payload table | **Resolved** | Payload table now shows analysis as cached server-side with invalidation on rerun |
| New 2 | Exclusion-window examples mixed into current Tier B scope | **Resolved** | Vision example annotated as post-B6 future-scope; Tier B v1 iteration is via suggest_next_step + plain rerun |
| New 3 | Stale labels (header, Tier A summary, TS note) | **Resolved** | Header updated to v5 "Design complete"; Tier A summary distinguishes Python vs MCP build_detector; TS note says "no architectural changes" not "no changes" |

**Note (2026-04-08):** Rounds 2-4 findings about RunSession, run_id, MCP session state, and TTL/scavenging are **superseded** by the Python-first redesign (Section 2.1.1). Those resolutions were correct for the MCP-first architecture but are no longer applicable. The underlying concerns (state management, serialization, cleanup) are now handled by Python object lifetime in a persistent interpreter session.

**Review concluded.** 4 design rounds (18 findings resolved). Spec revised to Python-first Tier B on 2026-04-08 (v6) after implementation experience showed MCP friction on Windows. Implementation code review (3 rounds, 9 findings) is tracked separately in the codebase commit history.

---

## Appendix A: TSB-AD Algorithm Reference

40 algorithms across 3 categories (NeurIPS 2024):

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

## Appendix C: AD-AGENT Architecture Reference

Four agents (arXiv:2505.12594):
- **Processor**: data preprocessing (LLM generates code)
- **Detection**: model selection + execution (LLM selects from 10 models)
- **Explanation**: result interpretation (LLM narration)
- **Adaptation**: iterative refinement (WIP)

Limitations: OpenAI-only, CLI-only, 10 models, $0.02-0.05/call.
