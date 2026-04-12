# PyOD V3 Documentation Redesign

**Date:** 2026-04-12
**Status:** Draft (v1)

---

## 1. Goal

Restructure all user-facing documentation around two axes:
1. **Full modality coverage** (tabular, time series, graph, text, image) — on top
2. **Three layers of usage** (classic API, intelligent engine, agentic workflow) — the narrative spine

The current docs lead with the 2019-era "5 lines of code" ECOD example. V3 needs to tell a bigger story: PyOD is not just a bag of detectors, it is an intelligent anomaly detection platform that any agent can drive.

---

## 2. Three-Layer Positioning

| Layer | Name | Audience | Hero Example |
|-------|------|----------|-------------|
| 1 | Classic API | Experts who know which detector they want | `IForest().fit(X); scores = clf.decision_scores_` |
| 2 | Smart Engine | Users who want PyOD to choose + compare | `state = ADEngine().investigate(X); print(state.quality)` |
| 3 | Agentic | AI agents driving OD through conversation | Full investigate → iterate → report loop |

Layer 3 is the **crown jewel** — the differentiator that no other library offers. It should be the most prominent example, not buried in a utility table.

---

## 3. README.rst Restructure

### Current structure (problems)
1. Badges
2. "Read Me First" — V2 features, EmbeddingOD
3. "About PyOD" — history, 2017 origins
4. 5-line ECOD example (Layer 1 hero)
5. EmbeddingOD text example
6. "Selecting the Right Algorithm"
7. Citations
8. Benchmarks
9. Algorithm tables (tabular, TS, graph)
10. API cheatsheet
11. Quick Start tutorial

**Problems:** Layer 2/3 invisible. ADEngine mentioned only in utility table footnote. No modality overview. No positioning of when to use which approach.

### Proposed structure

```
1. Title + badges
2. "Read Me First" (rewritten)
   - One paragraph: "PyOD detects anomalies across tabular, time series,
     graph, text, and image data. 60+ algorithms. Three ways to use it."
   - Modality icons/badges: tabular | TS | graph | text | image
3. "Three Ways to Use PyOD" (NEW — the narrative core)
   3a. Layer 1: Classic (5 lines) — existing ECOD example
   3b. Layer 2: Smart Engine — ADEngine investigate() example
   3c. Layer 3: Agentic Investigation (crown jewel — conversation dialogue)
4. "About PyOD" (shortened, moved down)
5. Installation
6. Benchmarks (compact, already done)
7. Implemented Algorithms (existing tables, unchanged)
8. API Cheatsheet (existing, unchanged)
9. Quick Start (existing, unchanged)
10. Citations
```

### Section 3 content (the core change)

**3a. Layer 1: Classic API (5 lines)**

```rst
**Layer 1: Pick a detector, fit, predict** (``pip install pyod``):

.. code-block:: python

    from pyod.models.iforest import IForest
    clf = IForest()
    clf.fit(X_train)
    y_scores = clf.decision_scores_        # training anomaly scores
    y_test_scores = clf.decision_function(X_test)  # test scores

Works for tabular and most time series detectors. Graph detectors (``pyg_`` prefix)
and MatrixProfile are transductive — use ``decision_scores_`` after ``fit()``,
no out-of-sample ``predict``. Full algorithm list below.
```

**3b. Layer 2: Smart Engine**

```rst
**Layer 2: Let PyOD choose, compare, and assess** (``pip install pyod``):

.. code-block:: python

    from pyod.utils.ad_engine import ADEngine
    engine = ADEngine()
    state = engine.investigate(X)           # runs top-3 detectors, compares
    print(state.analysis['summary'])        # "47 anomalies detected (4.7%)..."
    print(state.quality['verdict'])         # "high" — results are trustworthy

ADEngine profiles your data, selects benchmark-backed detectors, runs them,
computes consensus scores, and assesses result quality — all in one call.
```

**3c. Layer 3: Agentic Investigation (crown jewel — conversation format)**

The Layer 3 hero is a **dialogue**, not a code block. This is the canonical presentation for both README and docs/index.rst.

```rst
**Layer 3: Any AI agent becomes an OD expert**

.. code-block:: text

    User: I have server metrics from the last 24 hours. Find anomalies.

    Agent: I profiled your data: 1440 samples, 12 features (tabular).
           Running 3 detectors: IForest, ECOD, KNN...
           47 anomalies detected (3.3%). Detectors agree (Spearman 0.91).
           Quality: high. Top anomaly at timestamp 03:42.

    User: Too many. Focus on the critical ones only.

    Agent: Lowering threshold to top 1%... 14 anomalies detected.
           Concentrated between 03:30-04:15. Quality: high.

    User: What is driving these?

    Agent: Top contributing features: cpu_usage (z=4.2),
           memory_pressure (z=3.8), disk_io_wait (z=3.1).

    User: Good. Give me the report.

    Agent: [Generates investigation report with consensus scores,
           detector comparison, and anomaly details]

Behind the scenes, the agent calls ``ADEngine().investigate(data)``
then ``iterate()``, ``run()``, ``analyze()``, ``report()`` —
following ``state.next_action`` at each step. No OD knowledge required.
See `agentic example <examples/agentic_example.py>`_.
```

---

## 4. docs/index.rst Restructure

Mirror the README changes but with Sphinx roles:

1. Change title from "Welcome to PyOD V2 documentation!" to "PyOD Documentation"
2. Add "Three Ways to Use PyOD" section after "Read Me First" — Layer 1 and 2 as code blocks, Layer 3 as the same conversation dialogue format from README (using `.. code-block:: text`)
3. Add a "Data Types" summary section listing all modalities with detector counts
4. Keep existing algorithm tables unchanged
5. Reorganize toctree to add a "Getting Started" section that links to the three-layer examples

---

## 5. New Example Files

### 5a. `examples/investigate_example.py` (Layer 2 hero)

```python
"""Example: One-shot anomaly investigation with ADEngine.

ADEngine profiles your data, selects the best detectors,
runs them, computes consensus, and assesses result quality.
"""
from pyod.utils.ad_engine import ADEngine
from pyod.utils.data import generate_data

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data(
        n_train=500, n_test=200, contamination=0.1)

    engine = ADEngine()
    state = engine.investigate(X_train)

    print("=== Investigation Summary ===")
    print(state.analysis['summary'])
    print()
    print("Quality: %s (%.2f)" % (
        state.quality['verdict'], state.quality['overall']))
    print("Detectors used: %d" % state.consensus['n_detectors'])
    print("Agreement: %.2f" % state.consensus['agreement'])
    print()
    print("=== Top 5 Anomalies ===")
    for a in state.analysis['consensus_analysis']['top_anomalies'][:5]:
        print("  Index %d, score %.4f" % (a['index'], a['score']))
```

### 5b. Agentic example (Layer 3 crown jewel)

The crown jewel is NOT a code example — it is a **conversation demo** showing how users interact with an AI agent backed by PyOD. This goes in README.rst and docs as a dialogue, not a script.

**Format: conversation between user and agent**

```rst
**Layer 3: Any AI agent becomes an OD expert**

.. code-block:: text

    User: I have server metrics from the last 24 hours. Find anomalies.

    Agent: I profiled your data: 1440 samples, 12 features (tabular).
           Running 3 detectors: IForest, ECOD, KNN...
           Done. 47 anomalies detected (3.3%). Detectors agree
           (Spearman 0.91). Quality: high.
           Top anomaly at timestamp 03:42 (score 0.98).

    User: Too many. Focus on the critical ones only.

    Agent: Lowering threshold to top 1%... Re-running...
           14 anomalies detected (1.0%). Quality: high.
           Concentrated between 03:30-04:15.

    User: What is driving these?

    Agent: Top contributing features:
           - cpu_usage (z-score 4.2)
           - memory_pressure (z-score 3.8)
           - disk_io_wait (z-score 3.1)

    User: Good. Give me the report.

    Agent: [Generates markdown report with consensus scores,
           detector comparison, and top anomalies]

Behind the scenes, the agent calls ``ADEngine().investigate(data)``
then ``iterate()``, ``run()``, ``analyze()``, ``report()`` —
following ``state.next_action`` at each step. No OD knowledge required.
See `agentic example <examples/agentic_example.py>`_.
```

The canonical Layer 3 dialogue is Section 3c above. This section defines the expanded `examples/agentic_example.py` — a runnable script that simulates the conversation programmatically (prints dialogue with actual ADEngine calls behind each step). The example file may include more detail than the README hero, but the README/docs always use Section 3c's version.

---

## 6. docs/example.rst Reorganization

### Current structure
- Featured Tutorials (external links)
- EmbeddingOD example
- kNN example
- Model Combination example
- Thresholding example

### Proposed structure
```
1. Featured Tutorials (keep external links)
2. Three Approaches to PyOD
   2a. Layer 1: Classic fit/predict (link to knn_example.py)
   2b. Layer 2: ADEngine investigation (link to investigate_example.py)
   2c. Layer 3: Agentic workflow (link to agentic_example.py, with full walkthrough)
3. Examples by Data Type
   3a. Tabular (links to all tabular examples)
   3b. Time Series (links to ts_* examples)
   3c. Graph (links to pyg_* examples)
   3d. Text/Image (link to embedding_od_example.py)
4. Advanced Topics
   4a. Model Combination
   4b. Thresholding
   4c. Model Persistence
```

---

## 7. Scope

**In scope:**
- README.rst restructure (major)
- docs/index.rst restructure (major)
- docs/example.rst reorganization (major)
- `examples/investigate_example.py` (new)
- `examples/agentic_example.py` (new, crown jewel)
- Update `examples/ad_engine_example.py` header to say "Layer 2"

**Not in scope (deferred):**
- New notebooks (can come later)
- Per-modality ADEngine examples (investigate works for all)
- MCP integration examples (Python-first)
- Sphinx theme changes
- conf.py changes
- API reference changes (auto-generated)

---

## 8. Key Messaging

**One-liner:** "PyOD: Anomaly Detection for Any Data, Any Workflow"

**Elevator pitch:** "60+ detectors across tabular, time series, graph, text, and image data. Three ways to use it: classic fit/predict for experts, ADEngine for intelligent orchestration, and a full agentic workflow where AI agents drive the investigation."

**What makes V3 different:** "V3 adds an investigation engine that any AI agent can use to do expert-level anomaly detection without knowing OD. Profile data, compare detectors, assess quality, iterate on feedback — all through a guided workflow."
