# ADEngine Tier A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship PyOD's intelligent agent layer (Tier A): a structured knowledge base of all 46+ detectors, an `ADEngine` class for data profiling and detection planning, and an MCP server that lets any LLM agent query PyOD's expertise.

**Architecture:** Three layers: (1) Knowledge base (`pyod/utils/knowledge/*.json`) as single source of truth, (2) `ADEngine` Python class reading the knowledge base for profiling, planning, and detector construction, (3) MCP server + Claude Code skill as thin wrappers around ADEngine. Tier A is stateless -- no `run_id` sessions (that's Tier B).

**Tech Stack:** Python 3.8+, numpy, scikit-learn, PyOD BaseDetector, `mcp` SDK (optional extra), JSON for knowledge base.

**Spec:** `docs/superpowers/specs/2026-04-07-pyod-expansion-design.md` (v5)

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `pyod/utils/knowledge/__init__.py` | Knowledge base loader (reads JSON files, caches in memory) |
| `pyod/utils/knowledge/algorithms.json` | Algorithm registry: 46+ detectors with metadata, status, benchmarks |
| `pyod/utils/knowledge/benchmarks.json` | Benchmark results: ADBench, NLP-ADBench, TSB-AD |
| `pyod/utils/knowledge/routing_rules.json` | Structured routing predicates for detection planning |
| `pyod/utils/knowledge/papers.json` | Paper citations |
| `pyod/utils/ad_engine.py` | ADEngine lifecycle class (profile, plan, build, detect, knowledge queries) |
| `pyod/mcp_server.py` | MCP server wrapping ADEngine (Tier A tools only) |
| `pyod/test/test_ad_engine.py` | Tests for ADEngine |
| `pyod/test/test_knowledge.py` | Tests for knowledge base loading and validation |
| `skills/od-expert/SKILL.md` | Claude Code pointer skill |
| `examples/ad_engine_example.py` | Usage example |

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/auto_model_selector.py` | Add deprecation warning, thin shim on ADEngine |
| `setup.py` | Add `knowledge/*.json` to `package_data`, add `mcp` extra |
| `MANIFEST.in` | Include `knowledge/*.json` |

---

## Task 1: Knowledge Base Loader

**Files:**
- Create: `pyod/utils/knowledge/__init__.py`
- Test: `pyod/test/test_knowledge.py`

- [ ] **Step 1: Write the failing test for knowledge loader**

```python
# pyod/test/test_knowledge.py
# -*- coding: utf-8 -*-

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.knowledge import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_loads_algorithms(self):
        algos = self.kb.algorithms
        assert isinstance(algos, dict)
        assert len(algos) > 40  # we have 46+ detectors

    def test_algorithm_has_required_fields(self):
        algos = self.kb.algorithms
        required = {'class_path', 'full_name', 'status', 'data_types',
                    'category', 'strengths', 'weaknesses'}
        for name, entry in algos.items():
            for field in required:
                assert field in entry, \
                    f"Algorithm '{name}' missing field '{field}'"

    def test_algorithm_status_values(self):
        for name, entry in self.kb.algorithms.items():
            assert entry['status'] in ('shipped', 'experimental', 'planned'), \
                f"Algorithm '{name}' has invalid status '{entry['status']}'"

    def test_loads_benchmarks(self):
        benchmarks = self.kb.benchmarks
        assert isinstance(benchmarks, dict)
        assert 'ADBench' in benchmarks
        assert 'NLP_ADBench' in benchmarks

    def test_loads_routing_rules(self):
        rules = self.kb.routing_rules
        assert isinstance(rules, dict)
        assert 'rules' in rules
        assert len(rules['rules']) > 0

    def test_routing_rule_has_required_fields(self):
        for rule in self.kb.routing_rules['rules']:
            assert 'id' in rule
            assert 'conditions' in rule
            assert 'recommendations' in rule
            for cond in rule['conditions']:
                assert 'field' in cond
                assert 'op' in cond
                assert 'value' in cond

    def test_loads_papers(self):
        papers = self.kb.papers
        assert isinstance(papers, dict)
        assert 'pyod' in papers

    def test_get_algorithm(self):
        algo = self.kb.get_algorithm('ECOD')
        assert algo is not None
        assert algo['status'] == 'shipped'
        assert 'tabular' in algo['data_types']

    def test_get_algorithm_missing_returns_none(self):
        assert self.kb.get_algorithm('NonExistent') is None

    def test_list_by_data_type(self):
        tabular = self.kb.list_by_data_type('tabular')
        assert len(tabular) > 30  # most detectors are tabular
        text = self.kb.list_by_data_type('text')
        assert 'EmbeddingOD' in [a['name'] for a in text]

    def test_list_by_status(self):
        shipped = self.kb.list_by_status('shipped')
        assert len(shipped) >= 46
        planned = self.kb.list_by_status('planned')
        # TimeSeriesOD should be planned
        names = [a['name'] for a in planned]
        assert 'TimeSeriesOD' in names

    def test_caching(self):
        # Second access should return the same object (cached)
        a1 = self.kb.algorithms
        a2 = self.kb.algorithms
        assert a1 is a2


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_knowledge.py -v`
Expected: FAIL with `ImportError: cannot import name 'KnowledgeBase'`

- [ ] **Step 3: Write the knowledge base loader**

```python
# pyod/utils/knowledge/__init__.py
# -*- coding: utf-8 -*-
"""Knowledge base for PyOD's intelligent agent layer.

Loads structured JSON files containing algorithm metadata,
benchmark results, routing rules, and paper citations.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import json
import os


class KnowledgeBase:
    """Loader and accessor for PyOD's structured knowledge base.

    Reads JSON files from the knowledge directory and provides
    query methods for algorithm metadata, benchmarks, and routing.

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge directory. If None, uses the bundled
        directory shipped with PyOD.
    """

    def __init__(self, knowledge_dir=None):
        if knowledge_dir is None:
            knowledge_dir = os.path.join(
                os.path.dirname(__file__))
        self._dir = knowledge_dir
        self._algorithms = None
        self._benchmarks = None
        self._routing_rules = None
        self._papers = None

    def _load_json(self, filename):
        path = os.path.join(self._dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def algorithms(self):
        if self._algorithms is None:
            self._algorithms = self._load_json('algorithms.json')
        return self._algorithms

    @property
    def benchmarks(self):
        if self._benchmarks is None:
            self._benchmarks = self._load_json('benchmarks.json')
        return self._benchmarks

    @property
    def routing_rules(self):
        if self._routing_rules is None:
            self._routing_rules = self._load_json('routing_rules.json')
        return self._routing_rules

    @property
    def papers(self):
        if self._papers is None:
            self._papers = self._load_json('papers.json')
        return self._papers

    def get_algorithm(self, name):
        """Get algorithm metadata by name. Returns None if not found."""
        return self.algorithms.get(name)

    def list_by_data_type(self, data_type, status='shipped'):
        """List algorithms supporting a given data type."""
        results = []
        for name, entry in self.algorithms.items():
            if data_type in entry.get('data_types', []):
                if status == 'all' or entry.get('status') == status:
                    results.append({'name': name, **entry})
        return results

    def list_by_status(self, status):
        """List algorithms with a given status."""
        results = []
        for name, entry in self.algorithms.items():
            if entry.get('status') == status:
                results.append({'name': name, **entry})
        return results
```

- [ ] **Step 4: Run test to verify it fails (JSON files don't exist yet)**

Run: `python -m pytest pyod/test/test_knowledge.py -v`
Expected: FAIL with `FileNotFoundError` for `algorithms.json`

- [ ] **Step 5: Commit loader (tests will pass after Task 2 provides JSON files)**

```bash
git add pyod/utils/knowledge/__init__.py pyod/test/test_knowledge.py
git commit -m "feat: add knowledge base loader for ADEngine (tests pending JSON data)"
```

---

## Task 2: Algorithm Registry (`algorithms.json`)

This is the largest single task -- populating metadata for all 46+ detectors.

**Files:**
- Create: `pyod/utils/knowledge/algorithms.json`

- [ ] **Step 1: Generate the algorithm registry**

The registry must include every detector in `pyod/models/`. Each entry needs: `class_path`, `full_name`, `status`, `data_types`, `category`, `complexity`, `strengths`, `weaknesses`, `best_for`, `avoid_when`, `benchmark_refs`, `benchmark_rank`, `paper`, `default_params`, `preprocessing_mode`, `requires`, `version_added`.

For the 10 models that have existing `model_analysis_jsons`, fold their strengths/weaknesses labels into the new format.

The complete JSON file is large (~46 entries). Write it with accurate metadata for each detector. Here is the structure -- every shipped model in `pyod/models/` gets an entry:

```json
{
  "ECOD": {
    "class_path": "pyod.models.ecod.ECOD",
    "full_name": "Empirical Cumulative Distribution Based Outlier Detection",
    "status": "shipped",
    "data_types": ["tabular"],
    "category": "probabilistic",
    "complexity": {"time": "O(n*d)", "space": "O(n*d)"},
    "strengths": ["Parameter-free", "Fast on high-dimensional data", "Interpretable per-feature scores"],
    "weaknesses": ["Assumes feature independence", "Struggles with complex feature interactions"],
    "best_for": "High-dimensional tabular data where speed and interpretability matter",
    "avoid_when": "Strong feature correlations exist",
    "benchmark_refs": ["ADBench"],
    "benchmark_rank": {"ADBench_overall": 5, "ADBench_high_dim": 2},
    "paper": {"id": "ecod", "short": "Li et al., TKDE 2022"},
    "default_params": {"contamination": 0.1},
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.6.0"
  }
}
```

**All models to include** (grouped by category):

Probabilistic: ECOD, ABOD, COPOD, MAD, SOS, QMCD, KDE, Sampling, GMM
Linear: PCA, KPCA, MCD, CD, OCSVM, LMDD
Proximity: LOF, COF, CBLOF, LOCI, HBOS, HDBSCAN, KNN, SOD, ROD
Ensemble: IForest, INNE, DIF, FeatureBagging, LSCP, LODA, SUOD, XGBOD
Deep Learning: AutoEncoder, VAE, SO_GAAL, MO_GAAL, DeepSVDD, AnoGAN, ALAD, AE1SVM, DevNet
Graph-based: RGraph, LUNAR
Embedding: EmbeddingOD (preprocessing_mode: internal), MultiModalOD (preprocessing_mode: internal)
Planned: TimeSeriesOD (status: planned)

- [ ] **Step 2: Verify the knowledge loader tests pass**

Run: `python -m pytest pyod/test/test_knowledge.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add pyod/utils/knowledge/algorithms.json
git commit -m "feat: add algorithm registry with 46+ detector entries"
```

---

## Task 3: Benchmarks, Routing Rules, and Papers JSON

**Files:**
- Create: `pyod/utils/knowledge/benchmarks.json`
- Create: `pyod/utils/knowledge/routing_rules.json`
- Create: `pyod/utils/knowledge/papers.json`

- [ ] **Step 1: Write `benchmarks.json`**

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

- [ ] **Step 2: Write `routing_rules.json`**

Use the structured predicate format from the spec. Include rules for: tabular (high-dim fast, high-dim accurate, low-dim small, low-dim large, general balanced), text, image, multimodal. Time series rules included but with `status_required: "shipped"` (will be filtered out until TimeSeriesOD ships).

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
        {"detector": "ECOD", "params": {}, "confidence": 0.9},
        {"detector": "HBOS", "params": {}, "confidence": 0.85},
        {"detector": "IForest", "params": {}, "confidence": 0.8}
      ],
      "reason": "High-dimensional tabular + speed priority: parameter-free fast methods",
      "evidence": ["ADBench"]
    },
    {
      "id": "tabular_high_dim_accurate",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "tabular"},
        {"field": "n_features", "op": "gte", "value": 100},
        {"field": "priority", "op": "eq", "value": "accuracy"}
      ],
      "recommendations": [
        {"detector": "IForest", "params": {}, "confidence": 0.9},
        {"detector": "ECOD", "params": {}, "confidence": 0.85},
        {"detector": "COPOD", "params": {}, "confidence": 0.8}
      ],
      "reason": "High-dimensional tabular + accuracy: ensemble-friendly methods",
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
      "id": "tabular_low_dim_large",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "tabular"},
        {"field": "n_features", "op": "lt", "value": 20},
        {"field": "n_samples", "op": "gte", "value": 5000}
      ],
      "recommendations": [
        {"detector": "IForest", "params": {}, "confidence": 0.85},
        {"detector": "ECOD", "params": {}, "confidence": 0.8},
        {"detector": "INNE", "params": {}, "confidence": 0.75}
      ],
      "reason": "Low-dim large dataset: tree-based methods scale well",
      "evidence": ["ADBench"]
    },
    {
      "id": "tabular_balanced",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "tabular"}
      ],
      "recommendations": [
        {"detector": "IForest", "params": {}, "confidence": 0.85},
        {"detector": "ECOD", "params": {}, "confidence": 0.8},
        {"detector": "KNN", "params": {}, "confidence": 0.75}
      ],
      "reason": "General tabular: robust all-rounders from ADBench top-5",
      "evidence": ["ADBench"]
    },
    {
      "id": "text_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "text"}
      ],
      "recommendations": [
        {"detector": "EmbeddingOD", "params": {}, "preset": "for_text", "confidence": 0.9}
      ],
      "reason": "Text data: EmbeddingOD.for_text() with benchmark-informed defaults",
      "evidence": ["NLP_ADBench"]
    },
    {
      "id": "image_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "image"}
      ],
      "recommendations": [
        {"detector": "EmbeddingOD", "params": {}, "preset": "for_image", "confidence": 0.85}
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
        {"detector": "TimeSeriesOD", "params": {"detector": "IForest"}, "confidence": 0.85}
      ],
      "reason": "Time series: windowed IForest (TSB-AD top performer)",
      "evidence": ["TSB_AD"],
      "note": "TimeSeriesOD is status=planned; filtered until shipped"
    },
    {
      "id": "multimodal_default",
      "conditions": [
        {"field": "data_type", "op": "eq", "value": "multimodal"}
      ],
      "recommendations": [
        {"detector": "MultiModalOD", "params": {}, "confidence": 0.8}
      ],
      "reason": "Multi-modal data: score fusion across per-modality detectors",
      "evidence": []
    }
  ]
}
```

- [ ] **Step 3: Write `papers.json`**

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
    "venue": "NeurIPS 2022",
    "url": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268f24b8e26c8f26a7-Abstract-Datasets_and_Benchmarks.html"
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
  },
  "copod": {
    "title": "COPOD: Copula-Based Outlier Detection",
    "authors": "Li, Zhao, Botta, Ionescu, Hu",
    "venue": "ICDM 2020"
  },
  "suod": {
    "title": "SUOD: Accelerating Large-Scale Unsupervised Heterogeneous Outlier Detection",
    "authors": "Zhao, Hu, Botta, Ionescu, Chen",
    "venue": "MLSys 2021"
  }
}
```

- [ ] **Step 4: Run knowledge loader tests**

Run: `python -m pytest pyod/test/test_knowledge.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyod/utils/knowledge/benchmarks.json pyod/utils/knowledge/routing_rules.json pyod/utils/knowledge/papers.json
git commit -m "feat: add benchmarks, routing rules, and papers to knowledge base"
```

---

## Task 4: ADEngine Core — `profile_data()` and `plan_detection()`

**Files:**
- Create: `pyod/utils/ad_engine.py`
- Test: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests for profile_data**

```python
# pyod/test/test_ad_engine.py
# -*- coding: utf-8 -*-

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine


class TestProfileData(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_tabular_array(self):
        X = np.random.randn(1000, 20)
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'tabular'
        assert profile['n_samples'] == 1000
        assert profile['n_features'] == 20

    def test_tabular_1d_is_tabular_not_ts(self):
        """1D numeric arrays default to tabular, not time_series."""
        X = np.random.randn(500)
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'tabular'

    def test_explicit_time_series_override(self):
        X = np.random.randn(500)
        profile = self.engine.profile_data(X, data_type='time_series')
        assert profile['data_type'] == 'time_series'

    def test_text_list(self):
        X = ["hello world", "anomaly detection", "test sentence"]
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'text'
        assert profile['n_samples'] == 3

    def test_dict_is_multimodal(self):
        X = {'text': ["hello"], 'tabular': np.array([[1, 2]])}
        profile = self.engine.profile_data(X)
        assert profile['data_type'] == 'multimodal'

    def test_has_nan_detection(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        profile = self.engine.profile_data(X)
        assert profile['has_nan'] is True

    def test_no_nan(self):
        X = np.random.randn(100, 5)
        profile = self.engine.profile_data(X)
        assert profile['has_nan'] is False

    def test_dimensionality_class_high(self):
        X = np.random.randn(100, 200)
        profile = self.engine.profile_data(X)
        assert profile['dimensionality_class'] == 'high'

    def test_dimensionality_class_low(self):
        X = np.random.randn(100, 5)
        profile = self.engine.profile_data(X)
        assert profile['dimensionality_class'] == 'low'


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestProfileData -v`
Expected: FAIL with `ImportError: cannot import name 'ADEngine'`

- [ ] **Step 3: Implement `profile_data()`**

```python
# pyod/utils/ad_engine.py
# -*- coding: utf-8 -*-
"""ADEngine: Intelligent anomaly detection lifecycle engine.

Handles data profiling, detection planning, detector construction,
and knowledge queries. Works as a standalone Python API (no LLM
required) or as the backend for MCP/agent interfaces.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import importlib
import os

import numpy as np

from .knowledge import KnowledgeBase


class ADEngine:
    """Anomaly detection lifecycle engine.

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge base directory. If None, uses bundled.
    """

    def __init__(self, knowledge_dir=None):
        self.kb = KnowledgeBase(knowledge_dir=knowledge_dir)

    def profile_data(self, X, data_type=None):
        """Profile the input data.

        Parameters
        ----------
        X : array-like, list, or dict
            Input data.
        data_type : str or None
            Explicit override. One of 'tabular', 'text', 'image',
            'time_series', 'multimodal', 'graph'.

        Returns
        -------
        profile : dict
        """
        if data_type is not None:
            detected_type = data_type
        else:
            detected_type = self._sniff_data_type(X)

        profile = {'data_type': detected_type}

        if detected_type == 'text':
            profile['n_samples'] = len(X)
        elif detected_type == 'image':
            profile['n_samples'] = len(X)
        elif detected_type == 'multimodal':
            first_key = next(iter(X))
            first_val = X[first_key]
            profile['n_samples'] = len(first_val)
            profile['modalities'] = list(X.keys())
        else:
            # tabular, time_series, or graph
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            profile['n_samples'] = arr.shape[0]
            profile['n_features'] = arr.shape[1]
            profile['has_nan'] = bool(np.isnan(arr).any())
            profile['dtype'] = str(arr.dtype)

            # Dimensionality class
            n_feat = arr.shape[1]
            if n_feat <= 10:
                profile['dimensionality_class'] = 'low'
            elif n_feat <= 100:
                profile['dimensionality_class'] = 'medium'
            else:
                profile['dimensionality_class'] = 'high'

            if detected_type == 'time_series':
                profile['n_timestamps'] = arr.shape[0]
                profile['channels'] = arr.shape[1]

        return profile

    def _sniff_data_type(self, X):
        """Conservative data type detection."""
        if isinstance(X, dict):
            return 'multimodal'
        if isinstance(X, (list, tuple)) and len(X) > 0:
            sample = X[:min(20, len(X))]
            if all(isinstance(x, str) for x in sample):
                if self._looks_like_image_paths(sample[:5]):
                    return 'image'
                return 'text'
        # All numeric arrays → tabular (conservative default)
        return 'tabular'

    @staticmethod
    def _looks_like_image_paths(samples):
        """Check if string samples look like image file paths."""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif',
                      '.tiff', '.webp'}
        for s in samples:
            ext = os.path.splitext(s)[1].lower()
            if ext not in image_exts:
                return False
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestProfileData -v`
Expected: All PASS

- [ ] **Step 5: Write failing tests for plan_detection**

Add to `pyod/test/test_ad_engine.py`:

```python
class TestPlanDetection(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_tabular_high_dim_speed(self):
        profile = {'data_type': 'tabular', 'n_samples': 10000,
                   'n_features': 200, 'dimensionality_class': 'high'}
        plan = self.engine.plan_detection(profile, priority='speed')
        assert plan['detector_name'] == 'ECOD'
        assert plan['confidence'] > 0
        assert 'reason' in plan
        assert 'evidence' in plan

    def test_tabular_low_dim_small(self):
        profile = {'data_type': 'tabular', 'n_samples': 1000,
                   'n_features': 5, 'dimensionality_class': 'low'}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] in ('KNN', 'LOF', 'CBLOF',
                                          'IForest', 'ECOD')

    def test_text_routes_to_embedding(self):
        profile = {'data_type': 'text', 'n_samples': 100}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] == 'EmbeddingOD'
        assert plan.get('preset') == 'for_text'

    def test_image_routes_to_embedding(self):
        profile = {'data_type': 'image', 'n_samples': 50}
        plan = self.engine.plan_detection(profile)
        assert plan['detector_name'] == 'EmbeddingOD'
        assert plan.get('preset') == 'for_image'

    def test_plan_has_alternatives(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(profile)
        assert 'alternatives' in plan
        assert isinstance(plan['alternatives'], list)

    def test_planned_detector_filtered_out(self):
        """TimeSeriesOD is planned, should not be recommended."""
        profile = {'data_type': 'time_series', 'n_samples': 1000,
                   'n_features': 1}
        plan = self.engine.plan_detection(profile)
        # Should fall back or return a note, not recommend TimeSeriesOD
        assert plan['detector_name'] != 'TimeSeriesOD'

    def test_constraints_exclude_detector(self):
        profile = {'data_type': 'tabular', 'n_samples': 5000,
                   'n_features': 50, 'dimensionality_class': 'medium'}
        plan = self.engine.plan_detection(
            profile, constraints={'exclude_detectors': ['IForest', 'ECOD']})
        assert plan['detector_name'] not in ('IForest', 'ECOD')

    def test_plan_is_closed_schema(self):
        profile = {'data_type': 'tabular', 'n_samples': 1000,
                   'n_features': 10, 'dimensionality_class': 'low'}
        plan = self.engine.plan_detection(profile)
        allowed_keys = {'detector_name', 'preset', 'params',
                        'preprocessing', 'threshold_strategy',
                        'threshold_value', 'reason', 'evidence',
                        'confidence', 'alternatives', 'note'}
        for key in plan:
            assert key in allowed_keys, \
                f"Unexpected key '{key}' in plan"
```

- [ ] **Step 6: Implement `plan_detection()`**

Add to `ADEngine` class in `pyod/utils/ad_engine.py`:

```python
    def plan_detection(self, profile, priority='balanced',
                       constraints=None):
        """Plan a detection pipeline.

        Parameters
        ----------
        profile : dict
            Output of profile_data().
        priority : str
            'speed', 'accuracy', or 'balanced'.
        constraints : dict or None
            Optional: {'exclude_detectors': [...],
            'require_interpretable': True, ...}

        Returns
        -------
        plan : dict (DetectionPlan, closed schema)
        """
        constraints = constraints or {}
        exclude = set(constraints.get('exclude_detectors', []))

        # Evaluate routing rules
        matched = self._evaluate_rules(profile, priority)

        # Filter by status and constraints
        valid = []
        for rec in matched:
            name = rec['detector']
            algo = self.kb.get_algorithm(name)
            if algo is None:
                continue
            if algo.get('status') != 'shipped':
                continue
            if name in exclude:
                continue
            valid.append(rec)

        if not valid:
            # Fallback: IForest is always a safe choice for tabular
            return self._make_plan(
                detector_name='IForest', params={},
                reason='Fallback: no routing rule matched; '
                       'IForest is a robust general-purpose detector',
                evidence=['ADBench'], confidence=0.5,
                alternatives=[], note='No specific rule matched')

        best = valid[0]
        alternatives = [self._make_plan(
            detector_name=r['detector'],
            params=r.get('params', {}),
            preset=r.get('preset'),
            reason='', evidence=[], confidence=r.get('confidence', 0.5),
            alternatives=[]) for r in valid[1:3]]

        rule_reason = ''
        rule_evidence = []
        for rule in self.kb.routing_rules.get('rules', []):
            recs = rule.get('recommendations', [])
            if recs and recs[0].get('detector') == best['detector']:
                rule_reason = rule.get('reason', '')
                rule_evidence = rule.get('evidence', [])
                break

        return self._make_plan(
            detector_name=best['detector'],
            params=best.get('params', {}),
            preset=best.get('preset'),
            reason=rule_reason,
            evidence=rule_evidence,
            confidence=best.get('confidence', 0.7),
            alternatives=alternatives)

    def _evaluate_rules(self, profile, priority):
        """Evaluate routing rules against a profile. Returns matched
        recommendations sorted by confidence."""
        rules = self.kb.routing_rules.get('rules', [])
        all_recs = []

        for rule in rules:
            if self._rule_matches(rule, profile, priority):
                for rec in rule.get('recommendations', []):
                    all_recs.append(rec)

        # Deduplicate by detector name, keep highest confidence
        seen = {}
        for rec in all_recs:
            name = rec['detector']
            if name not in seen or \
                    rec.get('confidence', 0) > seen[name].get('confidence', 0):
                seen[name] = rec
        return sorted(seen.values(),
                      key=lambda r: r.get('confidence', 0),
                      reverse=True)

    def _rule_matches(self, rule, profile, priority):
        """Check if all conditions in a rule match the profile."""
        for cond in rule.get('conditions', []):
            field = cond['field']
            op = cond['op']
            value = cond['value']

            if field == 'priority':
                actual = priority
            else:
                actual = profile.get(field)

            if actual is None:
                return False
            if not self._eval_condition(actual, op, value):
                return False
        return True

    @staticmethod
    def _eval_condition(actual, op, value):
        """Evaluate a single condition predicate."""
        if op == 'eq':
            return actual == value
        if op == 'lt':
            return float(actual) < float(value)
        if op == 'lte':
            return float(actual) <= float(value)
        if op == 'gt':
            return float(actual) > float(value)
        if op == 'gte':
            return float(actual) >= float(value)
        if op == 'in':
            return actual in value
        return False

    @staticmethod
    def _make_plan(detector_name, params=None, preset=None,
                   reason='', evidence=None, confidence=0.5,
                   alternatives=None, note=None):
        """Construct a closed-schema DetectionPlan dict."""
        plan = {
            'detector_name': detector_name,
            'params': params or {},
            'reason': reason,
            'evidence': evidence or [],
            'confidence': confidence,
            'alternatives': alternatives or [],
        }
        if preset:
            plan['preset'] = preset
        if note:
            plan['note'] = note
        return plan
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest pyod/test/test_ad_engine.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add ADEngine with profile_data() and plan_detection()"
```

---

## Task 5: ADEngine — `build_detector()`, `detect()`, Knowledge Queries

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests**

Add to `pyod/test/test_ad_engine.py`:

```python
from pyod.models.base import BaseDetector


class TestBuildDetector(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_build_returns_base_detector(self):
        plan = {'detector_name': 'IForest', 'params': {}}
        clf = self.engine.build_detector(plan)
        assert isinstance(clf, BaseDetector)

    def test_build_with_params(self):
        plan = {'detector_name': 'KNN', 'params': {'n_neighbors': 10}}
        clf = self.engine.build_detector(plan)
        assert clf.n_neighbors == 10

    def test_build_with_preset(self):
        plan = {'detector_name': 'EmbeddingOD', 'params': {},
                'preset': 'for_text'}
        clf = self.engine.build_detector(plan)
        assert isinstance(clf, BaseDetector)

    def test_build_unknown_detector_raises(self):
        plan = {'detector_name': 'NonExistentDetector', 'params': {}}
        with self.assertRaises(ValueError):
            self.engine.build_detector(plan)

    def test_build_planned_detector_raises(self):
        plan = {'detector_name': 'TimeSeriesOD', 'params': {}}
        with self.assertRaises(ValueError):
            self.engine.build_detector(plan)


class TestDetectShortcut(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(200, 10)

    def test_detect_returns_result(self):
        result = self.engine.detect(self.X_train)
        assert 'plan' in result
        assert 'scores' in result
        assert 'labels' in result
        assert 'n_anomalies' in result
        assert len(result['scores']) == 200

    def test_detect_with_explicit_type(self):
        result = self.engine.detect(self.X_train, data_type='tabular')
        assert result['plan']['detector_name'] in (
            'IForest', 'ECOD', 'KNN', 'LOF', 'CBLOF', 'HBOS',
            'COPOD', 'INNE')


class TestKnowledgeQueries(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()

    def test_list_detectors(self):
        detectors = self.engine.list_detectors()
        assert len(detectors) >= 46
        names = [d['name'] for d in detectors]
        assert 'ECOD' in names
        assert 'IForest' in names

    def test_list_detectors_by_type(self):
        text_dets = self.engine.list_detectors(data_type='text')
        names = [d['name'] for d in text_dets]
        assert 'EmbeddingOD' in names

    def test_explain_detector(self):
        info = self.engine.explain_detector('ECOD')
        assert info['full_name'] is not None
        assert 'strengths' in info
        assert 'weaknesses' in info

    def test_explain_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.engine.explain_detector('FakeDetector')

    def test_compare_detectors(self):
        comparison = self.engine.compare_detectors(
            names=['ECOD', 'IForest', 'KNN'])
        assert len(comparison) == 3

    def test_get_benchmarks(self):
        benchmarks = self.engine.get_benchmarks()
        assert 'ADBench' in benchmarks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestBuildDetector -v`
Expected: FAIL with `AttributeError: 'ADEngine' object has no attribute 'build_detector'`

- [ ] **Step 3: Implement `build_detector()`**

Add to `ADEngine` class:

```python
    def build_detector(self, plan):
        """Build and return an unfitted detector from a plan.

        Parameters
        ----------
        plan : dict (DetectionPlan)
            Output of plan_detection().

        Returns
        -------
        detector : BaseDetector
        """
        name = plan['detector_name']
        algo = self.kb.get_algorithm(name)
        if algo is None:
            raise ValueError("Unknown detector '%s'" % name)
        if algo.get('status') not in ('shipped', 'experimental'):
            raise ValueError(
                "Detector '%s' has status '%s' and cannot be built. "
                % (name, algo.get('status', 'unknown')))

        # Handle presets (e.g., EmbeddingOD.for_text())
        preset = plan.get('preset')
        if preset:
            return self._build_from_preset(name, preset, plan.get('params', {}))

        # Standard instantiation
        class_path = algo['class_path']
        module_path, class_name = class_path.rsplit('.', 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        params = plan.get('params', {})
        return cls(**params)

    @staticmethod
    def _build_from_preset(detector_name, preset, extra_params):
        """Build a detector using a factory preset."""
        if detector_name == 'EmbeddingOD':
            from pyod.models.embedding import EmbeddingOD
            if preset == 'for_text':
                return EmbeddingOD.for_text(**extra_params)
            elif preset == 'for_image':
                return EmbeddingOD.for_image(**extra_params)
        raise ValueError("Unknown preset '%s' for '%s'"
                         % (preset, detector_name))
```

- [ ] **Step 4: Implement `detect()` shortcut**

```python
    def detect(self, X_train, X_test=None, data_type=None,
               priority='balanced'):
        """One-shot anomaly detection: profile → plan → build → fit.

        Returns
        -------
        result : dict
            Keys: 'plan', 'scores', 'labels', 'threshold',
            'n_anomalies', 'detector'.
        """
        profile = self.profile_data(X_train, data_type=data_type)
        plan = self.plan_detection(profile, priority=priority)
        clf = self.build_detector(plan)
        clf.fit(X_train)

        result = {
            'plan': plan,
            'scores': clf.decision_scores_,
            'labels': clf.labels_,
            'threshold': clf.threshold_,
            'n_anomalies': int(clf.labels_.sum()),
            'detector': clf,
        }

        if X_test is not None:
            test_scores = clf.decision_function(X_test)
            test_labels = clf.predict(X_test)
            result['scores_test'] = test_scores
            result['labels_test'] = test_labels

        return result
```

- [ ] **Step 5: Implement knowledge query methods**

```python
    def list_detectors(self, data_type=None, status='shipped'):
        """List available detectors."""
        if data_type:
            return self.kb.list_by_data_type(data_type, status=status)
        if status == 'all':
            return [{'name': k, **v}
                    for k, v in self.kb.algorithms.items()]
        return self.kb.list_by_status(status)

    def explain_detector(self, name):
        """Explain a detector."""
        algo = self.kb.get_algorithm(name)
        if algo is None:
            raise ValueError("Unknown detector '%s'" % name)
        return {'name': name, **algo}

    def compare_detectors(self, names=None, data_type=None, top_k=3):
        """Compare detectors."""
        if names:
            return [self.explain_detector(n) for n in names]
        detectors = self.list_detectors(data_type=data_type)
        return detectors[:top_k]

    def get_benchmarks(self, benchmark='all'):
        """Get benchmark results."""
        if benchmark == 'all':
            return self.kb.benchmarks
        return {benchmark: self.kb.benchmarks.get(benchmark)}
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest pyod/test/test_ad_engine.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add build_detector, detect shortcut, and knowledge queries to ADEngine"
```

---

## Task 6: AutoModelSelector Deprecation Shim

**Files:**
- Modify: `pyod/utils/auto_model_selector.py`

- [ ] **Step 1: Add deprecation warning at module level**

At the top of `auto_model_selector.py`, after imports, add:

```python
import warnings

warnings.warn(
    "AutoModelSelector is deprecated and will be removed in PyOD v2.3.0. "
    "Use pyod.utils.ad_engine.ADEngine instead. "
    "See: https://pyod.readthedocs.io/en/latest/ad_engine.html",
    FutureWarning,
    stacklevel=2
)
```

- [ ] **Step 2: Verify warning is emitted**

Run: `python -c "from pyod.utils.auto_model_selector import AutoModelSelector" 2>&1`
Expected: FutureWarning about deprecation

- [ ] **Step 3: Commit**

```bash
git add pyod/utils/auto_model_selector.py
git commit -m "chore: add deprecation warning to AutoModelSelector (replaced by ADEngine)"
```

---

## Task 7: MCP Server (Tier A Tools)

**Files:**
- Create: `pyod/mcp_server.py`

- [ ] **Step 1: Write the MCP server with Tier A tools**

```python
# pyod/mcp_server.py
# -*- coding: utf-8 -*-
"""PyOD MCP Server: Agent interface for anomaly detection.

Exposes PyOD's ADEngine as MCP tools that any LLM agent can call.
Tier A: knowledge queries + stateless planning.

Usage:
    python -m pyod.mcp_server
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import annotations

import json
import os
import sys


def _check_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError:
        print("MCP server requires the 'mcp' package. "
              "Install with: pip install pyod[mcp]",
              file=sys.stderr)
        sys.exit(1)


FastMCP = _check_mcp()

from pyod.utils.ad_engine import ADEngine

mcp = FastMCP("pyod")
engine = ADEngine()


def _to_json(obj):
    """Serialize result to JSON string."""
    return json.dumps(obj, indent=2, default=str)


@mcp.tool()
def profile_data(data_path: str, data_type: str = "auto") -> str:
    """Profile a dataset for anomaly detection.

    Loads data from path, detects data type and characteristics.
    Returns a JSON profile for use with plan_detection().

    Args:
        data_path: Path to data file (CSV, NPY, JSON).
        data_type: Override. One of 'tabular', 'text', 'image', or 'auto'.
    """
    X = _load_data(data_path)
    dt = None if data_type == "auto" else data_type
    return _to_json(engine.profile_data(X, data_type=dt))


@mcp.tool()
def plan_detection(
    data_profile: str,
    priority: str = "balanced",
    constraints: str = ""
) -> str:
    """Plan an anomaly detection pipeline.

    Returns a DetectionPlan with detector, params, reason, and evidence.

    Args:
        data_profile: JSON string from profile_data().
        priority: 'speed', 'accuracy', or 'balanced'.
        constraints: Optional JSON, e.g. '{"exclude_detectors": ["ECOD"]}'.
    """
    profile = json.loads(data_profile)
    cons = json.loads(constraints) if constraints else None
    return _to_json(engine.plan_detection(profile, priority, cons))


@mcp.tool()
def build_detector(plan: str) -> str:
    """Get constructor metadata for a detector from a plan.

    Returns import path, class name, validated params, and a
    ready-to-run Python code snippet for instantiation.

    Args:
        plan: JSON string from plan_detection().
    """
    plan_dict = json.loads(plan)
    name = plan_dict['detector_name']
    algo = engine.kb.get_algorithm(name)
    if algo is None:
        return _to_json({"error": "Unknown detector", "name": name})

    preset = plan_dict.get('preset')
    params = plan_dict.get('params', {})

    if preset:
        code = "from pyod.models.embedding import EmbeddingOD\n"
        code += "clf = EmbeddingOD.%s(%s)" % (
            preset,
            ', '.join('%s=%r' % (k, v) for k, v in params.items()))
    else:
        class_path = algo['class_path']
        module_path, class_name = class_path.rsplit('.', 1)
        code = "from %s import %s\n" % (module_path, class_name)
        if params:
            code += "clf = %s(%s)" % (
                class_name,
                ', '.join('%s=%r' % (k, v) for k, v in params.items()))
        else:
            code += "clf = %s()" % class_name

    return _to_json({
        "detector_name": name,
        "class_path": algo['class_path'],
        "params": params,
        "preset": preset,
        "code_snippet": code,
    })


@mcp.tool()
def list_detectors(data_type: str = "", status: str = "shipped") -> str:
    """List available PyOD detectors.

    Args:
        data_type: Filter by data type (tabular, text, image, etc.).
        status: Filter by status (shipped, planned, all).
    """
    return _to_json(engine.list_detectors(
        data_type=data_type or None, status=status))


@mcp.tool()
def explain_detector(name: str) -> str:
    """Explain a PyOD detector: how it works, strengths, weaknesses,
    benchmark performance, and recommended use cases."""
    try:
        return _to_json(engine.explain_detector(name))
    except ValueError as e:
        return _to_json({"error": str(e)})


@mcp.tool()
def compare_detectors(
    names: str = "",
    data_type: str = "tabular",
    top_k: int = 3
) -> str:
    """Compare detectors for a given data type.

    Args:
        names: Comma-separated detector names. If empty, top-k for type.
        data_type: Data type to compare for.
        top_k: Number of top detectors.
    """
    name_list = [n.strip() for n in names.split(',')] if names else None
    return _to_json(engine.compare_detectors(name_list, data_type, top_k))


@mcp.tool()
def get_benchmarks(benchmark: str = "all") -> str:
    """Get benchmark results (ADBench, NLP-ADBench, TSB-AD)."""
    return _to_json(engine.get_benchmarks(benchmark))


def _load_data(path):
    """Load data from file path."""
    import numpy as np

    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, allow_pickle=True)
    elif ext == '.npz':
        data = np.load(path, allow_pickle=True)
        return data[data.files[0]]
    elif ext == '.csv':
        import csv
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]
        return np.array(rows, dtype=np.float64)
    elif ext == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif ext == '.mat':
        from scipy.io import loadmat
        data = loadmat(path)
        if 'X' in data:
            return data['X']
        # Return first non-metadata key
        for key in data:
            if not key.startswith('_'):
                return data[key]
    else:
        raise ValueError("Unsupported file format: %s" % ext)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 2: Verify the server file is importable (no syntax errors)**

Run: `python -c "import ast; ast.parse(open('pyod/mcp_server.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add pyod/mcp_server.py
git commit -m "feat: add MCP server with Tier A tools (knowledge + planning)"
```

---

## Task 8: Claude Code Skill, Packaging, and Example

**Files:**
- Create: `skills/od-expert/SKILL.md`
- Modify: `setup.py`
- Create: `examples/ad_engine_example.py`

- [ ] **Step 1: Write the Claude Code skill**

```markdown
# skills/od-expert/SKILL.md
---
name: od-expert
description: Anomaly detection expert. Drives PyOD's ADEngine for data profiling, detection planning, algorithm explanation, and benchmark comparison. Works for tabular, text, and image data.
---

You are an anomaly detection expert backed by PyOD's ADEngine.

## When to activate
- User has data and wants anomaly detection
- User asks "which detector should I use?"
- User asks about PyOD algorithms or benchmarks
- User asks to compare detection methods

## How to work
Do NOT embed detection knowledge in your responses. Instead:

1. If PyOD MCP tools are available, use them:
   - `profile_data` to understand the data
   - `plan_detection` to get a recommendation
   - `build_detector` to get instantiation code
   - `list_detectors`, `explain_detector`, `compare_detectors`, `get_benchmarks` for knowledge queries
2. If MCP is not available, import and call ADEngine directly:
   ```python
   from pyod.utils.ad_engine import ADEngine
   engine = ADEngine()
   ```
3. For knowledge queries, read from `pyod/utils/knowledge/*.json`.

## Lifecycle flow
profile_data -> plan_detection -> build_detector (get code) -> user runs detection
```

- [ ] **Step 2: Update setup.py packaging**

Add `knowledge/*.json` to `package_data` and `mcp` to `extras_require`:

In `setup.py`, change:

```python
    package_data={
        'pyod.utils': ['model_analysis_jsons/*.json'],
    },
```

to:

```python
    package_data={
        'pyod.utils': ['model_analysis_jsons/*.json'],
        'pyod.utils.knowledge': ['*.json'],
    },
```

And in `extras_require`, add `'mcp': ['mcp>=1.0'],` and add `'mcp>=1.0'` to the `'all'` list.

- [ ] **Step 3: Write the usage example**

```python
# examples/ad_engine_example.py
"""ADEngine: Intelligent anomaly detection in 3 lines.

Demonstrates PyOD's ADEngine for automatic detector selection
and anomaly detection across data types.
"""
from pyod.utils.ad_engine import ADEngine
from pyod.utils.data import generate_data

# Generate sample data
X_train, X_test, y_train, y_test = generate_data(
    n_train=300, n_test=100, n_features=20, contamination=0.1)

# Initialize the engine
engine = ADEngine()

# === One-shot detection ===
result = engine.detect(X_train)
print("Detector chosen:", result['plan']['detector_name'])
print("Reason:", result['plan']['reason'])
print("Anomalies found:", result['n_anomalies'])
print()

# === Step-by-step lifecycle ===
# 1. Profile the data
profile = engine.profile_data(X_train)
print("Data profile:", profile)

# 2. Plan detection
plan = engine.plan_detection(profile, priority='speed')
print("Plan:", plan['detector_name'], "-", plan['reason'])

# 3. Build detector
clf = engine.build_detector(plan)
print("Detector:", clf)

# 4. Fit and predict
clf.fit(X_train)
print("Training anomalies:", clf.labels_.sum())
print("Test scores:", clf.decision_function(X_test)[:5])
print()

# === Knowledge queries ===
print("=== Available text detectors ===")
for d in engine.list_detectors(data_type='text'):
    print(f"  {d['name']}: {d['full_name']}")

print()
print("=== ECOD explained ===")
info = engine.explain_detector('ECOD')
print(f"  {info['full_name']}")
print(f"  Best for: {info['best_for']}")
print(f"  Strengths: {', '.join(info['strengths'][:3])}")

print()
print("=== ADBench results ===")
bench = engine.get_benchmarks('ADBench')
print(f"  Top 5: {bench['ADBench']['rankings']['overall_top_5']}")
```

- [ ] **Step 4: Run the example**

Run: `python examples/ad_engine_example.py`
Expected: Prints detector selection, profile, plan, results, and knowledge queries without error.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest pyod/test/test_knowledge.py pyod/test/test_ad_engine.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add skills/od-expert/SKILL.md setup.py examples/ad_engine_example.py
git commit -m "feat: add Claude Code skill, packaging updates, and ADEngine example"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Layer 1: Knowledge base (algorithms.json, benchmarks.json, routing_rules.json, papers.json) → Tasks 1-3
- [x] Layer 2: ADEngine (profile_data, plan_detection, build_detector, detect, knowledge queries) → Tasks 4-5
- [x] Layer 3a: MCP server (Tier A tools) → Task 7
- [x] Layer 3b: Claude Code skill → Task 8
- [x] AutoModelSelector deprecation → Task 6
- [x] Packaging (package_data, mcp extra) → Task 8
- [x] Example → Task 8
- [x] Tests → Tasks 1, 4, 5
- [ ] OpenAI schema export (`to_openai_tools()`) → deferred to a follow-up task (low priority, minimal code)

**Placeholder scan:** No TBD, TODO, or "implement later" found.

**Type consistency:** `ADEngine`, `KnowledgeBase`, `DetectionPlan` dict schema, method signatures all consistent across tasks.

**Not in scope (Tier B):** `run_detection()` execution, `RunSession`, `analyze_results()`, `explain_findings()`, `suggest_next_step()`, `generate_report()`, stateful MCP tools. These get their own plan when Tier A ships.
