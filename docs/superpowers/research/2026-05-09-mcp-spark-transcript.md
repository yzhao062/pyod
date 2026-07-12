# pyod MCP server stdio transcript

**Captured:** 2026-05-09T19:55:05.343749+00:00
**Server cmd:** `python -m pyod.mcp_server`
**Client:** mcp Python SDK stdio client
**Host:** Spark (DGX OS, Linux aarch64), py312 conda env


## Server-advertised tools (7)

- `build_detector`: Get constructor metadata for a detector from a plan.
- `compare_detectors`: Compare detectors for a given data type.
- `explain_detector`: Explain a PyOD detector: how it works, strengths, weaknesses,
- `get_benchmarks`: Get benchmark results (ADBench, NLP-ADBench, TSB-AD).
- `list_detectors`: List available PyOD detectors.
- `plan_detection`: Plan an anomaly detection pipeline.
- `profile_data`: Profile a dataset for anomaly detection.


## 1. list_detectors(data_type=tabular, status=shipped)

**Args:** `{"data_type": "tabular", "status": "shipped"}`

**Result:**
```
[
  {
    "name": "ECOD",
    "class_path": "pyod.models.ecod.ECOD",
    "full_name": "Empirical Cumulative Distribution Functions",
    "status": "shipped",
    "data_types": [
      "tabular"
    ],
    "category": "probabilistic",
    "complexity": {
      "time": "O(n * d * log(n))",
      "space": "O(n * d)"
    },
    "strengths": [
      "Parameter-free and highly interpretable",
      "Fast computation with parallelization support",
      "Strong benchmark performance across diverse datasets",
      "No assumption on data distribution"
    ],
    "weaknesses": [
      "May struggle with strongly correlated features",
      "Assumes outliers deviate in at least one marginal dimension"
    ],
    "best_for": "General-purpose outlier detection when speed and interpretability are priorities",
    "avoid_when": "Features are heavily correlated and outliers only manifest in joint distributions",
    "benchmark_refs": [
      "ADBench"
    ],
    "benchmark_rank": {
      "ADBench_overall": 5
    },
    "paper": {
      "id": "ecod",
      "short": "Li et al., TKDE 2022"
    },
    "default_params": {
      "contamination": 0.1,
      "n_jobs": 1
    },
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.9.0"
  },
  {
    "name": "ABOD",
    "class_path": "pyod.models.abod.ABOD",
    "full_name": "Angle-Based Outlier Detection",
    "status": "shipped",
    "data_types": [
      "tabular"
    ],
    "category": "probabilistic",
    "complexity": {
      "time": "O(n^2 * d) for fast, O(n^3 * d) for default",
      "space": "O(n^2)"
    },
    "strengths": [
      "Effective in high-dimensional spaces",
      "Not affected by distance concentration in high dimensions",
      "Good theoretical foundation"
    ],
    "weaknesses": [
      "Computationally expensive, especially in default mode",
      "Sensitive to noise in low-dimensional data",
      "Requires sufficient neighbors for angle variance estimation"
    ],
    "best_for": "High-dimensional datasets where distance-based methods suffer from curse of dimensionality",
    "avoid_when": "Dataset is very large or low-dimensional where distance-based methods work well",
    "benchmark_refs": [
      "ADBench"
    ],
    "benchmark_rank": {},
    "paper": {
      "id": "abod",
      "short": "Kriegel et al., KDD 2008"
    },
    "default_params": {
      "contamination": 0.1,
      "n_neighbors": 5,
      "method": "fast"
    },
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.5.0"
  },
  {
    "name": "COPOD",
    "class_path": "pyod.models.copod.COPOD",
    "full_name": "Copula-Based Outlier Detection",
    "status": "shipped",
    "data_types": [
      "tabular"
    ],
    "category": "probabilistic",
    "complexity": {
      "time": "O(n * d * log(n))",
      "space": "O(n * d)"
    },
    "strengths": [
      "Parameter-free and highly interpretable",
      "Fast computation with parallelization support",
      "Models tail probabilities via empirical copulas"
    ],
    "weaknesses": [
      "Assumes feature-wise independence for tail modeling",
      "May miss complex multivariate interactions"
    ],
    "best_for": "Large-scale datasets where speed and interpretability matter and features are roughly independent",
    "avoid_when": "Outliers only appear in joint distributions with strong feature dependencies",
    "benchmark_refs": [
      "ADBench"
    ],
    "benchmark_rank": {
      "ADBench_overall": 6
    },
    "paper": {
      "id": "copod",
      "short": "Li et al., ICDM 2020"
    },
    "default_params": {
      "contamination": 0.1,
      "n_jobs": 1
    },
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.7.0"
  },
  {
    "name": "MAD",
    "class_path": "pyod.models.mad.MAD",
    "full_name": "Median Absolute Deviation",
    "status": "shipped",
    "data_type
... [truncated to 4000 chars]
```



## 2. explain_detector(name=ECOD)

**Args:** `{"name": "ECOD"}`

**Result:**
```
{
  "name": "ECOD",
  "class_path": "pyod.models.ecod.ECOD",
  "full_name": "Empirical Cumulative Distribution Functions",
  "status": "shipped",
  "data_types": [
    "tabular"
  ],
  "category": "probabilistic",
  "complexity": {
    "time": "O(n * d * log(n))",
    "space": "O(n * d)"
  },
  "strengths": [
    "Parameter-free and highly interpretable",
    "Fast computation with parallelization support",
    "Strong benchmark performance across diverse datasets",
    "No assumption on data distribution"
  ],
  "weaknesses": [
    "May struggle with strongly correlated features",
    "Assumes outliers deviate in at least one marginal dimension"
  ],
  "best_for": "General-purpose outlier detection when speed and interpretability are priorities",
  "avoid_when": "Features are heavily correlated and outliers only manifest in joint distributions",
  "benchmark_refs": [
    "ADBench"
  ],
  "benchmark_rank": {
    "ADBench_overall": 5
  },
  "paper": {
    "id": "ecod",
    "short": "Li et al., TKDE 2022"
  },
  "default_params": {
    "contamination": 0.1,
    "n_jobs": 1
  },
  "preprocessing_mode": "external",
  "requires": [],
  "version_added": "0.9.0"
}
```



## 3. compare_detectors(data_type=tabular, top_k=3)

**Args:** `{"data_type": "tabular", "top_k": 3}`

**Result:**
```
[
  {
    "name": "ECOD",
    "class_path": "pyod.models.ecod.ECOD",
    "full_name": "Empirical Cumulative Distribution Functions",
    "status": "shipped",
    "data_types": [
      "tabular"
    ],
    "category": "probabilistic",
    "complexity": {
      "time": "O(n * d * log(n))",
      "space": "O(n * d)"
    },
    "strengths": [
      "Parameter-free and highly interpretable",
      "Fast computation with parallelization support",
      "Strong benchmark performance across diverse datasets",
      "No assumption on data distribution"
    ],
    "weaknesses": [
      "May struggle with strongly correlated features",
      "Assumes outliers deviate in at least one marginal dimension"
    ],
    "best_for": "General-purpose outlier detection when speed and interpretability are priorities",
    "avoid_when": "Features are heavily correlated and outliers only manifest in joint distributions",
    "benchmark_refs": [
      "ADBench"
    ],
    "benchmark_rank": {
      "ADBench_overall": 5
    },
    "paper": {
      "id": "ecod",
      "short": "Li et al., TKDE 2022"
    },
    "default_params": {
      "contamination": 0.1,
      "n_jobs": 1
    },
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.9.0"
  },
  {
    "name": "ABOD",
    "class_path": "pyod.models.abod.ABOD",
    "full_name": "Angle-Based Outlier Detection",
    "status": "shipped",
    "data_types": [
      "tabular"
    ],
    "category": "probabilistic",
    "complexity": {
      "time": "O(n^2 * d) for fast, O(n^3 * d) for default",
      "space": "O(n^2)"
    },
    "strengths": [
      "Effective in high-dimensional spaces",
      "Not affected by distance concentration in high dimensions",
      "Good theoretical foundation"
    ],
    "weaknesses": [
      "Computationally expensive, especially in default mode",
      "Sensitive to noise in low-dimensional data",
      "Requires sufficient neighbors for angle variance estimation"
    ],
    "best_for": "High-dimensional datasets where distance-based methods suffer from curse of dimensionality",
    "avoid_when": "Dataset is very large or low-dimensional where distance-based methods work well",
    "benchmark_refs": [
      "ADBench"
    ],
    "benchmark_rank": {},
    "paper": {
      "id": "abod",
      "short": "Kriegel et al., KDD 2008"
    },
    "default_params": {
      "contamination": 0.1,
      "n_neighbors": 5,
      "method": "fast"
    },
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.5.0"
  },
  {
    "name": "COPOD",
    "class_path": "pyod.models.copod.COPOD",
    "full_name": "Copula-Based Outlier Detection",
    "status": "shipped",
    "data_types": [
      "tabular"
    ],
    "category": "probabilistic",
    "complexity": {
      "time": "O(n * d * log(n))",
      "space": "O(n * d)"
    },
    "strengths": [
      "Parameter-free and highly interpretable",
      "Fast computation with parallelization support",
      "Models tail probabilities via empirical copulas"
    ],
    "weaknesses": [
      "Assumes feature-wise independence for tail modeling",
      "May miss complex multivariate interactions"
    ],
    "best_for": "Large-scale datasets where speed and interpretability matter and features are roughly independent",
    "avoid_when": "Outliers only appear in joint distributions with strong feature dependencies",
    "benchmark_refs": [
      "ADBench"
    ],
    "benchmark_rank": {
      "ADBench_overall": 6
    },
    "paper": {
      "id": "copod",
      "short": "Li et al., ICDM 2020"
    },
    "default_params": {
      "contamination": 0.1,
      "n_jobs": 1
    },
    "preprocessing_mode": "external",
    "requires": [],
    "version_added": "0.7.0"
  }
]
```



## 4. get_benchmarks(benchmark=all)

**Args:** `{"benchmark": "all"}`

**Result:**
```
{
  "ADBench": {
    "paper": {
      "id": "adbench",
      "short": "Han et al., NeurIPS 2022"
    },
    "scope": "tabular",
    "n_datasets": 57,
    "n_algorithms": 30,
    "rankings": {
      "overall_top_5": [
        "ECOD",
        "IForest",
        "KNN",
        "COPOD",
        "HBOS"
      ],
      "high_dim_top_3": [
        "ECOD",
        "COPOD",
        "IForest"
      ],
      "low_dim_top_3": [
        "KNN",
        "LOF",
        "CBLOF"
      ]
    },
    "key_finding": "No single algorithm dominates; ensemble of top-5 is robust"
  },
  "NLP_ADBench": {
    "paper": {
      "id": "nlp_adbench",
      "short": "Li et al., EMNLP 2025"
    },
    "scope": "text",
    "n_datasets": 8,
    "n_algorithms": 19,
    "rankings": {
      "overall_top_5": [
        "OpenAI+LUNAR",
        "OpenAI+LOF",
        "OpenAI+AE",
        "MiniLM+KNN",
        "BERT+LOF"
      ]
    },
    "key_finding": "Embedding quality >> detector choice; two-step beats end-to-end"
  },
  "TSB_AD": {
    "paper": {
      "id": "tsb_ad",
      "short": "Liu & Paparrizos, NeurIPS 2024"
    },
    "scope": "time_series",
    "n_datasets": 1070,
    "n_algorithms": 40,
    "rankings": {
      "overall_top_5": [
        "IForest",
        "LOF",
        "POLY",
        "KNN",
        "KShapeAD"
      ],
      "subsequence_top_3": [
        "MatrixProfile",
        "SAND",
        "Series2Graph"
      ]
    },
    "key_finding": "Classical methods competitive with deep; MatrixProfile strong on subsequence anomalies"
  },
  "BOND": {
    "paper": {
      "id": "liu2022bond",
      "short": "Liu et al., NeurIPS 2022"
    },
    "scope": "graph",
    "n_datasets": 14,
    "n_algorithms": 14,
    "rankings": {
      "deep_top_3": [
        "DOMINANT",
        "CoLA",
        "CONAD"
      ],
      "classical_top_2": [
        "Radar",
        "ANOMALOUS"
      ]
    },
    "key_finding": "DOMINANT and CoLA are most reliable deep methods; classical MF methods competitive on small graphs"
  }
}
```



## 5. profile_data(data_path=<cardio.csv>, data_type=auto)

**Args:** `{"data_path": "/home/yzhao062/projects/pyod/examples/data/cardio.csv", "data_type": "auto"}`

**Result:**
```
{
  "data_type": "tabular",
  "n_samples": 1831,
  "n_features": 21,
  "has_nan": false,
  "dtype": "float64",
  "dimensionality_class": "medium"
}
```



## 6. plan_detection(data_profile=<from step 5>, priority=balanced)

**Args:** `{"data_profile": "<JSON output of step 5>", "priority": "balanced"}`

**Result:**
```
{
  "detector_name": "IForest",
  "params": {},
  "reason": "General tabular: robust all-rounders from ADBench top-5",
  "evidence": [
    "ADBench"
  ],
  "confidence": 0.85,
  "alternatives": [
    {
      "detector_name": "ECOD",
      "params": {},
      "reason": "General tabular: robust all-rounders from ADBench top-5",
      "evidence": [
        "ADBench"
      ],
      "confidence": 0.8,
      "alternatives": []
    },
    {
      "detector_name": "KNN",
      "params": {},
      "reason": "General tabular: robust all-rounders from ADBench top-5",
      "evidence": [
        "ADBench"
      ],
      "confidence": 0.75,
      "alternatives": []
    }
  ]
}
```



## 7. build_detector(plan=<from step 6>)

**Args:** `{"plan": "<JSON output of step 6>"}`

**Result:**
```
{
  "detector_name": "IForest",
  "class_path": "pyod.models.iforest.IForest",
  "params": {},
  "preset": null,
  "code_snippet": "from pyod.models.iforest import IForest\nclf = IForest()"
}
```



## Tier-B probe (expected MISSING)

- `run_detection`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: run_detection
- `analyze_results`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: analyze_results
- `explain_findings`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: explain_findings
- `suggest_next_step`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: suggest_next_step
- `investigate`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: investigate
- `iterate`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: iterate
- `validate`: UNEXPECTEDLY PRESENT (err=True) -> Unknown tool: validate
