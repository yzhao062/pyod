Layer 2: ADEngine Intelligent Orchestration
============================================

ADEngine is PyOD's intelligent anomaly detection engine. It profiles your data, selects benchmark-backed detectors from PyOD's 60+ catalog, runs multiple detectors in parallel, computes consensus scores, and assesses result quality, all in one call.

Use Layer 2 when you are not sure which detector to pick.

Full example: `investigate_example.py <https://github.com/yzhao062/pyod/blob/development/examples/investigate_example.py>`_

----

One-shot Investigation
----------------------

The simplest way to use ADEngine is ``investigate()``, which runs the full workflow (profile, plan, run, analyze) in one call:

.. code-block:: python

    import numpy as np
    from pyod.utils.ad_engine import ADEngine

    np.random.seed(42)  # for reproducible output
    data = np.genfromtxt('examples/data/cardio.csv', delimiter=',', skip_header=1)
    X = data[:, :-1]

    engine = ADEngine()
    state = engine.investigate(X)

    print(state.analysis['summary'])
    # "172 anomalies detected out of 1831 samples (9.4%) by
    #  consensus of 3 detectors. Quality: high (0.83)."

    print(state.quality['verdict'])            # 'high'
    print(state.quality['overall'])            # 0.83

----

Multi-Detector Consensus
------------------------

ADEngine runs the top-3 detectors from PyOD's knowledge base and computes a consensus:

.. code-block:: python

    state = engine.investigate(X)

    # Per-detector results
    for result in state.results:
        print("%s: %d anomalies in %.2fs" % (
            result['detector_name'],
            result['n_anomalies'],
            result['runtime_seconds']))

    # Consensus
    print("Consensus scores:", state.consensus['scores'][:5])
    print("Agreement (Spearman):", state.consensus['agreement'])
    print("Disagreements:", len(state.consensus['disagreements']))

----

Quality Assessment
------------------

ADEngine quantifies how trustworthy the results are through three metrics:

* **Separation** -- ratio of anomaly scores to inlier scores ([0, 1])
* **Agreement** -- mean pairwise Spearman correlation between detectors ([0, 1])
* **Stability** -- Jaccard index of top-k sets under +/- 20% contamination ([0, 1])

.. code-block:: python

    q = state.quality
    print("Separation:", q['separation'])      # 1.00
    print("Agreement:",  q['agreement'])       # 0.68
    print("Stability:",  q['stability'])       # 0.82
    print("Overall:",    q['overall'])         # 0.83
    print("Verdict:",    q['verdict'])         # 'high'

Verdicts are ``'high'`` (>=0.7), ``'medium'`` (>=0.4), or ``'low'`` (<0.4).

----

Session API (Step by Step)
--------------------------

For finer control, use the session methods individually:

.. code-block:: python

    engine = ADEngine()
    state = engine.start(X)                    # phase='profiled'
    state = engine.plan(state)                 # phase='planned'
    state = engine.run(state)                  # phase='detected'
    state = engine.analyze(state)              # phase='analyzed'

At each step, ``state.next_action`` tells the caller what to do next. This is how agents follow the workflow without knowing OD.

----

Iteration
---------

If the results are not what you want, iterate with structured feedback:

.. code-block:: python

    # Adjust contamination (structured feedback)
    state = engine.iterate(state, {"action": "adjust_contamination", "value": 0.05})
    state = engine.run(state)
    state = engine.analyze(state)

    # Exclude a detector
    state = engine.iterate(state, {"action": "exclude", "detectors": ["IForest"]})

    # Add a detector
    state = engine.iterate(state, {"action": "include", "detectors": ["ECOD"]})

    # Or pass natural language (best-effort parsing):
    state = engine.iterate(state, "too many false positives")
    # state.next_action == 'confirm_with_user' with proposed_change

----

Reports
-------

Generate text or JSON reports:

.. code-block:: python

    text_report = engine.report(state, format='text')
    print(text_report)

    json_report = engine.report(state, format='json')
    # Native Python dict with session + best_detector sections

----

Data Type Routing
-----------------

ADEngine routes automatically based on data type:

============ ============================================== =====================
Data Type    Default Detectors                              Benchmark
============ ============================================== =====================
Tabular      IForest, ECOD, KNN                             ADBench (NeurIPS 2022)
Time Series  KShape, SpectralResidual, TimeSeriesOD         TSB-AD (NeurIPS 2024)
Graph        DOMINANT, CoLA, Radar                          BOND (NeurIPS 2022)
Text         EmbeddingOD (MiniLM + KNN)                     NLP-ADBench
Image        EmbeddingOD (DINOv2 + KNN)                     NLP-ADBench
============ ============================================== =====================

The data type is auto-detected from the input (``dict`` → multimodal, ``list[str]`` → text, numpy array → tabular/time_series, PyG ``Data`` → graph). You can override with ``engine.investigate(X, data_type='time_series')``.
