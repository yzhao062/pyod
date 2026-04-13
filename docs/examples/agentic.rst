Layer 3: Agentic Investigation
===============================

PyOD 3's ``od-expert`` skill lets any AI agent drive a full anomaly detection investigation through natural conversation. The agent handles benchmark-backed detector selection, multi-detector consensus, quality assessment, and iteration on user feedback, all without OD knowledge.

.. figure:: ../figs/agentic-demo.png
   :alt: PyOD 3 agentic investigation demo on cardiotocography dataset
   :align: center
   :width: 720

   5-turn agentic conversation on the UCI Cardiotocography dataset
   (1,831 fetal heart rate recordings, 21 clinical features).

* **Runnable script**: `agentic_example.py <https://github.com/yzhao062/pyod/blob/development/examples/agentic_example.py>`_
* **Interactive demo**: `agentic_demo.html <https://htmlpreview.github.io/?https://github.com/yzhao062/pyod/blob/development/examples/agentic_demo.html>`_ (open in a browser for the full visual mockup)

----

How It Works
------------

When a user asks about anomalies in their data, PyOD's ``od-expert`` skill auto-activates based on intent keywords. The agent:

1. **Profiles the data** -- detects modality (tabular, time series, graph, text, image)
2. **Selects detectors** -- picks top-3 from PyOD's 60+ catalog based on benchmark evidence (ADBench, TSB-AD, BOND)
3. **Runs in parallel** -- executes all detectors, computes rank-normalized consensus
4. **Assesses quality** -- separation, agreement, stability (each in [0, 1])
5. **Iterates on feedback** -- user says "too many" → engine lowers contamination and re-runs
6. **Generates reports** -- markdown or JSON output with per-detector detail

Activation Paths
----------------

PyOD 3 reaches agents through three paths. Pick whichever matches your stack:

**Claude Code / Claude Desktop**
    The ``od-expert`` skill ships as package data inside the pyod wheel. Install it into Claude Code's skill directory with two commands:

    .. code-block:: bash

        pip install pyod
        pyod-install-skill              # installs to ~/.claude/skills/od-expert/

    For a project-local install (the skill is only active when running Claude Code inside that project), use ``pyod-install-skill --project`` (installs to ``./skills/od-expert/`` in the current working directory). After installing, the skill auto-activates when users mention anomaly detection; restart your Claude Code session to pick it up.

**MCP-compatible agents**
    Run ``python -m pyod.mcp_server`` to expose PyOD tools. Any MCP-compatible LLM picks them based on intent. This path is packaged with PyOD and requires only ``pip install pyod``.

**Python apps / custom agents**
    Import and call directly: ``from pyod.utils.ad_engine import ADEngine``. The agent calls ``engine.investigate(data)`` and follows ``state.next_action`` at each step.

----

Conversation Walkthrough
------------------------

**Dataset:** UCI Cardiotocography: 1,831 fetal heart rate recordings with 21 clinical features. Shipped with PyOD at ``examples/data/cardio.csv``.

**Turn 1.** User asks for anomaly detection:

.. code-block:: text

    User: I have a cardiotocography dataset with 1,831 fetal
          heart rate recordings and 21 clinical features
          (cardio.csv). Find abnormal cases that might indicate
          fetal distress.

    Agent: [od-expert skill activated on anomaly-detection intent]
           Profiled as tabular data. PyOD has 58+ detectors; based
           on ADBench (NeurIPS 2022, 57 datasets) I picked the top 3:
             - IForest  tree ensemble, ADBench top-5
             - ECOD     parameter-free, empirical CDF
             - KNN      proximity-based
           Running all 3 in parallel...
           172 anomalies (9.4%). Agreement 0.68. Quality: HIGH (0.83).

           Top case #1656, all 3 detectors agree:
             IForest  score  0.1696  (threshold 0.0000, 100th pctile)
             ECOD     score 60.5560  (threshold 37.4205, 100th pctile)
             KNN      score  7.5600  (threshold 3.1251,  100th pctile)

Behind the scenes:

.. code-block:: python

    import os
    import numpy as np
    from pyod.utils.ad_engine import ADEngine

    np.random.seed(42)  # for reproducibility
    data = np.genfromtxt(
        os.path.join('examples', 'data', 'cardio.csv'),
        delimiter=',', skip_header=1)
    X = data[:, :-1]

    engine = ADEngine()
    state = engine.investigate(X)
    # state.phase == 'analyzed'
    # state.consensus has scores, labels, agreement
    # state.analysis has best_detector, per_detector_analysis
    # state.quality has separation, agreement, stability, overall

**Turn 2.** User wants a stricter cutoff:

.. code-block:: text

    User: 172 is too many for clinical review. Show only the top 3%.

    Agent: Re-running all 3 detectors with contamination=0.03.
           Each detector recomputes its threshold; consensus is
           majority vote. 52 cases flagged (2.8%). Quality: HIGH.

.. code-block:: python

    state = engine.iterate(
        state, {"action": "adjust_contamination", "value": 0.03})
    state = engine.run(state)
    state = engine.analyze(state)

**Turn 3.** User asks about feature contributions:

.. code-block:: text

    User: What clinical features are driving these?

    Agent: Analyzing top case #1656 via IForest (highest Spearman
           correlation with consensus):
             feature_19  z=6.49
             feature_8   z=5.62
             feature_6   z=4.37
             feature_16  z=3.91
             feature_15  z=3.87
           Cardiotocography features encode fetal heart rate
           variability and deceleration patterns. High z-scores
           here are consistent with fetal distress indicators
           per FIGO guidelines.

.. code-block:: python

    best_idx = state.analysis['best_detector_index']
    best_result = state.results[best_idx]
    explanations = engine.explain_findings(best_result, X=X, top_k=3)

**Turn 4.** User asks about a different modality:

.. code-block:: text

    User: What if I had continuous time-series of these metrics?

    Agent: Same API, different modality. PyOD ships 5 stable time
           series detectors (plus 2 experimental). Based on TSB-AD
           (NeurIPS 2024, 1070 datasets):
             - KShape            #2 overall, shape-based clustering
             - MatrixProfile     subsequence anomalies
             - SpectralResidual  point anomalies
           Just call engine.investigate(ts_data); same workflow.

**Turn 5.** User asks for a final report:

.. code-block:: python

    report = engine.report(state, format='text')
    # Returns markdown report with session summary, quality bars,
    # best detector details, and top anomalies.

    # Or JSON for programmatic consumption:
    report_dict = engine.report(state, format='json')

----

The Session API
---------------

The agentic workflow is built on :class:`~pyod.utils.ad_engine.ADEngine`'s session API. Each method advances the investigation and sets ``state.next_action`` to guide the agent:

========================== ========================================================
Method                     Purpose
========================== ========================================================
``start(X)``               Profile data, return ``InvestigationState``
``plan(state)``            Select top-N detectors from PyOD's catalog
``run(state)``             Execute all detectors, compute consensus
``analyze(state)``         Per-detector analysis, quality metrics, best-detector
``iterate(state, fb)``     Adjust plan based on structured or NL feedback
``report(state)``          Generate markdown or JSON report
``investigate(X)``         One-shot: ``start`` + ``plan`` + ``run`` + ``analyze``
========================== ========================================================

Feedback can be structured (dict) or natural language (string):

.. code-block:: python

    # Structured (executes immediately)
    engine.iterate(state, {"action": "adjust_contamination", "value": 0.05})
    engine.iterate(state, {"action": "exclude", "detectors": ["IForest"]})
    engine.iterate(state, {"action": "include", "detectors": ["ECOD"]})
    engine.iterate(state, {"action": "rerun"})

    # Natural language (parsed to action, may need confirmation)
    engine.iterate(state, "too many false positives")
    engine.iterate(state, "try without IForest")

----

Why This Is Different
---------------------

Without PyOD 3, an AI agent wrapping a library like scikit-learn would:

1. Pick one detector (probably the wrong one)
2. Run it once with default parameters
3. Return raw scores without quality assessment
4. Rely on the LLM to interpret results from first principles

With PyOD 3:

1. Selects detectors via benchmark-backed routing (ADBench, TSB-AD, BOND)
2. Runs top-3 in parallel and builds consensus
3. Assesses result quality quantitatively
4. Uses PyOD's domain knowledge (routing rules, detector metadata) rather than the LLM's general reasoning

The agent becomes an OD expert through the library, not despite it.
