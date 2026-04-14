Layer 3: Agentic Investigation
===============================

PyOD 3's ``od-expert`` skill lets any AI agent drive a full anomaly detection investigation through natural conversation. The agent handles benchmark-backed detector selection, multi-detector consensus, quality assessment, adaptive escalation, and iteration on user feedback, all without requiring the user to be an OD expert.

.. figure:: ../figs/agentic-demo.png
   :alt: PyOD 3 agentic investigation demo on a diabetes screening dataset
   :align: center
   :width: 720

   4-turn agentic conversation on a diabetes screening dataset
   (768 patients, 8 clinical features; shipped with PyOD as
   ``examples/data/pima.csv``). The dark callouts alongside the agent's
   turns show the ``od-expert`` skill's live decision-making: modality
   triage, top-10 pitfall checks, and the 11 adaptive escalation triggers.

* **Runnable script**: `agentic_example.py <https://github.com/yzhao062/pyod/blob/development/examples/agentic_example.py>`_
* **Interactive demo**: `agentic_demo.html <https://htmlpreview.github.io/?https://github.com/yzhao062/pyod/blob/development/examples/agentic_demo.html>`_ (open in a browser for the full visual walkthrough with skill decision panels)

----

What the ``od-expert`` skill encodes
------------------------------------

In v3.2.0, the skill grew from a 78-line API reference to roughly 1000 lines of expert content split across an always-loaded ``SKILL.md`` and six on-demand reference files. The content encodes:

* **A master decision tree** that routes the user's data to one of five modalities (tabular, time series, graph, text, image) based on observable properties.
* **Top-10 critical pitfalls**, always in the agent's working memory: unscaled features for distance-based detectors, contamination mismatches, deep learning on tiny data, missing optional extras like ``pyod[graph]``, raw-score reporting, single-detector runs, and so on. The agent walks each one before calling ``engine.run``.
* **11 adaptive escalation triggers** that decide when the agent proceeds autonomously vs. when it pauses to ask the user. Examples: modality ambiguity (T1), contamination uncertainty (T2), detector disagreement post-run (T3), high-stakes domain hints (T8 -- medical, fraud, safety), labels mentioned but not provided (T5). If none of the triggers fire, the agent runs end-to-end without interruption.
* **On-demand reference files** for each modality (``references/tabular.md``, ``references/time_series.md``, ``references/graph.md``, ``references/text_image.md``, plus ``references/workflow.md`` and ``references/pitfalls.md``). The agent loads them when the modality decision routes there.
* **A KB-derived detector list** for each modality, refreshed from ``pyod.utils.knowledge`` by ``scripts/regen_skill.py`` at build time. Every detector name in the skill is mechanically validated against the live KB by a CI safety net test (``pyod/test/test_skill_kb_consistency.py``) so drift fails the build.
* **An API safety net** (``pyod/test/test_skill_api_refs.py``) that walks ``ADEngine`` and ``InvestigationState`` via a live dry run and validates every ``state.X`` / ``state.X['key']`` / ``engine.X`` reference in the skill content. Added in v3.2.1 after a regression that shipped invented API names in v3.2.0.

See :doc:`the skill maintenance methodology guide <../skill_maintenance>` for the full pattern and for how to add a new skill.

----

How It Works
------------

When a user asks about anomalies in their data, PyOD's ``od-expert`` skill auto-activates based on intent keywords. The agent then:

1. **Walks the master decision tree** -- timestamps, graph structure, text/image, or tabular? Load the matching ``references/<modality>.md``.
2. **Walks the top-10 pitfall checklist** -- is any pitfall active for this data? Example: feature scale ratio > 100 triggers Pitfall 1 (unscaled features for distance-based detectors) and the agent recommends a pre-scaling step or flags it in the report.
3. **Walks the 11 escalation triggers** -- does anything about the request call for a pause? Example: "medical screening" fires Trigger 8 (high-stakes domain) and the agent commits to dual-detector validation and a confidence caveat.
4. **Selects detectors** -- calls ``engine.plan(state)`` to pick the top-3 from PyOD's 61-detector catalog based on benchmark evidence (ADBench, TSB-AD, BOND). Each plan entry in ``state.plans`` has ``detector_name``, ``confidence``, ``reason``, ``evidence``.
5. **Runs in parallel** -- executes all selected detectors and builds a rank-normalized consensus in ``state.consensus``.
6. **Re-walks a subset of triggers post-run** -- detector disagreement (T3), weak quality (T4), suspiciously clean results (T10). If any fire, the agent hedges the report or iterates.
7. **Generates a report** -- Markdown or JSON, always including a "what I assumed and why" block that lists the contamination rate, the detectors used, the best detector, and any caveats the trigger/pitfall walk surfaced.

The agent's decisions at each of these steps are visible in the interactive demo's dark "od-expert" panels.

----

Activation Paths
----------------

PyOD 3 reaches agents through three paths. Pick whichever matches your stack:

**Claude Code / Claude Desktop / Codex**
    The ``od-expert`` skill ships as package data inside the pyod wheel.
    Two install modes are supported:

    .. code-block:: bash

        pip install pyod

        # Claude Code / Claude Desktop: user-global install
        pyod install skill              # installs to ~/.claude/skills/od-expert/

        # Codex: project-local install (Codex reads ./skills/, has no user-global dir)
        pyod install skill --project    # installs to ./skills/od-expert/

    After installing, run ``pyod info`` to confirm the skill is detected
    and to see which agent stacks are present (``pyod info`` checks both
    ``~/.claude/`` and ``~/.codex/`` and recommends the right install
    command). The legacy ``pyod-install-skill`` command from v3.0.0 still
    works as an alias for ``pyod install skill``.

**MCP-compatible agents**
    Install the optional MCP extra, then launch the server:

    .. code-block:: bash

        pip install pyod[mcp]
        pyod mcp serve              # alias for ``python -m pyod.mcp_server``

    Any MCP-compatible LLM picks up the tools based on intent.

**Python apps / custom agents**
    Import and call directly: ``from pyod.utils.ad_engine import ADEngine``. The agent calls ``engine.investigate(data)`` and follows ``state.next_action`` at each step.

----

Conversation Walkthrough
------------------------

**Dataset:** Diabetes screening dataset, 768 patients with 8 clinical features (glucose, blood pressure, skin thickness, insulin, BMI, pedigree, age, pregnancies). Shipped with PyOD at ``examples/data/pima.csv``.

Why this dataset? It exercises the skill's machinery: the feature scale ratio is ~347x (Pitfall 1 fires), and the user's "medical screening" context fires Trigger 8 (high-stakes domain). On a happy-path dataset the side callouts would be mostly empty; on this data, they catch real problems.

**Turn 1.** User asks for screening:

.. code-block:: text

    User: I have a diabetes screening dataset
          -- 768 patients with 8 clinical features (glucose,
          blood pressure, BMI, insulin, age, etc.). Find
          unusual patient profiles that might need follow-up.
          This is a medical screening context.

    [od-expert skill: pre-run decisions]
      Master tree     -> tabular (no timestamps, no graph, no text/image)
      Pitfall #1      -> FIRES: std ratio 347x, note as report caveat
      Pitfalls #2-10  -> all pass
      Trigger T8      -> FIRES: medical screening; dual-detector validation
      Other triggers  -> not applicable
      Plan            -> KNN (0.85), IForest (0.85), LOF (0.80)

    Agent: Profiled as tabular, low-dim, n=768 x 8. Running
           KNN + IForest + LOF from the ADBench top-5 for
           low-dim small datasets. Scale mismatch noted for
           the final report.

           Results: 62 flagged (8.1%), separation 0.96,
           agreement 0.59, quality HIGH (0.79). Top case:
           patient #13. KNN strongest individually.

Behind the scenes:

.. code-block:: python

    import numpy as np
    from pyod.utils.ad_engine import ADEngine

    np.random.seed(42)
    data = np.genfromtxt('examples/data/pima.csv',
                         delimiter=',', skip_header=1)
    X = data[:, :-1]

    engine = ADEngine()
    state = engine.investigate(X)
    # state.plans[0]['detector_name']          == 'KNN'
    # state.analysis['best_detector']          == 'KNN'
    # state.analysis['consensus_analysis']['n_anomalies']    == 62
    # state.analysis['consensus_analysis']['anomaly_ratio']  == 0.081
    # state.quality['separation']              == 0.956
    # state.quality['agreement']               == 0.592
    # state.quality['stability']               == 0.814
    # state.quality['overall']                 == 0.787
    # state.quality['verdict']                 == 'high'

**Turn 2.** User asks what is driving the flags:

.. code-block:: text

    User: What clinical features are driving the flags?

    Agent: Analyzing patient #13 via KNN (best detector).
           engine.explain_findings returns absolute z-score
           magnitudes (|z|); raw values are cross-checked
           manually to show direction:
             insulin       |z|=6.65  (846 uU/mL)
             age           |z|=2.19  (59 yrs)
             glucose       |z|=2.13  (189 mg/dL)
             pregnancies   |z|=0.84  (1)
           Metabolic-syndrome profile: extreme insulin,
           elevated glucose, older age.

.. code-block:: python

    best_idx = state.analysis['best_detector_index']
    best_result = state.results[best_idx]
    explanations = engine.explain_findings(best_result, X=X, top_k=5)
    # explanations[0]['contributing_features'][0] ==
    #   {'feature': 4, 'z_score': 6.65}  # insulin (absolute magnitude)

**Turn 3.** User reveals labels exist. Trigger 5 fires:

.. code-block:: text

    User: I should mention -- I actually have diabetes
          diagnosis labels for the full cohort.

    [od-expert skill: new turn, re-evaluating triggers]
      Trigger T5 -> FIRES: labels mentioned but the session
                    is on the unsupervised path
      Recommended switch -> drop out of ADEngine session and
                            call XGBOD directly (pyod[xgboost])

    Agent: That changes the plan. The ADEngine session API is
           unsupervised in v3.2.x. For labelled data, supervised
           detection (XGBOD) typically lifts AUC 10-20 points.
           I would switch to the classic fit / predict path.

.. code-block:: python

    # ADEngine session is unsupervised in v3.2.x; for labels,
    # use the classic fit / predict path directly:
    from pyod.models.xgbod import XGBOD
    clf = XGBOD()
    clf.fit(X, y_labels)               # supervised training
    scores = clf.decision_function(X)  # anomaly scores
    labels_pred = clf.predict(X)       # binary labels

**Turn 4.** User asks for the unsupervised report for now:

.. code-block:: python

    report = engine.report(state, format='text')
    # Includes the quality bars, the selected detectors, the
    # best detector, and an explicit "assumptions and caveats"
    # block citing the scale mismatch (Pitfall 1), the
    # observed anomaly ratio, the high-stakes caveat (Trigger
    # 8), and the label-availability note (Trigger 5).

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
``plan(state)``            Select top-N detectors, populate ``state.plans``
``run(state)``             Execute all detectors, fill ``state.consensus``
``analyze(state)``         Populate ``state.quality`` and ``state.analysis``
``iterate(state, fb)``     Adjust plan based on structured or NL feedback
``report(state)``          Generate markdown or JSON report
``investigate(X)``         One-shot: ``start`` + ``plan`` + ``run`` + ``analyze``
========================== ========================================================

Key ``state`` fields the agent reads:

* ``state.profile`` -- dict with ``data_type``, ``n_samples``, ``n_features``, ``has_nan``, ``dtype``, ``dimensionality_class``.
* ``state.plans`` -- list of plan dicts with ``detector_name``, ``confidence``, ``reason``, ``evidence``, ``alternatives``.
* ``state.consensus`` -- dict with ``scores``, ``labels``, ``n_detectors``, ``agreement``.
* ``state.quality`` -- dict with ``separation``, ``agreement``, ``stability``, ``overall``, ``verdict``, ``explanation``.
* ``state.analysis`` -- dict with ``consensus_analysis`` (containing ``n_anomalies``, ``anomaly_ratio``, ``top_anomalies``, etc.), ``best_detector``, ``best_detector_index``, ``per_detector_analysis``.
* ``state.next_action`` -- dict with ``action`` in {``report_to_user``, ``iterate``, ``confirm_with_user``}, plus ``reason``, ``summary``, and sometimes ``suggestion`` / ``proposed_change``.

Feedback to ``iterate(state, fb)`` can be structured (dict) or natural language (string):

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

With PyOD 3 and the v3.2.0 ``od-expert`` skill:

1. Walks a master decision tree to pick the right modality and detector family.
2. Walks a top-10 pitfall checklist before running, catching problems like scale mismatch or DL on tiny data.
3. Walks 11 adaptive escalation triggers to decide when to pause and ask vs. run autonomously.
4. Selects detectors via benchmark-backed routing (ADBench, TSB-AD, BOND).
5. Runs top-3 in parallel and builds rank-normalized consensus.
6. Re-checks quality-related triggers post-run and hedges the report accordingly.
7. Always reports the assumptions and caveats, including the scale mismatch, contamination, and any triggered escalations.

The agent becomes an OD expert through the library, not despite it.
