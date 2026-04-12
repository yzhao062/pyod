Examples
========

PyOD offers three ways to do anomaly detection, from simple to fully agentic. This section is organized around those three layers, plus by data type and advanced topics.

.. image:: ../figs/agentic-demo.png
   :alt: PyOD V3 agentic investigation demo
   :align: center
   :width: 720

----

Three Approaches to PyOD
------------------------

**Layer 1: Classic API**. Pick a detector, fit, predict. Best when you know which detector you want.

.. code-block:: python

    from pyod.models.iforest import IForest
    clf = IForest()
    clf.fit(X_train)
    scores = clf.decision_scores_

See :doc:`tabular`, :doc:`timeseries`, :doc:`graph`, :doc:`embedding`.

**Layer 2: ADEngine**. Let PyOD choose, compare, and assess. Best when you do not know which detector to use.

.. code-block:: python

    from pyod.utils.ad_engine import ADEngine
    state = ADEngine().investigate(X)
    print(state.analysis['summary'])

See :doc:`adengine`.

**Layer 3: Agentic Investigation**. Any AI agent drives the workflow through natural conversation. Best for interactive use with user feedback.

.. code-block:: text

    User:  I have some cardiotocography data. Can you find abnormal cases?
    Agent: Found 172 anomalies in 1831 recordings (9.4%) by consensus of 3
           detectors (IForest, ECOD, KNN). Agreement is 0.68 (Spearman).
           Top case is recording #1656 (IForest raw score 0.17, 100th pctile).
           Should I dig deeper or iterate?
    User:  Focus on the top 50 and cross-check against the clinical labels.
    Agent: [runs analyze() with tighter top-k, validates against ground truth]

Behind the scenes the agent calls ``engine.investigate()``, ``iterate()``, and ``report()``,
following ``state.next_action`` at each step. See :doc:`agentic` for the full walkthrough.

----

Examples by Data Type
---------------------

* :doc:`tabular`: 50+ detectors for tabular data (ECOD, IForest, KNN, LOF, ...)
* :doc:`timeseries`: 5 shipped + 2 experimental time series detectors (KShape, MatrixProfile, SpectralResidual, ...)
* :doc:`graph`: 8 graph detectors (DOMINANT, CoLA, CONAD, ...)
* :doc:`embedding`: Text and image detection via foundation model embeddings

----

Advanced Topics
---------------

* :doc:`combination`: Model combination (average, max, AOM, MOA)
* :doc:`thresholding`: Data-driven threshold selection

----

Featured Tutorials
------------------

External tutorials and articles about PyOD:

* **Analytics Vidhya**: `An Awesome Tutorial to Learn Outlier Detection in Python using PyOD Library <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_
* **KDnuggets**: `Intuitive Visualization of Outlier Detection Methods <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_
* **Towards Data Science**: `Anomaly Detection for Dummies <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_
* **awesome-machine-learning**: `General-Purpose Machine Learning <https://github.com/josephmisiti/awesome-machine-learning#python-general-purpose>`_

.. toctree::
   :maxdepth: 1
   :hidden:

   agentic
   adengine
   tabular
   timeseries
   graph
   embedding
   combination
   thresholding
