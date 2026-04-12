Benchmarks
==========

PyOD's detector catalog is backed by three peer-reviewed benchmark suites. The :class:`~pyod.utils.ad_engine.ADEngine` routing rules pull their recommendations directly from these studies, so the suggestions users get from Layer 2 and Layer 3 are tied to reproducible evidence.


ADBench (Tabular)
-----------------

`ADBench <https://github.com/Minqi824/ADBench>`_ :cite:`a-han2022adbench` is a 45-page study evaluating 30 anomaly detection algorithms on 57 tabular benchmark datasets (NeurIPS 2022). It is the de-facto reference for PyOD's tabular detector routing.

.. image:: https://github.com/Minqi824/ADBench/blob/main/figs/ADBench.png?raw=true
   :target: https://github.com/Minqi824/ADBench/blob/main/figs/ADBench.png?raw=true
   :alt: ADBench organization

For a simpler visualization, see the comparison driver `compare_all_models.py <https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py>`_.

.. image:: https://github.com/yzhao062/pyod/blob/development/examples/ALL.png?raw=true
   :target: https://github.com/yzhao062/pyod/blob/development/examples/ALL.png?raw=true
   :alt: Comparison of all tabular detectors


TSB-AD (Time Series)
--------------------

`TSB-AD <https://github.com/TheDatumOrg/TSB-AD>`_ :cite:`a-liu2024tsb` is a time-series anomaly detection benchmark of 40 algorithms across 1,070 datasets (NeurIPS 2024). PyOD ships 5 ADEngine-routed stable time-series detectors (TimeSeriesOD, MatrixProfile, SpectralResidual, KShape #2, LSTMAD), selected by ``ADEngine`` based on TSB-AD rankings, plus 2 experimental implementations (SAND, AnomalyTransformer) that are available via direct class import but not yet included in routing. See :doc:`examples/timeseries` for usage.


BOND (Graph)
------------

`BOND <https://arxiv.org/abs/2206.10071>`_ :cite:`a-liu2022bond` benchmarks 14 graph anomaly detection algorithms on 14 datasets (NeurIPS 2022). PyOD's graph detectors (DOMINANT #1 deep, CoLA #2 deep, CONAD, AnomalyDAE, GUIDE, Radar, ANOMALOUS, SCAN) are routed by ADEngine based on BOND results. See :doc:`examples/graph` for usage.


NLP-ADBench (Text)
------------------

`NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_ evaluates 19 methods on 8 text datasets. A key finding is that a two-step approach (foundation model embeddings + an unsupervised detector) beats end-to-end NLP anomaly detection. PyOD implements this as :class:`~pyod.models.embedding.EmbeddingOD`. See :doc:`examples/embedding` for usage.
