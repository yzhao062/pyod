.. pyod documentation master file, created by
   sphinx-quickstart on Sun May 27 10:56:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyOD V3 documentation!
=================================


.. image:: https://img.shields.io/pypi/v/pyod.svg?color=brightgreen
   :target: https://pypi.org/project/pyod/
   :alt: PyPI version


.. image:: https://anaconda.org/conda-forge/pyod/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyod
   :alt: Anaconda version


.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://pyod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


.. image:: https://img.shields.io/github/stars/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/pyod.svg?color=blue
   :target: https://github.com/yzhao062/pyod/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/pyod
   :target: https://pepy.tech/project/pyod
   :alt: Downloads

.. image:: https://github.com/yzhao062/pyod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/yzhao062/pyod/actions/workflows/testing.yml
   :alt: Testing


.. image:: https://coveralls.io/repos/github/yzhao062/pyod/badge.svg
   :target: https://coveralls.io/github/yzhao062/pyod
   :alt: Coverage Status


.. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
   :target: https://codeclimate.com/github/yzhao062/Pyod/maintainability
   :alt: Maintainability


.. image:: https://img.shields.io/github/license/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/blob/master/LICENSE
   :alt: License


.. image:: https://img.shields.io/badge/ADBench-benchmark_results-pink
   :target: https://github.com/Minqi824/ADBench
   :alt: Benchmark

----

.. note::

   **New in V3.** Any AI agent can now run expert-level anomaly detection on your data. Just ask.

PyOD V3 is the most comprehensive Python library for anomaly detection. Four pillars:

===========================  ========================================================================================
Pillar                       What it means
===========================  ========================================================================================
Multi-Modal                  60+ detectors across **tabular, time series, graph, text, and image** data, one API
Full Lifecycle               From raw data to explained anomalies and next-step guidance in a single call
Agentic                      Ask in plain English, and AI agents run expert-level detection without OD expertise
Most Used                    `38+ million downloads <https://pepy.tech/project/pyod>`_; benchmark-backed routing (ADBench, TSB-AD, BOND, NLP-ADBench)
===========================  ========================================================================================

**Outlier Detection with 5 Lines of Code** (``pip install pyod``):

.. code-block:: python

    from pyod.models.iforest import IForest
    clf = IForest()
    clf.fit(X_train)
    y_train_scores = clf.decision_scores_          # training anomaly scores
    y_test_scores = clf.decision_function(X_test)   # test anomaly scores

**Three ways to use PyOD:**

=========  =====================  ======================================================================  ============================
Layer      Name                   When to use                                                             Entry point
=========  =====================  ======================================================================  ============================
1          Classic API            You know which detector you want                                        :doc:`examples/tabular`
2          ADEngine               You want PyOD to choose, compare, and assess automatically              :doc:`examples/adengine`
3          Agentic Investigation  You want an AI agent to drive OD through natural conversation           :doc:`examples/agentic`
=========  =====================  ======================================================================  ============================

Layers 2 and 3 are powered by :class:`~pyod.utils.ad_engine.ADEngine`, PyOD's intelligent orchestration core. Layer 3 adds the ``od-expert`` skill that auto-activates in Claude Code and MCP-compatible agents.

.. figure:: figs/agentic-demo.png
   :alt: PyOD V3 agentic investigation demo on cardiotocography dataset
   :align: center
   :width: 720

   A real 5-turn agentic conversation on the UCI Cardiotocography
   dataset (1,831 recordings, 21 clinical features).

See :doc:`examples/agentic` for the full walkthrough.

**How PyOD V3 gets triggered:**

* **Claude Code / Claude Desktop**: Copy `skills/od-expert/SKILL.md <https://github.com/yzhao062/pyod/tree/development/skills/od-expert>`_ from the repo into your project ``skills/`` directory or ``~/.claude/skills/``; the skill then auto-activates when users mention anomaly detection. ``pip install pyod`` installs the Python package but does not install the skill file itself.
* **MCP-compatible agents**: Run ``python -m pyod.mcp_server`` to expose PyOD tools. Any MCP-compatible LLM picks them based on intent.
* **Python apps / custom agents**: ``from pyod.utils.ad_engine import ADEngine`` and call ``engine.investigate(data)`` directly.

**PyOD Ecosystem & Resources**:
`ADBench <https://github.com/Minqi824/ADBench>`_ (tabular benchmark) :cite:`a-han2022adbench` | `TSB-AD <https://github.com/TheDatumOrg/TSB-AD>`_ (time series) :cite:`a-liu2024tsb` | `BOND <https://arxiv.org/abs/2206.10071>`_ (graph) :cite:`a-liu2022bond` | `NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_ (NLP) :cite:`a-li2024nlp` | `AD-LLM <https://arxiv.org/abs/2412.11142>`_ (LLM-based AD) :cite:`a-yang2024ad` | `Resources <https://github.com/yzhao062/anomaly-detection-resources>`_

----

About PyOD
^^^^^^^^^^

PyOD, established in 2017, is the longest-running and most widely used Python library for `anomaly detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_. With `38+ million downloads <https://pepy.tech/project/pyod>`_, it serves both academic research and commercial products worldwide.

V3 extends the library with :class:`~pyod.utils.ad_engine.ADEngine` (intelligent orchestration) and the ``od-expert`` skill (agentic workflow), while keeping the classic ``fit``/``predict`` API fully backward-compatible. V3 is built on SUOD :cite:`a-zhao2021suod` for fast parallel training and numba JIT for per-model speedups.

**Citing PyOD**:

If you use PyOD in a scientific publication, we would appreciate citations to the following paper(s):

`PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection <https://arxiv.org/abs/2412.12154>`_ is available as a preprint. If you use PyOD in a scientific publication, we would appreciate citations to the following paper::

    @inproceedings{chen2025pyod,
      title={Pyod 2: A python library for outlier detection with llm-powered model selection},
      author={Chen, Sihan and Qian, Zhuangzhuang and Siu, Wingchun and Hu, Xingcan and Li, Jiaqi and Li, Shawn and Qin, Yuehan and Yang, Tiankai and Xiao, Zhuo and Ye, Wanghao and others},
      booktitle={Companion Proceedings of the ACM on Web Conference 2025},
      pages={2807--2810},
      year={2025}
    }

`PyOD paper <http://www.jmlr.org/papers/volume20/19-011/19-011.pdf>`_ is published in `Journal of Machine Learning Research (JMLR) <http://www.jmlr.org/>`_ (MLOSS track).::

    @article{zhao2019pyod,
        author  = {Zhao, Yue and Nasrullah, Zain and Li, Zheng},
        title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
        journal = {Journal of Machine Learning Research},
        year    = {2019},
        volume  = {20},
        number  = {96},
        pages   = {1-7},
        url     = {http://jmlr.org/papers/v20/19-011.html}
    }

or::

    Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.


For a broader perspective on anomaly detection, see our NeurIPS papers on `ADBench <https://arxiv.org/abs/2206.09426>`_ :cite:`a-han2022adbench` and `ADGym <https://arxiv.org/abs/2309.15376>`_.


----

Benchmarks
^^^^^^^^^^

* `ADBench <https://github.com/Minqi824/ADBench>`_ :cite:`a-han2022adbench`: 30 algorithms on 57 tabular datasets. See `comparison <https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py>`_.
* `NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_ :cite:`a-li2024nlp`: 19 methods on 8 text datasets. Two-step (embedding + detector) beats end-to-end.
* `TSB-AD <https://github.com/TheDatumOrg/TSB-AD>`_ :cite:`a-liu2024tsb`: 40 algorithms on 1070 time series datasets (NeurIPS 2024).
* `BOND <https://arxiv.org/abs/2206.10071>`_ :cite:`a-liu2022bond`: 14 graph anomaly detection algorithms on 14 datasets (NeurIPS 2022).


Implemented Algorithms
======================

PyOD is organized into two functional groups: **(i) Detection Algorithms**, with dedicated subsections for tabular, time series, and graph data (EmbeddingOD inside the tabular table adds multi-modal support for text and image via foundation model encoders); and **(ii) Utility Functions** for data generation, evaluation, and intelligent orchestration.

**(i-a) Tabular & Multi-Modal Detection Algorithms** :

===================  ================  ======================================================================================================  =====  ===================================================  ======================================================
Type                 Abbr              Algorithm                                                                                               Year   Class                                                Ref
===================  ================  ======================================================================================================  =====  ===================================================  ======================================================
Probabilistic        ECOD              Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions                        2022   :class:`pyod.models.ecod.ECOD`                       :cite:`a-li2021ecod`
Probabilistic        COPOD             COPOD: Copula-Based Outlier Detection                                                                   2020   :class:`pyod.models.copod.COPOD`                     :cite:`a-li2020copod`
Probabilistic        ABOD              Angle-Based Outlier Detection                                                                           2008   :class:`pyod.models.abod.ABOD`                       :cite:`a-kriegel2008angle`
Probabilistic        FastABOD          Fast Angle-Based Outlier Detection using approximation                                                  2008   :class:`pyod.models.abod.ABOD`                       :cite:`a-kriegel2008angle`
Probabilistic        MAD               Median Absolute Deviation (MAD)                                                                         1993   :class:`pyod.models.mad.MAD`                         :cite:`a-iglewicz1993detect`
Probabilistic        SOS               Stochastic Outlier Selection                                                                            2012   :class:`pyod.models.sos.SOS`                         :cite:`a-janssens2012stochastic`
Probabilistic        QMCD              Quasi-Monte Carlo Discrepancy outlier detection                                                         2001   :class:`pyod.models.qmcd.QMCD`                       :cite:`a-fang2001wrap`
Probabilistic        KDE               Outlier Detection with Kernel Density Functions                                                         2007   :class:`pyod.models.kde.KDE`                         :cite:`a-latecki2007outlier`
Probabilistic        Sampling          Rapid distance-based outlier detection via sampling                                                     2013   :class:`pyod.models.sampling.Sampling`               :cite:`a-sugiyama2013rapid`
Probabilistic        GMM               Probabilistic Mixture Modeling for Outlier Analysis                                                            :class:`pyod.models.gmm.GMM`                         :cite:`a-aggarwal2015outlier` [Ch.2]
Linear Model         PCA               Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)   2003   :class:`pyod.models.pca.PCA`                         :cite:`a-shyu2003novel`
Linear Model         KPCA              Kernel Principal Component Analysis                                                                     2007   :class:`pyod.models.kpca.KPCA`                       :cite:`a-hoffmann2007kernel`
Linear Model         MCD               Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores)                    1999   :class:`pyod.models.mcd.MCD`                         :cite:`a-rousseeuw1999fast,a-hardin2004outlier`
Linear Model         CD                Use Cook's distance for outlier detection                                                               1977   :class:`pyod.models.cd.CD`                           :cite:`a-cook1977detection`
Linear Model         OCSVM             One-Class Support Vector Machines                                                                       2001   :class:`pyod.models.ocsvm.OCSVM`                     :cite:`a-scholkopf2001estimating`
Linear Model         LMDD              Deviation-based Outlier Detection (LMDD)                                                                1996   :class:`pyod.models.lmdd.LMDD`                       :cite:`a-arning1996linear`
Proximity-Based      LOF               Local Outlier Factor                                                                                    2000   :class:`pyod.models.lof.LOF`                         :cite:`a-breunig2000lof`
Proximity-Based      COF               Connectivity-Based Outlier Factor                                                                       2002   :class:`pyod.models.cof.COF`                         :cite:`a-tang2002enhancing`
Proximity-Based      Incr. COF         Memory Efficient Connectivity-Based Outlier Factor (slower but reduce storage complexity)               2002   :class:`pyod.models.cof.COF`                         :cite:`a-tang2002enhancing`
Proximity-Based      CBLOF             Clustering-Based Local Outlier Factor                                                                   2003   :class:`pyod.models.cblof.CBLOF`                     :cite:`a-he2003discovering`
Proximity-Based      LOCI              LOCI: Fast outlier detection using the local correlation integral                                       2003   :class:`pyod.models.loci.LOCI`                       :cite:`a-papadimitriou2003loci`
Proximity-Based      HBOS              Histogram-based Outlier Score                                                                           2012   :class:`pyod.models.hbos.HBOS`                       :cite:`a-goldstein2012histogram`
Proximity-Based      HDBSCAN           Density-based clustering based on hierarchical density estimates                                        2013   :class:`pyod.models.hdbscan.HDBSCAN`                 :cite:`a-campello2013density`
Proximity-Based      kNN               k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score                  2000   :class:`pyod.models.knn.KNN`                         :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`
Proximity-Based      AvgKNN            Average kNN (use the average distance to k nearest neighbors as the outlier score)                      2002   :class:`pyod.models.knn.KNN`                         :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`
Proximity-Based      MedKNN            Median kNN (use the median distance to k nearest neighbors as the outlier score)                        2002   :class:`pyod.models.knn.KNN`                         :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`
Proximity-Based      SOD               Subspace Outlier Detection                                                                              2009   :class:`pyod.models.sod.SOD`                         :cite:`a-kriegel2009outlier`
Proximity-Based      ROD               Rotation-based Outlier Detection                                                                        2020   :class:`pyod.models.rod.ROD`                         :cite:`a-almardeny2020novel`
Outlier Ensembles    IForest           Isolation Forest                                                                                        2008   :class:`pyod.models.iforest.IForest`                 :cite:`a-liu2008isolation,a-liu2012isolation`
Outlier Ensembles    INNE              Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles                                      2018   :class:`pyod.models.inne.INNE`                       :cite:`a-bandaragoda2018isolation`
Outlier Ensembles    DIF               Deep Isolation Forest for Anomaly Detection                                                             2023   :class:`pyod.models.dif.DIF`                         :cite:`a-xu2023dif`
Outlier Ensembles    FB                Feature Bagging                                                                                         2005   :class:`pyod.models.feature_bagging.FeatureBagging`  :cite:`a-lazarevic2005feature`
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                       2019   :class:`pyod.models.lscp.LSCP`                       :cite:`a-zhao2019lscp`
Outlier Ensembles    XGBOD             Extreme Boosting Based Outlier Detection **(Supervised)**                                               2018   :class:`pyod.models.xgbod.XGBOD`                     :cite:`a-zhao2018xgbod`
Outlier Ensembles    LODA              Lightweight On-line Detector of Anomalies                                                               2016   :class:`pyod.models.loda.LODA`                       :cite:`a-pevny2016loda`
Outlier Ensembles    SUOD              SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**          2021   :class:`pyod.models.suod.SUOD`                       :cite:`a-zhao2021suod`
Neural Networks      AutoEncoder       Fully connected AutoEncoder (use reconstruction error as the outlier score)                             2015   :class:`pyod.models.auto_encoder.AutoEncoder`        :cite:`a-aggarwal2015outlier` [Ch.3]
Neural Networks      VAE               Variational AutoEncoder (use reconstruction error as the outlier score)                                 2013   :class:`pyod.models.vae.VAE`                         :cite:`a-kingma2013auto`
Neural Networks      Beta-VAE          Variational AutoEncoder (all customized loss term by varying gamma and capacity)                        2018   :class:`pyod.models.vae.VAE`                         :cite:`a-burgess2018understanding`
Neural Networks      SO_GAAL           Single-Objective Generative Adversarial Active Learning                                                 2019   :class:`pyod.models.so_gaal.SO_GAAL`                 :cite:`a-liu2019generative`
Neural Networks      MO_GAAL           Multiple-Objective Generative Adversarial Active Learning                                               2019   :class:`pyod.models.mo_gaal.MO_GAAL`                 :cite:`a-liu2019generative`
Neural Networks      DeepSVDD          Deep One-Class Classification                                                                           2018   :class:`pyod.models.deep_svdd.DeepSVDD`              :cite:`a-ruff2018deepsvdd`
Neural Networks      AnoGAN            Anomaly Detection with Generative Adversarial Networks                                                  2017   :class:`pyod.models.anogan.AnoGAN`                   :cite:`a-schlegl2017unsupervised`
Neural Networks      ALAD              Adversarially learned anomaly detection                                                                 2018   :class:`pyod.models.alad.ALAD`                       :cite:`a-zenati2018adversarially`
Neural Networks      DevNet            Deep Anomaly Detection with Deviation Networks                                                          2019   :class:`pyod.models.devnet.DevNet`                   :cite:`a-pang2019deep`
Neural Networks      AE1SVM            Autoencoder-based One-class Support Vector Machine                                                      2019   :class:`pyod.models.ae1svm.AE1SVM`                   :cite:`a-nguyen2019scalable`
Graph-based          R-Graph           Outlier detection by R-graph                                                                            2017   :class:`pyod.models.rgraph.RGraph`                   :cite:`a-you2017provable`
Graph-based          LUNAR             LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks                               2022   :class:`pyod.models.lunar.LUNAR`                     :cite:`a-goodge2022lunar`
Embedding-based      EmbeddingOD       Multi-modal anomaly detection via foundation model embeddings (text, image)                             2025   :class:`pyod.models.embedding.EmbeddingOD`           :cite:`a-li2024nlp`
===================  ================  ======================================================================================================  =====  ===================================================  ======================================================


Ensemble methods (IForest, INNE, DIF, FB, LSCP, LODA, SUOD, XGBOD) are included in the table above. Score combination functions (average, maximization, AOM, MOA, median, majority vote) are in :mod:`pyod.models.combination`.

**(i-b) Time Series Anomaly Detection** :

All time series detectors use the same ``fit``/``predict``/``decision_function`` API as tabular detectors, with one exception: ``MatrixProfile`` is transductive (train-only; use ``decision_scores_`` and ``labels_`` after ``fit()``, no out-of-sample ``predict``).

**Input format**: numpy array of shape ``(n_timestamps,)`` for univariate or ``(n_timestamps, n_channels)`` for multivariate. Each row is one timestep; columns are channels/features. Pandas DataFrames and lists are auto-converted. **Output**: ``decision_scores_`` of shape ``(n_timestamps,)`` with one anomaly score per timestep.

**Time series detection in 3 lines**:

.. code-block:: python

    from pyod.models.ts_kshape import KShape      # or any TS detector
    clf = KShape(window_size=20)
    clf.fit(X_train)                               # shape (n_timestamps,) or (n_timestamps, n_channels)
    scores = clf.decision_scores_                  # per-timestamp anomaly scores

Algorithm rankings from `TSB-AD benchmark <https://github.com/TheDatumOrg/TSB-AD>`_ :cite:`a-liu2024tsb` (NeurIPS 2024, 1070 datasets):

===================  ==================  ======================================================================================================  =====  ==============================================================  ======================================================
Type                 Abbr                Algorithm                                                                                               Year   Class                                                           Ref
===================  ==================  ======================================================================================================  =====  ==============================================================  ======================================================
Windowed Bridge      TimeSeriesOD        Any PyOD detector on sliding windows of time series                                                     2026   :class:`pyod.models.ts_od.TimeSeriesOD`
Subsequence          MatrixProfile       Matrix Profile (STOMP): nearest-neighbor distance, transductive (train-only)                            2016   :class:`pyod.models.ts_matrix_profile.MatrixProfile`            :cite:`a-yeh2016matrix`
Frequency            SpectralResidual    Spectral Residual: FFT-based saliency detection                                                         2019   :class:`pyod.models.ts_spectral_residual.SpectralResidual`      :cite:`a-ren2019time`
Clustering           KShape              k-Shape clustering for subsequence anomaly detection (#2 in TSB-AD)                                     2015   :class:`pyod.models.ts_kshape.KShape`                           :cite:`a-paparrizos2015kshape`
Streaming            SAND                Streaming anomaly detection with drift adaptation (experimental)                                        2021   :class:`pyod.models.ts_sand.SAND`                               :cite:`a-boniol2021sand`
Deep Learning        LSTMAD              LSTM prediction error with Mahalanobis distance scoring                                                 2015   :class:`pyod.models.ts_lstm.LSTMAD`                             :cite:`a-malhotra2015long`
Deep Learning        AnomalyTransformer  Transformer with association discrepancy (experimental)                                                 2022   :class:`pyod.models.ts_anomaly_transformer.AnomalyTransformer`  :cite:`a-xu2022anomaly`
===================  ==================  ======================================================================================================  =====  ==============================================================  ======================================================


**(i-c) Graph Anomaly Detection** (``pip install pyod[graph]``):

All graph detectors are **transductive** in v1: use ``decision_scores_`` and ``labels_`` after ``fit()``. No out-of-sample ``predict``. Input: PyG ``Data`` object with ``x`` (node features) and ``edge_index`` (COO edges). SCAN works without features.

**Graph detection in 3 lines** (``pip install pyod[graph]``):

.. code-block:: python

    from pyod.models.pyg_dominant import DOMINANT
    clf = DOMINANT(hidden_dim=64, epochs=100)
    clf.fit(data)                                  # PyG Data object
    scores = clf.decision_scores_                  # per-node anomaly scores

Algorithm rankings from `BOND benchmark <https://arxiv.org/abs/2206.10071>`_ :cite:`a-liu2022bond` (NeurIPS 2022, 14 datasets):

.. list-table::
   :widths: 18 18 45 5 25 10
   :header-rows: 1

   * - Type
     - Abbr
     - Algorithm
     - Year
     - Class
     - Ref
   * - GCN Autoencoder
     - DOMINANT
     - GCN AE, structure + attribute reconstruction (#1 BOND deep)
     - 2019
     - :class:`pyod.models.pyg_dominant.DOMINANT`
     - :cite:`a-ding2019dominant`
   * - Contrastive
     - CoLA
     - Contrastive self-supervised, local neighbor context (#2 BOND deep)
     - 2022
     - :class:`pyod.models.pyg_cola.CoLA`
     - :cite:`a-liu2022cola`
   * - Contrastive+AE
     - CONAD
     - Contrastive with anomalous-view injection + dual reconstruction
     - 2022
     - :class:`pyod.models.pyg_conad.CONAD`
     - :cite:`a-xu2022conad`
   * - Attention AE
     - AnomalyDAE
     - GAT structure encoder + MLP attribute encoder
     - 2020
     - :class:`pyod.models.pyg_anomalydae.AnomalyDAE`
     - :cite:`a-fan2020anomalydae`
   * - Motif AE
     - GUIDE
     - Dual GCN AE on original + triangle-motif adjacency
     - 2021
     - :class:`pyod.models.pyg_guide.GUIDE`
     - :cite:`a-yuan2021guide`
   * - Matrix Factor.
     - Radar
     - Residual analysis via matrix factorization
     - 2017
     - :class:`pyod.models.pyg_radar.Radar`
     - :cite:`a-li2017radar`
   * - Matrix Factor.
     - ANOMALOUS
     - Joint MF with Laplacian regularization
     - 2018
     - :class:`pyod.models.pyg_anomalous.ANOMALOUS`
     - :cite:`a-peng2018anomalous`
   * - Structural
     - SCAN
     - Structural clustering, no features needed
     - 2007
     - :class:`pyod.models.pyg_scan.SCAN`
     - :cite:`a-xu2007scan`


**(ii) Utility Functions**:

===================  ===============================================  =====================================================================================================================================================
Type                 Name                                             Function
===================  ===============================================  =====================================================================================================================================================
Data                 :func:`~pyod.utils.data.generate_data`           Synthesized data generation; normal data from multivariate Gaussian, outliers from uniform distribution
Data                 :func:`~pyod.utils.data.generate_data_clusters`  Synthesized data generation in clusters for more complex patterns
Data                 :func:`~pyod.utils.data.generate_ts_data`        Synthesized time series data with point and subsequence anomalies
Evaluation           :func:`~pyod.utils.data.evaluate_print`          Print ROC-AUC and Precision @ Rank n for a detector
Evaluation           :func:`~pyod.utils.utility.precision_n_scores`   Calculate Precision @ Rank n
Utility              :func:`~pyod.utils.utility.get_label_n`          Turn raw outlier scores into binary labels by assigning 1 to the top n scores
Stat                 :func:`~pyod.utils.stat_models.wpearsonr`        Calculate the weighted Pearson correlation of two samples
Encoding             :func:`~pyod.utils.encoders.resolve_encoder`     Resolve an encoder from a string, BaseEncoder instance, or callable
Encoding             SentenceTransformerEncoder                       Encode text via sentence-transformers models (see :doc:`pyod.utils <pyod.utils>`)
Encoding             OpenAIEncoder                                    Encode text via OpenAI Embeddings API (see :doc:`pyod.utils <pyod.utils>`)
Encoding             HuggingFaceEncoder                               Encode text or images via HuggingFace transformers (see :doc:`pyod.utils <pyod.utils>`)
Intelligence         :class:`~pyod.utils.ad_engine.ADEngine`          Intelligent anomaly detection lifecycle engine: profiling, planning, execution, analysis, and reporting
===================  ===============================================  =====================================================================================================================================================



API Cheatsheet & Reference
==========================

The following APIs are applicable for all detector models for easy use.

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector. y is ignored in unsupervised methods.
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict raw anomaly score of X using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_confidence`: Predict the model's sample-wise confidence (available in predict and predict_proba).


Key Attributes of a fitted model:

* :attr:`pyod.models.base.BaseDetector.decision_scores_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* :attr:`pyod.models.base.BaseDetector.labels_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   examples/index
   benchmark

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced Topics

   model_persistence
   fast_train
   thresholding


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api_cc
   pyod.models.tabular
   pyod.models.timeseries
   pyod.models.graph
   pyod.models.embedding
   pyod.ad_engine
   pyod.utils


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   impact
   issues
   relevant_knowledge
   pubs
   faq
   about


----


.. rubric:: References

.. bibliography::
   :cited:
   :labelprefix: A
   :keyprefix: a-
