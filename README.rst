Python Outlier Detection (PyOD) 3
==================================

**PyOD 3: Agentic Anomaly Detection At Scale**

|badge_pypi| |badge_anaconda| |badge_docs| |badge_stars| |badge_forks| |badge_downloads| |badge_testing| |badge_coverage| |badge_maintainability| |badge_license| |badge_benchmark|

.. |badge_pypi| image:: https://img.shields.io/pypi/v/pyod.svg?color=brightgreen
   :target: https://pypi.org/project/pyod/
   :alt: PyPI version

.. |badge_anaconda| image:: https://anaconda.org/conda-forge/pyod/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyod
   :alt: Anaconda version

.. |badge_docs| image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://pyod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. |badge_stars| image:: https://img.shields.io/github/stars/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/stargazers
   :alt: GitHub stars

.. |badge_forks| image:: https://img.shields.io/github/forks/yzhao062/pyod.svg?color=blue
   :target: https://github.com/yzhao062/pyod/network
   :alt: GitHub forks

.. |badge_downloads| image:: https://pepy.tech/badge/pyod
   :target: https://pepy.tech/project/pyod
   :alt: Downloads

.. |badge_testing| image:: https://github.com/yzhao062/pyod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/yzhao062/pyod/actions/workflows/testing.yml
   :alt: Testing


.. |badge_coverage| image:: https://coveralls.io/repos/github/yzhao062/pyod/badge.svg
   :target: https://coveralls.io/github/yzhao062/pyod
   :alt: Coverage Status

.. |badge_maintainability| image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
   :target: https://codeclimate.com/github/yzhao062/Pyod/maintainability
   :alt: Maintainability

.. |badge_license| image:: https://img.shields.io/github/license/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/blob/master/LICENSE
   :alt: License

.. |badge_benchmark| image:: https://img.shields.io/badge/ADBench-benchmark_results-pink
   :target: https://github.com/Minqi824/ADBench
   :alt: Benchmark


-----

    **PyOD is now agentic.** Any AI agent can drive an expert-level anomaly detection investigation on your data in plain English. The classic ``fit``/``predict`` API stays unchanged.

-----

PyOD 3 is the most comprehensive Python library for anomaly detection. Four pillars:

===========================  ========================================================================================
Pillar                       What it means
===========================  ========================================================================================
Multi-Modal                  60+ detectors across **tabular, time series, graph, text, and image** data, one API
Full Lifecycle               From raw data to explained anomalies and next-step guidance in a single call
Agentic                      Ask in plain English, and AI agents run expert-level detection without OD expertise
Most Used                    38+ million downloads; benchmark-backed routing (ADBench, TSB-AD, BOND, NLP-ADBench)
===========================  ========================================================================================

**Outlier Detection with 5 Lines of Code** (``pip install pyod``):

.. code-block:: python

    from pyod.models.iforest import IForest
    clf = IForest()
    clf.fit(X_train)
    y_train_scores = clf.decision_scores_          # training anomaly scores
    y_test_scores = clf.decision_function(X_test)   # test anomaly scores

**Three ways to use PyOD:**

=========  =====================  ======================================================================  =======================================
Layer      Name                   When to use                                                             Entry point
=========  =====================  ======================================================================  =======================================
1          Classic API            You know which detector you want                                        `Layer 1 examples <https://pyod.readthedocs.io/en/latest/examples/tabular.html>`__
2          ADEngine               You want PyOD to choose, compare, and assess automatically              `Layer 2 walkthrough <https://pyod.readthedocs.io/en/latest/examples/adengine.html>`__
3          Agentic Investigation  You want an AI agent to drive OD through natural conversation           `Layer 3 walkthrough <https://pyod.readthedocs.io/en/latest/examples/agentic.html>`__
=========  =====================  ======================================================================  =======================================

Layers 2 and 3 are powered by ``ADEngine``, PyOD's intelligent orchestration core. Layer 3 adds two agentic activation paths: the ``od-expert`` skill for Claude Code (copy `skills/od-expert/SKILL.md <https://github.com/yzhao062/pyod/tree/development/skills/od-expert>`__ into your project ``skills/`` directory) and an MCP server (``python -m pyod.mcp_server``) that works with any MCP-compatible LLM out of the box.

.. image:: https://raw.githubusercontent.com/yzhao062/pyod/development/docs/figs/agentic-demo.png
   :alt: PyOD 3 agentic investigation demo on cardiotocography dataset
   :align: center
   :width: 720

The figure above shows a real 5-turn agentic conversation on the UCI Cardiotocography dataset. See the `full walkthrough <https://pyod.readthedocs.io/en/latest/examples/agentic.html>`__, runnable `agentic example <https://github.com/yzhao062/pyod/blob/development/examples/agentic_example.py>`__, or interactive `HTML demo <https://htmlpreview.github.io/?https://github.com/yzhao062/pyod/blob/development/examples/agentic_demo.html>`__.

**PyOD Ecosystem & Resources**:
`NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_ (NLP anomaly detection) | `TODS <https://github.com/datamllab/tods>`_ (time-series) | `PyGOD <https://pygod.org/>`_ (graph) | `ADBench <https://github.com/Minqi824/ADBench>`_ (benchmark) | `AD-LLM <https://arxiv.org/abs/2412.11142>`_ (LLM-based AD) [#Yang2024ad]_ | `Resources <https://github.com/yzhao062/anomaly-detection-resources>`_

----


About PyOD
^^^^^^^^^^

PyOD, established in 2017, is the longest-running and most widely used Python library for anomaly detection. With `38+ million downloads <https://pepy.tech/project/pyod>`_, it serves both academic research (featured in `Analytics Vidhya <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_, `KDnuggets <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_, and `Towards Data Science <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_) and commercial products.

V3 extends the library with ``ADEngine`` (intelligent orchestration) and the ``od-expert`` skill (agentic workflow), while keeping the classic ``fit``/``predict`` API fully backward-compatible. V3 is built on SUOD [#Zhao2021SUOD]_ for fast parallel training and numba JIT for per-model speedups.

**Impact & Recognition**:

===================================  ===========================================================================
Area                                 Examples
===================================  ===========================================================================
Space & science                      European Space Agency `OPS-SAT spacecraft telemetry benchmark <https://www.nature.com/articles/s41597-025-05035-3>`_ (*Nature Scientific Data*, 2025) uses PyOD for all 30 algorithms.
Enterprise deployment                Walmart (1M+ daily pricing updates, KDD 2019), Databricks (Kakapo framework integrating PyOD with MLflow/Hyperopt; insider-threat detection solution), IQVIA (123K+ pharmacy claims), Altair AI Studio, Ericsson (patent `WO2023166515A1 <https://patents.google.com/patent/WO2023166515A1>`_).
Books                                `Outlier Detection in Python <https://www.manning.com/books/outlier-detection-in-python>`_ (Brett Kennedy, Manning); *Handbook of Anomaly Detection with Python* (Chris Kuo, Columbia); `Finding Ghosts in Your Data <https://link.springer.com/book/10.1007/978-1-4842-8870-2>`_ (Kevin Feasel, Apress).
Courses                              DataCamp `Anomaly Detection in Python <https://www.datacamp.com/courses/anomaly-detection-in-python>`_ (19M+ platform learners), Manning `liveProject <https://www.manning.com/liveproject/using-pyod-and-ensembles-methods>`_, O'Reilly video edition, multiple Udemy courses.
Podcasts                             `Talk Python To Me #497 <https://talkpython.fm/episodes/show/497/outlier-detection-with-python>`_, `Real Python Podcast #208 <https://realpython.com/podcasts/rpp/208/>`_.
International                        Tutorials in 5 non-English languages: Chinese (CSDN, Zhihu, 搜狐, 机器之心, `aidoczh.com <https://www.aidoczh.com>`_ full doc translation), Japanese, Korean, German, Spanish.
===================================  ===========================================================================

See the `full impact page <https://pyod.readthedocs.io/en/latest/impact.html>`_ on Read the Docs for the complete list of citations, enterprise deployments, patents, and media coverage.

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


For a broader perspective on anomaly detection, see our NeurIPS papers on `ADBench <https://arxiv.org/abs/2206.09426>`_ [#Han2022ADBench]_ and `ADGym <https://arxiv.org/abs/2309.15376>`_.


**Table of Contents**:

* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `Benchmarks <#benchmarks>`_
* `Implemented Algorithms <#implemented-algorithms>`_ (Tabular, Time Series, Graph, Embedding)
* `Additional Topics <#additional-topics>`_ (Model Save/Load, SUOD, Thresholding)
* `Quick Start for Outlier Detection <#quick-start-for-outlier-detection>`_
* `How to Contribute <#how-to-contribute>`_
* `Inclusion Criteria <#inclusion-criteria>`_

----

Installation
^^^^^^^^^^^^

PyOD is designed for easy installation using either **pip** or **conda**. We recommend using the latest version of PyOD due to frequent updates and enhancements:

.. code-block:: bash

   pip install pyod            # normal install
   pip install --upgrade pyod  # or update if needed

.. code-block:: bash

   conda install -c conda-forge pyod

Alternatively, you can clone and run the setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   cd pyod
   pip install .

**Required Dependencies**:

* Python 3.9 or higher
* joblib
* matplotlib
* numpy>=1.19
* numba>=0.51
* scipy>=1.5.1
* scikit_learn>=0.22.0

**Optional Dependencies** (install only what you need):

.. list-table::
   :widths: 33 33 34
   :header-rows: 0

   * - ``pytorch``: deep learning models (AutoEncoder, VAE, DeepSVDD)
     - ``suod``: SUOD acceleration framework
     - ``xgboost``: XGBOD supervised detector
   * - ``combo``: model combination, FeatureBagging
     - ``pythresh``: data-driven thresholding
     - ``sentence-transformers``: EmbeddingOD text
   * - ``openai``: EmbeddingOD with OpenAI embeddings
     - ``transformers``, ``torch``: EmbeddingOD image, HuggingFace encoder
     - ``torch_geometric``: graph detectors (``pip install pyod[graph]``)

----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

The full API Reference is split by modality at `PyOD Documentation <https://pyod.readthedocs.io/en/latest/>`_: `Tabular <https://pyod.readthedocs.io/en/latest/pyod.models.tabular.html>`_, `Time Series <https://pyod.readthedocs.io/en/latest/pyod.models.timeseries.html>`_, `Graph <https://pyod.readthedocs.io/en/latest/pyod.models.graph.html>`_, `Embedding <https://pyod.readthedocs.io/en/latest/pyod.models.embedding.html>`_, `ADEngine <https://pyod.readthedocs.io/en/latest/pyod.ad_engine.html>`_, `Utilities <https://pyod.readthedocs.io/en/latest/pyod.utils.html>`_. Below is a quick cheatsheet for all detectors:

* **fit(X)**: Fit the detector. The parameter y is ignored in unsupervised methods.
* **decision_function(X)**: Predict raw anomaly scores for X using the fitted detector.
* **predict(X)**: Determine whether a sample is an outlier or not as binary labels using the fitted detector.
* **predict_proba(X)**: Estimate the probability of a sample being an outlier using the fitted detector.
* **predict_confidence(X)**: Assess the model's confidence on a per-sample basis (applicable in predict and predict_proba) [#Perini2020Quantifying]_.
* **predict_with_rejection(X)**\ : Allow the detector to reject (i.e., abstain from making) highly uncertain predictions (output = -2) [#Perini2023Rejection]_.

**Key Attributes of a fitted model**:

* **decision_scores_**: Outlier scores of the training data. Higher scores typically indicate more abnormal behavior. Outliers usually have higher scores.
* **labels_**: Binary labels of the training data, where 0 indicates inliers and 1 indicates outliers/anomalies.


----


Benchmarks
^^^^^^^^^^

* `ADBench <https://github.com/Minqi824/ADBench>`_ [#Han2022ADBench]_: 30 algorithms on 57 tabular datasets. See `comparison <https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py>`_.
* `NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_: 19 methods on 8 text datasets. Two-step (embedding + detector) beats end-to-end.
* `TSB-AD <https://github.com/TheDatumOrg/TSB-AD>`_ [#Liu2024TSB]_: 40 algorithms on 1070 time series datasets (NeurIPS 2024).
* `BOND <https://arxiv.org/abs/2206.10071>`_ [#Liu2022BOND]_: 14 graph anomaly detection algorithms on 14 datasets (NeurIPS 2022).

----

Additional Topics
^^^^^^^^^^^^^^^^^

* `Model Save & Load <https://pyod.readthedocs.io/en/latest/model_persistence.html>`_: Use joblib or pickle for saving and loading PyOD models. See `example <https://github.com/yzhao062/pyod/blob/master/examples/save_load_model_example.py>`_.
* `Fast Train with SUOD <https://pyod.readthedocs.io/en/latest/fast_train.html>`_: Accelerate training and prediction with the SUOD framework [#Zhao2021SUOD]_. See `example <https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py>`_.
* `Thresholding Outlier Scores <https://pyod.readthedocs.io/en/latest/thresholding.html>`_: Data-driven approaches for setting contamination levels via `PyThresh <https://github.com/KulikDM/pythresh>`_.

----



Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD is organized into two functional groups: **(i) Detection Algorithms**, with dedicated subsections for tabular, time series, and graph data (EmbeddingOD inside the tabular table adds multi-modal support for text and image via foundation model encoders); and **(ii) Utility Functions** for data generation, evaluation, and intelligent orchestration.

**(i-a) Tabular & Multi-Modal Detection Algorithms** :

.. list-table::
   :widths: 15 14 58 5 8
   :header-rows: 1

   * - Type
     - Abbr
     - Algorithm
     - Year
     - Ref
   * - Probabilistic
     - ECOD
     - Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions (`example <https://github.com/yzhao062/pyod/blob/development/examples/ecod_example.py>`_)
     - 2022
     - [#Li2021ECOD]_
   * - Probabilistic
     - ABOD
     - Angle-Based Outlier Detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/abod_example.py>`_)
     - 2008
     - [#Kriegel2008Angle]_
   * - Probabilistic
     - FastABOD
     - Fast Angle-Based Outlier Detection using approximation (`example <https://github.com/yzhao062/pyod/blob/development/examples/abod_example.py>`_)
     - 2008
     - [#Kriegel2008Angle]_
   * - Probabilistic
     - COPOD
     - COPOD: Copula-Based Outlier Detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/copod_example.py>`_)
     - 2020
     - [#Li2020COPOD]_
   * - Probabilistic
     - MAD
     - Median Absolute Deviation (MAD) (`example <https://github.com/yzhao062/pyod/blob/development/examples/mad_example.py>`_)
     - 1993
     - [#Iglewicz1993How]_
   * - Probabilistic
     - SOS
     - Stochastic Outlier Selection (`example <https://github.com/yzhao062/pyod/blob/development/examples/sos_example.py>`_)
     - 2012
     - [#Janssens2012Stochastic]_
   * - Probabilistic
     - QMCD
     - Quasi-Monte Carlo Discrepancy outlier detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/qmcd_example.py>`_)
     - 2001
     - [#Fang2001Wrap]_
   * - Probabilistic
     - KDE
     - Outlier Detection with Kernel Density Functions (`example <https://github.com/yzhao062/pyod/blob/development/examples/kde_example.py>`_)
     - 2007
     - [#Latecki2007Outlier]_
   * - Probabilistic
     - Sampling
     - Rapid distance-based outlier detection via sampling (`example <https://github.com/yzhao062/pyod/blob/development/examples/sampling_example.py>`_)
     - 2013
     - [#Sugiyama2013Rapid]_
   * - Probabilistic
     - GMM
     - Probabilistic Mixture Modeling for Outlier Analysis (`example <https://github.com/yzhao062/pyod/blob/development/examples/gmm_example.py>`_)
     -
     - [#Aggarwal2015Outlier]_ [Ch.2]
   * - Linear Model
     - PCA
     - Principal Component Analysis (sum of weighted projected distances to eigenvector hyperplanes) (`example <https://github.com/yzhao062/pyod/blob/development/examples/pca_example.py>`_)
     - 2003
     - [#Shyu2003A]_
   * - Linear Model
     - KPCA
     - Kernel Principal Component Analysis (`example <https://github.com/yzhao062/pyod/blob/development/examples/kpca_example.py>`_)
     - 2007
     - [#Hoffmann2007Kernel]_
   * - Linear Model
     - MCD
     - Minimum Covariance Determinant (Mahalanobis distances as outlier scores) (`example <https://github.com/yzhao062/pyod/blob/development/examples/mcd_example.py>`_)
     - 1999
     - [#Hardin2004Outlier]_ [#Rousseeuw1999A]_
   * - Linear Model
     - CD
     - Cook's distance for outlier detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/cd_example.py>`_)
     - 1977
     - [#Cook1977Detection]_
   * - Linear Model
     - OCSVM
     - One-Class Support Vector Machines (`example <https://github.com/yzhao062/pyod/blob/development/examples/ocsvm_example.py>`_)
     - 2001
     - [#Scholkopf2001Estimating]_
   * - Linear Model
     - LMDD
     - Deviation-based Outlier Detection (LMDD) (`example <https://github.com/yzhao062/pyod/blob/development/examples/lmdd_example.py>`_)
     - 1996
     - [#Arning1996A]_
   * - Proximity-Based
     - LOF
     - Local Outlier Factor (`example <https://github.com/yzhao062/pyod/blob/development/examples/lof_example.py>`_)
     - 2000
     - [#Breunig2000LOF]_
   * - Proximity-Based
     - COF
     - Connectivity-Based Outlier Factor (`example <https://github.com/yzhao062/pyod/blob/development/examples/cof_example.py>`_)
     - 2002
     - [#Tang2002Enhancing]_
   * - Proximity-Based
     - (Incr.) COF
     - Memory Efficient Connectivity-Based Outlier Factor (slower, reduced storage) (`example <https://github.com/yzhao062/pyod/blob/development/examples/cof_example.py>`_)
     - 2002
     - [#Tang2002Enhancing]_
   * - Proximity-Based
     - CBLOF
     - Clustering-Based Local Outlier Factor (`example <https://github.com/yzhao062/pyod/blob/development/examples/cblof_example.py>`_)
     - 2003
     - [#He2003Discovering]_
   * - Proximity-Based
     - LOCI
     - LOCI: Fast outlier detection via local correlation integral (`example <https://github.com/yzhao062/pyod/blob/development/examples/loci_example.py>`_)
     - 2003
     - [#Papadimitriou2003LOCI]_
   * - Proximity-Based
     - HBOS
     - Histogram-based Outlier Score (`example <https://github.com/yzhao062/pyod/blob/development/examples/hbos_example.py>`_)
     - 2012
     - [#Goldstein2012Histogram]_
   * - Proximity-Based
     - HDBSCAN
     - Density-based clustering via hierarchical density estimates (`example <https://github.com/yzhao062/pyod/blob/development/examples/hdbscan_example.py>`_)
     - 2013
     - [#Campello2013Density]_
   * - Proximity-Based
     - kNN
     - k Nearest Neighbors (distance to k-th neighbor as outlier score) (`example <https://github.com/yzhao062/pyod/blob/development/examples/knn_example.py>`_)
     - 2000
     - [#Ramaswamy2000Efficient]_
   * - Proximity-Based
     - AvgKNN
     - Average kNN (average distance to k neighbors as outlier score) (`example <https://github.com/yzhao062/pyod/blob/development/examples/knn_example.py>`_)
     - 2002
     - [#Angiulli2002Fast]_
   * - Proximity-Based
     - MedKNN
     - Median kNN (median distance to k neighbors as outlier score) (`example <https://github.com/yzhao062/pyod/blob/development/examples/knn_example.py>`_)
     - 2002
     - [#Angiulli2002Fast]_
   * - Proximity-Based
     - SOD
     - Subspace Outlier Detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/sod_example.py>`_)
     - 2009
     - [#Kriegel2009Outlier]_
   * - Proximity-Based
     - ROD
     - Rotation-based Outlier Detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/rod_example.py>`_)
     - 2020
     - [#Almardeny2020A]_
   * - Outlier Ensembles
     - IForest
     - Isolation Forest (`example <https://github.com/yzhao062/pyod/blob/development/examples/iforest_example.py>`_)
     - 2008
     - [#Liu2008Isolation]_
   * - Outlier Ensembles
     - INNE
     - Isolation-based Anomaly Detection via Nearest-Neighbor Ensembles (`example <https://github.com/yzhao062/pyod/blob/development/examples/inne_example.py>`_)
     - 2018
     - [#Bandaragoda2018Isolation]_
   * - Outlier Ensembles
     - DIF
     - Deep Isolation Forest for Anomaly Detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/dif_example.py>`_)
     - 2023
     - [#Xu2023Deep]_
   * - Outlier Ensembles
     - FB
     - Feature Bagging (`example <https://github.com/yzhao062/pyod/blob/development/examples/feature_bagging_example.py>`_)
     - 2005
     - [#Lazarevic2005Feature]_
   * - Outlier Ensembles
     - LSCP
     - LSCP: Locally Selective Combination of Parallel Outlier Ensembles (`example <https://github.com/yzhao062/pyod/blob/development/examples/lscp_example.py>`_)
     - 2019
     - [#Zhao2019LSCP]_
   * - Outlier Ensembles
     - XGBOD
     - Extreme Boosting Based Outlier Detection **(Supervised)** (`example <https://github.com/yzhao062/pyod/blob/development/examples/xgbod_example.py>`_)
     - 2018
     - [#Zhao2018XGBOD]_
   * - Outlier Ensembles
     - LODA
     - Lightweight On-line Detector of Anomalies (`example <https://github.com/yzhao062/pyod/blob/development/examples/loda_example.py>`_)
     - 2016
     - [#Pevny2016Loda]_
   * - Outlier Ensembles
     - SUOD
     - SUOD: Accelerating Large-scale Unsupervised Heterogeneous OD **(Acceleration)** (`example <https://github.com/yzhao062/pyod/blob/development/examples/suod_example.py>`_)
     - 2021
     - [#Zhao2021SUOD]_
   * - Neural Networks
     - AutoEncoder
     - Fully connected AutoEncoder (reconstruction error as outlier score) (`example <https://github.com/yzhao062/pyod/blob/development/examples/auto_encoder_example.py>`_)
     -
     - [#Aggarwal2015Outlier]_ [Ch.3]
   * - Neural Networks
     - VAE
     - Variational AutoEncoder (reconstruction error as outlier score) (`example <https://github.com/yzhao062/pyod/blob/development/examples/vae_example.py>`_)
     - 2013
     - [#Kingma2013Auto]_
   * - Neural Networks
     - Beta-VAE
     - Variational AutoEncoder with customized loss (gamma and capacity) (`example <https://github.com/yzhao062/pyod/blob/development/examples/vae_example.py>`_)
     - 2018
     - [#Burgess2018Understanding]_
   * - Neural Networks
     - SO_GAAL
     - Single-Objective Generative Adversarial Active Learning (`example <https://github.com/yzhao062/pyod/blob/development/examples/so_gaal_example.py>`_)
     - 2019
     - [#Liu2019Generative]_
   * - Neural Networks
     - MO_GAAL
     - Multiple-Objective Generative Adversarial Active Learning (`example <https://github.com/yzhao062/pyod/blob/development/examples/mo_gaal_example.py>`_)
     - 2019
     - [#Liu2019Generative]_
   * - Neural Networks
     - DeepSVDD
     - Deep One-Class Classification (`example <https://github.com/yzhao062/pyod/blob/development/examples/deepsvdd_example.py>`_)
     - 2018
     - [#Ruff2018Deep]_
   * - Neural Networks
     - AnoGAN
     - Anomaly Detection with Generative Adversarial Networks
     - 2017
     - [#Schlegl2017Unsupervised]_
   * - Neural Networks
     - ALAD
     - Adversarially learned anomaly detection (`example <https://github.com/yzhao062/pyod/blob/development/examples/alad_example.py>`_)
     - 2018
     - [#Zenati2018Adversarially]_
   * - Neural Networks
     - AE1SVM
     - Autoencoder-based One-class Support Vector Machine (`example <https://github.com/yzhao062/pyod/blob/development/examples/ae1svm_example.py>`_)
     - 2019
     - [#Nguyen2019scalable]_
   * - Neural Networks
     - DevNet
     - Deep Anomaly Detection with Deviation Networks (`example <https://github.com/yzhao062/pyod/blob/development/examples/devnet_example.py>`_)
     - 2019
     - [#Pang2019Deep]_
   * - Graph-based
     - R-Graph
     - Outlier detection by R-graph (`example <https://github.com/yzhao062/pyod/blob/development/examples/rgraph_example.py>`_)
     - 2017
     - [#You2017Provable]_
   * - Graph-based
     - LUNAR
     - LUNAR: Unifying Local OD Methods via Graph Neural Networks (`example <https://github.com/yzhao062/pyod/blob/development/examples/lunar_example.py>`_)
     - 2022
     - [#Goodge2022Lunar]_
   * - Embedding-based
     - EmbeddingOD
     - Multi-modal anomaly detection via foundation model embeddings, text and image (`example <https://github.com/yzhao062/pyod/blob/development/examples/embedding_od_example.py>`_)
     - 2025
     - [#Li2024NLPADBench]_


Ensemble methods (IForest, INNE, DIF, FB, LSCP, LODA, SUOD, XGBOD) are included in the table above. Score combination functions (average, maximization, AOM, MOA, median, majority vote) are in ``pyod.models.combination``. See `API docs <https://pyod.readthedocs.io/en/latest/pyod.models.tabular.html>`_ for details.


**(i-b) Time Series Anomaly Detection** :

All time series detectors use the same ``fit``/``predict``/``decision_function`` API as tabular detectors, with one exception: ``MatrixProfile`` is transductive (train-only; use ``decision_scores_`` and ``labels_`` after ``fit()``, no out-of-sample ``predict``).

**Input format**: numpy array of shape ``(n_timestamps,)`` for univariate or ``(n_timestamps, n_channels)`` for multivariate. Each row is one timestep; columns are channels/features. Pandas DataFrames and lists are auto-converted. **Output**: ``decision_scores_`` of shape ``(n_timestamps,)`` with one anomaly score per timestep.

**Time series detection in 3 lines**:

.. code-block:: python

    from pyod.models.ts_kshape import KShape      # or any TS detector
    clf = KShape(window_size=20)
    clf.fit(X_train)                               # shape (n_timestamps,) or (n_timestamps, n_channels)
    scores = clf.decision_scores_                  # per-timestamp anomaly scores

Algorithm rankings from `TSB-AD benchmark <https://github.com/TheDatumOrg/TSB-AD>`_ [#Liu2024TSB]_ (NeurIPS 2024, 1070 datasets):

.. list-table::
   :widths: 15 18 50 5 12
   :header-rows: 1

   * - Type
     - Abbr
     - Algorithm
     - Year
     - Ref
   * - Windowed Bridge
     - TimeSeriesOD
     - Any PyOD detector on sliding windows (`example <https://github.com/yzhao062/pyod/blob/development/examples/ts_od_example.py>`_)
     - 2026
     -
   * - Subsequence
     - MatrixProfile
     - Matrix Profile via STOMP, transductive (`example <https://github.com/yzhao062/pyod/blob/development/examples/ts_matrix_profile_example.py>`_)
     - 2016
     - [#Yeh2016Matrix]_
   * - Frequency
     - SpectralResidual
     - Spectral Residual: FFT-based saliency (`example <https://github.com/yzhao062/pyod/blob/development/examples/ts_spectral_residual_example.py>`_)
     - 2019
     - [#Ren2019Time]_
   * - Clustering
     - KShape
     - k-Shape clustering (#2 in TSB-AD) (`example <https://github.com/yzhao062/pyod/blob/development/examples/ts_kshape_example.py>`_)
     - 2015
     - [#Paparrizos2015KShape]_
   * - Streaming
     - SAND
     - Streaming with drift adaptation, experimental (`example <https://github.com/yzhao062/pyod/blob/development/examples/ts_sand_example.py>`_)
     - 2021
     - [#Boniol2021SAND]_
   * - Deep Learning
     - LSTMAD
     - LSTM prediction error + Mahalanobis scoring
     - 2015
     - [#Malhotra2015Long]_
   * - Deep Learning
     - AnomalyTransformer
     - Transformer with association discrepancy (experimental)
     - 2022
     - [#Xu2022Anomaly]_


**(i-c) Graph Anomaly Detection** (``pip install pyod[graph]``):

All graph detectors are **transductive** in v1: use ``decision_scores_`` and ``labels_`` after ``fit()``. No out-of-sample ``predict``. Input: PyG ``Data`` object with ``x`` (node features) and ``edge_index`` (COO edges). SCAN works without features.

**Graph detection in 3 lines** (``pip install pyod[graph]``):

.. code-block:: python

    from pyod.models.pyg_dominant import DOMINANT
    clf = DOMINANT(hidden_dim=64, epochs=100)
    clf.fit(data)                                  # PyG Data object
    scores = clf.decision_scores_                  # per-node anomaly scores

Algorithm rankings from `BOND benchmark <https://arxiv.org/abs/2206.10071>`_ [#Liu2022BOND]_ (NeurIPS 2022, 14 datasets):

.. list-table::
   :widths: 18 18 45 5 14
   :header-rows: 1

   * - Type
     - Abbr
     - Algorithm
     - Year
     - Ref
   * - GCN Autoencoder
     - DOMINANT
     - GCN AE, structure + attribute reconstruction (#1 BOND deep) (`dominant example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_dominant_example.py>`_)
     - 2019
     - [#Ding2019DOMINANT]_
   * - Contrastive
     - CoLA
     - Contrastive self-supervised, local neighbor context (#2 BOND deep) (`cola example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_cola_example.py>`_)
     - 2022
     - [#Liu2022CoLA]_
   * - Contrastive+AE
     - CONAD
     - Contrastive with anomalous-view injection + dual reconstruction (`conad example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_conad_example.py>`_)
     - 2022
     - [#Xu2022CONAD]_
   * - Attention AE
     - AnomalyDAE
     - GAT structure encoder + MLP attribute encoder (`anomalydae example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_anomalydae_example.py>`_)
     - 2020
     - [#Fan2020AnomalyDAE]_
   * - Motif AE
     - GUIDE
     - Dual GCN AE on original + triangle-motif adjacency (`guide example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_guide_example.py>`_)
     - 2021
     - [#Yuan2021GUIDE]_
   * - Matrix Factor.
     - Radar
     - Residual analysis via matrix factorization (`radar example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_radar_example.py>`_)
     - 2017
     - [#Li2017Radar]_
   * - Matrix Factor.
     - ANOMALOUS
     - Joint MF with Laplacian regularization (`anomalous example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_anomalous_example.py>`_)
     - 2018
     - [#Peng2018ANOMALOUS]_
   * - Structural
     - SCAN
     - Structural clustering, no features needed (`scan example <https://github.com/yzhao062/pyod/blob/development/examples/pyg_scan_example.py>`_)
     - 2007
     - [#Xu2007SCAN]_


**(ii) Utility Functions**:

===================  ============================  =====================================================================================================================================================
Type                 Name                          Function
===================  ============================  =====================================================================================================================================================
Data                 generate_data                 Synthesized data generation; normal data from multivariate Gaussian, outliers from uniform distribution
Data                 generate_data_clusters        Synthesized data generation in clusters for more complex patterns
Evaluation           evaluate_print                Print ROC-AUC and Precision @ Rank n for a detector
Evaluation           precision_n_scores            Calculate Precision @ Rank n
Utility              get_label_n                   Turn raw outlier scores into binary labels by assigning 1 to the top n scores
Stat                 wpearsonr                     Calculate the weighted Pearson correlation of two samples
Encoding             resolve_encoder               Resolve an encoder from a string name, BaseEncoder instance, or callable
Encoding             SentenceTransformerEncoder     Encode text via sentence-transformers models (e.g., MiniLM, mpnet)
Encoding             OpenAIEncoder                 Encode text via OpenAI Embeddings API (text-embedding-3-small/large)
Encoding             HuggingFaceEncoder            Encode text or images via HuggingFace transformers (BERT, DINOv2, CLIP)
===================  ============================  =====================================================================================================================================================

----

Quick Start for Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyOD has been well acknowledged by the machine learning community with a few featured posts and tutorials.

**Analytics Vidhya**: `An Awesome Tutorial to Learn Outlier Detection in Python using PyOD Library <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_

**KDnuggets**: `Intuitive Visualization of Outlier Detection Methods <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_, `An Overview of Outlier Detection Methods from PyOD <https://www.kdnuggets.com/2019/06/overview-outlier-detection-methods-pyod.html>`_

**Towards Data Science**: `Anomaly Detection for Dummies <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_

`"examples/knn_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py>`_
demonstrates the basic API of using kNN detector. **It is noted that the API across all other algorithms are consistent/similar**.

More detailed instructions for running examples can be found in `examples directory <https://github.com/yzhao062/pyod/blob/master/examples>`_.


#. Initialize a kNN detector, fit the model, and make the prediction.

   .. code-block:: python


       from pyod.models.knn import KNN   # kNN detector

       # train kNN detector
       clf_name = 'KNN'
       clf = KNN()
       clf.fit(X_train)

       # get the prediction label and outlier scores of the training data
       y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
       y_train_scores = clf.decision_scores_  # raw outlier scores

       # get the prediction on the test data
       y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
       y_test_scores = clf.decision_function(X_test)  # outlier scores

       # it is possible to get the prediction confidence as well
       y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

#. Evaluate the prediction by ROC and Precision @ Rank n (p@n).

   .. code-block:: python

       from pyod.utils.data import evaluate_print
       
       # evaluate and print the results
       print("\nOn Training Data:")
       evaluate_print(clf_name, y_train, y_train_scores)
       print("\nOn Test Data:")
       evaluate_print(clf_name, y_test, y_test_scores)


#. See a sample output & visualization.


   .. code-block:: python


       On Training Data:
       KNN ROC:1.0, precision @ rank n:1.0

       On Test Data:
       KNN ROC:0.9989, precision @ rank n:0.9

   .. code-block:: python


       visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
           y_test_pred, show_figure=True, save_figure=False)

----

Reference
^^^^^^^^^


.. [#Aggarwal2015Outlier] Aggarwal, C.C., 2015. Outlier analysis. In Data mining (pp. 237-263). Springer, Cham.

.. [#Aggarwal2015Theoretical] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.\ *ACM SIGKDD Explorations Newsletter*\ , 17(1), pp.24-47.

.. [#Aggarwal2017Outlier] Aggarwal, C.C. and Sathe, S., 2017. Outlier ensembles: An introduction. Springer.

.. [#Almardeny2020A] Almardeny, Y., Boujnah, N. and Cleary, F., 2020. A Novel Outlier Detection Method for Multivariate Data. *IEEE Transactions on Knowledge and Data Engineering*.

.. [#Angiulli2002Fast] Angiulli, F. and Pizzuti, C., 2002, August. Fast outlier detection in high dimensional spaces. In *European Conference on Principles of Data Mining and Knowledge Discovery* pp. 15-27.

.. [#Arning1996A] Arning, A., Agrawal, R. and Raghavan, P., 1996, August. A Linear Method for Deviation Detection in Large Databases. In *KDD* (Vol. 1141, No. 50, pp. 972-981).

.. [#Bandaragoda2018Isolation] Bandaragoda, T. R., Ting, K. M., Albrecht, D., Liu, F. T., Zhu, Y., and Wells, J. R., 2018, Isolation-based anomaly detection using nearest-neighbor ensembles. *Computational Intelligence*\ , 34(4), pp. 968-998.

.. [#Breunig2000LOF] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. *ACM Sigmod Record*\ , 29(2), pp. 93-104.

.. [#Burgess2018Understanding] Burgess, Christopher P., et al. "Understanding disentangling in beta-VAE." arXiv preprint arXiv:1804.03599 (2018).

.. [#Campello2013Density] Campello, R.J.G.B., Moulavi, D. and Sander, J., 2013, April. Density-based clustering based on hierarchical density estimates. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining* (pp. 160-172). Springer.

.. [#Cook1977Detection] Cook, R.D., 1977. Detection of influential observation in linear regression. Technometrics, 19(1), pp.15-18.

.. [#Chen2024PyOD] Chen, S., Qian, Z., Siu, W., Hu, X., Li, J., Li, S., Qin, Y., Yang, T., Xiao, Z., Ye, W. and Zhang, Y., 2024. PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection. arXiv preprint arXiv:2412.12154.

.. [#Fang2001Wrap] Fang, K.T. and Ma, C.X., 2001. Wrap-around L2-discrepancy of random sampling, Latin hypercube and uniform designs. Journal of complexity, 17(4), pp.608-624.

.. [#Goldstein2012Histogram] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In *KI-2012: Poster and Demo Track*\ , pp.59-63.

.. [#Goodge2022Lunar] Goodge, A., Hooi, B., Ng, S.K. and Ng, W.S., 2022, June. Lunar: Unifying local outlier detection methods via graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence.

.. [#Gopalan2019PIDForest] Gopalan, P., Sharan, V. and Wieder, U., 2019. PIDForest: Anomaly Detection via Partial Identification. In Advances in Neural Information Processing Systems, pp. 15783-15793.

.. [#Han2022ADBench] Han, S., Hu, X., Huang, H., Jiang, M. and Zhao, Y., 2022. ADBench: Anomaly Detection Benchmark. arXiv preprint arXiv:2206.09426.

.. [#Hardin2004Outlier] Hardin, J. and Rocke, D.M., 2004. Outlier detection in the multiple cluster setting using the minimum covariance determinant estimator. *Computational Statistics & Data Analysis*\ , 44(4), pp.625-638.

.. [#He2003Discovering] He, Z., Xu, X. and Deng, S., 2003. Discovering cluster-based local outliers. *Pattern Recognition Letters*\ , 24(9-10), pp.1641-1650.

.. [#Hoffmann2007Kernel] Hoffmann, H., 2007. Kernel PCA for novelty detection. Pattern recognition, 40(3), pp.863-874.

.. [#Iglewicz1993How] Iglewicz, B. and Hoaglin, D.C., 1993. How to detect and handle outliers (Vol. 16). Asq Press.

.. [#Janssens2012Stochastic] Janssens, J.H.M., Huszár, F., Postma, E.O. and van den Herik, H.J., 2012. Stochastic outlier selection. Technical report TiCC TR 2012-001, Tilburg University, Tilburg Center for Cognition and Communication, Tilburg, The Netherlands.

.. [#Kingma2013Auto] Kingma, D.P. and Welling, M., 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

.. [#Kriegel2008Angle] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In *KDD '08*\ , pp. 444-452. ACM.

.. [#Kriegel2009Outlier] Kriegel, H.P., Kröger, P., Schubert, E. and Zimek, A., 2009, April. Outlier detection in axis-parallel subspaces of high dimensional data. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining*\ , pp. 831-838. Springer, Berlin, Heidelberg.

.. [#Latecki2007Outlier] Latecki, L.J., Lazarevic, A. and Pokrajac, D., 2007, July. Outlier detection with kernel density functions. In International Workshop on Machine Learning and Data Mining in Pattern Recognition (pp. 61-75). Springer, Berlin, Heidelberg.

.. [#Lazarevic2005Feature] Lazarevic, A. and Kumar, V., 2005, August. Feature bagging for outlier detection. In *KDD '05*. 2005.

.. [#Li2024NLPADBench] Li, Y., Li, J., Xiao, Z., Yang, T., Nian, Y., Hu, X. and Zhao, Y., 2025. NLP-ADBench: NLP Anomaly Detection Benchmark. In *Findings of the Association for Computational Linguistics: EMNLP 2025*.

.. [#Li2019MADGAN] Li, D., Chen, D., Jin, B., Shi, L., Goh, J. and Ng, S.K., 2019, September. MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. In *International Conference on Artificial Neural Networks* (pp. 703-716). Springer, Cham.

.. [#Li2020COPOD] Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X. COPOD: Copula-Based Outlier Detection. *IEEE International Conference on Data Mining (ICDM)*, 2020.

.. [#Li2021ECOD] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, H. G. ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions. *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, 2022.

.. [#Liu2008Isolation] Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *International Conference on Data Mining*\ , pp. 413-422. IEEE.

.. [#Liu2019Generative] Liu, Y., Li, Z., Zhou, C., Jiang, Y., Sun, J., Wang, M. and He, X., 2019. Generative adversarial active learning for unsupervised outlier detection. *IEEE Transactions on Knowledge and Data Engineering*.

.. [#Nguyen2019scalable] Nguyen, M.N. and Vien, N.A., 2019. Scalable and interpretable one-class svms with deep learning and random fourier features. In *Machine Learning and Knowledge Discovery in Databases: European Conference*, ECML PKDD, 2018.

.. [#Pang2019Deep] Pang, Guansong, Chunhua Shen, and Anton Van Den Hengel. "Deep anomaly detection with deviation networks." In *KDD*, pp. 353-362. 2019.

.. [#Papadimitriou2003LOCI] Papadimitriou, S., Kitagawa, H., Gibbons, P.B. and Faloutsos, C., 2003, March. LOCI: Fast outlier detection using the local correlation integral. In *ICDE '03*, pp. 315-326. IEEE.

.. [#Pevny2016Loda] Pevný, T., 2016. Loda: Lightweight on-line detector of anomalies. *Machine Learning*, 102(2), pp.275-304.

.. [#Perini2020Quantifying] Perini, L., Vercruyssen, V., Davis, J. Quantifying the confidence of anomaly detectors in their example-wise predictions. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD)*, 2020.

.. [#Perini2023Rejection] Perini, L., Davis, J. Unsupervised anomaly detection with rejection. In *Proceedings of the Thirty-Seven Conference on Neural Information Processing Systems (NeurIPS)*, 2023.

.. [#Ramaswamy2000Efficient] Ramaswamy, S., Rastogi, R. and Shim, K., 2000, May. Efficient algorithms for mining outliers from large data sets. *ACM Sigmod Record*\ , 29(2), pp. 427-438.

.. [#Rousseeuw1999A] Rousseeuw, P.J. and Driessen, K.V., 1999. A fast algorithm for the minimum covariance determinant estimator. *Technometrics*\ , 41(3), pp.212-223.

.. [#Ruff2018Deep] Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S.A., Binder, A., Müller, E. and Kloft, M., 2018, July. Deep one-class classification. In *International conference on machine learning* (pp. 4393-4402). PMLR.

.. [#Schlegl2017Unsupervised] Schlegl, T., Seeböck, P., Waldstein, S.M., Schmidt-Erfurth, U. and Langs, G., 2017, June. Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. In International conference on information processing in medical imaging (pp. 146-157). Springer, Cham.

.. [#Scholkopf2001Estimating] Scholkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. and Williamson, R.C., 2001. Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7), pp.1443-1471.

.. [#Shyu2003A] Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. *MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING*.

.. [#Sugiyama2013Rapid] Sugiyama, M. and Borgwardt, K., 2013. Rapid distance-based outlier detection via sampling. Advances in neural information processing systems, 26.

.. [#Tang2002Enhancing] Tang, J., Chen, Z., Fu, A.W.C. and Cheung, D.W., 2002, May. Enhancing effectiveness of outlier detections for low density patterns. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining*, pp. 535-548. Springer, Berlin, Heidelberg.

.. [#Wang2020adVAE] Wang, X., Du, Y., Lin, S., Cui, P., Shen, Y. and Yang, Y., 2019. adVAE: A self-adversarial variational autoencoder with Gaussian anomaly prior knowledge for anomaly detection. *Knowledge-Based Systems*.

.. [#Xu2023Deep] Xu, H., Pang, G., Wang, Y., Wang, Y., 2023. Deep isolation forest for anomaly detection. *IEEE Transactions on Knowledge and Data Engineering*.

.. [#Yang2024ad] Yang, T., Nian, Y., Li, S., Xu, R., Li, Y., Li, J., Xiao, Z., Hu, X., Rossi, R., Ding, K. and Hu, X., 2024. AD-LLM: Benchmarking Large Language Models for Anomaly Detection. arXiv preprint arXiv:2412.11142.

.. [#You2017Provable] You, C., Robinson, D.P. and Vidal, R., 2017. Provable self-representation based outlier detection in a union of subspaces. In Proceedings of the IEEE conference on computer vision and pattern recognition.

.. [#Zenati2018Adversarially] Zenati, H., Romain, M., Foo, C.S., Lecouat, B. and Chandrasekhar, V., 2018, November. Adversarially learned anomaly detection. In 2018 IEEE International conference on data mining (ICDM) (pp. 727-736). IEEE.

.. [#Zhao2018XGBOD] Zhao, Y. and Hryniewicki, M.K. XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning. *IEEE International Joint Conference on Neural Networks*\ , 2018.

.. [#Zhao2019LSCP] Zhao, Y., Nasrullah, Z., Hryniewicki, M.K. and Li, Z., 2019, May. LSCP: Locally selective combination in parallel outlier ensembles. In *Proceedings of the 2019 SIAM International Conference on Data Mining (SDM)*, pp. 585-593. Society for Industrial and Applied Mathematics.

.. [#Zhao2021SUOD] Zhao, Y., Hu, X., Cheng, C., Wang, C., Wan, C., Wang, W., Yang, J., Bai, H., Li, Z., Xiao, C., Wang, Y., Qiao, Z., Sun, J. and Akoglu, L. (2021). SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection. *Conference on Machine Learning and Systems (MLSys)*.

.. [#Boniol2021SAND] Boniol, P., Paparrizos, J., Palpanas, T. and Franklin, M.J., 2021. SAND: Streaming Subsequence Anomaly Detection. *Proceedings of the VLDB Endowment*, 14(10), pp.1717-1729.

.. [#Malhotra2015Long] Malhotra, P., Vig, L., Shroff, G. and Agarwal, P., 2015. Long Short Term Memory Networks for Anomaly Detection in Time Series. In *European Symposium on Artificial Neural Networks (ESANN)*.

.. [#Paparrizos2015KShape] Paparrizos, J. and Gravano, L., 2015. k-Shape: Efficient and Accurate Clustering of Time Series. In *Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data*, pp.1855-1870.

.. [#Ren2019Time] Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T., Yang, M., Tong, J. and Zhang, Q., 2019. Time-Series Anomaly Detection Service at Microsoft. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp.3009-3017.

.. [#Xu2022Anomaly] Xu, J., Wu, H., Wang, J. and Long, M., 2022. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. In *International Conference on Learning Representations (ICLR)*.

.. [#Yeh2016Matrix] Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F., Mueen, A. and Keogh, E., 2016. Matrix Profile I: All Pairs Similarity Joins for Time Series Subsequences. In *2016 IEEE 16th International Conference on Data Mining (ICDM)*, pp.1317-1322.

.. [#Ding2019DOMINANT] Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019. Deep Anomaly Detection on Attributed Networks. In *Proceedings of the 2019 SIAM International Conference on Data Mining*, pp.594-602. SIAM.

.. [#Liu2022CoLA] Liu, Y., Li, Z., Pan, S., Gool, T., Xiang, T. and Gong, B., 2022. Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning. In *Proceedings of the ACM Web Conference 2022*, pp.2137-2147.

.. [#Xu2022CONAD] Xu, Z., Huang, X., Zhao, Y., Dong, Y. and Li, J., 2022. Contrastive Attributed Network Anomaly Detection with Data Augmentation. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining*, pp.444-457. Springer.

.. [#Fan2020AnomalyDAE] Fan, H., Zhang, F. and Li, Z., 2020. AnomalyDAE: Dual Autoencoder for Anomaly Detection on Attributed Networks. In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management*, pp.747-756.

.. [#Yuan2021GUIDE] Yuan, X., Zhou, N., Yu, S., Huang, H., Chen, Z. and Xia, F., 2021. Higher-Order Structure Based Anomaly Detection on Attributed Networks. In *2021 IEEE International Conference on Big Data*, pp.2691-2700. IEEE.

.. [#Li2017Radar] Li, J., Dani, H., Hu, X. and Liu, H., 2017. Radar: Residual Analysis for Anomaly Detection in Attributed Networks. In *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence*, pp.2152-2158.

.. [#Peng2018ANOMALOUS] Peng, Z., Luo, M., Li, J., Liu, H. and Zheng, Q., 2018. ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks. In *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence*, pp.3529-3535.

.. [#Xu2007SCAN] Xu, X., Yuruk, N., Feng, Z. and Schweiger, T.A.J., 2007. SCAN: A Structural Clustering Algorithm for Networks. In *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp.824-833.

.. [#Liu2024TSB] Liu, Q., Boniol, P., Palpanas, T. and Paparrizos, J., 2024. TSB-AD: Towards A Reliable Time-Series Anomaly Detection Benchmark. In *Advances in Neural Information Processing Systems (NeurIPS)*.

.. [#Liu2022BOND] Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K., Sun, L., Li, J., Chen, G.H., Jia, Z. and Yu, P.S., 2022. BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs. In *Advances in Neural Information Processing Systems (NeurIPS)*.
