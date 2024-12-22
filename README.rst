Python Outlier Detection (PyOD) V2
==================================

**Deployment & Documentation & Stats & License**

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


Read Me First
^^^^^^^^^^^^^

Welcome to PyOD, a well-developed and easy-to-use Python library for detecting anomalies in multivariate data. Whether you are working with a small-scale project or large datasets, PyOD provides a range of algorithms to fit your needs.

**PyOD Version 2 is now available** (`Paper <https://www.arxiv.org/abs/2412.12154>`_) [#Chen2024PyOD]_, featuring:

* **Expanded Deep Learning Support**: Integrates 12 modern neural models into a single PyTorch-based framework, bringing the total number of outlier detection methods to 45.
* **Enhanced Performance and Ease of Use**: Models are optimized for efficiency and consistent performance across different datasets.
* **LLM-based Model Selection**: Automated model selection guided by a large language model reduces manual tuning and assists users who may have limited experience with outlier detection.

**Additional Resources**:

* **NLP Anomaly Detection**: `NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_ provides both NLP anonaly detection datasets and algorithms
* **Time-series Outlier Detection**: `TODS <https://github.com/datamllab/tods>`_
* **Graph Outlier Detection**: `PyGOD <https://pygod.org/>`_
* **Performance Comparison & Datasets**: Our 45-page `anomaly detection benchmark paper <https://openreview.net/forum?id=foA_SFQ9zo0>`_ and `ADBench <https://github.com/Minqi824/ADBench>`_, comparing 30 algorithms on 57 datasets
* **PyOD on Distributed Systems**: `PyOD on Databricks <https://www.databricks.com/blog/2023/03/13/unsupervised-outlier-detection-databricks.html>`_
* **Learn More**: `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_

**Check out our latest research in 2025 on LLM-based anomaly detection** [#Yang2024ad]_: `AD-LLM: Benchmarking Large Language Models for Anomaly Detection <https://arxiv.org/abs/2412.11142>`_.

----


About PyOD
^^^^^^^^^^

PyOD, established in 2017, has become a go-to **Python library** for **detecting anomalous/outlying objects** in multivariate data. This exciting yet challenging field is commonly referred to as `Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_ or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.

PyOD includes more than 50 detection algorithms, from classical LOF (SIGMOD 2000) to the cutting-edge ECOD and DIF (TKDE 2022 and 2023). Since 2017, PyOD has been successfully used in numerous academic research projects and commercial products with more than `26 million downloads <https://pepy.tech/project/pyod>`_. It is also well acknowledged by the machine learning community with various dedicated posts/tutorials, including `Analytics Vidhya <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_, `KDnuggets <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_, and `Towards Data Science <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_.

**PyOD is featured for**:

* **Unified, User-Friendly Interface** across various algorithms.
* **Wide Range of Models**, from classic techniques to the latest deep learning methods in **PyTorch**.
* **High Performance & Efficiency**, leveraging `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_ for JIT compilation and parallel processing.
* **Fast Training & Prediction**, achieved through the SUOD framework [#Zhao2021SUOD]_.

**Outlier Detection with 5 Lines of Code**:

.. code-block:: python

    # Example: Training an ECOD detector
    from pyod.models.ecod import ECOD
    clf = ECOD()
    clf.fit(X_train)
    y_train_scores = clf.decision_scores_  # Outlier scores for training data
    y_test_scores = clf.decision_function(X_test)  # Outlier scores for test data


**Selecting the Right Algorithm:** Unsure where to start? Consider these robust and interpretable options:

- `ECOD <https://github.com/yzhao062/pyod/blob/master/examples/ecod_example.py>`_: Example of using ECOD for outlier detection
- `Isolation Forest <https://github.com/yzhao062/pyod/blob/master/examples/iforest_example.py>`_: Example of using Isolation Forest for outlier detection

Alternatively, explore `MetaOD <https://github.com/yzhao062/MetaOD>`_ for a data-driven approach.

**Citing PyOD**:

If you use PyOD in a scientific publication, we would appreciate citations to the following paper(s):

`PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection <https://arxiv.org/abs/2412.12154>`_ is available as a preprint. If you use PyOD in a scientific publication, we would appreciate citations to the following paper::

    @article{zhao2024pyod2,
        author  = {Chen, Sihan and Qian, Zhuangzhuang and Siu, Wingchun and Hu, Xingcan and Li, Jiaqi and Li, Shawn and Qin, Yuehan and Yang, Tiankai and Xiao, Zhuo and Ye, Wanghao and Zhang, Yichi and Dong, Yushun and Zhao, Yue},
        title   = {PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection},
        journal = {arXiv preprint arXiv:2412.12154},
        year    = {2024}
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


For a broader perspective on anomaly detection, see our NeurIPS papers `ADBench: Anomaly Detection Benchmark Paper <https://arxiv.org/abs/2206.09426>`_ and `ADGym: Design Choices for Deep Anomaly Detection <https://arxiv.org/abs/2309.15376>`_::

    @article{han2022adbench,
        title={Adbench: Anomaly detection benchmark},
        author={Han, Songqiao and Hu, Xiyang and Huang, Hailiang and Jiang, Minqi and Zhao, Yue},
        journal={Advances in Neural Information Processing Systems},
        volume={35},
        pages={32142--32159},
        year={2022}
    }

    @article{jiang2023adgym,
        title={ADGym: Design Choices for Deep Anomaly Detection},
        author={Jiang, Minqi and Hou, Chaochuan and Zheng, Ao and Han, Songqiao and Huang, Hailiang and Wen, Qingsong and Hu, Xiyang and Zhao, Yue},
        journal={Advances in Neural Information Processing Systems},
        volume={36},
        year={2023}
    }


**Table of Contents**:

* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `ADBench Benchmark and Datasets <#adbench-benchmark-and-datasets>`_
* `Model Save & Load <#model-save--load>`_
* `Fast Train with SUOD <#fast-train-with-suod>`_
* `Thresholding Outlier Scores <#thresholding-outlier-scores>`_
* `Implemented Algorithms <#implemented-algorithms>`_
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

* Python 3.8 or higher
* joblib
* matplotlib
* numpy>=1.19
* numba>=0.51
* scipy>=1.5.1
* scikit_learn>=0.22.0

**Optional Dependencies (see details below)**:

* combo (optional, required for models/combination.py and FeatureBagging)
* pytorch (optional, required for AutoEncoder, and other deep learning models)
* suod (optional, required for running SUOD model)
* xgboost (optional, required for XGBOD)
* pythresh (optional, required for thresholding)

----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

The full API Reference is available at `PyOD Documentation <https://pyod.readthedocs.io/en/latest/pyod.html>`_. Below is a quick cheatsheet for all detectors:

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


ADBench Benchmark and Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We just released a 45-page, the most comprehensive `ADBench: Anomaly Detection Benchmark <https://arxiv.org/abs/2206.09426>`_ [#Han2022ADBench]_.
The fully `open-sourced ADBench <https://github.com/Minqi824/ADBench>`_ compares 30 anomaly detection algorithms on 57 benchmark datasets.

The organization of **ADBench** is provided below:

.. image:: https://github.com/Minqi824/ADBench/blob/main/figs/ADBench.png?raw=true
   :target: https://github.com/Minqi824/ADBench/blob/main/figs/ADBench.png?raw=true
   :alt: benchmark-fig


For a simpler visualization, we make **the comparison of selected models** via
`compare_all_models.py <https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py>`_\.

.. image:: https://github.com/yzhao062/pyod/blob/development/examples/ALL.png?raw=true
   :target: https://github.com/yzhao062/pyod/blob/development/examples/ALL.png?raw=true
   :alt: Comparison_of_All



----

Model Save & Load
^^^^^^^^^^^^^^^^^

PyOD takes a similar approach of sklearn regarding model persistence.
See `model persistence <https://scikit-learn.org/stable/modules/model_persistence.html>`_ for clarification.

In short, we recommend to use joblib or pickle for saving and loading PyOD models.
See `"examples/save_load_model_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/save_load_model_example.py>`_ for an example.
In short, it is simple as below:

.. code-block:: python

    from joblib import dump, load

    # save the model
    dump(clf, 'clf.joblib')
    # load the model
    clf = load('clf.joblib')

It is known that there are challenges in saving neural network models.
Check `#328 <https://github.com/yzhao062/pyod/issues/328#issuecomment-917192704>`_
and `#88 <https://github.com/yzhao062/pyod/issues/88#issuecomment-615343139>`_
for temporary workaround.


----


Fast Train with SUOD
^^^^^^^^^^^^^^^^^^^^

**Fast training and prediction**: it is possible to train and predict with
a large number of detection models in PyOD by leveraging SUOD framework [#Zhao2021SUOD]_.
See  `SUOD Paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_
and  `SUOD example <https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py>`_.


.. code-block:: python

    from pyod.models.suod import SUOD

    # initialized a group of outlier detectors for acceleration
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35),
                     COPOD(), IForest(n_estimators=100),
                     IForest(n_estimators=200)]

    # decide the number of parallel process, and the combination method
    # then clf can be used as any outlier detection model
    clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average',
               verbose=False)

----

Thresholding Outlier Scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more data-based approach can be taken when setting the contamination level. By using a thresholding method, guessing an arbitrary value can be replaced with tested techniques for separating inliers and outliers. Refer to `PyThresh <https://github.com/KulikDM/pythresh>`_ for a more in-depth look at thresholding.

.. code-block:: python

    from pyod.models.knn import KNN
    from pyod.models.thresholds import FILTER

    # Set the outlier detection and thresholding methods
    clf = KNN(contamination=FILTER())


See supported thresholding methods in `thresholding <https://github.com/yzhao062/pyod/blob/master/docs/thresholding.rst>`_.

----



Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of four major functional groups:

**(i) Individual Detection Algorithms** :

===================  ==================  ======================================================================================================  =====  ========================================
Type                 Abbr                Algorithm                                                                                               Year   Ref
===================  ==================  ======================================================================================================  =====  ========================================
Probabilistic        ECOD                Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions                        2022   [#Li2021ECOD]_
Probabilistic        ABOD                Angle-Based Outlier Detection                                                                           2008   [#Kriegel2008Angle]_
Probabilistic        FastABOD            Fast Angle-Based Outlier Detection using approximation                                                  2008   [#Kriegel2008Angle]_
Probabilistic        COPOD               COPOD: Copula-Based Outlier Detection                                                                   2020   [#Li2020COPOD]_
Probabilistic        MAD                 Median Absolute Deviation (MAD)                                                                         1993   [#Iglewicz1993How]_
Probabilistic        SOS                 Stochastic Outlier Selection                                                                            2012   [#Janssens2012Stochastic]_
Probabilistic        QMCD                Quasi-Monte Carlo Discrepancy outlier detection                                                         2001   [#Fang2001Wrap]_
Probabilistic        KDE                 Outlier Detection with Kernel Density Functions                                                         2007   [#Latecki2007Outlier]_
Probabilistic        Sampling            Rapid distance-based outlier detection via sampling                                                     2013   [#Sugiyama2013Rapid]_
Probabilistic        GMM                 Probabilistic Mixture Modeling for Outlier Analysis                                                            [#Aggarwal2015Outlier]_ [Ch.2]
Linear Model         PCA                 Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)   2003   [#Shyu2003A]_
Linear Model         KPCA                Kernel Principal Component Analysis                                                                     2007   [#Hoffmann2007Kernel]_
Linear Model         MCD                 Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores)                    1999   [#Hardin2004Outlier]_ [#Rousseeuw1999A]_
Linear Model         CD                  Use Cook's distance for outlier detection                                                               1977   [#Cook1977Detection]_
Linear Model         OCSVM               One-Class Support Vector Machines                                                                       2001   [#Scholkopf2001Estimating]_
Linear Model         LMDD                Deviation-based Outlier Detection (LMDD)                                                                1996   [#Arning1996A]_
Proximity-Based      LOF                 Local Outlier Factor                                                                                    2000   [#Breunig2000LOF]_
Proximity-Based      COF                 Connectivity-Based Outlier Factor                                                                       2002   [#Tang2002Enhancing]_
Proximity-Based      (Incremental) COF   Memory Efficient Connectivity-Based Outlier Factor (slower but reduce storage complexity)               2002   [#Tang2002Enhancing]_
Proximity-Based      CBLOF               Clustering-Based Local Outlier Factor                                                                   2003   [#He2003Discovering]_
Proximity-Based      LOCI                LOCI: Fast outlier detection using the local correlation integral                                       2003   [#Papadimitriou2003LOCI]_
Proximity-Based      HBOS                Histogram-based Outlier Score                                                                           2012   [#Goldstein2012Histogram]_
Proximity-Based      kNN                 k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score)                 2000   [#Ramaswamy2000Efficient]_
Proximity-Based      AvgKNN              Average kNN (use the average distance to k nearest neighbors as the outlier score)                      2002   [#Angiulli2002Fast]_
Proximity-Based      MedKNN              Median kNN (use the median distance to k nearest neighbors as the outlier score)                        2002   [#Angiulli2002Fast]_
Proximity-Based      SOD                 Subspace Outlier Detection                                                                              2009   [#Kriegel2009Outlier]_
Proximity-Based      ROD                 Rotation-based Outlier Detection                                                                        2020   [#Almardeny2020A]_
Outlier Ensembles    IForest             Isolation Forest                                                                                        2008   [#Liu2008Isolation]_
Outlier Ensembles    INNE                Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles                                      2018   [#Bandaragoda2018Isolation]_
Outlier Ensembles    DIF                 Deep Isolation Forest for Anomaly Detection                                                             2023   [#Xu2023Deep]_
Outlier Ensembles    FB                  Feature Bagging                                                                                         2005   [#Lazarevic2005Feature]_
Outlier Ensembles    LSCP                LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                       2019   [#Zhao2019LSCP]_
Outlier Ensembles    XGBOD               Extreme Boosting Based Outlier Detection **(Supervised)**                                               2018   [#Zhao2018XGBOD]_
Outlier Ensembles    LODA                Lightweight On-line Detector of Anomalies                                                               2016   [#Pevny2016Loda]_
Outlier Ensembles    SUOD                SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**          2021   [#Zhao2021SUOD]_
Neural Networks      AutoEncoder         Fully connected AutoEncoder (use reconstruction error as the outlier score)                                    [#Aggarwal2015Outlier]_ [Ch.3]
Neural Networks      VAE                 Variational AutoEncoder (use reconstruction error as the outlier score)                                 2013   [#Kingma2013Auto]_
Neural Networks      Beta-VAE            Variational AutoEncoder (all customized loss term by varying gamma and capacity)                        2018   [#Burgess2018Understanding]_
Neural Networks      SO_GAAL             Single-Objective Generative Adversarial Active Learning                                                 2019   [#Liu2019Generative]_
Neural Networks      MO_GAAL             Multiple-Objective Generative Adversarial Active Learning                                               2019   [#Liu2019Generative]_
Neural Networks      DeepSVDD            Deep One-Class Classification                                                                           2018   [#Ruff2018Deep]_
Neural Networks      AnoGAN              Anomaly Detection with Generative Adversarial Networks                                                  2017   [#Schlegl2017Unsupervised]_
Neural Networks      ALAD                Adversarially learned anomaly detection                                                                 2018   [#Zenati2018Adversarially]_
Neural Networks      AE1SVM              Autoencoder-based One-class Support Vector Machine                                                      2019   [#Nguyen2019scalable]_
Neural Networks      DevNet              Deep Anomaly Detection with Deviation Networks                                                          2019   [#Pang2019Deep]_
Graph-based          R-Graph             Outlier detection by R-graph                                                                            2017   [#You2017Provable]_
Graph-based          LUNAR               LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks                               2022   [#Goodge2022Lunar]_
===================  ==================  ======================================================================================================  =====  ========================================


**(ii) Outlier Ensembles & Outlier Detector Combination Frameworks**:

===================  ================  =====================================================================================================  =====  ========================================
Type                 Abbr              Algorithm                                                                                              Year   Ref
===================  ================  =====================================================================================================  =====  ========================================
Outlier Ensembles    FB                Feature Bagging                                                                                        2005   [#Lazarevic2005Feature]_
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                      2019   [#Zhao2019LSCP]_
Outlier Ensembles    XGBOD             Extreme Boosting Based Outlier Detection **(Supervised)**                                              2018   [#Zhao2018XGBOD]_
Outlier Ensembles    LODA              Lightweight On-line Detector of Anomalies                                                              2016   [#Pevny2016Loda]_
Outlier Ensembles    SUOD              SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**         2021   [#Zhao2021SUOD]_
Outlier Ensembles    INNE              Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles                                     2018   [#Bandaragoda2018Isolation]_
Combination          Average           Simple combination by averaging the scores                                                             2015   [#Aggarwal2015Theoretical]_
Combination          Weighted Average  Simple combination by averaging the scores with detector weights                                       2015   [#Aggarwal2015Theoretical]_
Combination          Maximization      Simple combination by taking the maximum scores                                                        2015   [#Aggarwal2015Theoretical]_
Combination          AOM               Average of Maximum                                                                                     2015   [#Aggarwal2015Theoretical]_
Combination          MOA               Maximization of Average                                                                                2015   [#Aggarwal2015Theoretical]_
Combination          Median            Simple combination by taking the median of the scores                                                  2015   [#Aggarwal2015Theoretical]_
Combination          majority Vote     Simple combination by taking the majority vote of the labels (weights can be used)                     2015   [#Aggarwal2015Theoretical]_
===================  ================  =====================================================================================================  =====  ========================================


**(iii) Utility Functions**:

===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Type                 Name                    Function                                                                                                                                               Documentation
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Data                 generate_data           Synthesized data generation; normal data is generated by a multivariate Gaussian and outliers are generated by a uniform distribution                  `generate_data <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.data.generate_data>`_
Data                 generate_data_clusters  Synthesized data generation in clusters; more complex data patterns can be created with multiple clusters                                              `generate_data_clusters <https://pyod.readthedocs.io/en/latest/pyod.utils.html#pyod.utils.data.generate_data_clusters>`_
Stat                 wpearsonr               Calculate the weighted Pearson correlation of two samples                                                                                              `wpearsonr <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.stat_models.wpearsonr>`_
Utility              get_label_n             Turn raw outlier scores into binary labels by assign 1 to top n outlier scores                                                                         `get_label_n <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.get_label_n>`_
Utility              precision_n_scores      calculate precision @ rank n                                                                                                                           `precision_n_scores <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.precision_n_scores>`_
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================

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

Visualization (\ `knn_figure <https://raw.githubusercontent.com/yzhao062/pyod/master/examples/KNN.png>`_\ ):

.. image:: https://raw.githubusercontent.com/yzhao062/pyod/master/examples/KNN.png
   :target: https://raw.githubusercontent.com/yzhao062/pyod/master/examples/KNN.png
   :alt: kNN example figure

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
