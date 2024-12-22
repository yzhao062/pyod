.. pyod documentation master file, created by
   sphinx-quickstart on Sun May 27 10:56:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyOD V2 documentation!
=================================


**Deployment & Documentation & Stats & License**

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

Read Me First
^^^^^^^^^^^^^

Welcome to PyOD, a comprehensive but easy-to-use Python library for detecting anomalies in multivariate data. Whether you are working with a small-scale project or large datasets, PyOD provides a range of algorithms to suit your needs.

**PyOD Version 2 is now available** (`Paper <https://www.arxiv.org/abs/2412.12154>`_) :cite:`a-chen2024pyod`, featuring:

* **Expanded Deep Learning Support**: Integrates 12 modern neural models into a single PyTorch-based framework, bringing the total number of outlier detection methods to 45.
* **Enhanced Performance and Ease of Use**: Models are optimized for efficiency and consistent performance across different datasets.
* **LLM-based Model Selection**: Automated model selection guided by a large language model reduces manual tuning and assists users who may have limited experience with outlier detection.

**Additional Resources**:

* **NLP Anomaly Detection**: `NLP-ADBench <https://github.com/USC-FORTIS/NLP-ADBench>`_ provides both NLP anomaly detection datasets and algorithms :cite:`a-li2024nlp`
* **Time-series Outlier Detection**: `TODS <https://github.com/datamllab/tods>`_
* **Graph Outlier Detection**: `PyGOD <https://pygod.org/>`_
* **Performance Comparison & Datasets**: We have a 45-page, comprehensive `anomaly detection benchmark paper <https://openreview.net/forum?id=foA_SFQ9zo0>`_. The fully `open-sourced ADBench <https://github.com/Minqi824/ADBench>`_ compares 30 anomaly detection algorithms on 57 benchmark datasets.
* **PyOD on Distributed Systems**: You can also run `PyOD on Databricks <https://www.databricks.com/blog/2023/03/13/unsupervised-outlier-detection-databricks.html>`_
* **Learn More**: `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_

**Check out our latest research on LLM-based anomaly detection** :cite:`a-yang2024ad`: `AD-LLM: Benchmarking Large Language Models for Anomaly Detection <https://arxiv.org/abs/2412.11142>`_.

----

About PyOD
^^^^^^^^^^

PyOD, established in 2017, has become a go-to **Python library** for **detecting anomalous/outlying objects** in multivariate data. This exciting yet challenging field is commonly referred to as `Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_ or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.

PyOD includes more than 50 detection algorithms, from classical LOF (SIGMOD 2000) to the cutting-edge ECOD and DIF (TKDE 2022 and 2023). Since 2017, PyOD has been successfully used in numerous academic research projects and commercial products with more than `26 million downloads <https://pepy.tech/project/pyod>`_. It is also well acknowledged by the machine learning community with various dedicated posts/tutorials, including `Analytics Vidhya <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_, `KDnuggets <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_, and `Towards Data Science <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_.

**PyOD is featured for**:

* **Unified, User-Friendly Interface** across various algorithms.
* **Wide Range of Models**, from classic techniques to the latest deep learning methods in **PyTorch**.
* **High Performance & Efficiency**, leveraging `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_ for JIT compilation and parallel processing.
* **Fast Training & Prediction**, achieved through the SUOD framework :cite:`a-zhao2021suod`.

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


----

ADBench Benchmark and Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We just released a 45-page, the most comprehensive `ADBench: Anomaly Detection Benchmark <https://arxiv.org/abs/2206.09426>`_ :cite:`a-han2022adbench`.
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


Implemented Algorithms
======================

PyOD toolkit consists of three major functional groups:

**(i) Individual Detection Algorithms** :

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
===================  ================  ======================================================================================================  =====  ===================================================  ======================================================


**(ii) Outlier Ensembles & Outlier Detector Combination Frameworks**:


===================  ================  =====================================================================================================  =====  ===================================================  ======================================================
Type                 Abbr              Algorithm                                                                                              Year   Ref
===================  ================  =====================================================================================================  =====  ===================================================  ======================================================
Outlier Ensembles                      Feature Bagging                                                                                        2005   :class:`pyod.models.feature_bagging.FeatureBagging`  :cite:`a-lazarevic2005feature`
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                      2019   :class:`pyod.models.lscp.LSCP`                       :cite:`a-zhao2019lscp`
Outlier Ensembles    XGBOD             Extreme Boosting Based Outlier Detection **(Supervised)**                                              2018   :class:`pyod.models.xgbod.XGBOD`                     :cite:`a-zhao2018xgbod`
Outlier Ensembles    LODA              Lightweight On-line Detector of Anomalies                                                              2016   :class:`pyod.models.loda.LODA`                       :cite:`a-pevny2016loda`
Outlier Ensembles    SUOD              SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**         2021   :class:`pyod.models.suod.SUOD`                       :cite:`a-zhao2021suod`
Combination          Average           Simple combination by averaging the scores                                                             2015   :func:`pyod.models.combination.average`              :cite:`a-aggarwal2015theoretical`
Combination          Weighted Average  Simple combination by averaging the scores with detector weights                                       2015   :func:`pyod.models.combination.average`              :cite:`a-aggarwal2015theoretical`
Combination          Maximization      Simple combination by taking the maximum scores                                                        2015   :func:`pyod.models.combination.maximization`         :cite:`a-aggarwal2015theoretical`
Combination          AOM               Average of Maximum                                                                                     2015   :func:`pyod.models.combination.aom`                  :cite:`a-aggarwal2015theoretical`
Combination          MOA               Maximum of Average                                                                                     2015   :func:`pyod.models.combination.moa`                  :cite:`a-aggarwal2015theoretical`
Combination          Median            Simple combination by taking the median of the scores                                                  2015   :func:`pyod.models.combination.median`               :cite:`a-aggarwal2015theoretical`
Combination          majority Vote     Simple combination by taking the majority vote of the labels (weights can be used)                     2015   :func:`pyod.models.combination.majority_vote`        :cite:`a-aggarwal2015theoretical`
===================  ================  =====================================================================================================  =====  ===================================================  ======================================================


**(iii) Utility Functions**:

===================  ==============================================  =====================================================================================================================================================
Type                 Name                                            Function
===================  ==============================================  =====================================================================================================================================================
Data                 :func:`pyod.utils.data.generate_data`           Synthesized data generation; normal data is generated by a multivariate Gaussian and outliers are generated by a uniform distribution
Data                 :func:`pyod.utils.data.generate_data_clusters`  Synthesized data generation in clusters; more complex data patterns can be created with multiple clusters
Stat                 :func:`pyod.utils.stat_models.wpearsonr`        Calculate the weighted Pearson correlation of two samples
Utility              :func:`pyod.utils.utility.get_label_n`          Turn raw outlier scores into binary labels by assign 1 to top n outlier scores
Utility              :func:`pyod.utils.utility.precision_n_scores`   calculate precision @ rank n
===================  ==============================================  =====================================================================================================================================================



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
   model_persistence
   fast_train
   example
   benchmark


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api_cc
   pyod


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

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
