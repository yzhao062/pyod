.. pyod documentation master file, created by
   sphinx-quickstart on Sun May 27 10:56:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyOD documentation!
==============================

**Deployment & Documentation & Stats**

.. image:: https://badge.fury.io/py/pyod.svg
   :target: https://badge.fury.io/py/pyod
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://pyod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/yzhao062/pyod/master
   :alt: Binder


.. image:: https://img.shields.io/github/stars/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/Pyod/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/Pyod/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/pyod
   :target: https://pepy.tech/project/pyod
   :alt: Downloads


.. image:: https://pepy.tech/badge/pyod/month
   :target: https://pepy.tech/project/pyod
   :alt: Downloads

-----

**Build Status & Code Coverage & Maintainability**


.. image:: https://ci.appveyor.com/api/projects/status/1kupdy87etks5n3r/branch/master?svg=true
   :target: https://ci.appveyor.com/project/yzhao062/pyod/branch/master
   :alt: Build status


.. image:: https://travis-ci.org/yzhao062/pyod.svg?branch=master
   :target: https://travis-ci.org/yzhao062/pyod
   :alt: Build Status


.. image:: https://coveralls.io/repos/github/yzhao062/pyod/badge.svg
   :target: https://coveralls.io/github/yzhao062/pyod
   :alt: Coverage Status


.. image:: https://circleci.com/gh/yzhao062/pyod.svg?style=svg
    :target: https://circleci.com/gh/yzhao062/pyod


.. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
   :target: https://codeclimate.com/github/yzhao062/Pyod/maintainability
   :alt: Maintainability


----

PyOD is a comprehensive and scalable **Python toolkit** for **detecting outlying objects** in
multivariate data. This exciting yet challenging field is commonly referred as
`Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.
Since 2017, PyOD :cite:`a-zhao2019pyod` has been successfully used in various
academic researches and commercial products :cite:`a-zhao2018xgbod,a-zhao2018dcso,a-zhao2019lscp`.
PyOD is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced models**\ , including **Neural Networks/Deep Learning** and **Outlier Ensembles**.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Compatible with both Python 2 & 3** (scikit-learn compatible as well).

**Important Notes**\ :
PyOD contains neural network based models, e.g., AutoEncoders, which are
implemented in Keras. However, PyOD would **NOT** install **Keras** and/or
**TensorFlow** automatically. This reduces the risk of damaging your local copies.
If you want to use neural net based models, you should install Keras and back-end libraries like TensorFlow manually.
An instruction is provided: `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_.
Similarly, some models, e.g., XGBOD, depend on **xgboost**, which would **NOT** be installed by default.

**Key Links and Resources**\ :

* `View the latest codes on Github <https://github.com/yzhao062/pyod>`_
* `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyod/master>`_
* `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_


----


Quick Introduction
==================

PyOD toolkit consists of three major groups of functionalities:

**(i) Individual Detection Algorithms** :

1. Linear Models for Outlier Detection:

===================  ================  ======================================================================================================  =====  ===================================================  ======================================================
Type                 Abbr              Algorithm                                                                                               Year   Class                                                Ref
===================  ================  ======================================================================================================  =====  ===================================================  ======================================================
Linear Model         PCA               Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)   2003   :class:`pyod.models.pca.PCA`                         :cite:`a-shyu2003novel`
Linear Model         MCD               Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores)                    1999   :class:`pyod.models.mcd.MCD`                         :cite:`a-rousseeuw1999fast,a-hardin2004outlier`
Linear Model         OCSVM             One-Class Support Vector Machines                                                                       2003   :class:`pyod.models.ocsvm.OCSVM`                     :cite:`a-ma2003time`
Proximity-Based      LOF               Local Outlier Factor                                                                                    2000   :class:`pyod.models.lof.LOF`                         :cite:`a-breunig2000lof`
Proximity-Based      CBLOF             Clustering-Based Local Outlier Factor                                                                   2003   :class:`pyod.models.cblof.CBLOF`                     :cite:`a-he2003discovering`:
Proximity-Based      LOCI              LOCI: Fast outlier detection using the local correlation integral                                       2003   :class:`pyod.models.loci.LOCI`                       :cite:`a-papadimitriou2003loci`
Proximity-Based      HBOS              Histogram-based Outlier Score                                                                           2012   :class:`pyod.models.hbos.HBOS`                       :cite:`a-goldstein2012histogram`
Proximity-Based      kNN               k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score                  2000   :class:`pyod.models.knn.KNN`                         :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`
Proximity-Based      AvgKNN            Average kNN (use the average distance to k nearest neighbors as the outlier score)                      2002   :class:`pyod.models.knn.KNN`                         :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`
Proximity-Based      MedKNN            Median kNN (use the median distance to k nearest neighbors as the outlier score)                        2002   :class:`pyod.models.knn.KNN`                         :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`
Probabilistic        ABOD              Angle-Based Outlier Detection                                                                           2008   :class:`pyod.models.abod.ABOD`                       :cite:`a-kriegel2008angle`
Probabilistic        FastABOD          Fast Angle-Based Outlier Detection using approximation                                                  2008   :class:`pyod.models.abod.ABOD`                       :cite:`a-kriegel2008angle`
Probabilistic        SOS               Stochastic Outlier Selection                                                                            2012   :class:`pyod.models.sos.SOS`                         :cite:`a-janssens2012stochastic`
Outlier Ensembles    IForest           Isolation Forest                                                                                        2008   :class:`pyod.models.iforest.IForest`                 :cite:`a-liu2008isolation,a-liu2012isolation`
Outlier Ensembles                      Feature Bagging                                                                                         2005   :class:`pyod.models.feature_bagging.FeatureBagging`  :cite:`a-lazarevic2005feature`
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                       2019   :class:`pyod.models.lscp.LSCP`                       :cite:`a-zhao2019lscp`
Outlier Ensembles    XGBOD             Extreme Boosting Based Outlier Detection **(Supervised)**                                               2018   :class:`pyod.models.xgbod.XGBOD`                     :cite:`a-zhao2018xgbod`
Neural Networks      AutoEncoder       Fully connected AutoEncoder (use reconstruction error as the outlier score)                             2015   :class:`pyod.models.auto_encoder.AutoEncoder`        :cite:`a-aggarwal2015outlier`
Neural Networks      SO_GAAL           Single-Objective Generative Adversarial Active Learning                                                 2019   :class:`pyod.models.so_gaal.SO_GAAL`                 :cite:`a-liu2018generative`
Neural Networks      MO_GAAL           Multiple-Objective Generative Adversarial Active Learning                                               2019   :class:`pyod.models.mo_gaal.MO_GAAL`                 :cite:`a-liu2018generative`
===================  ================  ======================================================================================================  =====  ===================================================  ======================================================


**(ii) Outlier Ensembles & Outlier Detector Combination Frameworks**:


===================  ================  =====================================================================================================  =====  ===================================================  ======================================================
Type                 Abbr              Algorithm                                                                                              Year   Ref
===================  ================  =====================================================================================================  =====  ===================================================  ======================================================
Outlier Ensembles                      Feature Bagging                                                                                        2005   :class:`pyod.models.feature_bagging.FeatureBagging`  :cite:`a-lazarevic2005feature`
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                      2019   :class:`pyod.models.lscp.LSCP`                       :cite:`a-zhao2019lscp`
Combination          Average           Simple combination by averaging the scores                                                             2015   :func:`pyod.models.combination.average`              :cite:`a-aggarwal2015theoretical`
Combination          Weighted Average  Simple combination by averaging the scores with detector weights                                       2015   :func:`pyod.models.combination.average`              :cite:`a-aggarwal2015theoretical`
Combination          Maximization      Simple combination by taking the maximum scores                                                        2015   :func:`pyod.models.combination.maximization`         :cite:`a-aggarwal2015theoretical`
Combination          AOM               Average of Maximum                                                                                     2015   :func:`pyod.models.combination.aom`                  :cite:`a-aggarwal2015theoretical`
Combination          MOA               Maximization of Average                                                                                2015   :func:`pyod.models.combination.moa`                  :cite:`a-aggarwal2015theoretical`
===================  ================  =====================================================================================================  =====  ===================================================  ======================================================


**(iii) Utility Functions**:

===================  =============================================  =====================================================================================================================================================
Type                 Name                                           Function
===================  =============================================  =====================================================================================================================================================
Data                 :func:`pyod.utils.data.generate_data`          Synthesized data generation; normal data is generated by a multivariate Gaussian and outliers are generated by a uniform distribution
Stat                 :func:`pyod.utils.stat_models.wpearsonr`       Calculate the weighted Pearson correlation of two samples
Utility              :func:`pyod.utils.utility.get_label_n`         Turn raw outlier scores into binary labels by assign 1 to top n outlier scores
Utility              :func:`pyod.utils.utility.precision_n_scores`  calculate precision @ rank n
===================  =============================================  =====================================================================================================================================================

**Comparison of all implemented models** are made available below
(`Code <https://github.com/yzhao062/Pyod/blob/master/examples/compare_all_models.py>`_, `Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/Pyod/master>`_):

For Jupyter Notebooks, please navigate to **"/notebooks/Compare All Models.ipynb"**

.. figure:: figs/ALL.png
    :alt: Comparison of all implemented models

Key APIs & Attributes
=====================

The following APIs are applicable for all detector models for easy use.

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector. y is optional for unsupervised methods.
* :func:`pyod.models.base.BaseDetector.fit_predict`: Fit detector first and then predict whether a particular sample is an outlier or not.
* :func:`pyod.models.base.BaseDetector.fit_predict_score`: Fit the detector, predict on samples, and evaluate the model by predefined metrics, e.g., ROC.
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict raw anomaly score of X using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier using the fitted detector.

Key Attributes of a fitted model:

* :attr:`pyod.models.base.BaseDetector.decision_scores_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* :attr:`pyod.models.base.BaseDetector.labels_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started


   install
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
   todo
   about


Quick Links
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: References

.. bibliography:: zreferences.bib
   :cited:
   :labelprefix: A
   :keyprefix: a-