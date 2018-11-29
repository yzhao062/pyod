.. pyod documentation master file, created by
   sphinx-quickstart on Sun May 27 10:56:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyOD documentation!
==============================

.. image:: https://badge.fury.io/py/pyod.svg
    :target: https://badge.fury.io/py/pyod
.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/yzhao062/pyod/master
.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
    :target: https://pyod.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://ci.appveyor.com/api/projects/status/1kupdy87etks5n3r/branch/master?svg=true
    :target: https://ci.appveyor.com/project/yzhao062/pyod/branch/master
.. image:: https://travis-ci.org/yzhao062/pyod.svg?branch=master
    :target: https://travis-ci.org/yzhao062/pyod
.. image:: https://coveralls.io/repos/github/yzhao062/pyod/badge.svg
    :target: https://coveralls.io/github/yzhao062/pyod
.. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
    :target: https://codeclimate.com/github/yzhao062/Pyod/maintainability
    :alt: Maintainability
.. image:: https://img.shields.io/github/stars/yzhao062/pyod.svg
    :alt: GitHub stars
    :target: https://github.com/yzhao062/pyod
.. image:: https://img.shields.io/github/forks/yzhao062/pyod.svg
    :alt: GitHub forks
    :target: https://github.com/yzhao062/pyod
.. image:: https://pepy.tech/badge/pyod
    :alt: Downloads
    :target: https://pepy.tech/project/pyod
.. image:: https://pepy.tech/badge/pyod/month
    :alt: Downloads per Month
    :target: https://pepy.tech/project/pyod


**Py**\ thon \ **O**\ utlier \ **D**\ etection (PyOD) is a comprehensive and
scalable **Python toolkit** for **detecting outlying objects** in multivariate data.
This exciting yet challenging field is commonly referred as `Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.

Since 2017, PyOD has been successfully used in various academic researches
:cite:`a-zhao2018xgbod,a-zhao2018dcso` and commercial products. PyOD is featured for:

- **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
- **Advanced models**, including **Neural Networks/Deep Learning** and **Outlier Ensembles**.
- **Optimized performance with JIT and parallelization** when possible, using numba and parallelization.
- **Compatible with both Python 2 & 3** (scikit-learn compatible as well).

**Important Notes**:
PyOD contains some neural network based models, e.g., AutoEncoders, which are
implemented in keras. However, PyOD would **NOT** install **keras** and/or **tensorflow** automatically. This
reduces the risk of damaging your local installations.
So you should install keras and a back-end lib like tensorflow, if you want
It is fairly easy to install and an instruction is provided `here <https://github.com/yzhao062/Pyod/issues/19>`_.


**Key Links**:

- `View the latest codes on Github <https://github.com/yzhao062/Pyod>`_
- `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyod/master>`_
- `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_


Important Functionalities
=========================
PyOD toolkit consists of three major groups of functionalities: (i) outlier
detection algorithms; (ii) outlier ensemble frameworks and (iii) outlier
detection utility functions.

**Individual Detection Algorithms**:

1. Linear Models for Outlier Detection:

  i. **PCA: Principal Component Analysis** (use the sum of
     weighted projected distances to the eigenvector hyperplane as outlier
     scores) :cite:`a-shyu2003novel`: :class:`pyod.models.pca.PCA`
  ii. **MCD: Minimum Covariance Determinant** (use the mahalanobis distances
      as the outlier scores) :cite:`a-rousseeuw1999fast,a-hardin2004outlier`: :class:`pyod.models.mcd.MCD`
  iii. **One-Class Support Vector Machines** :cite:`a-ma2003time`: :class:`pyod.models.ocsvm.OCSVM`

2. Proximity-Based Outlier Detection Models:

  i. **LOF: Local Outlier Factor** :cite:`a-breunig2000lof`: :class:`pyod.models.lof.LOF`
  ii. **CBLOF: Clustering-Based Local Outlier Factor** :cite:`a-he2003discovering`: :class:`pyod.models.cblof.CBLOF`
  iii. **kNN: k Nearest Neighbors** (use the distance to the kth nearest
       neighbor as the outlier score) :cite:`a-ramaswamy2000efficient,a-angiulli2002fast`: :class:`pyod.models.knn.KNN`
  iv. **Average kNN** (use the average distance to k nearest neighbors as
      the outlier score): :class:`pyod.models.knn.KNN`
  v. **Median kNN** (use the median distance to k nearest neighbors
     as the outlier score): :class:`pyod.models.knn.KNN`
  vi. **HBOS: Histogram-based Outlier Score** :cite:`a-goldstein2012histogram`: :class:`pyod.models.hbos.HBOS`

3. Probabilistic Models for Outlier Detection:

  i. **ABOD: Angle-Based Outlier Detection** :cite:`a-kriegel2008angle`: :class:`pyod.models.abod.ABOD`
  ii. **FastABOD: Fast Angle-Based Outlier Detection using approximation** :cite:`a-kriegel2008angle`: :class:`pyod.models.abod.ABOD`
  iii. **SOS: Stochastic Outlier Selection** :cite:`a-janssens2012stochastic`: :class:`pyod.models.sos.SOS`

4. Outlier Ensembles and Combination Frameworks

  i. **Isolation Forest** :cite:`a-liu2008isolation,a-liu2012isolation`: :class:`pyod.models.iforest.IForest`
  ii. **Feature Bagging** :cite:`a-lazarevic2005feature`: :class:`pyod.models.feature_bagging.FeatureBagging`

5. Neural Networks and Deep Learning Models (implemented in Keras):

  i. **AutoEncoder with Fully Connected NN** :cite:`a-aggarwal2015outlier`: :class:`pyod.models.auto_encoder.AutoEncoder`

    FAQ regarding AutoEncoder in PyOD and debugging advices: `known issues <https://github.com/yzhao062/Pyod/issues/19>`_

**Outlier Detector/Scores Combination Frameworks**:

  1. **Feature Bagging**: build various detectors on random selected features :cite:`a-lazarevic2005feature`: :class:`pyod.models.feature_bagging.FeatureBagging`
  2. **Average** & **Weighted Average**: simply combine scores by averaging :cite:`a-aggarwal2015theoretical`: :func:`pyod.models.combination.average`
  3. **Maximization**: simply combine scores by taking the maximum across all
     base detectors :cite:`a-aggarwal2015theoretical`: :func:`pyod.models.combination.maximization`
  4. **Average of Maximum (AOM)** :cite:`a-aggarwal2015theoretical`: :func:`pyod.models.combination.aom`
  5. **Maximum of Average (MOA)** :cite:`a-aggarwal2015theoretical`: :func:`pyod.models.combination.moa`
  6. **Threshold Sum (Thresh)** :cite:`a-aggarwal2015theoretical`

**Comparison of all implemented models** are made available below
(`Code <https://github.com/yzhao062/Pyod/blob/master/examples/compare_all_models.py>`_, `Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/Pyod/master>`_):

For Jupyter Notebooks, please navigate to **"/notebooks/Compare All Models.ipynb"**

.. figure:: figs/ALL.png
    :alt: Comparison of all implemented models

Key APIs & Attributes
=====================

The following APIs are applicable for all detector models for easy use.

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector.
* :func:`pyod.models.base.BaseDetector.fit_predict`: Fit detector and predict if a particular sample is an outlier or not.
* :func:`pyod.models.base.BaseDetector.fit_predict_evaluate`: Fit, predict and then evaluate with predefined metrics (ROC and precision @ rank n).
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict anomaly score of X of the base classifiers.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not. The model must be fitted first.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier. The model must be fitted first.

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