.. pyod documentation master file, created by
   sphinx-quickstart on Sun May 27 10:56:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyOD Documentation
================================
.. image:: https://badge.fury.io/py/pyod.svg
    :target: https://badge.fury.io/py/pyod
.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
    :target: https://pyod.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://travis-ci.org/yzhao062/Pyod.svg?branch=master
    :target: https://travis-ci.org/yzhao062/Pyod
.. image:: https://coveralls.io/repos/github/yzhao062/Pyod/badge.svg?branch=master
    :target: https://coveralls.io/github/yzhao062/Pyod?branch=master&service=github
.. image:: https://img.shields.io/github/stars/yzhao062/Pyod.svg
    :alt: GitHub stars
    :target: https://github.com/yzhao062/Pyod
.. image:: https://img.shields.io/github/forks/yzhao062/Pyod.svg
    :alt: GitHub forks
    :target: https://github.com/yzhao062/Pyod

**Py**\ thon \ **O**\ utlier \ **D**\ etection (PyOD) is a comprehensive Python toolkit
to **identify outlying objects** in data with both unsupervised and supervised approaches.
This exciting yet challenging field is commonly referred as `Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.
The toolkit has been successfully used in various academic researches [4, 8] and commercial products.
Unlike existing libraries, PyOD provides:

- **Unified and consistent APIs** across various anomaly detection algorithms.
- **Compatibility with both Python 2 and 3**. All implemented algorithms are also **scikit-learn compatible**.
- **Advanced functions**, e.g., **Outlier Ensemble Frameworks** to combine multiple detectors.
- **Detailed API Reference, Examples and Tests** for better reliability.

**Key Links**:

- `View the latest codes on Github <https://github.com/yzhao062/Pyod>`_
- `Current version on PyPI <https://pypi.org/project/pyod/>`_
- `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_


Important Functions
================================
PyOD toolkit consists of three major groups of functionalities: (i) outlier
detection algorithms; (ii) outlier ensemble frameworks and (iii) outlier
detection utility functions.

**Individual Detection Algorithms**:

1. Linear Models for Outlier Detection:

  i. **PCA: Principal Component Analysis** (use the sum of
     weighted projected distances to the eigenvector hyperplane as the outlier
     scores) [10]: :class:`pyod.models.pca.PCA`
  ii. **One-Class Support Vector Machines** [3]: :class:`pyod.models.ocsvm.OCSVM`

2. Proximity-Based Outlier Detection Models:

  i. **LOF: Local Outlier Factor** [1]: :class:`pyod.models.lof.LOF`
  ii. **kNN: k Nearest Neighbors** (use the distance to the kth nearest
      neighbor as the outlier score): :class:`pyod.models.knn.KNN`
  iii. **Average kNN** (use the average distance to k nearest neighbors as
       the outlier score): :class:`pyod.models.knn.KNN`
  iv. **Median kNN** (use the median distance to k nearest neighbors
      as the outlier score): :class:`pyod.models.knn.KNN`
  v. **HBOS: Histogram-based Outlier Score** [5]: :class:`pyod.models.hbos.HBOS`

3. Probabilistic Models for Outlier Detection:

  i. **ABOD: Angle-Based Outlier Detection** [7]: :class:`pyod.models.abod.ABOD`
  ii. **FastABOD: Fast Angle-Based Outlier Detection using approximation** [7]: :class:`pyod.models.abod.ABOD`

4. Outlier Ensembles and Combination Frameworks

  i. **Isolation Forest** [2]: :class:`pyod.models.iforest.IForest`
  ii. **Feature Bagging** [9]: :class:`pyod.models.feature_bagging.FeatureBagging`

**Outlier Ensembles** (Outlier Score Combination Frameworks):

  1. **Feature Bagging**: build various detectors on random selected features [9]
  2. **Average** & **Weighted Average**: simply combine scores by averaging [6]: :func:`pyod.models.combination.average`
  3. **Maximization**: simply combine scores by taking the maximum across all
     base detectors [6]: :func:`pyod.models.combination.maximization`
  4. **Average of Maximum (AOM)** [6]: :func:`pyod.models.combination.aom`
  5. **Maximum of Average (MOA)** [6]: :func:`pyod.models.combination.moa`
  6. **Threshold Sum (Thresh)** [6]

**Utility Functions for Outlier Detection**, see :mod:`pyod.utils`.

  1. :func:`pyod.utils.utility.score_to_label`: converting raw outlier scores to binary labels
  2. :func:`pyod.utils.utility.precision_n_scores`: one of the popular evaluation metrics for outlier mining (precision @ rank n)
  3. :func:`pyod.utils.data.generate_data`: generate pseudo data for outlier detection experiment
  4. :func:`pyod.utils.stat_models.wpearsonr`:: weighted pearson is useful in pseudo ground truth generation

Contents
====================

.. toctree::
   :maxdepth: 2

   install
   example
   api_cc
   pyod

Reference
++++++++++++

[1] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. In *ACM SIGMOD Record*, pp. 93-104. ACM.

[2] Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *ICDM '08*, pp. 413-422. IEEE.

[3] Ma, J. and Perkins, S., 2003, July. Time-series novelty detection using one-class support vector machines. In *IJCNN' 03*, pp. 1741-1745. IEEE.

[4] Y. Zhao and M.K. Hryniewicki, "DCSO: Dynamic Combination of Detector Scores for Outlier Ensembles," *ACM SIGKDD Workshop on Outlier Detection De-constructed*, 2018. Submitted, under review.

[5] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In *KI-2012: Poster and Demo Track*, pp.59-63.

[6] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.*ACM SIGKDD Explorations Newsletter*, 17(1), pp.24-47.

[7] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In *KDD '08*, pp. 444-452. ACM.

[8] Y. Zhao and M.K. Hryniewicki, "XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning," *IEEE International Joint Conference on Neural Networks*, 2018.

[9] Lazarevic, A. and Kumar, V., 2005, August. Feature bagging for outlier detection. In *KDD '05*. 2005.

[10] Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. *MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING*.

==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`