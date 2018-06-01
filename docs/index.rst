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
Unlike existing libraries, PyOD provides:

- **Unified and consistent APIs** across various anomaly detection algorithms for easy use.
- **Compatibility with Python 2 and 3**. All implemented algorithms are **scikit-learn compatible** as well.
- Additional functionalities, e.g., **Detector Combination Frameworks** for ensemble learning.
- **Detailed API Reference, Examples and Tests** for better readability and reliability.

**The toolbox has been successfully used in various academic researches [4, 8] and commercial products.
It is currently under active development**. However,
the primary purpose of the toolkit is quick exploration. Using it as the final output should be cautious;
fine-tunning may be needed to generate meaningful results.
The authors can be reached out at yuezhao@cs.toronto.edu; comments, questions, pull requests and issues are welcome.
**Enjoy catching outliers!**

**Key Links**:

- `View the latest codes on Github <https://github.com/yzhao062/Pyod>`_
- `Current version on PyPI <https://pypi.org/project/pyod/>`_
- `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_


Important Functions
================================
The toolkit consists of three major groups of functionalities:

1. **Outlier detection algorithms**
    * Local Outlier Factor, LOF [1] :class:`pyod.models.lof.LOF`
    * Isolation Forest, iForest [2] :class:`pyod.models.iforest.IForest`
    * One-Class Support Vector Machines [3] :class:`pyod.models.ocsvm.OCSVM`
    * kNN Outlier Detection :class:`pyod.models.knn.KNN`
    * Average KNN Outlier Detection :class:`pyod.models.knn.KNN`
    * Median KNN Outlier Detection :class:`pyod.models.knn.KNN`
    * Histogram-based Outlier Score, HBOS [5] :class:`pyod.models.hbos.HBOS`
    * Angle-Based Outlier Detection, ABOD [7] :class:`pyod.models.abod.ABOD`
    * Fast Angle-Based Outlier Detection, FastABOD [7] :class:`pyod.models.abod.ABOD`

2. **Outlier ensemble frameworks**, see :mod:`pyod.models.combination`.
    * Feature bagging
    * Average of Maximum (AOM) [6] :func:`pyod.models.combination.aom`
    * Maximum of Average (MOA) [6] :func:`pyod.models.combination.moa`
    * Threshold Sum (Thresh) [6]

3. **Outlier detection utility functions**, see :mod:`pyod.utils`.
    * :func:`pyod.utils.utility.score_to_lable`: converting raw outlier scores to binary labels
    * :func:`pyod.utils.utility.precision_n_scores`: one of the popular evaluation metrics for outlier mining (precision @ rank n)
    * :func:`pyod.utils.load_data.generate_data`: generate pseudo data for outlier detection experiment
    * :func:`pyod.utils.stat_models.wpearsonr`:: weighted pearson is useful in pseudo ground truth generation

Contents
====================

.. toctree::
   :maxdepth: 2

   install
   example
   api_cc
   api

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

==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`