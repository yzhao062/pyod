.. pyod documentation master file, created by
   sphinx-quickstart on Sun May 27 10:56:38 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyOD Documentation
================================
.. image:: https://badge.fury.io/py/pyod.svg
    :target: https://badge.fury.io/py/pyod
.. image:: https://travis-ci.org/yzhao062/Pyod.svg?branch=master
    :target: https://travis-ci.org/yzhao062/Pyod
.. image:: https://coveralls.io/repos/github/yzhao062/Pyod/badge.svg?branch=master
    :target: https://coveralls.io/github/yzhao062/Pyod?branch=master
.. image:: https://img.shields.io/github/stars/yzhao062/Pyod.svg
    :alt: GitHub stars
    :target: https://github.com/yzhao062/Pyod/stargazers
.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
    :target: https://pyod.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**Py**\ thon \ **O**\ utlier \ **D**\ etection (PyOD) is a Python-based toolkit to identify outliers in data with both unsupervised and supervised algorithms.
It strives to provide unified APIs across for different anomaly detection algorithms.
The toolkit consists of three major groups of functionalities:

1. Outlier detection algorithms
    * Local Outlier Factor, LOF [1] :class:`pyod.models.lof.LOF`
    * Isolation Forest, iForest [2] :class:`pyod.models.iforest.IForest`
    * One-Class Support Vector Machines [3] :class:`pyod.models.ocsvm.OCSVM`
    * kNN Outlier Detection :class:`pyod.models.knn.KNN`
    * Average KNN Outlier Detection :class:`pyod.models.knn.KNN`
    * Median KNN Outlier Detection :class:`pyod.models.knn.KNN`
    * Broken, to fix: Global-Local Outlier Score From Hierarchies [4]
    * Histogram-based Outlier Score, HBOS [5] :class:`pyod.models.hbos.HBOS`
    * Angle-Based Outlier Setection, ABOD [7] :class:`pyod.models.abod.ABOD`

2. Outlier ensemble frameworks :mod:`pyod.models.combination`.
    * Feature bagging
    * Average of Maximum (AOM) [6] :func:`pyod.models.combination.aom`
    * Maximum of Average (MOA) [6] :func:`pyod.models.combination.moa`
    * Threshold Sum (Thresh) [6]

3. Outlier detection utility functions. See :mod:`pyod.utils`.
    * :func:`pyod.utils.utility.scores_to_lables`: converting raw outlier scores to binary labels
    * :func:`pyod.utils.utility.precision_n_scores`: one of the popular evaluation metrics for outlier mining (precision @ rank n)
    * :func:`pyod.utils.load_data.generate_data`: generate pseudo data for outlier detection experiment

More anomaly detection related resources, e.g., books, papers and videos,
can be found at `anomaly-detection-resources <https://github.com/yzhao062/anomaly-detection-resources>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   example
   api

Check `Github Repository <https://github.com/yzhao062/Pyod>`_
and `PyPI <https://pypi.org/project/pyod/>`_ for more information.

==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
