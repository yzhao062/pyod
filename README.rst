Python Outlier Detection (PyOD)
===============================


.. image:: https://badge.fury.io/py/pyod.svg
   :target: https://badge.fury.io/py/pyod
   :alt: PyPI version
 

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/yzhao062/pyod/master
   :alt: Binder


.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://pyod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://ci.appveyor.com/api/projects/status/1kupdy87etks5n3r/branch/master?svg=true
   :target: https://ci.appveyor.com/project/yzhao062/pyod/branch/master
   :alt: Build status


.. image:: https://travis-ci.org/yzhao062/pyod.svg?branch=master
   :target: https://travis-ci.org/yzhao062/pyod
   :alt: Build Status


.. image:: https://coveralls.io/repos/github/yzhao062/pyod/badge.svg
   :target: https://coveralls.io/github/yzhao062/pyod
   :alt: Coverage Status


.. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
   :target: https://codeclimate.com/github/yzhao062/Pyod/maintainability
   :alt: Maintainability


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

PyOD is a comprehensive and scalable **Python toolkit** for **detecting outlying objects** in 
multivariate data. This exciting yet challenging field is commonly referred as 
`Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.
Since 2017, PyOD has been successfully used in various academic researches [#Zhao2018DCSO]_ [#Zhao2018XGBOD]_ [#Zhao2019LSCP]_ and commercial products.
PyOD is featured for:


* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced models**\ , including **Neural Networks/Deep Learning** and **Outlier Ensembles**.
* **Optimized performance with JIT and parallelization** when possible, using numba and parallelization.
* **Compatible with both Python 2 & 3** (scikit-learn compatible as well).

**Important Notes**\ :
PyOD contains some neural network based models, e.g., AutoEncoders, which are
implemented in keras. However, PyOD would **NOT** install **keras** and/or **tensorflow** automatically. This
reduces the risk of damaging your local installations. 
So you should install keras and a back-end lib like tensorflow, if you want
to use neural net based models. An instruction is provided `issue19b <https://github.com/yzhao062/Pyod/issues/19>`_.

**Table of Contents**\ :


* `Key Links & Resources <#key-links-and-resources>`_
* `Quick Introduction <#quick-introduction>`_
* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `Quick Start for Outlier Detection <#quick-start-for-outlier-detection>`_
* `Quick Start for Combining Outlier Scores from Various Base Detectors <#quick-start-for-combining-outlier-scores-from-various-base-detectors>`_
* `How to Contribute and Collaborate <#how-to-contribute-and-collaborate>`_
* `Algorithm Benchmark <#algorithm-benchmark>`_


----

Key Links and Resources
^^^^^^^^^^^^^^^^^^^^^^^


* `View the latest codes on Github <https://github.com/yzhao062/pyod>`_
* `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyod/master>`_
* `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_


----

Quick Introduction
^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of three major groups of functionalities: (i) outlier 
detection algorithms; (ii) outlier ensemble frameworks and (iii) outlier 
detection utility functions.

**Individual Detection Algorithms** :


#. Linear Models for Outlier Detection:

   #. **PCA: Principal Component Analysis** (use the sum of
      weighted projected distances to the eigenvector hyperplane) [#Shyu2003A]_
   #. **MCD: Minimum Covariance Determinant** (use the mahalanobis distances 
      as the outlier scores) [#Hardin2004Outlier]_ [#Rousseeuw1999A]_
   #. **One-Class Support Vector Machines** [#Ma2003Time]_

#. Proximity-Based Outlier Detection Models:

   #. **LOF: Local Outlier Factor** [#Breunig2000LOF]_
   #. **CBLOF: Clustering-Based Local Outlier Factor** [#He2003Discovering]_
   #. **HBOS: Histogram-based Outlier Score** [#Goldstein2012Histogram]_
   #. **kNN: k Nearest Neighbors** (use the distance to the kth nearest 
      neighbor as the outlier score) [#Ramaswamy2000Efficient]_
   #. **Average kNN or kNN Sum** (use the average distance to k 
      nearest neighbors as the outlier score or sum all k distances) [#Angiulli2002Fast]_
   #. **Median kNN** Outlier Detection (use the median distance to k nearest 
      neighbors as the outlier score) [#Angiulli2002Fast]_

#. Probabilistic Models for Outlier Detection:

   #. **ABOD: Angle-Based Outlier Detection** [#Kriegel2008Angle]_
   #. **FastABOD: Fast Angle-Based Outlier Detection using approximation** [#Kriegel2008Angle]_
   #. **SOS: Stochastic Outlier Selection** [#Janssens2012Stochastic]_

#. Outlier Ensembles and Combination Frameworks

   #. **Isolation Forest** [#Liu2008Isolation]_
   #. **Feature Bagging** [#Lazarevic2005Feature]_

#. Neural Networks and Deep Learning Models (implemented in Keras)

   #. **AutoEncoder with Fully Connected NN** [#Aggarwal2015Outlier]_ [Chapter 3]

      FAQ regarding AutoEncoder in PyOD and debugging advice:
      `known issues <https://github.com/yzhao062/Pyod/issues/19>`_ 

**Outlier Detector/Scores Combination Frameworks**:

#. **Feature Bagging**\ : build various detectors on random selected features [#Lazarevic2005Feature]_
#. **Average** & **Weighted Average**\ : simply combine scores by averaging [#Aggarwal2015Theoretical]_
#. **Maximization**\ : simply combine scores by taking the maximum across all 
   base detectors [#Aggarwal2015Theoretical]_
#. **Average of Maximum (AOM)** [#Aggarwal2015Theoretical]_
#. **Maximum of Average (MOA)** [#Aggarwal2015Theoretical]_
#. **Threshold Sum (Thresh)** [#Aggarwal2015Theoretical]_

**Comparison of all implemented models** are made available below:
 (\ `Figure <https://raw.githubusercontent.com/yzhao062/Pyod/master/examples/ALL.png>`_\ , 
 `compare_all_models.py <https://github.com/yzhao062/Pyod/blob/master/examples/compare_all_models.py>`_\ ,
 `Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/Pyod/master>`_\ ):

For Jupyter Notebooks, please navigate to **"/notebooks/Compare All Models.ipynb"**


.. image:: https://raw.githubusercontent.com/yzhao062/Pyod/master/examples/ALL.png
   :target: https://raw.githubusercontent.com/yzhao062/Pyod/master/examples/ALL.png
   :alt: Comparision_of_All


----

Installation
^^^^^^^^^^^^

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as PyOD is updated frequently:

.. code-block:: bash

   pip install pyod
   pip install --upgrade pyod # make sure the latest version is installed!

Alternatively, install from github directly (\ **NOT Recommended**\ )

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   python setup.py install

**Required Dependencies**\ :


* Python 2.7, 3.5, 3.6, or 3.7
* nose
* numpy>=1.13
* numba>=0.35
* scipy>=0.19.1
* scikit_learn>=0.19.1

**Optional Dependencies (required for running examples and AutoEncoder)**\ :


* keras (optional, required if calling AutoEncoder, other backend works)
* matplotlib (optional, required for running examples)
* tensorflow (optional, required if calling AutoEncoder, other backend works)

**Known Issue 1**\ : PyOD depends on matplotlib, which would throw errors in conda
virtual environment on mac OS. See reasons and solutions `issue6 <https://github.com/yzhao062/Pyod/issues/6>`_.

**Known Issue 2**\ : PyOD builds on various packages, which most of them you should have
already installed. If you are installing PyOD in a fresh state (virtualenv),
downloading and installing the dependencies, e.g., TensorFlow, may take
**3-5 mins**.

**Known Issue 3**\ : If you are willing to run examples, matplotlib is required.
PyOD does not list it as a required package for eliminating the dependency.
Similarly, Keras and TensorFlow are listed as optional. However, they are
both required if you want to use neural network based models, such as
AutoEncoder. See reasons and solutions `issue19a <https://github.com/yzhao062/Pyod/issues/19>`_

----

API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://pyod.readthedocs.io/en/latest/pyod.html). API cheatsheet for all detectors:


* **fit(X)**\ : Fit detector.
* **fit_predict(X)**\ : Fit detector and predict if a particular sample is an outlier or not.
* **fit_predict_score(X, y)**\ : Fit, predict and then evaluate with predefined metrics (ROC and precision @ rank n).
* **decision_function(X)**\ : Predict anomaly score of X of the base classifiers.
* **predict(X)**\ : Predict if a particular sample is an outlier or not. The model must be fitted first.
* **predict_proba(X)**\ : Predict the probability of a sample being outlier. The model must be fitted first.

Key Attributes of a fitted model:


* **decision_scores**\ : The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* **labels_**\ : The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.

Full package structure can be found below:


* http://pyod.readthedocs.io/en/latest/genindex.html
* http://pyod.readthedocs.io/en/latest/py-modindex.html

----

Quick Start for Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See **examples directory** for more demos. `"examples/knn_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py>`_
demonstrates the basic APIs of PyOD using kNN detector. **It is noted the APIs for other detectors are similar**.

More detailed instruction of running examples can be found `examples. <https://github.com/yzhao062/pyod/blob/master/examples>`_


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

#. Evaluate the prediction by ROC and Precision@rank *n* (p@n):

   .. code-block:: python


       # evaluate and print the results
       print("\nOn Training Data:")
       evaluate_print(clf_name, y_train, y_train_scores)
       print("\nOn Test Data:")
       evaluate_print(clf_name, y_test, y_test_scores)


#. See a sample output & visualization


   .. code-block:: python


       On Training Data:
       KNN ROC:1.0, precision @ rank n:1.0

       On Test Data:
       KNN ROC:0.9989, precision @ rank n:0.9

   .. code-block:: python


       visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
           y_test_pred, show_figure=True, save_figure=False)

Visualization (\ `knn_figure <https://raw.githubusercontent.com/yzhao062/Pyod/master/examples/KNN.png>`_\ ):

.. image:: https://raw.githubusercontent.com/yzhao062/Pyod/master/examples/KNN.png
   :target: https://raw.githubusercontent.com/yzhao062/Pyod/master/examples/KNN.png
   :alt: kNN example figure


----

Quick Start for Combining Outlier Scores from Various Base Detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"examples/comb_example.py" illustrates the APIs for combining multiple base detectors
(\ `comb_example.py <https://github.com/yzhao062/Pyod/blob/master/examples/comb_example.py>`_\ ,
`Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/Pyod/master>`_\ ).

For Jupyter Notebooks, please navigate to **"/notebooks/Model Combination.ipynb"**

Given we have *n* individual outlier detectors, each of them generates an individual score for all samples.
The task is to combine the outputs from these detectors effectively
**Key Step: conducting Z-score normalization on raw scores before the combination.**
Four combination mechanisms are shown in this demo:


#. Average: take the average of all base detectors.
#. maximization : take the maximum score across all detectors as the score.
#. Average of Maximum (AOM): first randomly split n detectors in to p groups. For each group, use the maximum within the group as the group output. Use the average of all group outputs as the final output.
#. Maximum of Average (MOA): similarly to AOM, the same grouping is introduced. However, we use the average of a group as the group output, and use maximum of all group outputs as the final output.
   To better understand the merging techniques, refer to [6].

The walkthrough of the code example is provided:


#. Import models and generate sample data

   .. code-block:: python

       from pyod.models.knn import KNN
       from pyod.models.combination import aom, moa, average, maximization
       from pyod.utils.data import generate_data

       X, y = generate_data(train_only=True)  # load data

#. First initialize 20 kNN outlier detectors with different k (10 to 200), and get the outlier scores:

   .. code-block:: python

       # initialize 20 base detectors for combination
       k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                   150, 160, 170, 180, 190, 200]

       train_scores = np.zeros([X_train.shape[0], n_clf])
       test_scores = np.zeros([X_test.shape[0], n_clf])

       for i in range(n_clf):
           k = k_list[i]

           clf = KNN(n_neighbors=k, method='largest')
           clf.fit(X_train_norm)

           train_scores[:, i] = clf.decision_scores_
           test_scores[:, i] = clf.decision_function(X_test_norm)

#. Then the output codes are standardized into zero mean and unit variance before combination.


   .. code-block:: python

       from pyod.utils.utility import standardizer
       train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

#. Then four different combination algorithms are applied as described above:

   .. code-block:: python

       comb_by_average = average(test_scores_norm)
       comb_by_maximization = maximization(test_scores_norm)
       comb_by_aom = aom(test_scores_norm, 5) # 5 groups
       comb_by_moa = moa(test_scores_norm, 5)) # 5 groups

#. Finally, all four combination methods are evaluated with ROC and Precision @ Rank n:

   .. code-block:: bash

       Combining 20 kNN detectors
       Combination by Average ROC:0.9194, precision @ rank n:0.4531
       Combination by Maximization ROC:0.9198, precision @ rank n:0.4688
       Combination by AOM ROC:0.9257, precision @ rank n:0.4844
       Combination by MOA ROC:0.9263, precision @ rank n:0.4688

----

How to Contribute and Collaborate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project, and we are preparing
a manuscript at `JMLR <http://www.jmlr.org/mloss/>`_ (Track for open-source software).

If you are interested in contributing:


* Please first check Issue lists for "help wanted" tag and comment the one
  you are interested

* Fork the repository and add your improvement/modification/fix

* Create a pull request

To make sure the code has the same style and standard, please refer to models,
such as abod.py, hbos.py, or feature bagging for example.

You are also welcome to share your ideas by opening an issue or dropping me an email
at yuezhao@cs.toronto.edu :)

----

Algorithm Benchmark
^^^^^^^^^^^^^^^^^^^

To provide an overview and quick guidance of the implemented models, benchmark
is supplied.

In total, 17 benchmark data are used for comparision, all datasets could be
downloaded at `ODDS <http://odds.cs.stonybrook.edu/#table1>`_.

For each dataset, it is first split into 60% for training and 40% for testing.
All experiments are repeated 20 times independently with different samplings.
The mean of 20 trials are taken as the final result. Three evaluation metrics
are provided:


* The area under receiver operating characteristic (ROC) curve
* Precision @ rank n (P@N)
* Execution time

Check the latest result `benchmark <https://pyod.readthedocs.io/en/latest/benchmark.html>`_.
You are welcome to replicate this process by running
`benchmark.py <https://github.com/yzhao062/Pyod/blob/master/notebooks/benchmark.py>`_.

----

Reference
^^^^^^^^^


.. [#Aggarwal2015Outlier] Aggarwal, C.C., 2015. Outlier analysis. In Data mining (pp. 237-263). Springer, Cham.

.. [#Aggarwal2015Theoretical] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.\ *ACM SIGKDD Explorations Newsletter*\ , 17(1), pp.24-47.

.. [#Angiulli2002Fast] Angiulli, F. and Pizzuti, C., 2002, August. Fast outlier detection in high dimensional spaces. In *European Conference on Principles of Data Mining and Knowledge Discovery* pp. 15-27.

.. [#Breunig2000LOF] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. *ACM Sigmod Record*\ , 29(2), pp. 93-104.

.. [#Goldstein2012Histogram] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In *KI-2012: Poster and Demo Track*\ , pp.59-63.

.. [#Hardin2004Outlier] Hardin, J. and Rocke, D.M., 2004. Outlier detection in the multiple cluster setting using the minimum covariance determinant estimator. *Computational Statistics & Data Analysis*\ , 44(4), pp.625-638.

.. [#He2003Discovering] He, Z., Xu, X. and Deng, S., 2003. Discovering cluster-based local outliers. *Pattern Recognition Letters*\ , 24(9-10), pp.1641-1650.

.. [#Janssens2012Stochastic] Janssens, J.H.M., Huszár, F., Postma, E.O. and van den Herik, H.J., 2012. Stochastic outlier selection. Technical report TiCC TR 2012-001, Tilburg University, Tilburg Center for Cognition and Communication, Tilburg, The Netherlands.

.. [#Kriegel2008Angle] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In *KDD '08*\ , pp. 444-452. ACM.

.. [#Lazarevic2005Feature] Lazarevic, A. and Kumar, V., 2005, August. Feature bagging for outlier detection. In *KDD '05*. 2005.

.. [#Liu2008Isolation] Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *International Conference on Data Mining*\ , pp. 413-422. IEEE.

.. [#Ma2003Time] Ma, J. and Perkins, S., 2003, July. Time-series novelty detection using one-class support vector machines. In *IJCNN' 03*\ , pp. 1741-1745. IEEE.

.. [#Ramaswamy2000Efficient] Ramaswamy, S., Rastogi, R. and Shim, K., 2000, May. Efficient algorithms for mining outliers from large data sets. *ACM Sigmod Record*\ , 29(2), pp. 427-438).

.. [#Rousseeuw1999A] Rousseeuw, P.J. and Driessen, K.V., 1999. A fast algorithm for the minimum covariance determinant estimator. *Technometrics*\ , 41(3), pp.212-223.

.. [#Shyu2003A] Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. *MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING*.

.. [#Zhao2018DCSO] Zhao, Y. and Hryniewicki, M.K. DCSO: Dynamic Combination of Detector Scores for Outlier Ensembles. *ACM SIGKDD Workshop on Outlier Detection De-constructed (ODD v5.0)*\ , 2018.

.. [#Zhao2018XGBOD] Zhao, Y. and Hryniewicki, M.K. XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning. *IEEE International Joint Conference on Neural Networks*\ , 2018.

.. [#Zhao2019LSCP] Zhao, Y., Hryniewicki, M.K., Nasrullah, Z., and Li, Z. LSCP: Locally Selective Combination of Parallel Outlier Ensembles. *SIAM International Conference on Data Mining (SDM)*. **Currently Under Review**.
