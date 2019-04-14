Known Issues & Warnings
=======================

This is the central place to track known issues.


Installation
------------

There are some known dependency issues/notes. Refer
`installation <https://pyod.readthedocs.io/en/latest/install.html>`_
for more information.


Neural Networks
---------------

SO_GAAL and MO_GAAL may only work under Python 3.5+.


Differences between PyOD and scikit-learn
-----------------------------------------


Although PyOD is built on top of scikit-learn and inspired by its API design,
some differences should be noted:

- All models in PyOD follow the tradition that the outlying objects come with
  higher scores while the normal objects have lower scores. scikit-learn has
  an inverted design--lower scores stand for outlying objects.
- PyOD uses "0" to represent inliers and "1" to represent outliers. Differently,
  scikit-learn returns "-1" for anomalies/outliers and "1" for inliers.
- Although Isolation Forests, One-class SVM, and Local Outlier Factor are
  implemented in both PyOD and scikit-learn, users are not advised to mix the
  use of them, e.g., calling one model from PyOD and another model from scikit-learn.
  It is recommended to only use one library for consistency
  (for three models, the PyOD implementation is indeed a set of wrapper
  functions of scikit-learn).
- PyOD models may not work with scikit-learn's check_estimator function. Similarly,
  scikit-learn models would not work with PyOD's check_estimator function.