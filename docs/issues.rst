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


Discrepancy and difference between PyOD and scikit-learn
--------------------------------------------------------


Although PyOD is built on top of scikit-learn and inspired by its API design,
some differences should be noted:

- All models in PyOD follow the tradition that the outlying objects come with
  higher scores while the normal objects have lower scores. scikit-learn has
  an inverted design--lower scores stand for outlying objects.
- Although Isolation Forests, One-class SVM, Local Outlier Factor exist in both PyOD
  and scikit-learn, users should not mix them by calling two libraries.
  It is recommended to use only one library for detecting anomalies for consistency
  (the PyOD implementations of these three models are indeed a set of wrapper functions of scikit-learn).
- PyOD models may not work with scikit-learn's check_estimator function.