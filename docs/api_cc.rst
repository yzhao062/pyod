API CheatSheet
==============

The following APIs are applicable for all detector models for easy use.

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector. y is ignored in unsupervised methods.
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict raw anomaly score of X using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier using the fitted detector.


Key Attributes of a fitted model:

* :attr:`pyod.models.base.BaseDetector.decision_scores_`: The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* :attr:`pyod.models.base.BaseDetector.labels_`: The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


**Note** \ : fit_predict() and fit_predict_score() are deprecated in V0.6.9 due
to consistency issue and will be removed in V0.8.0. To get the binary labels
of the training data X_train, one should call clf.fit(X_train) and use
:attr:`pyod.models.base.BaseDetector.labels_`, instead of calling clf.predict(X_train).


See base class definition below:

pyod.models.base module
-----------------------

.. automodule:: pyod.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

