API CheatSheet
==============

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


**Fast training and prediction**: it is possible to train and predict with
a large number of detection models in PyOD by leveraging SUOD framework :cite:`zhao2021suod`.
See  `SUOD Paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_
and  `repository <https://github.com/yzhao062/SUOD>`_.


See base class definition below:

pyod.models.base module
-----------------------

.. automodule:: pyod.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

