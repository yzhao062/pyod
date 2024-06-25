API CheatSheet
==============

The full API Reference is available at `PyOD Documentation <https://pyod.readthedocs.io/en/latest/pyod.html>`_. Below is a quick cheatsheet for all detectors:

* :func:`pyod.models.base.BaseDetector.fit`: The parameter y is ignored in unsupervised methods.
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict raw anomaly scores for X using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict`: Determine whether a sample is an outlier or not as binary labels using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Estimate the probability of a sample being an outlier using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_confidence`: Assess the model's confidence on a per-sample basis (applicable in predict and predict_proba) :cite:`a-perini2020quantifying`.


**Key Attributes of a fitted model**:

* :attr:`pyod.models.base.BaseDetector.decision_scores_`: Outlier scores of the training data. Higher scores typically indicate more abnormal behavior. Outliers usually have higher scores.
  Outliers tend to have higher scores.
* :attr:`pyod.models.base.BaseDetector.labels_`: Binary labels of the training data, where 0 indicates inliers and 1 indicates outliers/anomalies.


See base class definition below:

pyod.models.base module
-----------------------

.. automodule:: pyod.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

