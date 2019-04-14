API CheatSheet
==============

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector. y is optional for unsupervised methods.
* :func:`pyod.models.base.BaseDetector.fit_predict`: Fit detector first and then predict whether a particular sample is an outlier or not.
* :func:`pyod.models.base.BaseDetector.fit_predict_score`: Fit the detector, predict on samples, and evaluate the model by predefined metrics, e.g., ROC.
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict raw anomaly score of X using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not using the fitted detector.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier using the fitted detector.

See base class definition below:

pyod.models.base module
-----------------------

.. automodule:: pyod.models.base
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

