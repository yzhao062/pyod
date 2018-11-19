API CheatSheet
==============

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector.
* :func:`pyod.models.base.BaseDetector.fit_predict`: Fit detector and predict if a particular sample is an outlier or not.
* :func:`pyod.models.base.BaseDetector.fit_predict_evaluate`: Fit, predict and then evaluate with predefined metrics (ROC and precision @ rank n).
* :func:`pyod.models.base.BaseDetector.decision_function`: Predict anomaly score of X of the base classifiers.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not. The model must be fitted first.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier. The model must be fitted first.

See full API reference :doc:`pyod`.

