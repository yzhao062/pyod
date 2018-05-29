API CheatSheet
==================

* :func:`pyod.models.base.BaseDetector.fit`: Fit detector.
* :func:`pyod.models.base.BaseDetector.fit_predict`: Fit detector and predict if a particular sample is an outlier or not.
* :func:`pyod.models.base.BaseDetector.decision_function`: Return raw outlier scores of a sample.
* :func:`pyod.models.base.BaseDetector.predict`: Predict if a particular sample is an outlier or not. The model must be fitted first.
* :func:`pyod.models.base.BaseDetector.predict_proba`: Predict the probability of a sample being outlier. The model must be fitted first.
* :func:`pyod.models.base.BaseDetector.predict_rank`: Predict the outlyingness rank of a sample.
* :func:`pyod.models.base.BaseDetector.evaluate`: Print out the roc and precision @ rank n.

See full API reference :doc:`api`.