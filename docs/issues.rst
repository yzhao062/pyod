Known Issues & Warnings
=======================

This is the central place to track known issues and behavioral notes.


Installation
------------

See :doc:`install` for dependency notes. Heavier modalities are optional: install ``pytorch`` for neural detectors, ``torch_geometric`` for graph detectors, and ``sentence-transformers`` / ``openai`` / ``transformers`` for text and image detection via :class:`~pyod.models.embedding.EmbeddingOD`.


Differences between PyOD and scikit-learn
-----------------------------------------

PyOD is built on top of scikit-learn and inspired by its API design, but some conventions differ:

* **Score direction.** PyOD uses the convention that outlying samples receive higher scores, while normal samples receive lower scores. scikit-learn uses the inverted convention (lower scores mean more anomalous).
* **Label values.** PyOD uses ``0`` for inliers and ``1`` for outliers. scikit-learn returns ``1`` for inliers and ``-1`` for anomalies.
* **Do not mix implementations.** Although Isolation Forest, One-Class SVM, and Local Outlier Factor exist in both libraries, mixing PyOD and scikit-learn instances of the same model in a single pipeline is not recommended. Use one library consistently (PyOD's versions of these three are wrappers around scikit-learn).
* **check_estimator compatibility.** PyOD models may not pass scikit-learn's ``check_estimator``, and scikit-learn models may not pass PyOD's ``check_estimator``. The two validators enforce different contracts.
