Layer 1: Time Series Anomaly Detection
========================================

PyOD ships 5 stable time series detectors plus 2 experimental (``SAND``, ``AnomalyTransformer``), all using the same ``fit``/``predict``/``decision_function`` API. The stable five are what ``ADEngine.list_detectors(data_type='time_series')`` returns; the experimental pair is available by class import but not routed by ADEngine yet. Rankings from `TSB-AD benchmark <https://github.com/TheDatumOrg/TSB-AD>`_ :cite:`a-liu2024tsb` (NeurIPS 2024, 1,070 datasets, 40 algorithms).

**Input format**: numpy array of shape ``(n_timestamps,)`` for univariate or ``(n_timestamps, n_channels)`` for multivariate.

**Output**: ``decision_scores_`` of shape ``(n_timestamps,)``.

.. code-block:: python

    from pyod.models.ts_kshape import KShape
    clf = KShape(window_size=20)
    clf.fit(X_train)
    scores = clf.decision_scores_

----

Detectors
---------

.. list-table::
   :widths: 16 52 8 24
   :header-rows: 1

   * - Type
     - Detector
     - Year
     - Ref
   * - Windowed
     - `TimeSeriesOD <https://github.com/yzhao062/pyod/blob/development/examples/ts_od_example.py>`__: any PyOD detector on sliding windows
     - 2026
     -
   * - Subsequence
     - `MatrixProfile <https://github.com/yzhao062/pyod/blob/development/examples/ts_matrix_profile_example.py>`__: STOMP, transductive
     - 2016
     - Yeh et al.
   * - Frequency
     - `SpectralResidual <https://github.com/yzhao062/pyod/blob/development/examples/ts_spectral_residual_example.py>`__: FFT saliency
     - 2019
     - Ren et al.
   * - Clustering
     - `KShape <https://github.com/yzhao062/pyod/blob/development/examples/ts_kshape_example.py>`__: shape-based, #2 TSB-AD overall
     - 2015
     - Paparrizos et al.
   * - Streaming
     - `SAND <https://github.com/yzhao062/pyod/blob/development/examples/ts_sand_example.py>`__: drift adaptation
     - 2021
     - Boniol et al.
   * - Deep Learning
     - LSTMAD: LSTM prediction + Mahalanobis
     - 2015
     - Malhotra et al.
   * - Deep Learning
     - AnomalyTransformer: attention discrepancy
     - 2022
     - Xu et al.

----

Transductive vs Inductive
--------------------------

Most time series detectors support both ``fit()`` and ``decision_function(X_test)``. One exception:

* ``MatrixProfile`` is **transductive**: use ``decision_scores_`` after ``fit()``, no out-of-sample ``predict()``.
