Layer 1: Tabular Anomaly Detection
====================================

PyOD has 44 tabular detectors covering probabilistic, linear, proximity, ensemble, immune-inspired, and deep learning approaches. All use the same ``fit``/``predict``/``decision_function`` API.

.. code-block:: python

    from pyod.models.iforest import IForest
    clf = IForest()
    clf.fit(X_train)
    y_train_scores = clf.decision_scores_
    y_test_scores = clf.decision_function(X_test)

----

Recommended Starting Points
----------------------------

Based on `ADBench <https://github.com/Minqi824/ADBench>`__ (NeurIPS 2022, 57 datasets, 30 algorithms):

* `ECOD <https://github.com/yzhao062/pyod/blob/master/examples/ecod_example.py>`__ -- parameter-free, highly interpretable, top ADBench performance
* `IForest <https://github.com/yzhao062/pyod/blob/master/examples/iforest_example.py>`__ -- tree ensemble, scales to high dimensions
* `KNN <https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py>`__ -- proximity-based, good baseline
* `LOF <https://github.com/yzhao062/pyod/blob/master/examples/lof_example.py>`__ -- density-based, good for local anomalies
* `COPOD <https://github.com/yzhao062/pyod/blob/master/examples/copod_example.py>`__ -- copula-based, fast

----

All Tabular Examples
--------------------

**Probabilistic:** `ECOD <https://github.com/yzhao062/pyod/blob/master/examples/ecod_example.py>`__, `COPOD <https://github.com/yzhao062/pyod/blob/master/examples/copod_example.py>`__, `ABOD <https://github.com/yzhao062/pyod/blob/master/examples/abod_example.py>`__, `MAD <https://github.com/yzhao062/pyod/blob/master/examples/mad_example.py>`__, `SOS <https://github.com/yzhao062/pyod/blob/master/examples/sos_example.py>`__, `QMCD <https://github.com/yzhao062/pyod/blob/master/examples/qmcd_example.py>`__, `KDE <https://github.com/yzhao062/pyod/blob/master/examples/kde_example.py>`__, `Sampling <https://github.com/yzhao062/pyod/blob/master/examples/sampling_example.py>`__, `GMM <https://github.com/yzhao062/pyod/blob/master/examples/gmm_example.py>`__

**Linear Models:** `PCA <https://github.com/yzhao062/pyod/blob/master/examples/pca_example.py>`__, `KPCA <https://github.com/yzhao062/pyod/blob/master/examples/kpca_example.py>`__, `MCD <https://github.com/yzhao062/pyod/blob/master/examples/mcd_example.py>`__, `CD <https://github.com/yzhao062/pyod/blob/master/examples/cd_example.py>`__, `OCSVM <https://github.com/yzhao062/pyod/blob/master/examples/ocsvm_example.py>`__, `LMDD <https://github.com/yzhao062/pyod/blob/master/examples/lmdd_example.py>`__

**Proximity-Based:** `LOF <https://github.com/yzhao062/pyod/blob/master/examples/lof_example.py>`__, `COF <https://github.com/yzhao062/pyod/blob/master/examples/cof_example.py>`__, `CBLOF <https://github.com/yzhao062/pyod/blob/master/examples/cblof_example.py>`__, `LOCI <https://github.com/yzhao062/pyod/blob/master/examples/loci_example.py>`__, `HBOS <https://github.com/yzhao062/pyod/blob/master/examples/hbos_example.py>`__, `HDBSCAN <https://github.com/yzhao062/pyod/blob/master/examples/hdbscan_example.py>`__, `KNN <https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py>`__, `SOD <https://github.com/yzhao062/pyod/blob/master/examples/sod_example.py>`__, `ROD <https://github.com/yzhao062/pyod/blob/master/examples/rod_example.py>`__

**Outlier Ensembles:** `IForest <https://github.com/yzhao062/pyod/blob/master/examples/iforest_example.py>`__, `INNE <https://github.com/yzhao062/pyod/blob/master/examples/inne_example.py>`__, `DIF <https://github.com/yzhao062/pyod/blob/master/examples/dif_example.py>`__, `Feature Bagging <https://github.com/yzhao062/pyod/blob/master/examples/feature_bagging_example.py>`__, `LSCP <https://github.com/yzhao062/pyod/blob/master/examples/lscp_example.py>`__, `XGBOD <https://github.com/yzhao062/pyod/blob/master/examples/xgbod_example.py>`__, `LODA <https://github.com/yzhao062/pyod/blob/master/examples/loda_example.py>`__, `SUOD <https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py>`__

**Neural Networks:** `AutoEncoder <https://github.com/yzhao062/pyod/blob/master/examples/auto_encoder_example.py>`__, `VAE <https://github.com/yzhao062/pyod/blob/master/examples/vae_example.py>`__, `DeepSVDD <https://github.com/yzhao062/pyod/blob/master/examples/deepsvdd_example.py>`__, `SO_GAAL <https://github.com/yzhao062/pyod/blob/master/examples/so_gaal_example.py>`__, `MO_GAAL <https://github.com/yzhao062/pyod/blob/master/examples/mo_gaal_example.py>`__, AnoGAN, `ALAD <https://github.com/yzhao062/pyod/blob/master/examples/alad_example.py>`__, `AE1SVM <https://github.com/yzhao062/pyod/blob/master/examples/ae1svm_example.py>`__, `DevNet <https://github.com/yzhao062/pyod/blob/master/examples/devnet_example.py>`__

----

Negative Selection for Novelty Detection
----------------------------------------

:class:`~pyod.models.nsa.NSA` is a negative-selection-inspired detector with
three strategies: ``binary``, ``fixed`` and ``variable``. They share one
estimator API and have separate detector generation and matching behavior.
The binary strategy supports ``hamming``, ``rcontiguous`` and ``rchunk``
matching after per-feature median quantization. The real-valued strategies
fit a min-max transformation on the training data and sample candidate
detectors from an expanded box around that transformed data.

The foundations are the negative-selection principle of
:cite:`forrest1994self`, real-valued negative selection
:cite:`gonzalez2003anomaly`, and variable-radius detection
:cite:`ji2004real`. These are explicitly **NSA-inspired adaptations**, with
continuous scores for the PyOD API; they are not complete reproductions of
the named methods in those papers.

Fit on **known normal/self observations only**. An anomalous training
observation is also treated as self and can suppress detection around it.
The ``contamination`` parameter sets PyOD's threshold on training scores;
it does not remove contaminated observations or define the geometric
detector-matching boundary. Larger scores mean a stronger detector match.

Finite sampled coverage leaves gaps, especially in high dimensions.
Real-valued scores can decrease for points far outside the sampling box;
this detector does not guarantee increasingly large scores with distance
from the training set. Binary quantization discards magnitudes. Sampling
can produce fewer than ``n_detectors`` detectors, and fitting raises an
error if no valid detector can be generated. There is no ``partial_fit``.

.. code-block:: python

    from pyod.models.nsa import NSA

    clf = NSA(strategy='variable', n_detectors=100, random_state=42)
    clf.fit(X_normal_train)
    novelty_scores = clf.decision_function(X_test)

See the runnable `nsa_example.py <https://github.com/yzhao062/pyod/blob/development/examples/nsa_example.py>`__
for all three strategies. The detector is registered with ``ADEngine`` for
explicit construction; no benchmark rank or automatic routing rule is
assigned to it.

----

Example Walkthrough
-------------------

Full example: `knn_example.py <https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py>`__

1. Import and generate data:

.. code-block:: python

    from pyod.models.knn import KNN
    from pyod.utils.data import generate_data, evaluate_print

    contamination = 0.1
    X_train, X_test, y_train, y_test = generate_data(
        n_train=200, n_test=100, contamination=contamination)

2. Fit and predict:

.. code-block:: python

    clf = KNN()
    clf.fit(X_train)

    y_train_pred = clf.labels_                  # 0: inlier, 1: outlier
    y_train_scores = clf.decision_scores_       # raw scores
    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

3. Evaluate:

.. code-block:: python

    evaluate_print('KNN', y_test, y_test_scores)
    # KNN ROC:0.9989, precision @ rank n:0.9
