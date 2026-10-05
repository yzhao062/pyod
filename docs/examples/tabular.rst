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

:class:`~pyod.models.nsa.NSA` is one negative-selection-inspired estimator
with distinct generation strategies. Binary matching has three rules, giving
13 configurations across eleven strategies. The public API uses mechanism
names, with no aliases claiming unimplemented published algorithms.

.. list-table:: Implemented generation mechanisms
   :header-rows: 1
   :widths: 16 44 40

   * - Strategy
     - Mechanism and source
     - Explicit adaptation or limit
   * - ``binary``
     - Hamming, contiguous-run, or fixed-position chunk matching;
       :cite:`forrest1994self`
     - Median bit per feature; signed score instead of a binary alarm.
   * - ``fixed``
     - Random fixed-radius spheres censored against self balls;
       :cite:`gonzalez2003anomaly`
     - No adaptive population or auxiliary classifier.
   * - ``variable``
     - Radius equals nearest-self distance minus self radius;
       :cite:`ji2004real`
     - Bounded draws and center suppression; no coverage certificate.
   * - ``coverage``
     - Frozen detector sets tested on fresh uniform non-self probes;
       :cite:`ji2009vdetector`
     - Exact binomial lower bounds with alpha spending replace the original
       normal approximation. Budgets can end fitting without certification.
   * - ``grid``
     - Sparse orthant index, radius-ordered filtering, contained-sphere
       removal; :cite:`zhang2013grid`
     - Exact nearest search replaces neighbor-only search; bounded depth and
       count stopping replace the paper's coverage stopping.
   * - ``hierarchical``
     - Refined self-cluster balls and restricted candidate boxes;
       :cite:`chen2011hierarchical`
     - Deterministic Euclidean covers and level budgets; no fractional metric,
       built-in PCA, or published coverage termination.
   * - ``voronoi``
     - Shared vertices of Voronoi cells clipped to the domain;
       :cite:`zhu2017quick`
     - Two or three features; rectangular domain, finite vertex budget,
       optional detector truncation; no distributed Map/Reduce classifier.
   * - ``deterministic``
     - Regular lattice, censored-column boundary selection, decaying
       repulsion; :cite:`barontini2019deterministic`
     - Two features; 30 movement steps, safe rejection of failures, optional
       thinning. No complete-coverage claim after censoring or thinning.
   * - ``suppressed``
     - Distance-based self partition, boundary identification, reverse
       self detectors that veto negative matches; :cite:`li2010suppression`
     - Clean-self interpretation retains outlier selves as reverse detectors;
       bounded draws replace published statistical termination.

   * - ``dual``
     - Adjusted KMeans self envelopes plus negative detectors inside them;
       :cite:`zheng2013dual`
     - Bounded cluster-count search reports infeasible radius targets;
       enclosing radii are expanded by self_radius for self-ball protection.

   * - ``annealed``
     - Monte Carlo population sizing and overlap-energy minimization by
       Metropolis moves; :cite:`gonzalez2003randomized`
     - Fixed finite cooling schedule and strict self-ball censoring replace
       unspecified initialization and soft-only constraints; no optimality
       or asymptotic convergence claim.

These are **NSA-inspired adaptations** with explicit implementation choices,
not complete reproductions or claimed replications of the papers' results.

Fit on **known normal/self observations only**. An anomalous training
observation is also treated as self and can suppress detection around it.
The ``contamination`` parameter sets PyOD's threshold on training scores;
it does not remove contaminated observations or define the geometric
matching boundary. Larger scores mean stronger evidence of non-self.
All real strategies fit a min-max transformation using training data only.
Their bounded domain expands the transformed training bounds by
``sampling_margin``. Geometry strategies require positive domain widths.

For ordinary real strategies, scores are the maximum signed distance inside
any detector sphere. ``suppressed`` additionally takes the minimum with the
signed distance outside the nearest reverse self ball. Thus positive scores
mean a negative match surviving suppression. The ``dual`` strategy takes
the maximum with the signed distance outside all enclosing self balls, so
it also recognizes points far outside the learned envelopes. Binary scores use signed
matching margins. Scores do not depend on other rows in an inference batch.

``coverage`` exposes ``coverage_reached_``, ``coverage_lower_bound_`` and
``coverage_test_count_``. It freezes the population during each complete
``coverage_samples``-probe test and draws fresh probes, rejecting self points.
The one-sided exact binomial bound in round t spends
``(1 - coverage_confidence) / (t * (t + 1))``. The sum of those error budgets
is bounded by ``1 - coverage_confidence``. This controls erroneous coverage
certificates across adaptive rounds under the bounded uniform sampling model.
It does **not** certify anomaly recall, generalization, or coverage outside
the domain. An incomplete round cannot certify coverage. If either budget
prevents certification, fitting warns and ``coverage_reached_`` is false.

``annealed`` first uses at most 256 Monte Carlo probes (at most a quarter
of its candidate budget, with a one-probe minimum) to estimate non-self
volume and an initial detector target capped by ``n_detectors``. It minimizes
Gaussian pair-overlap energy plus a self-affinity penalty of weight one.
Temperature starts at one, proposal neighborhoods start at twice the detector
radius, and both shrink by 0.9 per sweep. Each sweep makes at most twice the
population size in attempts and stops after population-size acceptances.
The best visited population is returned. These finite, documented choices
are adaptations; volume-based sizing is a heuristic without a confidence
certificate. All probes, initialization draws and optimization attempts
share ``max_candidates``. Diagnostics report the target, initial/best energy,
and accepted moves. Strict self-ball censoring applies to every accepted move.

``max_candidates`` counts random draws for sampled strategies, unique cell
vertices for ``voronoi``, and initial lattice sites for ``deterministic``.
The deterministic movement limit is separate: at most 30 steps per selected
boundary site. ``voronoi`` fails explicitly when vertex enumeration exceeds
its budget; it never returns a falsely complete partial tessellation.
``n_detectors`` caps retained negative detectors. Geometry truncation warns.
``generation_diagnostics_`` exposes the mechanism-specific details.

Finite coverage leaves gaps, especially in high dimensions. Real-valued
scores can decrease far outside the domain. Binary quantization discards
magnitudes. Fitting raises if no valid detector can be generated. No
incremental or labeled-feedback API is provided. Voronoi cell enumeration
solves one linear program per unique self sample, so use modest training
sets in two or three dimensions; no speedup over random generation is claimed.

.. code-block:: python

    from pyod.models.nsa import NSA

    clf = NSA(strategy='coverage', n_detectors=200,
              target_coverage=0.9, max_candidates=20000, random_state=42)
    clf.fit(X_normal_train)
    novelty_scores = clf.decision_function(X_test)
    print(clf.coverage_reached_, clf.coverage_lower_bound_)

The runnable `nsa_example.py <https://github.com/yzhao062/pyod/blob/development/examples/nsa_example.py>`__
retains the original raw-feature real-data demonstration and IForest baseline.
`nsa_researched_example.py <https://github.com/yzhao062/pyod/blob/development/examples/nsa_researched_example.py>`__
compares all ten real strategies and IForest on a fixed two-component PCA
representation, fitted only on normal training data. Two components are
chosen before evaluation to include the two-dimensional geometry methods.
This is a separate representation; its scores must not be compared directly
with the raw-feature example to claim an algorithm improvement.
Both examples print held-out AUROC and average precision without test tuning.

The estimator is registered with ``ADEngine`` for explicit construction;
no benchmark rank or automatic routing rule is assigned to it.

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
