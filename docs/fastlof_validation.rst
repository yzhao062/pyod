FastLOF validation
==================

Scope and baseline
------------------

The revision was synchronized with development commit
``01300361021f8aeb07ed8e0521ee5cc6256c86c8`` and the affected-module
suite was rerun against its current base and utility code.

This revision corrects novelty reachability, refitting, candidate pruning,
premature stopping, metric consistency, and precision. Pruning remains a
heuristic approximation; threshold=0 processes all chunk pairs.
The unchanged baseline is PR #699 commit
``3710aee14547ac49177074d54f181fa76121545b``.

Regression evidence
-------------------

The existing FastLOF and LOF tests passed on the baseline (43 tests).
The new focused numerical regressions produced 10 failures on the baseline
(including subtest failures). The corrected affected-module suite passes
75 tests and 31 subtests on Python 3.11.5. Commands:

.. code-block:: console

   python -m pytest pyod/test/test_fastlof.py pyod/test/test_fastlof_regression.py pyod/test/test_lof.py pyod/test/test_knowledge.py -q

The baseline check ran with ``python check_original.py``. The local runner
imports the unchanged source as pyod.models._fastlof_original, assigns it
to sys.modules["pyod.models.fastlof"], and calls pytest with:

.. code-block:: console

   python -m pytest pyod/test/test_fastlof_regression.py -q --tb=short -k "novelty_matches or refit_replaces or float64_large or weighted_minkowski or scaled_metrics or precomputed or later_diagonal"

This selects
novelty, refit, float64 precision, weighted Minkowski, globally scaled
metrics, precomputed rejection, and the later-diagonal fixture. It excludes
invalid-neighbor kernel calls, which are unsafe in the original code.

Tests compare numerical training and held-out novelty scores against
scikit-learn LocalOutlierFactor, with one chunk and multiple chunks without
pruning. They also exercise both pruning directions, float32 inputs,
duplicates, neighbor clamping, refits with changed shapes, metric aliases,
zero cosine vectors, and catalog construction.

A zero cosine vector has tied distances to all nonzero rows. Valid tie
choices may change its LOF, so that test requires finite zero-row scores
and reference agreement on untied rows.

Quality and runtime
-------------------

Data: bundled pima, cardio, and shuttle. Their Git blob hashes match the
baseline repository. Split: 70% training / 30% held out, stratified with
random_state=42. StandardScaler is fitted only on training data.
FastLOF uses n_neighbors=20, threshold=1.1, algorithm=auto, and seeds
0, 17, and 42. Quality numbers below are means over all three seeds.
Times are medians of complete fit plus first held-out novelty query.
Repeat-query time and sampled process RSS are retained in the JSON output.

Environment: python=3.11.5, platform=Windows-10-10.0.26200-SP0, numpy=1.26.4, scipy=1.17.0, sklearn=1.8.0, numba=0.59.1, threads=4

Pima
~~~~

* original: training AUC/AP 0.635865/0.442239;
  held-out AUC/AP 0.588093/0.432081;
  fit + first query 0.027502 s; sampled process RSS 248.2 MiB.
* fixed: training AUC/AP 0.646101/0.448317;
  held-out AUC/AP 0.600247/0.428664;
  fit + first query 0.022531 s; sampled process RSS 248.4 MiB.
* exact_auto: training AUC/AP 0.636348/0.439503;
  held-out AUC/AP 0.587078/0.418167;
  fit + first query 0.036500 s; sampled process RSS 248.4 MiB.
* exact_brute: training AUC/AP 0.636348/0.439503;
  held-out AUC/AP 0.587078/0.418167;
  fit + first query 0.017016 s; sampled process RSS 248.2 MiB.

Cardio
~~~~~~

* original: training AUC/AP 0.650601/0.188164;
  held-out AUC/AP 0.569176/0.164818;
  fit + first query 0.021034 s; sampled process RSS 250.1 MiB.
* fixed: training AUC/AP 0.635064/0.181317;
  held-out AUC/AP 0.628551/0.180593;
  fit + first query 0.020925 s; sampled process RSS 250.5 MiB.
* exact_auto: training AUC/AP 0.590660/0.170696;
  held-out AUC/AP 0.569151/0.160350;
  fit + first query 0.026361 s; sampled process RSS 250.6 MiB.
* exact_brute: training AUC/AP 0.590660/0.170696;
  held-out AUC/AP 0.569151/0.160350;
  fit + first query 0.018256 s; sampled process RSS 250.1 MiB.

Shuttle
~~~~~~~

* original: training AUC/AP 0.515567/0.116457;
  held-out AUC/AP 0.510493/0.129332;
  fit + first query 3.352240 s; sampled process RSS 294.0 MiB.
* fixed: training AUC/AP 0.514497/0.112984;
  held-out AUC/AP 0.543878/0.151153;
  fit + first query 3.164129 s; sampled process RSS 327.6 MiB.
* exact_auto: training AUC/AP 0.523993/0.117280;
  held-out AUC/AP 0.542033/0.138427;
  fit + first query 5.119506 s; sampled process RSS 276.0 MiB.
* exact_brute: training AUC/AP 0.523983/0.117291;
  held-out AUC/AP 0.541729/0.138415;
  fit + first query 2.373129 s; sampled process RSS 276.7 MiB.

Interpretation and limits
-------------------------

Held-out AUC improves on all three datasets. Training AUC and average
precision do not improve uniformly; cardio training AUC and shuttle
training average precision decline. The repairs restore valid candidate
processing and consistent distances. They are correctness changes, and
the pruning heuristic can still change ranking quality by dataset and seed.
These declines are disclosed rather than offset by tuning the test data.

Corrected shuttle training is slower than the original implementation.
Compare end-to-end times with both exact auto and exact brute: these
measurements do not substantiate a universal or order-of-magnitude speedup.
This experiment covers up to 49,097 input rows, not the larger datasets
claimed in the original PR description.

Cold first-use fit/query times (including compilation, with a fresh Numba
cache directory) were original: 6.407 s, fixed: 6.466 s.
Main timings follow warm-up. Numba, OpenMP, and BLAS are configured for
four threads. Timings are machine-specific; no tests ran concurrently
with the reported final benchmark.

RSS is sampled every 5 ms in one shared benchmark process. It includes
imports, JIT code, and retained allocator memory, and can miss short peaks.
It is a process-level observation, not an isolated detector memory bound.

Reproduction
------------

Save the baseline fastlof.py from the commit above to a local Python file.
Then run from the repository root, with NUMBA_NUM_THREADS, OMP_NUM_THREADS,
and OPENBLAS_NUM_THREADS set to 4:

.. code-block:: console

   python examples/fastlof_benchmark.py --baseline path/to/original_fastlof.py --output fastlof_benchmark.json

psutil is optional and enables RSS sampling. Use a fresh NUMBA_CACHE_DIR
to reproduce first-use compilation measurements. Without --baseline, the
script measures corrected FastLOF and both exact reference configurations.

Contribution disclosure
-----------------------

Codex assisted with implementation, regression tests, and this measured
report. Codex read the full model diff. Human review of the full diff is
not recorded. The affected-module tests and benchmark above actually ran.
The complete CI matrix and a Sphinx documentation build were not run
locally. Sphinx is unavailable in the local test environment.
