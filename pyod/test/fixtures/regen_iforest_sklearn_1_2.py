# -*- coding: utf-8 -*-
"""Regenerate the sklearn 1.2.x IsolationForest fixture used by
``test_persistence``.

Run inside an environment with ``scikit-learn==1.2.2`` (or any sklearn
that predates the ``missing_go_to_left`` Tree-node field introduced in
1.3). Writes three artifacts next to this script:

* ``iforest_sklearn_1_2_x.joblib`` - a tiny ``IsolationForest`` saved
  via raw ``joblib.dump``. Loading this on any sklearn >= 1.3 raises
  ``ValueError: node array from the pickle has an incompatible dtype``
  unless routed through ``pyod.utils.persistence.compat_load`` (or
  through ``load()`` which falls through automatically).
* ``iforest_sklearn_1_2_x_expected_scores.npy`` - the
  ``decision_function(X)`` array as observed in the 1.2.x save
  environment. Pinned for diagnostic and historical comparison only;
  the matching unit test does NOT assert score equality against this
  array because cross-version IsolationForest predict-side state
  (``_decision_path_lengths``, ``monotonic_cst``, ...) is added/changed
  in sklearn versions beyond 1.3 in ways that are not Tree-node-dtype
  drift, and therefore outside ``compat_load``'s documented scope.
  End-to-end predict round-trip is covered by the synthetic
  ``_make_aged_pickle`` helper in ``test_persistence.py``, which ages a
  model fit on the current sklearn and avoids unrelated cross-version
  attribute drift.
* ``iforest_sklearn_1_2_x_meta.json`` - machine-readable record of the
  package versions, random state, and dataset shape used at save time.

See ``README.md`` in this directory for the rationale and the regen
procedure.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent

# The synthetic input is fixed by seed and shape; the regen procedure
# must reproduce it bit-for-bit so the saved scores match.
_RNG_SEED = 0
_N_SAMPLES = 16
_N_FEATURES = 5


def main() -> None:
    rng = np.random.RandomState(_RNG_SEED)
    X = rng.randn(_N_SAMPLES, _N_FEATURES)

    clf = IsolationForest(n_estimators=1, random_state=0)
    clf.fit(X)

    artifact_path = HERE / "iforest_sklearn_1_2_x.joblib"
    scores_path = HERE / "iforest_sklearn_1_2_x_expected_scores.npy"
    meta_path = HERE / "iforest_sklearn_1_2_x_meta.json"

    joblib.dump(clf, artifact_path)
    scores = clf.decision_function(X).astype(np.float64)
    np.save(scores_path, scores)

    meta = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "rng_seed": _RNG_SEED,
        "n_samples": _N_SAMPLES,
        "n_features": _N_FEATURES,
        "estimator": "sklearn.ensemble.IsolationForest",
        "estimator_params": {
            "n_estimators": 1,
            "random_state": 0,
        },
        "output": "decision_function(X)",
        "output_shape": list(scores.shape),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"wrote {artifact_path.name}")
    print(f"wrote {scores_path.name}")
    print(f"wrote {meta_path.name}")
    print(f"saved scores: {scores}")


if __name__ == "__main__":
    sys.exit(main())
