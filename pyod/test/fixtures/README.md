# Test fixtures

This directory contains binary artifacts that exercise cross-sklearn-version
behavior in `pyod.utils.persistence`. Everything here is committed to the
repo so the matching tests run hermetically; regeneration is a deliberate
maintenance act, not part of `pytest`.

## `iforest_sklearn_1_2_x.*`

The smallest possible test case for `compat_load`'s sklearn Tree-node-dtype
realignment path.

| File | Bytes | What it is |
| --- | --- | --- |
| `iforest_sklearn_1_2_x.joblib` | ~3 KB | `sklearn.ensemble.IsolationForest(n_estimators=1, random_state=0)` fit on a 16x5 RNG-seed-0 matrix, saved via raw `joblib.dump` from sklearn 1.2.2 |
| `iforest_sklearn_1_2_x_expected_scores.npy` | 256 B | `decision_function(X)` as observed in the 1.2.2 save environment. Pinned for historical reference; the test does NOT assert score equality (see "Test-assertion scope" below) |
| `iforest_sklearn_1_2_x_meta.json` | 388 B | Machine-readable record of the package versions, RNG seed, and dataset shape used at save time |
| `regen_iforest_sklearn_1_2.py` | ~3 KB | The regen script that produces the three files above. Idempotent given the same sklearn env |

The 1.2.2 pickle layout omits the `missing_go_to_left` field that sklearn 1.3
added to the Tree node dtype. Loading this artifact on any sklearn >= 1.3
raises `ValueError: node array from the pickle has an incompatible dtype`.
`pyod.utils.persistence.compat_load` patches joblib's BUILD-opcode dispatch
so the saved Tree state is realigned to the running dtype (zero-fill on
`missing_go_to_left` from `_TREE_NODE_FIELD_DEFAULTS`) before sklearn's own
`Tree.__setstate__` raises. `load()` falls through to `compat_load()`
automatically on the same error.

## Test-assertion scope

`test_compat_load_with_committed_fixture` asserts:

- raw `joblib.load(path)` raises a `ValueError` whose message starts with
  the documented dtype-mismatch prefix (so the fall-through trigger is
  observed end-to-end, not just hypothetical),
- `compat_load(path)` returns an `IsolationForest`,
- the loaded model's `estimators_[0].tree_.__getstate__()['nodes'].dtype`
  equals the running sklearn's `NODE_DTYPE` (the dtype was actually repaired,
  not just silently accepted),
- a single `UserWarning` was emitted recommending re-fit.

The test does NOT assert end-to-end `decision_function` equality against
`iforest_sklearn_1_2_x_expected_scores.npy`. Cross-version sklearn
`IsolationForest` predict-side state (private cached arrays like
`_decision_path_lengths` introduced in 1.8, class attributes like
`monotonic_cst` introduced on tree estimators in 1.4) is added or changed
in versions beyond 1.3 in ways that are not Tree-node-dtype drift, and
therefore outside `compat_load`'s documented scope. End-to-end predict
round-trip is covered by the synthetic `_make_aged_pickle` helper in
`pyod/test/test_persistence.py`: that helper ages a model fit on the
current sklearn, so all attributes other than the surgically stripped
Tree-dtype field are exactly what the running sklearn expects.

The two fixture layers are complementary on purpose:

- **synthetic aged pickle** (in-process): proves the Tree-dtype repair
  mechanic end-to-end, including `predict` round-trip, on every test
  matrix cell. Cheap. Deterministic. Cannot catch real pickle-layout
  changes a hand-written reducer would miss.
- **committed binary fixture** (this directory): proves `compat_load`
  handles a real pre-1.3 sklearn pickle without raising, and that the
  realignment actually fires on the saved Tree dtype. Catches anything
  in the real pickle format the synthetic reducer would not reproduce.

## Regenerating the fixture

The fixture only needs regeneration when sklearn introduces a documented
breaking Tree-dtype change (the kind that would require a new entry in
`_TREE_NODE_FIELD_DEFAULTS` or `_TREE_NODE_FIELD_RENAMES` in
`pyod/utils/persistence.py`). In that case:

1. Create a fresh miniforge env that pins the legacy sklearn:

   ```bash
   mamba create -n pyod-sklearn-12 -c conda-forge \
       python=3.10 "scikit-learn=1.2.2" "numpy=1.24.*" "scipy=1.10.*" joblib -y
   ```

2. Run the regen script with that env's Python:

   ```bash
   "$HOME/miniforge3/envs/pyod-sklearn-12/bin/python" \
       pyod/test/fixtures/regen_iforest_sklearn_1_2.py
   ```

   On Windows, the equivalent PowerShell path is:

   ```powershell
   & "$env:USERPROFILE\miniforge3\envs\pyod-sklearn-12\python.exe" `
       pyod\test\fixtures\regen_iforest_sklearn_1_2.py
   ```

3. Re-run `pytest pyod/test/test_persistence.py` on the current sklearn
   to confirm the new fixture still triggers the dtype-mismatch path
   and that `compat_load` still repairs it.

4. Commit all four files (the regen script, the `.joblib`, the
   `_expected_scores.npy`, and the `_meta.json`) together with the
   `_TREE_NODE_FIELD_DEFAULTS` / `_TREE_NODE_FIELD_RENAMES` update and a
   CHANGES.txt entry.

Do not edit the binary files by hand; always regenerate via the script.
Do not regenerate the fixture as part of routine development; the point
of pinning a real 1.2.x artifact is that it does not drift with the
developer's local sklearn.

## Security

`pickle` and `joblib` load arbitrary Python code. Treat these fixtures as
trusted because they are produced by the regen script under version
control. Do not import them in any context where a third-party fixture
file could substitute for the committed one.
