# -*- coding: utf-8 -*-
"""Tests for ``pyod.utils.persistence``: ``save``, ``load``, ``compat_load``.

Test layers (see ``PLAN-model-persistence.md`` for the full design):

* Phase 1 tests cover ``compat_load`` on synthetic aged pickles and the
  committed binary fixture. Exercises the sklearn Tree-node-dtype
  realignment mechanic end-to-end.
* Phase 2 tests cover ``save`` / ``load`` envelope round-trips, version
  drift detection, strict mode, ``return_metadata``, and ``load()``'s
  automatic fall-through to ``compat_load`` on the documented dtype
  error.

Synthetic aged pickles are produced by ``_make_aged_pickle``, which
fits a detector on the current sklearn and then rewires each
``estimator_.tree_`` to a small ``_OldDtypeTree`` shim. The shim's
``__reduce__`` emits ``(Tree, args, modified_state)`` so that on load,
joblib calls ``Tree.__setstate__`` with the modified state - raising
the same dtype error a real legacy pickle would, and exercising
``compat_load``'s BUILD-dispatch override on a real ``Tree`` instance.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import copy
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
from numpy.testing import assert_allclose

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.ensemble import IsolationForest
from sklearn.tree._tree import Tree as _SkTree

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data
from pyod.utils import persistence as _persistence
from pyod.utils.persistence import (
    _CURRENT_PERSISTENCE_VERSION,
    _DTYPE_MISMATCH_PREFIX,
    _TREE_NODE_FIELD_DEFAULTS,
    compat_load,
    load,
    save,
)


FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


# ---------------------------------------------------------------------
# Helpers: build pickles whose Tree-node dtype mimics an older sklearn
# ---------------------------------------------------------------------

def _current_node_dtype() -> np.dtype:
    from sklearn.tree import _tree
    if hasattr(_tree, 'NODE_DTYPE'):
        return _tree.NODE_DTYPE
    n_classes = np.array([1], dtype=np.intp)
    return _tree.Tree(1, n_classes, 1).__getstate__()['nodes'].dtype


def _strip_field_from_nodes(nodes_arr: np.ndarray, field: str) -> np.ndarray:
    """Return a structured array identical to ``nodes_arr`` with one field
    removed. Used to simulate a pre-1.3 sklearn save."""
    names = [n for n in nodes_arr.dtype.names if n != field]
    new_dtype = np.dtype([(n, nodes_arr.dtype.fields[n][0]) for n in names])
    out = np.zeros(len(nodes_arr), dtype=new_dtype)
    for n in names:
        out[n] = nodes_arr[n]
    return out


def _add_field_to_nodes(
        nodes_arr: np.ndarray, field: str, sample_dtype: str) -> np.ndarray:
    """Add a NEW field to nodes_arr to simulate a future sklearn that
    introduces a field PyOD has not yet allowlisted."""
    descr = [(n, nodes_arr.dtype.fields[n][0]) for n in nodes_arr.dtype.names]
    descr.append((field, np.dtype(sample_dtype)))
    out = np.zeros(len(nodes_arr), dtype=np.dtype(descr))
    for n in nodes_arr.dtype.names:
        out[n] = nodes_arr[n]
    return out


def _retype_field_in_nodes(
        nodes_arr: np.ndarray, field: str, new_dtype: str) -> np.ndarray:
    """Change one field's dtype kind/itemsize to simulate an incompatible
    upstream redefinition."""
    descr = [
        (n, np.dtype(new_dtype) if n == field else nodes_arr.dtype.fields[n][0])
        for n in nodes_arr.dtype.names
    ]
    out = np.zeros(len(nodes_arr), dtype=np.dtype(descr))
    for n in nodes_arr.dtype.names:
        if n == field:
            out[n] = nodes_arr[n].astype(new_dtype)
        else:
            out[n] = nodes_arr[n]
    return out


def _rename_field_in_nodes(
        nodes_arr: np.ndarray, old: str, new: str) -> np.ndarray:
    """Rename a field in nodes_arr to simulate a future upstream rename.
    ``old`` must exist in the current dtype; the returned array has
    ``new`` instead of ``old`` with the same dtype and values."""
    descr = []
    for n in nodes_arr.dtype.names:
        if n == old:
            descr.append((new, nodes_arr.dtype.fields[n][0]))
        else:
            descr.append((n, nodes_arr.dtype.fields[n][0]))
    out = np.zeros(len(nodes_arr), dtype=np.dtype(descr))
    for n in nodes_arr.dtype.names:
        target = new if n == old else n
        out[target] = nodes_arr[n]
    return out


def _byteswap_field_in_nodes(
        nodes_arr: np.ndarray, field: str) -> np.ndarray:
    """Flip the byte order of one field's dtype only. The realignment
    should accept this as a compatible difference."""
    descr = []
    for n in nodes_arr.dtype.names:
        existing = nodes_arr.dtype.fields[n][0]
        if n == field:
            descr.append((n, existing.newbyteorder('S')))
        else:
            descr.append((n, existing))
    out = np.zeros(len(nodes_arr), dtype=np.dtype(descr))
    for n in nodes_arr.dtype.names:
        out[n] = nodes_arr[n]
    return out


def _build_tree_from_args(n_features, n_classes, n_outputs):
    """Module-level constructor used by ``_OldDtypeTree.__reduce__``.
    Must be importable so pickle can resolve it on load."""
    return _SkTree(n_features, n_classes, n_outputs)


class _OldDtypeTree:
    """Shim that replaces a fitted tree's ``tree_`` attribute. On pickle,
    emits ``(_build_tree_from_args, args, modified_state)``. On unpickle,
    a real ``Tree`` is constructed and ``__setstate__`` is called with
    the modified state - the same code path a legacy artifact takes."""

    def __init__(self, real_tree, *, transform):
        self._args = (
            real_tree.n_features,
            real_tree.n_classes,
            real_tree.n_outputs,
        )
        st = real_tree.__getstate__()
        new_nodes = transform(st['nodes'])
        self._state = dict(st)
        self._state['nodes'] = new_nodes

    def __reduce__(self):
        return (_build_tree_from_args, self._args, self._state)


def _make_aged_detector(
        detector,
        *,
        transform=None,
        stripped_field='missing_go_to_left'):
    """Return a deep copy of ``detector`` with each ``tree_`` replaced by
    an ``_OldDtypeTree`` shim. The shim ages the nodes array via
    ``transform`` (default: strip ``stripped_field``)."""
    if transform is None:
        def transform(nodes):
            return _strip_field_from_nodes(nodes, stripped_field)

    aged = copy.deepcopy(detector)
    estimators = _iter_tree_estimators(aged)
    for est in estimators:
        est.tree_ = _OldDtypeTree(est.tree_, transform=transform)
    return aged


def _iter_tree_estimators(detector):
    """Yield the per-tree sklearn estimators of an IForest or
    sklearn.IsolationForest. PyOD's IForest proxies ``estimators_``
    through to its ``detector_`` attribute."""
    estimators = getattr(detector, 'estimators_', None)
    if estimators is None:
        raise ValueError(
            'detector has no estimators_; expected a tree ensemble')
    return list(estimators)


def _make_aged_pickle(detector, path, *, transform=None,
                     stripped_field='missing_go_to_left', compress=0):
    """Save ``detector`` to ``path`` after rewiring its trees through
    ``_OldDtypeTree``. ``transform`` overrides the per-nodes-array
    transformation; the default strips ``stripped_field``."""
    aged = _make_aged_detector(
        detector, transform=transform, stripped_field=stripped_field)
    joblib.dump(aged, path, compress=compress)
    return path


# ---------------------------------------------------------------------
# Fixtures shared across test classes
# ---------------------------------------------------------------------

def _fit_small_iforest(n_estimators=1, random_state=0):
    rng = np.random.RandomState(random_state)
    X = rng.randn(64, 4)
    clf = IForest(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X)
    return clf, X


def _fit_small_sklearn_iforest(n_estimators=1, random_state=0):
    rng = np.random.RandomState(random_state)
    X = rng.randn(64, 4)
    clf = IsolationForest(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X)
    return clf, X


def _fit_small_ecod():
    """A non-Tree-bearing detector for the compat_load pass-through test."""
    X_train, _, _, _ = generate_data(
        n_train=80, n_test=20, contamination=0.1, random_state=42)
    clf = ECOD(contamination=0.1).fit(X_train)
    return clf, X_train


# =====================================================================
# Phase 1: compat_load mechanic
# =====================================================================

class TestCompatLoadCore(unittest.TestCase):
    """Phase 1 tests for ``compat_load`` and its helpers."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _tmp(self, name='artifact.joblib'):
        return os.path.join(self._tmpdir.name, name)

    # ------------------------------------------------------------
    # Realignment on synthetic aged pickles
    # ------------------------------------------------------------

    def test_compat_load_realigns_tree_dtype_iforest_1_estimator(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=1)
        path = _make_aged_pickle(clf, self._tmp())
        # Raw joblib.load should fail with the documented dtype error.
        with self.assertRaises(ValueError) as cm:
            joblib.load(path)
        self.assertTrue(
            str(cm.exception).startswith(_DTYPE_MISMATCH_PREFIX),
            f'unexpected error message: {cm.exception}')
        # compat_load succeeds and the Tree dtype is repaired.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            out = compat_load(path)
        self.assertIsInstance(out, IsolationForest)
        loaded_dtype = out.estimators_[0].tree_.__getstate__()['nodes'].dtype
        self.assertEqual(loaded_dtype, _current_node_dtype())

    def test_compat_load_handles_multiple_trees(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=5)
        path = _make_aged_pickle(clf, self._tmp())
        out = compat_load(path)
        self.assertEqual(len(out.estimators_), 5)
        for est in out.estimators_:
            dtype = est.tree_.__getstate__()['nodes'].dtype
            self.assertEqual(dtype, _current_node_dtype())

    def test_compat_load_preserves_predictions_within_tolerance(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=3)
        expected = clf.decision_function(X).astype(np.float64)
        path = _make_aged_pickle(clf, self._tmp())
        out = compat_load(path)
        got = out.decision_function(X).astype(np.float64)
        # Predictions on inputs WITHOUT missing values must match within
        # float tolerance; the zero-filled ``missing_go_to_left`` default
        # is only exercised when the input contains NaN.
        assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_compat_load_warns_when_realigning_a_tree(self):
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        path = _make_aged_pickle(clf, self._tmp())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            compat_load(path)
        realign = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and 'realigned' in str(w.message).lower()
            and 're-fit' in str(w.message).lower()
        ]
        self.assertEqual(len(realign), 1,
                         f'expected exactly one realign warning, got {caught}')

    # ------------------------------------------------------------
    # Silent pass-through for non-tree artifacts
    # ------------------------------------------------------------

    def test_compat_load_silent_on_non_tree_passthrough(self):
        clf, X = _fit_small_ecod()
        path = self._tmp('ecod.joblib')
        joblib.dump(clf, path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = compat_load(path)
        realign = [w for w in caught
                   if 'realigned' in str(w.message).lower()]
        self.assertEqual(len(realign), 0,
                         'compat_load must not warn on non-tree artifacts')
        self.assertIsInstance(out, ECOD)
        # Predictions still work post-load.
        out.decision_function(X)

    # ------------------------------------------------------------
    # Compression levels
    # ------------------------------------------------------------

    def test_compat_load_with_compress_levels(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=2)
        expected = clf.decision_function(X).astype(np.float64)
        for level in (0, 3, 9):
            with self.subTest(compress=level):
                path = self._tmp(f'aged_c{level}.joblib')
                _make_aged_pickle(clf, path, compress=level)
                out = compat_load(path)
                got = out.decision_function(X).astype(np.float64)
                assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    # ------------------------------------------------------------
    # Committed binary fixture
    # ------------------------------------------------------------

    def test_compat_load_with_committed_fixture(self):
        fixture = FIXTURES_DIR / 'iforest_sklearn_1_2_x.joblib'
        self.assertTrue(fixture.is_file(),
                        f'fixture missing: {fixture}')
        # Raw joblib.load on this real 1.2.2 artifact must trigger the
        # documented dtype-mismatch error.
        with self.assertRaises(ValueError) as cm:
            joblib.load(fixture)
        self.assertTrue(
            str(cm.exception).startswith(_DTYPE_MISMATCH_PREFIX))
        # compat_load succeeds and repairs the dtype. End-to-end
        # decision_function is NOT asserted because sklearn IsolationForest
        # introduced predict-side state (``_decision_path_lengths``,
        # ``monotonic_cst``) in versions beyond 1.3 that legacy 1.2.2
        # pickles cannot carry; that is downstream sklearn API drift,
        # not Tree-node-dtype drift, and is out of scope for
        # ``compat_load``. The synthetic aged-pickle tests above cover
        # end-to-end predict round-trip on the same dtype mechanic.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = compat_load(fixture)
        self.assertIsInstance(out, IsolationForest)
        self.assertEqual(len(out.estimators_), 1)
        node_dtype = out.estimators_[0].tree_.__getstate__()['nodes'].dtype
        self.assertEqual(node_dtype, _current_node_dtype())
        realign = [w for w in caught
                   if issubclass(w.category, UserWarning)
                   and 'realigned' in str(w.message).lower()]
        self.assertEqual(len(realign), 1)

    # ------------------------------------------------------------
    # Allowlist gate
    # ------------------------------------------------------------

    def test_compat_load_rejects_unknown_added_field(self):
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        # Simulate a future sklearn that adds an unknown field to NODE_DTYPE.
        # The current sklearn's NODE_DTYPE will NOT have this field, so
        # the realignment path won't even be entered for this scenario
        # via the normal route. To exercise the unknown-added-field gate,
        # we patch ``_TREE_NODE_FIELD_DEFAULTS`` temporarily so the
        # current dtype's field set differs from the saved one in a way
        # that has no default. Instead of patching that constant, we
        # simulate by REMOVING a current field from the saved pickle
        # that is NOT in _TREE_NODE_FIELD_DEFAULTS.
        # Pick a known-current field that has no allowlist entry.
        current = _current_node_dtype()
        candidates = [n for n in current.names
                      if n not in _TREE_NODE_FIELD_DEFAULTS]
        self.assertTrue(candidates,
                        'no current Tree field lacks a default; '
                        'rejection gate is untestable in this sklearn')
        unknown = candidates[0]

        def strip_unknown(nodes):
            return _strip_field_from_nodes(nodes, unknown)

        path = _make_aged_pickle(clf, self._tmp(), transform=strip_unknown)
        with self.assertRaises(ValueError) as cm:
            compat_load(path)
        self.assertIn(unknown, str(cm.exception))
        self.assertIn('default', str(cm.exception).lower())

    def test_compat_load_rejects_incompatible_field_dtype(self):
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        # Test two incompatible changes: int64 -> int32 (itemsize change)
        # and int64 -> uint64 (signedness change). Both must raise.
        for new_dtype in ('<i4', '<u8'):
            with self.subTest(new_dtype=new_dtype):
                def retype(nodes, _nd=new_dtype):
                    return _retype_field_in_nodes(nodes, 'feature', _nd)

                safe = new_dtype.replace('<', 'le_').replace('>', 'be_')
                path = _make_aged_pickle(
                    clf, self._tmp(f'retype_{safe}.joblib'),
                    transform=retype)
                with self.assertRaises(ValueError) as cm:
                    compat_load(path)
                msg = str(cm.exception)
                self.assertIn('feature', msg)
                self.assertTrue(
                    any(token in msg.lower()
                        for token in ('kind', 'itemsize', 'signedness')),
                    f'rejection message missing dtype rationale: {msg}')

    def test_compat_load_accepts_byte_order_difference(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=1)
        expected = clf.decision_function(X).astype(np.float64)

        def swap(nodes):
            return _byteswap_field_in_nodes(nodes, 'feature')

        path = _make_aged_pickle(clf, self._tmp(), transform=swap)
        out = compat_load(path)
        got = out.decision_function(X).astype(np.float64)
        assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_compat_load_resolves_field_rename_without_default(self):
        # Simulate a future upstream rename: saved dtype has
        # 'feature_legacy', current sklearn has 'feature'. The rename
        # carries data forward; no entry in _TREE_NODE_FIELD_DEFAULTS
        # for 'feature' should be required (it is not "added", it is a
        # rename target). Patches the module-level rename table for the
        # duration of this test only.
        clf, X = _fit_small_sklearn_iforest(n_estimators=1)
        expected = clf.decision_function(X).astype(np.float64)

        def rename(nodes):
            return _rename_field_in_nodes(
                nodes, old='feature', new='feature_legacy')

        original_renames = dict(_persistence._TREE_NODE_FIELD_RENAMES)
        _persistence._TREE_NODE_FIELD_RENAMES.clear()
        _persistence._TREE_NODE_FIELD_RENAMES.update(
            {'feature_legacy': 'feature'})
        self.addCleanup(
            lambda: (
                _persistence._TREE_NODE_FIELD_RENAMES.clear()
                or _persistence._TREE_NODE_FIELD_RENAMES.update(
                    original_renames)))

        path = _make_aged_pickle(clf, self._tmp(), transform=rename)
        out = compat_load(path)
        # Data carried forward through the rename: predictions
        # match because 'feature' values are intact.
        got = out.decision_function(X).astype(np.float64)
        assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    # ------------------------------------------------------------
    # mmap_mode
    # ------------------------------------------------------------

    def test_compat_load_mmap_mode_documented_limit(self):
        # mmap_mode='r' is forwarded to joblib. The realigned ``nodes``
        # ndarray is allocated in regular memory; other arrays in the
        # artifact may still be memory-mapped. The test asserts that
        # compat_load returns successfully and the model can predict.
        clf, X = _fit_small_sklearn_iforest(n_estimators=1)
        expected = clf.decision_function(X).astype(np.float64)
        path = _make_aged_pickle(clf, self._tmp())
        # mmap_mode requires uncompressed (compress=0) artifacts; our
        # helper defaults to compress=0.
        try:
            out = compat_load(path, mmap_mode='r')
        except Exception as exc:
            self.fail(
                f'compat_load(..., mmap_mode="r") raised unexpectedly: {exc}')
        got = out.decision_function(X).astype(np.float64)
        assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# =====================================================================
# Phase 2: save / load envelope
# =====================================================================

class TestSaveLoadRoundtrip(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _tmp(self, name='artifact.joblib'):
        return os.path.join(self._tmpdir.name, name)

    def test_save_load_round_trip_iforest(self):
        clf, X = _fit_small_iforest(n_estimators=2)
        expected = clf.decision_function(X)
        path = self._tmp()
        save(clf, path)
        out = load(path)
        self.assertIsInstance(out, IForest)
        got = out.decision_function(X)
        assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_save_writes_envelope_keys(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = self._tmp()
        save(clf, path, metadata={'note': 'hello'})
        raw = joblib.load(path)
        self.assertIsInstance(raw, dict)
        # Envelope shape: every documented field present with the right type.
        expected_keys = {
            '_pyod_persistence_version', 'pyod_version', 'sklearn_version',
            'numpy_version', 'scipy_version', 'joblib_version',
            'python_version', 'saved_at', 'model_class', 'metadata', 'model',
        }
        self.assertEqual(set(raw.keys()), expected_keys)
        self.assertEqual(raw['_pyod_persistence_version'],
                         _CURRENT_PERSISTENCE_VERSION)
        for k in ('pyod_version', 'sklearn_version', 'numpy_version',
                  'scipy_version', 'joblib_version', 'python_version',
                  'saved_at', 'model_class'):
            self.assertIsInstance(raw[k], str, f'{k} should be str')
        self.assertEqual(raw['metadata'], {'note': 'hello'})
        self.assertIsInstance(raw['model'], IForest)

    def test_save_load_with_user_metadata(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        meta = {'dataset': 'demo', 'run_id': 42, 'tags': ['unit', 'test']}
        path = self._tmp()
        save(clf, path, metadata=meta)
        model, env = load(path, return_metadata=True)
        self.assertIsInstance(model, IForest)
        self.assertEqual(env['metadata'], meta)


class TestLoadLegacy(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _tmp(self, name='artifact.joblib'):
        return os.path.join(self._tmpdir.name, name)

    def test_load_legacy_raw_joblib_passes_through(self):
        clf, X = _fit_small_iforest(n_estimators=1)
        expected = clf.decision_function(X)
        path = self._tmp()
        joblib.dump(clf, path)  # NOT through save() — raw legacy artifact.
        out = load(path)
        self.assertIsInstance(out, IForest)
        assert_allclose(out.decision_function(X), expected,
                        rtol=1e-12, atol=1e-12)

    def test_load_strict_rejects_raw_joblib(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = self._tmp()
        joblib.dump(clf, path)
        with self.assertRaises(ValueError) as cm:
            load(path, strict=True)
        self.assertIn('envelope', str(cm.exception).lower())


# ---------------------------------------------------------------------
# Helpers for envelope-construction tests
# ---------------------------------------------------------------------

def _write_envelope(path, *, model, overrides=None):
    """Write a hand-crafted envelope for tests that need to control
    dependency-version fields. Mirrors ``save()`` but lets the test
    inject specific values via ``overrides``."""
    import pyod
    import platform
    import sklearn
    import scipy
    from datetime import datetime, timezone

    envelope = {
        '_pyod_persistence_version': _CURRENT_PERSISTENCE_VERSION,
        'pyod_version': pyod.__version__,
        'sklearn_version': sklearn.__version__,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'joblib_version': joblib.__version__,
        'python_version': platform.python_version(),
        'saved_at': datetime.now(timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%SZ'),
        'model_class': f'{type(model).__module__}.{type(model).__name__}',
        'metadata': None,
        'model': model,
    }
    if overrides:
        envelope.update(overrides)
    joblib.dump(envelope, path)
    return path


class TestLoadVersionChecks(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _tmp(self, name='artifact.joblib'):
        return os.path.join(self._tmpdir.name, name)

    def test_load_warns_on_sklearn_version_mismatch(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = _write_envelope(
            self._tmp(), model=clf,
            overrides={'sklearn_version': '0.0.0-test-saved'})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            load(path)
        sklearn_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and 'sklearn_version' in str(w.message)
            and '0.0.0-test-saved' in str(w.message)
        ]
        self.assertEqual(len(sklearn_warns), 1)

    def test_load_warns_on_joblib_version_mismatch(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = _write_envelope(
            self._tmp(), model=clf,
            overrides={'joblib_version': '0.0.0-test-saved'})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            load(path)
        warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and 'joblib_version' in str(w.message)
            and '0.0.0-test-saved' in str(w.message)
        ]
        self.assertEqual(len(warns), 1)

    def test_load_strict_raises_on_warn_version_drift(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        # Drift in any ``warn``-severity dependency raises under strict.
        # ``python_version`` drift never raises (info severity).
        for field in ('sklearn_version', 'joblib_version',
                      'numpy_version', 'scipy_version'):
            with self.subTest(field=field):
                path = _write_envelope(
                    self._tmp(f'drift_{field}.joblib'),
                    model=clf,
                    overrides={field: '0.0.0-test-saved'})
                with self.assertRaises(ValueError) as cm:
                    load(path, strict=True)
                self.assertIn(field, str(cm.exception))

    def test_load_info_severity_python_version_drift_is_silent(self):
        # python_version drift is severity 'info': non-strict load must
        # NOT emit a UserWarning, and strict load must NOT raise. After
        # a compat repair, strict mode still raises (a repair happened),
        # but the message must be the no-drift compat message, not the
        # drift one.
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = _write_envelope(
            self._tmp('info_drift.joblib'),
            model=clf,
            overrides={'python_version': '0.0.0-test-saved'})

        # Non-strict: completely silent (info-only drift is diagnostic).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = load(path)
        self.assertIsInstance(out, IForest)
        drift_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and 'python_version' in str(w.message)
        ]
        self.assertEqual(drift_warns, [],
                         'info-severity python_version drift must not warn')

        # Strict: must NOT raise for info-only drift.
        out_strict = load(path, strict=True)
        self.assertIsInstance(out_strict, IForest)

        # After-compat strict path: aged detector + info-only drift.
        # Strict still raises (a repair happened) but the message must
        # be the no-drift compat message, not a dependency-drift one.
        clf_sk, _ = _fit_small_sklearn_iforest(n_estimators=1)
        aged = _make_aged_detector(clf_sk)
        path_after = _write_envelope(
            self._tmp('info_drift_after_compat.joblib'),
            model=aged,
            overrides={'python_version': '0.0.0-test-saved'})
        with self.assertRaises(ValueError) as cm:
            load(path_after, strict=True)
        msg = str(cm.exception)
        self.assertNotIn('python_version', msg,
                         'info-severity drift must not surface in '
                         'strict-after-compat error')
        self.assertIn('compat', msg.lower())

    def test_load_rejects_future_schema_version(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = _write_envelope(
            self._tmp(), model=clf,
            overrides={'_pyod_persistence_version':
                       _CURRENT_PERSISTENCE_VERSION + 99})
        with self.assertRaises(ValueError) as cm:
            load(path)
        msg = str(cm.exception)
        self.assertIn('schema version', msg.lower())
        self.assertIn('upgrade pyod', msg.lower())

    def test_load_strict_raises_after_compat_no_drift(self):
        # Build a Phase 2 envelope that records the running sklearn
        # version exactly, but whose ``model`` field is a tree
        # estimator with the OLD dtype layout. ``load(strict=True)``
        # must raise because a compat repair occurred even though no
        # recorded drift exists.
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        aged = _make_aged_detector(clf)
        path = _write_envelope(self._tmp(), model=aged)
        with self.assertRaises(ValueError) as cm:
            load(path, strict=True)
        msg = str(cm.exception)
        self.assertIn('compat', msg.lower())
        # The original dtype-mismatch error should be in the cause chain.
        cause = cm.exception.__cause__
        self.assertIsNotNone(cause)
        self.assertTrue(str(cause).startswith(_DTYPE_MISMATCH_PREFIX))

    def test_load_return_metadata_true_returns_tuple(self):
        clf, _ = _fit_small_iforest(n_estimators=1)
        path = self._tmp()
        save(clf, path, metadata={'k': 'v'})
        result = load(path, return_metadata=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        model, env = result
        self.assertIsInstance(model, IForest)
        self.assertIsInstance(env, dict)
        # Envelope-without-model must not include 'model'.
        self.assertNotIn('model', env)
        self.assertIn('_pyod_persistence_version', env)


class TestLoadAutoFallthrough(unittest.TestCase):
    """``load()`` must fall through to ``compat_load`` on the documented
    sklearn Tree dtype error, routing the recovered object through the
    same envelope/legacy handler."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _tmp(self, name='artifact.joblib'):
        return os.path.join(self._tmpdir.name, name)

    def test_load_auto_fallthrough_raw_legacy_to_compat(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=1)
        expected = clf.decision_function(X).astype(np.float64)
        path = _make_aged_pickle(clf, self._tmp())
        out = load(path)
        self.assertIsInstance(out, IsolationForest)
        got = out.decision_function(X).astype(np.float64)
        assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_load_auto_fallthrough_envelope_returns_model(self):
        clf, X = _fit_small_sklearn_iforest(n_estimators=1)
        aged = _make_aged_detector(clf)
        path = _write_envelope(self._tmp(), model=aged)
        out = load(path)
        # Must be the model, not the envelope dict.
        self.assertIsInstance(out, IsolationForest)
        self.assertNotIsInstance(out, dict)

    def test_load_auto_fallthrough_envelope_with_return_metadata(self):
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        aged = _make_aged_detector(clf)
        path = _write_envelope(
            self._tmp(), model=aged,
            overrides={'metadata': {'tag': 'aged'}})
        result = load(path, return_metadata=True)
        self.assertIsInstance(result, tuple)
        model, env = result
        self.assertIsInstance(model, IsolationForest)
        self.assertIsInstance(env, dict)
        self.assertNotIn('model', env)
        self.assertEqual(env['metadata'], {'tag': 'aged'})

    def test_load_auto_fallthrough_envelope_strict_raises_on_drift(self):
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        aged = _make_aged_detector(clf)
        path = _write_envelope(
            self._tmp(), model=aged,
            overrides={'sklearn_version': '0.0.0-test-saved'})
        with self.assertRaises(ValueError) as cm:
            load(path, strict=True)
        cause = cm.exception.__cause__
        self.assertIsNotNone(cause)
        self.assertTrue(str(cause).startswith(_DTYPE_MISMATCH_PREFIX))
        msg = str(cm.exception)
        self.assertIn('compat', msg.lower())
        self.assertIn('sklearn_version', msg)

    def test_load_auto_fallthrough_envelope_emits_version_warning(self):
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        aged = _make_aged_detector(clf)
        path = _write_envelope(
            self._tmp(), model=aged,
            overrides={'sklearn_version': '0.0.0-test-saved'})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            load(path)
        # Both the version-drift warning and the post-compat re-fit
        # recommendation should fire.
        drift = [w for w in caught
                 if 'sklearn_version' in str(w.message)
                 and '0.0.0-test-saved' in str(w.message)]
        compat = [w for w in caught
                  if 'recovered' in str(w.message).lower()
                  or 're-fit' in str(w.message).lower()]
        self.assertGreaterEqual(len(drift), 1)
        self.assertGreaterEqual(len(compat), 1)

    def test_load_auto_fallthrough_preserves_exception_chain(self):
        # If compat_load itself raises (e.g., unknown added field), the
        # raised exception must be chained from the original dtype error.
        clf, _ = _fit_small_sklearn_iforest(n_estimators=1)
        current = _current_node_dtype()
        unknown = next(n for n in current.names
                       if n not in _TREE_NODE_FIELD_DEFAULTS)

        def strip_unknown(nodes):
            return _strip_field_from_nodes(nodes, unknown)

        path = _make_aged_pickle(
            clf, self._tmp(), transform=strip_unknown)
        with self.assertRaises(ValueError) as cm:
            load(path)
        cause = cm.exception.__cause__
        self.assertIsNotNone(cause,
                             'compat fall-through must chain the original')
        self.assertTrue(str(cause).startswith(_DTYPE_MISMATCH_PREFIX))

    def test_load_auto_fallthrough_trigger_is_exact_prefix(self):
        # Pin the public fall-through contract at the joblib.load level:
        # a ValueError whose message does NOT start with the documented
        # dtype prefix must propagate directly (no compat_load call); a
        # ValueError that DOES start with the prefix must trigger
        # compat_load. Monkey-patching joblib.load is the only way to
        # exercise this gate independently of unrelated error paths
        # (schema validation, envelope parsing) that surface AFTER
        # joblib.load returns.
        path = self._tmp()
        # The path must exist on disk because load() opens it before
        # the monkey-patched joblib.load is reached.
        Path(path).write_bytes(b'')

        compat_calls = []

        def fake_compat_load(p, mmap_mode=None):
            compat_calls.append(str(p))
            raise RuntimeError(
                'fake_compat_load should not run for this branch')

        # Case 1: non-prefix ValueError. compat_load must NOT be called;
        # the original ValueError must propagate untouched.
        non_prefix_exc = ValueError('completely unrelated joblib error')

        def fake_load_non_prefix(*args, **kwargs):
            raise non_prefix_exc

        original_joblib_load = _persistence.joblib.load
        original_compat_load = _persistence.compat_load
        _persistence.joblib.load = fake_load_non_prefix
        _persistence.compat_load = fake_compat_load
        try:
            with self.assertRaises(ValueError) as cm:
                load(path)
            self.assertIs(cm.exception, non_prefix_exc,
                          'non-prefix ValueError must propagate unchanged')
        finally:
            _persistence.joblib.load = original_joblib_load
            _persistence.compat_load = original_compat_load
        self.assertEqual(compat_calls, [],
                         'non-prefix ValueError must NOT trigger compat_load')

        # Case 2: prefix ValueError. compat_load MUST be called.
        prefix_exc = ValueError(
            _DTYPE_MISMATCH_PREFIX
            + ': expected layout X, got layout Y')

        def fake_load_prefix(*args, **kwargs):
            raise prefix_exc

        _persistence.joblib.load = fake_load_prefix
        _persistence.compat_load = fake_compat_load
        try:
            # compat_load is faked to raise; load() chains it from the
            # original. We only need to confirm compat_load was called.
            with self.assertRaises(Exception):
                load(path)
        finally:
            _persistence.joblib.load = original_joblib_load
            _persistence.compat_load = original_compat_load
        self.assertEqual(len(compat_calls), 1,
                         'prefix ValueError MUST trigger compat_load exactly '
                         'once')


if __name__ == '__main__':
    unittest.main(verbosity=2)
