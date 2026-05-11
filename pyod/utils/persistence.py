# -*- coding: utf-8 -*-
"""Cross-sklearn-version model persistence for PyOD.

This module is the recommended way to save and load PyOD detectors.
It wraps `joblib` with two capabilities the raw `joblib.dump` /
`joblib.load` path does not provide:

1. A versioned envelope written by `save()`. The envelope records the
   PyOD, sklearn, numpy, scipy, joblib, and Python versions in effect
   at save time. `load()` compares the envelope against the running
   environment and emits a `UserWarning` when any binary-format
   dependency drifts; `load(..., strict=True)` raises instead. This
   lets users detect dependency drift before it surprises them in
   production.
2. A `compat_load()` helper that loads legacy artifacts whose sklearn
   `Tree` node dtype no longer matches the running sklearn (a recurring
   user pain documented in issue #519). `compat_load` uses joblib's
   own unpickler with the BUILD-opcode dispatch entry patched so that
   sklearn `Tree` state is realigned to the running dtype before
   `sklearn.tree._tree.Tree.__setstate__` sees it.

`load()` automatically falls through to `compat_load()` when the
underlying `joblib.load` raises the specific sklearn dtype `ValueError`,
so users who only call `load()` get the rescue path transparently.

WARNING: pickle and joblib load arbitrary Python code. Load only from
trusted sources. The compat_load helper does not change this security
model.

See `docs/model_persistence.rst` for the user-facing guide.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import annotations

import pickle
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from joblib.numpy_pickle import NumpyUnpickler


__all__ = ["save", "load", "compat_load"]


# ----------------------------------------------------------------------
# Module constants
# ----------------------------------------------------------------------

_CURRENT_PERSISTENCE_VERSION = 1
"""Newest envelope schema version this module writes and reads."""

# Conservative, allowlist-driven dtype realignment for sklearn Tree
# node arrays. Adding a new entry is a deliberate maintenance act:
# it pairs with a regression test, a CHANGES.txt note, and a
# documentation update.
_TREE_NODE_FIELD_DEFAULTS: dict[str, Any] = {
    "missing_go_to_left": 0,
}
"""Tree-node fields the loader may zero-fill when missing from the saved
dtype. Defaults must match what sklearn's pre-existing behavior implied
for legacy models. `missing_go_to_left=0` mirrors the
"don't route on missingness" behavior of pre-1.3 sklearn."""

_TREE_NODE_FIELD_RENAMES: dict[str, str] = {}
"""Tree-node fields the loader may map from an old name to a new name.
Empty in v1 because sklearn has not renamed any tree fields
historically; populated only when an upstream rename is observed."""

_DTYPE_MISMATCH_PREFIX = "node array from the pickle has an incompatible dtype"
"""Exact-prefix string that triggers `load()`'s auto-fall-through to
`compat_load`. If sklearn changes the error text, fall-through stops
firing and the original error propagates — which is the safe failure
mode because it preserves diagnostic context."""

# Version drift checks performed by `load()`. Each row: envelope key,
# callable returning the running value, severity. `warn` entries emit
# UserWarning when drift is detected and escalate to ValueError under
# strict mode. `info` entries never raise; they are recorded for
# diagnostics only.
_VERSION_CHECKS: list[tuple[str, Any, str]] = [
    ("sklearn_version", lambda: _running_version("sklearn"), "warn"),
    ("joblib_version", lambda: joblib.__version__, "warn"),
    ("numpy_version", lambda: np.__version__, "warn"),
    ("scipy_version", lambda: _running_version("scipy"), "warn"),
    ("python_version", lambda: platform.python_version(), "info"),
]


def _running_version(package_name: str) -> str:
    """Resolve the running version of an optional dependency."""
    if package_name == "sklearn":
        import sklearn
        return sklearn.__version__
    if package_name == "scipy":
        import scipy
        return scipy.__version__
    raise KeyError(package_name)


# ----------------------------------------------------------------------
# save
# ----------------------------------------------------------------------

def save(model: Any, path: Any, metadata: dict | None = None) -> None:
    """Save a fitted PyOD detector with a versioned envelope.

    The envelope records every dependency version that can affect
    pickle/joblib layout, plus a save timestamp and a user-supplied
    metadata dict. The actual model object is written via
    ``joblib.dump``; the only difference from raw ``joblib.dump(clf,
    path)`` is that the model is wrapped in a header dict the
    matching ``load()`` recognizes.

    Parameters
    ----------
    model : Any
        The fitted detector to save. Anything picklable will work; PyOD
        BaseDetector subclasses are the typical case.
    path : str or pathlib.Path
        Destination file path.
    metadata : dict or None
        Optional user-supplied metadata (training dataset id, feature
        schema hash, run id, anything). No schema is imposed; the dict
        round-trips as-is.

    Returns
    -------
    None

    Notes
    -----
    Loading the file with raw ``joblib.load`` returns the envelope
    dict, not the model. Use ``load()`` from this module to unwrap.
    """
    import pyod
    envelope = {
        "_pyod_persistence_version": _CURRENT_PERSISTENCE_VERSION,
        "pyod_version": pyod.__version__,
        "sklearn_version": _running_version("sklearn"),
        "numpy_version": np.__version__,
        "scipy_version": _running_version("scipy"),
        "joblib_version": joblib.__version__,
        "python_version": platform.python_version(),
        "saved_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "model_class": f"{type(model).__module__}."
                       f"{type(model).__name__}",
        "metadata": metadata,
        "model": model,
    }
    joblib.dump(envelope, path)


# ----------------------------------------------------------------------
# load
# ----------------------------------------------------------------------

def load(
        path: Any,
        strict: bool = False,
        return_metadata: bool = False) -> Any:
    """Load a PyOD detector saved by `save()` or by raw joblib.dump.

    `load()` understands three input shapes:

    1. An envelope dict written by `save()`. The envelope's recorded
       dependency versions are compared against the running
       environment. Drift in sklearn, joblib, numpy, or scipy emits a
       `UserWarning`; `strict=True` raises `ValueError` instead.
    2. A raw detector object written by `joblib.dump(clf, path)` on a
       previous PyOD release. Returned as-is when `strict=False`;
       raises under `strict=True` because legacy artifacts have no
       envelope to verify.
    3. A file that fails the initial `joblib.load` with the
       sklearn `Tree` node dtype error. `load()` falls through to
       `compat_load(path)` and routes the recovered object through
       the same envelope/legacy handler. See module docstring.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the artifact.
    strict : bool, default False
        When True, version drift in any `warn`-severity dependency
        raises `ValueError`. `info`-severity drift (Python version)
        never raises. Legacy artifacts without an envelope also raise
        under strict mode.
    return_metadata : bool, default False
        When True, return ``(model, envelope_without_model_field)``
        instead of just the model. For legacy artifacts the second
        element is ``None``.

    Returns
    -------
    model : Any
        The unpickled model. When `return_metadata=True`, returns
        ``(model, envelope_dict_or_None)``.

    Raises
    ------
    ValueError
        On schema-version mismatch, strict-mode drift, strict-mode
        legacy artifacts, or after a successful compat repair under
        strict mode.
    """
    try:
        obj = joblib.load(path)
    except ValueError as exc:
        if not str(exc).startswith(_DTYPE_MISMATCH_PREFIX):
            raise
        return _handle_compat_fallthrough(
            path, exc, strict, return_metadata)
    return _handle_loaded_object(
        obj, strict, return_metadata,
        after_compat=False, original_exc=None)


def _handle_loaded_object(
        obj: Any,
        strict: bool,
        return_metadata: bool,
        *,
        after_compat: bool,
        original_exc: BaseException | None) -> Any:
    """Route a loaded top-level object through envelope/legacy handlers.

    Shared by the non-fall-through path and by the
    post-`compat_load` path; the `after_compat` flag changes the
    strict-mode behavior (strict ALWAYS raises after a compat repair).
    """
    if _is_envelope(obj):
        _validate_schema_version(obj)
        drift = _check_versions(obj)
        warn_drift = [d for d in drift if d[3] == "warn"]
        if strict:
            if after_compat:
                if warn_drift:
                    raise ValueError(
                        _format_strict_compat_drift_msg(warn_drift)
                    ) from original_exc
                raise ValueError(
                    "load(strict=True): artifact required compatibility "
                    "repair (sklearn Tree dtype realignment) even "
                    "though recorded dependency versions match the "
                    "running environment. Re-save or re-fit the model "
                    "to remove the dependency on compat_load."
                ) from original_exc
            if warn_drift:
                raise ValueError(_format_drift_msg(warn_drift))
        else:
            if warn_drift:
                warnings.warn(
                    _format_drift_msg(warn_drift),
                    UserWarning, stacklevel=3)
            if after_compat:
                warnings.warn(
                    "load(): recovered model after sklearn Tree dtype "
                    "realignment. Re-save with save() to update the "
                    "envelope, or re-fit on the current sklearn for "
                    "the most reliable predictions.",
                    UserWarning, stacklevel=3)
        model = obj["model"]
        if return_metadata:
            envelope_no_model = {k: v for k, v in obj.items()
                                 if k != "model"}
            return model, envelope_no_model
        return model
    # Raw legacy detector
    if strict:
        if after_compat:
            raise ValueError(
                "load(strict=True): legacy artifact (no envelope) "
                "required compatibility repair. Strict mode cannot "
                "verify a legacy artifact and cannot return a repaired "
                "model. Re-save with save() or re-fit."
            ) from original_exc
        raise ValueError(
            "load(strict=True): artifact is a raw legacy save with no "
            "envelope; strict mode requires an envelope produced by "
            "save(). Use strict=False to load anyway.")
    if after_compat:
        warnings.warn(
            "load(): recovered a legacy artifact after sklearn Tree "
            "dtype realignment. Re-save with save() to opt in to the "
            "versioned envelope going forward, or re-fit on the "
            "current sklearn.",
            UserWarning, stacklevel=3)
    if return_metadata:
        return obj, None
    return obj


def _handle_compat_fallthrough(
        path: Any,
        original_exc: BaseException,
        strict: bool,
        return_metadata: bool) -> Any:
    try:
        obj = compat_load(path)
    except Exception as compat_exc:
        raise compat_exc from original_exc
    return _handle_loaded_object(
        obj, strict, return_metadata,
        after_compat=True, original_exc=original_exc)


def _is_envelope(obj: Any) -> bool:
    return (isinstance(obj, dict)
            and "_pyod_persistence_version" in obj
            and "model" in obj)


def _validate_schema_version(envelope: dict) -> None:
    v = envelope.get("_pyod_persistence_version")
    if not isinstance(v, int):
        raise ValueError(
            "load(): envelope has unsupported "
            f"_pyod_persistence_version={v!r}; expected an integer.")
    if v > _CURRENT_PERSISTENCE_VERSION:
        raise ValueError(
            f"load(): envelope schema version {v} is newer than this "
            f"PyOD release supports (max {_CURRENT_PERSISTENCE_VERSION}). "
            "Upgrade PyOD to read this artifact.")
    if v < 1:
        raise ValueError(
            f"load(): envelope schema version {v} is unrecognized.")
    # v in [1, _CURRENT_PERSISTENCE_VERSION] — supported by this release.


def _check_versions(envelope: dict) -> list[tuple[str, str, str, str]]:
    """Return a list of (field, saved, running, severity) tuples for
    every recorded dependency that drifted from the running version."""
    drift = []
    for field, runner, severity in _VERSION_CHECKS:
        saved = envelope.get(field)
        if saved is None:
            continue
        try:
            running = runner()
        except Exception:
            # Optional dep missing at load time; treat as no drift.
            continue
        if saved != running:
            drift.append((field, saved, running, severity))
    return drift


def _format_drift_msg(drift: list[tuple[str, str, str, str]]) -> str:
    warn_rows = [d for d in drift if d[3] == "warn"]
    if not warn_rows:
        return ""
    parts = ", ".join(
        f"{field}={saved!r} (running {running!r})"
        for field, saved, running, _ in warn_rows)
    return (
        "load(): dependency drift detected between saved envelope and "
        f"running environment: {parts}. Predictions may differ from "
        "what the model was trained to produce. Consider re-fitting on "
        "the current environment.")


def _format_strict_compat_drift_msg(
        drift: list[tuple[str, str, str, str]]) -> str:
    warn_rows = [d for d in drift if d[3] == "warn"]
    parts = ", ".join(
        f"{field}={saved!r} (running {running!r})"
        for field, saved, running, _ in warn_rows)
    return (
        "load(strict=True): artifact required compatibility repair "
        "(sklearn Tree dtype realignment) and recorded dependency "
        f"versions also drifted: {parts}. Re-save or re-fit the "
        "model.")


# ----------------------------------------------------------------------
# compat_load
# ----------------------------------------------------------------------

def compat_load(path: Any, mmap_mode: str | None = None) -> Any:
    """Load an artifact whose sklearn Tree node dtype no longer matches.

    Mirrors `joblib.load` but plugs a dispatch-table override into
    joblib's unpickler so that sklearn `Tree` state is realigned to
    the running sklearn dtype before `Tree.__setstate__` raises.

    Realignment is name-based and bounded by `_TREE_NODE_FIELD_DEFAULTS`
    plus `_TREE_NODE_FIELD_RENAMES`. Unknown added/removed fields,
    dtype kind/signedness/itemsize changes, and shape changes raise
    `ValueError`. Same-name byte-order-only differences realign safely.

    Emits a `UserWarning` recommending re-fit ONLY when at least one
    Tree was actually realigned. A no-op pass-through on a non-tree
    artifact is silent.

    Parameters
    ----------
    path : str, pathlib.Path, or file-like
        The artifact to load.
    mmap_mode : str or None, default None
        Forwarded to joblib's underlying load path. Supported values
        mirror joblib's: None, 'r', 'r+', 'w+', 'c'.

    Returns
    -------
    obj : Any
        The raw top-level object from the file (a fitted detector for
        legacy raw saves; an envelope dict for Phase 2 saves). Callers
        that need envelope unwrapping should use `load()`.
    """
    trees_realigned = [0]

    class _CompatNumpyUnpickler(NumpyUnpickler):
        dispatch = NumpyUnpickler.dispatch.copy()

        def load_build(self):
            if len(self.stack) >= 2:
                state = self.stack[-1]
                inst = self.stack[-2]
                if _is_sklearn_tree(inst) and isinstance(state, dict):
                    new_state = _maybe_realign_tree_state(state)
                    if new_state is not state:
                        self.stack[-1] = new_state
                        trees_realigned[0] += 1
            return super().load_build()

    _CompatNumpyUnpickler.dispatch[pickle.BUILD[0]] = (
        _CompatNumpyUnpickler.load_build)

    obj = _load_with_unpickler(path, _CompatNumpyUnpickler, mmap_mode)

    if trees_realigned[0] > 0:
        warnings.warn(
            f"compat_load: realigned {trees_realigned[0]} sklearn "
            "Tree(s) to the current sklearn dtype. Predictions on "
            "inputs WITH missing values may differ from what the "
            "original model would have produced because zero-filled "
            "defaults for newly-added node fields may not match the "
            "original training behavior. Re-fit on the current sklearn "
            "is recommended for the most reliable predictions.",
            UserWarning, stacklevel=2)
    return obj


def _load_with_unpickler(
        path: Any,
        unpickler_cls: type,
        mmap_mode: str | None) -> Any:
    """Mirror of joblib.load that swaps in a custom unpickler class.

    Re-uses joblib's `_validate_fileobject_and_memmap` so compressed
    files and mmap_mode follow the same code path as `joblib.load`.
    Requires joblib >= 1.5 because earlier versions lacked
    `_validate_fileobject_and_memmap` and used a different
    `NumpyUnpickler` constructor signature.
    """
    try:
        from joblib.numpy_pickle import (
            _validate_fileobject_and_memmap, load_compatibility)
    except ImportError as exc:
        raise ImportError(
            "compat_load requires joblib>=1.5 because it reuses "
            "joblib's validated file-object and mmap loader, which were "
            "added in joblib 1.5. Upgrade joblib, or use joblib.load "
            "for artifacts that do not need sklearn Tree dtype repair."
        ) from exc

    # Mirror joblib.load's normalization of Path and file-like input.
    if isinstance(path, Path):
        path = str(path)

    ensure_native_byte_order = mmap_mode is None

    def _run(file_handle, filename, validated_mmap_mode):
        if isinstance(file_handle, str):
            # Joblib pre-0.10 legacy format path.
            return load_compatibility(file_handle)
        unpickler = unpickler_cls(
            filename, file_handle,
            ensure_native_byte_order,
            mmap_mode=validated_mmap_mode)
        return unpickler.load()

    if hasattr(path, "read"):
        fobj = path
        filename = getattr(fobj, "name", "")
        with _validate_fileobject_and_memmap(
                fobj, filename, mmap_mode) as (fh, validated):
            return _run(fh, filename, validated)
    with open(path, "rb") as f:
        with _validate_fileobject_and_memmap(
                f, path, mmap_mode) as (fh, validated):
            return _run(fh, path, validated)


def _is_sklearn_tree(inst: Any) -> bool:
    """True iff `inst` is `sklearn.tree._tree.Tree`. Imported lazily
    so loading non-tree artifacts does not require sklearn to be
    importable on this code path."""
    try:
        from sklearn.tree._tree import Tree
    except Exception:
        return False
    return isinstance(inst, Tree)


def _current_tree_node_dtype() -> np.dtype:
    """Discover the running sklearn's Tree node dtype dynamically.

    Reads `sklearn.tree._tree.NODE_DTYPE` when available (sklearn
    >= 1.0). Falls back to introspecting an empty `Tree` instance.
    """
    from sklearn.tree import _tree
    if hasattr(_tree, "NODE_DTYPE"):
        return _tree.NODE_DTYPE
    n_classes = np.array([1], dtype=np.intp)
    t = _tree.Tree(1, n_classes, 1)
    return t.nodes.dtype


def _maybe_realign_tree_state(state: dict) -> dict:
    """Return `state` unchanged if no realignment is needed, otherwise
    return a new state dict with the `nodes` ndarray realigned to the
    running sklearn dtype. Raises `ValueError` on unsafe differences.
    """
    nodes = state.get("nodes")
    if not isinstance(nodes, np.ndarray):
        return state
    current = _current_tree_node_dtype()
    if nodes.dtype == current:
        return state

    saved_names = set(nodes.dtype.names or ())
    current_names = set(current.names or ())

    # Resolve recognized renames first. A field that is "added" from
    # current's perspective is treated as truly new only if no known
    # rename maps a removed saved field to it; otherwise the rename
    # carries the data forward and the default-required check does not
    # apply.
    raw_added = current_names - saved_names
    raw_removed = saved_names - current_names
    rename_map: dict[str, str] = {}
    for old_name in raw_removed:
        target = _TREE_NODE_FIELD_RENAMES.get(old_name)
        if target is None or target not in current_names:
            raise ValueError(
                "compat_load: saved Tree node dtype has field "
                f"{old_name!r} which the running sklearn does not "
                "recognize. Dropping it could change predictions. "
                "Re-fit on the current sklearn.")
        rename_map[old_name] = target
    rename_targets = set(rename_map.values())

    # Fields genuinely added in current (not produced by any rename).
    added = raw_added - rename_targets
    for name in added:
        if name not in _TREE_NODE_FIELD_DEFAULTS:
            raise ValueError(
                "compat_load: saved Tree node dtype is missing field "
                f"{name!r} which the running sklearn requires. No "
                "default is registered for this field, so silent "
                "zero-fill is unsafe. Re-fit the model on the current "
                "sklearn or add a compatibility entry to "
                "_TREE_NODE_FIELD_DEFAULTS with the correct default.")

    # Shared field dtypes must match modulo byte order.
    shared = saved_names & current_names
    for name in shared:
        if not _dtypes_compatible_modulo_endian(
                nodes.dtype.fields[name][0],
                current.fields[name][0]):
            raise ValueError(
                "compat_load: saved Tree node field "
                f"{name!r} has dtype {nodes.dtype.fields[name][0]!r} "
                f"but the running sklearn expects "
                f"{current.fields[name][0]!r}. Anything beyond a byte-"
                "order difference (kind, signedness, itemsize, shape) "
                "could change predictions. Re-fit on the current "
                "sklearn.")

    # Realign.
    new = np.zeros(len(nodes), dtype=current)
    for name in shared:
        new[name] = nodes[name]
    for old_name, new_name in rename_map.items():
        new[new_name] = nodes[old_name]
    for name in added:
        new[name] = _TREE_NODE_FIELD_DEFAULTS[name]
    return {**state, "nodes": new}


def _dtypes_compatible_modulo_endian(
        a: np.dtype, b: np.dtype) -> bool:
    """True iff `a` and `b` differ at most in byte order."""
    if a == b:
        return True
    # Compare the str representation with byte-order stripped.
    return (a.kind == b.kind
            and a.itemsize == b.itemsize
            and a.shape == b.shape
            and a.subdtype == b.subdtype)
