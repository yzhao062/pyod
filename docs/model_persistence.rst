Model Save and Load
===================

PyOD ships a small, versioned wrapper around ``joblib`` that solves
two recurring pain points: cross-sklearn-version compatibility for
saved models, and the absence of any record of *what* a saved model
was fit with. The recommended API lives in
:mod:`pyod.utils.persistence`.

Quick Start
-----------

.. code-block:: python

    from pyod.models.iforest import IForest
    from pyod.utils.persistence import save, load

    clf = IForest().fit(X_train)

    # Save with a versioned envelope.
    save(clf, "clf.pyod.joblib", metadata={"dataset": "demo"})

    # Later, in a possibly different environment:
    clf = load("clf.pyod.joblib")

    # Or get the envelope back alongside the model:
    clf, env = load("clf.pyod.joblib", return_metadata=True)
    print(env["sklearn_version"], env["saved_at"])

The complete example in
`examples/save_load_model_example.py <https://github.com/yzhao062/pyod/blob/master/examples/save_load_model_example.py>`_
also covers the legacy ``joblib.dump`` / ``joblib.load`` flow as a
secondary alternative.

Trust Boundary
--------------

``pickle`` and ``joblib`` deserialize arbitrary Python code. Load only
from sources you trust. This applies equally to raw ``joblib.load``,
raw ``pickle.load``, :func:`~pyod.utils.persistence.load`, and
:func:`~pyod.utils.persistence.compat_load`. The new wrapper does not
change this security model; it does not sandbox the unpickling step.

Why a Versioned Wrapper
-----------------------

Saving a fitted detector with plain ``joblib.dump`` writes the model
and nothing else. When a downstream user later calls ``joblib.load``,
the running environment's sklearn, numpy, scipy, joblib, and Python
versions may differ from the save environment in ways that change
predictions or break loading outright. Users on PyOD have reported
this exact failure mode (see issue
`#519 <https://github.com/yzhao062/pyod/issues/519>`_) when sklearn
evolves its internal pickle layout; the error message is
``ValueError: node array from the pickle has an incompatible dtype``.

:func:`~pyod.utils.persistence.save` records the dependency versions
in effect at save time alongside the model.
:func:`~pyod.utils.persistence.load` reads that envelope and emits a
clear warning when any binary-format dependency drifts, so the issue
surfaces at load time rather than during a later prediction
incident. The schema is documented and stable; future PyOD releases
will read envelopes written by earlier ones.

Loading Legacy Pickles
----------------------

If you already have artifacts saved with raw ``joblib.dump`` and they
fail to load with the dtype-mismatch error,
:func:`~pyod.utils.persistence.compat_load` repairs the most common
case: sklearn introduced a new Tree-node field (``missing_go_to_left``
in 1.3) and old pickles do not carry it. ``compat_load`` patches
joblib's unpickler so the saved Tree state is realigned to the
running sklearn's dtype before sklearn's own ``__setstate__`` raises.

.. code-block:: python

    from pyod.utils.persistence import compat_load

    clf = compat_load("legacy.joblib")
    # Re-save under the new envelope to avoid repeating the dance:
    from pyod.utils.persistence import save
    save(clf, "legacy_resaved.pyod.joblib")

You usually do not need to call ``compat_load`` directly.
:func:`~pyod.utils.persistence.load` falls through to ``compat_load``
automatically when ``joblib.load`` raises the documented dtype
error, and routes the recovered model through the same envelope or
legacy handler:

.. code-block:: python

    from pyod.utils.persistence import load

    clf = load("legacy.joblib")   # transparently recovers from dtype drift

The fall-through emits a ``UserWarning`` so the recovery does not
go unnoticed. Re-save with :func:`~pyod.utils.persistence.save` (or
re-fit on the current sklearn) to remove the dependency on the
compat path.

Decision Tree
~~~~~~~~~~~~~

::

    Saving a new model?
        -> use save(clf, path)

    Loading a model and load(path) works without warnings?
        -> done

    Loading a model and load(path) succeeds with a "recovered" warning?
        -> the artifact was repaired via compat_load; re-save with save()

    Loading a model and load(path) raises?
        -> if the error is about Tree-node dtype, try compat_load directly
           and check whether the warning recommends re-fit. If it cannot
           recover, re-fit on the current sklearn.

Cross-Sklearn-Version Compatibility
-----------------------------------

The most common cross-version failure is the sklearn Tree node dtype
evolving across minor releases. sklearn 1.3 added a
``missing_go_to_left`` field to its Tree node struct; older pickles
omit that field, and loading them on 1.3 or later raises
``ValueError: node array from the pickle has an incompatible dtype``.

:func:`~pyod.utils.persistence.compat_load` is the supported escape
hatch for this case. It is allowlist-driven and conservative:

* Missing fields in the saved dtype that PyOD has documented a safe
  default for (currently only ``missing_go_to_left = 0``, the
  pre-1.3 "do not route on missingness" behavior) are zero-filled.
* Missing fields without a documented default raise ``ValueError``
  rather than silently inventing a value.
* Field-level dtype changes beyond byte order (kind, signedness,
  itemsize, shape) raise ``ValueError`` rather than silently
  casting.
* Byte-order-only differences are realigned safely.

Two caveats apply. First, ``compat_load`` is best-effort: predictions
on inputs that contain missing values may differ from what the
original training would have produced, because zero-filled defaults
for fields like ``missing_go_to_left`` need not match what the
original training would have implied. The durable fix is to re-fit on
the current sklearn. Second, ``compat_load`` only repairs the Tree
node dtype. Other cross-version sklearn changes (newly required
private cached state, newly added class attributes) are out of
scope. If ``compat_load`` succeeds but predictions still fail with a
different sklearn error, re-fit on the current sklearn.

Troubleshooting
~~~~~~~~~~~~~~~

==================================================================  ==================================================================
Error text starts with                                              Recommended action
==================================================================  ==================================================================
``node array from the pickle has an incompatible dtype``            Try :func:`~pyod.utils.persistence.compat_load`. If it succeeds, re-save with :func:`~pyod.utils.persistence.save`. If it raises, re-fit.
``InconsistentVersionWarning`` (only a warning, not an error)       Safe to ignore; sklearn is reminding you the save and run versions differ. Re-save or re-fit when convenient.
Other sklearn unpickling errors                                     The artifact is incompatible beyond what ``compat_load`` repairs. Re-fit on the current sklearn.
==================================================================  ==================================================================

Strict Mode
-----------

For version-pinned production environments, pass ``strict=True`` to
:func:`~pyod.utils.persistence.load`:

.. code-block:: python

    from pyod.utils.persistence import load

    clf = load("prod.pyod.joblib", strict=True)

Under strict mode, any drift in sklearn, joblib, numpy, or scipy
raises ``ValueError`` rather than emitting a warning. Drift in the
Python version does not raise because it is informational only.
Strict mode also rejects raw legacy artifacts (no envelope to
compare against) and refuses to return a model that required a
``compat_load`` repair: strict callers must either re-save under the
current environment or re-fit.

Reading Envelope Metadata
-------------------------

``load(path, return_metadata=True)`` returns a ``(model, envelope)``
tuple where ``envelope`` is the full envelope dict minus the
``model`` field:

.. code-block:: python

    from pyod.utils.persistence import load

    clf, env = load("clf.pyod.joblib", return_metadata=True)
    print(env["pyod_version"], env["sklearn_version"])
    print(env["saved_at"], env["model_class"])
    print(env["metadata"])   # whatever you passed to save(... metadata=...)

A future PyOD release plans a true header-only ``inspect_artifact``
(reading metadata without unpickling the model), paired with a
``.pyod`` zip container that separates metadata from the model
payload. Until that ships, ``load(..., return_metadata=True)`` is
the supported way to introspect a saved artifact, and it does
unpickle the model.

Neural Network Models
---------------------

Saving deep-learning detectors that wrap ``torch.nn.Module`` (e.g.,
``AutoEncoder``, ``DeepSVDD``, ``VAE``) has separate constraints
that this module does not yet address; see issues
`#88 <https://github.com/yzhao062/pyod/issues/88#issuecomment-615343139>`_
and
`#328 <https://github.com/yzhao062/pyod/issues/328#issuecomment-917192704>`_
for the current workaround.
