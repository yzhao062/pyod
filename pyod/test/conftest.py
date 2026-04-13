"""Pytest configuration for PyOD tests.

Conditional collection skip for torch-dependent test modules when
torch is not installed.

Rationale: on macOS CI we deliberately do NOT install PyTorch because
of the upstream NNPACK slowdown on Apple Silicon
(https://github.com/pytorch/pytorch/issues/107534), which makes
conv-heavy torch tests run roughly 36x slower than on Linux or
Windows and exhausts any reasonable GitHub Actions job timeout.
Ubuntu and Windows CI install torch and exercise the full torch test
surface, so the macOS skip is a coverage relocation, not a coverage
regression.

This file is a no-op whenever ``import torch`` succeeds, so it does
not affect local development, Linux CI, Windows CI, or any future
environment where torch imports cleanly. When PyTorch fixes NNPACK
on Apple Silicon upstream, restore the unified install step in
.github/workflows/testing.yml and this file automatically becomes
inert.

TODO: remove both this shim and the macOS-specific install step in
.github/workflows/testing.yml once
https://github.com/pytorch/pytorch/issues/107534 is resolved and a
fixed PyTorch wheel is released.
"""

collect_ignore_glob = []

try:
    import torch  # noqa: F401
except ImportError:
    # Test modules that import torch (or torch_geometric) at module
    # load time. Keep this list in sync with torch-dependent tests.
    collect_ignore_glob = [
        # Deep learning detectors (torch-based)
        "test_auto_encoder.py",
        "test_vae.py",
        "test_deepsvdd.py",
        "test_so_gaal.py",
        "test_so_gaal_new.py",
        "test_mo_gaal.py",
        "test_anogan.py",
        "test_alad.py",
        "test_ae1svm.py",
        "test_devnet.py",
        "test_dif.py",
        "test_lunar.py",
        "test_embedding.py",
        # Shared deep learning machinery
        "test_base_dl.py",
        "test_torch_utility.py",
        # Torch-based time series detectors
        "test_ts_lstm.py",
        "test_ts_anomaly_transformer.py",
        # PyG graph detectors (all require torch_geometric)
        "test_pyg_*.py",
    ]
