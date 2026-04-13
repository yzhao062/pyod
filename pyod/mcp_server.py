# -*- coding: utf-8 -*-
"""PyOD MCP Server: Agent interface for anomaly detection.

Exposes PyOD's ADEngine as MCP tools that any LLM agent can call.
Tier A: knowledge queries + stateless planning.

Usage:
    python -m pyod.mcp_server

Note: On Windows with antivirus software (e.g., Bitdefender), the MCP
server subprocess may be blocked. If MCP is unavailable, use ADEngine
directly in Python: ``from pyod.utils.ad_engine import ADEngine``.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import annotations

import json
import keyword
import os
import sys


def _check_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError:
        print("MCP server requires the 'mcp' package. "
              "Install with: pip install pyod[mcp]",
              file=sys.stderr)
        sys.exit(1)


FastMCP = _check_mcp()

from pyod.utils.ad_engine import ADEngine

mcp = FastMCP("pyod")
engine = ADEngine()


def _to_json(obj):
    """Serialize result to JSON string."""
    return json.dumps(obj, indent=2, default=str)


@mcp.tool()
def profile_data(data_path: str, data_type: str = "auto") -> str:
    """Profile a dataset for anomaly detection.

    Loads data from path, detects data type and characteristics.
    Returns a JSON profile for use with plan_detection().

    Args:
        data_path: Path to data file (CSV, NPY, JSON).
        data_type: Override. One of 'tabular', 'text', 'image', or 'auto'.
    """
    X = _load_data(data_path)
    dt = None if data_type == "auto" else data_type
    return _to_json(engine.profile_data(X, data_type=dt))


@mcp.tool()
def plan_detection(
    data_profile: str,
    priority: str = "balanced",
    constraints: str = ""
) -> str:
    """Plan an anomaly detection pipeline.

    Returns a DetectionPlan with detector, params, reason, and evidence.

    Args:
        data_profile: JSON string from profile_data().
        priority: 'speed', 'accuracy', or 'balanced'.
        constraints: Optional JSON, e.g. '{"exclude_detectors": ["ECOD"]}'.
    """
    try:
        profile = json.loads(data_profile)
    except (json.JSONDecodeError, TypeError) as e:
        return _to_json({"error": "Invalid JSON", "details": str(e)})
    if not isinstance(profile, dict):
        return _to_json({"error": "data_profile must be a JSON object"})
    try:
        cons = json.loads(constraints) if constraints else None
    except (json.JSONDecodeError, TypeError) as e:
        return _to_json({"error": "Invalid JSON", "details": str(e)})
    if cons is not None and not isinstance(cons, dict):
        return _to_json({"error": "constraints must be a JSON object"})
    return _to_json(engine.plan_detection(profile, priority, cons))


@mcp.tool()
def build_detector(plan: str) -> str:
    """Get constructor metadata for a detector from a plan.

    Returns import path, class name, params, and a Python code
    snippet for instantiation. Params are passed through from
    the plan without constructor signature validation.

    Args:
        plan: JSON string from plan_detection().
    """
    try:
        plan_dict = json.loads(plan)
    except (json.JSONDecodeError, TypeError) as e:
        return _to_json({"error": "Invalid JSON", "details": str(e)})
    if not isinstance(plan_dict, dict):
        return _to_json({"error": "plan must be a JSON object"})
    name = plan_dict.get('detector_name', '')
    algo = engine.kb.get_algorithm(name)
    if algo is None:
        return _to_json({"error": "Unknown detector", "name": name})

    preset = plan_dict.get('preset')
    params = plan_dict.get('params', {})
    if not isinstance(params, dict):
        return _to_json({"error": "params must be a JSON object"})

    # Validate preset is only used with EmbeddingOD
    if preset and name != 'EmbeddingOD':
        return _to_json({"error": "Preset only valid for EmbeddingOD",
                         "detector": name, "preset": preset})

    # Validate preset against known allowlist
    _VALID_PRESETS = {'for_text', 'for_image'}
    if preset and preset not in _VALID_PRESETS:
        return _to_json({"error": "Unknown preset", "preset": preset})

    # Validate param keys are simple identifiers (no injection)
    for key in params:
        if not key.isidentifier() or keyword.iskeyword(key):
            return _to_json({"error": "Invalid parameter name", "key": key})

    if preset:
        code = "from pyod.models.embedding import EmbeddingOD\n"
        param_str = ', '.join('%s=%r' % (k, v) for k, v in params.items())
        code += "clf = EmbeddingOD.%s(%s)" % (preset, param_str)
    else:
        class_path = algo['class_path']
        module_path, class_name = class_path.rsplit('.', 1)
        code = "from %s import %s\n" % (module_path, class_name)
        if params:
            param_str = ', '.join('%s=%r' % (k, v) for k, v in params.items())
            code += "clf = %s(%s)" % (class_name, param_str)
        else:
            code += "clf = %s()" % class_name

    return _to_json({
        "detector_name": name,
        "class_path": algo['class_path'],
        "params": params,
        "preset": preset,
        "code_snippet": code,
    })


@mcp.tool()
def list_detectors(data_type: str = "", status: str = "shipped") -> str:
    """List available PyOD detectors.

    Args:
        data_type: Filter by data type (tabular, text, image, etc.).
        status: Filter by status (shipped, planned, all).
    """
    return _to_json(engine.list_detectors(
        data_type=data_type or None, status=status))


@mcp.tool()
def explain_detector(name: str) -> str:
    """Explain a PyOD detector: how it works, strengths, weaknesses,
    benchmark performance, and recommended use cases."""
    try:
        return _to_json(engine.explain_detector(name))
    except ValueError as e:
        return _to_json({"error": str(e)})


@mcp.tool()
def compare_detectors(
    names: str = "",
    data_type: str = "tabular",
    top_k: int = 3
) -> str:
    """Compare detectors for a given data type.

    Args:
        names: Comma-separated detector names. If empty, top-k for type.
        data_type: Data type to compare for.
        top_k: Number of top detectors.
    """
    name_list = [n.strip() for n in names.split(',')] if names else None
    return _to_json(engine.compare_detectors(name_list, data_type, top_k))


@mcp.tool()
def get_benchmarks(benchmark: str = "all") -> str:
    """Get benchmark results (ADBench, NLP-ADBench, TSB-AD)."""
    return _to_json(engine.get_benchmarks(benchmark))


def _load_data(path):
    """Load data from file path."""
    import numpy as np

    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, allow_pickle=False)
    elif ext == '.npz':
        data = np.load(path, allow_pickle=False)
        return data[data.files[0]]
    elif ext == '.csv':
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        # Return features only (last column is label if present)
        if data.ndim == 2 and data.shape[1] > 1:
            return data[:, :-1]
        return data
    elif ext == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif ext == '.mat':
        from scipy.io import loadmat
        data = loadmat(path)
        if 'X' in data:
            return data['X']
        for key in data:
            if not key.startswith('_'):
                return data[key]
    else:
        raise ValueError("Unsupported file format: %s" % ext)


if __name__ == "__main__":
    mcp.run()
