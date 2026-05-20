"""Golden-snapshot lock for the legacy ``get_kb_for_routing`` view.

The KB redesign (v3.6) adds soft-KB scores as an opt-in ``scoring_view``.
The default view must stay byte-for-byte identical to the pre-redesign
behavior because the already-published section 5.4 substrate evidence
depends on it. This test captures the default-view output for a fixed set
of routing cases and asserts it never drifts.

Regenerate the golden ONLY from the pre-redesign code:
    python pyod/test/test_kb_routing_golden.py
"""
import json
from pathlib import Path

from pyod.utils.ad_engine import ADEngine

GOLDEN = Path(__file__).resolve().parent / "_kb_routing_golden.json"

CASES = [
    ("tabular_k3", {"data_type": "tabular", "n_samples": 1000, "n_features": 20}, 3, None),
    ("tabular_k5_exclude", {"data_type": "tabular", "n_samples": 5000, "n_features": 50}, 5,
     {"exclude_detectors": ["IForest", "LOF"], "data_type_strict": True}),
    ("tabular_strict_false", {"data_type": "tabular", "n_samples": 800, "n_features": 10}, 3,
     {"data_type_strict": False}),
    ("time_series_k3", {"data_type": "time_series", "n_samples": 10000, "n_features": 1}, 3, None),
    ("graph_k3", {"data_type": "graph", "n_samples": 2000, "n_features": 100}, 3, None),
    ("text_k3", {"data_type": "text", "n_samples": 3000, "n_features": 768}, 3, None),
    ("image_k3", {"data_type": "image", "n_samples": 500, "n_features": 1024}, 3, None),
    ("synthetic_k2", {"data_type": "synthetic", "n_samples": 1000, "n_features": 15}, 2, None),
]


def _run_all():
    eng = ADEngine()
    out = {}
    for name, profile, top_k, constraints in CASES:
        out[name] = eng.get_kb_for_routing(profile, top_k=top_k, constraints=constraints)
    return out


def _ser(obj):
    # sort_keys=False so the snapshot is a byte-identical lock on the legacy
    # view, including dict key order (the v3.6 soft-KB view must not perturb
    # the default view's field order, values, or detector ordering).
    return json.dumps(obj, sort_keys=False, indent=2, default=str)


def test_legacy_get_kb_for_routing_byte_identical():
    assert GOLDEN.exists(), (
        "golden missing; regenerate from pre-redesign code with "
        "`python pyod/test/test_kb_routing_golden.py`")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    current = _run_all()
    assert _ser(current) == _ser(golden), (
        "Default get_kb_for_routing output drifted from the golden snapshot. "
        "The soft-KB view must not perturb the legacy default view.")


if __name__ == "__main__":
    GOLDEN.write_text(_ser(_run_all()), encoding="utf-8")
    print(f"wrote {GOLDEN}")
