"""Tests for the opt-in soft-KB view of get_kb_for_routing (v3.6 step 5).

The legacy default view is locked separately by test_kb_routing_golden.py.
These tests cover the new scoring_view="soft_kb" branch: field attachment,
score-ordered sorting, within-modality semantics, top-level metadata,
graceful degradation when the score artifact is absent, and that the
default view is unperturbed at the field level.
"""
import pytest

from pyod.utils.ad_engine import ADEngine

PROFILE_TAB = {"data_type": "tabular", "n_samples": 1000, "n_features": 20}
PROFILE_TS = {"data_type": "time_series", "n_samples": 10000, "n_features": 1}
PROFILE_IMG = {"data_type": "image", "n_samples": 5000, "n_features": 0}

KB_FIELDS = {"kb_score", "kb_score_rank", "kb_interval", "kb_coverage_n",
             "kb_eligible_n", "kb_failure_n", "kb_missing_n", "kb_effective_n",
             "kb_metric_scope", "kb_source_benchmark", "kb_fallback_level"}


def _eng():
    return ADEngine()


def test_legacy_default_has_no_soft_kb_fields():
    out = _eng().get_kb_for_routing(PROFILE_TAB, top_k=3)
    assert "kb_version" not in out
    assert "kb_scoring_method" not in out
    for d in out["available_detectors"]:
        assert KB_FIELDS.isdisjoint(d.keys()), "legacy view leaked soft-KB fields"


def test_legacy_explicit_equals_default():
    eng = _eng()
    a = eng.get_kb_for_routing(PROFILE_TAB, top_k=3)
    b = eng.get_kb_for_routing(PROFILE_TAB, top_k=3, scoring_view="legacy")
    assert a == b


def test_soft_kb_attaches_fields_and_meta():
    out = _eng().get_kb_for_routing(PROFILE_TAB, top_k=3, scoring_view="soft_kb")
    assert out.get("kb_version") and out["kb_version"] != "unknown"
    assert out.get("kb_scoring_method") and out["kb_scoring_method"] != "unknown"
    dets = out["available_detectors"]
    assert dets, "expected tabular detectors"
    for d in dets:
        assert KB_FIELDS.issubset(d.keys()), "soft-KB fields not attached"
    assert any(d["kb_score"] is not None for d in dets), "no tabular detector scored"


def test_soft_kb_sorted_by_score_desc_nulls_last():
    dets = _eng().get_kb_for_routing(
        PROFILE_TAB, top_k=5, scoring_view="soft_kb")["available_detectors"]
    scored = [d["kb_score"] for d in dets if d["kb_score"] is not None]
    assert scored == sorted(scored, reverse=True), "scored detectors not descending"
    seen_null = False
    for d in dets:
        if d["kb_score"] is None:
            seen_null = True
        elif seen_null:
            pytest.fail("a scored detector appeared after an unscored one")


def test_soft_kb_score_rank_is_dense_and_null_for_unscored():
    dets = _eng().get_kb_for_routing(
        PROFILE_TAB, top_k=5, scoring_view="soft_kb")["available_detectors"]
    scored = [d for d in dets if d["kb_score"] is not None]
    # 1-based, contiguous, matching the post-sort (descending) order.
    assert [d["kb_score_rank"] for d in scored] == list(range(1, len(scored) + 1))
    for d in dets:
        if d["kb_score"] is None:
            assert d["kb_score_rank"] is None, "unscored detector got a rank"


def test_soft_kb_within_modality_only():
    dets = _eng().get_kb_for_routing(
        PROFILE_TAB, top_k=3, scoring_view="soft_kb")["available_detectors"]
    scored = [d for d in dets if d["kb_score"] is not None]
    assert scored, "expected at least one scored tabular detector"
    for d in scored:
        assert d["kb_fallback_level"] == "modality"
        assert d["kb_source_benchmark"], "scored detector missing source benchmark"
        assert 0.0 <= d["kb_score"] <= 1.0, "kb_score outside [0,1]"


def test_soft_kb_timeseries_modality():
    out = _eng().get_kb_for_routing(PROFILE_TS, top_k=3, scoring_view="soft_kb")
    scored = [d for d in out["available_detectors"] if d["kb_score"] is not None]
    assert scored, "expected at least one scored time-series detector"


def test_soft_kb_image_modality_all_null():
    # Image is a valid modality with detectors but no scores in kb_scores.json,
    # so the soft view must attach all-null fields (never a cross-modality score).
    out = _eng().get_kb_for_routing(PROFILE_IMG, top_k=3, scoring_view="soft_kb")
    assert out["kb_version"] != "unknown", "KB present; image is just unscored"
    dets = out["available_detectors"]
    assert dets, "expected image detectors"
    for d in dets:
        assert KB_FIELDS.issubset(d.keys())
        assert d["kb_score"] is None, "image detector should not be scored"
        assert d["kb_score_rank"] is None
        assert d["kb_fallback_level"] is None


def test_soft_kb_degrades_when_scores_file_absent(monkeypatch):
    # If _raw/kb_scores.json is missing (e.g., a wheel that failed to ship it),
    # the soft view must return a valid all-null structure, not crash.
    from pyod.utils.knowledge import KnowledgeBase
    orig = KnowledgeBase._load_json

    def _maybe_raise(self, filename):
        if "kb_scores" in str(filename):
            raise FileNotFoundError(filename)
        return orig(self, filename)

    monkeypatch.setattr(KnowledgeBase, "_load_json", _maybe_raise)
    out = ADEngine().get_kb_for_routing(
        PROFILE_TAB, top_k=3, scoring_view="soft_kb")
    assert out["kb_version"] == "unknown"
    assert out["kb_scoring_method"] == "unknown"
    dets = out["available_detectors"]
    assert dets, "available detectors should still be listed without scores"
    for d in dets:
        assert KB_FIELDS.issubset(d.keys()), "soft-KB fields must still attach"
        assert d["kb_score"] is None
        assert d["kb_score_rank"] is None
        assert d["kb_interval"] is None
        assert d["kb_coverage_n"] == 0
        assert d["kb_fallback_level"] is None


def test_invalid_scoring_view_raises():
    with pytest.raises(ValueError):
        _eng().get_kb_for_routing(PROFILE_TAB, scoring_view="bogus")
