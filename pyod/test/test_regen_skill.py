# -*- coding: utf-8 -*-
"""Tests for scripts/regen_skill.py — the KB-derived skill content generator."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "regen_skill.py"


def _import_regen_skill():
    """Import the generator script as a module by file path."""
    spec = importlib.util.spec_from_file_location("regen_skill", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_imports_cleanly():
    """`scripts/regen_skill.py` must be importable as a module."""
    assert SCRIPT_PATH.is_file(), f"missing: {SCRIPT_PATH}"
    mod = _import_regen_skill()
    assert hasattr(mod, "render_section"), "render_section function missing"
    assert hasattr(mod, "regen_file"), "regen_file function missing"
    assert hasattr(mod, "main"), "main function missing"


def test_render_section_known_names():
    """Each known section name must render non-empty content from the live KB."""
    mod = _import_regen_skill()
    known_sections = [
        "tabular-detector-list",
        "time-series-detector-list",
        "graph-detector-list",
        "text-image-detector-list",
        "total-detector-count",
        "benchmark-list",
    ]
    for name in known_sections:
        content = mod.render_section(name)
        assert isinstance(content, str), f"{name} must return str"
        assert content.strip(), f"{name} returned empty content"


def test_render_section_unknown_name_raises():
    """Unknown section names must raise KeyError, not return empty."""
    mod = _import_regen_skill()
    try:
        mod.render_section("nonexistent-section")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown section name")


def test_regen_file_replaces_markers_in_place(tmp_path):
    """regen_file replaces content between markers and leaves the rest alone."""
    mod = _import_regen_skill()
    test_file = tmp_path / "sample.md"
    test_file.write_text(
        "# Hand-written header\n"
        "Some hand-written prose.\n"
        "\n"
        "<!-- BEGIN KB-DERIVED: tabular-detector-list -->\n"
        "OLD STALE CONTENT\n"
        "<!-- END KB-DERIVED: tabular-detector-list -->\n"
        "\n"
        "More hand-written prose.\n",
        encoding="utf-8",
    )
    mod.regen_file(test_file)
    new_content = test_file.read_text(encoding="utf-8")
    # Hand-written prose preserved
    assert "Hand-written header" in new_content
    assert "Some hand-written prose." in new_content
    assert "More hand-written prose." in new_content
    # Stale content replaced
    assert "OLD STALE CONTENT" not in new_content
    # Markers preserved
    assert "<!-- BEGIN KB-DERIVED: tabular-detector-list -->" in new_content
    assert "<!-- END KB-DERIVED: tabular-detector-list -->" in new_content
    # New content is non-empty
    begin = new_content.index("<!-- BEGIN KB-DERIVED: tabular-detector-list -->")
    end = new_content.index("<!-- END KB-DERIVED: tabular-detector-list -->")
    section_body = new_content[begin:end]
    assert len(section_body) > 100, "regenerated section body looks too short"


def test_main_check_mode_returns_zero_on_no_diff(tmp_path):
    """main(['--check']) must return 0 when files are already up to date."""
    mod = _import_regen_skill()
    # Create a file, regenerate it once, then run --check on a directory
    # containing only that file. The check should pass.
    test_file = tmp_path / "sample.md"
    test_file.write_text(
        "<!-- BEGIN KB-DERIVED: tabular-detector-list -->\n"
        "<!-- END KB-DERIVED: tabular-detector-list -->\n",
        encoding="utf-8",
    )
    mod.regen_file(test_file)
    # Now check should report no diff
    result = mod.check_files([test_file])
    assert result == 0, f"check_files returned {result}, expected 0 (no diff)"


def test_main_check_mode_returns_nonzero_on_diff(tmp_path):
    """main(['--check']) must return non-zero when files would change."""
    mod = _import_regen_skill()
    test_file = tmp_path / "sample.md"
    test_file.write_text(
        "<!-- BEGIN KB-DERIVED: tabular-detector-list -->\n"
        "STALE CONTENT THAT IS DIFFERENT FROM REGEN\n"
        "<!-- END KB-DERIVED: tabular-detector-list -->\n",
        encoding="utf-8",
    )
    result = mod.check_files([test_file])
    assert result != 0, f"check_files returned {result}, expected non-zero (diff)"


def test_format_complexity_handles_dict_and_string():
    """_format_complexity must render KB dicts as readable strings, not reprs."""
    mod = _import_regen_skill()
    # Dict form (current KB schema)
    assert mod._format_complexity(
        {"time": "O(n log n)", "space": "O(n)"}
    ) == "time O(n log n), space O(n)"
    # String form (legacy fallback)
    assert mod._format_complexity("O(n)") == "O(n)"
    # Empty/missing
    assert mod._format_complexity(None) == "?"
    assert mod._format_complexity("") == "?"
    assert mod._format_complexity({}) == "?"
    # Dict-but-only-time
    assert mod._format_complexity({"time": "O(n)"}) == "time O(n)"


def test_format_paper_handles_dict_and_string():
    """_format_paper must extract a human-facing reference, not render the dict."""
    mod = _import_regen_skill()
    assert mod._format_paper(
        {"id": "iforest", "short": "Liu et al., ICDM 2008"}
    ) == "Liu et al., ICDM 2008"
    assert mod._format_paper("Foo et al. 2024") == "Foo et al. 2024"
    assert mod._format_paper(None) == ""
    assert mod._format_paper({}) == ""
    # Falls back to id when short is missing
    assert mod._format_paper({"id": "myalgo"}) == "myalgo"


def test_format_requires_maps_torch_geometric_to_graph_extra():
    """_format_requires must map raw KB tokens to pyproject extras."""
    mod = _import_regen_skill()
    # The critical mapping: KB has torch_geometric, pyproject exposes pyod[graph]
    assert mod._format_requires(["torch_geometric"]) == "pyod[graph]"
    # Self-mapping tokens still produce pyod[name]
    assert mod._format_requires(["torch"]) == "pyod[torch]"
    assert mod._format_requires(["xgboost"]) == "pyod[xgboost]"
    # Multi-element
    assert mod._format_requires(["torch", "xgboost"]) == "pyod[torch], pyod[xgboost]"
    # Empty
    assert mod._format_requires([]) == ""
    assert mod._format_requires(None) == ""


def test_format_requires_unknown_token_raises():
    """Unknown KB requires tokens must raise KeyError, not silently produce wrong hints."""
    mod = _import_regen_skill()
    try:
        mod._format_requires(["this_package_is_not_mapped"])
    except KeyError:
        return
    raise AssertionError(
        "expected KeyError for unmapped requires token; "
        "_REQUIRES_TO_EXTRA must stay in sync with pyproject.toml"
    )


def test_text_image_section_deduplicates_dual_modality_detectors():
    """`text-image-detector-list` must list EmbeddingOD/MultiModalOD exactly once.

    EmbeddingOD and MultiModalOD live in both `text` and `image` modalities
    in the KB. The naive concatenation approach (text-list + image-list)
    would render them twice. The combined renderer must dedupe.
    """
    mod = _import_regen_skill()
    rendered = mod.render_section("text-image-detector-list")
    # Each name must appear exactly once as a bullet header
    assert rendered.count("**EmbeddingOD**") == 1, (
        f"EmbeddingOD appears {rendered.count('**EmbeddingOD**')} times in:\n{rendered}"
    )
    assert rendered.count("**MultiModalOD**") == 1, (
        f"MultiModalOD appears {rendered.count('**MultiModalOD**')} times in:\n{rendered}"
    )


def test_render_detector_list_does_not_leak_dict_repr():
    """Smoke test: live KB rendering must not produce raw `{'time': ...}` strings."""
    mod = _import_regen_skill()
    for section in ["tabular-detector-list", "graph-detector-list",
                    "time-series-detector-list", "text-image-detector-list"]:
        rendered = mod.render_section(section)
        # If complexity were a raw dict-repr, it would contain "{'time'"
        assert "{'time'" not in rendered, (
            f"{section} contains raw complexity dict repr; _format_complexity broken"
        )
        # If paper were a raw dict-repr, it would contain "{'id'"
        assert "{'id'" not in rendered, (
            f"{section} contains raw paper dict repr; _format_paper broken"
        )


def test_graph_section_uses_pyod_graph_extra_not_torch_geometric():
    """The rendered graph detector list must say `pyod[graph]`, not raw token."""
    mod = _import_regen_skill()
    rendered = mod.render_section("graph-detector-list")
    # At least one graph detector should have a requires hint
    assert "requires:" in rendered, (
        "Expected at least one graph detector to render a 'requires:' line"
    )
    assert "pyod[graph]" in rendered, (
        f"Expected pyod[graph] in graph section, got:\n{rendered}"
    )
    # The raw token must NOT leak
    assert "pyod[torch_geometric]" not in rendered, (
        "Graph detector list leaked the raw torch_geometric token instead of "
        "mapping it to the pyod[graph] extra"
    )
