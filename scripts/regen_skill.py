#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate KB-derived sections in pyod skill files.

Reads pyod.utils.knowledge.algorithms and rewrites the content between
<!-- BEGIN KB-DERIVED: <section-name> --> and <!-- END KB-DERIVED: <section-name> -->
markers in every *.md file under pyod/skills/.

Hand-written content (everything outside the markers) is left untouched.

Usage:
    python scripts/regen_skill.py              # regenerate in place
    python scripts/regen_skill.py --check      # dry-run; exit 1 if any file would change
    python scripts/regen_skill.py --verbose    # print every regenerated section
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "pyod" / "skills"

# Regex to match a KB-DERIVED block. Captures the section name and the
# old body (everything between the markers). DOTALL so . matches newline.
_BLOCK_RE = re.compile(
    r"<!-- BEGIN KB-DERIVED: ([a-z0-9_\-]+) -->\n(.*?)<!-- END KB-DERIVED: \1 -->",
    re.DOTALL,
)

# Map raw KB requires tokens (Python package names) to user-facing pyproject
# extras. The KB stores the runtime dependency name (e.g. "torch_geometric"),
# but the install hint in the rendered skill must use the extra exposed in
# pyproject.toml (e.g. "graph"). Unknown tokens raise KeyError so the
# maintainer is forced to keep this mapping in sync with pyproject.toml.
_REQUIRES_TO_EXTRA = {
    "torch_geometric": "graph",
    "torch": "torch",
    "xgboost": "xgboost",
}


def _load_kb():
    """Load pyod.utils.knowledge.algorithms once."""
    from pyod.utils.ad_engine import ADEngine
    return ADEngine().kb.algorithms


def _format_complexity(complexity):
    """Format the KB complexity field as a one-line readable string.

    The KB stores complexity as a dict ``{"time": ..., "space": ...}``.
    Older entries may store a plain string. Empty/missing returns "?".
    """
    if not complexity:
        return "?"
    if isinstance(complexity, str):
        return complexity
    if isinstance(complexity, dict):
        time_s = complexity.get("time")
        space_s = complexity.get("space")
        parts = []
        if time_s:
            parts.append(f"time {time_s}")
        if space_s:
            parts.append(f"space {space_s}")
        return ", ".join(parts) if parts else "?"
    return str(complexity)


def _format_paper(paper):
    """Format the KB paper field as a short human-facing reference.

    The KB stores paper as a dict ``{"id": ..., "short": ...}`` where
    ``short`` is e.g. "Liu et al., ICDM 2008". Older entries may be a
    plain string. Returns empty string for missing/empty paper.
    """
    if not paper:
        return ""
    if isinstance(paper, str):
        return paper
    if isinstance(paper, dict):
        return paper.get("short", "") or paper.get("id", "")
    return str(paper)


def _format_requires(requires):
    """Format the KB requires list as comma-joined ``pyod[extra]`` hints.

    Each token must be in ``_REQUIRES_TO_EXTRA``; unknown tokens raise
    KeyError so the maintainer keeps the mapping current. Returns empty
    string for an empty/missing list.
    """
    if not requires:
        return ""
    extras = []
    for req in requires:
        if req not in _REQUIRES_TO_EXTRA:
            raise KeyError(
                f"Unknown KB requires token {req!r}. Add it to "
                f"_REQUIRES_TO_EXTRA in {__file__} (map to the matching "
                f"pyproject.toml extra)."
            )
        extras.append(f"pyod[{_REQUIRES_TO_EXTRA[req]}]")
    return ", ".join(extras)


def _select_algos(kb, modalities):
    """Return deduplicated (name, algo) tuples whose data_types include any modality.

    Used by both single-modality renderers and the combined text/image
    renderer. Iteration is in sorted-name order, and each detector
    appears at most once even if it lives in multiple modalities (e.g.
    ``EmbeddingOD`` is in both ``text`` and ``image``).
    """
    seen = set()
    items = []
    for name, algo in sorted(kb.items()):
        data_types = algo.get("data_types", [])
        if any(m in data_types for m in modalities) and name not in seen:
            seen.add(name)
            items.append((name, algo))
    return items


def _render_bullets(items):
    """Render a list of (name, algo) tuples as a markdown bullet list.

    Each bullet contains: name, full_name, complexity, best_for,
    avoid_when, requires (as ``pyod[extra]``), paper. Fields that are
    empty in the KB are silently omitted.
    """
    if not items:
        return "_No detectors registered._\n"
    lines = []
    for name, algo in items:
        full = algo.get("full_name", name)
        complexity = _format_complexity(algo.get("complexity"))
        best_for = algo.get("best_for", "")
        avoid_when = algo.get("avoid_when", "")
        requires = _format_requires(algo.get("requires", []))
        paper = _format_paper(algo.get("paper"))
        line = f"- **{name}** ({full}) — complexity: {complexity}"
        if best_for:
            line += f"; best for: {best_for}"
        if avoid_when:
            line += f"; avoid when: {avoid_when}"
        if requires:
            line += f"; requires: {requires}"
        if paper:
            line += f"; paper: {paper}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _render_detector_list(kb, modality):
    """Render a markdown bullet list of detectors for a single modality."""
    return _render_bullets(_select_algos(kb, [modality]))


def _render_combined_detector_list(kb, modalities):
    """Render a deduplicated bullet list across multiple modalities."""
    return _render_bullets(_select_algos(kb, modalities))


def _render_total_count(kb):
    """Render a one-line summary of total detector counts by modality."""
    counts = Counter()
    for algo in kb.values():
        for dt in algo.get("data_types", []):
            counts[dt] += 1
    total = len(kb)
    preferred = ["tabular", "time_series", "graph", "text", "image", "multimodal"]
    parts = []
    for key in preferred:
        if counts.get(key):
            label = key.replace("_", "-")
            parts.append(f"{counts[key]} {label}")
    for key, val in counts.items():
        if key not in preferred and val:
            parts.append(f"{val} {key}")
    breakdown = ", ".join(parts) if parts else "none"
    return f"PyOD ships **{total}** detectors total ({breakdown}).\n"


def _render_benchmark_list(kb):
    """Render a deduplicated list of benchmark refs cited in the KB."""
    refs = set()
    for algo in kb.values():
        for ref in algo.get("benchmark_refs", []) or []:
            refs.add(ref)
    if not refs:
        return "_No benchmark refs registered._\n"
    lines = ["Benchmarks referenced by PyOD detectors:"]
    for ref in sorted(refs):
        lines.append(f"- {ref}")
    return "\n".join(lines) + "\n"


# Section name → renderer function
_SECTION_RENDERERS = {
    "tabular-detector-list": lambda kb: _render_detector_list(kb, "tabular"),
    "time-series-detector-list": lambda kb: _render_detector_list(kb, "time_series"),
    "graph-detector-list": lambda kb: _render_detector_list(kb, "graph"),
    "text-image-detector-list": lambda kb: _render_combined_detector_list(
        kb, ["text", "image"]
    ),
    "total-detector-count": _render_total_count,
    "benchmark-list": _render_benchmark_list,
}


def render_section(section_name):
    """Render a named KB-derived section. Raises KeyError on unknown names."""
    renderer = _SECTION_RENDERERS[section_name]
    kb = _load_kb()
    return renderer(kb)


def regen_file(path):
    """Regenerate every KB-DERIVED block in a single file in place.

    Returns True if the file was modified, False if it was already up to date.
    """
    text = path.read_text(encoding="utf-8")
    kb = _load_kb()

    def _replace(match):
        section_name = match.group(1)
        if section_name not in _SECTION_RENDERERS:
            raise KeyError(
                f"Unknown KB-DERIVED section name {section_name!r} in {path}. "
                f"Add it to _SECTION_RENDERERS in {__file__}."
            )
        renderer = _SECTION_RENDERERS[section_name]
        new_body = renderer(kb)
        if not new_body.endswith("\n"):
            new_body += "\n"
        return (
            f"<!-- BEGIN KB-DERIVED: {section_name} -->\n"
            f"{new_body}"
            f"<!-- END KB-DERIVED: {section_name} -->"
        )

    new_text = _BLOCK_RE.sub(_replace, text)
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def find_skill_files():
    """Yield every *.md file under pyod/skills/ recursively."""
    for path in sorted(SKILLS_DIR.rglob("*.md")):
        yield path


def regen_all(verbose=False):
    """Regenerate every skill file. Returns the number of files modified."""
    modified = 0
    for path in find_skill_files():
        if regen_file(path):
            modified += 1
            if verbose:
                rel = path.relative_to(REPO_ROOT)
                print(f"regenerated: {rel}")
    return modified


def check_files(paths):
    """Dry-run: return 0 if no file would change, 1 otherwise.

    Used by --check mode and by tests. Does NOT mutate any file: it reads
    the file, computes what regen_file would write, compares, and reports.
    """
    kb = _load_kb()

    def _replace(match):
        section_name = match.group(1)
        if section_name not in _SECTION_RENDERERS:
            raise KeyError(f"Unknown KB-DERIVED section name {section_name!r}")
        renderer = _SECTION_RENDERERS[section_name]
        new_body = renderer(kb)
        if not new_body.endswith("\n"):
            new_body += "\n"
        return (
            f"<!-- BEGIN KB-DERIVED: {section_name} -->\n"
            f"{new_body}"
            f"<!-- END KB-DERIVED: {section_name} -->"
        )

    diffs = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        new_text = _BLOCK_RE.sub(_replace, text)
        if new_text != text:
            print(f"would regenerate: {path}", file=sys.stderr)
            diffs += 1
    return 1 if diffs else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate KB-derived sections in pyod skill files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: exit 1 if any file would change, 0 otherwise.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every regenerated file path.",
    )
    args = parser.parse_args(argv)

    if args.check:
        rc = check_files(list(find_skill_files()))
        if rc == 0:
            print("All skill files are up to date.")
        return rc

    n = regen_all(verbose=args.verbose)
    print(f"Regenerated {n} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
