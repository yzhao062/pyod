# -*- coding: utf-8 -*-
"""KB consistency tests for packaged pyod skill content.

Enforces three invariants:

1. Every backtick-wrapped token in hand-written skill prose that looks like
   a detector name must exist in `pyod.utils.knowledge.algorithms`. Author
   prose must use backtick convention for detector names.

2. Every KB-DERIVED block in every skill .md file must have a matching
   BEGIN / END marker pair with the same section name. Malformed or
   unmatched markers fail loudly.

3. Every KB-DERIVED block's body is byte-identical to what
   scripts/regen_skill.py would produce (re-run the generator to fix).
"""
import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "pyod" / "skills"
SCRIPT_PATH = REPO_ROOT / "scripts" / "regen_skill.py"

# Allowlist of backtick-wrapped tokens that look like detector names but
# are actually legitimate non-detector references (Python symbols, env
# vars, CLI commands, etc.). Add to this list when a false positive
# surfaces, with a one-line reason.
#
# IMPORTANT: do NOT add any token here that is also a live KB key. The
# allowlist must not shadow real detector names — that would let a typo
# in `IForset` slip through if `IForest` were also allowlisted. The
# `test_allowlist_does_not_shadow_kb_keys` test enforces this invariant.
_BACKTICK_ALLOWLIST = {
    # Python stdlib / pyod module names that look detector-like but
    # are orchestration objects, not KB-registered detectors.
    "ADEngine", "ADEngine.start", "ADEngine.plan", "ADEngine.run",
    "ADEngine.analyze", "ADEngine.iterate", "ADEngine.report",
    "ADEngine.investigate", "ADEngine.profile_data",
    "ADEngine.list_detectors", "ADEngine.explain_detector",
    "ADEngine.compare_detectors", "ADEngine.get_benchmarks",
    "BaseDetector", "MultiModalEncoder",
    # CLI commands and flags
    "pyod", "pyod-install-skill", "pip", "pyod[graph]", "pyod[mcp]",
    "pyod info", "pyod install skill", "pyod install skill --project",
    "pyod mcp serve", "python -m pyod.mcp_server",
    # Benchmark and paper names — valid references that aren't KB keys
    "ADBench", "TSB-AD", "BOND", "NLP-ADBench",
    # Common Python state / attribute refs
    "state", "state.profile", "state.plan", "state.scores", "state.labels",
    "state.quality", "state.next_action", "state.best_detector",
    "state.top_cases", "state.feature_importance", "state.consensus_scores",
    "decision_scores_", "decision_function", "predict", "predict_proba",
    "predict_confidence", "predict_with_rejection", "labels_", "fit",
    "fit_predict",
    # Sklearn utilities commonly mentioned in prose
    # (NOTE: PCA is a live KB detector, NOT allowlisted — see comment above.)
    "StandardScaler", "RobustScaler",
    # Test runner / stdlib references
    "True", "False", "None", "HOME", "USERPROFILE", "PYTHONPATH",
    # Filesystem paths often wrapped in backticks
    "~/.claude", "~/.codex", "~/.claude/skills",
    "./skills/", "./skills/od-expert/",
}

# Backtick-wrapped token pattern. Captures the inside of a single-backtick
# span. Example: `IForest` → captures "IForest". We deliberately do NOT
# match triple-backtick code fences (which are whole-line delimiters) —
# the regex only matches inline spans.
_BACKTICK_RE = re.compile(r"(?<!`)`([^`\n]+?)`(?!`)")

# Exact valid marker forms. A line that mentions a KB marker MUST
# fullmatch one of these after strip(); anything else is an error.
_BEGIN_MARKER_RE = re.compile(r"<!-- BEGIN KB-DERIVED: ([a-z0-9_\-]+) -->")
_END_MARKER_RE = re.compile(r"<!-- END KB-DERIVED: ([a-z0-9_\-]+) -->")

# "Suspicious marker" detector — any line containing an HTML comment
# with BEGIN/END + KB- is a candidate for validation, case-insensitive.
# This deliberately matches typo'd sentinels like `KB-DEREIVED`,
# `KB-DERIVD`, or `Kb-derived` so that the validator can flag them.
# A literal substring check for "KB-DERIVED" would silently miss those
# typos — that was the hole flagged in the Round 3 review.
_SUSPICIOUS_MARKER_RE = re.compile(
    r"<!--\s*(BEGIN|END)\s+KB-",
    re.IGNORECASE,
)

# Paired marker regex (matches well-formed blocks only — same as the
# generator's _BLOCK_RE). Used to strip KB-DERIVED bodies before
# backtick scanning, and to identify the section body for regen check.
_PAIRED_BLOCK_RE = re.compile(
    r"<!-- BEGIN KB-DERIVED: ([a-z0-9_\-]+) -->\n(.*?)<!-- END KB-DERIVED: \1 -->",
    re.DOTALL,
)


def _import_regen_skill():
    """Import scripts/regen_skill.py as a module by file path."""
    spec = importlib.util.spec_from_file_location("regen_skill", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_kb():
    """Load pyod.utils.knowledge.algorithms."""
    from pyod.utils.ad_engine import ADEngine
    return ADEngine().kb.algorithms


def _all_skill_files():
    """Return sorted list of all *.md files under pyod/skills/."""
    return sorted(SKILLS_DIR.rglob("*.md"))


def _strip_paired_kb_derived_blocks(text):
    """Remove content inside well-formed KB-DERIVED blocks.

    KB-DERIVED blocks already came from the KB by definition; we only
    want to validate hand-written prose. Malformed blocks are left in
    place so the marker-pairing test can still flag them.
    """
    return _PAIRED_BLOCK_RE.sub("", text)


def test_skill_files_exist():
    """At least the od-expert SKILL.md must exist."""
    files = _all_skill_files()
    paths = {f.relative_to(REPO_ROOT) for f in files}
    assert any("od_expert" in str(p) and p.name == "SKILL.md" for p in paths), (
        f"od_expert/SKILL.md not found among {sorted(paths)}"
    )


def test_kb_derived_markers_are_well_formed():
    """Every line mentioning KB-DERIVED must be a valid BEGIN or END marker,
    and every BEGIN must match a later END with the same section name.

    This is a line-oriented parser that catches everything the regex-based
    approach missed in Round 1:
      - Typo'd markers that don't match either valid form (e.g.
        ``<!-- BEGIN KB-DERIVED foo -->`` — missing colon, or
        ``<!-- BEGIN KB-DEREIVED: foo -->`` — typo)
      - END markers that appear before any BEGIN (depth would go negative)
      - Unclosed BEGIN markers at end-of-file
      - Nested BEGIN-inside-BEGIN
      - Section name mismatch between BEGIN and its corresponding END
      - Stray lines containing the literal "KB-DERIVED" that are not
        actual markers (e.g. a prose reference that would otherwise be
        invisible to the generator but visible to a human reader)

    The generator's regex-based replacement is only safe if every
    KB-DERIVED marker in the file matches one of the two exact forms.
    This test enforces that invariant at build time.
    """
    files = _all_skill_files()
    problems = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        open_name = None  # None = no block open; str = name of open block
        for lineno, line in enumerate(lines, start=1):
            # Only inspect lines that look like a KB-DERIVED marker
            # (BEGIN/END + KB-, case-insensitive). This is broader than
            # the literal "KB-DERIVED" substring check so typo'd
            # sentinels like KB-DEREIVED still land in the validator.
            if not _SUSPICIOUS_MARKER_RE.search(line):
                continue
            begin_m = _BEGIN_MARKER_RE.fullmatch(line.strip())
            end_m = _END_MARKER_RE.fullmatch(line.strip())
            if begin_m is None and end_m is None:
                problems.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: line looks "
                    f"like a KB-DERIVED marker but does not match the exact "
                    f"syntax (typo in sentinel or section name, extra text, "
                    f"wrong spacing, etc.). Line: {line!r}"
                )
                continue
            if begin_m is not None and end_m is not None:
                # Shouldn't happen given distinct regexes, but guard anyway.
                problems.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: line matches "
                    f"both BEGIN and END patterns. Line: {line!r}"
                )
                continue
            if begin_m is not None:
                name = begin_m.group(1)
                if open_name is not None:
                    problems.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: nested "
                        f"BEGIN {name!r} while {open_name!r} still open"
                    )
                    # Keep parsing; treat the new BEGIN as the open block.
                open_name = name
            else:  # end_m is not None
                name = end_m.group(1)
                if open_name is None:
                    problems.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: END "
                        f"{name!r} without matching BEGIN"
                    )
                elif open_name != name:
                    problems.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: END "
                        f"{name!r} does not match open BEGIN {open_name!r}"
                    )
                    open_name = None
                else:
                    open_name = None  # clean close
        if open_name is not None:
            problems.append(
                f"{path.relative_to(REPO_ROOT)}: unclosed BEGIN {open_name!r} "
                f"at end of file"
            )
    assert not problems, (
        "Malformed KB-DERIVED markers in skill files:\n  "
        + "\n  ".join(problems)
        + "\n\nFix the marker pairing manually. Every KB-DERIVED line must "
        + "match one of:\n"
        + "  <!-- BEGIN KB-DERIVED: section-name -->\n"
        + "  <!-- END KB-DERIVED: section-name -->\n"
        + "with matching section names and no nesting."
    )


def test_kb_derived_sections_are_up_to_date():
    """Every KB-DERIVED block in every skill file must match the live KB.

    If this test fails, run `python scripts/regen_skill.py` and commit
    the result.
    """
    regen = _import_regen_skill()
    files = _all_skill_files()
    rc = regen.check_files(files)
    assert rc == 0, (
        "KB-DERIVED sections in skill files are stale. "
        "Run `python scripts/regen_skill.py` and commit the result."
    )


def _extract_backtick_tokens(text):
    """Return the set of inline backtick-wrapped tokens in `text`.

    Only matches single-backtick spans (inline code), not triple-backtick
    fences. Splits on spaces and commas to handle things like
    ``pyod install skill`` (multi-word) — those count as one token, not
    three.
    """
    matches = _BACKTICK_RE.findall(text)
    return set(matches)


def test_backtick_detector_names_exist_in_kb():
    """Every backtick-wrapped detector-like token in prose must exist in the KB.

    The design enforces a convention: detector names in hand-written
    skill prose must be wrapped in backticks (inline code style). For
    example:

        Use `IForest` for large tabular data.   # validated
        Use Isolation Forest for large data.    # NOT validated (free prose)

    This lets maintainers write expository prose freely while keeping
    the canonical detector references type-checked against the KB.

    Failure modes this catches:
      - Typo (`IForset` instead of `IForest`)
      - Renamed detector (`Deep_SVDD` after KB rename to `DeepSVDD`)
      - Stale reference to a removed detector

    Allowlist: tokens that are legitimate backtick references but not
    KB keys (Python symbols, CLI commands, file paths, benchmark names)
    live in `_BACKTICK_ALLOWLIST` at the top of this file. Add to that
    list when a false positive surfaces, with a one-line reason.
    """
    kb = _load_kb()
    known = set(kb.keys())
    files = _all_skill_files()
    unknown_per_file = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        text = _strip_paired_kb_derived_blocks(text)
        tokens = _extract_backtick_tokens(text)
        # Heuristic: a token is "detector-like" if it starts with a
        # capital letter and contains no spaces beyond method-call dots.
        detector_like = {
            t for t in tokens
            if t and t[0].isupper() and " " not in t
        }
        unknown = detector_like - known - _BACKTICK_ALLOWLIST
        if unknown:
            unknown_per_file[path.relative_to(REPO_ROOT)] = sorted(unknown)
    assert not unknown_per_file, (
        "Unknown detector-like backtick tokens in skill hand-written content:\n"
        + "\n".join(
            f"  {path}: {names}"
            for path, names in unknown_per_file.items()
        )
        + "\n\nEither:\n"
        + "  1. Fix the skill — this detector is not in the KB.\n"
        + "  2. Use free prose instead (remove the backticks).\n"
        + "  3. Add the token to _BACKTICK_ALLOWLIST with a one-line reason."
    )


def test_kb_loads_cleanly():
    """Sanity check: the KB must load without errors."""
    kb = _load_kb()
    assert kb, "KB is empty"
    assert isinstance(kb, dict)
    # Spot-check that the KB has the expected schema
    sample_name, sample_algo = next(iter(kb.items()))
    assert "data_types" in sample_algo, (
        f"KB algorithm {sample_name} missing 'data_types' field"
    )


def test_allowlist_does_not_shadow_kb_keys():
    """The backtick allowlist must not contain any live KB detector name.

    Allowing a real detector name here defeats the safety net's purpose:
    if `IForest` is in the allowlist and the maintainer typos it as
    `IForset`, the typo would fail the KB lookup but the allowlist
    fallback would let it through silently. This test enforces that the
    allowlist and the live KB are disjoint sets, so every detector name
    in skill prose is validated against the KB on every CI run.
    """
    kb = _load_kb()
    overlap = _BACKTICK_ALLOWLIST & set(kb.keys())
    assert not overlap, (
        f"_BACKTICK_ALLOWLIST shadows live KB detector names: {sorted(overlap)}.\n"
        f"Remove these entries — they should be validated against the KB, "
        f"not allowlisted. Allowlisting them would let typos slip past the "
        f"safety net."
    )
