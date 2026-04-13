"""PyOD AI-agent skills, packaged as data-only subpackages.

PyOD ships Markdown skill files that Claude Code reads at session start
to enable benchmark-backed anomaly detection through natural conversation.
This subpackage bundles the files as package data; users copy them into
Claude Code's skill discovery directories via the ``pyod-install-skill``
console script.

Two-step install:

.. code-block:: bash

    pip install pyod
    pyod-install-skill              # copies to ~/.claude/skills/od-expert/
    pyod-install-skill --project    # copies to ./skills/od-expert/ instead

Programmatic use:

.. code-block:: python

    from pyod.skills import get_skill_path, install
    from pathlib import Path

    print(get_skill_path("od_expert"))
    install(Path.home() / ".claude" / "skills", skill_name="od-expert")

Note on naming: Python package names cannot contain hyphens, so the
subpackage that actually ships the file is ``pyod.skills.od_expert``
(underscore). Claude Code's skill directory convention uses the
hyphenated name from the skill frontmatter (``name: od-expert``), so
the installer writes to ``<target>/od-expert/SKILL.md`` to match the
canonical on-disk identifier. Both underscore and hyphen forms are
accepted as input to ``install()`` and ``--skill``; they are normalized
to the underscore form for package lookup and then mapped back to the
hyphen form for the install destination.
"""
from __future__ import annotations

import argparse
import importlib.resources
import shutil
import sys
from pathlib import Path

__all__ = ["get_skill_path", "install", "install_cli", "_run_install"]

# Python-package-name → Claude-Code-install-dirname mapping.
# Python package names cannot contain hyphens, but Claude Code's skill
# directory convention uses the hyphenated skill identifier from the
# SKILL.md frontmatter. When the two forms differ, record the mapping
# here so the installer writes to the canonical Claude Code directory.
_INSTALL_DIRNAME_MAP = {
    "od_expert": "od-expert",
}


def _normalize_to_package_name(skill: str) -> str:
    """Normalize user input to the Python subpackage name (underscore form).

    Accepts either the Python subpackage name (``od_expert``) or the
    Claude Code skill directory name (``od-expert``) and returns the
    subpackage name so ``importlib.resources`` can locate it.
    """
    reverse = {v: k for k, v in _INSTALL_DIRNAME_MAP.items()}
    return reverse.get(skill, skill)


def _install_dirname(skill_pkg: str) -> str:
    """Map a Python subpackage name to its Claude Code install dirname."""
    return _INSTALL_DIRNAME_MAP.get(skill_pkg, skill_pkg)


def get_skill_path(skill_name: str = "od_expert") -> Path:
    """Return the filesystem directory containing a packaged skill.

    Parameters
    ----------
    skill_name : str
        Name of the packaged skill. Accepts either the Python subpackage
        name (``"od_expert"``) or the Claude Code skill directory name
        (``"od-expert"``); both resolve to the same packaged directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the skill directory, which contains SKILL.md.
    """
    pkg_name = _normalize_to_package_name(skill_name)
    pkg = f"pyod.skills.{pkg_name}"
    files = importlib.resources.files(pkg)
    return Path(str(files))


def install(target_dir: Path, skill_name: str = "od_expert") -> Path:
    """Copy a packaged skill into a Claude Code skill directory.

    The skill is installed at ``target_dir / <install-dirname> / SKILL.md``,
    where ``<install-dirname>`` is the Claude Code canonical directory name
    for the skill (hyphenated form). If the destination already exists,
    it is overwritten.

    Parameters
    ----------
    target_dir : pathlib.Path
        Directory where Claude Code looks for skills, typically
        ``~/.claude/skills`` (user-global) or ``<project>/skills``
        (project-local).
    skill_name : str
        Name of the packaged skill. Accepts either the Python subpackage
        name (``"od_expert"``) or the Claude Code skill directory name
        (``"od-expert"``).

    Returns
    -------
    pathlib.Path
        Absolute path to the installed SKILL.md.
    """
    pkg_name = _normalize_to_package_name(skill_name)
    source_dir = get_skill_path(pkg_name)
    source_file = source_dir / "SKILL.md"
    if not source_file.is_file():
        raise FileNotFoundError(
            f"Packaged skill not found: {source_file}. "
            f"Reinstalling pyod may fix this."
        )
    install_name = _install_dirname(pkg_name)
    dest_dir = Path(target_dir).expanduser().resolve() / install_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "SKILL.md"
    shutil.copy2(source_file, dest_file)
    return dest_file


def _run_install(
    *,
    target: Path | None,
    project: bool,
    skill: str,
    list_skills: bool,
) -> int:
    """Shared install path for both `pyod install skill` and `pyod-install-skill`.

    Parameters mirror the argparse surface of both CLIs. Returns a shell
    exit code: 0 on success, 1 on a FileNotFoundError from the installer.
    """
    if list_skills:
        print("Available skills:")
        for pkg_name, install_name in _INSTALL_DIRNAME_MAP.items():
            source = get_skill_path(pkg_name) / "SKILL.md"
            marker = "ok" if source.is_file() else "MISSING"
            print(f"  {install_name} ({marker})")
        return 0

    if target is not None:
        resolved_target = target
        mode = "custom"
    elif project:
        resolved_target = Path.cwd() / "skills"
        mode = "project"
    else:
        resolved_target = Path.home() / ".claude" / "skills"
        mode = "user-global"

    try:
        dest = install(resolved_target, skill_name=skill)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    canonical = _install_dirname(_normalize_to_package_name(skill))
    print(f"Installed {canonical} skill to: {dest}")
    if mode == "user-global":
        print(
            "Claude Code will auto-activate this skill on anomaly-detection "
            "intent. Restart your Claude Code session to pick it up."
        )
    elif mode == "project":
        print(
            "The skill is now available as a project-local skill. Claude Code "
            "and Codex running inside this project directory will pick it up "
            "at their next session start."
        )
    else:
        print(
            "If this target is a known agent skill directory, restart the "
            "agent's session to pick up the skill."
        )
    return 0


def install_cli(argv: list[str] | None = None) -> int:
    """Console entry point for `pyod-install-skill` (legacy alias in v3.1.0+)."""
    parser = argparse.ArgumentParser(
        prog="pyod-install-skill",
        description=(
            "Install a pyod skill into Claude Code's skill directory. "
            "Prefer `pyod install skill` in v3.1.0+; this command is kept "
            "as a backward-compat alias."
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Custom target directory for the skill install. "
            "Takes precedence over --project."
        ),
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help=(
            "Install into ./skills/ in the current working directory "
            "instead of ~/.claude/skills/. Use this to enable the skill "
            "only for the project you are currently in."
        ),
    )
    parser.add_argument(
        "--skill",
        default="od-expert",
        help=(
            "Name of the packaged skill to install (default: od-expert). "
            "Both 'od-expert' and 'od_expert' are accepted."
        ),
    )
    parser.add_argument("--list", action="store_true", dest="list_skills")
    args = parser.parse_args(argv)
    return _run_install(
        target=args.target,
        project=args.project,
        skill=args.skill,
        list_skills=args.list_skills,
    )


if __name__ == "__main__":
    sys.exit(install_cli())
