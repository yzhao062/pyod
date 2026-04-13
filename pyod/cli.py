# -*- coding: utf-8 -*-
"""Unified command-line interface for PyOD.

Single entry point for the three agentic activation paths:

- ``pyod install skill``: copy the od-expert Claude Code skill into
  ``~/.claude/skills/od-expert/``.
- ``pyod info``: print version, detector counts, and install state for
  each activation path. Self-diagnostic.
- ``pyod mcp serve``: launch the MCP server (alias for
  ``python -m pyod.mcp_server``).
- ``pyod --help``: show all subcommands.

The legacy ``pyod-install-skill`` console script is kept as a
backward-compat alias. It shares the same ``_run_install`` helper as
``pyod install skill`` so their output is identical.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path


def _cmd_install_skill(args: argparse.Namespace) -> int:
    """Dispatch for `pyod install skill`. Delegates to the shared helper."""
    from pyod.skills import _run_install
    return _run_install(
        target=args.target,
        project=args.project,
        skill=args.skill,
        list_skills=args.list_skills,
    )


def _cmd_info(args: argparse.Namespace) -> int:
    """Dispatch for `pyod info`. Self-diagnostic; returns 0 in core installs."""
    import pyod

    # Real detector counts: iterate the KB instead of hardcoding buckets.
    # The KB stores a `data_types` list per algorithm (note the plural);
    # an algorithm may appear in multiple modalities.
    try:
        from pyod.utils.ad_engine import ADEngine
        engine = ADEngine()
        counts: Counter = Counter()
        for algo in engine.kb.algorithms.values():
            for dt in algo.get("data_types", []):
                counts[dt] += 1
        total = len(engine.kb.algorithms)
        ad_ok = True
    except Exception:
        counts = Counter()
        total = 0
        ad_ok = False

    # Classic API reachable?
    try:
        from pyod.models.iforest import IForest  # noqa: F401
        classic_ok = True
    except Exception:
        classic_ok = False

    # MCP extra availability — probe via find_spec, NEVER import the
    # server module. Importing pyod.mcp_server would execute its
    # module-level FastMCP check, which in v3.0.0 called sys.exit(1)
    # when mcp was missing. Task 1 fixes that upstream, but probing
    # via find_spec is still the right abstraction.
    #
    # Gotcha: find_spec("mcp.server.fastmcp") RAISES ModuleNotFoundError
    # when the parent `mcp` package is missing (it does not return None).
    # Probe the parent first.
    if importlib.util.find_spec("mcp") is None:
        mcp_available = False
    else:
        try:
            mcp_available = importlib.util.find_spec("mcp.server.fastmcp") is not None
        except ModuleNotFoundError:
            mcp_available = False

    # od-expert skill install state — check BOTH install paths that
    # `pyod install skill` supports: the user-global Claude Code
    # directory and the project-local `./skills/` directory used by
    # `--project`. Also detect sibling agent stacks (Codex) so the
    # output gives actionable guidance for non-Claude-Code users.
    # Claude Code has a user-global skill directory at ~/.claude/skills/;
    # Codex does not have an equivalent and instead reads project-local
    # skills from ./skills/<name>/ (the same path used by `--project`).
    claude_dir = Path.home() / ".claude"
    codex_dir = Path.home() / ".codex"
    user_skill_path = claude_dir / "skills" / "od-expert" / "SKILL.md"
    project_skill_path = Path.cwd() / "skills" / "od-expert" / "SKILL.md"
    user_installed = user_skill_path.is_file()
    project_installed = project_skill_path.is_file()
    agents_detected: list[str] = []
    if claude_dir.is_dir():
        agents_detected.append("Claude Code")
    if codex_dir.is_dir():
        agents_detected.append("Codex")

    # --- Output ---
    print(f"PyOD version:          {pyod.__version__}")

    if ad_ok:
        # Render in a stable order; unknown modalities fall through at the end.
        preferred = ["tabular", "time_series", "graph", "text", "image",
                     "multimodal"]
        parts = []
        for key in preferred:
            if counts.get(key):
                label = key.replace("_", "-")
                parts.append(f"{counts[key]} {label}")
        for key, val in counts.items():
            if key not in preferred and val:
                parts.append(f"{val} {key}")
        breakdown = ", ".join(parts) if parts else "none"
        print(f"Detectors (ADEngine):  {total} total ({breakdown})")
    else:
        print("Detectors (ADEngine):  ERROR (ADEngine did not load)")

    print(f"Classic API:           {'OK' if classic_ok else 'ERROR'}")
    print(f"ADEngine (Layer 2):    {'OK' if ad_ok else 'ERROR'}")
    if mcp_available:
        print("MCP extra:             OK (run: pyod mcp serve)")
    else:
        print("MCP extra:             NOT INSTALLED "
              "(install: pip install pyod[mcp])")

    if user_installed and project_installed:
        print(f"od-expert skill:       INSTALLED (user-global) at {user_skill_path}")
        print(f"                       INSTALLED (project)     at {project_skill_path}")
        if agents_detected:
            print(f"                       Detected agents: {', '.join(agents_detected)}")
    elif user_installed:
        print(f"od-expert skill:       INSTALLED (user-global) at {user_skill_path}")
        if "Codex" in agents_detected:
            print("                       Codex detected but does not read the user-global path.")
            print("                       For Codex: run `pyod install skill --project` in the project directory.")
    elif project_installed:
        print(f"od-expert skill:       INSTALLED (project) at {project_skill_path}")
        if agents_detected:
            print(f"                       Active for: {', '.join(agents_detected)}")
        if "Claude Code" in agents_detected:
            print("                       For a user-global Claude Code install, run `pyod install skill`.")
    elif agents_detected:
        print("od-expert skill:       NOT INSTALLED")
        print(f"                       Detected agents: {', '.join(agents_detected)}")
        if "Claude Code" in agents_detected:
            print("                       Claude Code (user-global): run `pyod install skill`")
        if "Codex" in agents_detected:
            print("                       Codex (project-local):     run `pyod install skill --project`")
    else:
        print("od-expert skill:       NOT INSTALLED (no agent stacks detected)")

    return 0


def _cmd_mcp_serve(args: argparse.Namespace) -> int:
    """Dispatch for `pyod mcp serve`. Delegates to `pyod.mcp_server.main`."""
    from pyod import mcp_server
    return mcp_server.main()


def main(argv: list[str] | None = None) -> int:
    """Unified `pyod` CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pyod",
        description=(
            "PyOD 3 command-line interface. Use `pyod <subcommand> --help` "
            "for details on each subcommand."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # pyod install ...
    install_p = sub.add_parser(
        "install", help="Install an activation-path component.",
    )
    # Use a non-overloaded dest name: `skill_p` below also has a
    # `--target` option, and argparse would otherwise write both to
    # `args.target`. The inner `--target` wins in practice, but the
    # collision is a readability trap.
    install_sub = install_p.add_subparsers(dest="install_component", required=True)

    # pyod install skill
    skill_p = install_sub.add_parser(
        "skill",
        help="Copy the od-expert Claude Code skill into a skill directory.",
    )
    skill_p.add_argument(
        "--target", type=Path, default=None,
        help="Custom target directory. Overrides --project.",
    )
    skill_p.add_argument(
        "--project", action="store_true",
        help="Install into ./skills/ in the current working directory.",
    )
    skill_p.add_argument(
        "--skill", default="od-expert",
        help=(
            "Name of the packaged skill to install (default: od-expert). "
            "Both 'od-expert' and 'od_expert' are accepted."
        ),
    )
    skill_p.add_argument(
        "--list", action="store_true", dest="list_skills",
        help="List available packaged skills and exit.",
    )
    skill_p.set_defaults(func=_cmd_install_skill)

    # pyod info
    info_p = sub.add_parser(
        "info",
        help="Print version, detector counts, and install state.",
    )
    info_p.set_defaults(func=_cmd_info)

    # pyod mcp ...
    mcp_p = sub.add_parser("mcp", help="MCP server commands.")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", required=True)
    serve_p = mcp_sub.add_parser("serve", help="Run the PyOD MCP server.")
    serve_p.set_defaults(func=_cmd_mcp_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
