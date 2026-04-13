# -*- coding: utf-8 -*-
"""Tests for the unified pyod CLI."""
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path


def test_pyod_cli_help():
    """Running `pyod --help` lists the three subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "install" in result.stdout
    assert "info" in result.stdout
    assert "mcp" in result.stdout


def test_pyod_info_runs():
    """`pyod info` prints version and detector counts, exit 0 in core install."""
    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "info"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "PyOD version" in result.stdout
    assert "detectors" in result.stdout.lower() or "Detectors" in result.stdout


def test_pyod_info_does_not_exit_without_mcp():
    """`pyod info` must not crash in a core install without the mcp extra.

    Regression test for the Task 1 mcp_server refactor: if
    pyod.mcp_server ever regresses to exiting at import time, `pyod info`
    would inherit the exit and this test would fail.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['mcp'] = None; "
         "from pyod.cli import main; sys.exit(main(['info']))"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"pyod info exited non-zero with mcp blocked: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _isolated_home_env(home: Path) -> dict:
    """Return an env dict with HOME/USERPROFILE pointing at a clean directory.

    Used by tests that need to exercise the ``pyod info`` skill-install-state
    branches without leaking state from the test runner's real home dir
    (which may have ``~/.claude`` or ``~/.codex`` installed for unrelated
    reasons).

    Also injects the repo root into ``PYTHONPATH`` so subprocesses that
    run ``python -m pyod.cli`` from a temp cwd can still import ``pyod``.
    On developer workstations with an editable install this is redundant;
    on CI the workflow does not install pyod, so this is load-bearing.
    """
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)  # Windows equivalent of HOME
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        _REPO_ROOT + os.pathsep + existing_pp if existing_pp else _REPO_ROOT
    )
    return env


def test_pyod_info_project_local_only(tmp_path):
    """`pyod info` distinguishes project-local-only from user-global install.

    Regression test for the expanded skill-install-state rendering in
    `_cmd_info`. Sets up:
      - a fake project-local install at `tmp_path/skills/od-expert/SKILL.md`
      - a clean HOME (no ~/.claude/ or ~/.codex/ under it)
    then runs `pyod info` with `cwd=tmp_path` and asserts the output
    reports the project-local install without claiming a user-global
    install or detecting any agent stack.
    """
    project_skill = tmp_path / "skills" / "od-expert"
    project_skill.mkdir(parents=True)
    (project_skill / "SKILL.md").write_text("---\nname: od-expert\n---\n")

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "info"],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        env=_isolated_home_env(fake_home),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "INSTALLED (project)" in result.stdout
    assert "INSTALLED (user-global)" not in result.stdout
    # No agent dirs in fake_home → no "Active for" or "Detected agents" line
    assert "Detected agents:" not in result.stdout


def test_pyod_info_codex_detected_no_skill(tmp_path):
    """`pyod info` detects Codex via ~/.codex/ and advises --project install.

    Sets up a fake HOME containing only ``~/.codex/`` (no Claude Code, no
    project-local skill). `pyod info` must report Codex as detected and
    recommend ``pyod install skill --project`` rather than the Claude-Code-
    specific user-global command.
    """
    fake_home = tmp_path / "fake_home"
    (fake_home / ".codex").mkdir(parents=True)

    # cwd is a pristine directory with no ./skills/od-expert/ install
    work = tmp_path / "work"
    work.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "info"],
        capture_output=True, text=True,
        cwd=str(work),
        env=_isolated_home_env(fake_home),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "NOT INSTALLED" in result.stdout
    assert "Codex" in result.stdout
    assert "pyod install skill --project" in result.stdout
    # Must NOT fall into the Claude-Code-specific branch
    assert "Claude Code (user-global)" not in result.stdout
    assert "no agent stacks detected" not in result.stdout


def test_pyod_info_codex_and_claude_both_detected(tmp_path):
    """Both ~/.claude/ and ~/.codex/ present, neither skill installed.

    Output must list both agents and show both install commands so the
    user knows which option fits their workflow.
    """
    fake_home = tmp_path / "fake_home"
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".codex").mkdir(parents=True)

    work = tmp_path / "work"
    work.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "info"],
        capture_output=True, text=True,
        cwd=str(work),
        env=_isolated_home_env(fake_home),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "NOT INSTALLED" in result.stdout
    # Pin the exact recommendation lines so a regression that drops one
    # branch cannot pass on a substring match (``pyod install skill`` is
    # a substring of ``pyod install skill --project``).
    assert "Claude Code (user-global): run `pyod install skill`" in result.stdout
    assert "Codex (project-local):" in result.stdout
    assert "`pyod install skill --project`" in result.stdout


def test_pyod_info_user_global_claude_plus_codex_detected(tmp_path):
    """User-global Claude install + Codex detected + no project install.

    The subtle branch flagged in Round 2 review. Claude's user-global
    skill does not help Codex, so the output must recommend `--project`
    for Codex even though Claude Code is already satisfied.
    """
    fake_home = tmp_path / "fake_home"
    # Pre-install od-expert user-globally for Claude Code
    user_skill = fake_home / ".claude" / "skills" / "od-expert"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("---\nname: od-expert\n---\n")
    # Codex also present
    (fake_home / ".codex").mkdir(parents=True)

    # Pristine cwd (no project-local install)
    work = tmp_path / "work"
    work.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "info"],
        capture_output=True, text=True,
        cwd=str(work),
        env=_isolated_home_env(fake_home),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "INSTALLED (user-global)" in result.stdout
    # Must explicitly flag the Codex gap and the --project remedy.
    assert "Codex detected but does not read the user-global path" in result.stdout
    assert "`pyod install skill --project`" in result.stdout


def test_install_skill_project_message_is_agent_neutral(tmp_path):
    """`pyod install skill --project` must not claim Claude-only activation.

    Regression test for the Round 2 finding that `_run_install` hardcoded
    a Claude-specific success message. For a project-local install, the
    output should use agent-neutral wording that covers both Claude Code
    and Codex.

    Uses explicit ``cwd`` + ``_isolated_home_env`` so the subprocess
    imports ``pyod`` via PYTHONPATH even on CI (no editable install).
    """
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "install", "skill", "--project"],
        capture_output=True, text=True,
        cwd=str(work),
        env=_isolated_home_env(fake_home),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert (work / "skills" / "od-expert" / "SKILL.md").is_file()
    # The Claude-only restart hint must NOT appear for project-local installs.
    assert "Restart your Claude Code session" not in result.stdout
    assert "Claude Code will auto-activate" not in result.stdout
    # The agent-neutral wording must appear.
    assert "project-local skill" in result.stdout
    assert "Codex" in result.stdout


def test_pyod_install_skill_to_target(tmp_path):
    """`pyod install skill --target <path>` writes od-expert/SKILL.md."""
    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "install", "skill",
         "--target", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "od-expert" / "SKILL.md").is_file()
    # Output should use the canonical hyphenated name.
    assert "od-expert" in result.stdout


def test_pyod_install_skill_canonical_name_on_underscore_input(tmp_path):
    """Passing `--skill od_expert` still prints the canonical `od-expert`."""
    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "install", "skill",
         "--skill", "od_expert", "--target", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Installed od-expert skill" in result.stdout
    assert "Installed od_expert skill" not in result.stdout


def test_pyod_install_skill_list():
    """`pyod install skill --list` prints available skills with canonical names."""
    result = subprocess.run(
        [sys.executable, "-m", "pyod.cli", "install", "skill", "--list"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "od-expert" in result.stdout


def test_legacy_and_unified_install_match_in_process(tmp_path):
    """`pyod-install-skill ...` and `pyod install skill ...` match in-process.

    Compares return code, stdout (with the path line scrubbed), and
    stderr. An entry-point regression that changes exit behavior would
    fail here.
    """
    from pyod.skills import install_cli
    from pyod.cli import main as cli_main

    tmp_legacy = tmp_path / "legacy"
    tmp_unified = tmp_path / "unified"

    buf_legacy_out = io.StringIO()
    buf_legacy_err = io.StringIO()
    with contextlib.redirect_stdout(buf_legacy_out), \
         contextlib.redirect_stderr(buf_legacy_err):
        rc_legacy = install_cli(["--target", str(tmp_legacy)])

    buf_unified_out = io.StringIO()
    buf_unified_err = io.StringIO()
    with contextlib.redirect_stdout(buf_unified_out), \
         contextlib.redirect_stderr(buf_unified_err):
        rc_unified = cli_main(["install", "skill", "--target", str(tmp_unified)])

    assert rc_legacy == 0
    assert rc_unified == 0
    assert rc_legacy == rc_unified

    def _scrub(text):
        return "\n".join(
            line for line in text.splitlines()
            if "Installed od-expert skill to:" not in line
        )

    assert _scrub(buf_legacy_out.getvalue()) == _scrub(buf_unified_out.getvalue())
    assert buf_legacy_err.getvalue() == buf_unified_err.getvalue()


def test_legacy_and_unified_install_match_subprocess(tmp_path):
    """Subprocess parity test: real console scripts produce matching output.

    Runs the real `pyod` and `pyod-install-skill` commands through the
    entry-point shims rather than importing the functions directly.
    This catches wiring regressions the in-process test would miss
    (e.g., a console_scripts entry pointing at the wrong function).
    Skipped if either executable is not on PATH.
    """
    import shutil

    pyod_exe = shutil.which("pyod")
    legacy_exe = shutil.which("pyod-install-skill")
    if not pyod_exe or not legacy_exe:
        import pytest
        pytest.skip("pyod and/or pyod-install-skill not on PATH (editable install not wired up)")

    tmp_legacy = tmp_path / "legacy"
    tmp_unified = tmp_path / "unified"

    r_legacy = subprocess.run(
        [legacy_exe, "--target", str(tmp_legacy)],
        capture_output=True, text=True,
    )
    r_unified = subprocess.run(
        [pyod_exe, "install", "skill", "--target", str(tmp_unified)],
        capture_output=True, text=True,
    )

    assert r_legacy.returncode == 0, r_legacy.stderr
    assert r_unified.returncode == 0, r_unified.stderr
    assert r_legacy.returncode == r_unified.returncode

    def _scrub(text):
        return "\n".join(
            line for line in text.splitlines()
            if "Installed od-expert skill to:" not in line
        )

    assert _scrub(r_legacy.stdout) == _scrub(r_unified.stdout)
    assert r_legacy.stderr == r_unified.stderr
    assert (tmp_legacy / "od-expert" / "SKILL.md").is_file()
    assert (tmp_unified / "od-expert" / "SKILL.md").is_file()
