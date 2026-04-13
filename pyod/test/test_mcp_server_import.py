# -*- coding: utf-8 -*-
"""Import-safety tests for pyod.mcp_server.

pyod.mcp_server must be importable in a core install that does not
have the optional ``mcp`` extra. The v3.0.0 implementation called
``sys.exit(1)`` at module import time if ``mcp.server.fastmcp`` was
missing, which made ``import pyod.mcp_server`` kill any parent
process that tried to probe MCP availability.
"""
import subprocess
import sys


def test_mcp_server_imports_without_mcp_extra():
    """`import pyod.mcp_server` does not exit the process.

    Runs in a subprocess so we do not need to unload `mcp` from this
    test runner's sys.modules. The subprocess blocks `mcp` from being
    imported by shadowing it with None, then attempts to import
    pyod.mcp_server. Exit code 0 means the module was import-safe.
    """
    script = (
        "import sys\n"
        "sys.modules['mcp'] = None  # block any real mcp import\n"
        "try:\n"
        "    import pyod.mcp_server  # must not sys.exit\n"
        "    print('IMPORTED_OK')\n"
        "except SystemExit as e:\n"
        "    print(f'IMPORT_EXITED_{e.code}')\n"
        "    sys.exit(2)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"import pyod.mcp_server raised SystemExit in a core install: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "IMPORTED_OK" in result.stdout


def test_mcp_server_exposes_main():
    """pyod.mcp_server.main is callable so the unified CLI can delegate."""
    import pyod.mcp_server as m
    assert callable(getattr(m, "main", None)), (
        "pyod.mcp_server.main must be a callable so `pyod mcp serve` "
        "can delegate to it."
    )


def test_mcp_server_main_returns_nonzero_without_mcp_extra():
    """main() must return a non-zero exit code when mcp is missing, not sys.exit at import."""
    script = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"
        "import pyod.mcp_server as m\n"
        "rc = m.main()\n"
        "print(f'RC={rc}')\n"
        "sys.exit(0 if rc != 0 else 3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "RC=" in result.stdout


def test_check_mcp_handles_missing_parent_package():
    """_check_mcp() must return None (not raise) when `mcp` is not installed.

    Regression test for the Round 2 finding that
    ``importlib.util.find_spec("mcp.server.fastmcp")`` raises
    ModuleNotFoundError instead of returning None when the parent
    `mcp` package is not installed. The probe must guard against this.
    """
    script = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"
        "import pyod.mcp_server as m\n"
        "result = m._check_mcp()\n"
        "assert result is None, f'expected None, got {result!r}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"_check_mcp raised or returned non-None when mcp is missing: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_main_registers_all_seven_tools_in_order(monkeypatch):
    """Positive-path test: main() registers the 7 canonical tools in order.

    Uses a fake FastMCP class that records every callable passed through
    `mcp.tool()(fn)` and asserts the full registration sequence. Also
    verifies mcp.run() is invoked exactly once.
    """
    import pyod.mcp_server as m

    registered: list[str] = []
    run_calls = {"n": 0}

    class _FakeMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def _decorator(fn):
                registered.append(fn.__name__)
                return fn
            return _decorator

        def run(self):
            run_calls["n"] += 1

    monkeypatch.setattr(m, "_check_mcp", lambda: _FakeMCP)

    rc = m.main()
    assert rc == 0
    assert run_calls["n"] == 1
    assert registered == [
        "profile_data",
        "plan_detection",
        "build_detector",
        "list_detectors",
        "explain_detector",
        "compare_detectors",
        "get_benchmarks",
    ]
