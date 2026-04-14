#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render examples/agentic_demo.html to docs/figs/agentic-demo.png.

Uses Playwright's headless Chromium to capture a single full-page
screenshot at viewport width 1200px (matching the demo's 1180px max
container width), with the top CAPTURE toolbar hidden via the
``body.capture`` CSS class.

Re-run this any time ``examples/agentic_demo.html`` changes so the
figure at ``docs/figs/agentic-demo.png`` (rendered on readthedocs via
``docs/examples/agentic.rst``) stays in sync with the HTML source.

Prerequisites (one-time):

.. code-block:: bash

    pip install playwright
    playwright install chromium

Usage:

.. code-block:: bash

    python scripts/render_agentic_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "examples" / "agentic_demo.html"
PNG = REPO / "docs" / "figs" / "agentic-demo.png"


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. Run:\n"
            "    pip install playwright\n"
            "    playwright install chromium",
            file=sys.stderr,
        )
        return 1

    if not HTML.is_file():
        print(f"ERROR: {HTML} not found", file=sys.stderr)
        return 1

    PNG.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1200, "height": 900},
            device_scale_factor=2,  # retina-sharp for docs
        )
        page = context.new_page()
        page.goto(HTML.as_uri())
        # Hide the top CAPTURE toolbar so it does not appear in the PNG.
        # The HTML wires this up via the `body.capture` class selector.
        page.evaluate("document.body.classList.add('capture')")
        # Wait for layout and web fonts to settle before capturing.
        page.wait_for_load_state("networkidle")
        # `full_page=True` captures the entire scroll height as a single
        # PNG, so there is no need to stitch multiple screenshots.
        page.screenshot(path=str(PNG), full_page=True)
        browser.close()

    size_kb = PNG.stat().st_size / 1024
    print(f"Wrote {PNG.relative_to(REPO)} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
