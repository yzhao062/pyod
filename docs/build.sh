#!/bin/bash
# Build PyOD docs locally.
#
# Usage:
#   bash docs/build.sh          # Build once
#   bash docs/build.sh clean    # Clean build
#   bash docs/build.sh watch    # Auto-rebuild on change (requires sphinx-autobuild)
#
# Dependencies (install once):
#   pip install sphinxcontrib-bibtex furo sphinx-rtd-theme sphinx-autobuild

set -e
cd "$(dirname "$0")"

MODE="${1:-build}"

case "$MODE" in
    clean)
        rm -rf _build
        echo "Cleaned _build/"
        ;;
    watch)
        exec sphinx-autobuild . _build/html --open-browser --port 8000
        ;;
    build|*)
        python -m sphinx -b html . _build/html
        echo
        echo "Built docs. Open: docs/_build/html/index.html"
        ;;
esac
