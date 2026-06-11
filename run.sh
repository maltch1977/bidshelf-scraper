#!/usr/bin/env bash
# Launcher for the BidShelf lead scraper.
# Looks for a venv either alongside this script or one directory up,
# so it works whether the repo is cloned directly into /opt/bidshelf-scraper
# or as a subfolder (e.g. /opt/bidshelf-scraper/app/).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$SCRIPT_DIR/../venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../venv/bin/activate"
else
    echo "FATAL: venv not found in $SCRIPT_DIR or $SCRIPT_DIR/.." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec python3 scraper.py
