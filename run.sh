#!/usr/bin/env bash
# Launcher for the BidShelf lead scraper on the VPS.
# Cron calls this. It activates the venv and runs scraper.py with logging.
set -e
cd "$(dirname "$0")"
source venv/bin/activate
exec python3 scraper.py
