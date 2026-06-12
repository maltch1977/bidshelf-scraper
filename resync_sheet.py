#!/usr/bin/env python3
"""
One-shot Sheet resync: pull every successfully-enriched lead from Supabase
and post it to the Google Sheet with the new sales-actionable schema.

Use when the Sheet schema changes or the Sheet gets cleared out. Safe to
re-run — Apollo dedup + the scraper's URL cache will prevent duplicates
on the *source-of-truth* (Supabase) side; you'll just get duplicate rows
in the Sheet if you run it twice.

Run order on the VPS after a Sheet wipe:
    python3 resync_sheet.py        # repost the firms we've already found
    python3 enrich_existing.py     # try the no_match firms again with the latest fixes
"""

import os
import sys
import time

# Reuse scraper's env loading + clients.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from scraper import (  # noqa: E402
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    Supabase,
    post_to_sheet,
)


def main() -> None:
    sb = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/leadgen_scraped_leads"
        "?enrichment_status=eq.found&select=*&order=extracted_at.asc",
        headers=sb.headers,
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    print(f"Found {len(rows)} enriched leads in Supabase.")

    for row in rows:
        print(f"  → {row.get('company_name')} | {row.get('contact_name')} ({row.get('contact_title')})")
        post_to_sheet(row)
        time.sleep(0.3)  # be gentle on the Apps Script webhook

    print("Done.")


if __name__ == "__main__":
    main()
