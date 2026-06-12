#!/usr/bin/env python3
"""
One-shot enrichment for existing leadgen_scraped_leads rows.

The original 9 leads were inserted before Consulti enrichment was wired in,
so they have null enrichment_status and don't carry contact info. This script:
  - fetches all rows with enrichment_status IS NULL
  - calls Consulti to find a senior contact at each firm
  - updates the row in Supabase
  - posts qualifying leads (with a contact) to the Google Sheet

Run once after deploying the scraper update.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Re-use scraper's module — same env loading, same client classes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from scraper import (  # noqa: E402
    CONSULTI_API_KEY,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    Consulti,
    Supabase,
    post_to_sheet,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    sb = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    consulti = Consulti(CONSULTI_API_KEY)

    credits = consulti.credits_remaining()
    if credits is not None:
        print(f"Consulti credits remaining: {credits}")

    print("Loading Apollo domains for cross-target check...")
    apollo_domains = sb.get_apollo_domains()
    print(f"  {len(apollo_domains)} Apollo domains loaded.")

    # Re-process anything that's not currently in 'found' state. That covers
    # NULL (never tried) plus 'no_match' (failed under a previous, stricter
    # filter and now deserves another shot).
    print("Fetching scraped_leads rows that need (re)enrichment...")
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/leadgen_scraped_leads"
        "?or=(enrichment_status.is.null,enrichment_status.eq.no_match)"
        "&select=*&order=extracted_at.asc",
        headers=sb.headers,
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()
    print(f"Found {len(rows)} rows to process.\n")

    found = 0
    no_match = 0
    apollo_skip = 0

    for row in rows:
        company = (row.get("company_name") or "").strip()
        domain = (row.get("domain") or "").strip()
        if not company or not domain:
            print(f"[skip] row {row.get('id')}: missing company/domain")
            continue

        print(f"[{row.get('id')}] {company} ({domain})")
        matches = consulti.search_at_firm(company, domain)
        time.sleep(0.5)
        best = Consulti.pick_best(matches)
        now = _now_iso()

        if not best or not best.get("email"):
            print("    → no contact")
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/leadgen_scraped_leads?id=eq.{row['id']}",
                headers={**sb.headers, "Prefer": "return=minimal"},
                json={"enrichment_source": "consulti", "enrichment_status": "no_match", "enriched_at": now},
                timeout=30,
            )
            no_match += 1
            continue

        # Apollo cross-target check via enriched domains
        cross = Consulti.enriched_domains(best) & apollo_domains
        if cross:
            hit = next(iter(cross))
            print(f"    → Apollo cross-target via {hit} — skip")
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/leadgen_scraped_leads?id=eq.{row['id']}",
                headers={**sb.headers, "Prefer": "return=minimal"},
                json={"enrichment_source": "consulti", "enrichment_status": f"apollo_cross_target:{hit}", "enriched_at": now},
                timeout=30,
            )
            apollo_skip += 1
            continue

        contact_name = (
            f"{(best.get('first_name') or '').strip()} "
            f"{(best.get('last_name') or '').strip()}"
        ).strip() or None
        update = {
            "contact_name":         contact_name,
            "contact_title":        best.get("job_title"),
            "contact_email":        best.get("email"),
            "contact_linkedin_url": best.get("linkedin_url"),
            "enrichment_source":    "consulti",
            "enrichment_status":    "found",
            "enriched_at":          now,
        }
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/leadgen_scraped_leads?id=eq.{row['id']}",
            headers={**sb.headers, "Prefer": "return=minimal"},
            json=update,
            timeout=30,
        )
        post_to_sheet({**row, **update})
        print(f"    → {contact_name} ({best.get('job_title')}) {best.get('email')}")
        found += 1

    print(f"\nDone. Found contact: {found}  No match: {no_match}  Apollo cross-target skip: {apollo_skip}")


if __name__ == "__main__":
    main()
