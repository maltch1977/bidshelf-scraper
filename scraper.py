#!/usr/bin/env python3
"""
BidShelf Lead Scraper v2 — Job Board Edition
============================================

Hunts job postings from commercial GCs / AEC firms hiring proposal-related roles.
The act of hiring IS the pain signal: when a firm publicly posts a "Proposal
Coordinator" or "Estimator" role, the proposal process is hurting them right now.

Pipeline:
    Tavily searches → Kimi extracts firm info → Apollo dedup (Supabase)
        → write to Supabase scraped_leads → POST to Google Sheet

State of the world:
    leadgen_apollo_leads     — your 7,884 existing Apollo prospects (DO NOT TARGET)
    leadgen_scraped_leads    — new firms this scraper finds
    leadgen_scraped_urls     — URL-level cache, never re-process a URL

Run:
    cd /opt/bidshelf-scraper && source venv/bin/activate && python scraper.py
"""

import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

import requests
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient


# ============================================================================
# LOAD .env (simple parser — no external dep needed)
# ============================================================================
def _load_env_file() -> None:
    """Load .env from the same dir as this script, or one level up.
    Lets us deploy the code in /opt/bidshelf-scraper/app/ while keeping
    the .env at /opt/bidshelf-scraper/.env (shared with the venv)."""
    here = Path(__file__).resolve().parent
    for env_path in (here / ".env", here.parent / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
            return


_load_env_file()


# ============================================================================
# CONFIG
# ============================================================================
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "moonshot-v1-128k")
KIMI_BASE_URL = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SHEETS_WEBHOOK_URL = os.environ.get("SHEETS_WEBHOOK_URL", "")
CONSULTI_API_KEY = os.environ.get("CONSULTI_API_KEY", "")

MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "10"))
TAVILY_DELAY_S = float(os.environ.get("TAVILY_DELAY_S", "2.0"))
LLM_DELAY_S = float(os.environ.get("LLM_DELAY_S", "1.0"))
CONSULTI_DELAY_S = float(os.environ.get("CONSULTI_DELAY_S", "0.5"))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", "12000"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

# Fail fast on missing secrets
for name in ("TAVILY_API_KEY", "KIMI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "CONSULTI_API_KEY"):
    if not os.environ.get(name):
        print(f"FATAL: {name} is not set in .env", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# QUERIES — job board searches for proposal/preconstruction/estimating roles
# ============================================================================
JOB_BOARD_QUERIES = [
    # --- LinkedIn Jobs (highest-yield for direct firm postings) ---
    'site:linkedin.com/jobs "proposal coordinator" "general contractor"',
    'site:linkedin.com/jobs "proposal manager" "construction"',
    'site:linkedin.com/jobs "proposal writer" "construction"',
    'site:linkedin.com/jobs "proposal coordinator" "AEC"',
    'site:linkedin.com/jobs "estimator" "commercial construction"',
    'site:linkedin.com/jobs "senior estimator" "general contractor"',
    'site:linkedin.com/jobs "preconstruction" "general contractor"',
    'site:linkedin.com/jobs "director of preconstruction" "construction"',
    'site:linkedin.com/jobs "business development" "general contractor" "proposal"',

    # --- Indeed (friendlier to search engines than LinkedIn) ---
    'site:indeed.com "proposal coordinator" "general contractor"',
    'site:indeed.com "proposal manager" "commercial construction"',
    'site:indeed.com "proposal writer" "construction"',
    'site:indeed.com "estimator" "general contractor"',
    'site:indeed.com "preconstruction manager" "construction"',
    'site:indeed.com "preconstruction" "AEC"',

    # --- ZipRecruiter ---
    'site:ziprecruiter.com "proposal coordinator" "construction"',
    'site:ziprecruiter.com "estimator" "general contractor"',
    'site:ziprecruiter.com "preconstruction" "commercial construction"',

    # --- Glassdoor ---
    'site:glassdoor.com/Job "proposal coordinator" "construction"',
    'site:glassdoor.com/Job "estimator" "general contractor"',
    'site:glassdoor.com/Job "preconstruction" "general contractor"',

    # --- AGC career center & industry boards ---
    'site:agc.org "proposal coordinator"',
    'site:agc.org "estimator"',
    'site:abc.org "proposal coordinator"',

    # --- Direct firm career pages (Tavily catches these via Google) ---
    '"careers" "proposal coordinator" "general contractor"',
    '"careers" "proposal manager" "commercial construction"',
    '"careers" "preconstruction" "AEC"',
    '"now hiring" "proposal coordinator" "construction"',
    '"join our team" "proposal coordinator" "general contractor"',
]


# ============================================================================
# EXTRACTION SCHEMA + PROMPT
# ============================================================================
class FirmExtraction(BaseModel):
    is_valid: bool = Field(
        ..., description="True only if this is a real job posting from a commercial GC / AEC firm hiring a proposal/preconstruction/estimating role."
    )
    company_name: Optional[str] = None
    domain: Optional[str] = Field(
        None,
        description="Firm website domain, lowercase, no www, no protocol, no path. Example: 'acmeconstruction.com'.",
    )
    website: Optional[str] = None
    linkedin_url: Optional[str] = None
    job_title: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    signal_type: Optional[str] = Field(
        None,
        description="hiring_proposal_role | hiring_estimator | hiring_preconstruction | hiring_bd_construction | hiring_other",
    )
    signal_heat: Optional[str] = Field(None, description="hot | warm | cool")
    reasoning: Optional[str] = None


EXTRACTION_SYSTEM_PROMPT = """\
You are a B2B lead qualification analyst for BidShelf — a platform that eliminates
manual proposal generation pain for commercial general contractors and AEC firms.

Your job: evaluate a job-posting URL and decide whether it represents a commercial
construction / AEC firm hiring a proposal-, estimating-, or preconstruction-related
role. If so, that's a hot pain signal — they need BidShelf right now.

═══════════════════════════════════════════
GATE 1 — IS THIS A JOB POSTING?
═══════════════════════════════════════════
ACCEPT ONLY if the content is a real, identifiable job posting from a hiring company:
  - Has a job title, responsibilities, requirements
  - Identifies the hiring firm (or strongly implies one)
  - Is currently or recently active

REJECT if:
  - It's a personal profile / resume (e.g. linkedin.com/in/...)
  - It's a forum post, article, blog, news piece, or video
  - It's a staffing agency / recruiter posting on behalf of an unnamed client
  - The hiring firm cannot be identified

═══════════════════════════════════════════
GATE 2 — IS THE HIRING FIRM IN OUR ICP?
═══════════════════════════════════════════
ACCEPT if the firm is in:
  - Commercial general contracting
  - AEC (architecture / engineering / construction)
  - Specialty trade contracting (mechanical, electrical, civil, concrete, roofing, glazing, etc.)
  - Design-build firms
  - Construction management

REJECT if the firm is in:
  - Federal / government / defense / IT services contracting
  - Healthcare, pharma, biotech
  - IT services, software, SaaS, ERP, Oracle / Salesforce consulting
  - Marketing / creative / advertising / PR agencies
  - Legal / accounting / financial services
  - Staffing / recruiting / executive search (even when hiring for construction — we cannot reach the end client)
  - Residential-only homebuilders (we want commercial)
  - Anything else not clearly commercial construction or AEC

═══════════════════════════════════════════
GATE 3 — IS THE ROLE A PAIN SIGNAL?
═══════════════════════════════════════════
The role must indicate proposal / RFP / estimating / preconstruction work.

HOT signals (signal_heat='hot'):
  - Proposal Coordinator, Proposal Manager, Proposal Writer, Proposal Specialist
  - RFP Manager, Capture Manager, Bid Manager
  - Director / VP of Proposals
  → signal_type='hiring_proposal_role'

WARM signals (signal_heat='warm'):
  - Estimator, Senior Estimator, Lead Estimator, Chief Estimator
  - Preconstruction Manager, Director of Preconstruction, VP of Preconstruction
  → signal_type='hiring_estimator' or 'hiring_preconstruction'

COOL signals (signal_heat='cool'):
  - Business Development with explicit proposal responsibility
  - Project Manager with explicit proposal duties
  → signal_type='hiring_bd_construction' or 'hiring_other'

If the role is unrelated to proposals/estimating/preconstruction → REJECT.

═══════════════════════════════════════════
EXTRACTION
═══════════════════════════════════════════
If all gates pass, extract:
  - company_name: the HIRING firm (NOT the job board)
  - domain: firm website, normalized (lowercase, no www, no protocol, no path).
            Example: 'acmeconstruction.com'.
  - website: firm website URL as-is
  - linkedin_url: firm's company LinkedIn URL if present
  - job_title: role being hired
  - city, state: hiring location
  - signal_type: one of the values above
  - signal_heat: 'hot' | 'warm' | 'cool'
  - reasoning: one sentence on why this passes

If any gate fails, set is_valid=false and explain WHICH gate failed in reasoning.
A bad lead is worse than no lead. When in doubt, reject.
"""


# ============================================================================
# SUPABASE CLIENT (uses raw REST API, no SDK needed)
# ============================================================================
class Supabase:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _paginated_select(self, path: str, column: str, page_size: int = 1000) -> Set[str]:
        out: Set[str] = set()
        offset = 0
        while True:
            r = requests.get(
                f"{self.url}/rest/v1/{path}",
                headers={**self.headers, "Range-Unit": "items", "Range": f"{offset}-{offset + page_size - 1}"},
                timeout=30,
            )
            r.raise_for_status()
            rows = r.json()
            for row in rows:
                v = row.get(column)
                if v:
                    out.add(str(v).strip().lower() if column == "domain" else str(v).strip())
            if len(rows) < page_size:
                break
            offset += page_size
        return out

    def get_apollo_domains(self) -> Set[str]:
        return self._paginated_select("leadgen_apollo_leads?select=domain&domain=not.is.null", "domain")

    def get_scraped_domains(self) -> Set[str]:
        return self._paginated_select("leadgen_scraped_leads?select=domain&domain=not.is.null", "domain")

    def get_scraped_urls(self) -> Set[str]:
        return self._paginated_select("leadgen_scraped_urls?select=url", "url")

    def record_url(self, url: str, qualified: bool, rejection_reason: Optional[str], source_query: str) -> None:
        body = {
            "url": url,
            "was_qualified": qualified,
            "rejection_reason": (rejection_reason or "")[:500] or None,
            "source_query": source_query,
        }
        try:
            r = requests.post(
                f"{self.url}/rest/v1/leadgen_scraped_urls",
                headers={**self.headers, "Prefer": "return=minimal"},
                json=body,
                timeout=30,
            )
            # 409 = already exists; that's fine
            if r.status_code not in (201, 204, 409):
                print(f"    [WARN] scraped_urls insert failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"    [WARN] scraped_urls insert error: {e}")

    def insert_lead(self, lead_row: dict) -> bool:
        try:
            r = requests.post(
                f"{self.url}/rest/v1/leadgen_scraped_leads",
                headers={**self.headers, "Prefer": "return=minimal"},
                json=lead_row,
                timeout=30,
            )
            if r.status_code in (201, 204):
                return True
            # 409 means domain dedup hit; we shouldn't get here often but it's safe
            if r.status_code == 409:
                print(f"    [WARN] scraped_leads conflict (likely race): {lead_row.get('domain')}")
                return False
            print(f"    [WARN] scraped_leads insert failed: {r.status_code} {r.text[:200]}")
            return False
        except Exception as e:
            print(f"    [WARN] scraped_leads insert error: {e}")
            return False

    def update_lead_sheet_posted(self, lead_id: int) -> None:
        # Used after a successful Sheet POST — sets posted_to_sheet_at.
        # We don't currently use this since we fire-and-forget the Sheet,
        # but it's here if we want to add retry logic later.
        pass


# ============================================================================
# KIMI EXTRACTION
# ============================================================================
def _clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_firm(content: str, source_url: str, kimi: OpenAI) -> Optional[FirmExtraction]:
    schema_str = json.dumps(FirmExtraction.model_json_schema(), indent=2)
    user_prompt = (
        f"Source URL: {source_url}\n\n"
        f"--- CONTENT START ---\n"
        f"{content[:MAX_CONTENT_LENGTH]}\n"
        f"--- CONTENT END ---\n\n"
        f"Respond with a single JSON object matching this JSON Schema. "
        f"Output ONLY valid JSON. No markdown fences, no preamble, no commentary.\n\n"
        f"{schema_str}"
    )
    try:
        completion = kimi.chat.completions.create(
            model=KIMI_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = completion.choices[0].message.content or ""
        cleaned = _clean_json_response(raw)
        parsed = json.loads(cleaned)
        return FirmExtraction.model_validate(parsed)
    except ValidationError as e:
        print(f"    [LLM] Validation error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"    [LLM] JSON decode error: {e}")
        return None
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "429" in err:
            print("    [LLM] Rate limited; sleeping 60s...")
            time.sleep(60)
            return extract_firm(content, source_url, kimi)
        print(f"    [LLM] Error: {e}")
        return None


def normalize_domain(s: Optional[str]) -> Optional[str]:
    if not s or not s.strip():
        return None
    s = s.strip().lower()
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.startswith("www."):
        s = s[4:]
    s = s.split("/")[0].split(":")[0]
    return s if s and "." in s else None


# ============================================================================
# CONSULTI ENRICHMENT
# ============================================================================
class Consulti:
    """Client for Consulti.ai's B2B lead search.

    Given a firm's name + domain, find a senior, proposal-relevant contact at the
    firm. Returns None if no usable contact is found — caller MUST silent-skip
    those firms so half-formed leads never reach the sales-team Sheet.
    """

    BASE = "https://www.consulti.ai/api/v1"

    # Title keywords we search Consulti for. Broad enough to catch variants
    # ("Director of Preconstruction", "VP Pre-Construction", etc.) since
    # Consulti's `titles` filter is substring-style.
    SEARCH_TITLES = [
        "Preconstruction",
        "Pre-Construction",
        "Business Development",
        "Proposal",
        "Estimator",
        "Estimating",
        "President",
        "CEO",
        "Owner",
    ]

    # Title-priority ranking — lower = higher priority. Used to pick the best
    # contact when Consulti returns multiple matches at a firm.
    # NOTE: no generic "Vice President" catchall — Consulti's title filter is a
    # substring match, so "President" inside "Vice President" returns lots of
    # off-target VPs (Insurance, Risk, HR, Finance). Require a specific keyword.
    TITLE_PRIORITIES = [
        ("preconstruction",      1),
        ("pre-construction",     1),
        ("proposal",             1),
        ("rfp",                  1),
        ("business development", 2),
        ("bd ",                  2),
        ("estimator",            3),
        ("estimating",           3),
        ("president",            4),
        ("ceo",                  4),
        ("chief executive",      4),
        ("owner",                4),
        ("principal",            4),
    ]

    def __init__(self, key: str):
        self.key = key
        self.headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    # Suffixes stripped when generating shorter search candidates.
    _SUFFIX_RE = re.compile(
        r"\s*(?:[,\-—]\s*)?(?:"
        r"a [A-Za-z]+ Company|a [A-Za-z]+ Corporation"
        r"|Incorporated|Inc|LLC|L\.L\.C\.?|Ltd|Limited|Corp|Corporation|Company|Co"
        r"|PLC|GmbH|LP|LLP|PA|PC|PLLC"
        r"|General Contractors?|Construction Group|Construction Company|Construction Corp"
        r"|Construction Inc|Construction Co|Construction"
        r"|Contracting|Contractors?|Builders?|Group"
        r")\.?\s*$",
        flags=re.IGNORECASE,
    )

    @classmethod
    def search_candidates(cls, company_name: str, domain: str = "") -> list:
        """Generate progressively shorter query variants for Consulti's
        substring company filter.

        Consulti indexes firms under cleaner short names ("HITT Contracting")
        than what Kimi extracts from job postings ("Hitt Contracting Inc").
        Long formal names fail substring match. We try the full name, then
        strip legal/industry suffixes, then fall back to a domain-derived
        token like 'Mastec' (from mastec.com).
        """
        name = (company_name or "").strip()
        out: list = []
        seen: Set[str] = set()

        def add(s: str):
            s = s.strip(" ,.-—\t")
            key = s.lower()
            if s and key not in seen and len(s) >= 3:
                seen.add(key)
                out.append(s)

        if name:
            add(name)
            # Apply suffix stripping up to twice (handles "Construction Inc" → "Inc" → done)
            cleaned = cls._SUFFIX_RE.sub("", name).strip()
            cleaned2 = cls._SUFFIX_RE.sub("", cleaned).strip()
            add(cleaned)
            add(cleaned2)

        # Domain fallback: "willmengconstruction.com" → "willmeng", "mastec.com" → "Mastec".
        if domain:
            sld = domain.split(".")[0].lower()
            for suffix in ("construction", "contracting", "contractors", "builders", "construct", "group", "inc", "corp"):
                if sld.endswith(suffix) and len(sld) > len(suffix) + 2:
                    sld = sld[: -len(suffix)]
                    break
            if sld and len(sld) >= 4:
                add(sld.capitalize())

        return out

    def search_at_firm(self, company_name: str, target_domain: str, size: int = 10) -> list:
        """Search Consulti for leads at a company by trying progressively
        shorter query variants until one returns results. Returns the first
        non-empty result set (or [] if no variant matches).

        Consulti's `company_domain` field often differs from what we scraped
        off the job posting (we get hittcontracting.com from a LinkedIn page,
        Consulti has hitt.com as the canonical work-email domain). We trust
        Consulti's company-name substring match to scope results.
        """
        for candidate in self.search_candidates(company_name, target_domain):
            leads = self._search_with_company(candidate, size)
            if leads:
                return leads
        return []

    def _search_with_company(self, company: str, size: int) -> list:
        body = {
            "company": company,
            "titles": self.SEARCH_TITLES,
            "countries": ["United States"],
            "size": size,
        }
        try:
            r = requests.post(
                f"{self.BASE}/leads/search",
                headers=self.headers,
                json=body,
                timeout=30,
            )
            if r.status_code != 200:
                print(f"    [Consulti] HTTP {r.status_code} for '{company}': {r.text[:160]}")
                return []
            return (r.json().get("leads") or [])
        except Exception as e:
            print(f"    [Consulti] error for '{company}': {e}")
            return []

    @classmethod
    def pick_best(cls, leads: list) -> Optional[dict]:
        """Return the highest-priority lead, or None if no lead has a
        proposal-relevant title. Title gate prevents Insurance/Risk/HR VPs from
        slipping through (Consulti's title filter is substring-style, so
        searching 'President' returns lots of 'Vice President of <anything>')."""
        if not leads:
            return None

        def score(lead):
            title = (lead.get("job_title") or "").lower()
            for keyword, tier in cls.TITLE_PRIORITIES:
                if keyword in title:
                    return tier
            return 99

        best = sorted(leads, key=score)[0]
        if score(best) >= 99:
            return None  # no proposal-relevant title in the result set
        return best

    @staticmethod
    def enriched_domains(contact: dict) -> set:
        """Extract all possible 'true' domains for a Consulti contact — used
        for Apollo cross-target checks. Returns a set of normalized domains
        from both the contact's email and Consulti's company_domain field."""
        out = set()
        email = (contact.get("email") or "").strip()
        if "@" in email:
            d = normalize_domain(email.rsplit("@", 1)[1])
            if d:
                out.add(d)
        cd = normalize_domain(contact.get("company_domain"))
        if cd:
            out.add(cd)
        return out

    def credits_remaining(self) -> Optional[int]:
        try:
            r = requests.get(f"{self.BASE}/credits", headers=self.headers, timeout=10)
            if r.status_code != 200:
                return None
            return r.json().get("data", {}).get("lead_credits")
        except Exception:
            return None


# ============================================================================
# GOOGLE SHEET PUSHER (fire-and-forget; CSV write removed since DB is canonical)
# ============================================================================
def post_to_sheet(lead_row: dict) -> None:
    """Push a fully-enriched lead row to the Google Sheet for the sales team."""
    if not SHEETS_WEBHOOK_URL:
        return
    payload = {
        "company_name":         lead_row.get("company_name") or "",
        "domain":               lead_row.get("domain") or "",
        "contact_name":         lead_row.get("contact_name") or "",
        "contact_title":        lead_row.get("contact_title") or "",
        "contact_email":        lead_row.get("contact_email") or "",
        "contact_linkedin_url": lead_row.get("contact_linkedin_url") or "",
        "signal_heat":          lead_row.get("signal_heat") or "",
        "signal_type":          lead_row.get("signal_type") or "",
        "job_title":            lead_row.get("job_title") or "",
        "job_posting_url":      lead_row.get("job_posting_url") or "",
        "city":                 lead_row.get("city") or "",
        "state":                lead_row.get("state") or "",
        "reasoning":            lead_row.get("reasoning") or "",
        "extracted_at":         lead_row.get("extracted_at") or "",
    }
    try:
        r = requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=15)
        if r.status_code != 200 or '"status":"ok"' not in r.text:
            print(f"    [Sheet] Warning: {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"    [Sheet] Warning: {e}")


# ============================================================================
# MAIN
# ============================================================================
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    print("\n[INFO] Shutdown requested, finishing current item...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main() -> None:
    print("=" * 70)
    print(" BidShelf Lead Scraper v2 — Job Board Edition")
    print("=" * 70)
    print(f" Provider     : Kimi ({KIMI_MODEL})")
    print(f" Supabase     : {SUPABASE_URL}")
    print(f" Queries      : {len(JOB_BOARD_QUERIES)}")
    print(f" Max/query    : {MAX_RESULTS_PER_QUERY}")
    print(f" Tavily delay : {TAVILY_DELAY_S}s")
    print(f" LLM delay    : {LLM_DELAY_S}s")
    print("=" * 70)

    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    kimi = OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)
    sb = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    consulti = Consulti(CONSULTI_API_KEY)

    credits = consulti.credits_remaining()
    if credits is not None:
        print(f"[INIT] Consulti credits remaining: {credits}")

    print("[INIT] Loading Apollo domains for dedup...")
    apollo_domains = sb.get_apollo_domains()
    print(f"[INIT] {len(apollo_domains)} Apollo domains loaded.")

    print("[INIT] Loading already-scraped domains...")
    scraped_domains = sb.get_scraped_domains()
    print(f"[INIT] {len(scraped_domains)} scraped domains loaded.")

    print("[INIT] Loading already-processed URLs...")
    scraped_urls = sb.get_scraped_urls()
    print(f"[INIT] {len(scraped_urls)} URLs already processed.")

    total_new = 0
    total_processed = 0
    start = time.time()

    for q_idx, query in enumerate(JOB_BOARD_QUERIES, 1):
        if _shutdown:
            break
        print(f"\n[{q_idx}/{len(JOB_BOARD_QUERIES)}] {query[:80]}")
        try:
            resp = tavily.search(query=query, max_results=MAX_RESULTS_PER_QUERY, search_depth="advanced")
            results = [
                {"url": str(r.get("url") or "").strip(), "content": str(r.get("content") or "").strip()}
                for r in resp.get("results", [])
                if r.get("url") and r.get("content")
            ]
        except Exception as e:
            print(f"  [Tavily] error: {e}")
            time.sleep(TAVILY_DELAY_S)
            continue

        print(f"  [Tavily] {len(results)} results.")
        time.sleep(TAVILY_DELAY_S)

        for r_idx, result in enumerate(results, 1):
            if _shutdown:
                break
            url = result["url"]
            content = result["content"]
            total_processed += 1

            if url in scraped_urls:
                print(f"    [{r_idx}] Skip (URL cached): {url[:80]}")
                continue

            print(f"    [{r_idx}] Analyzing: {url[:80]}")
            firm = extract_firm(content, url, kimi)
            time.sleep(LLM_DELAY_S)

            if firm is None:
                sb.record_url(url, qualified=False, rejection_reason="extraction_error", source_query=query)
                scraped_urls.add(url)
                continue

            if not firm.is_valid:
                reason = (firm.reasoning or "did not qualify")[:200]
                print(f"    [Skip] {reason[:100]}")
                sb.record_url(url, qualified=False, rejection_reason=reason, source_query=query)
                scraped_urls.add(url)
                continue

            domain = normalize_domain(firm.domain) or normalize_domain(firm.website)
            if not domain:
                print("    [Skip] No usable domain")
                sb.record_url(url, qualified=False, rejection_reason="no_domain", source_query=query)
                scraped_urls.add(url)
                continue

            if domain in apollo_domains:
                print(f"    [Skip] Apollo dedup: {domain}")
                sb.record_url(url, qualified=False, rejection_reason=f"apollo_dedup:{domain}", source_query=query)
                scraped_urls.add(url)
                continue

            if domain in scraped_domains:
                print(f"    [Skip] Already scraped: {domain}")
                sb.record_url(url, qualified=False, rejection_reason=f"already_scraped:{domain}", source_query=query)
                scraped_urls.add(url)
                continue

            # ---- CONSULTI ENRICHMENT ----
            # Without a usable contact, the lead can't be acted on, so we silent-skip.
            consulti_matches = consulti.search_at_firm(firm.company_name or "", domain)
            time.sleep(CONSULTI_DELAY_S)
            best_contact = Consulti.pick_best(consulti_matches)

            if not best_contact or not best_contact.get("email"):
                print(f"    [Skip] No Consulti contact for {domain}")
                sb.record_url(url, qualified=False, rejection_reason=f"no_consulti_contact:{domain}", source_query=query)
                scraped_urls.add(url)
                continue

            # Apollo cross-target check using the enriched (true) domains —
            # catches the case where we scraped a marketing/redirect domain
            # but the firm's actual work-email domain IS in your Apollo list.
            cross_domains = Consulti.enriched_domains(best_contact)
            apollo_hit = cross_domains & apollo_domains
            if apollo_hit:
                hit = next(iter(apollo_hit))
                print(f"    [Skip] Apollo cross-target via enriched domain: {hit}")
                sb.record_url(url, qualified=False, rejection_reason=f"apollo_cross_target_enriched:{hit}", source_query=query)
                scraped_urls.add(url)
                continue

            contact_name = (
                f"{(best_contact.get('first_name') or '').strip()} "
                f"{(best_contact.get('last_name') or '').strip()}"
            ).strip() or None

            # ---- NEW LEAD ----
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            lead_row = {
                "company_name": firm.company_name,
                "domain": domain,
                "website": firm.website,
                "linkedin_url": firm.linkedin_url,
                "job_posting_url": url,
                "job_title": firm.job_title,
                "signal_type": firm.signal_type,
                "signal_heat": firm.signal_heat,
                "city": firm.city,
                "state": firm.state,
                "reasoning": firm.reasoning,
                "source_query": query,
                "extracted_at": now,
                # contact (Consulti)
                "contact_name":         contact_name,
                "contact_title":        best_contact.get("job_title"),
                "contact_email":        best_contact.get("email"),
                "contact_phone":        None,  # /leads/search doesn't return phone
                "contact_linkedin_url": best_contact.get("linkedin_url"),
                "enrichment_source":    "consulti",
                "enrichment_status":    "found",
                "enriched_at":          now,
            }
            if sb.insert_lead(lead_row):
                scraped_domains.add(domain)
                total_new += 1
                heat = (firm.signal_heat or "?").upper()
                print(
                    f"    [LEAD {heat}] {firm.company_name} | {contact_name} ({best_contact.get('job_title')}) | "
                    f"{best_contact.get('email')}"
                )
                post_to_sheet(lead_row)
                sb.record_url(url, qualified=True, rejection_reason=None, source_query=query)
            else:
                sb.record_url(url, qualified=False, rejection_reason="insert_failed", source_query=query)
            scraped_urls.add(url)

        elapsed = time.time() - start
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"  [Progress] {total_processed} URLs | {total_new} new leads | {rate:.2f} URLs/sec")

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(" RUN COMPLETE")
    print(f" URLs processed : {total_processed}")
    print(f" New leads      : {total_new}")
    print(f" Time elapsed   : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
