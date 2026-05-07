"""
Write data/team_image_urls.csv: 3-letter team acronym -> Sports Reference CDN logo URL.

Fetches one Basketball Reference team page, parses the ssref tlogo URL, then builds
  https://cdn.ssref.net/req/<build>/tlogo/bbr/<ABBR>-<season_year>.png
for every NBA team (same mapping as src/api/app.py TEAM_ACRONYMS).

Example (Boston):
  https://cdn.ssref.net/req/202605010/tlogo/bbr/BOS-2026.png
"""

from __future__ import annotations

import csv
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_ORIGIN = "https://www.basketball-reference.com"
OUTPUT_CSV = REPO_ROOT / "data" / "team_image_urls.csv"

# Same full-name → 3-letter map as src/api/app.py (Performances use full names).
TEAM_NAME_TO_ACRONYM = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

PROBE_TEAM_ABBR = "BOS"

RATE_LIMIT_COOLDOWN_SECONDS = 300
MAX_429_RETRIES = 12

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Full logo URL on team pages (absolute on cdn.ssref.net).
TEAM_LOGO_URL_RE = re.compile(
    r"(https://cdn\.ssref\.net/req/\d+/tlogo/bbr/)([A-Z0-9]{2,4})-(\d{4})\.png"
)


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def fetch_with_429_retry(session: requests.Session, url: str) -> requests.Response | None:
    response = None
    for attempt in range(MAX_429_RETRIES):
        response = session.get(url, timeout=60)
        if response.status_code == 429:
            print(
                f"HTTP 429 from Basketball Reference; sleeping {RATE_LIMIT_COOLDOWN_SECONDS}s "
                f"(attempt {attempt + 1}/{MAX_429_RETRIES})..."
            )
            time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
            continue
        return response
    return response


def candidate_season_end_years() -> list[int]:
    """BR /teams/X/<Y>.html uses the second calendar year of the NBA season."""
    now = datetime.now()
    primary = now.year + 1 if now.month >= 10 else now.year
    return [primary, primary - 1, primary + 1]


def discover_logo_template(session: requests.Session) -> tuple[str, str]:
    """
    Returns (url_prefix, season_year_str) where each team URL is
    f'{prefix}{ACRONYM}-{season_year}.png'
    """
    last_status = None
    for end_year in candidate_season_end_years():
        team_url = f"{BASE_ORIGIN}/teams/{PROBE_TEAM_ABBR}/{end_year}.html"
        response = fetch_with_429_retry(session, team_url)
        last_status = getattr(response, "status_code", None)
        if response is None or response.status_code != 200:
            continue
        m = TEAM_LOGO_URL_RE.search(response.text)
        if m:
            prefix = m.group(1)
            year = m.group(3)
            print(f"Discovered logo template from {team_url}")
            return prefix, year
        # Fallback: any img / meta pointing at tlogo
        soup = soup_from_html(response.text)
        for img in soup.find_all("img", src=True):
            src = img.get("src", "")
            m2 = TEAM_LOGO_URL_RE.search(src)
            if m2:
                print(f"Discovered logo template from img src on {team_url}")
                return m2.group(1), m2.group(3)

    raise SystemExit(
        f"Could not find team logo URL on BR probe pages (last HTTP {last_status}). "
        "Try again later or update candidate_season_end_years()."
    )


def all_team_acronyms() -> list[str]:
    return sorted(set(TEAM_NAME_TO_ACRONYM.values()))


def main() -> None:
    acronyms = all_team_acronyms()

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        prefix, season_year = discover_logo_template(session)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team_acronym", "image_url"])
        for abbr in acronyms:
            writer.writerow([abbr, f"{prefix}{abbr}-{season_year}.png"])

    print(f"Logo URL prefix: {prefix}")
    print(f"Season year suffix: {season_year}")
    print(f"Wrote {len(acronyms)} row(s) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
