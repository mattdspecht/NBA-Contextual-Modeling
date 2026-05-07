"""
Write a CSV of Basketball Reference headshot URLs for every player_id.

Loads image_scraping/player_ids.csv. Fetches one probe player page to learn the
current /req/<build>/images/headshots/ prefix, then builds
  {prefix}{player_id}.jpg
for each row (no per-player HTTP).

Output: data/player_image_urls.csv (columns: player_id, image_url), same path the API reads.
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_ORIGIN = "https://www.basketball-reference.com"
DISCOVERY_PLAYER_URL = f"{BASE_ORIGIN}/players/a/antetgi01.html"

INPUT_CSV = Path(__file__).resolve().parent / "player_ids.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "player_image_urls.csv"

RATE_LIMIT_COOLDOWN_SECONDS = 300
MAX_429_RETRIES = 12

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

HEADSHOT_PREFIX_RE = re.compile(
    r"https://www\.basketball-reference\.com/req/\d+/images/headshots/",
)


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def headshot_url_from_player_html(html: str) -> str | None:
    soup = soup_from_html(html)
    meta = soup.find("div", id="meta")
    if meta is None:
        return None
    for img in meta.find_all("img"):
        src = (img.get("src") or "").strip()
        if "headshots" in src:
            return urljoin(BASE_ORIGIN, src)
    return None


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


def discover_headshots_prefix(session: requests.Session) -> str:
    response = fetch_with_429_retry(session, DISCOVERY_PLAYER_URL)
    if response is None or response.status_code != 200:
        raise SystemExit(
            f"Could not load discovery page {DISCOVERY_PLAYER_URL!r} "
            f"(status {getattr(response, 'status_code', None)})."
        )
    url = headshot_url_from_player_html(response.text)
    if not url:
        m = HEADSHOT_PREFIX_RE.search(response.text)
        if m:
            return m.group(0)
        raise SystemExit("Could not find headshot URL on discovery page.")
    if "/" not in url:
        raise SystemExit(f"Unexpected headshot URL: {url!r}")
    return url.rsplit("/", 1)[0] + "/"


def load_player_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Missing {path}; run export_player_ids.py first.")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "player_id" not in reader.fieldnames:
            raise SystemExit(f"{path} must have a player_id column.")
        return [r["player_id"].strip() for r in reader if r.get("player_id", "").strip()]


def main() -> None:
    player_ids = load_player_ids(INPUT_CSV)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        prefix = discover_headshots_prefix(session)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["player_id", "image_url"])
        for pid in player_ids:
            writer.writerow([pid, f"{prefix}{pid}.jpg"])

    print(f"Headshots prefix: {prefix}")
    print(f"Wrote {len(player_ids):,} row(s) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
