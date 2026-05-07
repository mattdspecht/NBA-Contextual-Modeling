"""
Incremental data updater for NBA Contextual Modeling.

Scrapes new schedule entries and box scores since the last update,
appends them to existing raw CSVs, then re-runs preprocess and build_db.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

STATE_PATH = REPO_ROOT / "datasets" / "refresh_state.json"
SCHEDULE_CSV = REPO_ROOT / "datasets" / "raw" / "schedule.csv"
RAW_CSV_TEMPLATE = str(REPO_ROOT / "datasets" / "raw" / "player_stats_{year}.csv")
CLEANER_SCRIPT = REPO_ROOT / "nba" / "etl" / "preprocess.py"
LOADER_SCRIPT = REPO_ROOT / "nba" / "etl" / "build_db.py"

INITIAL_LAST_UPDATED = "2026-05-04"
COOLDOWN_MINUTES = 60

BREF_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

NBA_MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june",
    10: "october", 11: "november", 12: "december",
}
NBA_MONTHS = set(NBA_MONTH_NAMES.keys())


def get_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_updated": INITIAL_LAST_UPDATED, "last_attempt": None, "scraped_urls": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_PATH)


def nba_year_for_month(calendar_year: int, month_number: int) -> int:
    """Oct/Nov/Dec of year N belong to NBA season N+1; Jan-June belong to season N."""
    return calendar_year + 1 if month_number >= 10 else calendar_year


def months_to_check(from_date: date, to_date: date) -> list[tuple[int, str]]:
    """Returns (nba_season_year, month_name) tuples for NBA months between from_date and to_date."""
    result = []
    year, month = from_date.year, from_date.month
    end_year, end_month = to_date.year, to_date.month

    while (year, month) <= (end_year, end_month):
        if month in NBA_MONTHS:
            nba_year = nba_year_for_month(year, month)
            result.append((nba_year, NBA_MONTH_NAMES[month]))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return result


def parse_schedule_date(date_str: str) -> Optional[date]:
    """Parse 'Tue, May 4, 2026' -> date object, None on failure."""
    try:
        return datetime.strptime(date_str.strip(), "%a, %b %d, %Y").date()
    except (ValueError, AttributeError):
        return None


def calendar_year_from_date_str(date_str: str) -> Optional[int]:
    """Extract calendar year from BBRef date string like 'Tue, May 4, 2026' -> 2026."""
    d = parse_schedule_date(date_str)
    return d.year if d else None


def run_incremental_refresh(status: dict) -> None:
    """Full incremental refresh. Mutates status dict in-place for progress tracking."""
    # Lazy imports: avoids circular import issues; REPO_ROOT must be on sys.path already.
    from nba.etl.schedule_scraper import fetch_month_schedule
    from nba.etl.game_scraper import (
        fetch_and_parse_box_score,
        OUTPUT_COLUMN_ORDER,
        REQUEST_DELAY_SECONDS,
    )

    try:
        state = get_state()
        last_updated = date.fromisoformat(state["last_updated"])
        today = date.today()

        # 1-day lookback: box scores sometimes post hours after game ends
        check_from = last_updated - timedelta(days=1)

        existing_sched = pd.read_csv(SCHEDULE_CSV)
        # known_urls: URLs already in schedule.csv — used only to avoid re-adding schedule rows
        known_urls: set[str] = set(
            existing_sched["box_score_url"].dropna().astype(str).str.strip()
        )
        known_urls.discard("")
        # scraped_urls: URLs whose box scores have been successfully fetched and written
        scraped_urls: set[str] = set(state.get("scraped_urls") or [])

        # Phase 1: Fetch new schedule entries
        status.update({"message": "Fetching schedule updates...", "progress": 0.0})

        month_list = months_to_check(check_from, today)
        new_sched_rows: list[dict] = []  # rows to append to schedule.csv
        new_rows: list[dict] = []        # rows that need box scores scraped

        with requests.Session() as session:
            session.headers.update({"User-Agent": BREF_USER_AGENT})
            for nba_year, month_name in month_list:
                rows, hit_rate_limit = fetch_month_schedule(session, nba_year, month_name)
                time.sleep(3.1)

                if hit_rate_limit:
                    status.update({
                        "status": "error",
                        "message": "Rate limited by Basketball-Reference. Try again later.",
                    })
                    return

                if not rows:
                    continue

                for row in rows:
                    url = (row.get("box_score_url") or "").strip()
                    if not url:
                        continue
                    row_date = parse_schedule_date(row["date"])
                    if row_date is None or row_date < check_from:
                        continue
                    if url not in known_urls:
                        new_sched_rows.append(row)
                        known_urls.add(url)
                    # Scrape box score if not already done, regardless of schedule status
                    if url not in scraped_urls:
                        new_rows.append(row)

        if new_sched_rows:
            pd.DataFrame(new_sched_rows).to_csv(SCHEDULE_CSV, mode="a", header=False, index=False)

        status.update({
            "message": f"Found {len(new_rows)} game(s) needing box scores. Scraping..."
            if new_rows else "No new games found.",
            "progress": 0.25,
        })

        if not new_rows:
            state["last_updated"] = today.isoformat()
            save_state(state)
            status.update({
                "status": "done",
                "message": "No new games found. Data is already up to date.",
                "progress": 1.0,
            })
            return

        # Phase 2: Scrape box scores for games not yet in raw CSVs
        total = len(new_rows)
        with requests.Session() as session:
            session.headers.update({"User-Agent": BREF_USER_AGENT})
            for i, row in enumerate(new_rows):
                cal_year = calendar_year_from_date_str(row["date"])
                if not cal_year:
                    continue

                output_csv = Path(RAW_CSV_TEMPLATE.format(year=cal_year))
                game_dict = {
                    "date": row["date"],
                    "arena": row.get("arena", ""),
                    "home_team": row["home_team"],
                    "visitor_team": row["visitor_team"],
                    "box_score_url": row["box_score_url"],
                }
                rows_data = fetch_and_parse_box_score(session, row["box_score_url"], game_dict)
                time.sleep(REQUEST_DELAY_SECONDS)

                if rows_data:
                    batch = pd.DataFrame(rows_data)
                    for col in OUTPUT_COLUMN_ORDER:
                        if col not in batch.columns:
                            batch[col] = ""
                    batch = batch[OUTPUT_COLUMN_ORDER]
                    # Guard against duplicates: skip if this game date+home already exists
                    if output_csv.exists():
                        existing = pd.read_csv(output_csv, usecols=["Date", "Home_Team"])
                        game_date = row["date"]
                        home = row["home_team"]
                        already_present = (
                            (existing["Date"] == game_date) & (existing["Home_Team"] == home)
                        ).any()
                    else:
                        already_present = False

                    if not already_present:
                        write_header = not output_csv.exists()
                        batch.to_csv(output_csv, mode="a", header=write_header, index=False)

                    scraped_urls.add(row["box_score_url"])
                    state["scraped_urls"] = list(scraped_urls)
                    save_state(state)

                status.update({
                    "message": f"Scraped {i + 1}/{total} games...",
                    "progress": round(0.25 + 0.50 * ((i + 1) / total), 3),
                })

        # Phase 3: Re-run cleaner via subprocess (cleaner.main() calls os.chdir())
        status.update({"message": "Processing all data...", "progress": 0.75})
        result = subprocess.run(
            [sys.executable, str(CLEANER_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            snippet = (result.stderr or result.stdout or "unknown error")[-500:]
            status.update({"status": "error", "message": f"Cleaner failed: {snippet}"})
            return

        # Phase 4: Re-run loader
        status.update({"message": "Updating database...", "progress": 0.88})
        result = subprocess.run(
            [sys.executable, str(LOADER_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            snippet = (result.stderr or result.stdout or "unknown error")[-500:]
            status.update({"status": "error", "message": f"Database update failed: {snippet}"})
            return

        # Phase 5: Persist new last_updated date
        state["last_updated"] = today.isoformat()
        save_state(state)

        status.update({
            "status": "done",
            "message": f"Refresh complete! Added {total} new game(s).",
            "progress": 1.0,
        })

    except Exception as e:
        status.update({"status": "error", "message": f"Unexpected error: {e}"})
