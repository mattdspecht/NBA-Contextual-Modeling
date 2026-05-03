import os
import re
import time
from bs4 import BeautifulSoup, Comment
import pandas as pd
import requests

TARGET_YEAR = "2020"
SCHEDULE_CSV = "data/raw/nba_schedule_2021_2026.csv"
OUTPUT_CSV = f"data/raw/nba_player_stats_{TARGET_YEAR}.csv"

BASE_ORIGIN = "https://www.basketball-reference.com"
REQUEST_DELAY_SECONDS = 3.1

BOX_BASIC_TABLE_ID_RE = re.compile(r"^box-[A-Z]+-game-basic$")

BR_STAT_KEYS = [
    "fg",
    "fga",
    "fg_pct",
    "fg3",
    "fg3a",
    "fg3_pct",
    "ft",
    "fta",
    "ft_pct",
    "orb",
    "drb",
    "trb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
    "plus_minus",
]

OUTPUT_STAT_COLUMNS = {
    "fg": "FG",
    "fga": "FGA",
    "fg_pct": "FG%",
    "fg3": "3P",
    "fg3a": "3PA",
    "fg3_pct": "3P%",
    "ft": "FT",
    "fta": "FTA",
    "ft_pct": "FT%",
    "orb": "ORB",
    "drb": "DRB",
    "trb": "TRB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TOV",
    "pf": "PF",
    "pts": "PTS",
    "plus_minus": "+/-",
}

COUNTING_STATS = {
    "fg",
    "fga",
    "fg3",
    "fg3a",
    "ft",
    "fta",
    "orb",
    "drb",
    "trb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
}


def parse_minutes_played(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]) + int(parts[1]) / 60.0
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def find_game_basic_tables(soup: BeautifulSoup):
    tables = soup.find_all("table", id=BOX_BASIC_TABLE_ID_RE)
    if tables:
        return tables

    found = []
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "game-basic" not in comment or "box-" not in comment:
            continue
        comment_soup = BeautifulSoup(comment, "html.parser")
        found.extend(comment_soup.find_all("table", id=BOX_BASIC_TABLE_ID_RE))
    return found


def should_skip_player_name(name: str) -> bool:
    n = name.strip().lower()
    if not n or n == "team totals":
        return True
    if "did not play" in n or "did not dress" in n or "not with team" in n:
        return True
    return False


def format_stat_cell(cell, br_key: str):
    if cell is None:
        return ""
    text = cell.get_text(strip=True)
    if text == "":
        return ""
    if br_key == "plus_minus":
        try:
            return int(text)
        except ValueError:
            return text
    if br_key in COUNTING_STATS:
        try:
            return int(float(text))
        except ValueError:
            return text
    return text


def parse_basic_table_rows(table, context: dict):
    rows_out = []
    tbody = table.find("tbody")
    if tbody is None:
        return rows_out

    for tr in tbody.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue

        player_th = tr.find("th", {"data-stat": "player"})
        if player_th is None:
            continue

        name = player_th.get_text(strip=True)
        if should_skip_player_name(name):
            continue

        link = player_th.find("a", href=True)
        if link is None:
            continue

        href = link.get("href", "")
        m = re.search(r"/players/[a-z]/([^/]+)\.html", href)
        if not m:
            continue
        player_id = m.group(1)

        mp_cell = tr.find("td", {"data-stat": "mp"})
        mp_raw = mp_cell.get_text(strip=True) if mp_cell else ""
        mp_val = parse_minutes_played(mp_raw)
        if mp_val is None:
            continue

        record = {
            **context,
            "player_name": name,
            "player_id": player_id,
            "MP": mp_val,
        }

        for br_key in BR_STAT_KEYS:
            col_name = OUTPUT_STAT_COLUMNS[br_key]
            cell = tr.find("td", {"data-stat": br_key})
            record[col_name] = format_stat_cell(cell, br_key)

        rows_out.append(record)

    return rows_out


def game_slug_from_box_url(box_score_url: str) -> str:
    base = os.path.basename(box_score_url or "")
    if base.endswith(".html"):
        return base[: -len(".html")]
    return base or "unknown"


def fetch_box_score_rows(session: requests.Session, box_score_url: str, context: dict):
    url = f"{BASE_ORIGIN}{box_score_url}"
    response = session.get(url, timeout=30)
    if response.status_code == 429:
        print(f"429 Too Many Requests at {url}. Stopping.")
        return None, True
    if response.status_code != 200:
        print(f"Skipping {url} due to HTTP {response.status_code}.")
        return [], False

    soup = BeautifulSoup(response.text, "html.parser")
    tables = find_game_basic_tables(soup)
    if len(tables) < 2:
        print(f"Expected 2 basic tables at {url}, found {len(tables)}.")

    all_rows = []
    for table in tables:
        all_rows.extend(parse_basic_table_rows(table, context))

    return all_rows, False


def main():
    schedule = pd.read_csv(SCHEDULE_CSV)
    schedule["date"] = schedule["date"].astype(str)
    games = schedule[schedule["date"].str.contains(TARGET_YEAR, regex=False)].copy()
    games = games.dropna(subset=["box_score_url"])
    games = games[games["box_score_url"].astype(str).str.len() > 0]
    games = games.drop_duplicates(subset=["box_score_url"]).reset_index(drop=True)

    total = len(games)
    if total == 0:
        print(f"No games found for TARGET_YEAR={TARGET_YEAR!r} in {SCHEDULE_CSV}.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)

    column_order = [
        "Date",
        "Arena",
        "Home_Team",
        "Visitor_Team",
        "player_name",
        "player_id",
        "MP",
        "FG",
        "FGA",
        "FG%",
        "3P",
        "3PA",
        "3P%",
        "FT",
        "FTA",
        "FT%",
        "ORB",
        "DRB",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "+/-",
    ]

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
        )

        for pos, (_, game) in enumerate(games.iterrows(), start=1):
            slug = game_slug_from_box_url(game["box_score_url"])

            context = {
                "Date": game["date"],
                "Arena": game["arena"],
                "Home_Team": game["home_team"],
                "Visitor_Team": game["visitor_team"],
            }

            rows, rate_limited = fetch_box_score_rows(session, game["box_score_url"], context)
            time.sleep(REQUEST_DELAY_SECONDS)

            if rate_limited:
                print("Stopped early due to rate limiting.")
                break

            if not rows:
                print(f"Processed {slug} ({pos}/{total})")
                continue

            batch = pd.DataFrame(rows)
            for col in column_order:
                if col not in batch.columns:
                    batch[col] = ""
            batch = batch[column_order]

            batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
            write_header = False
            print(f"Processed {slug} ({pos}/{total})")

    print(f"Finished. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
