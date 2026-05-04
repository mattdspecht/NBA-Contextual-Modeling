import os
import re
import time
from bs4 import BeautifulSoup, Comment
import pandas as pd
import requests

TARGET_YEAR = ["2021","2022","2023","2024","2025","2026"]
VERSION = "v2"
SCHEDULE_CSV = "data/raw/nba_schedule_2021_2026.csv"
OUTPUT_CSV_TEMPLATE = "data/raw/nba_player_stats_{year}_{version}.csv"

BASE_ORIGIN = "https://www.basketball-reference.com"
REQUEST_DELAY_SECONDS = 3.1
RATE_LIMIT_COOLDOWN_SECONDS = 300
MAX_429_RETRIES = 12

BOX_TABLE_ID_RE = re.compile(r"^box-([A-Z]+)-game-(basic|advanced)$")

BASIC_BR_TO_COL = {
    "mp": "MP",
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
    "game_score": "GmSc",
    "plus_minus": "+/-",
}

BASIC_COUNTING = {
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

BASIC_FLOAT_STATS = {"game_score", "fg_pct", "fg3_pct", "ft_pct"}

ADVANCED_BR_TO_COL = {
    "ts_pct": "adv_TS%",
    "efg_pct": "adv_eFG%",
    "fg3a_per_fga_pct": "adv_3PAr",
    "fta_per_fga_pct": "adv_FTr",
    "orb_pct": "adv_ORB%",
    "drb_pct": "adv_DRB%",
    "trb_pct": "adv_TRB%",
    "ast_pct": "adv_AST%",
    "stl_pct": "adv_STL%",
    "blk_pct": "adv_BLK%",
    "tov_pct": "adv_TOV%",
    "usg_pct": "adv_USG%",
    "off_rtg": "adv_ORtg",
    "def_rtg": "adv_DRtg",
    "bpm": "adv_BPM",
}

CONTEXT_COLUMNS = [
    "Date",
    "Arena",
    "Visitor_Team",
    "Home_Team",
    "player_team",
    "is_home",
    "opponent",
    "player_name",
    "player_id",
]

BASIC_COLUMN_ORDER = [BASIC_BR_TO_COL[k] for k in BASIC_BR_TO_COL]
ADVANCED_COLUMN_ORDER = list(ADVANCED_BR_TO_COL.values())

OUTPUT_COLUMN_ORDER = (
    CONTEXT_COLUMNS
    + BASIC_COLUMN_ORDER
    + ADVANCED_COLUMN_ORDER
)


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


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def find_box_tables_in_soup(soup: BeautifulSoup):
    tables = {}
    for table in soup.find_all("table", id=BOX_TABLE_ID_RE):
        m = BOX_TABLE_ID_RE.match(table.get("id", ""))
        if m:
            tables[(m.group(1), m.group(2))] = table

    if tables:
        return tables

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "box-" not in comment or "game-" not in comment:
            continue
        inner = soup_from_html(comment)
        for table in inner.find_all("table", id=BOX_TABLE_ID_RE):
            m = BOX_TABLE_ID_RE.match(table.get("id", ""))
            if m:
                tables[(m.group(1), m.group(2))] = table
    return tables


def player_team_full_name(soup: BeautifulSoup, table_id: str) -> str:
    anchor = soup.find("span", id=f"{table_id}_link")
    if anchor is None:
        return ""
    label = anchor.get("data-label") or ""
    return label.replace(" Basic and Advanced Stats", "").strip()


def should_skip_player_name(name: str) -> bool:
    n = name.strip().lower()
    if not n or n == "team totals":
        return True
    if "did not play" in n or "did not dress" in n or "not with team" in n:
        return True
    return False


def format_basic_cell(br_key: str, cell):
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
    if br_key in BASIC_COUNTING:
        try:
            return int(float(text))
        except ValueError:
            return text
    if br_key in BASIC_FLOAT_STATS:
        try:
            return float(text)
        except ValueError:
            return text
    return text


def format_advanced_cell(cell):
    if cell is None:
        return ""
    text = cell.get_text(strip=True)
    if text == "":
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def parse_advanced_rows_by_player_id(table):
    by_id = {}
    tbody = table.find("tbody")
    if tbody is None:
        return by_id

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
        pid = m.group(1)
        row = {}
        for br_key, col_name in ADVANCED_BR_TO_COL.items():
            if br_key == "mp":
                continue
            cell = tr.find("td", {"data-stat": br_key})
            row[col_name] = format_advanced_cell(cell)
        by_id[pid] = row
    return by_id


def parse_basic_rows_for_team(table, soup: BeautifulSoup, home_team: str, visitor_team: str, base_context: dict):
    rows = []
    table_id = table.get("id", "")
    player_team = player_team_full_name(soup, table_id)
    if not player_team:
        abbr_match = re.match(r"^box-([A-Z]+)-game-basic$", table_id)
        player_team = abbr_match.group(1) if abbr_match else ""

    ht = str(home_team).strip()
    vt = str(visitor_team).strip()
    is_home = bool(player_team and ht and player_team == ht)
    opponent = vt if is_home else ht

    team_context = {
        **base_context,
        "Visitor_Team": visitor_team,
        "Home_Team": home_team,
        "player_team": player_team,
        "is_home": is_home,
        "opponent": opponent,
    }

    tbody = table.find("tbody")
    if tbody is None:
        return rows

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
            **team_context,
            "player_name": name,
            "player_id": player_id,
            "MP": mp_val,
        }

        for br_key, col_name in BASIC_BR_TO_COL.items():
            if br_key == "mp":
                continue
            cell = tr.find("td", {"data-stat": br_key})
            record[col_name] = format_basic_cell(br_key, cell)

        rows.append(record)

    return rows


def parse_box_score_from_soup(soup: BeautifulSoup, home_team: str, visitor_team: str, base_context: dict):
    tables = find_box_tables_in_soup(soup)
    abbrs = {abbr for (abbr, kind) in tables}

    advanced_by_abbr = {}
    for abbr in abbrs:
        adv = tables.get((abbr, "advanced"))
        if adv is not None:
            advanced_by_abbr[abbr] = parse_advanced_rows_by_player_id(adv)

    all_rows = []
    for abbr in sorted(abbrs):
        basic = tables.get((abbr, "basic"))
        if basic is None:
            continue
        team_rows = parse_basic_rows_for_team(
            basic, soup, home_team=home_team, visitor_team=visitor_team, base_context=base_context
        )
        adv_map = advanced_by_abbr.get(abbr, {})
        for rec in team_rows:
            pid = rec["player_id"]
            adv_stats = adv_map.get(pid, {})
            for col in ADVANCED_COLUMN_ORDER:
                rec[col] = adv_stats.get(col, "")
            all_rows.append(rec)

    return all_rows


def game_slug_from_box_url(box_score_url: str) -> str:
    base = os.path.basename(box_score_url or "")
    if base.endswith(".html"):
        return base[: -len(".html")]
    return base or "unknown"


def fetch_with_429_retry(session: requests.Session, url: str):
    response = None
    for attempt in range(MAX_429_RETRIES):
        response = session.get(url, timeout=60)
        if response.status_code == 429:
            print(
                "\n"
                + "#" * 80
                + "\n"
                + "###  HTTP 429 — TOO MANY REQUESTS  ###\n"
                + "###  Basketball-Reference rate limit.  ###\n"
                + f"###  Sleeping {RATE_LIMIT_COOLDOWN_SECONDS}s then retrying same URL.  ###\n"
                + f"###  (attempt {attempt + 1} of {MAX_429_RETRIES})  ###\n"
                + "#" * 80
                + "\n"
            )
            time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
            continue
        return response
    if response is not None and response.status_code == 429:
        print(f"Still HTTP 429 after {MAX_429_RETRIES} attempts; giving up on this URL.")
    return response


def fetch_and_parse_box_score(session: requests.Session, box_score_url: str, game_row: dict):
    url = f"{BASE_ORIGIN}{box_score_url}"
    response = fetch_with_429_retry(session, url)

    if response.status_code != 200:
        print(f"Skipping {url} due to HTTP {response.status_code}.")
        return []

    soup = soup_from_html(response.text)
    tables = find_box_tables_in_soup(soup)
    if len([k for k in tables if k[1] == "basic"]) < 2:
        print(f"Expected 2 basic tables at {url}, found fewer.")

    base_context = {
        "Date": game_row["date"],
        "Arena": game_row["arena"],
    }

    rows = parse_box_score_from_soup(
        soup,
        home_team=game_row["home_team"],
        visitor_team=game_row["visitor_team"],
        base_context=base_context,
    )
    return rows


def main():
    if not TARGET_YEAR:
        print("TARGET_YEAR is empty; add at least one calendar year string, e.g. ['2024'].")
        return

    schedule = pd.read_csv(SCHEDULE_CSV)
    schedule["date"] = schedule["date"].astype(str)

    first_year = TARGET_YEAR[0]
    os.makedirs(
        os.path.dirname(OUTPUT_CSV_TEMPLATE.format(year=first_year, version=VERSION)) or ".",
        exist_ok=True,
    )

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

        for calendar_year in TARGET_YEAR:
            games = schedule[schedule["date"].str.contains(calendar_year, regex=False)].copy()
            games = games.dropna(subset=["box_score_url"])
            games = games[games["box_score_url"].astype(str).str.len() > 0]
            games = games.drop_duplicates(subset=["box_score_url"]).reset_index(drop=True)

            output_csv = OUTPUT_CSV_TEMPLATE.format(year=calendar_year, version=VERSION)
            total = len(games)

            if total == 0:
                print(f"No games found for year {calendar_year!r} in {SCHEDULE_CSV}.")
                continue

            write_header = not os.path.exists(output_csv)
            print(f"Starting year {calendar_year}: {total} games -> {output_csv}")
            year_t0 = time.perf_counter()

            for pos, (_, game) in enumerate(games.iterrows(), start=1):
                slug = game_slug_from_box_url(game["box_score_url"])
                game_dict = {
                    "date": game["date"],
                    "arena": game["arena"],
                    "home_team": game["home_team"],
                    "visitor_team": game["visitor_team"],
                    "box_score_url": game["box_score_url"],
                }

                rows = fetch_and_parse_box_score(session, game["box_score_url"], game_dict)
                time.sleep(REQUEST_DELAY_SECONDS)

                if not rows:
                    print(f"[{calendar_year}] Processed {slug} ({pos}/{total})")
                    continue

                batch = pd.DataFrame(rows)
                for col in OUTPUT_COLUMN_ORDER:
                    if col not in batch.columns:
                        batch[col] = ""
                batch = batch[OUTPUT_COLUMN_ORDER]

                batch.to_csv(output_csv, mode="a", header=write_header, index=False)
                write_header = False
                print(f"[{calendar_year}] Processed {slug} ({pos}/{total})")

            year_elapsed = time.perf_counter() - year_t0
            if year_elapsed < 3600:
                elapsed_str = f"{year_elapsed / 60:.1f} minutes"
            else:
                h, rem = divmod(year_elapsed, 3600)
                m, s = divmod(rem, 60)
                elapsed_str = f"{int(h)}h {int(m)}m {s:.0f}s"
            print(
                f"Finished year {calendar_year}. Output: {output_csv} "
                f"(time elapsed: {elapsed_str})"
            )

    print("Done.")


if __name__ == "__main__":
    main()
