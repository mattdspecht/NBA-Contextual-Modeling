import os
import time
from bs4 import BeautifulSoup, Comment
import pandas as pd
import requests


BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
SEASONS = range(2021, 2027)
MONTHS = [
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
]
OUTPUT_PATH = "datasets/raw/schedule.csv"
REQUEST_DELAY_SECONDS = 3.1


def get_schedule_table(soup: BeautifulSoup):
    table = soup.find("table", id="schedule")
    if table is not None:
        return table

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "id=\"schedule\"" not in comment:
            continue
        comment_soup = BeautifulSoup(comment, "html.parser")
        commented_table = comment_soup.find("table", id="schedule")
        if commented_table is not None:
            return commented_table

    return None


def parse_schedule_table(table, source_url: str):
    rows = []
    tbody = table.find("tbody")
    if tbody is None:
        return rows

    for row in tbody.find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue

        date_cell = row.find("th", {"data-stat": "date_game"})
        visitor_cell = row.find("td", {"data-stat": "visitor_team_name"})
        home_cell = row.find("td", {"data-stat": "home_team_name"})
        arena_cell = row.find("td", {"data-stat": "arena_name"})
        box_score_cell = row.find("td", {"data-stat": "box_score_text"})

        if date_cell is None or visitor_cell is None or home_cell is None:
            continue

        box_score_link_tag = box_score_cell.find("a") if box_score_cell is not None else None
        box_score_url = box_score_link_tag.get("href", "").strip() if box_score_link_tag else ""

        rows.append(
            {
                "date": date_cell.get_text(strip=True),
                "visitor_team": visitor_cell.get_text(strip=True),
                "home_team": home_cell.get_text(strip=True),
                "arena": arena_cell.get_text(strip=True) if arena_cell is not None else "",
                "box_score_url": box_score_url,
                "source_url": source_url,
            }
        )

    return rows


def fetch_month_schedule(session: requests.Session, year: int, month: str):
    url = BASE_URL.format(year=year, month=month)
    print(f"Scraping {month.capitalize()} {year}...")

    response = session.get(url, timeout=30)
    if response.status_code == 429:
        print(f"429 Too Many Requests received at {url}. Cooling down and stopping scrape.")
        return None, True

    if response.status_code != 200:
        print(f"Skipping {url} due to HTTP {response.status_code}.")
        return [], False

    soup = BeautifulSoup(response.text, "html.parser")
    table = get_schedule_table(soup)
    if table is None:
        print(f"No schedule table found for {month.capitalize()} {year}.")
        return [], False

    rows = parse_schedule_table(table, source_url=url)
    print(f"Scraping {month.capitalize()} {year}... Done ({len(rows)} games)")
    return rows, False


def main():
    all_rows = []
    should_stop = False

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

        for year in SEASONS:
            for month in MONTHS:
                rows, hit_rate_limit = fetch_month_schedule(session, year, month)
                time.sleep(REQUEST_DELAY_SECONDS)

                if hit_rate_limit:
                    should_stop = True
                    break

                if rows:
                    month_df = pd.DataFrame(rows)
                    all_rows.append(month_df)

            if should_stop:
                break

    if all_rows:
        master_df = pd.concat(all_rows, ignore_index=True)
    else:
        master_df = pd.DataFrame(
            columns=["date", "visitor_team", "home_team", "arena", "box_score_url", "source_url"]
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    master_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(master_df)} rows to {OUTPUT_PATH}")

    if should_stop:
        print("Scrape stopped early due to rate limiting (429).")


if __name__ == "__main__":
    main()
