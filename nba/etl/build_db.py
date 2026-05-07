# Load all performances into our DB (3NF schema)

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "datasets" / "processed" / "performances.csv"
DB_PATH = ROOT / "datasets" / "game_db.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "db_schema.sql"

PERF_CHUNK_SIZE = 5000


def _bool_to_int(val) -> int:
    if pd.isna(val):
        return 0
    if isinstance(val, bool):
        return int(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    if s in ("false", "0", "no", ""):
        return 0
    return 1 if val else 0


def _safe_int(val, default: int = 0) -> int:
    if pd.isna(val):
        return default
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_game_id(df: pd.DataFrame) -> pd.Series:
    date_part = df["Date"].astype(str).str.replace("-", "", regex=False)
    return date_part + "_" + df["Home_Team"].astype(str)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = build_game_id(out)
    return out


def run() -> None:
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = prepare_dataframe(df)
    print(f"  Rows: {len(df):,}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    print(f"Applying schema: {SCHEMA_PATH}")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())

    teams = pd.concat(
        [df["Home_Team"], df["Visitor_Team"]], ignore_index=True
    ).dropna()
    teams = teams.astype(str).unique()
    print(f"Inserting Teams (unique): {len(teams):,}")
    conn.executemany(
        "INSERT OR IGNORE INTO Teams (team_acronym) VALUES (?);",
        ((t,) for t in teams),
    )
    conn.commit()
    print("  Teams insert committed.")

    arenas = (
        df[["Arena", "arena_lat", "arena_lon"]]
        .dropna(subset=["Arena", "arena_lat", "arena_lon"])
        .drop_duplicates(subset=["Arena", "arena_lat", "arena_lon"])
    )
    print(f"Inserting Arenas (unique arena/lat/lon): {len(arenas):,}")
    conn.executemany(
        "INSERT OR IGNORE INTO Arenas (arena_name, latitude, longitude) VALUES (?, ?, ?);",
        (
            (
                row.Arena,
                float(row.arena_lat),
                float(row.arena_lon),
            )
            for row in arenas.itertuples(index=False)
        ),
    )
    conn.commit()
    print("  Arenas insert committed.")

    players = df[["player_id", "player_name"]].drop_duplicates(subset=["player_id"])
    players = players.dropna(subset=["player_id", "player_name"])
    print(f"Inserting Players (unique player_id): {len(players):,}")
    conn.executemany(
        "INSERT OR IGNORE INTO Players (player_id, player_name) VALUES (?, ?);",
        (
            (str(row.player_id), str(row.player_name))
            for row in players.itertuples(index=False)
        ),
    )
    conn.commit()
    print("  Players insert committed.")

    games = df[
        ["game_id", "Date", "Home_Team", "Visitor_Team", "Arena"]
    ].drop_duplicates(subset=["game_id"])
    games = games.dropna(subset=["game_id", "Date", "Home_Team", "Visitor_Team", "Arena"])
    print(f"Inserting Games (unique game_id): {len(games):,}")
    conn.executemany(
        """
        INSERT OR IGNORE INTO Games
        (game_id, game_date, home_team, visitor_team, arena_name)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            (
                str(row.game_id),
                str(row.Date),
                str(row.Home_Team),
                str(row.Visitor_Team),
                str(row.Arena),
            )
            for row in games.itertuples(index=False)
        ),
    )
    conn.commit()
    print("  Games insert committed.")

    # performances
    perf_sql = """
    INSERT OR IGNORE INTO Performances (
        game_id, player_id, player_team, is_home,
        miles_traveled, days_rest, is_back_to_back, altitude_impact,
        mp, fg, fga, fg_pct, fg3, fg3a, fg3_pct,
        ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, tov, pf, pts,
        gmsc, plus_minus,
        adv_ts_pct, adv_efg_pct, adv_3par, adv_ftr,
        adv_orb_pct, adv_drb_pct, adv_trb_pct, adv_ast_pct,
        adv_stl_pct, adv_blk_pct, adv_tov_pct, adv_usg_pct,
        adv_ortg, adv_drtg, adv_bpm
    ) VALUES (
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?
    );
    """

    d = df
    days_rest = d["days_rest"].apply(lambda x: _safe_int(x, 10))
    perf_df = pd.DataFrame(
        {
            "game_id": d["game_id"].astype(str),
            "player_id": d["player_id"].astype(str),
            "player_team": d["player_team"].astype(str),
            "is_home": d["is_home"].map(_bool_to_int),
            "miles_traveled": d["miles_traveled"].apply(lambda x: _safe_float(x, 0.0)),
            "days_rest": days_rest,
            "is_back_to_back": d["is_back_to_back"].map(_bool_to_int),
            "altitude_impact": d["altitude_impact"].map(_bool_to_int),
            "mp": d["MP"].astype(float),
            "fg": d["FG"].apply(lambda x: _safe_int(x, 0)),
            "fga": d["FGA"].apply(lambda x: _safe_int(x, 0)),
            "fg_pct": d["FG%"].apply(lambda x: _safe_float(x, 0.0)),
            "fg3": d["3P"].apply(lambda x: _safe_int(x, 0)),
            "fg3a": d["3PA"].apply(lambda x: _safe_int(x, 0)),
            "fg3_pct": d["3P%"].apply(lambda x: _safe_float(x, 0.0)),
            "ft": d["FT"].apply(lambda x: _safe_int(x, 0)),
            "fta": d["FTA"].apply(lambda x: _safe_int(x, 0)),
            "ft_pct": d["FT%"].apply(lambda x: _safe_float(x, 0.0)),
            "orb": d["ORB"].apply(lambda x: _safe_int(x, 0)),
            "drb": d["DRB"].apply(lambda x: _safe_int(x, 0)),
            "trb": d["TRB"].apply(lambda x: _safe_int(x, 0)),
            "ast": d["AST"].apply(lambda x: _safe_int(x, 0)),
            "stl": d["STL"].apply(lambda x: _safe_int(x, 0)),
            "blk": d["BLK"].apply(lambda x: _safe_int(x, 0)),
            "tov": d["TOV"].apply(lambda x: _safe_int(x, 0)),
            "pf": d["PF"].apply(lambda x: _safe_int(x, 0)),
            "pts": d["PTS"].apply(lambda x: _safe_int(x, 0)),
            "gmsc": d["GmSc"].apply(lambda x: _safe_float(x, 0.0)),
            "plus_minus": d["+/-"].apply(lambda x: _safe_int(x, 0)),
            "adv_ts_pct": d["adv_TS%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_efg_pct": d["adv_eFG%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_3par": d["adv_3PAr"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_ftr": d["adv_FTr"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_orb_pct": d["adv_ORB%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_drb_pct": d["adv_DRB%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_trb_pct": d["adv_TRB%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_ast_pct": d["adv_AST%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_stl_pct": d["adv_STL%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_blk_pct": d["adv_BLK%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_tov_pct": d["adv_TOV%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_usg_pct": d["adv_USG%"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_ortg": d["adv_ORtg"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_drtg": d["adv_DRtg"].apply(lambda x: _safe_float(x, 0.0)),
            "adv_bpm": d["adv_BPM"].apply(lambda x: _safe_float(x, 0.0)),
        }
    )

    total_perf = len(perf_df)
    print(f"Inserting Performances: {total_perf:,} rows in chunks of {PERF_CHUNK_SIZE:,}...")

    for start in range(0, total_perf, PERF_CHUNK_SIZE):
        end = min(start + PERF_CHUNK_SIZE, total_perf)
        sub = perf_df.iloc[start:end]
        chunk = [tuple(row) for row in sub.to_numpy()]
        conn.executemany(perf_sql, chunk)
        conn.commit()
        print(f"  Performances: committed rows {start + 1:,}–{end:,} / {total_perf:,}")

    print("\nFinal table row counts:")
    for table in ("Arenas", "Teams", "Players", "Games", "Performances"):
        (n,) = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()
        print(f"  {table}: {n:,}")

    conn.close()
    print(f"\nDatabase written to: {DB_PATH}")


if __name__ == "__main__":
    run()
