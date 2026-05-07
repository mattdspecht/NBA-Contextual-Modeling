"""
Transform raw NBA box score CSVs into a machine-learning-ready dataset.

Loads all `datasets/raw/*.csv`, merges arena coordinates, engineers travel/rest
features, applies cleaning rules, and writes `datasets/processed/performances.csv`.
"""

from __future__ import annotations

import glob
import math
import os

import pandas as pd

DEBUG_MODE = False

RAW_DATA_GLOB = os.path.join("datasets", "raw", "*.csv")
ARENA_COORDS_PATH = os.path.join("datasets", "assets", "arena_coords.csv")
PROCESSED_OUTPUT_PATH = os.path.join(
    "datasets", "processed", "performances.csv"
)

EARTH_RADIUS_MILES = 3958.8

HIGH_ALTITUDE_ARENA_KEYS = frozenset({"ball arena", "vivint arena"})


def load_raw_csvs(pattern: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found for pattern: {pattern}")
    frames = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def load_arena_coords(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"arena_name", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Arena coords file missing columns: {sorted(missing)}")
    df = df.copy()
    df["arena_key"] = df["arena_name"].astype(str).str.strip().str.casefold()
    return df.drop_duplicates(subset=["arena_key"], keep="first")


def merge_arena_coordinates(master: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["arena_key"] = master["Arena"].str.strip().str.casefold()
    coord_cols = coords[["arena_key", "latitude", "longitude"]].rename(
        columns={"latitude": "arena_lat", "longitude": "arena_lon"}
    )
    out = master.merge(coord_cols, on="arena_key", how="left")
    has_arena = out["Arena"].notna()
    missing_mask = has_arena & (out["arena_lat"].isna() | out["arena_lon"].isna())
    if missing_mask.any():
        bad = (
            out.loc[missing_mask, "Arena"]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        bad.sort()
        raise ValueError(
            "Arena coordinate merge failed for the following arena name(s) "
            f"({len(bad)} unique): {bad}"
        )
    return out


def haversine_miles_scalar(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance in miles. Returns NaN if any input coordinate is missing."""
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    rlat1, rlon1 = math.radians(float(lat1)), math.radians(float(lon1))
    rlat2, rlon2 = math.radians(float(lat2)), math.radians(float(lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(
        dlon / 2.0
    ) ** 2
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def haversine_miles_columnwise(df: pd.DataFrame) -> pd.Series:
    """Pairwise Haversine miles for current vs previous arena coordinates."""
    out: list[float] = []
    for a, b, c, d in zip(
        df["arena_lat"],
        df["arena_lon"],
        df["prev_arena_lat"],
        df["prev_arena_lon"],
    ):
        out.append(haversine_miles_scalar(a, b, c, d))
    return pd.Series(out, index=df.index, dtype="float64")


def add_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["player_id", "Date"], ascending=True).reset_index(drop=True)

    g = df.groupby("player_id", sort=False)
    df["prev_arena_lat"] = g["arena_lat"].shift(1)
    df["prev_arena_lon"] = g["arena_lon"].shift(1)
    df["prev_date"] = g["Date"].shift(1)

    df["miles_traveled"] = haversine_miles_columnwise(df)
    df.loc[df["prev_date"].isna(), "miles_traveled"] = 0.0

    df["days_rest"] = (df["Date"] - df["prev_date"]).dt.days

    long_rest = df["days_rest"] > 10
    df.loc[long_rest, "miles_traveled"] = 0.0
    df.loc[long_rest, "days_rest"] = 10

    df["is_back_to_back"] = df["days_rest"].eq(1)

    return df


def fill_percentage_nans_with_zero(df: pd.DataFrame) -> pd.DataFrame:
    pct_cols = [c for c in df.columns if "%" in c]
    if not pct_cols:
        return df
    df = df.copy()
    df[pct_cols] = df[pct_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def finalize_mp_and_altitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MP"] = pd.to_numeric(df["MP"], errors="coerce")
    df = df.dropna(subset=["MP"])
    df = df.loc[df["MP"] != 0].copy()

    arena_key = df["Arena"].astype(str).str.strip().str.casefold()
    df["altitude_impact"] = arena_key.isin(HIGH_ALTITUDE_ARENA_KEYS)

    return df


def print_summary(df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_players = df["player_id"].nunique()
    coord_nan = int(df["arena_lat"].isna().sum() + df["arena_lon"].isna().sum())
    print(f"Total rows: {n_rows}")
    print(f"Total unique players: {n_players}")
    if coord_nan == 0:
        print("Coordinates: no NaN values in arena_lat / arena_lon.")
    else:
        print(f"WARNING: Found {coord_nan} NaN coordinate cells (should be 0).")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo_root)

    df = load_raw_csvs(RAW_DATA_GLOB)
    if DEBUG_MODE:
        df = df.head(500).copy()

    coords = load_arena_coords(ARENA_COORDS_PATH)
    df = merge_arena_coordinates(df, coords)
    df = add_fatigue_features(df)
    df = fill_percentage_nans_with_zero(df)
    df = finalize_mp_and_altitude(df)

    df = df.drop(columns=["arena_key"], errors="ignore")

    os.makedirs(os.path.dirname(PROCESSED_OUTPUT_PATH), exist_ok=True)
    df.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print_summary(df)
    print(f"Wrote processed dataset to: {PROCESSED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
