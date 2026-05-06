"""
NBA player points prediction - inference-time feature engineering for the v2 model.

The single public entry point is build_prediction_features(), which fetches all
required data from the DB and returns a one-row DataFrame whose columns match
FEATURES_V2 exactly, ready to be piped into v2_mean_pipeline.pkl /
v2_q10_pipeline.pkl / v2_q90_pipeline.pkl.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import json
import joblib
import argparse
from datetime import datetime
from scipy.stats import norm

# Feature list must stay in sync with trainer_v2.FEATURES_V2
FEATURES_V2 = [
    "is_home", "miles_traveled", "days_rest", "is_back_to_back", "altitude_impact",
    "game_month", "is_playoffs",
    "player_roll5_pts", "player_roll5_mp", "player_ema3_pts",
    "player_roll10_pts", "player_roll10_mp",
    "player_roll10_adv_usg_pct", "player_roll10_adv_ts_pct", "player_roll10_adv_ast_pct",
    "player_roll10_gmsc", "player_roll10_adv_efg_pct", "player_roll10_adv_bpm",
    "player_roll10_adv_ortg", "player_roll10_adv_3par", "player_roll10_adv_tov_pct",
    "player_roll10_plus_minus", "player_roll10_fta", "player_roll10_fg3a",
    "player_roll10_adv_trb_pct",
    "player_roll30_pts", "player_roll30_mp", "player_roll30_adv_usg_pct",
    "player_roll30_gmsc", "player_roll30_adv_bpm",
    "player_ema5_pts", "player_ema5_mp", "player_ema5_adv_usg_pct",
    "player_std10_pts", "player_std10_mp",
    "player_trend_pts",
    "player_mp_trend",
    "player_roll10_pts_per_min",
    "matchup_hist_pts", "matchup_hist_mp",
    "matchup_hist_count", "matchup_hist_std_pts",
    "matchup_pts_diff",
    "team_roll10_pts_scored", "team_roll30_pts_scored",
    "opp_roll10_pts_allowed", "opp_roll30_pts_allowed",
    "opp_roll10_team_drtg", "opp_roll30_team_drtg",
    "opp_roll10_pts_scored",
    "player_roll10_pf",
    "expected_pts_prior",
    "usg_x_opp_drtg",
    "pts_x_team_scoring",
]


# ── DB FETCH HELPERS ──────────────────────────────────────────────────────────

def _get_player_data(conn: sqlite3.Connection, player_name: str) -> pd.DataFrame:
    """Last 30 games for a player, newest first, with all v2 stat columns."""
    query = """
    SELECT g.game_date, p.player_team,
           p.pts, p.mp, p.adv_usg_pct, p.adv_ts_pct, p.adv_ast_pct, p.gmsc,
           p.adv_efg_pct, p.adv_bpm, p.adv_ortg, p.adv_drtg, p.adv_3par,
           p.adv_tov_pct, p.plus_minus, p.fta, p.fg3a, p.adv_trb_pct, p.pf
    FROM Performances p
    JOIN Games g ON p.game_id = g.game_id
    JOIN Players pl ON p.player_id = pl.player_id
    WHERE pl.player_name = ?
    ORDER BY g.game_date DESC
    LIMIT 30
    """
    return pd.read_sql_query(query, conn, params=(player_name,))


def _get_opponent_stats(conn: sqlite3.Connection, opp_team: str) -> pd.DataFrame:
    """Last 30 games involving opp_team — pts_allowed, pts_scored, team_drtg."""
    query = """
    SELECT g.game_id, g.game_date,
           SUM(CASE WHEN p.player_team != ? THEN p.pts ELSE 0 END) AS pts_allowed,
           SUM(CASE WHEN p.player_team  = ? THEN p.pts ELSE 0 END) AS pts_scored,
           AVG(CASE WHEN p.player_team  = ? THEN p.adv_drtg ELSE NULL END) AS team_drtg
    FROM Games g
    JOIN Performances p ON g.game_id = p.game_id
    WHERE g.home_team = ? OR g.visitor_team = ?
    GROUP BY g.game_id, g.game_date
    ORDER BY g.game_date DESC
    LIMIT 30
    """
    return pd.read_sql_query(
        query, conn, params=(opp_team, opp_team, opp_team, opp_team, opp_team)
    )


def _get_team_offense(conn: sqlite3.Connection, player_team: str) -> pd.DataFrame:
    """Last 30 games for player_team — total points scored per game."""
    query = """
    SELECT g.game_id, g.game_date,
           SUM(p.pts) AS team_pts_scored
    FROM Games g
    JOIN Performances p ON g.game_id = p.game_id AND p.player_team = ?
    WHERE g.home_team = ? OR g.visitor_team = ?
    GROUP BY g.game_id, g.game_date
    ORDER BY g.game_date DESC
    LIMIT 30
    """
    return pd.read_sql_query(query, conn, params=(player_team, player_team, player_team))


def _get_matchup_history(conn: sqlite3.Connection,
                         player_name: str, opp_team: str) -> pd.DataFrame:
    """All historical games where player faced opp_team."""
    query = """
    SELECT g.game_date, p.pts, p.mp
    FROM Performances p
    JOIN Games g ON p.game_id = g.game_id
    JOIN Players pl ON p.player_id = pl.player_id
    WHERE pl.player_name = ?
      AND (g.home_team = ? OR g.visitor_team = ?)
      AND p.player_team != ?
    ORDER BY g.game_date DESC
    """
    return pd.read_sql_query(query, conn, params=(player_name, opp_team, opp_team, opp_team))


# ── STAT HELPERS ──────────────────────────────────────────────────────────────

def _roll_mean(series: pd.Series, n: int, min_n: int = 1) -> float:
    """Mean of the first n values (newest first). NaN when fewer than min_n exist."""
    vals = series.dropna().head(n)
    return float(vals.mean()) if len(vals) >= min_n else np.nan


def _roll_std(series: pd.Series, n: int, min_n: int = 2) -> float:
    vals = series.dropna().head(n)
    return float(vals.std()) if len(vals) >= min_n else np.nan


def _ewm(series: pd.Series, span: int, min_periods: int = 2) -> float:
    """EWM of a series stored newest-first. Returns the latest (most recent) value."""
    vals = series.dropna()
    if len(vals) < min_periods:
        return np.nan
    oldest_first = vals.iloc[::-1].reset_index(drop=True)
    return float(oldest_first.ewm(span=span, min_periods=min_periods).mean().iloc[-1])


# ── MAIN INFERENCE FUNCTION ───────────────────────────────────────────────────

def build_prediction_features(
    db_path: str,
    player_name: str,
    opp_team: str,
    is_home: int,
    days_rest: int,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Return (feature_df, None) on success or (None, error_message) on failure.
    feature_df is a one-row DataFrame with columns = FEATURES_V2.
    """
    conn = sqlite3.connect(db_path)
    try:
        player_df = _get_player_data(conn, player_name)
        if len(player_df) < 5:
            return None, f"Not enough data for '{player_name}' (found {len(player_df)} games)."

        player_team = str(player_df["player_team"].iloc[0])
        opp_df = _get_opponent_stats(conn, opp_team)
        if len(opp_df) < 5:
            return None, f"Not enough data for opponent '{opp_team}'."

        team_df = _get_team_offense(conn, player_team)
        matchup_df = _get_matchup_history(conn, player_name, opp_team)
    finally:
        conn.close()

    now = datetime.now()
    game_month = now.month
    is_playoffs = 1 if 4 <= game_month <= 6 else 0

    # ── PLAYER FORM ───────────────────────────────────────────────────────────
    pts = player_df["pts"]
    mp  = player_df["mp"]

    roll5_pts  = _roll_mean(pts, 5, min_n=3)
    roll5_mp   = _roll_mean(mp,  5, min_n=3)
    ema3_pts   = _ewm(pts, span=3, min_periods=2)

    adv_cols_10 = [
        "adv_usg_pct", "adv_ts_pct", "adv_ast_pct", "gmsc",
        "adv_efg_pct", "adv_bpm", "adv_ortg", "adv_3par", "adv_tov_pct",
        "plus_minus", "fta", "fg3a", "adv_trb_pct",
    ]
    roll10 = {c: _roll_mean(player_df[c], 10, min_n=5) for c in adv_cols_10}
    roll10_pts = _roll_mean(pts, 10, min_n=5)
    roll10_mp  = _roll_mean(mp,  10, min_n=5)
    roll10_pf  = _roll_mean(player_df["pf"], 10, min_n=5)

    roll30_pts       = _roll_mean(pts,                      30, min_n=10)
    roll30_mp        = _roll_mean(mp,                       30, min_n=10)
    roll30_adv_usg   = _roll_mean(player_df["adv_usg_pct"], 30, min_n=10)
    roll30_gmsc      = _roll_mean(player_df["gmsc"],        30, min_n=10)
    roll30_adv_bpm   = _roll_mean(player_df["adv_bpm"],     30, min_n=10)

    ema5_pts     = _ewm(pts,                      span=5, min_periods=3)
    ema5_mp      = _ewm(mp,                       span=5, min_periods=3)
    ema5_adv_usg = _ewm(player_df["adv_usg_pct"], span=5, min_periods=3)

    std10_pts = _roll_std(pts, 10, min_n=5)
    std10_mp  = _roll_std(mp,  10, min_n=5)

    trend_pts = (ema5_pts - roll30_pts) if not (np.isnan(ema5_pts) or np.isnan(roll30_pts)) else np.nan
    mp_trend  = (roll5_mp - roll30_mp)  if not (np.isnan(roll5_mp)  or np.isnan(roll30_mp))  else np.nan

    # pts-per-minute — exclude very low-minute games to avoid noise
    valid_mp = mp[mp > 3]
    valid_pts = pts[mp > 3]
    pts_per_min_series = valid_pts / valid_mp
    roll10_pts_per_min = _roll_mean(pts_per_min_series, 10, min_n=5)

    # ── MATCHUP HISTORY ───────────────────────────────────────────────────────
    if len(matchup_df) > 0:
        matchup_pts_s = matchup_df["pts"]
        matchup_mp_s  = matchup_df["mp"]
        hist_pts   = float(matchup_pts_s.mean())
        hist_mp    = float(matchup_mp_s.mean())
        hist_count = float(len(matchup_df))
        hist_std   = float(matchup_pts_s.std()) if len(matchup_df) >= 2 else (std10_pts if not np.isnan(std10_pts) else np.nan)
    else:
        # First time facing this opponent — fall back to player's overall baseline
        hist_pts   = roll30_pts if not np.isnan(roll30_pts) else roll10_pts
        hist_mp    = roll30_mp  if not np.isnan(roll30_mp)  else roll10_mp
        hist_count = 0.0
        hist_std   = std10_pts

    matchup_pts_diff = (hist_pts - roll30_pts) if not (np.isnan(hist_pts) or np.isnan(roll30_pts)) else np.nan

    # ── TEAM OFFENSIVE CONTEXT ────────────────────────────────────────────────
    if len(team_df) >= 5:
        team_roll10_pts = _roll_mean(team_df["team_pts_scored"], 10, min_n=5)
        team_roll30_pts = _roll_mean(team_df["team_pts_scored"], 30, min_n=10)
    else:
        team_roll10_pts = np.nan
        team_roll30_pts = np.nan

    # ── OPPONENT DEFENSE ──────────────────────────────────────────────────────
    opp_pts_allowed   = opp_df["pts_allowed"]
    opp_pts_scored    = opp_df["pts_scored"]
    opp_team_drtg     = opp_df["team_drtg"]

    opp_roll10_pts_allowed = _roll_mean(opp_pts_allowed, 10, min_n=5)
    opp_roll30_pts_allowed = _roll_mean(opp_pts_allowed, 30, min_n=10)
    opp_roll10_drtg        = _roll_mean(opp_team_drtg,   10, min_n=5)
    opp_roll30_drtg        = _roll_mean(opp_team_drtg,   30, min_n=10)
    opp_roll10_pts_scored  = _roll_mean(opp_pts_scored,  10, min_n=5)

    # ── INTERACTION FEATURES ──────────────────────────────────────────────────
    expected_pts_prior = (
        roll10_pts_per_min * ema5_mp
        if not (np.isnan(roll10_pts_per_min) or np.isnan(ema5_mp))
        else np.nan
    )
    usg_x_opp_drtg = (
        roll10["adv_usg_pct"] * opp_roll10_drtg
        if not (np.isnan(roll10["adv_usg_pct"]) or np.isnan(opp_roll10_drtg))
        else np.nan
    )
    pts_x_team_scoring = (
        roll10_pts * team_roll10_pts
        if not (np.isnan(roll10_pts) or np.isnan(team_roll10_pts))
        else np.nan
    )

    # ── ASSEMBLE FEATURE DICT ─────────────────────────────────────────────────
    feat = {
        "is_home":                    int(is_home),
        "miles_traveled":             0.0,   # arena lookup not available at inference
        "days_rest":                  int(days_rest),
        "is_back_to_back":            1 if days_rest <= 1 else 0,
        "altitude_impact":            0,
        "game_month":                 game_month,
        "is_playoffs":                is_playoffs,
        "player_roll5_pts":           roll5_pts,
        "player_roll5_mp":            roll5_mp,
        "player_ema3_pts":            ema3_pts,
        "player_roll10_pts":          roll10_pts,
        "player_roll10_mp":           roll10_mp,
        "player_roll10_adv_usg_pct":  roll10["adv_usg_pct"],
        "player_roll10_adv_ts_pct":   roll10["adv_ts_pct"],
        "player_roll10_adv_ast_pct":  roll10["adv_ast_pct"],
        "player_roll10_gmsc":         roll10["gmsc"],
        "player_roll10_adv_efg_pct":  roll10["adv_efg_pct"],
        "player_roll10_adv_bpm":      roll10["adv_bpm"],
        "player_roll10_adv_ortg":     roll10["adv_ortg"],
        "player_roll10_adv_3par":     roll10["adv_3par"],
        "player_roll10_adv_tov_pct":  roll10["adv_tov_pct"],
        "player_roll10_plus_minus":   roll10["plus_minus"],
        "player_roll10_fta":          roll10["fta"],
        "player_roll10_fg3a":         roll10["fg3a"],
        "player_roll10_adv_trb_pct":  roll10["adv_trb_pct"],
        "player_roll30_pts":          roll30_pts,
        "player_roll30_mp":           roll30_mp,
        "player_roll30_adv_usg_pct":  roll30_adv_usg,
        "player_roll30_gmsc":         roll30_gmsc,
        "player_roll30_adv_bpm":      roll30_adv_bpm,
        "player_ema5_pts":            ema5_pts,
        "player_ema5_mp":             ema5_mp,
        "player_ema5_adv_usg_pct":    ema5_adv_usg,
        "player_std10_pts":           std10_pts,
        "player_std10_mp":            std10_mp,
        "player_trend_pts":           trend_pts,
        "player_mp_trend":            mp_trend,
        "player_roll10_pts_per_min":  roll10_pts_per_min,
        "matchup_hist_pts":           hist_pts,
        "matchup_hist_mp":            hist_mp,
        "matchup_hist_count":         hist_count,
        "matchup_hist_std_pts":       hist_std,
        "matchup_pts_diff":           matchup_pts_diff,
        "team_roll10_pts_scored":     team_roll10_pts,
        "team_roll30_pts_scored":     team_roll30_pts,
        "opp_roll10_pts_allowed":     opp_roll10_pts_allowed,
        "opp_roll30_pts_allowed":     opp_roll30_pts_allowed,
        "opp_roll10_team_drtg":       opp_roll10_drtg,
        "opp_roll30_team_drtg":       opp_roll30_drtg,
        "opp_roll10_pts_scored":      opp_roll10_pts_scored,
        "player_roll10_pf":           roll10_pf,
        "expected_pts_prior":         expected_pts_prior,
        "usg_x_opp_drtg":             usg_x_opp_drtg,
        "pts_x_team_scoring":         pts_x_team_scoring,
    }

    X = pd.DataFrame([feat])[FEATURES_V2]
    return X, None


# ── CONTEXT HELPERS (used by app.py for response enrichment) ─────────────────

def get_player_recent_stats(db_path: str, player_name: str) -> pd.DataFrame:
    """Last 10 games for display purposes (date + pts + mp + opponent)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT g.game_date, p.pts, p.mp, p.adv_usg_pct, p.adv_ts_pct,
               CASE WHEN p.player_team = g.home_team THEN g.visitor_team
                    ELSE g.home_team END AS opponent
        FROM Performances p
        JOIN Games g ON p.game_id = g.game_id
        JOIN Players pl ON p.player_id = pl.player_id
        WHERE pl.player_name = ?
        ORDER BY g.game_date DESC LIMIT 10
        """,
        conn, params=(player_name,),
    )
    conn.close()
    return df


def get_opponent_recent_defense(db_path: str, opp_team: str) -> pd.DataFrame:
    """Last 10 games of pts_allowed for the opponent team."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT g.game_id, g.game_date,
               SUM(CASE WHEN p.player_team != ? THEN p.pts ELSE 0 END) AS pts_allowed
        FROM Games g
        JOIN Performances p ON g.game_id = p.game_id
        WHERE g.home_team = ? OR g.visitor_team = ?
        GROUP BY g.game_id, g.game_date
        ORDER BY g.game_date DESC LIMIT 10
        """,
        conn, params=(opp_team, opp_team, opp_team),
    )
    conn.close()
    return df


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Player Prop Predictor (v2)")
    parser.add_argument("--player",   required=True)
    parser.add_argument("--opp",      required=True, help="Opponent team acronym")
    parser.add_argument("--line",     type=float, required=True)
    parser.add_argument("--is_home",  type=int, default=1, choices=[0, 1])
    parser.add_argument("--days_rest",type=int, default=2)
    args = parser.parse_args()

    base_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path   = os.path.join(base_dir, "data", "nba_contextual.db")
    models_dir = os.path.join(base_dir, "models")

    X, err = build_prediction_features(db_path, args.player, args.opp, args.is_home, args.days_rest)
    if err:
        print(f"Error: {err}")
        return

    mean_pipe = joblib.load(os.path.join(models_dir, "v2_mean_pipeline.pkl"))
    q10_pipe  = joblib.load(os.path.join(models_dir, "v2_q10_pipeline.pkl"))
    q90_pipe  = joblib.load(os.path.join(models_dir, "v2_q90_pipeline.pkl"))

    expected_pts = float(mean_pipe.predict(X)[0])
    low_pts      = float(q10_pipe.predict(X)[0])
    high_pts     = float(q90_pipe.predict(X)[0])

    # P(over) using the quantile interval as a proxy for the distribution
    mid  = expected_pts
    half_width = max((high_pts - low_pts) / 2.0, 1.0)
    prob_over  = 1.0 - norm.cdf(args.line, loc=mid, scale=half_width * 0.78)  # ~80% interval ≈ 1.28σ
    prob_under = 1.0 - prob_over

    print("=" * 45)
    print(f"  PROP PREDICTOR (v2): {args.player.upper()}")
    print(f"  vs {args.opp.upper()}  |  {'HOME' if args.is_home else 'AWAY'}  |  {args.days_rest}d rest")
    print("=" * 45)
    print(f"  Expected pts:    {expected_pts:.1f}")
    print(f"  80% interval:    [{low_pts:.1f}, {high_pts:.1f}]")
    print(f"  Line:            {args.line}")
    print(f"  P(Over):         {prob_over:.1%}")
    print(f"  P(Under):        {prob_under:.1%}")
    print("=" * 45)


if __name__ == "__main__":
    main()
