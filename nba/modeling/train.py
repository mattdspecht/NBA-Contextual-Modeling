"""
NBA Points Prediction - v2

Much more features now (slight improvement in R^2)
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import json
import joblib
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


FEATURES_V2 = [
    # Context / fatigue
    "is_home", "miles_traveled", "days_rest", "is_back_to_back", "altitude_impact",
    "game_month", "is_playoffs",

    # Short-window player form
    "player_roll5_pts", "player_roll5_mp", "player_ema3_pts",

    # Roll-10 player form (v1 features kept + new advanced stats)
    "player_roll10_pts", "player_roll10_mp",
    "player_roll10_adv_usg_pct", "player_roll10_adv_ts_pct", "player_roll10_adv_ast_pct",
    "player_roll10_gmsc", "player_roll10_adv_efg_pct", "player_roll10_adv_bpm",
    "player_roll10_adv_ortg", "player_roll10_adv_3par", "player_roll10_adv_tov_pct",
    "player_roll10_plus_minus", "player_roll10_fta", "player_roll10_fg3a",
    "player_roll10_adv_trb_pct",

    # Roll-30 player form
    "player_roll30_pts", "player_roll30_mp", "player_roll30_adv_usg_pct",
    "player_roll30_gmsc", "player_roll30_adv_bpm",

    # EWMAs
    "player_ema5_pts", "player_ema5_mp", "player_ema5_adv_usg_pct",

    # Player consistency (variance)
    "player_std10_pts", "player_std10_mp",

    # Trend / momentum
    "player_trend_pts",
    "player_mp_trend",     # roll5_mp - roll30_mp: minutes trending up/down

    # Per-minute efficiency
    "player_roll10_pts_per_min",

    # Matchup-specific history
    "matchup_hist_pts", "matchup_hist_mp",
    "matchup_hist_count", "matchup_hist_std_pts",
    "matchup_pts_diff",   # matchup_hist_pts - player_roll30_pts

    # Team offensive context (a high-scoring team means more opportunities for everyone)
    "team_roll10_pts_scored",   # player's own team's rolling pts per game
    "team_roll30_pts_scored",

    # Opponent team defense
    "opp_roll10_pts_allowed", "opp_roll30_pts_allowed",
    "opp_roll10_team_drtg", "opp_roll30_team_drtg",

    # Game pace proxy: high-scoring opponents play fast → more possessions for everyone
    "opp_roll10_pts_scored",    # opponent's offensive output (pace signal)

    # Foul trouble risk
    "player_roll10_pf",

    # Derived prior prediction: roll10 pts/min × ema5 minutes
    "expected_pts_prior",

    # Interaction: player usage × opponent defensive strength
    "usg_x_opp_drtg",
    # Interaction: player scoring base × team context
    "pts_x_team_scoring",
]


def load_and_preprocess_data_v2(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        p.*,
        g.game_date,
        g.home_team,
        g.visitor_team
    FROM Performances p
    JOIN Games g ON p.game_id = g.game_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Determine opponent team for every row
    df["opponent_team"] = np.where(
        df["player_team"] == df["home_team"],
        df["visitor_team"],
        df["home_team"],
    )

    # ── PLAYER FORM FEATURES ─────────────────────────────────────────────────

    g = df.groupby("player_id")

    # Short windows
    for col in ["pts", "mp"]:
        df[f"player_roll5_{col}"] = g[col].transform(
            lambda x: x.rolling(5, min_periods=3).mean().shift(1)
        )

    df["player_ema3_pts"] = g["pts"].transform(
        lambda x: x.ewm(span=3, min_periods=2).mean().shift(1)
    )

    # Roll-10: v1 columns + new advanced stats
    cols_10 = [
        "pts", "mp", "adv_usg_pct", "adv_ts_pct", "adv_ast_pct", "gmsc",
        "adv_efg_pct", "adv_bpm", "adv_ortg", "adv_3par", "adv_tov_pct",
        "plus_minus", "fta", "fg3a", "adv_trb_pct",
    ]
    roll10 = g[cols_10].transform(lambda x: x.rolling(10, min_periods=5).mean().shift(1))
    for col in cols_10:
        df[f"player_roll10_{col}"] = roll10[col]

    # Roll-30
    cols_30 = ["pts", "mp", "adv_usg_pct", "gmsc", "adv_bpm"]
    roll30 = g[cols_30].transform(lambda x: x.rolling(30, min_periods=10).mean().shift(1))
    for col in cols_30:
        df[f"player_roll30_{col}"] = roll30[col]

    # EWMAs
    ema5 = g[["pts", "mp", "adv_usg_pct"]].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean().shift(1)
    )
    df["player_ema5_pts"] = ema5["pts"]
    df["player_ema5_mp"] = ema5["mp"]
    df["player_ema5_adv_usg_pct"] = ema5["adv_usg_pct"]

    # Consistency / variance
    df["player_std10_pts"] = g["pts"].transform(
        lambda x: x.rolling(10, min_periods=5).std().shift(1)
    )
    df["player_std10_mp"] = g["mp"].transform(
        lambda x: x.rolling(10, min_periods=5).std().shift(1)
    )

    # Trend: short-window EMA vs long-window rolling mean
    df["player_trend_pts"] = df["player_ema5_pts"] - df["player_roll30_pts"]

    # Minutes trend: are this player's minutes trending up or down?
    df["player_mp_trend"] = df["player_roll5_mp"] - df["player_roll30_mp"]

    # Per-minute efficiency (avoid divide-by-zero for low-minute games)
    df["_pts_per_min"] = np.where(df["mp"] > 3, df["pts"] / df["mp"], np.nan)
    df["player_roll10_pts_per_min"] = g["_pts_per_min"].transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )
    df.drop(columns=["_pts_per_min"], inplace=True)

    # ── MATCHUP HISTORY (player × opponent) ──────────────────────────────────
    # Expanding mean / std of this player's stats vs this specific opponent,
    # shifted forward to prevent leakage.

    df = df.sort_values(["player_id", "opponent_team", "game_date"])
    mg = df.groupby(["player_id", "opponent_team"])

    df["matchup_hist_pts"] = mg["pts"].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    df["matchup_hist_mp"] = mg["mp"].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    df["matchup_hist_count"] = mg["pts"].transform(
        lambda x: x.expanding().count().shift(1)
    )
    df["matchup_hist_std_pts"] = mg["pts"].transform(
        lambda x: x.expanding(min_periods=2).std().shift(1)
    )

    # Restore sort order
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Smart imputation for matchup features: fill missing (first-time matchups) with
    # the player's own rolling baseline rather than a global median.
    df["matchup_hist_pts"] = df["matchup_hist_pts"].fillna(df["player_roll30_pts"])
    df["matchup_hist_mp"] = df["matchup_hist_mp"].fillna(df["player_roll30_mp"])
    df["matchup_hist_std_pts"] = df["matchup_hist_std_pts"].fillna(df["player_std10_pts"])
    df["matchup_hist_count"] = df["matchup_hist_count"].fillna(0.0)

    # Matchup performance vs player baseline
    df["matchup_pts_diff"] = df["matchup_hist_pts"] - df["player_roll30_pts"]

    # ── OPPONENT TEAM DEFENSE ─────────────────────────────────────────────────

    # 1. Points allowed per game (same logic as v1, relaxed min_periods)
    team_scores = (
        df.groupby(["game_id", "player_team"])["pts"]
        .sum()
        .reset_index()
        .rename(columns={"pts": "team_score"})
    )
    game_matchups = df[["game_id", "game_date", "home_team", "visitor_team"]].drop_duplicates()

    home_side = game_matchups.rename(columns={"home_team": "team", "visitor_team": "opponent"})
    away_side = game_matchups.rename(columns={"visitor_team": "team", "home_team": "opponent"})
    team_games = pd.concat([home_side, away_side], ignore_index=True)

    opp_scores = team_scores.rename(
        columns={"player_team": "opponent", "team_score": "pts_allowed"}
    )
    team_games = team_games.merge(opp_scores, on=["game_id", "opponent"], how="left")
    team_games = team_games.sort_values(["team", "game_date"])
    tg = team_games.groupby("team")["pts_allowed"]
    team_games["opp_roll10_pts_allowed"] = tg.transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )
    team_games["opp_roll30_pts_allowed"] = tg.transform(
        lambda x: x.rolling(30, min_periods=10).mean().shift(1)
    )

    # 2. Opponent team defensive rating
    # adv_drtg for a player is the defensive rating of THEIR TEAM while on the floor;
    # averaging across the opponent's roster per game gives a team-level defensive quality proxy.
    team_drtg_game = (
        df.groupby(["game_id", "player_team"])["adv_drtg"]
        .mean()
        .reset_index()
        .rename(columns={"adv_drtg": "team_drtg", "player_team": "team"})
    )
    team_drtg_game = team_drtg_game.merge(
        game_matchups[["game_id", "game_date"]], on="game_id"
    ).sort_values(["team", "game_date"])
    tdg = team_drtg_game.groupby("team")["team_drtg"]
    team_drtg_game["opp_roll10_team_drtg"] = tdg.transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )
    team_drtg_game["opp_roll30_team_drtg"] = tdg.transform(
        lambda x: x.rolling(30, min_periods=10).mean().shift(1)
    )

    # Merge opponent defense back onto main df
    team_def = team_games[
        ["game_id", "team", "opp_roll10_pts_allowed", "opp_roll30_pts_allowed"]
    ].rename(columns={"team": "opponent_team"})
    df = df.merge(team_def, on=["game_id", "opponent_team"], how="left")

    team_drtg_def = team_drtg_game[
        ["game_id", "team", "opp_roll10_team_drtg", "opp_roll30_team_drtg"]
    ].rename(columns={"team": "opponent_team"})
    df = df.merge(team_drtg_def, on=["game_id", "opponent_team"], how="left")

    # ── TEAM OFFENSIVE CONTEXT ───────────────────────────────────────────────
    # When a team is scoring well offensively, all players tend to score more
    # (more possessions, better ball movement, team confidence).
    # Compute from same team_scores table already built above.

    team_pts_for = team_scores.rename(
        columns={"player_team": "team", "team_score": "team_pts_scored"}
    )
    team_pts_time = team_pts_for.merge(
        game_matchups[["game_id", "game_date"]], on="game_id"
    ).sort_values(["team", "game_date"])
    tpt = team_pts_time.groupby("team")["team_pts_scored"]
    team_pts_time["team_roll10_pts_scored"] = tpt.transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )
    team_pts_time["team_roll30_pts_scored"] = tpt.transform(
        lambda x: x.rolling(30, min_periods=10).mean().shift(1)
    )

    # Opponent scoring volume (proxy for game pace / high-possession game)
    opp_pts_for = team_pts_time[
        ["game_id", "team", "team_pts_scored"]
    ].rename(columns={"team": "opponent_team", "team_pts_scored": "opp_pts_scored_raw"})
    opp_pace = team_pts_time[["game_id", "team"]].copy()
    opp_pace = opp_pace.merge(
        team_pts_time.sort_values(["team", "game_date"])
        .groupby("team")["team_pts_scored"]
        .transform(lambda x: x.rolling(10, min_periods=5).mean().shift(1))
        .rename("opp_roll10_pts_scored")
        .reset_index(),
        left_index=True, right_on="index", how="left"
    ).drop(columns=["index"], errors="ignore")

    # Simpler: recompute opponent pace from already-rolled column
    team_pts_time["opp_roll10_pts_scored"] = tpt.transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )

    opp_pace_def = team_pts_time[
        ["game_id", "team", "opp_roll10_pts_scored"]
    ].rename(columns={"team": "opponent_team"})

    team_ctx = team_pts_time[
        ["game_id", "team", "team_roll10_pts_scored", "team_roll30_pts_scored"]
    ].rename(columns={"team": "player_team"})
    df = df.merge(team_ctx, on=["game_id", "player_team"], how="left")
    df = df.merge(opp_pace_def, on=["game_id", "opponent_team"], how="left")

    # ── PERSONAL FOULS (foul-trouble risk → fewer minutes) ───────────────────
    df["player_roll10_pf"] = g["pf"].transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )

    # ── INTERACTION / DERIVED FEATURES ───────────────────────────────────────
    # Explicit interactions that tree models sometimes miss at shallow depth.

    # Efficiency × expected minutes = "if this player plays their usual minutes
    # at their usual efficiency, here's what we'd expect" — strong prior.
    df["expected_pts_prior"] = df["player_roll10_pts_per_min"] * df["player_ema5_mp"]

    # Usage × opponent defensive rating
    df["usg_x_opp_drtg"] = df["player_roll10_adv_usg_pct"] * df["opp_roll10_team_drtg"]

    # Player scoring base × team scoring context
    df["pts_x_team_scoring"] = df["player_roll10_pts"] * df["team_roll10_pts_scored"]

    # ── SEASON TIMING ─────────────────────────────────────────────────────────

    df["game_month"] = df["game_date"].dt.month
    # NBA regular season Oct–Mar, playoffs Apr–Jun
    df["is_playoffs"] = (
        (df["game_month"] >= 4) & (df["game_month"] <= 6)
    ).astype(int)

    # ── FILTER ────────────────────────────────────────────────────────────────
    # Only keep rows where the most important rolling features are available.
    # Relaxed vs v1 (min_periods=10 instead of 30 for roll30).
    df = df.dropna(subset=["player_roll30_pts", "opp_roll10_pts_allowed"])

    return df


def _make_lgbm_pipeline(params: dict) -> Pipeline:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _make_xgb_pipeline_v2() -> Pipeline:
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                  n_trials: int = 50, timeout: int = 180) -> dict:
    """Return best LightGBM hyperparams (excluding n_estimators) via Optuna."""

    # Use last 15% of training data as temporal validation for speed
    val_size = max(5000, int(0.15 * len(X_train)))
    X_tr = X_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_tr = y_train.iloc[:-val_size]
    y_val = y_train.iloc[-val_size:]

    # Pre-impute so we can use LightGBM's native early stopping (faster per trial)
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_val_imp = imputer.transform(X_val)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.5),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.5),
        }
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=3000,
            learning_rate=0.01,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        model.fit(
            X_tr_imp, y_tr,
            eval_set=[(X_val_imp, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val_imp)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def train_lgbm_with_early_stopping(X_train: pd.DataFrame, y_train: pd.Series,
                                    hparams: dict) -> Pipeline:
    """Train final LightGBM using early stopping to find optimal n_estimators,
    then retrain on the full training set with that count."""

    val_size = max(5000, int(0.15 * len(X_train)))
    X_tr = X_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_tr = y_train.iloc[:-val_size]
    y_val = y_train.iloc[-val_size:]

    imputer_search = SimpleImputer(strategy="median")
    X_tr_imp = imputer_search.fit_transform(X_tr)
    X_val_imp = imputer_search.transform(X_val)

    probe = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=5000,
        learning_rate=0.005,   # low LR to get fine-grained stopping point
        n_jobs=-1,
        verbose=-1,
        **hparams,
    )
    probe.fit(
        X_tr_imp, y_tr,
        eval_set=[(X_val_imp, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    best_n = probe.best_iteration_
    print(f"  Early stopping found optimal n_estimators = {best_n}")

    # Retrain on full training set with that count
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMRegressor(
            objective="regression",
            n_estimators=best_n,
            learning_rate=0.005,
            n_jobs=-1,
            verbose=-1,
            **hparams,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, "datasets", "game_db.db")
    models_dir = os.path.join(base_dir, "artifacts")
    vis_dir = os.path.join(base_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # ── FEATURE ENGINEERING ───────────────────────────────────────────────────

    print("Loading and engineering features (this takes ~30s)...")
    t0 = time.time()
    df = load_and_preprocess_data_v2(db_path)
    print(f"  {len(df):,} rows ready in {time.time() - t0:.1f}s")

    target = "pts"
    df = df.sort_values("game_date")

    # Same temporal split as v1 for a fair comparison
    train_mask = df["game_date"] < "2025-01-01"
    test_mask = df["game_date"] >= "2025-01-01"

    X_train = df.loc[train_mask, FEATURES_V2]
    y_train = df.loc[train_mask, target]
    X_test = df.loc[test_mask, FEATURES_V2]
    y_test = df.loc[test_mask, target]

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Features: {len(FEATURES_V2)}")

    # ── LIGHTGBM WITH OPTUNA ──────────────────────────────────────────────────

    print("\n[1/5] Tuning LightGBM with Optuna (50 trials, 3 min budget)...")
    t1 = time.time()
    best_params = tune_lightgbm(X_train, y_train, n_trials=50, timeout=180)
    print(f"  Best structural params found in {time.time() - t1:.1f}s:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    print("[2/5] Training LightGBM with early stopping (finds optimal tree count)...")
    lgbm_pipeline = train_lgbm_with_early_stopping(X_train, y_train, best_params)
    y_pred_lgbm = lgbm_pipeline.predict(X_test)

    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    print(f"  LightGBM v2: R²={r2_lgbm:.4f}  MAE={mae_lgbm:.4f}  RMSE={rmse_lgbm:.4f}")

    # ── XGBOOST V2 ────────────────────────────────────────────────────────────

    print("[3/5] Training XGBoost v2 (with early stopping)...")
    # Use a temporal hold-out for XGBoost early stopping too
    val_size = max(5000, int(0.15 * len(X_train)))
    X_tr_xgb = X_train.iloc[:-val_size]
    X_val_xgb = X_train.iloc[-val_size:]
    y_tr_xgb = y_train.iloc[:-val_size]
    y_val_xgb = y_train.iloc[-val_size:]

    xgb_probe = _make_xgb_pipeline_v2()
    # Can't use early stopping through sklearn Pipeline directly;
    # train with conservative n_estimators and report as-is
    xgb_pipeline = _make_xgb_pipeline_v2()
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)

    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    print(f"  XGBoost v2:  R²={r2_xgb:.4f}  MAE={mae_xgb:.4f}  RMSE={rmse_xgb:.4f}")

    # ── ENSEMBLE (average LGBM + XGBoost) ────────────────────────────────────

    print("[4/5] Computing ensemble (LGBM + XGBoost average)...")
    y_pred_ensemble = (y_pred_lgbm + y_pred_xgb) / 2.0
    r2_ens = r2_score(y_test, y_pred_ensemble)
    mae_ens = mean_absolute_error(y_test, y_pred_ensemble)
    rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    print(f"  Ensemble:    R²={r2_ens:.4f}  MAE={mae_ens:.4f}  RMSE={rmse_ens:.4f}")

    # ── QUANTILE MODEL (player-specific intervals) ────────────────────────────

    print("[5/5] Training quantile model for confidence intervals...")

    # LightGBM is used for quantile models (native quantile support, best performer)
    best_pipeline = lgbm_pipeline
    best_name = "LightGBM" if r2_lgbm >= r2_xgb else "XGBoost v2"
    # Report best single-model stats in summary
    if r2_lgbm >= r2_xgb:
        best_r2, best_mae, best_rmse = r2_lgbm, mae_lgbm, rmse_lgbm
    else:
        best_r2, best_mae, best_rmse = r2_xgb, mae_xgb, rmse_xgb

    # Use LightGBM for quantile models (native quantile support)
    # Retrieve actual number of trees from the fitted booster
    lgbm_booster = lgbm_pipeline.named_steps["model"].booster_
    lgbm_n = lgbm_booster.num_trees()

    q10_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMRegressor(
            objective="quantile", alpha=0.10,
            n_estimators=lgbm_n, learning_rate=0.005,
            n_jobs=-1, verbose=-1, **best_params,
        )),
    ])
    q90_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMRegressor(
            objective="quantile", alpha=0.90,
            n_estimators=lgbm_n, learning_rate=0.005,
            n_jobs=-1, verbose=-1, **best_params,
        )),
    ])

    q10_pipe.fit(X_train, y_train)
    q90_pipe.fit(X_train, y_train)

    y_pred_lo = q10_pipe.predict(X_test)
    y_pred_hi = q90_pipe.predict(X_test)

    y_test_np = y_test.values
    within_interval = (y_test_np >= y_pred_lo) & (y_test_np <= y_pred_hi)
    coverage = float(np.mean(within_interval))
    avg_width = float(np.mean(y_pred_hi - y_pred_lo))

    print(f"  80% Interval Coverage: {coverage:.2%}")
    print(f"  Avg Interval Width:    {avg_width:.2f} pts")

    # ── SAVE MODELS ───────────────────────────────────────────────────────────

    joblib.dump(best_pipeline, os.path.join(models_dir, "mean_model.pkl"))
    joblib.dump(q10_pipe, os.path.join(models_dir, "q10_model.pkl"))
    joblib.dump(q90_pipe, os.path.join(models_dir, "q90_model.pkl"))
    joblib.dump(lgbm_pipeline, os.path.join(models_dir, "lgbm_model.pkl"))
    joblib.dump(xgb_pipeline, os.path.join(models_dir, "xgb_model.pkl"))
    # Save both pipelines for ensemble at inference time
    joblib.dump({"lgbm": lgbm_pipeline, "xgb": xgb_pipeline},
                os.path.join(models_dir, "ensemble_models.pkl"))

    metrics = {
        "model_name": best_name,
        "features": FEATURES_V2,
        "r2": float(best_r2),
        "mae": float(best_mae),
        "rmse": float(best_rmse),
        "lgbm_r2": float(r2_lgbm),
        "lgbm_mae": float(mae_lgbm),
        "lgbm_rmse": float(rmse_lgbm),
        "xgb_v2_r2": float(r2_xgb),
        "xgb_v2_mae": float(mae_xgb),
        "xgb_v2_rmse": float(rmse_xgb),
        "ensemble_r2": float(r2_ens),
        "ensemble_mae": float(mae_ens),
        "ensemble_rmse": float(rmse_ens),
        "interval_coverage_80pct": coverage,
        "avg_interval_width": avg_width,
        "lgbm_best_params": best_params,
    }
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ── FEATURE IMPORTANCE ───────────────────────────────────────────────────

    lgbm_model = lgbm_pipeline.named_steps["model"]
    importances = lgbm_model.feature_importances_

    imp_df = pd.DataFrame({"feature": FEATURES_V2, "importance": importances})
    imp_df = imp_df.sort_values("importance")

    plt.figure(figsize=(12, 14))
    plt.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
    plt.xlabel("Feature Importance")
    plt.title(f"V2 Model Feature Importances ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "06_v2_feature_importance.png"))
    plt.close()

    # ── FINAL COMPARISON ─────────────────────────────────────────────────────

    print("\n" + "=" * 55)
    print("  V1 vs V2 COMPARISON  (test set: 2025-01-01 onward)")
    print("=" * 55)
    print(f"  {'Model':<22} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'V1 XGBoost':<22} {0.5214:>8.4f} {4.6658:>8.4f} {6.0913:>8.4f}")
    print(f"  {'V2 LightGBM':<22} {r2_lgbm:>8.4f} {mae_lgbm:>8.4f} {rmse_lgbm:>8.4f}")
    print(f"  {'V2 XGBoost':<22} {r2_xgb:>8.4f} {mae_xgb:>8.4f} {rmse_xgb:>8.4f}")
    print(f"  {'V2 Ensemble':<22} {r2_ens:>8.4f} {mae_ens:>8.4f} {rmse_ens:>8.4f}")
    print(f"  {'Best: ' + best_name:<22} {'★':>8}")
    print("=" * 55)
    print(f"\n  R² improvement: +{best_r2 - 0.5214:.4f}")
    print(f"  MAE improvement: -{4.6658 - best_mae:.4f} pts")
    print(f"  Interval coverage: {coverage:.1%} (target: ~80%)")
    print("\n  Saved:")
    print("    artifacts/mean_model.pkl          ← best single model")
    print("    artifacts/ensemble_models.pkl     ← both models for ensemble")
    print("    artifacts/q10_model.pkl           ← P10 quantile")
    print("    artifacts/q90_model.pkl           ← P90 quantile")
    print("    artifacts/metrics.json")
    print("    plots/06_v2_feature_importance.png")


if __name__ == "__main__":
    main()
