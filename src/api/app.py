import os
import json
import joblib
import sqlite3
import threading
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional

# Add root directory to python path if not already there
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.predictor.predictor import get_player_stats, get_opponent_stats, calculate_ewma
from src.data.incremental_updater import get_state, save_state, run_incremental_refresh, COOLDOWN_MINUTES

db_path = os.path.join(base_dir, 'data', 'nba_contextual.db')
models_dir = os.path.join(base_dir, 'models')
_api_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(_api_dir, "static")

# Optional override: absolute path to a local .mp4 for the UI background.
# If unset, uses static/bg-loop.mp4 when present (served as /static/bg-loop.mp4).
BG_VIDEO_ENV = "NBA_PROP_BG_VIDEO"

def _default_background_video_path() -> str:
    return os.path.join(static_dir, "bg-loop.mp4")

# Global state for models
ml_models = {}

# Global refresh state — mutated in-place by the worker thread
_refresh_status: dict = {"status": "idle", "message": "", "progress": 0.0}
_refresh_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML model
    ml_models['pipeline_mean'] = joblib.load(os.path.join(models_dir, 'xgb_mean_pipeline.pkl'))
    with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
        metrics = json.load(f)
    ml_models['rmse'] = metrics['rmse']
    yield
    # Clean up
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    player: str
    opp_team: str
    is_home: int = 1
    days_rest: int = 2

TEAM_ACRONYMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

def team_name_to_acronym(team_name: str) -> str:
    if not team_name or not str(team_name).strip():
        return "UNK"
    return TEAM_ACRONYMS.get(str(team_name).strip(), "UNK")


def resolved_background_video_path() -> Optional[str]:
    raw = os.environ.get(BG_VIDEO_ENV, "").strip()
    if raw:
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.isfile(path) and path.lower().endswith(".mp4"):
            return path
    default = _default_background_video_path()
    if os.path.isfile(default):
        return default
    return None

@app.get("/api/players")
def get_players():
    # Return unique players and their most recent team/acronym
    conn = sqlite3.connect(db_path)
    query = """
    SELECT pl.player_name, p.player_team as team_name
    FROM (
        SELECT player_id, player_team, ROW_NUMBER() OVER(PARTITION BY player_id ORDER BY performance_id DESC) as rn
        FROM Performances
    ) p
    JOIN Players pl ON pl.player_id = p.player_id
    WHERE p.rn = 1
    ORDER BY pl.player_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df["team"] = df["team_name"].apply(team_name_to_acronym)
    
    return df.to_dict(orient="records")

@app.get("/api/config")
def ui_config():
    p = resolved_background_video_path()
    if not p:
        return {"backgroundVideoUrl": None}
    # Default bundle: direct static URL (reliable for <video> + Range requests in all browsers)
    if os.path.abspath(p) == os.path.abspath(_default_background_video_path()):
        return {"backgroundVideoUrl": "/static/bg-loop.mp4"}
    return {"backgroundVideoUrl": "/api/background-video"}


@app.get("/api/background-video")
def background_video():
    p = resolved_background_video_path()
    if not p:
        raise HTTPException(status_code=404, detail="No background video configured")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/teams")
def get_teams():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT DISTINCT team_acronym FROM Teams ORDER BY team_acronym", conn)
    conn.close()
    return {"teams": df['team_acronym'].tolist()}

@app.post("/api/predict")
def predict(req: PredictRequest):
    player_df = get_player_stats(db_path, req.player)
    opp_df = get_opponent_stats(db_path, req.opp_team)
    
    if len(player_df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough data for player {req.player}")
    if len(opp_df) < 10:
        raise HTTPException(status_code=400, detail=f"Not enough data for opponent {req.opp_team}")
        
    player_roll10_pts = player_df['pts'].head(10).mean()
    player_roll10_mp = player_df['mp'].head(10).mean()
    player_roll10_adv_usg_pct = player_df['adv_usg_pct'].head(10).mean()
    player_roll10_adv_ts_pct = player_df['adv_ts_pct'].head(10).mean()
    player_roll10_adv_ast_pct = player_df['adv_ast_pct'].head(10).mean()
    player_roll10_gmsc = player_df['gmsc'].head(10).mean()
    
    player_roll30_pts = player_df['pts'].mean()
    player_roll30_mp = player_df['mp'].mean()
    player_roll30_adv_usg_pct = player_df['adv_usg_pct'].mean()
    player_roll30_gmsc = player_df['gmsc'].mean()
    
    player_ema5_pts = calculate_ewma(player_df['pts'].head(5))
    player_ema5_mp = calculate_ewma(player_df['mp'].head(5))
    player_ema5_adv_usg_pct = calculate_ewma(player_df['adv_usg_pct'].head(5))
    
    opp_roll10_pts_allowed = opp_df['pts_allowed'].head(10).mean()
    opp_roll30_pts_allowed = opp_df['pts_allowed'].mean()
    
    features = [
        req.is_home, 0.0, req.days_rest, (1 if req.days_rest == 0 else 0), 0,
        player_roll10_pts, player_roll10_mp, player_roll10_adv_usg_pct, 
        player_roll10_adv_ts_pct, player_roll10_adv_ast_pct, player_roll10_gmsc,
        player_roll30_pts, player_roll30_mp, player_roll30_adv_usg_pct, player_roll30_gmsc,
        player_ema5_pts, player_ema5_mp, player_ema5_adv_usg_pct,
        opp_roll10_pts_allowed, opp_roll30_pts_allowed
    ]
    
    feature_names = [
        'is_home', 'miles_traveled', 'days_rest', 'is_back_to_back', 'altitude_impact',
        'player_roll10_pts', 'player_roll10_mp', 'player_roll10_adv_usg_pct', 
        'player_roll10_adv_ts_pct', 'player_roll10_adv_ast_pct', 'player_roll10_gmsc',
        'player_roll30_pts', 'player_roll30_mp', 'player_roll30_adv_usg_pct', 'player_roll30_gmsc',
        'player_ema5_pts', 'player_ema5_mp', 'player_ema5_adv_usg_pct',
        'opp_roll10_pts_allowed', 'opp_roll30_pts_allowed'
    ]
    
    X_pred = pd.DataFrame([features], columns=feature_names)
    
    pipeline_mean = ml_models['pipeline_mean']
    expected_pts = float(pipeline_mean.predict(X_pred)[0])
    rmse = float(ml_models['rmse'])

    last_game_date = str(player_df["game_date"].iloc[0]) if len(player_df) else None
    recent_pts = [float(x) for x in player_df["pts"].head(10).iloc[::-1].tolist()]

    player_ctx = {
        "roll10_pts": float(player_roll10_pts),
        "roll30_pts": float(player_roll30_pts),
        "ema5_pts": float(player_ema5_pts),
        "last_game_date": last_game_date,
        "recent_pts": recent_pts,
        "roll10_usg_pct": float(player_roll10_adv_usg_pct),
        "roll10_ts_pct": float(player_roll10_adv_ts_pct),
    }
    opponent_ctx = {
        "acronym": req.opp_team,
        "roll10_pts_allowed": float(opp_roll10_pts_allowed),
        "roll30_pts_allowed": float(opp_roll30_pts_allowed),
    }
    matchup_ctx = {
        "is_home": int(req.is_home),
        "days_rest": int(req.days_rest),
    }

    return {
        "expected_pts": expected_pts,
        "rmse": rmse,
        # Flat keys (always present) so clients never depend only on nested objects
        "roll10_pts": player_ctx["roll10_pts"],
        "roll30_pts": player_ctx["roll30_pts"],
        "ema5_pts": player_ctx["ema5_pts"],
        "last_game_date": last_game_date,
        "recent_pts": recent_pts,
        "roll10_usg_pct": player_ctx["roll10_usg_pct"],
        "roll10_ts_pct": player_ctx["roll10_ts_pct"],
        "opp_roll10_pts_allowed": opponent_ctx["roll10_pts_allowed"],
        "opp_roll30_pts_allowed": opponent_ctx["roll30_pts_allowed"],
        "opp_team": opponent_ctx["acronym"],
        "is_home": matchup_ctx["is_home"],
        "days_rest": matchup_ctx["days_rest"],
        "player": player_ctx,
        "opponent": opponent_ctx,
        "matchup": matchup_ctx,
    }

@app.get("/api/refresh-state")
def get_refresh_state():
    state = get_state()
    return {
        "last_updated": state.get("last_updated", "2026-05-04"),
        "status": _refresh_status["status"],
        "message": _refresh_status["message"],
        "progress": _refresh_status["progress"],
    }


@app.post("/api/refresh")
def trigger_refresh():
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=423, detail="A refresh is already in progress.")

    try:
        state = get_state()
        last_attempt = state.get("last_attempt")
        if last_attempt:
            try:
                last_dt = datetime.fromisoformat(last_attempt)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                remaining = int(COOLDOWN_MINUTES * 60 - elapsed)
                if remaining > 0:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Data was refreshed recently. Wait {remaining // 60 + 1} more minute(s).",
                        headers={"Retry-After": str(remaining)},
                    )
            except HTTPException:
                raise
            except (ValueError, TypeError):
                pass  # Malformed timestamp; proceed

        state["last_attempt"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        _refresh_status["status"] = "running"
        _refresh_status["message"] = "Starting refresh..."
        _refresh_status["progress"] = 0.0

        def _refresh_worker():
            try:
                run_incremental_refresh(_refresh_status)
            except Exception as e:
                _refresh_status["status"] = "error"
                _refresh_status["message"] = str(e)
            finally:
                _refresh_lock.release()

        threading.Thread(target=_refresh_worker, daemon=True).start()
        return JSONResponse(status_code=202, content={"detail": "Refresh started."})

    except HTTPException:
        _refresh_lock.release()
        raise
    except Exception:
        _refresh_lock.release()
        raise


# Serve Static Files (includes bg-loop.mp4)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))
