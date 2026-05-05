import sqlite3
import pandas as pd
import numpy as np
import os
import json
import joblib
import argparse
from scipy.stats import norm

def get_player_stats(db_path, player_name):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT g.game_date, p.pts, p.mp, p.adv_usg_pct, p.adv_ts_pct, p.adv_ast_pct, p.gmsc
    FROM Performances p
    JOIN Games g ON p.game_id = g.game_id
    JOIN Players pl ON p.player_id = pl.player_id
    WHERE pl.player_name = ?
    ORDER BY g.game_date DESC
    LIMIT 30
    """
    df = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()
    return df

def get_opponent_stats(db_path, opp_team):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT g.game_id, g.game_date, 
           SUM(CASE WHEN p.player_team != ? THEN p.pts ELSE 0 END) as pts_allowed
    FROM Games g
    JOIN Performances p ON g.game_id = p.game_id
    WHERE g.home_team = ? OR g.visitor_team = ?
    GROUP BY g.game_id
    ORDER BY g.game_date DESC
    LIMIT 30
    """
    df = pd.read_sql_query(query, conn, params=(opp_team, opp_team, opp_team))
    conn.close()
    return df

def calculate_ewma(series, span=5):
    # Pandas ewm calculates from old to new. Our data is sorted new to old (DESC).
    # We must reverse it, calculate ewma, and take the last value.
    s = series.iloc[::-1].reset_index(drop=True)
    ewma = s.ewm(span=span, min_periods=1).mean()
    return ewma.iloc[-1]

def main():
    parser = argparse.ArgumentParser(description="NBA Player Prop Over/Under Predictor")
    parser.add_argument("--player", type=str, required=True, help="Player Name (e.g., 'LeBron James')")
    parser.add_argument("--team", type=str, required=True, help="Player's Team Acronym (e.g., 'LAL')")
    parser.add_argument("--opp", type=str, required=True, help="Opponent Team Acronym (e.g., 'BOS')")
    parser.add_argument("--line", type=float, required=True, help="Over/Under Line (e.g., 24.5)")
    parser.add_argument("--is_home", type=int, default=1, choices=[0, 1], help="1 if Home, 0 if Away")
    parser.add_argument("--days_rest", type=int, default=2, help="Days of rest")
    
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, 'data', 'nba_contextual.db')
    models_dir = os.path.join(base_dir, 'models')
    
    # 1. Fetch Data
    player_df = get_player_stats(db_path, args.player)
    opp_df = get_opponent_stats(db_path, args.opp)
    
    if len(player_df) < 10:
        print(f"Error: Not enough data for {args.player}. Found {len(player_df)} games.")
        return
        
    if len(opp_df) < 10:
        print(f"Error: Not enough data for opponent {args.opp}.")
        return

    # 2. Engineer Features for the single prediction
    # player_df is sorted DESC (newest first). 
    # The first 10 rows are the most recent 10 games.
    
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
    
    # Context features (defaults or passed args)
    is_home = args.is_home
    miles_traveled = 0.0 # Defaulting for simplicity in a quick prop lookup
    days_rest = args.days_rest
    is_back_to_back = 1 if days_rest == 0 else 0
    altitude_impact = 0 # Defaulting
    
    # Create the feature vector
    features = [
        is_home, miles_traveled, days_rest, is_back_to_back, altitude_impact,
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
    
    # 3. Load Model & Metrics
    pipeline_mean = joblib.load(os.path.join(models_dir, 'xgb_mean_pipeline.pkl'))
    with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
        metrics = json.load(f)
        
    rmse = metrics['rmse']
    
    # 4. Predict
    expected_pts = pipeline_mean.predict(X_pred)[0]
    
    # 5. Calculate O/U Probability
    # P(Over) = 1 - CDF(line)
    prob_under = norm.cdf(args.line, loc=expected_pts, scale=rmse)
    prob_over = 1.0 - prob_under
    
    print("=" * 40)
    print(f"PROP PREDICTOR: {args.player.upper()}")
    print(f"MATCHUP: {args.team.upper()} vs {args.opp.upper()}")
    print("=" * 40)
    print(f"Expected Points (Model):  {expected_pts:.2f} pts")
    print(f"Historical RMSE (Sigma):  {rmse:.2f} pts")
    print("-" * 40)
    print(f"THE LINE: {args.line} points")
    print(f"Probability of OVER:  {prob_over:.1%}")
    print(f"Probability of UNDER: {prob_under:.1%}")
    print("=" * 40)

if __name__ == "__main__":
    main()
