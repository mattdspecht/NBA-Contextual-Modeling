import sqlite3
import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_preprocess_data(db_path):
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
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    
    # --- Player Form ---
    df = df.sort_values(['player_id', 'game_date'])
    
    cols_10 = ['pts', 'mp', 'adv_usg_pct', 'adv_ts_pct', 'adv_ast_pct', 'gmsc']
    cols_30 = ['pts', 'mp', 'adv_usg_pct', 'gmsc']
    
    grouped = df.groupby('player_id')
    
    rolling_10 = grouped[cols_10].transform(lambda x: x.rolling(10, min_periods=10).mean().shift(1))
    df[['player_roll10_' + c for c in cols_10]] = rolling_10
    
    rolling_30 = grouped[cols_30].transform(lambda x: x.rolling(30, min_periods=30).mean().shift(1))
    df[['player_roll30_' + c for c in cols_30]] = rolling_30
    
    # Exponential moving average for recent form (span=5)
    ema_5 = grouped[['pts', 'mp', 'adv_usg_pct']].transform(lambda x: x.ewm(span=5, min_periods=5).mean().shift(1))
    df[['player_ema5_pts', 'player_ema5_mp', 'player_ema5_adv_usg_pct']] = ema_5
    
    # --- Opponent Strength ---
    team_scores = df.groupby(['game_id', 'player_team'])['pts'].sum().reset_index()
    team_scores.rename(columns={'pts': 'team_score'}, inplace=True)
    
    game_matchups = df[['game_id', 'game_date', 'home_team', 'visitor_team']].drop_duplicates()
    
    home_stats = game_matchups.rename(columns={'home_team': 'team', 'visitor_team': 'opponent'})
    visitor_stats = game_matchups.rename(columns={'visitor_team': 'team', 'home_team': 'opponent'})
    team_games = pd.concat([home_stats, visitor_stats])
    
    opp_scores = team_scores.rename(columns={'player_team': 'opponent', 'team_score': 'pts_allowed'})
    team_games = team_games.merge(opp_scores, on=['game_id', 'opponent'], how='left')
    
    team_games = team_games.sort_values(['team', 'game_date'])
    team_grouped = team_games.groupby('team')['pts_allowed']
    
    team_games['opp_roll10_pts_allowed'] = team_grouped.transform(lambda x: x.rolling(10, min_periods=10).mean().shift(1))
    team_games['opp_roll30_pts_allowed'] = team_grouped.transform(lambda x: x.rolling(30, min_periods=30).mean().shift(1))
    
    df['opponent_team'] = np.where(df['player_team'] == df['home_team'], df['visitor_team'], df['home_team'])
    
    team_def = team_games[['game_id', 'team', 'opp_roll10_pts_allowed', 'opp_roll30_pts_allowed']].rename(columns={'team': 'opponent_team'})
    df = df.merge(team_def, on=['game_id', 'opponent_team'], how='left')
    
    df = df.dropna(subset=['player_roll30_pts', 'opp_roll30_pts_allowed', 'player_ema5_pts'])
    
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, 'data', 'nba_contextual.db')
    models_dir = os.path.join(base_dir, 'models')
    vis_dir = os.path.join(base_dir, 'visualizations')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    df = load_and_preprocess_data(db_path)
    
    features = [
        'is_home', 'miles_traveled', 'days_rest', 'is_back_to_back', 'altitude_impact',
        'player_roll10_pts', 'player_roll10_mp', 'player_roll10_adv_usg_pct', 
        'player_roll10_adv_ts_pct', 'player_roll10_adv_ast_pct', 'player_roll10_gmsc',
        'player_roll30_pts', 'player_roll30_mp', 'player_roll30_adv_usg_pct', 'player_roll30_gmsc',
        'player_ema5_pts', 'player_ema5_mp', 'player_ema5_adv_usg_pct',
        'opp_roll10_pts_allowed', 'opp_roll30_pts_allowed'
    ]
    target = 'pts'
    
    df = df.sort_values('game_date')
    train_mask = df['game_date'] < '2025-01-01'
    test_mask = df['game_date'] >= '2025-01-01'
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, target]
    X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, target]
    
    # Train expected value model to maximize R2
    print("Training Expected Value Model (reg:squarederror)...")
    pipeline_mean = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    pipeline_mean.fit(X_train, y_train)
    y_pred_mean = pipeline_mean.predict(X_test)
    r2_mean = r2_score(y_test, y_pred_mean)
    mae_mean = mean_absolute_error(y_test, y_pred_mean)
    print(f"Mean Model R2: {r2_mean:.4f}, MAE: {mae_mean:.4f}")
    
    from sklearn.metrics import mean_squared_error
    rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    
    print("Training Quantile Model (reg:quantileerror)...")
    pipeline_quant = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=[0.1, 0.5, 0.9],
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    pipeline_quant.fit(X_train, y_train)
    y_pred_quant = pipeline_quant.predict(X_test)
    
    y_pred_lower = y_pred_quant[:, 0]
    y_pred_median = y_pred_quant[:, 1]
    y_pred_upper = y_pred_quant[:, 2]
    
    # Combine predictions: Use mean for point prediction, quantiles for intervals
    mae_median = mean_absolute_error(y_test, y_pred_median)
    r2_median = r2_score(y_test, y_pred_median)
    
    y_test_np = y_test.values
    within_interval = (y_test_np >= y_pred_lower) & (y_test_np <= y_pred_upper)
    coverage = np.mean(within_interval)
    avg_interval_width = np.mean(y_pred_upper - y_pred_lower)
    
    print("-" * 40)
    print("Final Model Evaluation on Test Set")
    print(f"Point Prediction R2: {r2_mean:.4f}")
    print(f"Point Prediction MAE: {mae_mean:.4f}")
    print(f"Point Prediction RMSE: {rmse_mean:.4f}")
    print(f"80% Interval Coverage: {coverage:.2%}")
    print(f"Avg Interval Width:    {avg_interval_width:.2f} pts")
    print("-" * 40)
    
    xgb_model = pipeline_quant.named_steps['model']
    importances = xgb_model.feature_importances_
    
    indices = np.argsort(importances)
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features, sorted_importances, color='salmon')
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    importance_path = os.path.join(vis_dir, '05_feature_importance.png')
    plt.savefig(importance_path)
    plt.close()
    
    # Save the quantile model
    model_path = os.path.join(models_dir, 'xgb_fatigue_pipeline.pkl')
    joblib.dump(pipeline_quant, model_path)
    
    # Save the mean model too, so predictor can use both
    mean_model_path = os.path.join(models_dir, 'xgb_mean_pipeline.pkl')
    joblib.dump(pipeline_mean, mean_model_path)
    
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    metrics_data = {
        'r2': float(r2_mean),
        'mae': float(mae_mean),
        'rmse': float(rmse_mean),
        'r2_median': float(r2_median),
        'mae_median': float(mae_median),
        'interval_coverage': float(coverage),
        'avg_interval_width': float(avg_interval_width)
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

if __name__ == "__main__":
    main()
