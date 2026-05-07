# Predicting NBA Player Game Scores with Contextual Modeling

CS210 Final Project

This project focuses on predicting how many points an NBA player will score in a given game using web scraping, relational databases, contextual feature engineering, and machine learning. The core novelty is encoding real-world game context—travel distance, elevation, rest, and matchup history—as quantitative features derived from arena coordinates and schedule data.

---

## 1. Problem Definition and Relevance

NBA player scoring is one of the most widely predicted quantities in sports analytics, yet most public models reduce the problem to a weighted average of recent performance. This ignores factors that practitioners know matter: a player flying cross-country for a back-to-back game at altitude will perform differently than one playing at home on two days of rest. This project builds a model that explicitly encodes those factors.

### Course Connection

This project connects to CS210 course concepts through:
- **Web Scraping:** Extracting box scores and schedules from Basketball-Reference using BeautifulSoup and requests.
- **Data Cleaning:** Handling real-world datasets with missing values (DNPs, incomplete box scores), inconsistent player IDs, and unstandardized column names.
- **Data Management:** Designing a normalized relational schema (SQLite) to ensure referential integrity and efficient time-windowed queries.
- **Data Provenance:** Maintaining a documented, script-driven transformation pipeline from raw scraped CSVs to a trained model, with no manual edits.

### Use Cases

By predicting player scoring with calibrated confidence intervals, this project can help:
- **Daily fantasy players:** Identify undervalued or overvalued player props by comparing predictions to market lines.
- **Analysts:** Understand how travel schedules and opponent matchups shift a player's expected output.
- **Bettors:** Use the over/under probability calculator to assess whether a sportsbook line reflects realistic scoring expectations.

---

## 2. Novelty and Importance

### Gap in Current Tools

Existing NBA prediction tools and public models share a common limitation: they treat every game as if it were played under identical conditions. A player's trailing 10-game average is used as the prediction regardless of whether the next game is:
- A home game after two days of rest
- A road game in Denver (5,280 ft altitude) on the second night of a back-to-back after flying from Miami

This project addresses that gap directly. Arena GPS coordinates from `datasets/assets/arena_coords.csv` are used to compute the exact great-circle distance a team traveled between games. That distance, combined with schedule timestamps, generates features like `miles_traveled`, `days_rest`, `is_back_to_back`, and `altitude_impact` that are stored in the database and fed to the model. No existing public basketball prediction tool encodes fatigue this way at the individual game level.

The second novelty is **per-matchup history modeling**: rather than treating a player's scoring average as uniform across opponents, the model maintains an expanding mean, standard deviation, and game count for every unique (player, opponent) pair. A player who historically outperforms against a specific team gets credit for that tendency.

---

## 3. Data Description

### Data Source

All data is scraped from [Basketball-Reference](https://www.basketball-reference.com/), including:
- Season schedules (game dates, home/visitor teams, arena names, box score URLs): 2021–2026
- Individual game box scores (basic and advanced statistics per player): 2020–2026

Arena coordinates (latitude and longitude for all 30 NBA arenas) were manually compiled into a seed CSV and are used to compute travel distances at load time.

### Features

| Category | Features |
| :--- | :--- |
| **Fatigue & Context** | Home/away indicator, miles traveled from previous arena, days of rest, back-to-back flag, altitude impact, game month, playoff indicator |
| **Short-Window Form** | 5-game rolling points and minutes; 3-game exponential moving average of points |
| **Mid-Window Form** | 10-game rolling points, minutes, usage %, TS%, AST%, game score, eFG%, BPM, ORtg, 3PAr, TOV%, +/-, FTA, FG3A, TRB%, points per minute, personal fouls |
| **Long-Window Form** | 30-game rolling points, minutes, usage %, game score, BPM |
| **Exponential Averages** | 5-game EMA of points, minutes, and usage % |
| **Trend & Variance** | Standard deviation of points and minutes over 10 games; linear trend slope for both |
| **Matchup History** | Mean points, mean minutes, game count, standard deviation, and premium/discount vs. rolling baseline — all computed per (player, opponent) pair |
| **Opponent Defense** | Opponent L10 and L30 points allowed, L10 and L30 defensive rating, L10 points scored (pace proxy) |
| **Team Context** | Player's team L10 and L30 points scored |
| **Interactions** | Usage × opponent DRTG; points × team scoring; combined matchup-team prior |
| **Target** | Points scored in game (integer, regression) |

**Total: 54 features**

### Data Accessibility & Format

Raw data is stored in CSV format in `datasets/raw/` before being cleaned and migrated to a SQLite relational database at `datasets/game_db.db`.

---

## 4. Data Provenance

- **Source Tracking:** Raw data is extracted directly from Basketball-Reference and stored in `datasets/raw/` in its original, unedited format. It can be re-scraped by running `nba/etl/schedule_scraper.py` and `nba/etl/game_scraper.py`, or by triggering a refresh through the web UI.
- **Transformation Pipeline:** All cleaning and normalization (column standardization, DNP handling, type coercion) are performed through `nba/etl/preprocess.py`. No edits are made manually; the transition from raw to processed data is entirely script-driven and reproducible.
- **Incremental Updates:** `nba/etl/refresh.py` reads `datasets/refresh_state.json` to determine the last successful update date and fetches only new games, preserving the full historical record without re-scraping. The last update timestamp is stored in `refresh_state.json`.
- **Versioned Artifacts:** Original scraped datasets are stored in `datasets/raw/` and cleaned datasets in `datasets/processed/`. Trained model pipelines are serialized to `artifacts/`. Evaluation metrics are saved to `artifacts/metrics.json`.
- **Environment Locking:** All library versions and environment requirements are documented in `requirements.txt` to ensure consistent execution across systems.

---

## 5. Methodology

### Data Management (Extraction & Transformation)

- **Schedule Scraping (`nba/etl/schedule_scraper.py`):** Requests-based scraper extracts game schedules from Basketball-Reference season pages, capturing game dates, home and visiting teams, arena names, and links to individual box score pages.
- **Box Score Scraping (`nba/etl/game_scraper.py`):** Follows box score URLs from the schedule to extract per-player basic and advanced statistics for every game. Both regular season and playoff games are included.
- **Preprocessing (`nba/etl/preprocess.py`):** Standardizes column names, filters out DNP rows, handles missing advanced stats, and coerces types. No imputation is performed at this stage — missing values are preserved and handled at the feature engineering step.
- **Database Loading (`nba/etl/build_db.py`, `nba/etl/db_schema.sql`):** Cleaned data is loaded into a normalized SQLite database with five tables: `Arenas`, `Teams`, `Players`, `Games`, and `Performances`. Arena coordinates from `datasets/assets/arena_coords.csv` are joined at load time to compute `miles_traveled`, `days_rest`, `is_back_to_back`, and `altitude_impact` for every performance record. These context features are stored directly in `Performances` so they are available at both training and inference without recomputation.

### Machine Learning (`nba/modeling/train.py`)

- **Model Selection:**
  - **LightGBM (primary):** Gradient boosted decision trees tuned with Optuna (50 trials, 3-minute wall clock budget, time-series cross-validation). Handles non-linear interactions between context and form features efficiently.
  - **XGBoost (secondary):** Trained with the same feature set for ensemble averaging.
  - **Ensemble:** Simple average of LightGBM and XGBoost predictions, which marginally reduces RMSE.
  - **Quantile Models:** Two additional LightGBM models trained with `alpha=0.10` and `alpha=0.90` quantile regression objectives produce calibrated 80% confidence intervals for each prediction.
- **Temporal Split:** All games through the 2023–24 season are used for training; the 2024–25 season is held out as the test set. Random splitting is deliberately avoided to prevent look-ahead bias.
- **Leakage Prevention:** All rolling and expanding features are shifted by one game per player (grouped chronologically) before training, ensuring no target-game information is included in any feature.
- **Matchup Imputation:** When a player has no prior history against a given opponent, matchup history features are imputed from the player's own rolling baseline rather than a global mean, preserving player-level signal.
- **Evaluation:** Models are evaluated using **R² Score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**. Quantile model quality is assessed by empirical interval coverage (target: 80%) and average interval width.

---

## 6. Results

### Model Performance

Evaluated on the 2024–25 season (held-out test set):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| LightGBM | 0.531 | 4.59 pts | 6.00 pts |
| XGBoost | 0.526 | 4.60 pts | 6.03 pts |
| Ensemble | 0.530 | 4.59 pts | 6.00 pts |

**Confidence interval performance** (80% nominal target):
- Empirical coverage: **79.0%** (well-calibrated)
- Average interval width: **14.4 points** (Q10 to Q90)

### Key Insights

Feature importance (LightGBM gain) is dominated by the rolling scoring windows (`roll10_pts`, `roll5_pts`, `ema5_pts`), usage rate, and minutes played. Contextual features — particularly `miles_traveled`, `days_rest`, and `is_back_to_back` — provide meaningful secondary signal that a naive averages model omits entirely. Matchup history features contribute most strongly for players with dense opponent histories (veterans playing the same division rivals repeatedly).

The ~53% R² reflects the fundamental stochasticity of single-game NBA scoring: a 30-PPG player can score anywhere from 8 to 52 points depending on foul trouble, game script, and shot variance that no observational model can fully anticipate. An MAE of 4.6 points is competitive with published benchmarks for this task.

### Visualizations (`nba/modeling/charts.py`)

- **Feature Importance Chart:** LightGBM gain across all 54 features.
- **Actual vs. Predicted Scatter:** With diagonal reference line.
- **Residual Distribution:** Error histogram and Q-Q plot.
- **Interval Coverage Plot:** Empirical coverage vs. nominal quantile levels.

### Predictions (`nba/modeling/inference.py`)

At inference time, the predictor queries the database for a given player and opponent, assembles the 54-feature vector, and returns a point estimate with a calibrated 80% confidence interval. Predictions are exposed through the web UI and the `/api/predict` endpoint.

---

## 7. Implementation Details

### Directory Structure

```
NBA-Contextual-Modeling/
├── datasets/
│   ├── raw/                         # Original scraped CSVs (one per season + schedule)
│   ├── processed/                   # Cleaned CSV ready for DB load
│   ├── assets/                      # Seed data and image URL lookups
│   │   ├── arena_coords.csv         # Arena lat/lon for travel distance calculation
│   │   ├── player_images.csv        # NBA.com headshot URLs
│   │   └── team_images.csv          # Team logo URLs
│   ├── game_db.db                   # SQLite database (all seasons)
│   └── refresh_state.json           # Tracks last successful data update
├── artifacts/
│   ├── mean_model.pkl               # Primary LightGBM predictor
│   ├── q10_model.pkl                # 10th-percentile quantile model
│   ├── q90_model.pkl                # 90th-percentile quantile model
│   ├── xgb_model.pkl                # XGBoost predictor
│   ├── ensemble_models.pkl          # Both models bundled for ensemble inference
│   └── metrics.json                 # Evaluation metrics & best hyperparameters
├── nba/
│   ├── server/                      # FastAPI backend
│   │   ├── api.py
│   │   └── static/                  # index.html, script.js, style.css, assets
│   ├── etl/                         # ETL pipeline: scrape, preprocess, load
│   │   ├── schedule_scraper.py
│   │   ├── game_scraper.py
│   │   ├── preprocess.py
│   │   ├── build_db.py
│   │   ├── refresh.py
│   │   └── db_schema.sql
│   └── modeling/                    # ML: training, evaluation, inference
│       ├── train.py
│       ├── evaluate.py
│       ├── charts.py
│       └── inference.py
├── plots/                           # Generated visualization outputs
└── requirements.txt
```

---

## 8. Reproducibility and Execution Guide

### Prerequisites

- Python 3.10+

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd NBA-Contextual-Modeling
   ```

2. **Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Executing the Technical Implementation

1. **Scrape schedule:**
   ```bash
   python nba/etl/schedule_scraper.py
   ```

2. **Scrape box scores:**
   ```bash
   python nba/etl/game_scraper.py
   ```

3. **Preprocess data:**
   ```bash
   python nba/etl/preprocess.py
   ```

4. **Build database:**
   ```bash
   python nba/etl/build_db.py
   ```

5. **Train models:**
   ```bash
   python nba/modeling/train.py
   ```

6. **Generate visualizations:**
   ```bash
   python nba/modeling/charts.py
   ```

7. **Run a prediction from the command line:**
   ```bash
   python nba/modeling/inference.py --player "LeBron James" --opp LAC --line 24.5
   ```

8. **Refresh data incrementally (after initial setup):**
   ```bash
   python nba/etl/refresh.py
   ```

### Run Web UI

```bash
cd nba/server
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser.

---

## 9. Demonstration

> **Video walkthrough (8–10 min) coming soon.** This will cover the full pipeline — data collection through live predictions — with a demo of the web UI.
