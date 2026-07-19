# Football Predictor

A machine learning web app that predicts football match outcomes using live odds, Elo ratings, team form, head-to-head history, and league standings across 6 European leagues.

## Features

- **Live odds** fetched from The Odds API (Premier League, Championship, Serie A, La Liga, Bundesliga, Ligue 1)
- **ML predictions** — XGBoost model with 35 features, tuned via RandomizedSearchCV (50 iter × 5-fold CV)
- **Value bet detection** — highlights outcomes where model probability exceeds market-implied probability by ≥5pp
- **Model vs market comparison** with expected value (EV) per outcome
- **Confidence rating** (HIGH / MED / LOW) on each prediction
- **Team form** tooltips showing last 5 results with scores and opponents
- **League standings** panel for all 6 leagues
- **Date navigation** and league filter pills
- **Predict All** button to run all visible fixtures at once
- User authentication (register / login)

## Tech Stack

- **Backend** — Python, Flask, Flask-Login, SQLAlchemy (SQLite)
- **ML** — XGBoost, scikit-learn, pandas, numpy, joblib
- **Data sources** — [The Odds API](https://the-odds-api.com), [football-data.org](https://football-data.org)
- **Frontend** — Vanilla JS, CSS custom properties, Inter font

## Setup

### 1. Install dependencies

```bash
pip install flask flask-sqlalchemy flask-login werkzeug xgboost scikit-learn pandas numpy requests joblib
```

### 2. Add API keys

Open `app.py` and replace the placeholder values:

```python
ODDS_API_KEY          = 'your_odds_api_key_here'
FOOTBALL_DATA_API_KEY = 'your_football_data_key_here'
```

- Free Odds API key: https://the-odds-api.com
- Free football-data.org key: https://www.football-data.org

### 3. Build the dataset

```bash
python prepare_data.py
```

Processes the 137 historical CSV files in `my_football_data/` and outputs:
- `processed_football_data.csv` — feature matrix for training
- `team_elo.json`, `team_form.json`, `h2h_cache.json` — lookup caches

### 4. Train the model

```bash
python train_model.py
```

Runs RandomizedSearchCV over XGBoost and GradientBoosting, picks the winner, and saves:
- `football_model.joblib` — trained model + feature list
- `accuracy.txt` — 5-fold CV accuracy

Training takes ~5–10 minutes depending on hardware.

### 5. Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` — register an account and go to the dashboard.

## Project Structure

```
football-predictor/
├── app.py                  # Flask app — routes, caching, prediction logic
├── prepare_data.py         # Feature engineering pipeline
├── train_model.py          # Model training + evaluation
├── templates/
│   └── index.html          # Dashboard UI
├── static/                 # Static assets
├── my_football_data/       # 137 historical CSVs (E0, E1, I1, SP1, D1, F1)
└── .gitignore
```

## Model

The model uses 35 features per fixture:

| Category | Features |
|---|---|
| Odds | Consensus H/D/A odds, implied probabilities |
| Elo | Home Elo, away Elo, Elo difference |
| Form (all venues) | Points, goals scored/conceded per game (last 5) |
| Form (venue-split) | Same metrics split by home/away |
| Season standings | PPG, goal difference |
| Composite | Form diff, PPG diff, attack vs defence ratios |
| Momentum | Recent 3-game weighted form |
| Rest | Days since last match (home and away) |
| H2H | Home win rate, draw rate (last 10 meetings) |

Cross-validation accuracy: **~51%** (3-class: home win / draw / away win).
