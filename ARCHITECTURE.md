# Karachi AQI Predictor — Architecture & Developer Guide

> **For AI assistants (GitHub Copilot, etc.):** This file is the single source of truth for the project structure, tech stack, data flow, and coding conventions. Read this before generating any code.

---

## Project Summary

An end-to-end, fully serverless ML system that predicts Karachi's Air Quality Index (AQI) for the next 3 days. The system runs autonomously — data collection, model retraining, and predictions all happen in the cloud with no manual intervention. The user's machine can be completely off.

**Target city:** Karachi, Pakistan  
**Prediction horizon:** 72 hours (3 days ahead)  
**Update frequency:** Features every hour, model retrained every day

---

## Tech Stack

| Layer           | Technology                              | Purpose                               |
| --------------- | --------------------------------------- | ------------------------------------- |
| Data source     | AQICN API / OpenWeatherMap API          | Raw AQI + weather data for Karachi    |
| Feature store   | Hopsworks (free tier)                   | Store, version, and serve ML features |
| Model registry  | Hopsworks Model Registry                | Store trained model versions          |
| ML models       | Scikit-learn, XGBoost, TensorFlow/Keras | Random Forest, Ridge Regression, LSTM |
| Explainability  | SHAP                                    | Feature importance for predictions    |
| Orchestration   | GitHub Actions                          | Serverless cron scheduling            |
| Dashboard       | Streamlit (deployed to Streamlit Cloud) | User-facing web app                   |
| Language        | Python 3.10+                            | All pipeline scripts                  |
| Version control | Git + GitHub                            | Source of truth, triggers CI/CD       |

---

## Repository Structure

```
karachi-aqi-predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml      # Runs every hour — fetches + stores features
│       └── training_pipeline.yml     # Runs every day — retrains + saves model
│
├── pipelines/
│   ├── feature_pipeline.py           # Step 1: fetch → compute features → write to Hopsworks
│   ├── backfill_pipeline.py          # Step 1b: run feature_pipeline for historical date range
│   └── training_pipeline.py          # Step 2: read features → train → evaluate → save model
│
├── src/
│   ├── data/
│   │   ├── aqicn_client.py           # API wrapper for AQICN
│   │   └── openweather_client.py     # API wrapper for OpenWeatherMap
│   ├── features/
│   │   ├── feature_engineering.py    # All feature computation logic
│   │   └── hopsworks_client.py       # Hopsworks connection + read/write helpers
│   ├── models/
│   │   ├── random_forest.py          # RF model training + eval
│   │   ├── ridge_regression.py       # Ridge model training + eval
│   │   ├── lstm_model.py             # LSTM model (TensorFlow/Keras)
│   │   └── model_selector.py         # Picks best model by RMSE
│   └── utils/
│       ├── alerts.py                 # AQI threshold alerts logic
│       └── config.py                 # Constants, AQI thresholds, env var names
│
├── app/
│   ├── streamlit_app.py              # Main Streamlit entry point
│   ├── pages/
│   │   ├── forecast.py               # 3-day forecast page
│   │   ├── eda.py                    # EDA / historical trends page
│   │   ├── model_insights.py         # SHAP + feature importance page
│   │   └── alerts.py                 # Hazardous AQI alerts page
│   └── components/
│       ├── charts.py                 # Reusable Plotly chart functions
│       └── aqi_badge.py              # AQI color badge component
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # EDA scratch notebook (not used in prod)
│
├── tests/
│   ├── test_feature_engineering.py
│   └── test_model_selector.py
│
├── requirements.txt
├── .env.example                      # Template for secrets (never commit .env)
├── ARCHITECTURE.md                   # This file
└── README.md
```

---

## Data Flow (Step by Step)

```
[AQICN / OpenWeather API]
        │
        ▼  (every hour via GitHub Actions)
[feature_pipeline.py]
  - Fetch raw: pm25, pm10, o3, no2, so2, co, temperature, humidity, wind_speed
  - Compute features: hour_of_day, day_of_week, month, aqi_change_rate, rolling_avg_24h
  - Write to Hopsworks Feature Group: "aqi_features"
        │
        ▼  (Feature Store)
[Hopsworks — Feature Group: aqi_features]
        │
        ├──► [backfill_pipeline.py]  ← one-time run for historical data
        │
        ▼  (every day via GitHub Actions)
[training_pipeline.py]
  - Create Feature View from "aqi_features"
  - Split train/test (time-based split, no shuffle)
  - Train: RandomForest, Ridge, LSTM
  - Evaluate: RMSE, MAE, R²
  - Select best model
  - Save to Hopsworks Model Registry
        │
        ▼  (Model Registry)
[Hopsworks — Model Registry]
        │
        ▼  (Streamlit Cloud — always on)
[streamlit_app.py]
  - Load latest model from Model Registry
  - Load latest features from Feature Store
  - Compute 3-day forecast
  - Display: forecast + EDA + SHAP + alerts
        │
        ▼
[User — browser]
```

---

## Feature Engineering Reference

All features are computed in `src/features/feature_engineering.py`.

### Raw inputs (from API)
| Field         | Type     | Description                 |
| ------------- | -------- | --------------------------- |
| `pm25`        | float    | PM2.5 concentration (µg/m³) |
| `pm10`        | float    | PM10 concentration (µg/m³)  |
| `o3`          | float    | Ozone level                 |
| `no2`         | float    | Nitrogen dioxide            |
| `so2`         | float    | Sulfur dioxide              |
| `co`          | float    | Carbon monoxide             |
| `temperature` | float    | °C                          |
| `humidity`    | float    | % relative humidity         |
| `wind_speed`  | float    | m/s                         |
| `timestamp`   | datetime | UTC timestamp of reading    |

### Computed features
| Feature           | Formula                                   |
| ----------------- | ----------------------------------------- |
| `hour_of_day`     | `timestamp.hour`                          |
| `day_of_week`     | `timestamp.dayofweek` (0=Mon)             |
| `month`           | `timestamp.month`                         |
| `aqi_change_rate` | `(aqi_now - aqi_1h_ago) / 1h`             |
| `rolling_avg_24h` | rolling mean of AQI over past 24 readings |
| `rolling_std_24h` | rolling std of AQI over past 24 readings  |

### Target variable
- `aqi_next_72h`: AQI value 72 hours ahead (regression target)

---

## Hopsworks Usage Patterns

### Connecting
```python
import hopsworks

project = hopsworks.login(
    host="c.app.hopsworks.ai",
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)
fs = project.get_feature_store()
```

### Writing features (feature_pipeline.py)
```python
fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    primary_key=["timestamp"],
    event_time="timestamp",
    description="Hourly AQI features for Karachi"
)
fg.insert(df)  # df is a pandas DataFrame
```

### Reading features for training (training_pipeline.py)
```python
fg = fs.get_feature_group("aqi_features", version=1)
fv = fs.create_feature_view(
    name="aqi_feature_view",
    version=1,
    query=fg.select_all()
)
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
```

### Saving a model
```python
mr = project.get_model_registry()
model = mr.sklearn.create_model(
    name="aqi_random_forest",
    metrics={"rmse": rmse, "mae": mae, "r2": r2}
)
model.save(model_dir)
```

### Loading a model (in Streamlit app)
```python
mr = project.get_model_registry()
best_model = mr.get_best_model("aqi_random_forest", metric="rmse", direction="min")
model_dir = best_model.download()
model = joblib.load(model_dir + "/model.pkl")
```

---

## GitHub Actions Workflows

### Feature pipeline — runs every hour
```yaml
# .github/workflows/feature_pipeline.yml
name: Feature Pipeline
on:
  schedule:
    - cron: '0 * * * *'   # every hour
  workflow_dispatch:        # allow manual trigger

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python pipelines/feature_pipeline.py
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          AQICN_API_KEY: ${{ secrets.AQICN_API_KEY }}
```

### Training pipeline — runs every day
```yaml
# .github/workflows/training_pipeline.yml
name: Training Pipeline
on:
  schedule:
    - cron: '0 2 * * *'   # every day at 2am UTC
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python pipelines/training_pipeline.py
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
```

**Important:** All secrets go in GitHub → Settings → Secrets → Actions. Never hardcode API keys.

---

## ML Models

### Model selection strategy
All three models are trained on every run. The one with the lowest RMSE on the test set is saved to the Model Registry as the "production" model.

### Time-based train/test split
**Never use random shuffle for time series.** Always split by time:
```python
# CORRECT — time-based split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# WRONG — do not use this for time series
# from sklearn.model_selection import train_test_split (with shuffle=True)
```

### LSTM input shape
The LSTM expects a 3D input: `(samples, timesteps, features)`.  
Use a lookback window of 24 (past 24 hourly readings) to predict the next 72h AQI.

---

## AQI Thresholds (for alerts)

| AQI Range | Category                       | Color  |
| --------- | ------------------------------ | ------ |
| 0–50      | Good                           | Green  |
| 51–100    | Moderate                       | Yellow |
| 101–150   | Unhealthy for Sensitive Groups | Orange |
| 151–200   | Unhealthy                      | Red    |
| 201–300   | Very Unhealthy                 | Purple |
| 301+      | Hazardous                      | Maroon |

Alerts are triggered when predicted AQI > 150 for any of the 3 forecast days.

---

## Environment Variables

```bash
# .env.example — copy to .env locally, add to GitHub Secrets for CI/CD
HOPSWORKS_API_KEY=your_key_here
AQICN_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here   # optional fallback
```

---

## Coding Conventions

- All pipeline scripts are **standalone executables** — run with `python pipelines/script.py`
- Use `python-dotenv` locally to load `.env`; GitHub Actions injects secrets as env vars
- DataFrames always use lowercase snake_case column names
- Every function in `src/` has a docstring with Args and Returns
- Log with `logging` module (not print) in pipeline scripts
- All timestamps stored as UTC

---

## Deployment Checklist

1. [ ] Create Hopsworks free account → get API key
2. [ ] Get AQICN API key (free at aqicn.org/api)
3. [ ] Add secrets to GitHub repo (Settings → Secrets → Actions)
4. [ ] Run `backfill_pipeline.py` once manually to populate historical data
5. [ ] Run `training_pipeline.py` once manually to create first model
6. [ ] Enable GitHub Actions workflows
7. [ ] Deploy Streamlit app to Streamlit Cloud (connect GitHub repo)
8. [ ] Verify dashboard loads and shows predictions
