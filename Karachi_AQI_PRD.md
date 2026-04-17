# Karachi AQI Predictor

**Product Requirements Document (PRD)**

Version 1.0 | Internship Project

**Status: Draft**

---

# 1. Product Overview

Karachi AQI Predictor is a fully serverless, end-to-end machine learning system that forecasts Karachi's Air Quality Index (AQI) for the next 3 days. The system operates autonomously — data collection, model retraining, and prediction serving all run in the cloud without any manual intervention. The user's local machine does not need to be running.

---

# 2. Problem Statement

Karachi consistently ranks among the most polluted cities in Asia. Residents, healthcare workers, and urban planners lack access to reliable short-term AQI forecasts that would help them make informed decisions — whether to wear a mask, restrict outdoor activity, or issue public health alerts.

Existing AQI dashboards show real-time values but provide no forward-looking predictions. This project addresses that gap with a data-driven, automated forecasting system.

---

# 3. Goals and Non-Goals

## 3.1 Goals

- Predict AQI for the next 72 hours (3 daily values) with measurable accuracy
- Automate the full pipeline: data ingestion, feature storage, model retraining, and serving
- Provide explainable predictions using SHAP feature importance
- Alert users when predicted AQI crosses hazardous thresholds (>150)
- Deploy a publicly accessible dashboard that works 24/7 with no local infrastructure

## 3.2 Non-Goals

- Real-time (sub-hourly) AQI tracking — this is a forecasting product, not a live monitor
- Coverage of cities other than Karachi in v1
- Mobile app — web dashboard only in v1
- Custom user accounts or authentication

---

# 4. Technology Stack

| **Layer**      | **Technology**                    | **Reason**                                                  |
| -------------- | --------------------------------- | ----------------------------------------------------------- |
| Data ingestion | AQICN API / OpenWeatherMap        | Free tier, covers Karachi, includes pollutants + weather    |
| Feature store  | Hopsworks (free tier)             | Serverless, versioned feature groups, Python SDK            |
| Model registry | Hopsworks Model Registry          | Paired with feature store, stores model artifacts + metrics |
| ML models      | Scikit-learn, XGBoost, TensorFlow | Random Forest, Ridge Regression, LSTM                       |
| Explainability | SHAP                              | Feature importance for every prediction                     |
| CI/CD          | GitHub Actions                    | Free, serverless cron — no Docker or Airflow needed         |
| Dashboard      | Streamlit + Streamlit Cloud       | Fast to build, free deployment, Python-native               |
| Language       | Python 3.10+                      | Entire stack is Python                                      |

---

# 5. System Architecture

The system consists of four independently scheduled components that communicate through Hopsworks as the central data layer.

## 5.1 Feature Pipeline

Runs every hour via GitHub Actions. Fetches raw pollutant and weather data from the API, computes derived features, and writes them to the Hopsworks feature group.

- **Script:** `pipelines/feature_pipeline.py`
- **Schedule:** every hour (cron: `'0 * * * *'`)
- **Output:** new rows in Hopsworks feature group `'aqi_features'`

## 5.2 Historical Backfill

A one-time manual run that calls the feature pipeline for a range of past dates to generate enough historical data for model training. Target: at least 6 months of hourly data.

- **Script:** `pipelines/backfill_pipeline.py`
- Run once before training, re-run if more history is needed

## 5.3 Training Pipeline

Runs every day at 02:00 UTC via GitHub Actions. Reads features from the Feature Store, trains all three models, evaluates them, and saves the best one to the Model Registry.

- **Script:** `pipelines/training_pipeline.py`
- **Schedule:** daily at 2am UTC (cron: `'0 2 * * *'`)
- **Train/test split:** time-based (no shuffle) — 80% train, 20% test
- **Model selection:** lowest RMSE on test set wins
- **Output:** versioned model artifact in Hopsworks Model Registry

## 5.4 Web Application

A Streamlit app deployed to Streamlit Cloud. Loads the latest model and features from Hopsworks at runtime and displays the 3-day AQI forecast along with analytics.

- **Entry point:** `app/streamlit_app.py`
- Always on — hosted on Streamlit Cloud, no local server needed

---

# 6. Feature Requirements

## 6.1 Feature Engineering

All feature computation lives in `src/features/feature_engineering.py`.

| **Feature**     | **Type** | **Description**                       |
| --------------- | -------- | ------------------------------------- |
| pm25            | Raw      | PM2.5 concentration (µg/m³)           |
| pm10            | Raw      | PM10 concentration (µg/m³)            |
| o3              | Raw      | Ozone level                           |
| no2             | Raw      | Nitrogen dioxide                      |
| so2             | Raw      | Sulfur dioxide                        |
| co              | Raw      | Carbon monoxide                       |
| temperature     | Raw      | Air temperature in °C                 |
| humidity        | Raw      | Relative humidity (%)                 |
| wind_speed      | Raw      | Wind speed (m/s)                      |
| hour_of_day     | Computed | Hour extracted from timestamp (0–23)  |
| day_of_week     | Computed | Day of week (0=Monday, 6=Sunday)      |
| month           | Computed | Month (1–12)                          |
| aqi_change_rate | Computed | AQI delta vs 1 hour ago               |
| rolling_avg_24h | Computed | 24-hour rolling mean of AQI           |
| rolling_std_24h | Computed | 24-hour rolling std of AQI            |
| aqi_next_72h    | Target   | AQI 72 hours ahead — regression label |

---

# 7. Machine Learning Models

## 7.1 Models to Train

| **Model**        | **Library**        | **Notes**                                                       |
| ---------------- | ------------------ | --------------------------------------------------------------- |
| Random Forest    | Scikit-learn       | Baseline. Good on tabular data, handles non-linearity well.     |
| Ridge Regression | Scikit-learn       | Linear baseline. Fast to train, good interpretability.          |
| LSTM             | TensorFlow / Keras | Deep learning. Uses 24-step lookback window on sequential data. |

## 7.2 Evaluation Metrics

- **RMSE** — Root Mean Squared Error (primary selection metric)
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of determination

## 7.3 Model Selection Rule

After every training run, all three models are evaluated on the time-based test split. The model with the lowest RMSE is saved to the Hopsworks Model Registry as the active production model. The Streamlit app always loads the best model at startup.

---

# 8. Dashboard Requirements

## 8.1 Pages

| **Page**       | **File**                  | **Content**                                                               |
| -------------- | ------------------------- | ------------------------------------------------------------------------- |
| Forecast       | `pages/forecast.py`       | 3-day AQI forecast chart, current AQI badge, hazard level label           |
| EDA            | `pages/eda.py`            | Historical AQI trends, seasonality charts, pollutant correlations         |
| Model Insights | `pages/model_insights.py` | SHAP summary plot, top 5 feature importances, model metrics (RMSE/MAE/R²) |
| Alerts         | `pages/alerts.py`         | Triggered when any forecast day AQI > 150, shows health advisory          |

## 8.2 AQI Color Scale

| **AQI Range** | **Category**                   | **Color** |
| ------------- | ------------------------------ | --------- |
| 0 – 50        | Good                           | Green     |
| 51 – 100      | Moderate                       | Yellow    |
| 101 – 150     | Unhealthy for Sensitive Groups | Orange    |
| 151 – 200     | Unhealthy                      | Red       |
| 201 – 300     | Very Unhealthy                 | Purple    |
| 301+          | Hazardous                      | Maroon    |

---

# 9. CI/CD Pipeline

GitHub Actions is used as the serverless scheduler. No Airflow, no Docker, no local server required. Secrets (API keys) are stored in GitHub repository secrets and injected as environment variables at runtime.

| **Workflow file**                         | **Trigger**          | **Action**                    |
| ----------------------------------------- | -------------------- | ----------------------------- |
| `.github/workflows/feature_pipeline.yml`  | Every hour (cron)    | Run `feature_pipeline.py`     |
| `.github/workflows/training_pipeline.yml` | Every day at 2am UTC | Run `training_pipeline.py`    |
| Both workflows                            | `workflow_dispatch`  | Manual trigger from GitHub UI |

---

# 10. Environment Variables & Secrets

| **Variable**          | **Where to set**              | **Description**                            |
| --------------------- | ----------------------------- | ------------------------------------------ |
| `HOPSWORKS_API_KEY`   | GitHub Secrets + local `.env` | Hopsworks project API key                  |
| `AQICN_API_KEY`       | GitHub Secrets + local `.env` | AQICN data API key (free at aqicn.org/api) |
| `OPENWEATHER_API_KEY` | GitHub Secrets + local `.env` | OpenWeatherMap key (optional fallback)     |

---

# 11. Deployment Checklist

- [ ] Create Hopsworks free account at app.hopsworks.ai and get API key
- [ ] Register for AQICN API key at aqicn.org/api (free)
- [ ] Add `HOPSWORKS_API_KEY` and `AQICN_API_KEY` to GitHub Secrets
- [ ] Run `backfill_pipeline.py` once manually to populate 6 months of historical features
- [ ] Run `training_pipeline.py` once manually to create first model version
- [ ] Enable both GitHub Actions workflows
- [ ] Deploy Streamlit app to Streamlit Cloud by connecting GitHub repo
- [ ] Verify dashboard loads, shows 3-day forecast and AQI alerts

---

# 12. Success Metrics

| **Metric**                   | **Target**                      |
| ---------------------------- | ------------------------------- |
| RMSE on test set             | < 20 AQI units                  |
| MAE on test set              | < 15 AQI units                  |
| Feature pipeline reliability | < 5% missed hourly runs         |
| Dashboard uptime             | 99%+ (Streamlit Cloud SLA)      |
| Alert accuracy               | No false negatives on AQI > 200 |

---

*End of Document*
