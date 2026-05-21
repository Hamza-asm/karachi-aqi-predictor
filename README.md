 <p align="center">
    <img src="images/Title%20Banner.png" alt="Title Banner" width="1000"/>
 </p>



<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python" />
    <img src="https://img.shields.io/badge/Streamlit-Cloud-orange?logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/Hopsworks-Feature%20Store-007ACC" alt="Hopsworks" />
    <img src="https://img.shields.io/badge/Open%20Meteo-API-yellow" alt="Open-Meteo" />
    <img src="https://img.shields.io/badge/GitHub%20Actions-CI-blue" alt="GitHub Actions" />
</p>

> **Author:** Hamza Ali | **Role:** Data Science  
> **Goal:** End-to-end, fully serverless ML system that forecasts Karachi's Air Quality Index (AQI) at 24h, 48h, and 72h horizons.
---

## Introduction

This repository contains a production-oriented, serverless machine learning system that forecasts Karachi's AQI out to 72 hours. It is intentionally lightweight (four flat pipeline scripts), uses Hopsworks as the central data and model layer, and exposes a Streamlit dashboard for visualization and alerting. The project is built for reproducibility and operations: pipelines are runnable locally, CI is provided through GitHub Actions, and models & features are versioned in Hopsworks.

Key points:
- Predictive horizon: 24h, 48h, 72h forecasts (separate models per horizon).
- Data sources: Open-Meteo (weather + pollutant forecasts / history). Historical ground-truth label data was used during earlier experiments — see the PRD for details.
- Feature store & model registry: Hopsworks (free tier).
- Dashboard: Streamlit (deployed to Streamlit Cloud in production).

---

## Dashboard screenshots

<div align="center">
    <img src="images/ForecastOverview.png" alt="Forecast overview" width="800"/>
    <p><em>Forecast overview: 3-day AQI predictions, current AQI badge, and model comparison controls.</em></p>
</div>

<div align="center">
    <img src="images/ModelInsights.png" alt="Model insights" width="800"/>
    <p><em>Model insights: SHAP summary and top feature importances used to explain predictions.</em></p>
</div>

<div align="center">
    <img src="images/EDA.png" alt="EDA view" width="800"/>
    <p><em>Exploratory data analysis: historical AQI trends and pollutant correlation matrices.</em></p>
</div>

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
│      Open-Meteo API (forecast + historical)                      │
└────────────────┬───────────────────────────┬─────────────────────┘
                 │                           │
                 ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     4 PIPELINE SCRIPTS (flat file structure)     │
│                                                                  │
│  feature_pipeline.py     → fetch, engineer features, write FG    │
│  training_pipeline.py    → train models, evaluate, register      │
│  inference_pipeline.py   → predict next 24/48/72h, write preds   │
│  backfill_pipeline.py    → historical Open-Meteo backfill        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        HOPSWORKS                                 │
│  Feature Store (feature groups)  ·  Model Registry (3 models)    │
│  aqi_model_24h · aqi_model_48h · aqi_model_72h                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD (deployed)                      │
│  Real-time forecasts · EDA section · Model comparison tab        │
│  SHAP / LIME explainability · Dark theme cards · Plotly charts   │
└──────────────────────────────────────────────────────────────────┘
                             ▲
                  GitHub Actions CI/CD
              (scheduled pipeline runs)
```

**Key design decisions:**
- Flat 4-script structure over a `src/` package hierarchy — intentional simplicity, easier to debug
- Three separate per-horizon models rather than one multi-output model — cleaner experiment tracking
- Open-Meteo for features (free, no API key, 7-day forecast + 3-month history)
- Ground-truth labels (historical; US EPA standard)

---

## My Journey — From Idea to Deployed System

### Phase 1 — Design & Setup

Started with the goal of building a real, production-like ML system — not just a notebook. Chose a serverless architecture to avoid Docker/Airflow overhead. Mapped out the data sources: Open-Meteo for meteorological and forecast features (primary), with historical ground-truth AQI label data used during early experiments. Settled on Hopsworks as both the feature store and model registry since it offered a free tier that could handle the project's scale.

Initially considered OpenWeatherMap as an additional data source but dropped it due to API activation delays. GitHub Actions was chosen for orchestration over Airflow for the same simplicity-first reason.

### Phase 2 — Data Pipeline & Feature Engineering

Built the feature pipeline to pull hourly data from Open-Meteo (with historical ground-truth label data used during early experiments), engineer lag features (1h, 3h, 6h, 12h, 24h), rolling statistics, and cyclical time encodings (sin/cos for hour and day-of-week), then write everything into a Hopsworks feature group.

The backfill pipeline pulled 3 months of Open-Meteo historical data to bootstrap the feature group with enough history to train on.

### Phase 3 — The Distribution Shift Crisis (Biggest Setback)

This was the hardest part of the project. After training the initial models, the metrics were catastrophically bad — negative R². After investigation, the root cause was a **data distribution shift**: the synthetic Open-Meteo backfill data had a very different AQI mean than the later real label data. The model had effectively learned from two different distributions.

**Fix:** Deleted and rebuilt the feature group entirely — twice — until the feature group used only the later real label data from **2025-03-04 onward**. This single data quality fix had a bigger impact on model performance than any hyperparameter tuning.

**Lesson learned:** Data quality beats model complexity, every time.

### Phase 4 — Model Training

Trained three separate Ridge regression models (one per forecast horizon) on the clean dataset of 26 features. Final metrics after fixing the data:

| Horizon | R²    |
| ------- | ----- |
| 24h     | ~0.22 |
| 48h     | ~0.17 |
| 72h     | ~0.14 |

These numbers are honest. Papers reporting R² ≈ 0.99 for "AQI prediction" are almost always solving a **nowcasting** problem — using same-day inputs to predict same-day AQI — which is fundamentally easier. This project is a genuine **forecasting** problem with no future ground truth available at prediction time.

The use of shifted actuals as training targets was deliberately framed as a "perfect forecast proxy / upper-bound baseline" rather than "target leakage" — a more accurate framing for cold-start forecast scenarios.

### Phase 5 — Dashboard & Three Performance Bugs

Deployed a Streamlit dashboard with dark theme cards, Plotly charts, a real EDA section, a model comparison tab across all horizons, and SHAP/LIME explainability. Three bugs hit in production:

**Bug 1 — Model cache TTL too short**
Frequent large model re-downloads from Hopsworks were degrading performance. The model cache TTL was set too aggressively short. Fixed by extending it to a 6-hour TTL.

**Bug 2 — History cache never expiring**
The history cache was keyed on an object reference rather than a stable value, so it never actually expired or refreshed. Fixed by using the current UTC hour as the cache-bust key parameter.

**Bug 3 — Stale "Last Updated" timestamp**
The dashboard was displaying a stale timestamp that didn't reflect the actual data time. Fixed by reading the timestamp directly from the sorted history DataFrame rather than a cached variable.

### Phase 6 — The Hopsworks Infinite Load / Pipeline Hang

One of the most frustrating operational issues: the feature pipeline would occasionally run for **30+ minutes** without completing, blocking everything. This was caused by a Hopsworks job getting stuck — the job didn't fail cleanly, it just hung. The 7-day forecast data window would not get written to the feature group during these episodes.

**Fix:** Manually kill the hung job via the Hopsworks UI, then re-trigger the pipeline. After a kill + restart, the writes completed normally. This confirmed the issue was a stuck job, not a data or logic problem.

**Mitigation added:** Set explicit job timeouts and added monitoring so that hung runs don't silently block downstream inference and dashboard freshness.

### Phase 7 — CI/CD (In Progress)

GitHub Actions is configured for scheduled pipeline runs but not yet fully enabled. Deliberate choice: local development was stabilized first before handing off to automation. The preference was local-first, CI/CD second — enabling CI before the pipelines are stable just turns debugging into a log-reading exercise.

---

## Challenges & How I Managed Them

| Challenge                                 | Root Cause                                                                                    | Resolution                                                                                    |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Negative R² after initial training        | Distribution shift — synthetic backfill data vs. later real label data had very different AQI means | Deleted and rebuilt the feature group twice; kept only later real label data from 2025-03-04 onward |
| Hopsworks pipeline hanging for 30+ min    | Stuck job — didn't fail, just hung; 7-day forecast data not persisted                         | Manually killed the job via Hopsworks UI, re-triggered; added explicit timeouts               |
| Dashboard model re-downloads too frequent | Cache TTL set too short                                                                       | Extended model cache TTL to 6 hours                                                           |
| History cache never refreshing            | Cache keyed on object reference instead of stable value                                       | Re-keyed cache on UTC hour                                                                    |
| Stale "Last Updated" timestamp            | Reading from a cached variable instead of live data                                           | Read timestamp directly from sorted history DataFrame                                         |
| AQI standard mismatch                     | Ground-truth label source uses US EPA AQI; Open-Meteo website shows European EAQI               | Documented on dashboard; users informed the scales are not comparable                         |
| OpenWeatherMap API never activated        | Provider-side delay                                                                           | Dropped entirely; Open-Meteo covers all needed features for free                              |
| R² misinterpretation risk                 | Papers show ~0.99 but solve nowcasting, not forecasting                                       | Documented the distinction clearly; honest reporting of 0.14–0.22 range                       |

---

## Key Learnings

1. **Fix your data before tuning your model.** The distribution shift fix gave a larger R² improvement than any model change.
2. **Nowcasting ≠ Forecasting.** R² of 0.22 for a 24h-ahead cold-start AQI forecast is legitimate. Don't benchmark against papers solving a different problem.
3. **Simple architecture is a feature.** Four flat scripts beat a complex package hierarchy for a project of this scope.
4. **Operational bugs are real bugs.** Cache TTL, stuck jobs, and stale timestamps are production problems that matter just as much as model metrics.
5. **Serverless first.** Avoiding Docker/Airflow kept the project deployable without infrastructure overhead.

---

*Dataset start: 2025-03-04 | Features: 26 (lags, rolling stats, cyclical encodings) | Models: Ridge regression | Registry: Hopsworks*

## Dependencies & Run (local)

Follow these steps to set up the project locally and run pipelines or the dashboard. These instructions assume a Windows development machine (PowerShell) and a Python 3.10+ environment.

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Create a `.env` from the example and add required secrets (local testing only):

```powershell
copy .env.example .env
# Edit .env and populate HOPSWORKS_API_KEY and any other needed keys
```

4. Run pipelines locally (examples):

```powershell
# Feature pipeline (fetch + write features to Hopsworks)
python feature_pipeline.py

# Backfill (one-time historical run)
python backfill_pipeline.py --start 2024-11-01 --end 2025-03-01

# Training pipeline (train + register model)
python training_pipeline.py
```

5. Run the Streamlit dashboard locally:

```powershell
streamlit run streamlit_app.py
```

Notes:
- CI (GitHub Actions) runs the same scripts in a cloud environment — make sure secrets are in GitHub Secrets.
- If you use Hopsworks, set `HOPSWORKS_API_KEY` in `.env` (local) and in GitHub Secrets (CI).

---