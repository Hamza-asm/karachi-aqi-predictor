from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", message=r".*tf\.reset_default_graph.*")

import hopsworks
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from xgboost import XGBRegressor

from aqi_feature_utils import (
    build_training_frame,
    target_column,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LOOKBACK_WINDOW      = 24
DEFAULT_TRAIN_START  = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=92)).date().isoformat()
COVERAGE_THRESHOLD   = 0.05

BASE_FEATURE_COLS = [
    "aqi",
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temperature",
    "humidity",
    "wind_speed",
    "hour_of_day",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "hours_since_prev",
    "is_gap",
]

HORIZON_LAG_COLS = {
    24: ["aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h"],
    48: ["aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h"],
    72: ["aqi_lag_12h", "aqi_lag_24h"],
}

HORIZON_ROLLING_COLS = {
    24: ["rolling_mean_6h", "rolling_mean_24h", "rolling_std_6h", "rolling_std_24h"],
    48: ["rolling_mean_24h", "rolling_std_24h"],
    72: ["rolling_mean_24h", "rolling_std_24h"],
}

HORIZON_FORECAST_WINDOWS = {
    24: ["24h"],
    48: ["48h"],
    72: ["72h"],
}

# PM2.5 forecast directly maps into AQI via the target construction, so keep it out of training features.
FORECAST_BASE_FEATURES = ["fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]

# Forecast feature columns added by feature_pipeline + backfill
FORECAST_FEATURE_COLS = [
    f"{base}_{window}"
    for window in ["24h", "48h", "72h"]
    for base in FORECAST_BASE_FEATURES
]


@dataclass
class ModelResult:
    name:            str
    model_type:      str
    metrics:         dict
    artifact:        object
    all_metrics:     dict        # metrics for ALL candidates, shown on dashboard
    scaler:          StandardScaler | None = None


# Helpers
def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def get_horizon_feature_cols(df: pd.DataFrame, horizon_hours: int) -> list[str]:
    """
    Safely build the feature list for a specific horizon.
    Prevents duplicate columns and checks coverage threshold for forecast signals.
    """
    base_cols = BASE_FEATURE_COLS + HORIZON_LAG_COLS[horizon_hours] + HORIZON_ROLLING_COLS[horizon_hours]
    
    # Target only the forecast columns relevant to this specific horizon
    forecast_cols = [
        f"{base}_{window}"
        for window in HORIZON_FORECAST_WINDOWS.get(horizon_hours, [])
        for base in FORECAST_BASE_FEATURES
    ]
    
    valid_cols = []
    candidate_cols = base_cols + forecast_cols
    
    # Remove any accidental duplicates while preserving order
    unique_candidates = list(dict.fromkeys(candidate_cols))
    
    for col in unique_candidates:
        if col not in df.columns:
            logging.warning("Dropping missing feature column: %s", col)
            continue
            
        if col in forecast_cols:
            coverage = df[col].notna().mean()
            if coverage >= COVERAGE_THRESHOLD:
                valid_cols.append(col)
                logging.info("Including forecast col %s (coverage: %.1f%%)", col, coverage * 100)
            else:
                logging.info(
                    "Skipping forecast col %s (coverage: %.1f%% < %.1f%%)",
                    col, coverage * 100, COVERAGE_THRESHOLD * 100,
                )
        else:
            valid_cols.append(col)
            
    return valid_cols


# LSTM
def build_sequence_dataset(
    df: pd.DataFrame,
    horizon_hours: int,
    feat_cols: list[str],
    lookback_window: int = LOOKBACK_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    target_col = target_column(horizon_hours)
    x_values: list[np.ndarray] = []
    y_values: list[float]      = []

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    for end_idx in range(lookback_window - 1, len(ordered)):
        start_idx    = end_idx - lookback_window + 1
        window       = ordered.iloc[start_idx: end_idx + 1]
        target_value = ordered.iloc[end_idx][target_col]
        
        if pd.isna(target_value):
            continue
            
        sequence = window[feat_cols].to_numpy(dtype=np.float32)
        if np.isnan(sequence).any():
            continue
            
        x_values.append(sequence)
        y_values.append(float(target_value))

    if not x_values:
        raise RuntimeError(f"No LSTM sequences for horizon {horizon_hours}h")

    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def train_lstm(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test:  np.ndarray, y_test:  np.ndarray,
    feature_count: int,
) -> tuple[keras.Model, StandardScaler, dict]:
    scaler     = StandardScaler()
    train_flat = x_train.reshape(-1, feature_count)
    scaler.fit(train_flat)

    x_tr = scaler.transform(train_flat).reshape(x_train.shape)
    x_te = scaler.transform(x_test.reshape(-1, feature_count)).reshape(x_test.shape)

    model = keras.Sequential([
        keras.layers.Input(shape=(x_train.shape[1], feature_count)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
    model.fit(
        x_tr, y_train,
        validation_split=0.1,
        epochs=40, batch_size=32, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
    )
    preds   = model.predict(x_te, verbose=0).reshape(-1)
    metrics = evaluate(pd.Series(y_test), preds)
    return model, scaler, metrics


# Tabular models (RF + Ridge + XGBoost)
def train_tabular_models(
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test:  pd.DataFrame, y_test:  pd.Series,
) -> dict[str, tuple[object, dict]]:
    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "ridge": Ridge(alpha=1.0),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),
    }
    results: dict[str, tuple[object, dict]] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        preds   = model.predict(x_test)
        metrics = evaluate(y_test, preds)
        logging.info("%-15s  RMSE: %.4f  MAE: %.4f  R²: %.4f",
                     name, metrics["rmse"], metrics["mae"], metrics["r2"])
        results[name] = (model, metrics)
    return results


# Per-horizon trainer
def train_for_horizon(project: object, df: pd.DataFrame, horizon_hours: int) -> ModelResult:
    target_col  = target_column(horizon_hours)
    model_frame = build_training_frame(df, horizon_hours)
    feat_cols   = get_horizon_feature_cols(model_frame, horizon_hours)

    logging.info("Horizon %sh: using %s features", horizon_hours, len(feat_cols))

    # FIX 1: NEVER DROP ROWS BASED ON FC FEATURES. Only drop if the target is missing.
    model_frame = model_frame.dropna(subset=[target_col])

    # DROP rows with missing lag features for this horizon — these produce invalid
    # training examples (e.g. aqi_lag_24h NaNs). Only drop lag columns, not forecast cols.
    lag_cols = [c for c in HORIZON_LAG_COLS.get(horizon_hours, []) if c in model_frame.columns]
    if lag_cols:
        before_rows = len(model_frame)
        model_frame = model_frame.dropna(subset=lag_cols)
        dropped = before_rows - len(model_frame)
        if dropped:
            logging.info("Dropped %s rows with NaN lag features for %sh (rows: %s → %s)", dropped, horizon_hours, before_rows, len(model_frame))

    if model_frame.empty:
        raise RuntimeError(f"No training rows for horizon {horizon_hours}h")

    logging.info("Horizon %sh: %s rows, %s features", horizon_hours, len(model_frame), len(feat_cols))

    # FIX 4: SPLIT BEFORE DROPPING/IMPUTING NANS
    split_idx = int(len(model_frame) * 0.8)
    train     = model_frame.iloc[:split_idx].copy()
    test      = model_frame.iloc[split_idx:].copy()

    # FIX 2 & 3: Handle NaNs separately. 
    # Use ffill (safe for time series) and fallback to 0 for any leading NaNs.
    train[feat_cols] = train[feat_cols].ffill().fillna(0)
    test[feat_cols]  = test[feat_cols].ffill().fillna(0)

    x_train, y_train = train[feat_cols], train[target_col]
    x_test,  y_test  = test[feat_cols],  test[target_col]

    # Train tabular models
    tabular_results = train_tabular_models(x_train, y_train, x_test, y_test)

    # Train LSTM using the safely split and imputed data frames
    try:
        lstm_x_train, lstm_y_train = build_sequence_dataset(train, horizon_hours, feat_cols)
        lstm_x_test, lstm_y_test   = build_sequence_dataset(test, horizon_hours, feat_cols)
        
        lstm_model, lstm_scaler, lstm_metrics = train_lstm(
            lstm_x_train, lstm_y_train,
            lstm_x_test, lstm_y_test,
            feature_count=len(feat_cols),
        )
        logging.info("%-15s  RMSE: %.4f  MAE: %.4f  R²: %.4f",
                     "lstm", lstm_metrics["rmse"], lstm_metrics["mae"], lstm_metrics["r2"])
    except Exception as exc:
        logging.warning("LSTM training failed for %sh, skipping: %s", horizon_hours, exc)
        lstm_model, lstm_scaler, lstm_metrics = None, None, {"rmse": 9999, "mae": 9999, "r2": -9999}

    # All candidates including LSTM
    all_candidates = {
        **{name: (model, metrics, "sklearn", None)
           for name, (model, metrics) in tabular_results.items()},
    }
    if lstm_model is not None:
        all_candidates["lstm"] = (lstm_model, lstm_metrics, "tensorflow", lstm_scaler)

    # Select best by RMSE
    best_name = min(all_candidates, key=lambda k: all_candidates[k][1]["rmse"])
    best_model, best_metrics, best_type, best_scaler = all_candidates[best_name]
    logging.info("Best %sh model → %s  RMSE: %.4f  R²: %.4f",
                 horizon_hours, best_name, best_metrics["rmse"], best_metrics["r2"])

    # All metrics dict for dashboard
    all_metrics = {
        name: vals[1] for name, vals in all_candidates.items()
    }

    # Save to Hopsworks Model Registry
    with tempfile.TemporaryDirectory() as tmp:
        metadata = {
            "model_name":        best_name,
            "model_type":        best_type,
            "horizon_hours":     horizon_hours,
            "lookback_window":   LOOKBACK_WINDOW,
            "features":          feat_cols,
            "metrics":           best_metrics,
            "all_model_metrics": all_metrics,
        }
        with open(os.path.join(tmp, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if best_type == "tensorflow":
            best_model.save(os.path.join(tmp, "model.keras"))
            if best_scaler:
                joblib.dump(best_scaler, os.path.join(tmp, "scaler.pkl"))
            reg = project.get_model_registry()
            reg.tensorflow.create_model(
                name=f"aqi_model_{horizon_hours}h",
                metrics=best_metrics,
                description=f"Best {horizon_hours}h model: {best_name}",
            ).save(tmp)
        else:
            joblib.dump(best_model, os.path.join(tmp, "model.pkl"))
            reg = project.get_model_registry()
            reg.sklearn.create_model(
                name=f"aqi_model_{horizon_hours}h",
                metrics=best_metrics,
                description=f"Best {horizon_hours}h model: {best_name}",
            ).save(tmp)

    return ModelResult(
        name=best_name, model_type=best_type,
        metrics=best_metrics, artifact=best_model,
        all_metrics=all_metrics, scaler=best_scaler,
    )


# Entry point
def main() -> None:
    load_dotenv()

    host             = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    api_key          = os.getenv("HOPSWORKS_API_KEY")
    # Default to the project's DEFAULT_TRAIN_START (92 days) unless overridden.
    train_start_date = os.getenv("TRAIN_START_DATE", DEFAULT_TRAIN_START)

    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    project = hopsworks.login(host=host, api_key_value=api_key)
    feature_store_name = os.getenv("HOPSWORKS_FEATURE_STORE_NAME", "aqi_khi_serverless_featurestore")
    fs      = project.get_feature_store(name=feature_store_name)

    # Load and filter data
    fg  = fs.get_feature_group(name="aqi_features", version=1)
    raw = fg.read(online=False)
    logging.info("Raw data: %s rows, %s columns", *raw.shape)

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    # Keep only real AQI rows (Open-Meteo source) (aqi not null) after start date
    filtered = raw[
        (raw["timestamp"] >= pd.Timestamp(train_start_date, tz="UTC")) &
        (raw["aqi"].notna())
    ].copy()

    if filtered.empty:
        raise RuntimeError(f"No rows with non-null AQI after {train_start_date}")

    logging.info("Training on %s rows (%s → %s)",
                 len(filtered),
                 filtered["timestamp"].min().date(),
                 filtered["timestamp"].max().date())

    # Log forecast feature coverage
    fc_cols_present = [c for c in FORECAST_FEATURE_COLS if c in filtered.columns]
    fc_non_null     = filtered[fc_cols_present].notna().any(axis=1).sum() if fc_cols_present else 0
    logging.info("Forecast feature columns present: %s", len(fc_cols_present))
    logging.info("Rows with at least one forecast feature: %s / %s", fc_non_null, len(filtered))

    # Train all horizons
    for horizon in [24, 48, 72]:
        logging.info("=" * 50)
        logging.info("Training horizon: %sh", horizon)
        result = train_for_horizon(project, filtered, horizon)
        logging.info(
            "Registered aqi_model_%sh — best: %s  RMSE: %.4f  R²: %.4f",
            horizon, result.name, result.metrics["rmse"], result.metrics["r2"],
        )

    logging.info("=" * 50)
    logging.info("All horizons trained and registered.")


if __name__ == "__main__":
    main()