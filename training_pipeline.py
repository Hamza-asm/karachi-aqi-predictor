from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass

import hopsworks
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from aqi_feature_utils import (
    build_training_frame,
    drop_missing_training_rows,
    feature_columns,
    target_column,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LOOKBACK_WINDOW = 24
DEFAULT_TRAIN_START_DATE = "2025-03-04"


@dataclass
class ModelResult:
    name: str
    model_type: str
    metrics: dict
    artifact: object
    scaler: StandardScaler | None = None



def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}



def build_sequence_dataset(
    df: pd.DataFrame,
    horizon_hours: int,
    lookback_window: int = LOOKBACK_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    features = feature_columns()
    target_col = target_column(horizon_hours)
    x_values: list[np.ndarray] = []
    y_values: list[float] = []

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    for end_index in range(lookback_window - 1, len(ordered)):
        start_index = end_index - lookback_window + 1
        window = ordered.iloc[start_index : end_index + 1]
        target_value = ordered.iloc[end_index][target_col]
        if pd.isna(target_value):
            continue
        sequence = window[features].to_numpy(dtype=np.float32)
        if np.isnan(sequence).any():
            continue
        x_values.append(sequence)
        y_values.append(float(target_value))

    if not x_values:
        raise RuntimeError(f"No LSTM sequences could be constructed for horizon {horizon_hours}h")

    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)



def train_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_count: int,
) -> tuple[keras.Model, StandardScaler, dict]:
    scaler = StandardScaler()
    train_flat = x_train.reshape(-1, feature_count)
    scaler.fit(train_flat)

    x_train_scaled = scaler.transform(train_flat).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test.reshape(-1, feature_count)).reshape(x_test.shape)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x_train.shape[1], feature_count)),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    model.fit(
        x_train_scaled,
        y_train,
        validation_split=0.1,
        epochs=40,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
    )

    predictions = model.predict(x_test_scaled, verbose=0).reshape(-1)
    metrics = evaluate(pd.Series(y_test), predictions)
    return model, scaler, metrics



def train_tabular_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, tuple[object, dict]]:
    candidates: dict[str, object] = {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "ridge": Ridge(alpha=1.0),
    }
    results: dict[str, tuple[object, dict]] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate(y_test, predictions)
        logging.info("%s metrics: %s", name, metrics)
        results[name] = (model, metrics)

    return results



def train_for_horizon(project: object, df: pd.DataFrame, horizon_hours: int) -> ModelResult:
    target_col = target_column(horizon_hours)
    model_frame = build_training_frame(df, horizon_hours)
    required_columns = feature_columns() + [target_col]
    model_frame = drop_missing_training_rows(model_frame, required_columns)

    if model_frame.empty:
        raise RuntimeError(f"No training rows available for horizon {horizon_hours}h")

    split_idx = int(len(model_frame) * 0.8)
    train = model_frame.iloc[:split_idx].copy()
    test = model_frame.iloc[split_idx:].copy()

    x_train = train[feature_columns()]
    y_train = train[target_col]
    x_test = test[feature_columns()]
    y_test = test[target_col]

    tabular_results = train_tabular_models(x_train, y_train, x_test, y_test)

    lstm_x, lstm_y = build_sequence_dataset(model_frame, horizon_hours)
    lstm_split_idx = int(len(lstm_x) * 0.8)
    lstm_x_train = lstm_x[:lstm_split_idx]
    lstm_x_test = lstm_x[lstm_split_idx:]
    lstm_y_train = lstm_y[:lstm_split_idx]
    lstm_y_test = lstm_y[lstm_split_idx:]

    lstm_model, lstm_scaler, lstm_metrics = train_lstm_model(
        lstm_x_train,
        lstm_y_train,
        lstm_x_test,
        lstm_y_test,
        feature_count=len(feature_columns()),
    )
    logging.info("lstm metrics: %s", lstm_metrics)

    candidates = {
        **{name: (model, metrics, "sklearn", None) for name, (model, metrics) in tabular_results.items()},
        "lstm": (lstm_model, lstm_metrics, "tensorflow", lstm_scaler),
    }

    best_name = min(candidates, key=lambda key: candidates[key][1]["rmse"])
    best_model, best_metrics, best_type, scaler = candidates[best_name]

    logging.info("Best %sh model: %s with metrics %s", horizon_hours, best_name, best_metrics)

    with tempfile.TemporaryDirectory() as tmp:
        metadata = {
            "model_name": best_name,
            "model_type": best_type,
            "horizon_hours": horizon_hours,
            "lookback_window": LOOKBACK_WINDOW,
            "features": feature_columns(),
            "metrics": best_metrics,
            "candidate_metrics": {
                "random_forest": tabular_results["random_forest"][1],
                "ridge": tabular_results["ridge"][1],
                "lstm": lstm_metrics,
            },
        }

        metadata_path = os.path.join(tmp, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        if best_type == "tensorflow":
            model_path = os.path.join(tmp, "model.keras")
            best_model.save(model_path)
            if scaler is not None:
                joblib.dump(scaler, os.path.join(tmp, "scaler.pkl"))
            registry = project.get_model_registry()
            registered = registry.tensorflow.create_model(
                name=f"aqi_model_{horizon_hours}h",
                metrics=best_metrics,
                description=f"Best {horizon_hours}h AQI model: {best_name}",
            )
            registered.save(tmp)
        else:
            model_path = os.path.join(tmp, "model.pkl")
            joblib.dump(best_model, model_path)
            registry = project.get_model_registry()
            registered = registry.sklearn.create_model(
                name=f"aqi_model_{horizon_hours}h",
                metrics=best_metrics,
                description=f"Best {horizon_hours}h AQI model: {best_name}",
            )
            registered.save(tmp)

    return ModelResult(name=best_name, model_type=best_type, metrics=best_metrics, artifact=best_model, scaler=scaler)


def main() -> None:
    load_dotenv()

    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    train_start_date = os.getenv("TRAIN_START_DATE", DEFAULT_TRAIN_START_DATE)
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    project = hopsworks.login(host=host, api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="aqi_features", version=1)
    raw = fg.read(online=False)
    logging.info("Raw data shape: %s", raw.shape)
    logging.info("Raw data columns: %s", raw.columns.tolist())
    logging.info("Raw data dtypes:\n%s", raw.dtypes)

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    filtered = raw[raw["timestamp"] >= pd.Timestamp(train_start_date, tz="UTC")].copy()
    filtered = filtered[filtered["aqi"].notna()].copy()

    if filtered.empty:
        raise RuntimeError(f"No rows available after TRAIN_START_DATE={train_start_date} with non-null AQI")

    logging.info("Rows keeping (>= %s and AQI not null): %s", train_start_date, len(filtered))
    logging.info(
        "Date range keeping: %s -> %s",
        filtered["timestamp"].min(),
        filtered["timestamp"].max(),
    )

    horizon_results: dict[int, ModelResult] = {}
    for horizon_hours in [24, 48, 72]:
        logging.info("Training horizon: %sh", horizon_hours)
        horizon_results[horizon_hours] = train_for_horizon(project, filtered, horizon_hours)
        logging.info(
            "%sh best model %s (%s) metrics: %s",
            horizon_hours,
            horizon_results[horizon_hours].name,
            horizon_results[horizon_hours].model_type,
            horizon_results[horizon_hours].metrics,
        )

    logging.info("Completed training for horizons: %s", list(horizon_results.keys()))


if __name__ == "__main__":
    main()
