from __future__ import annotations

import json
import logging
import os
import tempfile

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_training_frame(df: pd.DataFrame, horizon: int = 72) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)

    data["hour_of_day"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["aqi_change_rate"] = data["aqi"].diff(1)
    data["rolling_avg_24h"] = data["aqi"].rolling(window=24, min_periods=24).mean()
    data["rolling_std_24h"] = data["aqi"].rolling(window=24, min_periods=24).std()
    data["aqi_next_72h"] = data["aqi"].shift(-horizon)

    return data.dropna().reset_index(drop=True)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    load_dotenv()

    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    project = hopsworks.login(host=host, api_key_value=api_key)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="aqi_features", version=1)
    raw = fg.read()

    train_df = build_training_frame(raw, horizon=72)

    feature_cols = [
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
        "aqi_change_rate",
        "rolling_avg_24h",
        "rolling_std_24h",
    ]

    split_idx = int(len(train_df) * 0.8)
    train = train_df.iloc[:split_idx]
    test = train_df.iloc[split_idx:]

    x_train = train[feature_cols]
    y_train = train["aqi_next_72h"]
    x_test = test[feature_cols]
    y_test = test["aqi_next_72h"]

    candidates = {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "ridge": Ridge(alpha=1.0),
    }

    best_name = None
    best_model = None
    best_metrics = None

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics = evaluate(y_test, pred)
        logging.info("%s metrics: %s", name, metrics)

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = name
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_name is not None and best_metrics is not None

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pkl")
        metadata_path = os.path.join(tmp, "metadata.json")

        joblib.dump(best_model, model_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": best_name, "features": feature_cols, "metrics": best_metrics}, f, indent=2)

        mr = project.get_model_registry()
        registered = mr.sklearn.create_model(
            name="aqi_best_model",
            metrics=best_metrics,
            description=f"Best model chosen by RMSE: {best_name}",
        )
        registered.save(tmp)

    logging.info("Best model '%s' saved to model registry with metrics %s", best_name, best_metrics)


if __name__ == "__main__":
    main()
