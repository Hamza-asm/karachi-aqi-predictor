from __future__ import annotations

import json
import os

import hopsworks
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Karachi AQI Predictor", layout="wide")


def _login() -> hopsworks.project.Project:
    load_dotenv()
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key and "HOPSWORKS_API_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_API_KEY"]

    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    return hopsworks.login(host=host, api_key_value=api_key)


def _latest_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)

    data["hour_of_day"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    data["aqi_change_rate"] = data["aqi"].diff(1)
    data["rolling_avg_24h"] = data["aqi"].rolling(window=24, min_periods=24).mean()
    data["rolling_std_24h"] = data["aqi"].rolling(window=24, min_periods=24).std()

    ready = data.dropna().reset_index(drop=True)
    latest = ready.tail(1)
    return ready, latest


def main() -> None:
    st.title("Karachi AQI Prediction Service")

    project = _login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(name="aqi_features", version=1)
    data = fg.read()

    ready, latest = _latest_features(data)

    best_model = mr.get_best_model("aqi_best_model", metric="rmse", direction="min")
    model_dir = best_model.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    metadata = {}
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    feature_cols = metadata.get(
        "features",
        [
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
        ],
    )

    pred = float(model.predict(latest[feature_cols])[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Current AQI", f"{float(ready['aqi'].iloc[-1]):.1f}")
    c2.metric("Predicted AQI (+72h)", f"{pred:.1f}")
    c3.metric("Rows in Feature Store", f"{len(data)}")

    st.subheader("Last 72 Hours AQI")
    plot_df = ready.tail(72).copy()
    st.line_chart(plot_df.set_index("timestamp")["aqi"])

    if metadata.get("metrics"):
        st.subheader("Best Model Metrics")
        st.json(metadata["metrics"])


if __name__ == "__main__":
    main()
