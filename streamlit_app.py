from __future__ import annotations

import json
import os

import hopsworks
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Karachi AQI Predictor", layout="wide")


def _aqi_category(value: float) -> str:
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for Sensitive Groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def _aqi_color(value: float) -> str:
    if value <= 50:
        return "#2e7d32"
    if value <= 100:
        return "#f9a825"
    if value <= 150:
        return "#ef6c00"
    if value <= 200:
        return "#c62828"
    if value <= 300:
        return "#6a1b9a"
    return "#4e342e"


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


def _feature_signal(model: object, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif hasattr(model, "coef_"):
        scores = np.abs(np.asarray(getattr(model, "coef_"), dtype=float))
    else:
        return pd.DataFrame(columns=["feature", "score"])

    if len(scores) != len(feature_cols):
        return pd.DataFrame(columns=["feature", "score"])

    frame = pd.DataFrame({"feature": feature_cols, "score": scores})
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    st.title("Karachi AQI Prediction Service")
    st.caption("Simple serverless dashboard covering forecast, trends, model quality, and alerts.")

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
    current_aqi = float(ready["aqi"].iloc[-1])
    category = _aqi_category(pred)
    category_color = _aqi_color(pred)
    row_count = len(data)

    st.header("Karachi AQI Dashboard")
    st.caption("Current monitoring, 72-hour forecast, and model status from Hopsworks.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current AQI", f"{current_aqi:.1f}")
    c2.metric("Predicted AQI (+72h)", f"{pred:.1f}")
    c3.metric("Rows in Feature Store", f"{row_count}")

    st.markdown(
        f"<div style='padding:0.85rem 1rem;border-left:6px solid {category_color};background:#f7f9fc;color:#1f2937;border-radius:10px;margin:0.5rem 0 1rem 0;'>"
        f"<strong>Forecast category:</strong> {category}"
        f" <span style='color:{category_color};font-weight:700;'>({pred:.1f})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if pred > 150:
        st.error("Predicted AQI exceeds 150. Health advisory: limit outdoor activity and use masks if needed.")
    else:
        st.success("Predicted AQI is below the hazardous alert threshold of 150.")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Recent AQI Trend")
        plot_df = ready.tail(72).copy()
        st.line_chart(plot_df.set_index("timestamp")["aqi"])

    with right:
        st.subheader("Model Metrics")
        if metadata.get("metrics"):
            metrics = metadata["metrics"]
            st.metric("RMSE", f"{metrics.get('rmse', float('nan')):.2f}")
            st.metric("MAE", f"{metrics.get('mae', float('nan')):.2f}")
            st.metric("R²", f"{metrics.get('r2', float('nan')):.2f}")
        else:
            st.info("Model metrics are not available in the saved metadata.")

        st.subheader("Top Feature Drivers")
        drivers = _feature_signal(model, feature_cols)
        if not drivers.empty:
            st.dataframe(drivers.head(5), use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance is not available for this model type.")

    st.subheader("Forecast Snapshot")
    forecast_df = pd.DataFrame(
        {
            "metric": ["Current AQI", "72h Forecast", "Forecast Category", "Feature Rows"],
            "value": [f"{current_aqi:.1f}", f"{pred:.1f}", category, f"{row_count}"],
        }
    )
    st.table(forecast_df)

    with st.expander("What this dashboard covers"):
        st.markdown(
            """
            - **Forecast**: current AQI and the 72-hour prediction from the latest model
            - **EDA**: the last 72 hours of observed AQI
            - **Model insights**: metric summary and top feature drivers
            - **Alerts**: highlighted when predicted AQI goes above 150
            """
        )


if __name__ == "__main__":
    main()
