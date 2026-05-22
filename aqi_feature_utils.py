from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

LAG_HOURS = [1, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 24]
TARGET_HORIZONS = [24, 48, 72]

FEATURE_COLUMNS = [
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
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_6h",
    "rolling_std_24h",
]


def pm25_to_aqi(pm25: float | None) -> float:
    """Convert PM2.5 concentration (µg/m³) to the US EPA AQI scale."""
    if pm25 is None or pd.isna(pm25):
        return float("nan")
    pm25 = float(pm25)
    if pm25 < 0:
        return 0.0
    if pm25 <= 12.0:
        return ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
    if pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    if pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    if pm25 <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    if pm25 <= 250.4:
        return ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    if pm25 <= 350.4:
        return ((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301
    if pm25 <= 500.4:
        return ((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401
    return 501.0


def aqi_category(aqi: float) -> tuple[str, str]:
    if aqi <= 50:
        return "Good", "#00e400"
    if aqi <= 100:
        return "Moderate", "#f97316"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    if aqi <= 200:
        return "Unhealthy", "#ff0000"
    if aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    return "Hazardous", "#7e0023"


def ensure_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "timestamp" not in data.columns:
        data["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = ensure_datetime_utc(df)
    data = data.sort_values("timestamp").reset_index(drop=True)

    data["hour_of_day"] = data["timestamp"].dt.hour.astype("int64")
    data["day_of_week"] = data["timestamp"].dt.dayofweek.astype("int64")
    data["month"] = data["timestamp"].dt.month.astype("int64")

    data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    gap_hours = data["timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    data["hours_since_prev"] = gap_hours.astype("float32")
    data["is_gap"] = (gap_hours > 1.0).astype("int8")

    if "aqi" in data.columns:
        data["aqi_lag_1h"] = data["aqi"].shift(1)
        data["aqi_lag_3h"] = data["aqi"].shift(3)
        data["aqi_lag_6h"] = data["aqi"].shift(6)
        data["aqi_lag_12h"] = data["aqi"].shift(12)
        data["aqi_lag_24h"] = data["aqi"].shift(24)
        data["rolling_mean_6h"] = data["aqi"].rolling(window=6, min_periods=1).mean()
        data["rolling_mean_24h"] = data["aqi"].rolling(window=24, min_periods=1).mean()
        data["rolling_std_6h"] = data["aqi"].rolling(window=6, min_periods=1).std()
        data["rolling_std_24h"] = data["aqi"].rolling(window=24, min_periods=1).std()
    else:
        for column in [
            "aqi_lag_1h",
            "aqi_lag_3h",
            "aqi_lag_6h",
            "aqi_lag_12h",
            "aqi_lag_24h",
            "rolling_mean_6h",
            "rolling_mean_24h",
            "rolling_std_6h",
            "rolling_std_24h",
        ]:
            data[column] = np.nan

    return data


def build_training_frame(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    # Prefer stored lag/rolling columns from the feature store. Do NOT recompute
    # lags from the `aqi` column here — that would alter the provenance of the
    # precomputed features and enable leakage. If lag/rolling columns are
    # missing, create them as NaN so the training pipeline can decide how to
    # handle/drop those rows.
    lag_cols = [
        "aqi_lag_1h",
        "aqi_lag_3h",
        "aqi_lag_6h",
        "aqi_lag_12h",
        "aqi_lag_24h",
        "rolling_mean_6h",
        "rolling_mean_24h",
        "rolling_std_6h",
        "rolling_std_24h",
    ]

    data = ensure_datetime_utc(df).sort_values("timestamp").reset_index(drop=True)

    # Ensure basic time-derived features exist (safe to compute here).
    for column in ["hour_of_day", "day_of_week", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        if column not in data.columns:
            if column == "hour_of_day":
                data[column] = data["timestamp"].dt.hour.astype("int64")
            elif column == "day_of_week":
                data[column] = data["timestamp"].dt.dayofweek.astype("int64")
            elif column == "month":
                data[column] = data["timestamp"].dt.month.astype("int64")
            elif column == "hour_sin":
                data[column] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
            elif column == "hour_cos":
                data[column] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
            elif column == "dow_sin":
                data[column] = np.sin(2 * np.pi * data["day_of_week"] / 7)
            elif column == "dow_cos":
                data[column] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    # Compute gap features only if they are not already present.
    if "hours_since_prev" not in data.columns or "is_gap" not in data.columns:
        gap_hours = data["timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
        if "hours_since_prev" not in data.columns:
            data["hours_since_prev"] = gap_hours.astype("float32")
        if "is_gap" not in data.columns:
            data["is_gap"] = (gap_hours > 1.0).astype("int8")

    # Ensure lag/rolling columns exist but DO NOT compute them here; leave as NaN
    # if not provided by the feature materialization step.
    for col in lag_cols:
        if col not in data.columns:
            data[col] = np.nan

    # Create target value by shifting the aqi column
    data[f"aqi_next_{horizon_hours}h"] = data["aqi"].shift(-horizon_hours)

    # CRITICAL FIX: only keep rows where the shifted target timestamp is
    # exactly `horizon_hours` ahead. This excludes rows where gaps cause the
    # target to be e.g. 30h, 60h, etc. ahead, which would corrupt training.
    data["_target_timestamp"] = data["timestamp"].shift(-horizon_hours)
    data["_actual_hours_ahead"] = (
        data["_target_timestamp"] - data["timestamp"]
    ).dt.total_seconds() / 3600

    before = len(data)
    data = data[data["_actual_hours_ahead"] == float(horizon_hours)].copy()
    data = data.drop(columns=["_target_timestamp", "_actual_hours_ahead"])
    dropped = before - len(data)
    if dropped:
        import logging
        logging.info(
            "Horizon %sh: dropped %s rows where target was not exactly %sh ahead (timestamp gaps)",
            horizon_hours, dropped, horizon_hours,
        )

    return data


def prepare_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = add_engineered_features(df)
    # The training pipeline uses ffill().fillna(0) on feature columns to be robust to missing values.
    # We apply the same logic here to ensure the latest rows are not dropped if some pollutants are missing.
    cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    data[cols] = data[cols].ffill().fillna(0)
    return data.reset_index(drop=True)


def build_feature_row_for_insert(history: pd.DataFrame, current_row: pd.DataFrame) -> pd.DataFrame:
    history_data = ensure_datetime_utc(history)
    current_data = ensure_datetime_utc(current_row)
    if "aqi" in history_data.columns:
        history_data = history_data[history_data["aqi"].notna()].copy()
    combined = pd.concat([history_data, current_data], ignore_index=True, sort=False)
    combined = add_engineered_features(combined)
    return combined.tail(1).reset_index(drop=True)


def feature_columns() -> list[str]:
    return FEATURE_COLUMNS.copy()


def target_column(horizon_hours: int) -> str:
    return f"aqi_next_{horizon_hours}h"


def safe_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def drop_missing_training_rows(df: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    columns = list(required_columns)
    return df.dropna(subset=columns).reset_index(drop=True)
