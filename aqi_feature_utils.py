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


def aqi_category(aqi: float) -> tuple[str, str]:
    if aqi <= 50:
        return "Good", "#00e400"
    if aqi <= 100:
        return "Moderate", "#ffff00"
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

    data["hour_of_day"] = data["timestamp"].dt.hour.astype(int)
    data["day_of_week"] = data["timestamp"].dt.dayofweek.astype(int)
    data["month"] = data["timestamp"].dt.month.astype(int)

    data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    if "aqi" in data.columns:
        data["aqi_lag_1h"] = data["aqi"].shift(1)
        data["aqi_lag_3h"] = data["aqi"].shift(3)
        data["aqi_lag_6h"] = data["aqi"].shift(6)
        data["aqi_lag_12h"] = data["aqi"].shift(12)
        data["aqi_lag_24h"] = data["aqi"].shift(24)
        data["rolling_mean_6h"] = data["aqi"].rolling(window=6, min_periods=6).mean()
        data["rolling_mean_24h"] = data["aqi"].rolling(window=24, min_periods=24).mean()
        data["rolling_std_6h"] = data["aqi"].rolling(window=6, min_periods=6).std()
        data["rolling_std_24h"] = data["aqi"].rolling(window=24, min_periods=24).std()
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
    data = add_engineered_features(df)
    data[f"aqi_next_{horizon_hours}h"] = data["aqi"].shift(-horizon_hours)
    return data


def prepare_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = add_engineered_features(df)
    return data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


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
