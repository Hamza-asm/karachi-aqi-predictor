from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

from aqi_feature_utils import build_feature_row_for_insert, safe_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KARACHI_LAT = 24.8608
KARACHI_LON = 67.0011

FEATURE_COLUMNS = [
    "timestamp",
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
]

# Forecast windows to compute: (label, start_hour, end_hour)
FORECAST_WINDOWS = [("24h", 0, 24), ("48h", 24, 48), ("72h", 48, 72)]


# ── AQI conversion ────────────────────────────────────────────────────────────

def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 concentration (µg/m³) to US AQI."""
    if pm25 is None or (isinstance(pm25, float) and pd.isna(pm25)):
        return float("nan")
    breakpoints = [
        (0.0,   12.0,  0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for lo_c, hi_c, lo_i, hi_i in breakpoints:
        if lo_c <= pm25 <= hi_c:
            return round((hi_i - lo_i) / (hi_c - lo_c) * (pm25 - lo_c) + lo_i)
    return 500.0


# ── Open-Meteo: current readings ─────────────────────────────────────────────

def fetch_openmeteo_current(lat: float = KARACHI_LAT, lon: float = KARACHI_LON) -> pd.DataFrame:
    """
    Fetch the current hour's air quality from Open-Meteo (free, no API key).
    Returns a single-row DataFrame with the same columns as the legacy label fetch.
    """
    # -- air quality ----------------------------------------------------------
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        [
            "pm2_5", "pm10", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
        ],
        "timezone":      "GMT",
        "forecast_days": 1,
    }
    aq_resp = requests.get(aq_url, params=aq_params, timeout=60)
    if aq_resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo AQ error: {aq_resp.status_code} {aq_resp.text[:200]}")

    aq_hourly = aq_resp.json()["hourly"]
    aq_times  = pd.to_datetime(aq_hourly["time"], utc=True)
    now       = pd.Timestamp.now(tz="UTC").floor("h")

    aq_df = pd.DataFrame({
        "timestamp": aq_times,
        "pm25":      pd.to_numeric(aq_hourly["pm2_5"],              errors="coerce"),
        "pm10":      pd.to_numeric(aq_hourly["pm10"],               errors="coerce"),
        "co":        pd.to_numeric(aq_hourly["carbon_monoxide"],    errors="coerce"),
        "no2":       pd.to_numeric(aq_hourly["nitrogen_dioxide"],   errors="coerce"),
        "so2":       pd.to_numeric(aq_hourly["sulphur_dioxide"],    errors="coerce"),
        "o3":        pd.to_numeric(aq_hourly["ozone"],              errors="coerce"),
    })

    row = aq_df[aq_df["timestamp"] == now].copy()
    if row.empty:
        logging.warning("Current hour not found in Open-Meteo AQ response; using latest row")
        row = aq_df.iloc[[-1]].copy()

    # -- weather (temp, humidity, wind) ---------------------------------------
    wx_url = "https://api.open-meteo.com/v1/forecast"
    wx_params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "timezone":      "GMT",
        "forecast_days": 1,
    }
    try:
        wx_resp   = requests.get(wx_url, params=wx_params, timeout=30)
        wx_hourly = wx_resp.json()["hourly"]
        wx_times  = pd.to_datetime(wx_hourly["time"], utc=True)
        wx_df = pd.DataFrame({
            "timestamp":   wx_times,
            "temperature": pd.to_numeric(wx_hourly["temperature_2m"],       errors="coerce"),
            "humidity":    pd.to_numeric(wx_hourly["relative_humidity_2m"], errors="coerce"),
            "wind_speed":  pd.to_numeric(wx_hourly["wind_speed_10m"],       errors="coerce"),
        })
        wx_row = wx_df[wx_df["timestamp"] == now]
        if not wx_row.empty:
            row["temperature"] = wx_row["temperature"].values[0]
            row["humidity"]    = wx_row["humidity"].values[0]
            row["wind_speed"]  = wx_row["wind_speed"].values[0]
        else:
            row["temperature"] = float("nan")
            row["humidity"]    = float("nan")
            row["wind_speed"]  = float("nan")
    except Exception as wx_exc:
        logging.warning("Open-Meteo weather fetch failed: %s", wx_exc)
        row["temperature"] = float("nan")
        row["humidity"]    = float("nan")
        row["wind_speed"]  = float("nan")

    # -- compute AQI from PM2.5 -----------------------------------------------
    row["aqi"] = row["pm25"].apply(pm25_to_aqi)

    row = row.reset_index(drop=True)
    logging.info("Open-Meteo current: %s", row.to_dict(orient="records")[0])
    return row[FEATURE_COLUMNS]


# ── Open-Meteo: 72-hour forecast ──────────────────────────────────────────────

def fetch_openmeteo_forecast(lat: float = KARACHI_LAT, lon: float = KARACHI_LON) -> pd.DataFrame:
    """
    Fetch hourly air quality forecast from Open-Meteo (free, no API key needed).
    Returns 7 days of hourly pm25, co, no2, so2, o3, dust, uv_index.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        ["pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                          "sulphur_dioxide", "ozone", "dust", "uv_index"],
        "timezone":      "GMT",
        "forecast_days": 7,
    }
    response = requests.get(url, params=params, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo forecast error: {response.status_code} {response.text[:200]}")

    hourly = response.json().get("hourly", {})
    times  = pd.to_datetime(hourly.get("time", []), utc=True)

    df = pd.DataFrame({
        "timestamp": times,
        "fc_pm25":   pd.to_numeric(hourly.get("pm2_5", []),            errors="coerce"),
        "fc_co":     pd.to_numeric(hourly.get("carbon_monoxide", []),  errors="coerce"),
        "fc_no2":    pd.to_numeric(hourly.get("nitrogen_dioxide", []), errors="coerce"),
        "fc_so2":    pd.to_numeric(hourly.get("sulphur_dioxide", []),  errors="coerce"),
        "fc_o3":     pd.to_numeric(hourly.get("ozone", []),            errors="coerce"),
        "fc_dust":   pd.to_numeric(hourly.get("dust", []),             errors="coerce"),
        "fc_uvi":    pd.to_numeric(hourly.get("uv_index", []),         errors="coerce"),
    })
    logging.info("Open-Meteo forecast: %s rows (%s → %s)",
                 len(df), df["timestamp"].min(), df["timestamp"].max())
    return df


# ── Forecast window aggregation ───────────────────────────────────────────────

def compute_forecast_windows(current_ts: pd.Timestamp, forecast_df: pd.DataFrame) -> dict:
    """
    For a given timestamp, compute mean forecast values
    for the 0-24h, 24-48h, and 48-72h windows ahead.
    Returns flat dict: fc_pm25_24h, fc_dust_48h, fc_uvi_72h, etc.
    """
    fc     = forecast_df.set_index("timestamp").sort_index()
    result = {}

    for label, h_start, h_end in FORECAST_WINDOWS:
        window_start = current_ts + pd.Timedelta(hours=h_start)
        window_end   = current_ts + pd.Timedelta(hours=h_end)
        window = fc.loc[(fc.index >= window_start) & (fc.index < window_end)]

        for col in ["fc_pm25", "fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]:
            key = f"{col}_{label}"
            result[key] = round(float(window[col].mean()), 4) if not window.empty else float("nan")

    return result


# ── Fallback ──────────────────────────────────────────────────────────────────

def fallback_current(history: pd.DataFrame) -> pd.DataFrame:
    """Build a best-effort row when Open-Meteo is unavailable."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if history is not None and not history.empty:
        latest_history = history.sort_values("timestamp").tail(1).copy().reset_index(drop=True)
        latest_history.loc[:, "timestamp"] = now
        logging.warning("Open-Meteo unavailable; reusing the latest feature-store row as fallback")
        return latest_history

    logging.warning("Open-Meteo unavailable and no history exists; creating empty fallback row")
    return pd.DataFrame([{col: float("nan") if col != "timestamp" else now
                          for col in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)


# ── Hopsworks history ─────────────────────────────────────────────────────────

def load_history(feature_store: object) -> pd.DataFrame:
    """Load existing feature group rows for lag/rolling computation."""
    try:
        fg      = feature_store.get_feature_group(name="aqi_features", version=1)
        history = fg.read(online=False)
        if history is None or history.empty:
            return pd.DataFrame()
        return history
    except Exception:
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    host              = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    # ── Step 1: connect Hopsworks ─────────────────────────────────────────────
    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    feature_store_name = os.getenv("HOPSWORKS_FEATURE_STORE_NAME", "aqi_khi_serverless_featurestore")
    fs      = project.get_feature_store(name=feature_store_name)
    history = load_history(fs)

    # ── Step 2: fetch current readings from Open-Meteo ────────────────────────
    try:
        latest = fetch_openmeteo_current()
    except Exception as exc:
        logging.warning("Open-Meteo current fetch failed, using fallback: %s", exc)
        latest = fallback_current(history)

    # ── Step 3: fetch forecast and compute window features ────────────────────
    try:
        forecast_df       = fetch_openmeteo_forecast()
        current_ts        = latest["timestamp"].iloc[0]
        forecast_features = compute_forecast_windows(current_ts, forecast_df)
        for col, val in forecast_features.items():
            latest[col] = val
        logging.info("Forecast windows added: %s", list(forecast_features.keys()))
    except Exception as exc:
        logging.warning("Open-Meteo forecast failed, using NaN placeholders: %s", exc)
        for label in ["24h", "48h", "72h"]:
            for col in ["fc_pm25", "fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]:
                latest[f"{col}_{label}"] = float("nan")

    # ── Step 4: compute lag/rolling features ──────────────────────────────────
    latest = build_feature_row_for_insert(history, latest)

    logging.info("Feature row columns : %s", latest.columns.tolist())
    logging.info("Feature row preview : %s", latest.to_dict(orient="records")[0])

    # ── Step 5: insert into feature group ─────────────────────────────────────
    fg = fs.get_or_create_feature_group(
        name        = "aqi_features",
        version     = 1,
        primary_key = ["timestamp"],
        event_time  = "timestamp",
        description = "Karachi AQI — Open-Meteo current + 72h forecast features",
    )
    latest = latest.drop(columns=['hours_since_prev', 'is_gap'], errors='ignore')
    fg.insert(latest)
    logging.info("Inserted %s row into feature group aqi_features:1", len(latest))


if __name__ == "__main__":
    main()