from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

from aqi_feature_utils import build_feature_row_for_insert, safe_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KARACHI_LAT = 24.8608
KARACHI_LON = 67.0011

AQICN_COLUMNS = [
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


def fetch_aqicn_current(city: str, api_key: str) -> pd.DataFrame:
    """Fetch current AQI and pollutant readings from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/"
    response = requests.get(url, params={"token": api_key}, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN API error: {payload}")

    data = payload.get("data", {})
    iaqi = data.get("iaqi", {})

    def val(key: str) -> float:
        raw = iaqi.get(key, {}).get("v")
        return safe_float(raw)

    raw_ts = data.get("time", {}).get("iso")
    now = datetime.now(timezone.utc)
    ts = pd.to_datetime(raw_ts, utc=True).to_pydatetime() if raw_ts else now
    if ts < now - timedelta(days=2):
        logging.warning("AQICN timestamp %s looks stale; using current UTC time instead", ts)
        ts = now
    ts = ts.replace(minute=0, second=0, microsecond=0)

    return pd.DataFrame([{
        "timestamp":   ts,
        "aqi":         safe_float(data.get("aqi")),
        "pm25":        val("pm25"),
        "pm10":        val("pm10"),
        "o3":          val("o3"),
        "no2":         val("no2"),
        "so2":         val("so2"),
        "co":          val("co"),
        "temperature": val("t"),
        "humidity":    val("h"),
        "wind_speed":  val("w"),
    }])


def fallback_aqicn_current(history: pd.DataFrame) -> pd.DataFrame:
    """Build a best-effort AQICN row when the live API is unavailable."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    if history is not None and not history.empty:
        latest_history = history.sort_values("timestamp").tail(1).copy().reset_index(drop=True)
        latest_history.loc[:, "timestamp"] = now
        logging.warning("AQICN unavailable; reusing the latest feature-store row as a fallback")
        return latest_history

    logging.warning("AQICN unavailable and no history exists; creating an empty fallback row")
    return pd.DataFrame([
        {
            "timestamp": now,
            "aqi": float("nan"),
            "pm25": float("nan"),
            "pm10": float("nan"),
            "o3": float("nan"),
            "no2": float("nan"),
            "so2": float("nan"),
            "co": float("nan"),
            "temperature": float("nan"),
            "humidity": float("nan"),
            "wind_speed": float("nan"),
        }
    ], columns=AQICN_COLUMNS)


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
        raise RuntimeError(f"Open-Meteo error: {response.status_code} {response.text[:200]}")

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


def main() -> None:
    load_dotenv()

    city              = os.getenv("AQI_CITY", "Karachi")
    host              = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    aqicn_api_key     = os.getenv("AQICN_API_KEY")

    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")
    if not aqicn_api_key:
        raise RuntimeError("AQICN_API_KEY is missing")

    # ── Step 1: connect Hopsworks early so we have a fallback if AQICN fails ──
    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    fs      = project.get_feature_store()
    history = load_history(fs)

    # ── Step 2: fetch current AQI from AQICN ──────────────────────────────────
    try:
        latest = fetch_aqicn_current(city=city, api_key=aqicn_api_key)
        logging.info("AQICN payload: %s", latest.to_dict(orient="records")[0])
    except Exception as exc:
        logging.warning("AQICN fetch failed, using fallback row instead: %s", exc)
        latest = fallback_aqicn_current(history)
        logging.info("Fallback AQICN payload: %s", latest.to_dict(orient="records")[0])

    # ── Step 3: fetch Open-Meteo forecast and compute window features ──────────
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

    # ── Step 4: compute lag/rolling features and insert into feature store ────
    latest  = build_feature_row_for_insert(history, latest)

    logging.info("Feature row columns : %s", latest.columns.tolist())
    logging.info("Feature row preview : %s", latest.to_dict(orient="records")[0])

    # ── Step 5: insert into feature group ─────────────────────────────────────
    fg = fs.get_or_create_feature_group(
        name        = "aqi_features",
        version     = 1,
        primary_key = ["timestamp"],
        event_time  = "timestamp",
        description = "Karachi AQI — AQICN real labels + Open-Meteo 72h forecast features",
    )
    fg.insert(latest)
    logging.info("Inserted %s row into feature group aqi_features:1", len(latest))


if __name__ == "__main__":
    main()