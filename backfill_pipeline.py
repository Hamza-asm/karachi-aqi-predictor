from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

from aqi_feature_utils import add_engineered_features, pm25_to_aqi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011
DEFAULT_HISTORY_DAYS = 92


# ── Open-Meteo: historical air quality ────────────────────────────────────────
def _fetch_historical_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch real measured hourly pollutant data from Open-Meteo air quality archive.
    No API key required.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "UTC",
        # FIX: pass as list, not comma-separated string
        "hourly": ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"],
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
        "pm25":      pd.to_numeric(hourly.get("pm2_5", []),             errors="coerce"),
        "pm10":      pd.to_numeric(hourly.get("pm10", []),              errors="coerce"),
        "o3":        pd.to_numeric(hourly.get("ozone", []),             errors="coerce"),
        "no2":       pd.to_numeric(hourly.get("nitrogen_dioxide", []),  errors="coerce"),
        "so2":       pd.to_numeric(hourly.get("sulphur_dioxide", []),   errors="coerce"),
        "co":        pd.to_numeric(hourly.get("carbon_monoxide", []),   errors="coerce"),
    })

    null_counts = df.isnull().sum()
    logging.info("Historical air quality rows fetched: %s", len(df))
    logging.info("Null counts per column:\n%s", null_counts)
    return df


# ── Open-Meteo: historical weather ────────────────────────────────────────────
def _fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch real measured hourly weather from Open-Meteo archive API.
    No API key required.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "UTC",
        # FIX: pass as list, not comma-separated string
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp":   pd.to_datetime(hourly.get("time", []), utc=True),
        "temperature": pd.to_numeric(hourly.get("temperature_2m", []),       errors="coerce"),
        "humidity":    pd.to_numeric(hourly.get("relative_humidity_2m", []), errors="coerce"),
        "wind_speed":  pd.to_numeric(hourly.get("wind_speed_10m", []),       errors="coerce"),
    })

    logging.info("Historical weather rows fetched: %s", len(df))
    return df


# ── Open-Meteo: air quality forecast ──────────────────────────────────────────
def _fetch_forecast_air_quality(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch hourly air quality forecast from Open-Meteo (up to 7 days ahead, free).
    Note: dust and uv_index are only available in the forecast, not historical.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "timezone":      "UTC",
        "forecast_days": 7,
        "hourly": ["pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone", "dust", "uv_index"],
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
        "fc_pm25":   pd.to_numeric(hourly.get("pm2_5", []),            errors="coerce"),
        "fc_co":     pd.to_numeric(hourly.get("carbon_monoxide", []),  errors="coerce"),
        "fc_no2":    pd.to_numeric(hourly.get("nitrogen_dioxide", []), errors="coerce"),
        "fc_so2":    pd.to_numeric(hourly.get("sulphur_dioxide", []),  errors="coerce"),
        "fc_o3":     pd.to_numeric(hourly.get("ozone", []),            errors="coerce"),
        "fc_dust":   pd.to_numeric(hourly.get("dust", []),             errors="coerce"),
        "fc_uvi":    pd.to_numeric(hourly.get("uv_index", []),         errors="coerce"),
    })
    logging.info("Forecast air quality rows fetched: %s", len(df))
    return df


# ── Forecast window aggregation ───────────────────────────────────────────────
def _add_forecast_windows(base_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in base_df, compute mean forecast values for
    the 0-24h, 24-48h, and 48-72h windows ahead of that row's timestamp.
    """
    forecast_df = forecast_df.set_index("timestamp").sort_index()
    result = base_df.copy()

    for horizon_label, h_start, h_end in [("24h", 0, 24), ("48h", 24, 48), ("72h", 48, 72)]:
        cols = ["fc_pm25", "fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]
        accumulators = {col: [] for col in cols}

        for ts in base_df["timestamp"]:
            window_start = ts + pd.Timedelta(hours=h_start)
            window_end   = ts + pd.Timedelta(hours=h_end)
            window = forecast_df.loc[
                (forecast_df.index >= window_start) &
                (forecast_df.index <  window_end)
            ]
            for col in cols:
                if window.empty or col not in window.columns:
                    accumulators[col].append(float("nan"))
                else:
                    accumulators[col].append(round(window[col].mean(), 4))

        for col in cols:
            result[f"{col}_{horizon_label}"] = accumulators[col]

    return result


# ── Main dataset builder ───────────────────────────────────────────────────────
def build_backfill_dataset(
    start_date:   str,
    end_date:     str,
    lat:          float,
    lon:          float,
    add_forecast: bool = True,
) -> pd.DataFrame:
    """Build the full backfill DataFrame."""

    air     = _fetch_historical_air_quality(lat, lon, start_date, end_date)
    weather = _fetch_historical_weather(lat, lon, start_date, end_date)

    data = (
        air.merge(weather, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Ensure correct dtypes
    for col in ["pm25", "pm10", "o3", "no2", "so2", "co",
                "temperature", "humidity", "wind_speed"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

    # Derive US AQI from PM2.5
    data["aqi"] = data["pm25"].apply(pm25_to_aqi)

    # Add time features + lag + rolling
    data = add_engineered_features(data)

    # ── Forecast windows ──────────────────────────────────────────────────────
    # Initialize all forecast columns with NaN first
    for horizon in ["24h", "48h", "72h"]:
        for col in ["fc_pm25", "fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]:
            data[f"{col}_{horizon}"] = float("nan")

    if add_forecast:
        # Step 4: Try to overlay real future forecasts onto the recent portion of our dataset
        try:
            forecast_df = _fetch_forecast_air_quality(lat, lon)
            data = _add_forecast_windows(data, forecast_df)
            logging.info("Real forecast windows added to recent rows.")
        except Exception as exc:
            logging.warning("Forecast fetch failed, will use shifted backfill only: %s", exc)

    # ── Backfill historical forecasts by shifting actuals ─────────────────────
    # Step 5: For rows where forecast windows are NaN (historical data), simulate them
    # by shifting the actual observed values. This is a reasonable approximation
    # since tomorrow's AQI closely follows today's pattern.
    logging.info("Filling missing forecast windows with shifted actuals...")
    for horizon, shift_hours in [("24h", -24), ("48h", -48), ("72h", -72)]:
        for col, src in [
            ("fc_pm25", "pm25"), ("fc_co", "co"), ("fc_no2", "no2"),
            ("fc_so2", "so2"),   ("fc_o3", "o3"),
        ]:
            key = f"{col}_{horizon}"
            data[key] = data[key].fillna(data[src].shift(shift_hours))

        # FIX: dust and uvi have no historical equivalent — fill with 0.0
        # instead of leaving NaN so training rows are not dropped.
        # 0.0 is a valid value (no dust event, nighttime UV).
        data[f"fc_dust_{horizon}"] = data[f"fc_dust_{horizon}"].fillna(0.0)
        data[f"fc_uvi_{horizon}"]  = data[f"fc_uvi_{horizon}"].fillna(0.0)

    # ── Final null report ─────────────────────────────────────────────────────
    null_counts = data.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        logging.warning("Columns still containing NaN after backfill:\n%s", null_cols)
    else:
        logging.info("No NaN values remaining in backfill dataset ✓")

    logging.info("Backfill dataset built: %s rows, %s columns", len(data), len(data.columns))
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features into Hopsworks")
    default_start = (datetime.now(timezone.utc).date() - timedelta(days=DEFAULT_HISTORY_DAYS)).isoformat()
    parser.add_argument("--start-date",  default=default_start, help="YYYY-MM-DD")
    parser.add_argument("--end-date",    default=datetime.now(timezone.utc).date().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--lat",         type=float, default=KARACHI_LAT)
    parser.add_argument("--lon",         type=float, default=KARACHI_LON)
    parser.add_argument("--no-forecast", action="store_true", help="Skip forecast window columns")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    host              = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    # Step 6: Trigger the actual data generation pipeline spanning the provided date range
    backfill_df = build_backfill_dataset(
        start_date   = args.start_date,
        end_date     = args.end_date,
        lat          = args.lat,
        lon          = args.lon,
        add_forecast = not args.no_forecast,
    )

    logging.info("Backfill rows prepared: %s", len(backfill_df))
    logging.info("Date range: %s → %s",
                 backfill_df["timestamp"].min(), backfill_df["timestamp"].max())

    # Step 7: Push the fully backfilled records to the Hopsworks feature group
    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    fs      = project.get_feature_store()
    fg      = fs.get_or_create_feature_group(
        name        = "aqi_features",
        version     = 1,
        primary_key = ["timestamp"],
        event_time  = "timestamp",
        description = "Hourly AQI features for Karachi — Open-Meteo pollutants + forecast windows",
    )

    # Step 8: Read previously ingested timestamps from the online store to avoid duplicates
    try:
        existing = fg.read(online=False)
        if existing is not None and not existing.empty:
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            existing_ts = set(existing["timestamp"])
            before = len(backfill_df)
            backfill_df = backfill_df[~backfill_df["timestamp"].isin(existing_ts)].copy()
            logging.info("Duplicate filter: %s → %s rows (skipped %s existing)",
                         before, len(backfill_df), before - len(backfill_df))
    except Exception as exc:
        logging.warning("Could not read existing timestamps (inserting all): %s", exc)

    if backfill_df.empty:
        logging.info("No new rows to insert — feature store is already up to date.")
        return

    # Step 9: Finally, perform the data insertion into Hopsworks
    fg.insert(backfill_df)
    logging.info(
        "Backfill completed. Inserted %s rows from %s to %s",
        len(backfill_df), args.start_date, args.end_date,
    )


if __name__ == "__main__":
    main()