from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from aqi_feature_utils import add_engineered_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _open_meteo_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
            "pm25": hourly.get("pm2_5", []),
            "pm10": hourly.get("pm10", []),
            "o3": hourly.get("ozone", []),
            "no2": hourly.get("nitrogen_dioxide", []),
            "so2": hourly.get("sulphur_dioxide", []),
            "co": hourly.get("carbon_monoxide", []),
        }
    )


def _open_meteo_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
            "temperature": hourly.get("temperature_2m", []),
            "humidity": pd.Series(hourly.get("relative_humidity_2m", []), dtype=float),
            "wind_speed": hourly.get("wind_speed_10m", []),
        }
    )


def build_backfill_dataset(start_date: str, end_date: str, lat: float, lon: float, aqi_mode: str) -> pd.DataFrame:
    air = _open_meteo_air_quality(lat, lon, start_date, end_date)
    weather = _open_meteo_weather(lat, lon, start_date, end_date)
    data = air.merge(weather, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    if aqi_mode == "pm25":
        data["aqi"] = pd.to_numeric(data["pm25"], errors="coerce")
    else:
        data["aqi"] = np.nan
    data = add_engineered_features(data)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features into Hopsworks")
    parser.add_argument("--start-date", default="2025-03-04", help="YYYY-MM-DD")
    parser.add_argument(
        "--end-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="YYYY-MM-DD",
    )
    parser.add_argument("--lat", type=float, default=24.8607)
    parser.add_argument("--lon", type=float, default=67.0011)
    parser.add_argument(
        "--aqi-mode",
        choices=["pm25", "null"],
        default="pm25",
        help="Backfill AQI behavior: pm25 proxy or null",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    backfill_df = build_backfill_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        lat=args.lat,
        lon=args.lon,
        aqi_mode=args.aqi_mode,
    )

    logging.info("Backfill rows prepared: %s", len(backfill_df))
    logging.info("Backfill date range: %s -> %s", backfill_df["timestamp"].min(), backfill_df["timestamp"].max())

    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="AQICN-only AQI features for Karachi with lag and cyclical encoding",
    )

    try:
        existing = fg.read(online=False)
        if existing is not None and not existing.empty:
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            existing_timestamps = set(existing["timestamp"])
            backfill_df = backfill_df[~backfill_df["timestamp"].isin(existing_timestamps)].copy()
            logging.info("Rows remaining after duplicate timestamp filter: %s", len(backfill_df))
    except Exception:
        pass

    fg.insert(backfill_df)
    logging.info(
        "Backfill completed. Inserted %s rows from %s to %s",
        len(backfill_df),
        args.start_date,
        args.end_date,
    )


if __name__ == "__main__":
    main()
