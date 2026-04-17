from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def fetch_aqicn_current(city: str, api_key: str) -> pd.DataFrame:
    """Fetch current AQI and pollutant/weather values for a city from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/"
    response = requests.get(url, params={"token": api_key}, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"AQICN API error: {payload}")

    data = payload.get("data", {})
    iaqi = data.get("iaqi", {})

    def val(key: str, default: float = 0.0) -> float:
        raw = iaqi.get(key, {}).get("v")
        return float(raw) if raw is not None else default

    raw_ts = data.get("time", {}).get("iso")
    if raw_ts:
        ts = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
    else:
        ts = datetime.now(timezone.utc)

    # Round to hour for consistent event time keys.
    ts = ts.replace(minute=0, second=0, microsecond=0)

    row = {
        "timestamp": ts,
        "aqi": float(data.get("aqi", 0.0)),
        "pm25": val("pm25"),
        "pm10": val("pm10"),
        "o3": val("o3"),
        "no2": val("no2"),
        "so2": val("so2"),
        "co": val("co"),
        "temperature": val("t"),
        "humidity": val("h"),
        "wind_speed": val("w"),
    }
    return pd.DataFrame([row])


def add_base_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["hour_of_day"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    return data


def main() -> None:
    load_dotenv()

    city = os.getenv("AQI_CITY", "Karachi")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    aqicn_api_key = os.getenv("AQICN_API_KEY")

    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")
    if not aqicn_api_key:
        raise RuntimeError("AQICN_API_KEY is missing")

    latest = fetch_aqicn_current(city=city, api_key=aqicn_api_key)
    latest = add_base_time_features(latest)

    logging.info("Raw AQICN data collected:")
    logging.info("\nDataFrame shape: %s", latest.shape)
    logging.info("\nDataFrame dtypes:\n%s", latest.dtypes)
    logging.info("\nDataFrame preview:\n%s", latest)
    logging.info("\nDataFrame info:\n%s", latest.info())

    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI and weather features for Karachi",
    )

    fg.insert(latest)
    logging.info("Inserted %s row into feature group aqi_features:1", len(latest))


if __name__ == "__main__":
    main()
