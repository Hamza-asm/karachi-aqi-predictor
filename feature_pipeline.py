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


def fetch_aqicn_current(city: str, api_key: str) -> pd.DataFrame:
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
    if raw_ts:
        ts = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
    else:
        ts = datetime.now(timezone.utc)

    ts = ts.replace(minute=0, second=0, microsecond=0)

    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "aqi": safe_float(data.get("aqi")),
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
        ]
    )


def load_history(feature_store: object) -> pd.DataFrame:
    try:
        fg = feature_store.get_feature_group(name="aqi_features", version=1)
        history = fg.read(online=False)
        if history is None or history.empty:
            return pd.DataFrame()
        return history
    except Exception:
        return pd.DataFrame()


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
    logging.info("Collected AQICN payload: %s", latest.to_dict(orient="records")[0])

    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    history = load_history(fs)
    latest = build_feature_row_for_insert(history, latest)

    logging.info("Feature row columns: %s", latest.columns.tolist())
    logging.info("Feature row preview: %s", latest.to_dict(orient="records")[0])

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="AQICN-only AQI features for Karachi with lag and cyclical encoding",
    )

    fg.insert(latest)
    logging.info("Inserted %s row into feature group aqi_features:1", len(latest))


if __name__ == "__main__":
    main()
