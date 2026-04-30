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


# ── Open-Meteo: historical air quality (real measured values) ──────────────────
def _fetch_historical_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch real measured hourly pollutant data from Open-Meteo air quality archive.
    No API key required. Returns pm25, pm10, o3, no2, so2, co.
    """
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

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly.get("time", []), utc=True),
        "pm25":      pd.to_numeric(hourly.get("pm2_5", []),              errors="coerce"),
        "pm10":      pd.to_numeric(hourly.get("pm10", []),               errors="coerce"),
        "o3":        pd.to_numeric(hourly.get("ozone", []),              errors="coerce"),
        "no2":       pd.to_numeric(hourly.get("nitrogen_dioxide", []),   errors="coerce"),
        "so2":       pd.to_numeric(hourly.get("sulphur_dioxide", []),    errors="coerce"),
        "co":        pd.to_numeric(hourly.get("carbon_monoxide", []),    errors="coerce"),
    })
    logging.info("Historical air quality rows fetched: %s", len(df))
    return df


# ── Open-Meteo: historical weather archive (real measured values) ──────────────
def _fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch real measured hourly weather from Open-Meteo archive API.
    No API key required. Returns temperature, humidity, wind_speed.
    """
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

    df = pd.DataFrame({
        "timestamp":   pd.to_datetime(hourly.get("time", []), utc=True),
        "temperature": pd.to_numeric(hourly.get("temperature_2m", []),       errors="coerce"),
        "humidity":    pd.to_numeric(hourly.get("relative_humidity_2m", []), errors="coerce"),
        "wind_speed":  pd.to_numeric(hourly.get("wind_speed_10m", []),       errors="coerce"),
    })
    logging.info("Historical weather rows fetched: %s", len(df))
    return df


# ── Open-Meteo: air quality forecast (future signal for 24h/48h/72h) ──────────
def _fetch_forecast_air_quality(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch hourly air quality forecast from Open-Meteo (up to 7 days ahead, free).
    Used to compute fc_pm25_24h, fc_pm10_24h etc. for the most recent backfill rows.
    No API key required.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "timezone":  "UTC",
        "forecast_days": 7,
        "hourly": "pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp":  pd.to_datetime(hourly.get("time", []), utc=True),
        "fc_pm25":    pd.to_numeric(hourly.get("pm2_5", []),            errors="coerce"),
        "fc_co":      pd.to_numeric(hourly.get("carbon_monoxide", []),  errors="coerce"),
        "fc_no2":     pd.to_numeric(hourly.get("nitrogen_dioxide", []), errors="coerce"),
        "fc_so2":     pd.to_numeric(hourly.get("sulphur_dioxide", []),  errors="coerce"),
        "fc_o3":      pd.to_numeric(hourly.get("ozone", []),            errors="coerce"),
        "fc_dust":    pd.to_numeric(hourly.get("dust", []),             errors="coerce"),
        "fc_uvi":     pd.to_numeric(hourly.get("uv_index", []),         errors="coerce"),
    })
    logging.info("Forecast air quality rows fetched: %s", len(df))
    return df


# ── Build forecast feature windows ────────────────────────────────────────────
def _add_forecast_windows(base_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in base_df, compute mean forecast values for
    the 0-24h, 24-48h, and 48-72h windows ahead of that row's timestamp.
    Adds columns: fc_pm25_24h, fc_co_24h, fc_no2_24h, fc_so2_24h, fc_o3_24h,
                  fc_dust_24h, fc_uvi_24h and the same for 48h / 72h.
    Rows where forecast window is not available get NaN — that is acceptable.
    """
    forecast_df = forecast_df.set_index("timestamp").sort_index()
    result = base_df.copy()

    for horizon_label, h_start, h_end in [("24h", 0, 24), ("48h", 24, 48), ("72h", 48, 72)]:
        pm25_vals, co_vals, no2_vals, so2_vals, o3_vals, dust_vals, uvi_vals = [], [], [], [], [], [], []

        for ts in base_df["timestamp"]:
            window_start = ts + pd.Timedelta(hours=h_start)
            window_end   = ts + pd.Timedelta(hours=h_end)
            window = forecast_df.loc[
                (forecast_df.index >= window_start) &
                (forecast_df.index <  window_end)
            ]
            if window.empty:
                pm25_vals.append(float("nan"))
                co_vals.append(float("nan"))
                no2_vals.append(float("nan"))
                so2_vals.append(float("nan"))
                o3_vals.append(float("nan"))
                dust_vals.append(float("nan"))
                uvi_vals.append(float("nan"))
            else:
                pm25_vals.append(round(window["fc_pm25"].mean(), 4))
                co_vals.append(round(window["fc_co"].mean(), 4))
                no2_vals.append(round(window["fc_no2"].mean(), 4))
                so2_vals.append(round(window["fc_so2"].mean(), 4))
                o3_vals.append(round(window["fc_o3"].mean(), 4))
                dust_vals.append(round(window["fc_dust"].mean(), 4))
                uvi_vals.append(round(window["fc_uvi"].mean(),  4))

        result[f"fc_pm25_{horizon_label}"] = pm25_vals
        result[f"fc_co_{horizon_label}"] = co_vals
        result[f"fc_no2_{horizon_label}"] = no2_vals
        result[f"fc_so2_{horizon_label}"] = so2_vals
        result[f"fc_o3_{horizon_label}"] = o3_vals
        result[f"fc_dust_{horizon_label}"] = dust_vals
        result[f"fc_uvi_{horizon_label}"]  = uvi_vals

    return result


# ── Main dataset builder ───────────────────────────────────────────────────────
def build_backfill_dataset(
    start_date: str,
    end_date:   str,
    lat:        float,
    lon:        float,
    add_forecast: bool = True,
) -> pd.DataFrame:
    """
    Build the full backfill DataFrame.
    """
    air     = _fetch_historical_air_quality(lat, lon, start_date, end_date)
    weather = _fetch_historical_weather(lat, lon, start_date, end_date)

    data = (
        air.merge(weather, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    data["humidity"] = pd.to_numeric(data["humidity"], errors="coerce").astype(float)
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce").astype(float)
    data["wind_speed"] = pd.to_numeric(data["wind_speed"], errors="coerce").astype(float)
    for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

    # Use PM2.5 as the historical AQI proxy so model training has labels.
    # We convert the concentration (µg/m³) to the US EPA AQI scale for consistency.
    data["aqi"] = data["pm25"].apply(pm25_to_aqi)

    # Add time features + lag + rolling
    data = add_engineered_features(data)

    # Add Open-Meteo real forecast windows (only available near current date)
    if add_forecast:
        try:
            forecast_df = _fetch_forecast_air_quality(lat, lon)
            data = _add_forecast_windows(data, forecast_df)
            logging.info("Real forecast windows added to recent rows.")
        except Exception as exc:
            logging.warning("Forecast fetch failed, creating empty forecast columns: %s", exc)
            for horizon in ["24h", "48h", "72h"]:
                for col in ["fc_pm25", "fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]:
                    data[f"{col}_{horizon}"] = float("nan")
    else:
        for horizon in ["24h", "48h", "72h"]:
            for col in ["fc_pm25", "fc_co", "fc_no2", "fc_so2", "fc_o3", "fc_dust", "fc_uvi"]:
                data[f"{col}_{horizon}"] = float("nan")

    # ── THE "IDEAL CONDITIONS" BACKFILL HACK ──
    # Simulates historical forecasts by shifting actual weather backwards.
    # Uses fillna() so it never overwrites the real forecasts retrieved above.
    logging.info("Simulating historical forecasts for remaining missing rows...")
    for horizon, shift_hours in [("24h", -24), ("48h", -48), ("72h", -72)]:
        data[f"fc_pm25_{horizon}"] = data[f"fc_pm25_{horizon}"].fillna(data["pm25"].shift(shift_hours))
        data[f"fc_co_{horizon}"]   = data[f"fc_co_{horizon}"].fillna(data["co"].shift(shift_hours))
        data[f"fc_no2_{horizon}"]  = data[f"fc_no2_{horizon}"].fillna(data["no2"].shift(shift_hours))
        data[f"fc_so2_{horizon}"]  = data[f"fc_so2_{horizon}"].fillna(data["so2"].shift(shift_hours))
        data[f"fc_o3_{horizon}"]   = data[f"fc_o3_{horizon}"].fillna(data["o3"].shift(shift_hours))
        
        # Note: Dust and UVI are not returned in the standard historical endpoints.
        # They will remain NaN and be cleanly dropped by the trainer script's coverage filter.
        # The 5 primary pollutants above are more than enough signal for the models.

    logging.info("Backfill dataset built: %s rows, %s columns", len(data), len(data.columns))
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features into Hopsworks")
    default_start = (datetime.now(timezone.utc).date() - timedelta(days=DEFAULT_HISTORY_DAYS)).isoformat()
    parser.add_argument("--start-date", default=default_start,          help="YYYY-MM-DD")
    parser.add_argument("--end-date",   default=datetime.now(timezone.utc).date().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--lat",        type=float, default=KARACHI_LAT)
    parser.add_argument("--lon",        type=float, default=KARACHI_LON)
    parser.add_argument("--no-forecast", action="store_true", help="Skip forecast window columns")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")

    backfill_df = build_backfill_dataset(
        start_date    = args.start_date,
        end_date      = args.end_date,
        lat           = args.lat,
        lon           = args.lon,
        add_forecast  = not args.no_forecast,
    )

    logging.info("Backfill rows prepared: %s", len(backfill_df))
    logging.info("Date range: %s → %s", backfill_df["timestamp"].min(), backfill_df["timestamp"].max())

    project = hopsworks.login(host=host, api_key_value=hopsworks_api_key)
    fs      = project.get_feature_store()
    fg      = fs.get_or_create_feature_group(
        name        = "aqi_features",
        version     = 1,
        primary_key = ["timestamp"],
        event_time  = "timestamp",
        description = "Hourly AQI features for Karachi — AQICN labels + Open-Meteo pollutants + forecast windows",
    )

    # Filter out timestamps already in the feature store
    try:
        existing = fg.read(online=False)
        if existing is not None and not existing.empty:
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            existing_ts = set(existing["timestamp"])
            before = len(backfill_df)
            backfill_df = backfill_df[~backfill_df["timestamp"].isin(existing_ts)].copy()
            logging.info("Duplicate filter: %s → %s rows", before, len(backfill_df))
    except Exception as exc:
        logging.warning("Could not read existing timestamps: %s", exc)

    if backfill_df.empty:
        logging.info("No new rows to insert — feature store is already up to date.")
        return

    fg.insert(backfill_df)
    logging.info(
        "Backfill completed. Inserted %s rows from %s to %s",
        len(backfill_df), args.start_date, args.end_date,
    )


if __name__ == "__main__":
    main()