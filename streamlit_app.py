from __future__ import annotations

import json
import os
import textwrap
import warnings
from dataclasses import dataclass
import hopsworks
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st
from dotenv import load_dotenv

try:
    from streamlit_autorefresh import st_autorefresh
    AUTREFRESH_AVAILABLE = True
except Exception:
    st_autorefresh = None
    AUTREFRESH_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LimeTabularExplainer = None
    LIME_AVAILABLE = False

from aqi_feature_utils import (
    aqi_category,
    feature_columns,
    prepare_prediction_frame,
    pm25_to_aqi,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* Loading state */
.loading-shell {
    background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(10,14,26,0.96));
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}
.loading-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.7rem;
}
.loading-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #3b82f6;
}
.loading-subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
}
.loading-track {
    position: relative;
    height: 10px;
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 999px;
    overflow: hidden;
}
.loading-fill {
    position: absolute;
    top: 0;
    left: -42%;
    width: 42%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, rgba(29,78,216,0.12) 0%, #3b82f6 42%, #06b6d4 100%);
    box-shadow: 0 0 18px rgba(59, 130, 246, 0.35);
    animation: loadingSweep 1.25s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}
.loading-dots {
    display: inline-flex;
    gap: 0.3rem;
    margin-left: 0.4rem;
}
.loading-dots span {
    width: 0.35rem;
    height: 0.35rem;
    border-radius: 999px;
    background: #06b6d4;
    opacity: 0.35;
    animation: dotBounce 1.1s infinite ease-in-out;
}
.loading-dots span:nth-child(2) { animation-delay: 0.12s; }
.loading-dots span:nth-child(3) { animation-delay: 0.24s; }
@keyframes loadingPulse {
    from { transform: scaleX(0.94); filter: brightness(0.95); }
    to { transform: scaleX(1); filter: brightness(1.15); }
}
@keyframes loadingSweep {
    0% { left: -42%; }
    55% { left: 30%; }
    100% { left: 110%; }
}
@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
    40% { transform: translateY(-3px); opacity: 1; }
}

/* Header */
.aqi-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
}
.aqi-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f0f6ff;
    line-height: 1;
}
.aqi-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.35rem;
    font-weight: 400;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.aqi-timestamp {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #475569;
    text-align: right;
}
.aqi-attribution {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 300;
    letter-spacing: 0.04em;
    color: #94a3b8;
    text-align: right;
    white-space: nowrap;
}

/* KPI strip */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f0f6ff;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* Alert banners */
.alert-danger {
    background: linear-gradient(135deg, #2d0a0a, #3d1515);
    border: 1px solid #dc2626;
    border-left: 4px solid #dc2626;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 600;
    color: #fca5a5;
}
.alert-ok {
    background: linear-gradient(135deg, #071a0f, #0d2b18);
    border: 1px solid #16a34a;
    border-left: 4px solid #16a34a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 500;
    color: #86efac;
}

/* Forecast cards */
.forecast-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
    margin-bottom: 2rem;
}
.forecast-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.forecast-card:hover { border-color: #3b82f6; }
.forecast-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60px;
    background: linear-gradient(0deg, rgba(59,130,246,0.05), transparent);
    pointer-events: none;
}
.fc-horizon {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.25rem;
}
.fc-confidence {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 1rem;
    font-weight: 400;
}
.fc-aqi {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    color: #f0f6ff;
    margin-bottom: 0.75rem;
}
.fc-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.fc-confidence-bar {
    margin-top: 1rem;
    height: 3px;
    border-radius: 2px;
    background: #1e2d4a;
    overflow: hidden;
}
.fc-confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d4a;
}

/* Metrics table */
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.metrics-table th {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid #1e2d4a;
    background: #0d1526;
}
.metrics-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #111827;
    color: #cbd5e1;
}
.metrics-table tr:last-child td { border-bottom: none; }
.metrics-table tr:hover td { background: #111827; }
.metric-good { color: #34d399; font-family: 'Space Mono', monospace; }
.metric-mid  { color: #fbbf24; font-family: 'Space Mono', monospace; }
.metric-bad  { color: #f87171; font-family: 'Space Mono', monospace; }

/* Experiment history */
.exp-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.exp-step {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: #3b82f6;
    background: #0f1f3d;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    white-space: nowrap;
    margin-top: 0.1rem;
}
.exp-text {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.5;
}
.exp-outcome {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.25rem;
}
.exp-outcome.good { color: #34d399; }
.exp-outcome.bad  { color: #f87171; }
.exp-outcome.info { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIDENCE_LABELS = {
    24: ("High confidence", 90),
    48: ("Medium confidence", 65),
    72: ("Indicative only", 35),
}

AQI_COLORS = {
    "Good":                                 "#22c55e",
    "Moderate":                             "#f97316",
    "Unhealthy for Sensitive Groups":   "#f97316",
    "Unhealthy":                            "#ef4444",
    "Very Unhealthy":                   "#a855f7",
    "Hazardous":                            "#dc2626",
}


# ── Data classes & helpers ─────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    horizon: int
    model_name: str
    model_version: str
    model_type: str
    metrics: dict
    all_model_metrics: dict[str, dict] | None
    model: object
    model_dir: str
    scaler: object | None = None
    lookback_window: int = 24
    feature_cols: list[str] | None = None


@st.cache_resource(show_spinner=False)
def _login() -> hopsworks.project.Project:
    load_dotenv()
    host = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key and "HOPSWORKS_API_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_API_KEY"]
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is missing")
    return hopsworks.login(host=host, api_key_value=api_key)


def _feature_store_name() -> str:
    load_dotenv()
    return os.getenv("HOPSWORKS_FEATURE_STORE_NAME", "aqi_khi_serverless_featurestore")


def _latest_model_version(_mr, horizon: int) -> int:
    """Get the LATEST version number (not the best by RMSE)."""
    model_name = f"aqi_model_{horizon}h"
    # Ask the registry for all models and pick the highest version for this name.
    models = _mr.get_models(model_name)
    versions: list[int] = []
    for model in models or []:
        if getattr(model, "name", None) != model_name:
            continue
        version = getattr(model, "version", None)
        if version is not None:
            versions.append(int(version))
    if not versions:
        raise RuntimeError(f"No versions found for {model_name}")
    return max(versions)


@st.cache_resource(show_spinner=False)
def _load_model_bundle(_mr, horizon: int, model_version: int) -> ModelBundle:
    # model_version is part of the cache key, so a new registry version refreshes the download.
    model_name = f"aqi_model_{horizon}h"
    registered_model = _mr.get_model(model_name, version=model_version)
    model_dir = registered_model.download()

    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise RuntimeError(f"metadata.json missing for aqi_model_{horizon}h")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_type    = metadata.get("model_type", "sklearn")
    feature_cols  = metadata.get("features", feature_columns())
    lookback      = int(metadata.get("lookback_window", 24))
    metrics       = metadata.get("metrics", {})
    all_metrics   = metadata.get("all_model_metrics", {})

    # Conditionally load TF dependencies only if required
    if model_type == "tensorflow":
        from tensorflow import keras
        model  = keras.models.load_model(os.path.join(model_dir, "model.keras"))
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    else:
        model  = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = None

    return ModelBundle(
        horizon=horizon, model_name=metadata.get("model_name", f"aqi_model_{horizon}h"),
        model_version=model_version,
        model_type=model_type, metrics=metrics, model=model, model_dir=model_dir,
        all_model_metrics=all_metrics, scaler=scaler, lookback_window=lookback,
        feature_cols=feature_cols,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_history(_fs, _cache_bust: str) -> pd.DataFrame:
    """Load AQI history from the online feature store."""
    fg = _fs.get_feature_group(name="aqi_features", version=1)
    online_data = pd.DataFrame()

    try:
        # Online store gives the absolute latest row
        online_data = fg.read(online=True)
    except Exception as e:
        st.sidebar.warning(f"Note: Online store unavailable ({e}).")

    with st.sidebar.expander("🔍 Data Diagnostics", expanded=False):
        st.write(f"Raw Online Rows: {len(online_data)}")
        if not online_data.empty:
            st.write(f"Latest Online: {pd.to_datetime(online_data['timestamp'], utc=True).max()}")

    if online_data.empty:
        try:
            offline_data = fg.read(online=False)
        except Exception as e:
            st.sidebar.error(f"Error: Offline store read failed ({e}).")
            return pd.DataFrame()

        if offline_data.empty:
            return pd.DataFrame()

        offline_data["timestamp"] = pd.to_datetime(offline_data["timestamp"], utc=True)
        offline_data = offline_data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        offline_data = offline_data[offline_data["aqi"].notna()].copy()
        if offline_data.empty:
            return pd.DataFrame()
        return prepare_prediction_frame(offline_data)

    online_data["timestamp"] = pd.to_datetime(online_data["timestamp"], utc=True)
    online_data = online_data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    online_data = online_data[online_data["aqi"].notna()].copy()

    if online_data.empty:
        return pd.DataFrame()

    return prepare_prediction_frame(online_data)


def _loading_state_html(step: str, percent: int, detail: str) -> str:
    percent = max(0, min(100, percent))
    return f"""
    <div class="loading-shell">
        <div class="loading-row">
            <div>
                <div class="loading-title">Loading dashboard</div>
                <div class="loading-subtitle">{step}<span class="loading-dots"><span></span><span></span><span></span></span></div>
            </div>
            <div class="loading-subtitle">{percent}%</div>
        </div>
        <div class="loading-track">
            <div class="loading-fill" style="width:{percent}%;"></div>
        </div>
        <div class="loading-subtitle" style="margin-top:0.65rem;">{detail}</div>
    </div>
    """


def _predict(bundle: ModelBundle, history: pd.DataFrame) -> float:
    if history.empty:
        raise RuntimeError("No history available for prediction")
    if bundle.model_type == "tensorflow":
        if len(history) < bundle.lookback_window:
            raise RuntimeError(f"Need {bundle.lookback_window} rows for LSTM")
        window = history.tail(bundle.lookback_window)[bundle.feature_cols].to_numpy(dtype=np.float32)
        scaled = bundle.scaler.transform(
            window.reshape(-1, len(bundle.feature_cols))
        ).reshape(1, bundle.lookback_window, len(bundle.feature_cols))
        return float(bundle.model.predict(scaled, verbose=0).reshape(-1)[0])
    return float(bundle.model.predict(history.tail(1)[bundle.feature_cols])[0])


def _shap_importance(bundle: ModelBundle, history: pd.DataFrame) -> pd.DataFrame:
    cache_key = f"shap::{bundle.model_dir}::{len(history)}::{history['timestamp'].iloc[-1].value if not history.empty else 0}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    if bundle.model_type == "tensorflow":
        result = pd.DataFrame(columns=["feature", "importance"])
        st.session_state[cache_key] = result
        return result

    sample  = history.tail(min(200, len(history)))
    x       = sample[bundle.feature_cols]
    try:
        if hasattr(bundle.model, "coef_"):
            explainer = shap.LinearExplainer(bundle.model, x, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(x)
        else:
            explainer = shap.TreeExplainer(bundle.model)
            shap_vals = explainer.shap_values(x)
    except Exception:
        # Some XGBoost/SHAP version combinations cannot parse the serialized base_score.
        # Fall back to a fast feature-importance profile instead of a slow permutation explainer.
        if hasattr(bundle.model, "feature_importances_"):
            importance = np.asarray(bundle.model.feature_importances_, dtype=float)
            if importance.ndim == 1 and len(importance) == len(bundle.feature_cols):
                importance = np.abs(importance)
                total = float(importance.sum())
                if total > 0:
                    importance = importance / total
                result = (
                    pd.DataFrame({"feature": bundle.feature_cols, "importance": importance})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
                st.session_state[cache_key] = result
                return result

        result = pd.DataFrame(columns=["feature", "importance"])
        st.session_state[cache_key] = result
        return result
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    importance = np.abs(np.asarray(shap_vals)).mean(axis=0)
    return (
        pd.DataFrame({"feature": bundle.feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _lime_importance(bundle: ModelBundle, history: pd.DataFrame) -> pd.DataFrame:
    cache_key = f"lime::{bundle.model_dir}::{len(history)}::{history['timestamp'].iloc[-1].value if not history.empty else 0}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    if bundle.model_type == "tensorflow":
        result = pd.DataFrame(columns=["feature", "weight"])
        st.session_state[cache_key] = result
        return result

    background = history.tail(min(500, len(history)))[bundle.feature_cols].copy()
    if background.empty:
        result = pd.DataFrame(columns=["feature", "weight"])
        st.session_state[cache_key] = result
        return result

    row = history.tail(1)[bundle.feature_cols].iloc[0].to_numpy()

    if bundle.model_type == "xgboost" or not LIME_AVAILABLE:
        background_mean = background.mean(numeric_only=True)
        background_std = background.std(numeric_only=True).replace(0, 1).fillna(1)
        if hasattr(bundle.model, "feature_importances_"):
            weights = np.asarray(bundle.model.feature_importances_, dtype=float)
            if weights.ndim != 1 or len(weights) != len(bundle.feature_cols):
                weights = np.ones(len(bundle.feature_cols), dtype=float)
        else:
            weights = np.ones(len(bundle.feature_cols), dtype=float)

        total = float(np.abs(weights).sum())
        if total > 0:
            weights = weights / total

        signed_local = ((row - background_mean.to_numpy()) / background_std.to_numpy()) * weights
        result = (
            pd.DataFrame({"feature": bundle.feature_cols, "weight": signed_local})
            .reindex(columns=["feature", "weight"])
            .sort_values("weight", key=lambda s: np.abs(s), ascending=False)
            .reset_index(drop=True)
        )
        st.session_state[cache_key] = result
        return result

    def _lime_predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=bundle.feature_cols)
        return bundle.model.predict(frame)

    try:
        explainer = LimeTabularExplainer(
            training_data=background.to_numpy(),
            feature_names=bundle.feature_cols,
            mode="regression",
            discretize_continuous=False,
            random_state=42,
        )
        explanation = explainer.explain_instance(row, _lime_predict, num_features=min(10, len(bundle.feature_cols)))
        result = pd.DataFrame(explanation.as_list(), columns=["feature", "weight"])
    except Exception:
        # LIME can fail on some model/data combinations.
        # Return a lightweight local proxy so the dashboard still shows a chart.
        background_mean = background.mean(numeric_only=True)
        background_std = background.std(numeric_only=True).replace(0, 1).fillna(1)
        result = (
            pd.DataFrame({
                "feature": bundle.feature_cols,
                "weight": ((row - background_mean.to_numpy()) / background_std.to_numpy())
            })
            .sort_values("weight", key=lambda s: np.abs(s), ascending=False)
            .reset_index(drop=True)
        )

    st.session_state[cache_key] = result
    return result


def _r2_color(r2: float | None) -> str:
    if r2 is None: return ""
    if r2 >= 0.35: return "metric-good"
    if r2 >= 0.10: return "metric-mid"
    return "metric-bad"


def _rmse_color(rmse: float | None) -> str:
    if rmse is None: return ""
    if rmse <= 12:  return "metric-good"
    if rmse <= 18:  return "metric-mid"
    return "metric-bad"


def _fmt_metric(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.4f}"


def _model_display_name(model_name: str) -> str:
    return model_name.replace("_", " ").upper()


def _display_aqi_value(value: float | None) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return pm25_to_aqi(float(value))


def _comparison_frame(bundle: ModelBundle) -> pd.DataFrame:
    metrics = dict(bundle.all_model_metrics or {})
    if bundle.model_name not in metrics:
        metrics[bundle.model_name] = bundle.metrics

    preferred_order = ["random_forest", "ridge", "xgboost", "lstm"]
    ordered_names = [name for name in preferred_order if name in metrics]
    ordered_names.extend(name for name in metrics if name not in ordered_names)

    rows: list[dict[str, object]] = []
    for name in ordered_names:
        values = metrics.get(name, {}) or {}
        rows.append(
            {
                "model": name,
                "label": _model_display_name(name),
                "rmse": values.get("rmse"),
                "mae": values.get("mae"),
                "r2": values.get("r2"),
            }
        )

    return pd.DataFrame(rows)


def _comparison_chart(frame: pd.DataFrame, selected_model: str) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("RMSE ↓", "MAE ↓", "R² ↑"),
        horizontal_spacing=0.08,
    )

    color_map = ["#06b6d4" if model == selected_model else "#1e3a5f" for model in frame["model"]]
    metrics = [("rmse", 1), ("mae", 2), ("r2", 3)]

    for metric_name, col_idx in metrics:
        fig.add_trace(
            go.Bar(
                x=frame["label"],
                y=frame[metric_name],
                marker=dict(color=color_map),
                text=[_fmt_metric(value) for value in frame[metric_name]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>" + metric_name.upper() + ": %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=12),
        margin=dict(l=0, r=0, t=30, b=0),
        height=360,
        bargap=0.35,
    )
    for axis in ["xaxis", "xaxis2", "xaxis3"]:
        fig.layout[axis].tickfont = dict(size=10)
        fig.layout[axis].gridcolor = "rgba(0,0,0,0)"
    for axis in ["yaxis", "yaxis2", "yaxis3"]:
        fig.layout[axis].gridcolor = "#1e2d4a"
        fig.layout[axis].zeroline = False

    return fig


def _model_buttons(bundle: ModelBundle, comparison_frame: pd.DataFrame, ui_prefix: str = "comparison") -> str:
    state_key = f"{ui_prefix}_selected_model_{bundle.horizon}h"
    if state_key not in st.session_state:
        st.session_state[state_key] = bundle.model_name if bundle.model_name in set(comparison_frame["model"]) else comparison_frame.iloc[0]["model"]
    return st.session_state[state_key]


def _run_eda_points() -> list[tuple[str, str]]:
    return [
        ("Study scope", "Eleven regression models were compared across linear, nonlinear, and ensemble families."),
        ("Input features", "PM2.5, PM10, NO2, SO2, CO, and O3 were used to build AQI predictions."),
        ("Best model", "Random Forest achieved the strongest reported test performance (R² = 0.9987, RMSE = 3.25)."),
        ("Key drivers", "PM2.5 and PM10 were the dominant predictors; gaseous pollutants contributed far less."),
        ("Explainability", "SHAP and PDPs confirmed that particulate matter carries most of the AQI signal."),
    ]


def _render_metric_matrix(bundle: ModelBundle, selected_model: str) -> None:
    comparison_frame = _comparison_frame(bundle)
    if comparison_frame.empty:
        st.info("No model metrics were stored for this horizon yet.")
        return

    selected_row = comparison_frame[comparison_frame["model"] == selected_model].iloc[0]
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Selected model", _model_display_name(selected_model))
    with summary_cols[1]:
        st.metric("RMSE", _fmt_metric(selected_row["rmse"]))
    with summary_cols[2]:
        st.metric("R²", _fmt_metric(selected_row["r2"]))

    st.plotly_chart(_comparison_chart(comparison_frame, selected_model), width='stretch')

    st.dataframe(
        comparison_frame[["label", "rmse", "mae", "r2"]].rename(
            columns={"label": "model", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_model_explainability(bundle: ModelBundle, history: pd.DataFrame, selected_model: str) -> None:
    st.markdown("#### Feature Contribution Overview")

    shap_df = _shap_importance(bundle, history)
    lime_df = _lime_importance(bundle, history)

    left, right = st.columns(2)
    with left:
        st.markdown("**SHAP**")
        if shap_df.empty:
            st.info("SHAP is not available for this model type.")
        else:
            top10 = shap_df.head(10)
            fig = go.Figure(go.Bar(
                x=top10["importance"],
                y=top10["feature"],
                orientation="h",
                marker=dict(
                    color=top10["importance"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
                    showscale=False,
                ),
                text=[f"{v:.4f}" for v in top10["importance"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8", size=12),
                margin=dict(l=0, r=80, t=10, b=0),
                height=340,
                xaxis=dict(gridcolor="#1e2d4a", zeroline=False, tickfont=dict(size=11)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12, color="#e2e8f0"), autorange="reversed"),
            )
            st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("**LIME**")
        if not LIME_AVAILABLE:
            st.info("Install `lime` to enable local explanations.")
        elif lime_df.empty:
            st.info("LIME is not available for this model type or there is not enough history yet.")
        else:
            fig = go.Figure(go.Bar(
                x=lime_df["weight"],
                y=lime_df["feature"],
                orientation="h",
                marker=dict(color=np.where(lime_df["weight"] >= 0, "#22c55e", "#ef4444")),
                text=[f"{v:.4f}" for v in lime_df["weight"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>LIME: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8", size=12),
                margin=dict(l=0, r=80, t=10, b=0),
                height=340,
                xaxis=dict(gridcolor="#1e2d4a", zeroline=False, tickfont=dict(size=11)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12, color="#e2e8f0"), autorange="reversed"),
            )
            st.plotly_chart(fig, width='stretch')


def _eda_feature_overview(history: pd.DataFrame, max_features: int = 6) -> None:
    cols = [c for c in (history.columns.tolist()) if c not in ("timestamp", "aqi")]
    if not cols:
        st.info("No feature columns available for EDA.")
        return
    features = cols[:max_features]
    sample = history.tail(2000).copy()

    rows = (len(features) + 2) // 3
    fig = make_subplots(rows=rows, cols=3, subplot_titles=features)
    for i, feat in enumerate(features):
        r = i // 3 + 1
        c = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=sample[feat], nbinsx=40, marker=dict(color="#3b82f6"), showlegend=False),
            row=r, col=c,
        )
        fig.update_xaxes(title_text=feat, row=r, col=c)
    fig.update_layout(height=220 * rows, margin=dict(t=30, b=10, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')


def _eda_corr_heatmap(history: pd.DataFrame, max_features: int = 12) -> None:
    cols = [c for c in (history.columns.tolist()) if c not in ("timestamp", "aqi")]
    if not cols:
        return
    features = cols[:max_features]
    sample = history.tail(2000)[features].corr()
    fig = go.Figure(data=go.Heatmap(
        z=sample.values,
        x=sample.columns,
        y=sample.index,
        colorscale="Viridis",
        colorbar=dict(title="corr", tickfont=dict(color="#94a3b8")),
    ))
    fig.update_layout(height=480, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')


def _eda_scatter_matrix(history: pd.DataFrame, features: list[str] | None = None) -> None:
    import plotly.express as px

    cols = [c for c in (history.columns.tolist()) if c not in ("timestamp", "aqi")]
    if not cols:
        return
    if features is None:
        features = cols[:4]
    sample = history.tail(1000)[features]
    if sample.empty:
        return
    fig = px.scatter_matrix(sample, dimensions=features, color_discrete_sequence=["#06b6d4"]) 
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=600, margin=dict(t=30, b=10, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="aqi-header">
        <div>
            <div class="aqi-title">☁️ Karachi AQI Predictor</div>
            <div class="aqi-subtitle">Real-time forecasting · Open-Meteo data · Hopsworks feature store</div>
        </div>
        <div class="aqi-attribution">Designed &amp; Developed by Hamza Ali Khan</div>
    </div>
    """, unsafe_allow_html=True)

    loading_slot = st.empty()
    loading_slot.markdown(
        _loading_state_html(
            "Connecting to Hopsworks",
            12,
            "Authenticating and preparing the dashboard",
        ),
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Connecting to Hopsworks…"):
        project = _login()
        fs      = project.get_feature_store(name=_feature_store_name())
        mr      = project.get_model_registry()

    loading_slot.markdown(
        _loading_state_html(
            "Connected to Hopsworks",
            35,
            "Reading feature store history and model registry",
        ),
        unsafe_allow_html=True,
    )

    # --- Live controls: manual refresh and optional auto-refresh ---
    controls_col1, controls_col2 = st.columns([1, 3])
    with controls_col1:
        if st.button("Refresh now"):
            _load_history.clear()
            _load_model_bundle.clear()
            st.rerun()
    with controls_col2:
        auto = st.checkbox("Auto-refresh", value=False)
        interval = st.selectbox("Interval (s)", [30, 60, 120, 300, 600], index=1)

    if auto:
        if AUTREFRESH_AVAILABLE:
            st_autorefresh(interval=interval * 1000, key="aqi_autorefresh")
        else:
            st.caption("Install streamlit-autorefresh to enable timed reruns; manual refresh still works.")

    with st.spinner("Loading feature store history…"):
        # Fix 2: Cache Busting with Current Hour
        current_hour = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d-%H")
        history = _load_history(fs, current_hour)

    loading_slot.markdown(
        _loading_state_html(
            "Feature history loaded",
            62,
            "Loading the latest model bundle and forecast output",
        ),
        unsafe_allow_html=True,
    )

    if history.empty:
        loading_slot.empty()
        st.error("No real AQI rows are available yet. Run the feature pipeline first.")
        return

    # ── Load models & predict (always fetch latest model during each run) ─────
    bundles:     dict[int, ModelBundle] = {}
    predictions: dict[int, float]       = {}

    for horizon in [24, 48, 72]:
        try:
            model_version = _latest_model_version(mr, horizon)
            b = _load_model_bundle(mr, horizon, model_version)
            bundles[horizon]     = b
            predictions[horizon] = _predict(b, history)
        except Exception as exc:
            loading_slot.empty()
            st.error(f"Failed to load {horizon}h model: {exc}")
            return

    loading_slot.markdown(
        _loading_state_html(
            "Dashboard ready",
            100,
            "Rendering forecast cards, trend chart, and model comparison",
        ),
        unsafe_allow_html=True,
    )

    current_aqi   = float(history["aqi"].iloc[-1])
    display_predictions = {horizon: float(value) for horizon, value in predictions.items()}
    current_label, current_color = aqi_category(current_aqi)
    latest_ts_utc = history["timestamp"].iloc[-1]
    # Clamp displayed timestamp so the dashboard never shows a future time
    now_utc = pd.Timestamp.now(tz="UTC")
    if latest_ts_utc > now_utc:
        latest_ts_utc = now_utc
    latest_ts     = latest_ts_utc.tz_convert("Asia/Karachi").strftime("%Y-%m-%d %I:%M %p PKT")
    any_alert     = any(v > 150 for v in display_predictions.values())

    # ── Alert banner ──────────────────────────────────────────────────────────
    if any_alert:
        bad_horizons = [f"{h}h" for h, v in display_predictions.items() if v > 150]
        st.markdown(f"""
        <div class="alert-danger">
            <span style="font-size:1.4rem">⚠️</span>
            <span>Unhealthy AQI levels predicted for: {', '.join(bad_horizons)}.
            Sensitive groups should avoid outdoor activity.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-ok">
            <span style="font-size:1.2rem">✅</span>
            <span>All forecast horizons are within acceptable AQI levels.</span>
        </div>""", unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-card">
            <div class="kpi-label">Current AQI</div>
            <div class="kpi-value">{current_aqi:.1f}</div>
            <div class="kpi-sub" style="color:{current_color};font-weight:600;">{current_label}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Best forecast model</div>
            <div class="kpi-value" style="font-size:1.2rem;padding-top:0.3rem;">
                {bundles[24].model_name.upper()}
            </div>
            <div class="kpi-sub">Selected by lowest RMSE</div>
            <div class="kpi-sub">Registry version: {bundles[24].model_version}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Training rows</div>
            <div class="kpi-value">{len(history):,}</div>
            <div class="kpi-sub">Clean Open-Meteo-only labels</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Last updated</div>
            <div class="kpi-value" style="font-size:0.95rem;padding-top:0.4rem;">{latest_ts}</div>
            <div class="kpi-sub">Hourly via GitHub Actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fix 3: Scoped Tabs
    forecast_tab, comparison_tab, eda_tab = st.tabs(["Forecast", "Model Comparison", "EDA"])

    loading_slot.empty()

    with forecast_tab:
        st.markdown('<div class="section-header">72-Hour Forecast</div>', unsafe_allow_html=True)

        cards_html = '<div class="forecast-grid">'
        for horizon in [24, 48, 72]:
            pred              = display_predictions[horizon]
            label, color      = aqi_category(pred)
            conf_label, conf_pct = CONFIDENCE_LABELS[horizon]
            text_color        = "#111" if color in ("#22c55e", "#eab308") else "#fff"
            cards_html += f"""
            <div class="forecast-card">
                <div class="fc-horizon">+{horizon} hours</div>
                <div class="fc-confidence">{conf_label}</div>
                <div class="fc-aqi">{pred:.1f}</div>
                <div class="fc-badge" style="background:{color};color:{text_color};">{label}</div>
                <div class="fc-confidence-bar">
                    <div class="fc-confidence-fill" style="width:{conf_pct}%;"></div>
                </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Recent AQI Trend (72h)</div>', unsafe_allow_html=True)
        trend = history.tail(72).copy()
        trend["timestamp_local"] = trend["timestamp"].dt.tz_convert("Asia/Karachi")
        trend["display_aqi"] = trend["pm25"].apply(pm25_to_aqi) if "pm25" in trend.columns else trend["aqi"]
        fig   = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["timestamp_local"], y=trend["display_aqi"],
            mode="lines",
            line=dict(color="#3b82f6", width=2.5, shape="spline", smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
            name="AQI",
            hovertemplate="<b>%{x|%b %d %I:%M %p PKT}</b><br>AQI: %{y:.1f}<extra></extra>",
        ))
        last_ts  = trend["timestamp_local"].iloc[-1]
        for horizon, pred in predictions.items():
            display_pred = display_predictions[horizon]
            fig.add_trace(go.Scatter(
                x=[last_ts + pd.Timedelta(hours=horizon)],
                y=[display_pred],
                mode="markers+text",
                marker=dict(size=10, color="#06b6d4", symbol="diamond"),
                text=[f"+{horizon}h"],
                textposition="top center",
                textfont=dict(size=10, color="#94a3b8"),
                name=f"{horizon}h forecast",
                hovertemplate=f"<b>+{horizon}h forecast</b><br>AQI: {display_pred:.1f}<extra></extra>",
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8", size=12),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            showlegend=False,
            xaxis=dict(gridcolor="#1e2d4a", zeroline=False, tickformat="%b %d %I:%M %p", tickfont=dict(size=11)),
            yaxis=dict(
                gridcolor="#1e2d4a", zeroline=False,
                tickfont=dict(size=11),
                range=[max(0, trend["display_aqi"].min() - 5), trend["display_aqi"].max() + 8],
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

    with comparison_tab:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        st.caption("Buttons select the model to inspect. The matrix shows all trained candidates for the chosen horizon.")

        comparison_tabs = st.tabs(["24h", "48h", "72h"])
        for tab, horizon in zip(comparison_tabs, [24, 48, 72]):
            with tab:
                bundle = bundles[horizon]
                comparison_frame = _comparison_frame(bundle)
                if comparison_frame.empty:
                    st.info("No model metrics were stored for this horizon yet.")
                    continue

                # Auto-select the best registered model for this horizon (buttons removed)
                if bundle.model_name in set(comparison_frame["model"]):
                    selected_model = bundle.model_name
                else:
                    selected_model = comparison_frame.iloc[0]["model"]

                _render_metric_matrix(bundle, selected_model)
                st.caption("Showing model metrics for the best candidate (auto-selected by RMSE).")

    with eda_tab:
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        # ── Compute EDA stats from real data (converted to Local Time) ─────────
        eda_df = history.copy()
        eda_df["timestamp"] = pd.to_datetime(eda_df["timestamp"], utc=True).dt.tz_convert("Asia/Karachi")
        eda_df["hour"]  = eda_df["timestamp"].dt.hour
        eda_df["month"] = eda_df["timestamp"].dt.month
        eda_df["month_name"] = eda_df["timestamp"].dt.strftime("%b %Y")

        aqi_mean   = eda_df["aqi"].mean()
        aqi_std    = eda_df["aqi"].std()
        aqi_min    = eda_df["aqi"].min()
        aqi_max    = eda_df["aqi"].max()
        aqi_median = eda_df["aqi"].median()
        date_min   = eda_df["timestamp"].min().strftime("%b %d, %Y")
        date_max   = eda_df["timestamp"].max().strftime("%b %d, %Y")
        total_rows = len(eda_df)
        total_cols = len(eda_df.columns)
        good_pct   = (eda_df["aqi"] <= 50).mean() * 100
        moderate_pct = ((eda_df["aqi"] > 50) & (eda_df["aqi"] <= 100)).mean() * 100
        unhealthy_pct = (eda_df["aqi"] > 100).mean() * 100
        pm25_corr  = eda_df[["aqi", "pm25"]].corr().iloc[0, 1]
        peak_hour  = eda_df.groupby("hour")["aqi"].mean().idxmax()
        clean_hour = eda_df.groupby("hour")["aqi"].mean().idxmin()

        # ── Dataset Info Card ─────────────────────────────────────────────────
        dataset_overview_html = textwrap.dedent("""
        <div style="background:#111827;border:1px solid #1e2d4a;border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1.5rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;color:#3b82f6;
                        margin-bottom:1rem;">Dataset Overview</div>
        """) + textwrap.dedent(f"""
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
                <div>
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.3rem;">Total Records</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.5rem;
                                font-weight:700;color:#f0f6ff;">{total_rows:,}</div>
                    <div style="font-size:0.72rem;color:#475569;">hourly observations</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.3rem;">Date Range</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem;
                                font-weight:700;color:#f0f6ff;">{date_min}</div>
                    <div style="font-size:0.72rem;color:#475569;">to {date_max}</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.3rem;">Features</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.5rem;
                                font-weight:700;color:#f0f6ff;">{total_cols}</div>
                    <div style="font-size:0.72rem;color:#475569;">engineered columns</div>
                </div>
                <div>
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.06em;margin-bottom:0.3rem;">Data Source</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.85rem;
                                font-weight:700;color:#f0f6ff;">Open-Meteo API</div>
                    <div style="font-size:0.72rem;color:#475569;">US EPA AQI standard</div>
                </div>
            </div>
        </div>
        """)
        st.markdown("\n".join(line.lstrip() for line in dataset_overview_html.splitlines()), unsafe_allow_html=True)

        # ── Descriptive Stats Card ────────────────────────────────────────────
        descriptive_stats_html = textwrap.dedent("""
        <div style="background:#111827;border:1px solid #1e2d4a;border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1.5rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase;color:#3b82f6;
                        margin-bottom:1rem;">AQI Descriptive Statistics</div>
        """) + textwrap.dedent(f"""
            <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0.75rem;margin-bottom:1.2rem;">
                <div style="text-align:center;background:#0d1526;border-radius:10px;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">Mean</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#3b82f6;">{aqi_mean:.1f}</div>
                </div>
                <div style="text-align:center;background:#0d1526;border-radius:10px;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">Median</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#06b6d4;">{aqi_median:.1f}</div>
                </div>
                <div style="text-align:center;background:#0d1526;border-radius:10px;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">Std Dev</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#f0f6ff;">{aqi_std:.1f}</div>
                </div>
                <div style="text-align:center;background:#0d1526;border-radius:10px;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">Min</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#22c55e;">{aqi_min:.1f}</div>
                </div>
                <div style="text-align:center;background:#0d1526;border-radius:10px;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">Max</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#ef4444;">{aqi_max:.1f}</div>
                </div>
                <div style="text-align:center;background:#0d1526;border-radius:10px;padding:0.8rem 0.5rem;">
                    <div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">PM2.5 Corr</div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#a78bfa;">{pm25_corr:.3f}</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
                <div style="background:#071a0f;border:1px solid #16a34a;border-radius:8px;
                            padding:0.7rem 1rem;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.7rem;color:#86efac;font-weight:600;">🟢 Good (AQI ≤ 50)</div>
                        <div style="font-size:0.72rem;color:#475569;margin-top:0.15rem;">Air quality satisfactory</div>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#22c55e;">{good_pct:.1f}%</div>
                </div>
                <div style="background:#1a1200;border:1px solid #ca8a04;border-radius:8px;
                            padding:0.7rem 1rem;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.7rem;color:#fde68a;font-weight:600;">🟡 Moderate (51–100)</div>
                        <div style="font-size:0.72rem;color:#475569;margin-top:0.15rem;">Acceptable for most</div>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#eab308;">{moderate_pct:.1f}%</div>
                </div>
                <div style="background:#2d0a0a;border:1px solid #dc2626;border-radius:8px;
                            padding:0.7rem 1rem;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.7rem;color:#fca5a5;font-weight:600;">🔴 Unhealthy (> 100)</div>
                        <div style="font-size:0.72rem;color:#475569;margin-top:0.15rem;">Health risk present</div>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:#ef4444;">{unhealthy_pct:.1f}%</div>
                </div>
            </div>
            <div style="margin-top:1rem;font-size:0.78rem;color:#64748b;line-height:1.6;">
                Karachi's AQI averaged <b style="color:#e2e8f0;">{aqi_mean:.1f}</b> across {total_rows:,} hourly readings
                from {date_min} to {date_max}. PM2.5 shows a strong correlation of
                <b style="color:#a78bfa;">{pm25_corr:.3f}</b> with AQI, consistent with research literature identifying
                PM2.5 as the dominant AQI driver. Air quality peaks at hour
                <b style="color:#ef4444;">{peak_hour}:00</b> and is cleanest around
                <b style="color:#22c55e;">{clean_hour}:00</b> on average.
                Note: Values follow the <b style="color:#e2e8f0;">US EPA AQI standard</b>; data sourced from Open-Meteo.
            </div>
        </div>
        """)
        st.markdown("\n".join(line.lstrip() for line in descriptive_stats_html.splitlines()), unsafe_allow_html=True)

        # ── Chart 1 & 2 side by side ──────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: PM2.5 vs AQI scatter
        with chart_col1:
            st.markdown('<div class="section-header">PM2.5 vs AQI</div>', unsafe_allow_html=True)
            scatter_df = eda_df[["pm25"]].dropna().copy()
            scatter_df["aqi_calc"] = scatter_df["pm25"].apply(pm25_to_aqi)
            if scatter_df.empty:
                st.info("No PM2.5 values available for the scatter chart.")
                return
            scatter_df = scatter_df.sample(min(2000, len(scatter_df)), random_state=42)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df["pm25"],
                y=scatter_df["aqi_calc"],
                mode="markers",
                marker=dict(
                    color=scatter_df["aqi_calc"],
                    colorscale=[
                        [0.0,  "#22c55e"],
                        [0.2,  "#eab308"],
                        [0.4,  "#f97316"],
                        [0.6,  "#ef4444"],
                        [0.8,  "#a855f7"],
                        [1.0,  "#dc2626"],
                    ],
                    size=4,
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="AQI", font=dict(color="#94a3b8", size=11)),
                        tickfont=dict(color="#94a3b8", size=10),
                        thickness=12,
                    ),
                ),
                hovertemplate="PM2.5: %{x:.1f}<br>AQI: %{y:.1f}<extra></extra>",
            ))
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                xaxis=dict(title="PM2.5 (µg/m³)", gridcolor="#1e2d4a", zeroline=False),
                yaxis=dict(title="AQI", gridcolor="#1e2d4a", zeroline=False),
            )
            st.plotly_chart(fig_scatter, width='stretch')
            st.caption(f"Correlation: {pm25_corr:.3f} — PM2.5 is the strongest single predictor of AQI in Karachi.")

        # Chart 2: Hourly AQI pattern
        with chart_col2:
            st.markdown('<div class="section-header">Average AQI by Hour of Day</div>', unsafe_allow_html=True)
            hourly = eda_df.groupby("hour")["aqi"].agg(["mean", "std"]).reset_index()
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(
                x=hourly["hour"],
                y=hourly["mean"] + hourly["std"],
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
                fillcolor="rgba(59,130,246,0.1)",
            ))
            fig_hour.add_trace(go.Scatter(
                x=hourly["hour"],
                y=hourly["mean"] - hourly["std"],
                mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.1)",
                showlegend=False, hoverinfo="skip",
            ))
            fig_hour.add_trace(go.Scatter(
                x=hourly["hour"],
                y=hourly["mean"],
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5, shape="spline", smoothing=0.8),
                marker=dict(size=6, color="#06b6d4"),
                hovertemplate="Hour %{x}:00<br>Avg AQI: %{y:.1f}<extra></extra>",
                name="Mean AQI",
            ))
            fig_hour.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                showlegend=False,
                xaxis=dict(
                    title="Hour of Day (UTC)",
                    gridcolor="#1e2d4a", zeroline=False,
                    tickvals=list(range(0, 24, 3)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                ),
                yaxis=dict(title="Mean AQI", gridcolor="#1e2d4a", zeroline=False),
            )
            st.plotly_chart(fig_hour, width='stretch')
            st.caption(f"Peak pollution at {peak_hour}:00 UTC, cleanest air at {clean_hour}:00 UTC. Shaded band = ±1 std dev.")

        # ── Chart 3 & 4 side by side ──────────────────────────────────────────
        chart_col3, chart_col4 = st.columns(2)

        # Chart 3: Monthly average AQI bar chart
        with chart_col3:
            st.markdown('<div class="section-header">Monthly Average AQI</div>', unsafe_allow_html=True)
            monthly = (
                eda_df.groupby(eda_df["timestamp"].dt.to_period("M"))["aqi"]
                .mean()
                .reset_index()
            )
            monthly["period_str"] = monthly["timestamp"].astype(str)
            monthly_colors = [
                "#22c55e" if v <= 50 else
                "#eab308" if v <= 100 else
                "#f97316" if v <= 150 else
                "#ef4444"
                for v in monthly["aqi"]
            ]
            fig_monthly = go.Figure(go.Bar(
                x=monthly["period_str"],
                y=monthly["aqi"].round(1),
                marker=dict(color=monthly_colors, opacity=0.85),
                text=[f"{v:.1f}" for v in monthly["aqi"]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
                hovertemplate="<b>%{x}</b><br>Avg AQI: %{y:.1f}<extra></extra>",
            ))
            fig_monthly.add_hline(
                y=50, line=dict(color="#22c55e", dash="dot", width=1),
                annotation_text="Good", annotation_font=dict(color="#22c55e", size=10),
            )
            fig_monthly.add_hline(
                y=100, line=dict(color="#eab308", dash="dot", width=1),
                annotation_text="Moderate", annotation_font=dict(color="#eab308", size=10),
            )
            fig_monthly.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                xaxis=dict(title="Month", gridcolor="rgba(0,0,0,0)", tickangle=-30, tickfont=dict(size=10)),
                yaxis=dict(title="Mean AQI", gridcolor="#1e2d4a", zeroline=False),
                bargap=0.25,
            )
            st.plotly_chart(fig_monthly, width='stretch')
            worst_month = monthly.loc[monthly["aqi"].idxmax(), "period_str"]
            best_month  = monthly.loc[monthly["aqi"].idxmin(), "period_str"]
            st.caption(f"Worst month: {worst_month} ({monthly['aqi'].max():.1f}) · Best month: {best_month} ({monthly['aqi'].min():.1f}). Color = AQI category.")

        # Chart 4: Full AQI history line chart
        with chart_col4:
            st.markdown('<div class="section-header">AQI Over Time (Full History)</div>', unsafe_allow_html=True)
            daily_avg = (
                eda_df.set_index("timestamp")["aqi"]
                .resample("D").mean()
                .reset_index()
                .dropna()
            )
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=daily_avg["timestamp"],
                y=daily_avg["aqi"],
                mode="lines",
                line=dict(color="#3b82f6", width=1.5, shape="spline", smoothing=0.6),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.06)",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Daily avg AQI: %{y:.1f}<extra></extra>",
                name="Daily avg AQI",
            ))
            for threshold, color, label in [
                (50,  "#22c55e", "Good"),
                (100, "#eab308", "Moderate"),
                (150, "#f97316", "Unhealthy (SG)"),
            ]:
                fig_history.add_hline(
                    y=threshold,
                    line=dict(color=color, dash="dot", width=1),
                    annotation_text=label,
                    annotation_position="top right",
                    annotation_font=dict(color=color, size=9),
                )
            fig_history.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                showlegend=False,
                xaxis=dict(gridcolor="#1e2d4a", zeroline=False, tickformat="%b %Y", tickfont=dict(size=10)),
                yaxis=dict(
                    title="Daily Avg AQI",
                    gridcolor="#1e2d4a", zeroline=False,
                    range=[0, min(eda_df["aqi"].max() + 15, 200)],
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_history, width='stretch')
            st.caption(f"Daily averages smoothed for clarity. Dotted lines = US EPA AQI category thresholds.")

        # ── SHAP + LIME feature relationships ─────
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Feature Relationships (SHAP & LIME Analysis)</div>', unsafe_allow_html=True)
        st.caption("Overview of global feature importance and local interpretability for the primary forecasting model.")

        primary_bundle = bundles.get(24) or (next(iter(bundles.values())) if bundles else None)
        if primary_bundle is not None:
            selected_model = primary_bundle.model_name
            reference_name = "Ridge Regression" if selected_model == "ridge" else _model_display_name(selected_model)
            st.markdown(f"**Reference Model: {reference_name}**")
            with st.expander(f"Feature relations reference — {reference_name}", expanded=True):
                try:
                    _render_model_explainability(primary_bundle, history, selected_model)
                except Exception as exc:
                    st.info("Explainability charts are unavailable for this model bundle right now.")
                    st.caption(f"Fallback reason: {exc.__class__.__name__}")

    st.markdown("""
    <p style="font-size:0.75rem;color:#334155;text-align:center;margin-top:2rem;padding-top:1rem;
    border-top:1px solid #1e2d4a;">
        Karachi AQI Predictor · Built with Hopsworks · GitHub Actions · Streamlit · By Hamza Ali Khan
    </p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("The dashboard encountered an unexpected error while rendering.")
        st.stop()