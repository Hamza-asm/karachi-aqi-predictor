from __future__ import annotations

import json
import os
from dataclasses import dataclass

import hopsworks
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from dotenv import load_dotenv
from tensorflow import keras

from aqi_feature_utils import (
    aqi_category,
    feature_columns,
    prepare_prediction_frame,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
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
    "Good":                             "#22c55e",
    "Moderate":                         "#eab308",
    "Unhealthy for Sensitive Groups":   "#f97316",
    "Unhealthy":                        "#ef4444",
    "Very Unhealthy":                   "#a855f7",
    "Hazardous":                        "#dc2626",
}


# ── Data classes & helpers ─────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    horizon: int
    model_name: str
    model_type: str
    metrics: dict
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


@st.cache_resource(show_spinner=False)
def _load_model_bundle(_mr, horizon: int) -> ModelBundle:
    registered_model = _mr.get_best_model(f"aqi_model_{horizon}h", metric="rmse", direction="min")
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

    if model_type == "tensorflow":
        model  = keras.models.load_model(os.path.join(model_dir, "model.keras"))
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    else:
        model  = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = None

    return ModelBundle(
        horizon=horizon, model_name=metadata.get("model_name", f"aqi_model_{horizon}h"),
        model_type=model_type, metrics=metrics, model=model, model_dir=model_dir,
        scaler=scaler, lookback_window=lookback, feature_cols=feature_cols,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_history(_fs) -> pd.DataFrame:
    fg   = _fs.get_feature_group(name="aqi_features", version=1)
    data = fg.read(online=False)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    data = data[data["aqi"].notna()].copy()
    return prepare_prediction_frame(data)


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
    if bundle.model_type == "tensorflow":
        return pd.DataFrame(columns=["feature", "importance"])
    sample  = history.tail(min(200, len(history)))
    x       = sample[bundle.feature_cols]
    if hasattr(bundle.model, "coef_"):
        explainer   = shap.LinearExplainer(bundle.model, x, feature_perturbation="interventional")
    else:
        explainer   = shap.TreeExplainer(bundle.model)
    shap_vals = explainer.shap_values(x)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    importance = np.abs(np.asarray(shap_vals)).mean(axis=0)
    return (
        pd.DataFrame({"feature": bundle.feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


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


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="aqi-header">
        <div>
            <div class="aqi-title">🌫️ Karachi AQI Predictor</div>
            <div class="aqi-subtitle">Real-time forecasting · AQICN data · Hopsworks feature store</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Connecting to Hopsworks…"):
        project = _login()
        fs      = project.get_feature_store()
        mr      = project.get_model_registry()

    with st.spinner("Loading feature store history…"):
        history = _load_history(fs)

    if history.empty:
        st.error("No real AQICN rows are available yet. Run the feature pipeline first.")
        return

    # ── Load models & predict ─────────────────────────────────────────────────
    bundles:     dict[int, ModelBundle] = {}
    predictions: dict[int, float]       = {}

    for horizon in [24, 48, 72]:
        try:
            b = _load_model_bundle(mr, horizon)
            bundles[horizon]     = b
            predictions[horizon] = _predict(b, history)
        except Exception as exc:
            st.error(f"Failed to load {horizon}h model: {exc}")
            return

    current_aqi   = float(history["aqi"].iloc[-1])
    current_label, current_color = aqi_category(current_aqi)
    latest_ts     = history["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    any_alert     = any(v > 150 for v in predictions.values())

    # ── Alert banner ──────────────────────────────────────────────────────────
    if any_alert:
        bad_horizons = [f"{h}h" for h, v in predictions.items() if v > 150]
        st.markdown(f"""
        <div class="alert-danger">
            <span style="font-size:1.4rem">⚠️</span>
            <span>Hazardous AQI predicted for: {', '.join(bad_horizons)}.
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
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Training rows</div>
            <div class="kpi-value">{len(history):,}</div>
            <div class="kpi-sub">Clean AQICN-only labels</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Last updated</div>
            <div class="kpi-value" style="font-size:0.95rem;padding-top:0.4rem;">{latest_ts}</div>
            <div class="kpi-sub">Hourly via GitHub Actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Forecast cards ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">72-Hour Forecast</div>', unsafe_allow_html=True)

    cards_html = '<div class="forecast-grid">'
    for horizon in [24, 48, 72]:
        pred              = predictions[horizon]
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

    # ── AQI Trend chart ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recent AQI Trend (72h)</div>', unsafe_allow_html=True)

    trend = history.tail(72).copy()
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["timestamp"], y=trend["aqi"],
        mode="lines",
        line=dict(color="#3b82f6", width=2.5, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        name="AQI",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y:.1f}<extra></extra>",
    ))
    # Add forecast dots
    last_ts  = trend["timestamp"].iloc[-1]
    last_aqi = trend["aqi"].iloc[-1]
    for horizon, pred in predictions.items():
        fig.add_trace(go.Scatter(
            x=[last_ts + pd.Timedelta(hours=horizon)],
            y=[pred],
            mode="markers+text",
            marker=dict(size=10, color="#06b6d4", symbol="diamond"),
            text=[f"+{horizon}h"],
            textposition="top center",
            textfont=dict(size=10, color="#94a3b8"),
            name=f"{horizon}h forecast",
            hovertemplate=f"<b>+{horizon}h forecast</b><br>AQI: {pred:.1f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        showlegend=False,
        xaxis=dict(
            gridcolor="#1e2d4a", zeroline=False,
            tickformat="%b %d %H:%M", tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="#1e2d4a", zeroline=False,
            tickfont=dict(size=11),
            range=[max(0, trend["aqi"].min() - 5), trend["aqi"].max() + 8],
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance by Horizon</div>', unsafe_allow_html=True)

    rows_html = ""
    for horizon in [24, 48, 72]:
        b    = bundles[horizon]
        rmse = b.metrics.get("rmse")
        mae  = b.metrics.get("mae")
        r2   = b.metrics.get("r2")
        conf_label, conf_pct = CONFIDENCE_LABELS[horizon]
        rmse_txt = _fmt_metric(rmse)
        mae_txt  = _fmt_metric(mae)
        r2_txt   = _fmt_metric(r2)
        rows_html += f"""
        <tr>
            <td><span style="font-family:'Space Mono',monospace;color:#f0f6ff;font-weight:700;">+{horizon}h</span></td>
            <td>{b.model_name.upper()}</td>
            <td>{conf_label}</td>
            <td class="{_rmse_color(rmse)}">{rmse_txt}</td>
            <td class="{_rmse_color(mae)}">{mae_txt}</td>
            <td class="{_r2_color(r2)}">{r2_txt}</td>
        </tr>"""

    st.markdown(f"""
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Horizon</th><th>Model</th><th>Confidence</th>
                <th>RMSE ↓</th><th>MAE ↓</th><th>R² ↑</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    <p style="font-size:0.75rem;color:#475569;margin-top:0.75rem;">
        72h forecasting is inherently harder — R² degrades with horizon length even in professional systems.
        RMSE of ~10 AQI units is the practical floor without future weather forecast data.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP feature importance ───────────────────────────────────────────────
    st.markdown('<div class="section-header">24h Model — SHAP Feature Importance</div>', unsafe_allow_html=True)

    shap_df = _shap_importance(bundles[24], history)
    if shap_df.empty:
        st.info("SHAP not available for this model type.")
    else:
        top10   = shap_df.head(10)
        max_imp = top10["importance"].max()

        fig2 = go.Figure(go.Bar(
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
            textfont=dict(size=11, color="#94a3b8"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8", size=12),
            margin=dict(l=0, r=80, t=10, b=0),
            height=360,
            xaxis=dict(
                gridcolor="#1e2d4a", zeroline=False,
                range=[0, max_imp * 1.25],
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(size=12, color="#e2e8f0"),
                autorange="reversed",
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Experiment history ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Experiment History</div>', unsafe_allow_html=True)

    experiments = [
        {
            "step": "EXP 01",
            "text": "Initial model trained on 23,809 rows including synthetic Open-Meteo AQI backfill (2023–2025).",
            "outcome": "R² –0.33, RMSE 20.70 — negative R² means model worse than mean baseline.",
            "cls": "bad",
        },
        {
            "step": "EXP 02",
            "text": "Distribution check confirmed 79% mean AQI shift between backfill (18.56) and real AQICN (33.23). Label mismatch identified as root cause.",
            "outcome": "Synthetic backfill deleted. Feature group reset.",
            "cls": "info",
        },
        {
            "step": "EXP 03",
            "text": "Retrained on clean AQICN-only data (2025-03-04 onward, 4,777 rows). No synthetic labels.",
            "outcome": "R² still negative — label mismatch was not the only bottleneck.",
            "cls": "bad",
        },
        {
            "step": "EXP 04",
            "text": "Added lag features (1h, 3h, 6h, 12h, 24h), rolling statistics (mean/std over 6h and 24h), and cyclical time encoding (sin/cos for hour and day-of-week). Rebuilt feature group with consistent schema.",
            "outcome": "24h R² 0.2284, RMSE 10.03 — 50% RMSE reduction. Positive R² achieved.",
            "cls": "good",
        },
        {
            "step": "EXP 05",
            "text": "Split into 3 separate models per horizon (24h, 48h, 72h). Ridge Regression outperformed Random Forest and LSTM on all horizons with current data volume.",
            "outcome": "72h R² 0.1312 — accepted as practical ceiling without future weather data.",
            "cls": "info",
        },
    ]

    for exp in experiments:
        st.markdown(f"""
        <div class="exp-card">
            <div class="exp-step">{exp['step']}</div>
            <div>
                <div class="exp-text">{exp['text']}</div>
                <div class="exp-outcome {exp['cls']}">→ {exp['outcome']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.75rem;color:#334155;text-align:center;margin-top:2rem;padding-top:1rem;
    border-top:1px solid #1e2d4a;">
        Karachi AQI Predictor · Built with Hopsworks · GitHub Actions · Streamlit ·
    </p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()