Here are the exact instructions for Copilot — give these verbatim:

TASK: Migrate streamlit_app.py from Hopsworks to BigQuery + GCS. DO NOT change any CSS, styling, HTML, charts, or UI logic.
Step 1 — Replace imports at the top:
Remove:
pythonimport hopsworks
Add:
pythonfrom google.cloud import bigquery
from google.cloud import storage
from google.oauth2 import service_account

Step 2 — Add GCP credential setup right after imports (before st.set_page_config):
python# Setup GCP credentials from environment
load_dotenv()
gcp_key = os.getenv("GCP_KEY_JSON") or (st.secrets.get("GCP_KEY_JSON") if hasattr(st, "secrets") else None)
if gcp_key:
    if not os.path.exists("gcp-key.json"):
        with open("gcp-key.json", "w") as f:
            f.write(gcp_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "aqi-predictor-497110")
GCS_BUCKET = "aqi-predictor-497110-features"
BQ_TABLE = "aqi-predictor-497110.aqi_features.features"

Step 3 — Replace _login() function entirely:
Remove:
python@st.cache_resource(show_spinner=False)
def _login() -> hopsworks.project.Project:
    ...
Replace with:
python@st.cache_resource(show_spinner=False)
def _get_gcp_clients():
    bq_client = bigquery.Client(project=GCP_PROJECT)
    gcs_client = storage.Client(project=GCP_PROJECT)
    return bq_client, gcs_client

Step 4 — Replace _feature_store_name() function:
Remove entire function. Not needed anymore.

Step 5 — Replace _latest_model_version() function:
Remove entire function.
Replace with:
pythondef _latest_model_version(gcs_client, horizon: int) -> str:
    bucket = gcs_client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"models/aqi_model_{horizon}h/latest/metadata.json")
    return blob.updated.isoformat() if blob.exists() else "latest"

Step 6 — Replace _load_model_bundle() function:
Remove entire existing function.
Replace with:
python@st.cache_resource(show_spinner=False)
def _load_model_bundle(_gcs_client, horizon: int, cache_bust: str) -> ModelBundle:
    bucket = _gcs_client.bucket(GCS_BUCKET)
    prefix = f"models/aqi_model_{horizon}h/latest"

    with tempfile.TemporaryDirectory() as tmp:
        for filename in ["metadata.json", "model.pkl", "model.keras", "scaler.pkl"]:
            blob = bucket.blob(f"{prefix}/{filename}")
            if blob.exists():
                blob.download_to_filename(os.path.join(tmp, filename))

        metadata_path = os.path.join(tmp, "metadata.json")
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"metadata.json missing for aqi_model_{horizon}h")

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        model_type   = metadata.get("model_type", "sklearn")
        feature_cols = metadata.get("features", feature_columns())
        lookback     = int(metadata.get("lookback_window", 24))
        metrics      = metadata.get("metrics", {})
        all_metrics  = metadata.get("all_model_metrics", {})

        if model_type == "tensorflow":
            from tensorflow import keras
            model  = keras.models.load_model(os.path.join(tmp, "model.keras"))
            scaler_path = os.path.join(tmp, "scaler.pkl")
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        else:
            model  = joblib.load(os.path.join(tmp, "model.pkl"))
            scaler = None

        return ModelBundle(
            horizon=horizon,
            model_name=metadata.get("model_name", f"aqi_model_{horizon}h"),
            model_version=cache_bust,
            model_type=model_type,
            metrics=metrics,
            model=model,
            model_dir=tmp,
            all_model_metrics=all_metrics,
            scaler=scaler,
            lookback_window=lookback,
            feature_cols=feature_cols,
        )

Step 7 — Replace _load_history() function:
Remove entire existing function.
Replace with:
python@st.cache_data(ttl=3600, show_spinner=False)
def _load_history(_bq_client, _cache_bust: str) -> pd.DataFrame:
    query = f"""
        SELECT * FROM `{BQ_TABLE}`
        ORDER BY timestamp DESC
        LIMIT 500
    """
    df = _bq_client.query(query).to_dataframe()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[df['aqi'].notna()].copy()
    return prepare_prediction_frame(df)

Step 8 — Replace the main() data loading block:
Find this section in main():
pythonwith st.spinner("Connecting to Hopsworks…"):
    project = _login()
    fs      = project.get_feature_store(name=_feature_store_name())
    mr      = project.get_model_registry()
Replace with:
pythonwith st.spinner("Connecting to GCP…"):
    bq_client, gcs_client = _get_gcp_clients()

Step 9 — Replace history loading in main():
Find:
pythonhistory = _load_history(fs, current_hour)
Replace with:
pythonhistory = _load_history(bq_client, current_hour)

Step 10 — Replace model loading loop in main():
Find:
pythonfor horizon in [24, 48, 72]:
    try:
        model_version = _latest_model_version(mr, horizon)
        b = _load_model_bundle(mr, horizon, model_version)
Replace with:
pythonfor horizon in [24, 48, 72]:
    try:
        model_version = _latest_model_version(gcs_client, horizon)
        b = _load_model_bundle(gcs_client, horizon, model_version)

Step 11 — Update loading state text in main():
Find:
python"Connecting to Hopsworks",
Replace with:
python"Connecting to GCP",
Find:
python"Connected to Hopsworks",
Replace with:
python"Connected to GCP",
Find:
pythonst.markdown('<div class="aqi-subtitle">Real-time forecasting · Open-Meteo data · Hopsworks feature store</div>',
Replace with:
pythonst.markdown('<div class="aqi-subtitle">Real-time forecasting · Open-Meteo data · BigQuery feature store</div>',
Find at bottom of file:
pythonKarachi AQI Predictor · Built with Hopsworks · GitHub Actions · Streamlit · By Hamza Ali Khan
Replace with:
pythonKarachi AQI Predictor · Built with BigQuery · GCS · GitHub Actions · Streamlit · By Hamza Ali Khan

Step 12 — Add missing import at top:
pythonimport tempfile

DO NOT change anything else. All CSS, charts, HTML, EDA logic, SHAP/LIME, forecast cards stay exactly the same.