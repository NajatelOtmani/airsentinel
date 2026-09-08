import pandas as pd
import streamlit as st
from api_client import ensure_logged_in, get

st.set_page_config(page_title="AirSentinel — Anomaly Explorer", layout="wide")

if not ensure_logged_in():
    st.stop()

st.title("🔍 Anomaly Explorer")

# --- Sensor selector ---
try:
    sensors_resp = get("/api/v1/sensors/")
    sensor_ids = sensors_resp.get("sensors", [])
except Exception as e:
    st.error(f"Failed to load sensors: {e}")
    st.stop()

if not sensor_ids:
    st.warning("No sensors found.")
    st.stop()

selected = st.selectbox("Select sensor", sensor_ids)

# --- Fetch anomalies for selected sensor ---
try:
    resp = get(f"/api/v1/anomalies/{selected}", params={"limit": 100})
    anomalies = resp.get("anomalies", [])
except Exception as e:
    st.error(f"Failed to load anomalies: {e}")
    st.stop()

st.subheader(f"Anomalies for {selected}")

if not anomalies:
    st.info("No anomalies detected for this sensor yet.")
else:
    df = pd.DataFrame(anomalies)

    # --- Filterable table ---
    col1, col2 = st.columns(2)
    with col1:
        min_pm25 = st.slider(
            "Min PM2.5", 0.0, float(df["pm25"].max()) if "pm25" in df else 100.0, 0.0
        )
    with col2:
        aqi_categories = (
            df["aqi_category"].dropna().unique().tolist()
            if "aqi_category" in df
            else []
        )
        selected_categories = st.multiselect(
            "AQI Category", aqi_categories, default=aqi_categories
        )

    filtered = df.copy()
    if "pm25" in filtered:
        filtered = filtered[filtered["pm25"] >= min_pm25]
    if selected_categories and "aqi_category" in filtered:
        filtered = filtered[filtered["aqi_category"].isin(selected_categories)]

    st.dataframe(filtered, use_container_width=True)

    # --- Reconstruction error chart (if column exists) ---
    st.subheader("Reconstruction Error Over Time")
    if "label_residual" in filtered.columns and "timestamp" in filtered.columns:
        chart_df = filtered[["timestamp", "label_residual"]].sort_values("timestamp")
        st.line_chart(chart_df.set_index("timestamp"))
    else:
        st.info("No reconstruction error data available for this view.")

# --- SHAP waterfall (from your Day 7-10 explainability assets) ---
st.subheader("SHAP Feature Importance")
shap_path = "reports/figures/shap_waterfall.png"
try:
    st.image(shap_path, caption="SHAP Feature Importance (from latest model run)")
except Exception:
    st.info(
        "SHAP waterfall image not found. Run the explainability pipeline (run_pipeline.py) to generate it."
    )
