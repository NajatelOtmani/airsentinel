import pandas as pd
import pydeck as pdk
import streamlit as st
from api_client import ensure_logged_in, get

st.set_page_config(page_title="AirSentinel — City AQI Overview", layout="wide")

if not ensure_logged_in():
    st.stop()

st.title("🌍 City AQI Overview")

# --- Fetch sensor list ---
try:
    sensors_resp = get("/api/v1/sensors/")
    sensor_ids = sensors_resp.get("sensors", [])
except Exception as e:
    st.error(f"Failed to load sensors: {e}")
    st.stop()

if not sensor_ids:
    st.warning("No sensors found.")
    st.stop()

# --- Fetch latest reading per sensor for the map ---
rows = []
for sid in sensor_ids:
    try:
        latest = get(f"/api/v1/sensors/{sid}/latest")
        if "error" not in latest:
            rows.append(latest)
    except Exception:
        continue

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No sensor readings available yet.")
    st.stop()

# --- Zone map ---
st.subheader("Zone Map")


def aqi_color(aqi):
    if pd.isna(aqi):
        return [128, 128, 128, 160]
    if aqi <= 50:
        return [0, 200, 0, 180]
    elif aqi <= 100:
        return [255, 220, 0, 180]
    elif aqi <= 150:
        return [255, 140, 0, 180]
    else:
        return [220, 0, 0, 180]


df["color"] = df["epa_aqi"].apply(aqi_color)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["longitude", "latitude"],
    get_fill_color="color",
    get_radius=400,
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=df["latitude"].mean(),
    longitude=df["longitude"].mean(),
    zoom=10,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{location_id}\nAQI: {epa_aqi}\nPM2.5: {pm25}"},
    )
)

# --- Live gauges ---
st.subheader("Live Gauges")
cols = st.columns(min(len(df), 4))
for i, (_, row) in enumerate(df.iterrows()):
    with cols[i % len(cols)]:
        st.metric(
            label=row["location_id"],
            value=f"{row.get('epa_aqi', 'N/A')} AQI",
            delta=f"PM2.5: {row.get('pm25', 'N/A')} µg/m³",
        )

# --- 12-step forecast ribbon ---
st.subheader("12h Forecast")
selected = st.selectbox("Select sensor for forecast", sensor_ids)
if selected:
    try:
        forecast = get(f"/api/v1/forecasts/{selected}")
        st.json(forecast)
    except Exception as e:
        st.error(f"Forecast unavailable: {e}")
