import pandas as pd
import pydeck as pdk
import streamlit as st

from api_client import ensure_logged_in, get
from styles import aqi_badge, inject_css, page_header

# Set page configuration
st.set_page_config(page_title="AirSentinel — Overview", layout="wide")

# Inject custom styling
inject_css()

# Authentication check
if not ensure_logged_in():
    st.stop()

# Sticky Top Page Header
page_header("Live data · Network", "Air Quality Overview", "Last 24 hours across your monitoring network")

# Fetch active sensor IDs
try:
    sensor_ids = get("/api/v1/sensors/").get("sensors", [])
except Exception as e:
    st.error(f"Failed to load sensors: {e}")
    st.stop()

# Fetch latest reading per sensor
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

# Compute summary metrics safely
avg_pm25 = df["pm25"].mean() if "pm25" in df else 0.0
avg_aqi = int(df["epa_aqi"].mean()) if "epa_aqi" in df else 0
worst = df.loc[df["epa_aqi"].idxmax()] if "epa_aqi" in df else df.iloc[0]

# Render KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Network Avg PM2.5</div>
        <div class="kpi-value">{avg_pm25:.1f} µg/m³</div>
        <div class="kpi-sub">{aqi_badge(avg_aqi)}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Sensors Online</div>
        <div class="kpi-value">{len(df)}</div>
        <div class="kpi-sub">Across active zones</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    worst_loc = worst.get("location_id", "Unknown")
    worst_aqi = int(worst.get("epa_aqi", 0))
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Worst Zone</div>
        <div class="kpi-value">{worst_loc}</div>
        <div class="kpi-sub">{aqi_badge(worst_aqi)}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    total_anomalies = df["is_anomaly"].sum() if "is_anomaly" in df else 0
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-label">Active Anomalies</div>
        <div class="kpi-value">{int(total_anomalies)}</div>
        <div class="kpi-sub">Right now</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Zone Map")


def aqi_color(aqi):
    if pd.isna(aqi):
        return [128, 128, 128, 160]
    if aqi <= 50:
        return [168, 240, 59, 200]
    elif aqi <= 100:
        return [240, 201, 59, 200]
    else:
        return [240, 96, 59, 200]


df["color"] = df["epa_aqi"].apply(aqi_color) if "epa_aqi" in df else [[128, 128, 128, 160]] * len(df)

# Interactive PyDeck Map
st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius=400,
                pickable=True,
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=df["latitude"].mean() if "latitude" in df else 0.0,
            longitude=df["longitude"].mean() if "longitude" in df else 0.0,
            zoom=10,
        ),
        tooltip={"text": "{location_id}\nAQI: {epa_aqi}\nPM2.5: {pm25}"},
        map_style=None,
    )
)

# Detailed Pollutant Breakdown Table
st.markdown("#### Pollutant Breakdown by Zone")
display_cols = [c for c in ["location_id", "pm25", "pm10", "no2", "o3", "epa_aqi", "aqi_category"] if c in df.columns]

st.dataframe(
    df[display_cols],
    use_container_width=True,
    hide_index=True,
)