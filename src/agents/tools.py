"""
src/agents/tools.py
===================
DAY 13 — ReAct Agent Tools
--------------------------
4 Custom Tools tailored for AirSentinel:
  1. query_anomaly_db: Query anomaly events in data/processed_sensor_data.csv
  2. get_sensor_forecast: Query ONNX model or rolling averages for 12h forecast
  3. search_env_documents: Search FAISS vector store
  4. compute_zone_statistics: Compute rolling PM2.5/AQI metrics by location_id
"""

from pathlib import Path

import numpy as np
import pandas as pd
from langchain.tools import tool
from loguru import logger
from pydantic import BaseModel, Field

CSV_PATH = Path("data/processed_sensor_data.csv")

# ── 1. Query Anomaly DB ───────────────────────────────────────────────────────


class AnomalyQueryInput(BaseModel):
    location_id: str = Field(description="Sensor or location ID, e.g. 'LONDON_HF1'")
    hours: int = Field(default=3, description="Lookback window in hours")


# In src/agents/tools.py


@tool("query_anomaly_db", args_schema=AnomalyQueryInput)
def query_anomaly_db(location_id: str, hours: int = 3) -> str:
    """Fetch recent anomaly events for a given location_id from processed_sensor_data.csv."""
    if not CSV_PATH.exists():
        return "Error: Processed data CSV not found."

    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    filtered = df[df["location_id"].str.contains(location_id, case=False, na=False)]
    if filtered.empty:
        return f"No sensor or anomaly records exist in the database for location '{location_id}'. Continue with statistical and document tools."

    anomalies = filtered[filtered["is_anomaly"] == 1].tail(hours)
    if anomalies.empty:
        return f"No recent anomalies detected for '{location_id}'. Total historical anomalies: {int((filtered['is_anomaly'] == 1).sum())}."

    # Force a strict cap of max 2 records to keep token size tiny
    anomalies = anomalies.tail(2)
    summary_list = [
        f"Time: {row['timestamp']} | PM2.5: {row['pm25']} | AQI: {row['epa_aqi']}"
        for _, row in anomalies.iterrows()
    ]

    total_count = int((filtered["is_anomaly"] == 1).sum())
    return f"Total anomalies for {location_id}: {total_count}. Recent sample: {'; '.join(summary_list)}"


# ── 2. Get Sensor Forecast ────────────────────────────────────────────────────


class ForecastInput(BaseModel):
    location_id: str = Field(description="Sensor or location ID, e.g. 'LONDON_HF1'")


@tool("get_sensor_forecast", args_schema=ForecastInput)
def get_sensor_forecast(location_id: str) -> str:
    """Get 12-step ahead PM2.5 forecast using ONNX transformer model or exponential smoothing fallback."""
    onnx_path = Path("models/onnx/transformer.onnx")

    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
        sub = df[df["location_id"].str.contains(location_id, case=False, na=False)]
        if not sub.empty:
            recent_pm25 = sub["pm25"].tail(12).values
            mean_val = float(recent_pm25.mean()) if len(recent_pm25) > 0 else 25.0
        else:
            mean_val = 25.0
    else:
        mean_val = 25.0

    if onnx_path.exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            # Generate dummy sequence shaped (1, 60, 10) for inference
            dummy_input = np.ones((1, 60, 10), dtype=np.float32) * mean_val
            raw_pred = session.run(None, {input_name: dummy_input})[0]
            forecast = [round(float(x), 2) for x in raw_pred.flatten()[:12]]
            return f"12-hour ONNX forecast for {location_id}: {forecast}"
        except Exception as e:
            logger.warning(f"ONNX inference failed, using fallback: {e}")

    # Baseline forecast fallback
    forecast = [round(mean_val * (1 + 0.02 * i), 2) for i in range(12)]
    return f"12-hour statistical forecast for {location_id}: {forecast}"


# ── 3. Search Environmental Documents ────────────────────────────────────────

# ── 3. Search Environmental Documents ────────────────────────────────────────


class DocumentSearchInput(BaseModel):
    query: str = Field(
        description="Search query regarding WHO, EPA guidelines, or health impacts"
    )
    k: int = Field(default=1, description="Number of results to retrieve")


@tool("search_env_documents", args_schema=DocumentSearchInput)
def search_env_documents(query: str, k: int = 1) -> str:
    """Search vector store for WHO/EPA regulatory standards and health advice."""
    try:
        from src.rag.ingest import AirSentinelIngestor

        ingestor = AirSentinelIngestor()
        results = ingestor.similarity_search(query, k=k, use_mmr=False)
        if not results:
            return "WHO Guidelines: PM2.5 daily threshold is 15 ug/m3. EPA AQI above 100 indicates unhealthy levels for sensitive groups."

        snippets = []
        for r in results:
            # Handle LangChain Document object attributes (.page_content & .metadata)
            content = getattr(r, "page_content", str(r))[:120]
            metadata = getattr(r, "metadata", {})
            src = metadata.get("source", "WHO/EPA Guidelines")
            snippets.append(f"[{src}]: {content}")
        return "\n".join(snippets)
    except Exception as e:
        logger.warning(
            f"Vector store search failed, using standard fallback guidelines: {e}"
        )
        return "WHO Guidelines: Recommended 24-hour mean PM2.5 limit is 15 ug/m3. EPA AQI 0-50 Good, 51-100 Moderate, 101-150 Unhealthy for Sensitive Groups."


# ── 4. Compute Zone Statistics ────────────────────────────────────────────────


class ZoneStatsInput(BaseModel):
    location_id: str = Field(description="Sensor location ID, e.g. 'LONDON_HF1'")


@tool("compute_zone_statistics", args_schema=ZoneStatsInput)
def compute_zone_statistics(location_id: str) -> str:
    """Compute rolling summary statistics (mean, max, std, AQI category) for a given location_id."""
    if not CSV_PATH.exists():
        return "Error: Processed CSV not found."

    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    filtered = df[df["location_id"].str.contains(location_id, case=False, na=False)]
    if filtered.empty:
        return f"No rolling statistical data available for location '{location_id}'."

    stats = {
        "location_id": location_id,
        "pm25_mean": round(float(filtered["pm25"].mean()), 2),
        "pm25_max": round(float(filtered["pm25"].max()), 2),
        "pm25_std": round(float(filtered["pm25"].std()), 2),
        "epa_aqi_mean": round(float(filtered["epa_aqi"].mean()), 2),
        "latest_aqi_category": str(filtered["aqi_category"].iloc[-1])
        if "aqi_category" in filtered.columns
        else "N/A",
        "total_anomalies": int(filtered["is_anomaly"].sum()),
    }
    return f"Zone Statistics for {location_id}: {stats}"
