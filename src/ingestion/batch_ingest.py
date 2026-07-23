"""
src/ingestion/batch_ingest.py
==============================
Single-sweep (batch) version of AirSentinel ingestion.

api_producer.py's `main()` polls forever and streams each reading to Kafka —
that's correct for the live system, but wrong for a batch pipeline/CLI run.
This module performs ONE polling sweep across all registered stations and
returns the result as a flat DataFrame, persisting it as a timestamped
Parquet snapshot under `data/raw/`. It deliberately reuses
`load_station_registry`, `generate_mock_metrics`, and `OpenAQClient` from
the existing producer/worker modules instead of duplicating that logic.
"""

import os
from datetime import datetime

import pandas as pd
from loguru import logger

from src.ingestion.api_producer import (generate_mock_metrics,
                                        load_station_registry)
from src.ingestion.api_worker import OpenAQClient


def run_batch_ingest(output_dir: str = "data/raw") -> pd.DataFrame:
    """Perform a single polling sweep across all registered stations.

    Args:
        output_dir: Directory to write the timestamped raw snapshot to.

    Returns:
        DataFrame of raw sensor readings, one row per station, matching the
        payload shape produced by the streaming producer.

    Raises:
        SystemExit: propagated from load_station_registry() if
            infrastructure/stations.json is missing or incomplete.
    """
    logger.info("Loading station registry...")
    stations = load_station_registry()
    logger.info(f"Loaded {len(stations)} stations for batch ingestion.")

    client = OpenAQClient()
    records = []

    for station in stations:
        station_id = station["id"]
        station_name = station["name"]
        location_id = station["openaq_location_id"]

        logger.info(f"Fetching latest metrics for {station_id} ({station_name})...")
        metrics = client.fetch_latest_metrics(location_id, station_name)

        if not metrics:
            logger.warning(f"No live data for {station_id}; using simulated fallback.")
            metrics = generate_mock_metrics()

        records.append(
            {
                "sensor_id": station_id,
                "location": station_name,
                "latitude": station["lat"],
                "longitude": station["lon"],
                "timestamp": metrics["timestamp"],
                "pm25": metrics["pm25"],
                "pm10": metrics["pm10"],
                "co": metrics["co"],
                "no2": metrics["no2"],
                "o3": metrics["o3"],
                "temperature": None,
                "humidity": None,
            }
        )

    df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)
    ts_label = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(output_dir, f"ingest_{ts_label}.parquet")
    df.to_parquet(out_path, index=False)
    logger.success(f"Batch ingest complete: {len(df)} rows written to {out_path}")

    return df
