"""
One-off backfill: pulls historical measurements from OpenAQ v3 for every
station in stations.json and writes them to data/historical_sensor_data.csv.

Safe to run alongside the live producer/consumer — writes to a SEPARATE
file so there's no concurrent-write risk with the streaming CSV append
happening in consumer.py.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

BASE_URL = "https://api.openaq.org/v3"
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "data",
    "historical_sensor_data.csv",
)
STATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "infrastructure",
    "stations.json",
)


def load_stations():
    with open(STATIONS_PATH, "r") as f:
        return json.load(f)["stations"]


def fetch_location_sensors(session: requests.Session, location_id: int) -> list[int]:
    """Retrieves all sensor IDs linked to an OpenAQ location."""
    url = f"{BASE_URL}/locations/{location_id}"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return []

    sensors = results[0].get("sensors", [])
    return [s["id"] for s in sensors if "id" in s]


def fetch_sensor_measurements(
    session: requests.Session, sensor_id: int, date_from: datetime, date_to: datetime
) -> list[dict]:
    """Pulls paginated historical measurements for a specific sensor ID."""
    records = []
    page = 1

    # ISO 8601 string formatting
    from_str = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_str = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
        params = {
            "datetime_from": from_str,
            "datetime_to": to_str,
            "limit": 1000,
            "page": page,
        }
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        records.extend(results)
        if len(results) < 1000:
            break
        page += 1

    return records


def fetch_station_history(
    session: requests.Session, location_id: int, days_back: int
) -> list[dict]:
    """Fetches measurements across all sensors tied to an OpenAQ location."""
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=days_back)

    sensor_ids = fetch_location_sensors(session, location_id)
    if not sensor_ids:
        logger.warning(f"No sensors found for location_id {location_id}.")
        return []

    all_records = []
    for sid in sensor_ids:
        records = fetch_sensor_measurements(session, sid, date_from, date_to)
        all_records.extend(records)

    return all_records


def flatten_records(
    records: list[dict], station_id: str, station_name: str, lat: float, lon: float
) -> pd.DataFrame:
    """Pivots raw per-parameter measurement rows into one row per timestamp."""
    rows: dict[str, dict] = {}
    for r in records:
        # Get UTC timestamp
        period = r.get("period", {})
        dt_info = period.get("datetimeFrom", {})
        ts = dt_info.get("utc") if isinstance(dt_info, dict) else None

        if not ts:
            continue

        param = r.get("parameter", {}).get("name")
        value = r.get("value")

        if ts not in rows:
            rows[ts] = {
                "location_id": station_id,
                "sensor_id": station_id,
                "location": station_name,
                "latitude": lat,
                "longitude": lon,
                "timestamp": ts,
                "pm25": None,
                "pm10": None,
                "co": None,
                "no2": None,
                "o3": None,
            }

        if param in rows[ts]:
            rows[ts][param] = value

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(list(rows.values()))


def main(days_back: int = 30):
    stations = load_stations()
    session = requests.Session()
    api_key = os.getenv("OPENAQ_API_KEY")
    if api_key:
        session.headers.update({"X-API-Key": api_key})

    all_dfs = []
    for station in stations:
        logger.info(f"Backfilling {station['id']} ({days_back}d history)...")
        try:
            records = fetch_station_history(
                session, station["openaq_location_id"], days_back
            )
            if not records:
                logger.warning(f"No historical records returned for {station['id']}.")
                continue

            df = flatten_records(
                records, station["id"], station["name"], station["lat"], station["lon"]
            )
            if df.empty:
                logger.warning(f"No valid flattened rows for {station['id']}.")
                continue

            all_dfs.append(df)
            logger.success(
                f"{station['id']}: {len(df)} historical timestamps backfilled."
            )
        except requests.RequestException as e:
            logger.error(f"Failed to backfill {station['id']}: {e}")
            continue

    if not all_dfs:
        logger.error("No data backfilled for any station.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Successfully wrote {len(combined)} total rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main(days_back=30)
