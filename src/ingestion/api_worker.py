import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("air_sentinel.api_worker")


class OpenAQClient:
    BASE_URL = "https://api.openaq.org/v3"

    def __init__(self):
        api_key = os.getenv("OPENAQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAQ_API_KEY is not set. Register at explore.openaq.org/register "
                "and add the key to your .env file."
            )
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

        # Cache of location_id -> {sensor_id: parameter_name}
        self._sensor_maps: Dict[int, Dict[int, str]] = {}

    def _get_sensor_map(self, location_id: int) -> Dict[int, str]:
        """Fetches and caches the sensor_id -> parameter_name mapping for a location."""
        if location_id in self._sensor_maps:
            return self._sensor_maps[location_id]

        url = f"{self.BASE_URL}/locations/{location_id}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return {}

        sensor_map = {
            sensor["id"]: {
                "name": sensor["parameter"]["name"],
                "unit": sensor["parameter"]["units"],
            }
            for sensor in results[0].get("sensors", [])
        }
        self._sensor_maps[location_id] = sensor_map
        return sensor_map

    def fetch_latest_metrics(
        self, location_id: int, station_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Queries OpenAQ v3 for the latest sensor readings at a given location_id
        and flattens them into a standardized payload dictionary.
        """
        try:
            sensor_map = self._get_sensor_map(location_id)
            if not sensor_map:
                logger.warning(
                    f"No sensors registered for {station_name} (location_id={location_id})"
                )
                return None

            url = f"{self.BASE_URL}/locations/{location_id}/latest"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.warning(
                    f"No latest readings returned for station: {station_name}"
                )
                return None

            flattened_payload = {
                "pm25": None,
                "pm10": None,
                "co": None,
                "no2": None,
                "o3": None,
                "timestamp": None,
            }

            for reading in results:
                sensor_info = sensor_map.get(reading.get("sensorsId"))
                if not sensor_info:
                    continue
                param_name = sensor_info["name"]
                if param_name in flattened_payload:
                    value = float(reading["value"])
                    if param_name == "co" and sensor_info["unit"] == "ppm":
                        value *= 1.145  # normalize ppm -> µg/m³ at standard conditions
                    flattened_payload[param_name] = value

            if not flattened_payload["timestamp"]:
                flattened_payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

            return flattened_payload

        except requests.RequestException as e:
            logger.error(
                f"Failed to communicate with OpenAQ for {station_name}: {str(e)}"
            )
            return None
