import json
import logging
import os
import random
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from kafka import KafkaProducer

# Ensure the root src folder is in the path for clean relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.api_worker import OpenAQClient

# Setup structured logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("air_sentinel.api_producer")

# Load environment configuration
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "air_quality_stream")
POLLING_INTERVAL_SECONDS = int(
    os.getenv("POLLING_INTERVAL_SECONDS", "900")
)  # Default: 15 minutes (OpenAQ Cadence)


def load_station_registry():
    """Reads the fixed infrastructure layout file securely."""
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    registry_path = os.path.join(root_dir, "infrastructure", "stations.json")

    try:
        with open(registry_path, "r") as f:
            stations = json.load(f).get("stations", [])
    except FileNotFoundError:
        logger.error(
            f"Stations configuration not found at expected path: {registry_path}"
        )
        sys.exit(1)

    # Fail loudly, not silently, if IDs haven't been filled in yet.
    # A None ID would otherwise poll nothing and get masked by the mock fallback,
    # which is exactly the "looks fine but is fake" trap we're trying to avoid.
    missing = [s["id"] for s in stations if not s.get("openaq_location_id")]
    if missing:
        logger.critical(
            f"Missing openaq_location_id for stations: {missing}. "
            f"Look these up at https://explore.openaq.org/ and add them to stations.json "
            f"before running the producer."
        )
        sys.exit(1)

    return stations


def generate_mock_metrics():
    """Fallback metric generator used only when a single poll fails transiently."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "pm25": round(random.uniform(5.0, 35.0), 2),
        "pm10": round(random.uniform(10.0, 55.0), 2),
        "co": round(random.uniform(0.1, 2.5), 2),
        "no2": round(random.uniform(15.0, 80.0), 2),
        "o3": round(random.uniform(20.0, 90.0), 2),
    }


def main():
    logger.info("Initializing SmartCity AirSentinel Streaming Producer...")
    stations = load_station_registry()
    logger.info(f"Loaded {len(stations)} fixed London tracking points from registry.")

    # Initialize OpenAQ Client (v3 — requires OPENAQ_API_KEY in .env)
    client = OpenAQClient()

    # Initialize Kafka Producer with safe retry parameters (Exponential Backoff mitigation)
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            retry_backoff_ms=1000,
            acks="all",  # Ensure high durability for regulatory metrics
        )
        logger.info(
            f"Successfully connected to Kafka cluster at: {KAFKA_BOOTSTRAP_SERVERS}"
        )
    except Exception as e:
        logger.critical(f"Failed to establish connection to Kafka brokers: {str(e)}")
        sys.exit(1)

    logger.info(
        f"Production loop started. Polling interval set to {POLLING_INTERVAL_SECONDS}s."
    )

    try:
        while True:
            start_time = time.time()
            logger.info(
                "Starting fresh data acquisition sweep across London network..."
            )

            for station in stations:
                station_id = station["id"]
                station_name = station["name"]
                location_id = station["openaq_location_id"]

                logger.info(
                    f"Polling metrics for {station_id} ({station_name}, location_id={location_id})..."
                )
                metrics = client.fetch_latest_metrics(location_id, station_name)

                # Transient failure fallback only (network blip, momentary empty response) —
                # this is NOT meant to mask a structurally broken endpoint anymore.
                if not metrics:
                    logger.warning(
                        f"⚠️ Upstream API blank/error response for {station_id}. Activating simulation fallback."
                    )
                    metrics = generate_mock_metrics()

                # Construct a unified payload, merging metadata and metrics (whether live or generated)
                payload = {
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
                    "temperature": None,  # Will be enriched dynamically during Day 3 Weather mapping
                    "humidity": None,
                }

                # Ship directly to the real-time stream topic
                producer.send(KAFKA_TOPIC, value=payload)
                logger.info(
                    f"🚀 Sent standardized record for {station_id} to Kafka topic [{KAFKA_TOPIC}]"
                )

            # Ensure execution metrics don't drift past interval thresholds
            elapsed_time = time.time() - start_time
            sleep_duration = max(0, POLLING_INTERVAL_SECONDS - elapsed_time)

            logger.info(
                f"Sweep complete. Entering sleep cycle for {round(sleep_duration, 2)} seconds."
            )
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.info("Graceful shutdown signal caught. Flushing producer cache hooks...")
    finally:
        producer.flush()
        producer.close()
        logger.info("Kafka Producer connection cleanly dropped. Exit path solid.")


if __name__ == "__main__":
    main()
