import json
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer

# Ensure the root src folder is in the path for clean imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean import for Day 3 Feature Engine
from features.engineering import AirQualityFeatureEngineer
from ingestion.validation import SensorSchema

# Project relative path to data/processed_sensor_data.csv
PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "processed_sensor_data.csv",
)
# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("air_sentinel.consumer")

load_dotenv()
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "air_quality_stream")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "validation_processor")


def process_batch(raw_messages):
    """
    Transforms raw Kafka packets into a pandas DataFrame and applies
    the Pandera data validation schema contract.
    """
    parsed_records = []

    for msg in raw_messages:
        try:
            payload = json.loads(msg.value.decode("utf-8"))
            parsed_records.append(payload)
        except Exception as e:
            logger.error(f"Malformed JSON dropped from stream: {str(e)}")
            continue

    if not parsed_records:
        return None

    # Convert batch to DataFrame for vectorized validation
    # Convert batch to DataFrame for vectorized validation
    df = pd.DataFrame(parsed_records)

    # Crucial step: cast string timestamps to actual pandas datetime objects
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    # Impute missing pm25 within this micro-batch before schema validation
    if "pm25" in df.columns and df["pm25"].isna().any():
        n_missing = df["pm25"].isna().sum()
        logger.warning(
            f"⚠️ {n_missing} record(s) missing pm25 — imputing via batch median."
        )
        df["pm25"] = df["pm25"].fillna(df["pm25"].median())
        df["pm25"] = df["pm25"].fillna(0.0)  # fallback if entire batch was NaN

    try:
        # Validate data against the Pandera contract model
        validated_df = SensorSchema.validate(df)
        logger.info(
            f"✔ Successfully validated batch of {len(validated_df)} telemetry entries."
        )
        return validated_df

    except Exception as e:
        # Real-world defense: isolate corrupted records instead of letting them crash the container
        logger.critical(
            f"❌ Schema validation failed for batch! Sending to dead-letter pipeline. Error: {str(e)}"
        )
        return None


def persist_features(enriched_df: pd.DataFrame) -> None:
    """Appends enriched feature rows to the rolling training CSV, writing a header only once."""
    if enriched_df.empty:
        logger.warning("Attempted to persist an empty DataFrame. Skipping.")
        return

    # Ensure the 'data/' folder exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    file_exists = os.path.isfile(PROCESSED_DATA_PATH)

    # Append to CSV (write header only if file is brand new)
    enriched_df.to_csv(
        PROCESSED_DATA_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )
    logger.info(f"💾 Appended {len(enriched_df)} rows to {PROCESSED_DATA_PATH}")


def main():
    logger.info("Initializing AirSentinel Validation Consumer...")

    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=CONSUMER_GROUP,
            auto_offset_reset="earliest",  # Replays missed streams if the consumer drops offline
            enable_auto_commit=True,
            value_deserializer=lambda x: x,  # Leave decoding to process_batch to catch exceptions safely
        )
        logger.info(
            f"Listening to Kafka topic [{KAFKA_TOPIC}] as group '{CONSUMER_GROUP}'..."
        )
    except Exception as e:
        logger.critical(f"Failed to bind Kafka consumer runtime: {str(e)}")
        sys.exit(1)

    # Initialize our Day 3 engineering state engine
    engineer = AirQualityFeatureEngineer(
        sensor_locations_path=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "infrastructure",
            "stations.json",
        )
    )

    batch_buffer = []
    max_batch_size = 5  # Matches our London topology footprint per check-in window

    try:
        for message in consumer:
            batch_buffer.append(message)

            # Process records in window micro-batches
            if len(batch_buffer) >= max_batch_size:
                logger.info(
                    f"Buffer threshold reached ({len(batch_buffer)}). Executing validation sweep..."
                )
                validated_data = process_batch(batch_buffer)

                if validated_data is not None:
                    # Apply Feature Engineering safely on the validated DataFrame
                    try:
                        enriched_data = engineer.extract_features(validated_data)
                        logger.info("✨ Day 3 Features Engineered Successfully!")

                        # Let's inspect the columns to verify they're active
                        logger.info(
                            f"Columns in Enriched DF: {list(enriched_data.columns)}"
                        )
                        persist_features(enriched_data)

                        # Next Step: db_writer.bulk_insert(enriched_data) will go here!

                    except Exception as feat_err:
                        logger.error(
                            f"⚠️ Feature engineering enrichment failed: {str(feat_err)}"
                        )

                # Clear buffer safely for next polling window
                batch_buffer.clear()

    except KeyboardInterrupt:
        logger.info(
            "Termination intercept received. Shutting down consumer hooks cleanly..."
        )
    finally:
        consumer.close()
        logger.info("Kafka consumer loop dropped safely. Pipeline at rest.")


if __name__ == "__main__":
    main()
