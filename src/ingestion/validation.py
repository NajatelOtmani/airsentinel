"""
src/ingestion/validation.py
=============================
Pandera schema contract for AirSentinel sensor readings, plus a shared
`validate_batch()` helper. Previously this validation logic lived inline
inside consumer.py's `process_batch()`; it's extracted here so the batch
pipeline (src/pipeline.py) can call the exact same contract the streaming
consumer uses, instead of re-implementing it.
"""

from typing import Optional

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import Series


class SensorSchema(pa.DataFrameModel):
    """Data contract for a single validated sensor reading."""

    # Core operational fields MUST exist and cannot be null
    timestamp: Series[pd.Timestamp] = pa.Field(nullable=False)
    sensor_id: Series[str] = pa.Field(nullable=False, str_startswith="LONDON_")
    location: Series[str] = pa.Field(nullable=False)

    # Real-world pollutants can be null if a station lacks that parameter,
    # but if present must respect realistic atmospheric ranges.
    pm25: Series[float] = pa.Field(ge=0, le=500, nullable=True)
    pm10: Series[float] = pa.Field(ge=0, le=600, nullable=True)
    co: Series[float] = pa.Field(ge=0, le=50, nullable=True)
    no2: Series[float] = pa.Field(ge=0, le=1000, nullable=True)
    o3: Series[float] = pa.Field(ge=0, le=1000, nullable=True)

    # Meteorological baselines
    temperature: Series[float] = pa.Field(ge=-20, le=60, nullable=True)
    humidity: Series[float] = pa.Field(ge=0, le=100, nullable=True)

    # Geospatial data tracking coordinates
    latitude: Series[float] = pa.Field(ge=-90, le=90, nullable=True)
    longitude: Series[float] = pa.Field(ge=-180, le=180, nullable=True)

    class Config:
        strict = True  # Disallow extra unmapped columns to block schema drift
        coerce = True  # Automatically cast safe data types (e.g., int -> float)


def validate_batch(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Apply the SensorSchema contract to a raw batch of sensor readings.

    This is the same validation step consumer.py runs inline per Kafka
    micro-batch, factored out so the batch CLI pipeline can reuse it.

    Args:
        df: Raw, unvalidated DataFrame of sensor readings. The `timestamp`
            column may be str or datetime — it will be coerced if needed.

    Returns:
        The validated DataFrame if the batch passes the schema contract,
        otherwise None. Callers are responsible for dead-lettering or
        aborting downstream steps when None is returned.
    """
    if df is None or df.empty:
        logger.warning("validate_batch called with an empty or missing DataFrame.")
        return None

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    try:
        validated_df = SensorSchema.validate(df)
        logger.info(
            f"Validated batch of {len(validated_df)} records against SensorSchema."
        )
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation failed: {e}")
        return None
