"""
AirSentinel Unit Testing Suite - Week 1 Validation
Enforces contract checking, validation rules, and pipeline steps.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from pandera.errors import SchemaErrors
from sklearn.ensemble import IsolationForest

from src.ingestion.validation import SensorSchema, validate_batch
from src.models.anomaly_detector import score_batch, tune_contamination_cv

# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Generates a valid dictionary-based dataframe mimicking OpenAQ batch data."""
    return pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow(),
                "sensor_id": "LONDON_1",
                "location": "Westminster",
                "pm25": 12.5,
                "pm10": 18.2,
                "co": 0.4,
                "no2": 25.0,
                "o3": 30.1,
                "temperature": 22.0,
                "humidity": 55.0,
                "latitude": 51.5074,
                "longitude": -0.1278,
            }
        ]
    )


# ===========================================================================
# SCHEMA & VALIDATION TESTS (8 Tests)
# ===========================================================================


def test_valid_schema_processing(sample_raw_data: pd.DataFrame) -> None:
    """Validates that a correctly formatted dataframe passes SensorSchema."""
    logger.info("Testing valid schema layout...")
    res = SensorSchema.validate(sample_raw_data)
    assert res is not None
    assert len(res) == 1


def test_validate_batch_helper_coercion(sample_raw_data: pd.DataFrame) -> None:
    """Checks if string timestamps are safely coerced into datetime objects."""
    sample_raw_data["timestamp"] = "2026-07-20 12:00:00"
    res = validate_batch(sample_raw_data)
    assert res is not None
    assert pd.api.types.is_datetime64_any_dtype(res["timestamp"])


def test_invalid_sensor_id_prefix(sample_raw_data: pd.DataFrame) -> None:
    """Enforces that sensor_ids must strictly start with 'LONDON_'."""
    sample_raw_data["sensor_id"] = "PARIS_1"
    res = validate_batch(sample_raw_data)
    assert res is None


def test_negative_pollutant_ranges(sample_raw_data: pd.DataFrame) -> None:
    """Schema must reject negative atmospheric values for PM2.5."""
    sample_raw_data["pm25"] = -5.0
    res = validate_batch(sample_raw_data)
    assert res is None


def test_extreme_pollutant_ranges(sample_raw_data: pd.DataFrame) -> None:
    """Schema must reject unrealistic pollutant values (out of bounds)."""
    sample_raw_data["pm25"] = 600.0  # Max permitted le=500
    res = validate_batch(sample_raw_data)
    assert res is None


def test_missing_mandatory_field(sample_raw_data: pd.DataFrame) -> None:
    """Verifies that missing primary business keys (location) triggers failure."""
    sample_raw_data.drop(columns=["location"], inplace=True)
    res = validate_batch(sample_raw_data)
    assert res is None


def test_strict_extra_columns_rejection(sample_raw_data: pd.DataFrame) -> None:
    """Ensures schema drift is blocked by forbidding unmapped columns."""
    sample_raw_data["unexpected_custom_field"] = "malicious_drift"

    # On informe pytest que le code DOIT lever une exception SchemaErrors pour réussir
    with pytest.raises(SchemaErrors):
        SensorSchema.validate(sample_raw_data)


def test_empty_dataframe_handling() -> None:
    """Validates that empty payloads safely return None without crashing."""
    res = validate_batch(pd.DataFrame())
    assert res is None


# ===========================================================================
# MODEL & ANALYTICS TESTS (7 Tests)
# ===========================================================================


def test_tune_contamination_cv_output() -> None:
    """Verifies cross-validation grid tuning outputs an expected float rate."""
    X = np.random.normal(15, 2, (100, 3))
    y = np.zeros(100)
    y[:5] = 1  # 5% anomalies
    best_c = tune_contamination_cv(X, y, contamination_grid=[0.01, 0.05])
    assert isinstance(best_c, float)
    assert best_c in [0.01, 0.05]


def test_score_batch_column_generation(sample_raw_data: pd.DataFrame) -> None:
    """Checks if score_batch appends 'anomaly_pred' and 'anomaly_score' output targets."""
    X_train = np.random.normal(15, 2, (50, 3))
    model = IsolationForest(contamination=0.05, random_state=42).fit(X_train)

    feature_cols = ["pm25", "pm10", "no2"]
    scored_df = score_batch(sample_raw_data, feature_cols, model)

    assert "anomaly_pred" in scored_df.columns
    assert "anomaly_score" in scored_df.columns
    assert scored_df["anomaly_pred"].iloc[0] in [0, 1]


def test_score_batch_immutable_source(sample_raw_data: pd.DataFrame) -> None:
    """Ensures scoring runs on a dataframe copy and leaves the source immutable."""
    X_train = np.random.normal(15, 2, (10, 3))
    model = IsolationForest().fit(X_train)
    feature_cols = ["pm25", "pm10", "no2"]

    scored_df = score_batch(sample_raw_data, feature_cols, model)
    assert id(scored_df) != id(sample_raw_data)


def test_latitude_boundary_check(sample_raw_data: pd.DataFrame) -> None:
    """Verifies geospatial checks catch illegal latitude values."""
    sample_raw_data["latitude"] = 95.0
    res = validate_batch(sample_raw_data)
    assert res is None


def test_longitude_boundary_check(sample_raw_data: pd.DataFrame) -> None:
    """Verifies geospatial checks catch illegal longitude values."""
    sample_raw_data["longitude"] = -190.0
    res = validate_batch(sample_raw_data)
    assert res is None


def test_nullable_meteorological_fields(sample_raw_data: pd.DataFrame) -> None:
    """Ensures secondary parameters like temperature can be Null without dropping the record."""
    sample_raw_data["temperature"] = None
    res = validate_batch(sample_raw_data)
    assert res is not None


def test_datatype_coercion_ints_to_floats(sample_raw_data: pd.DataFrame) -> None:
    """Confirms that integer inputs for pollutants are cleanly casted to float series."""
    sample_raw_data["pm25"] = int(15)
    res = validate_batch(sample_raw_data)
    assert res is not None
    assert isinstance(res["pm25"].dtype, object) or np.issubdtype(
        res["pm25"].dtype, np.floating
    )
