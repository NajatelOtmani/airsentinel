"""create_initial_hypertables

Revision ID: 75eda45d7acc
Revises:
Create Date: 2026-06-30 19:29:41.840983

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "75eda45d7acc"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Force the TimescaleDB extension to load first
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # 2. Create your 3 new application tables
    op.create_table(
        "sensor_readings",
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("sensor_id", sa.String(length=50), nullable=False),
        sa.Column("location", sa.String(length=100), nullable=False),
        sa.Column("pm25", sa.Float(), nullable=False),
        sa.Column("pm10", sa.Float(), nullable=False),
        sa.Column("co2", sa.Float(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("humidity", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("timestamp", "sensor_id"),
    )

    op.create_table(
        "aqi_features",
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("sensor_id", sa.String(length=50), nullable=False),
        sa.Column("pm25_rolling_avg_1h", sa.Float(), nullable=False),
        sa.Column("co2_hourly_delta", sa.Float(), nullable=False),
        sa.Column("aqi_subindex", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("timestamp", "sensor_id"),
    )

    op.create_table(
        "anomaly_events",
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("event_id", sa.UUID(), nullable=False),
        sa.Column("sensor_id", sa.String(length=50), nullable=False),
        sa.Column("metric_targeted", sa.String(length=50), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("timestamp", "event_id"),
    )

    # 3. Convert those tables into specialized time hypertables
    op.execute(
        "SELECT create_hypertable('sensor_readings', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 day');"
    )
    op.execute(
        "SELECT create_hypertable('aqi_features', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 day');"
    )


def downgrade() -> None:
    op.drop_table("anomaly_events")
    op.drop_table("aqi_features")
    op.drop_table("sensor_readings")
