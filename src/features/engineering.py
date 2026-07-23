"""
=============================================================================
src/data/features.py
AirSentinel — Day 3: Complete Feature Engineering Pipeline
=============================================================================

WHAT THIS FILE DOES (big picture):
    Raw sensor readings (timestamp, location_id, pollutant, value) come in
    from Kafka/TimescaleDB.  This module transforms them into a rich feature
    matrix that the ML models (Week 2) will consume.

PIPELINE STAGES BUILT HERE:
    1. Rolling statistics     — 1h / 6h / 24h mean, std, min, max
    2. EPA AQI computation    — PM2.5, PM10, CO, NO2, O3 official formula
    3. STL decomposition      — trend / seasonality / residual per sensor
    4. Cyclical time encoding — hour-of-day, day-of-week as sin/cos pairs
    5. Spatial correlation    — nearest-3-sensor mean per pollutant
    6. Spike labelling        — ground-truth anomaly flags for model training

WHY EACH STEP EXISTS:
    • Rolling stats give the model memory of recent history without storing
      a full sequence (that's the LSTM's job in Week 2).
    • AQI converts raw µg/m³ into a standard 0-500 index; thresholds are
      legally defined by the EPA and map directly to health categories.
    • STL isolates the residual component — the part that is NOT routine
      trend or daily seasonality.  Anomalies live in the residual.
    • Sin/cos encoding wraps circular time so hour 23 is numerically close
      to hour 0 (standard label encoding breaks this relationship).
    • Spatial features: if sensor A spikes but its 3 neighbours are normal,
      that's a localised emission event — a key anomaly signal.
    • Spike labels give us supervised ground truth to evaluate Week 1/2 models.

=============================================================================
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from statsmodels.tsa.seasonal import STL

# ---------------------------------------------------------------------------
# Logger — structured output, visible in Grafana Loki or your terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("airsentinel.features")


# ===========================================================================
# SECTION 1 — EPA AQI BREAKPOINTS
# ===========================================================================
#
# The EPA defines AQI using piecewise-linear interpolation between official
# concentration breakpoints.  Each row is:
#   (C_low, C_high, I_low, I_high)
# where C is the pollutant concentration and I is the AQI index.
#
# Source: EPA Technical Assistance Document for the AQI (May 2016)
# https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
# ===========================================================================

# PM2.5 breakpoints — 24-hour average, units: µg/m³
PM25_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0.0, 12.0, 0, 50),  # Good
    (12.1, 35.4, 51, 100),  # Moderate
    (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
    (55.5, 150.4, 151, 200),  # Unhealthy
    (150.5, 250.4, 201, 300),  # Very Unhealthy
    (250.5, 500.4, 301, 500),  # Hazardous
]

# PM10 breakpoints — 24-hour average, units: µg/m³
PM10_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

# CO breakpoints — 8-hour average, units: ppm
CO_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 50.4, 301, 500),
]

# NO2 breakpoints — 1-hour average, units: ppb
NO2_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 2049, 301, 500),
]

# O3 breakpoints — 8-hour average, units: ppb
O3_BREAKPOINTS: List[Tuple[float, float, int, int]] = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

# Map pollutant column name → its breakpoint table
POLLUTANT_BREAKPOINTS: Dict[str, List[Tuple]] = {
    "pm25": PM25_BREAKPOINTS,
    "pm10": PM10_BREAKPOINTS,
    "co": CO_BREAKPOINTS,
    "no2": NO2_BREAKPOINTS,
    "o3": O3_BREAKPOINTS,
}

# Lower bound of the "Unhealthy" AQI category (151-200) per pollutant,
# used as a simple threshold-based anomaly trigger in label_spike_events().
EPA_UNHEALTHY_THRESHOLDS: Dict[str, float] = {
    "pm25": 55.5,
    "pm10": 255,
    "co": 12.5,
    "no2": 361,
    "o3": 0.086,
}

# EPA AQI health category labels (for reporting / dashboard display)
AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


# ===========================================================================
# SECTION 2 — CORE EPA AQI FORMULA
# ===========================================================================


def _piecewise_aqi(
    concentration: float, breakpoints: List[Tuple[float, float, int, int]]
) -> int:
    """
    Generic EPA piecewise-linear AQI formula.

    THE MATH:
        AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low

    This is just a linear interpolation between two known (concentration, AQI)
    anchor points.  The EPA defines these anchor points per pollutant in their
    Technical Assistance Document.

    Args:
        concentration: Measured pollutant concentration (units depend on pollutant).
        breakpoints:   Official EPA breakpoint table for that pollutant.

    Returns:
        Integer AQI value in [0, 500].  Returns 0 for invalid input.
    """
    if concentration < 0 or np.isnan(concentration):
        # Invalid reading — return 0 rather than crashing.
        # In production this should also trigger a data quality alert.
        return 0

    # Walk the breakpoint table to find which segment this concentration falls in
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            # Linear interpolation — the core EPA formula
            aqi = ((i_high - i_low) / (c_high - c_low)) * (
                concentration - c_low
            ) + i_low
            return int(round(aqi))

    # Above the highest breakpoint → cap at 500 (Hazardous)
    return 500


def compute_aqi_for_pollutant(concentration: float, pollutant: str) -> int:
    """
    Public wrapper: compute AQI for any supported pollutant.

    Args:
        concentration: Raw sensor reading.
        pollutant:     One of: 'pm25', 'pm10', 'co', 'no2', 'o3'.

    Returns:
        Integer AQI score.

    Example:
        >>> compute_aqi_for_pollutant(35.5, "pm25")
        101   # First value in the "Unhealthy for Sensitive Groups" range
    """
    pollutant = pollutant.lower().strip()
    if pollutant not in POLLUTANT_BREAKPOINTS:
        log.warning("Unknown pollutant '%s' — AQI not computed.", pollutant)
        return -1
    return _piecewise_aqi(concentration, POLLUTANT_BREAKPOINTS[pollutant])


def aqi_category(aqi_value: int) -> Tuple[str, str]:
    """
    Returns (label, hex_color) for a given AQI score.
    Used by the Streamlit dashboard for color-coded zone cards.
    """
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return label, color
    return "Hazardous", "#7e0023"


# ===========================================================================
# SECTION 3 — ROLLING STATISTICS
# ===========================================================================
#
# WHY NOT USE pandas .rolling() directly?
#   pandas .rolling() works on the full dataframe in one shot, which is fine
#   for batch processing.  But our pipeline also runs in streaming mode where
#   each sensor emits one reading at a time.  The history_buffers dict acts
#   as a per-sensor sliding window that works identically in both modes.
#
# WINDOWS CHOSEN:
#   1h  — captures transient spikes (traffic, industrial events)
#   6h  — captures medium-term trends (morning rush, afternoon peaks)
#   24h — captures full diurnal cycle; used for AQI standard averaging
# ===========================================================================


class RollingStatsCalculator:
    """
    Maintains a per-sensor, time-indexed sliding window of historical readings.

    State is kept in self.history_buffers so it survives across multiple
    calls (i.e., it works correctly in streaming mode as readings arrive
    one-by-one from Kafka).

    Usage:
        calc = RollingStatsCalculator(window_hours=24)
        features = calc.compute(sensor_id, timestamp, value)
    """

    def __init__(self, window_hours: int = 24):
        """
        Args:
            window_hours: Maximum history to keep per sensor (hours).
                          Readings older than this are discarded to save memory.
        """
        self.window_hours = window_hours
        # Dict[sensor_id → pd.DataFrame with DatetimeIndex and column 'value']
        self.history_buffers: Dict[str, pd.DataFrame] = {}

    def _update_buffer(
        self, sensor_id: str, timestamp: pd.Timestamp, value: float
    ) -> pd.DataFrame:
        """
        Append the new reading to this sensor's buffer, then prune old rows.

        IMPORTANT: We sort by index after each concat.  Without sorting,
        out-of-order Kafka messages (which happen in practice) would break
        the time-window slicing below.
        """
        new_row = pd.DataFrame({"value": [value]}, index=[timestamp])

        if sensor_id not in self.history_buffers:
            self.history_buffers[sensor_id] = new_row
        else:
            self.history_buffers[sensor_id] = pd.concat(
                [self.history_buffers[sensor_id], new_row]
            ).sort_index()  # ← sort guarantees correct time-window slicing

        # Prune: drop anything older than window_hours
        cutoff = timestamp - pd.Timedelta(hours=self.window_hours)
        self.history_buffers[sensor_id] = self.history_buffers[sensor_id].loc[
            self.history_buffers[sensor_id].index >= cutoff
        ]

        return self.history_buffers[sensor_id]

    def compute(
        self,
        sensor_id: str,
        timestamp: pd.Timestamp,
        value: float,
        pollutant: str = "pm25",
    ) -> Dict[str, float]:
        """
        Compute rolling statistics for one sensor reading.

        Returns a flat dict of feature names → values, ready to be
        concatenated into the main feature DataFrame.
        """
        history = self._update_buffer(sensor_id, timestamp, value)

        # Slice each time window from the full 24h buffer
        h1 = history.loc[history.index >= timestamp - pd.Timedelta(hours=1)]
        h6 = history.loc[history.index >= timestamp - pd.Timedelta(hours=6)]
        h24 = history  # the full buffer IS the 24h window

        prefix = pollutant  # e.g. "pm25_mean_1h"

        features: Dict[str, float] = {}

        for window_label, window_df in [("1h", h1), ("6h", h6), ("24h", h24)]:
            v = window_df["value"]
            n = len(v)
            features[f"{prefix}_mean_{window_label}"] = float(v.mean())
            features[f"{prefix}_std_{window_label}"] = float(v.std()) if n > 1 else 0.0
            features[f"{prefix}_min_{window_label}"] = float(v.min())
            features[f"{prefix}_max_{window_label}"] = float(v.max())
            # Count of readings in window — useful as a data-quality signal
            features[f"{prefix}_count_{window_label}"] = int(n)

        return features


# ===========================================================================
# SECTION 4 — CYCLICAL TIME ENCODING
# ===========================================================================
#
# THE PROBLEM WITH PLAIN INTEGERS:
#   If you encode hour as 0–23, a model sees hour 23 and hour 0 as 23 units
#   apart.  But 11pm and midnight are only 1 hour apart in real life.
#   This creates a hard discontinuity at midnight that confuses gradient-based
#   models.
#
# THE FIX — SIN/COS ENCODING:
#   Project the integer onto the unit circle:
#       sin(2π × hour / 24)   →  captures "where in the day"
#       cos(2π × hour / 24)   →  captures the other axis
#   Now hour 23 and hour 0 are very close together in 2D space. ✓
#
#   Same logic applies to day-of-week (period=7) and month (period=12).
# ===========================================================================


def add_cyclical_time_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Add sin/cos encodings for hour-of-day, day-of-week, and month-of-year.

    Args:
        df:            Input DataFrame containing a datetime timestamp column.
        timestamp_col: Name of the datetime column.

    Returns:
        DataFrame with 6 new columns appended (does NOT modify df in place).
    """
    result = df.copy()
    ts = result[timestamp_col]

    # Hour of day: period = 24
    hours = ts.dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Day of week: period = 7  (Monday=0, Sunday=6)
    days = ts.dt.dayofweek
    result["day_sin"] = np.sin(2 * np.pi * days / 7.0)
    result["day_cos"] = np.cos(2 * np.pi * days / 7.0)

    # Month of year: period = 12  (captures seasonal patterns)
    months = ts.dt.month
    result["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * months / 12.0)

    log.debug("Cyclical time features added: hour, day, month.")
    return result


# ===========================================================================
# SECTION 5 — STL DECOMPOSITION (CORRIGÉE)
# ===========================================================================


def add_stl_features(series: pd.Series, period: int = 24) -> pd.DataFrame:
    """
    Run STL decomposition on a single sensor's time series.

    Args:
        series: pd.Series of pollutant values, indexed by timestamp, sorted.
        period: Seasonal period (24 = daily cycle for hourly data).

    Returns:
        DataFrame (same index as series) with columns: trend, seasonal, residual.
    """
    if len(series) < 2 * period:
        raise ValueError(
            f"Series too short for STL (need >= {2 * period} points, got {len(series)})"
        )

    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    return pd.DataFrame(
        {
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
        },
        index=series.index,
    )


def add_stl_features_all_sensors(
    df: pd.DataFrame,
    sensor_col: str = "location_id",
    value_col: str = "value",
    timestamp_col: str = "timestamp",
    period: int = 24,
) -> pd.DataFrame:
    result_parts = []

    for sensor_id, group in df.groupby(sensor_col):
        series = group.set_index(timestamp_col)[value_col].sort_index()

        try:
            stl_df = add_stl_features(series, period=period)
            stl_df = stl_df.rename(
                columns={
                    "trend": "stl_trend",
                    "seasonal": "stl_seasonal",
                    "residual": "stl_residual",
                }
            )
            # Correction de l'index pour éviter les conflits au reset_index
            stl_df.index.name = timestamp_col
            stl_df[sensor_col] = sensor_id
            result_parts.append(stl_df.reset_index())
        except ValueError as e:
            log.warning("STL skipped for sensor %s: %s", sensor_id, e)

    if not result_parts:
        log.warning("No STL decomposition was possible — returning df unchanged.")
        return df

    stl_combined = pd.concat(result_parts).set_index([sensor_col, timestamp_col])

    # Fusion propre avec le dataframe principal
    df = df.set_index([sensor_col, timestamp_col])
    df = df.join(
        stl_combined[["stl_trend", "stl_seasonal", "stl_residual"]], how="left"
    )
    df = df.reset_index()

    log.info("STL decomposition complete for %d sensors.", len(result_parts))
    return df


# ===========================================================================
# SECTION 6 — SPATIAL CORRELATION FEATURES
# ===========================================================================
#
# THE IDEA:
#   If sensor A reads PM2.5 = 180 µg/m³ but its 3 nearest neighbours read
#   20–30 µg/m³, this is almost certainly a local emission source (factory,
#   truck) right next to sensor A — a high-priority anomaly.
#
#   If all sensors spike simultaneously, this is a city-wide pollution event
#   (dust storm, atmospheric inversion) — a different type of anomaly with
#   different response required.
#
#   The SPATIAL MEAN and SPATIAL DEVIATION encode this neighbourhood context
#   as a feature so the model can distinguish local vs. city-wide events.
#
# IMPLEMENTATION:
#   We use scipy.spatial.cKDTree for fast nearest-neighbour lookup.
#   cKDTree is a k-d tree (O(log n) queries) — much faster than computing
#   all pairwise distances (O(n²)) when you have 20+ sensors.
# ===========================================================================


def add_spatial_features(
    df: pd.DataFrame,
    sensor_locations: pd.DataFrame,
    sensor_col: str = "location_id",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    n_neighbors: int = 3,
) -> pd.DataFrame:
    """
    Add spatial context features: nearest-N-sensor mean and deviation.

    Args:
        df:               Main feature DataFrame with sensor readings.
        sensor_locations: DataFrame with columns [sensor_col, 'latitude', 'longitude'].
                          One row per sensor.  Coordinates must be in decimal degrees.
        sensor_col:       Column in df identifying the sensor.
        timestamp_col:    Column in df with timestamps.
        value_col:        Pollutant measurement column.
        n_neighbors:      How many nearest sensors to average (default 3).

    Returns:
        df with two new columns:
            'spatial_neighbor_mean'  — mean reading of the N nearest sensors
            'spatial_deviation'      — how much this sensor deviates from that mean
    """
    # Build the k-d tree from sensor lat/lon coordinates
    # Note: for small city-scale distances, Euclidean distance on lat/lon is
    # a good enough approximation.  For large regions use Haversine distance.
    coords = sensor_locations[["latitude", "longitude"]].values
    sensor_ids = sensor_locations[sensor_col].values
    tree = cKDTree(coords)

    # Pre-compute the N nearest neighbours for every sensor (done once)
    # distances shape: (n_sensors, n_neighbors+1); +1 because the sensor itself
    # is its own nearest neighbour (distance=0), which we skip.
    distances, indices = tree.query(coords, k=n_neighbors + 1)

    # Build a lookup: sensor_id → list of n_neighbors nearest sensor_ids
    neighbor_map: Dict[str, List[str]] = {}
    for i, sid in enumerate(sensor_ids):
        # Skip index 0 (itself), take indices 1..n_neighbors
        neighbor_ids = [sensor_ids[j] for j in indices[i][1 : n_neighbors + 1]]
        neighbor_map[str(sid)] = [str(n) for n in neighbor_ids]

    result = df.copy()
    neighbor_means = []
    deviations = []

    # For each timestamp, look up the concurrent readings from neighbour sensors
    # and compute the mean.  We iterate by timestamp group for efficiency.
    for ts, group in df.groupby(timestamp_col):
        # Build a dict of {sensor_id: value} for this timestamp slice
        ts_readings: Dict[str, float] = dict(
            zip(group[sensor_col].astype(str), group[value_col])
        )

        for _, row in group.iterrows():
            sid = str(row[sensor_col])
            neighbors = neighbor_map.get(sid, [])

            # Collect the values of neighbours that have a reading at this ts
            neighbor_values = [ts_readings[n] for n in neighbors if n in ts_readings]

            if neighbor_values:
                n_mean = float(np.mean(neighbor_values))
            else:
                # Edge case: no neighbour data at this timestamp (e.g. sensor offline)
                n_mean = float(row[value_col])

            neighbor_means.append(n_mean)
            deviations.append(float(row[value_col]) - n_mean)

    result["spatial_neighbor_mean"] = neighbor_means
    result["spatial_deviation"] = deviations

    log.info("Spatial features computed (k=%d neighbours).", n_neighbors)
    return result


# ===========================================================================
# SECTION 7 — SPIKE / ANOMALY LABELLING (CORRIGÉE)
# ===========================================================================


def label_spike_events(
    df: pd.DataFrame,
    value_col: str = "value",
    pollutant: str = "pm25",
    zscore_window: str = "24h",
    zscore_threshold: float = 3.0,
    residual_col: Optional[str] = "stl_residual",
    residual_threshold: float = 2.5,
) -> pd.DataFrame:
    result = df.copy()

    # ── Method 1: EPA Threshold ────────────────────────────────────────────
    threshold = EPA_UNHEALTHY_THRESHOLDS.get(pollutant.lower(), 55.5)
    result["label_threshold"] = (result[value_col] >= threshold).astype(int)

    # ── Method 2: Z-Score (rolling, per-sensor) ────────────────────────────
    # Correction : On garde l'index temporel actif pendant le calcul du z-score
    result = result.sort_values("timestamp").set_index("timestamp")

    roll_mean_parts, roll_std_parts = [], []
    for sensor_id, group in result.groupby("location_id"):
        r = group[value_col].rolling(zscore_window, min_periods=2)
        roll_mean_parts.append(r.mean())
        roll_std_parts.append(r.std())

    rolling_mean = pd.concat(roll_mean_parts).sort_index()
    rolling_std = pd.concat(roll_std_parts).sort_index().replace(0, np.nan)

    # L'alignement des index fonctionne maintenant parfaitement (DatetimeIndex vs DatetimeIndex)
    zscore = (result[value_col] - rolling_mean) / rolling_std
    result["label_zscore"] = (zscore.abs() >= zscore_threshold).fillna(0).astype(int)

    # On peut maintenant restaurer le timestamp en colonne
    result = result.reset_index()

    # ── Method 3: STL Residual ─────────────────────────────────────────────
    if residual_col and residual_col in result.columns:
        residual_std = result[residual_col].std()
        result["label_residual"] = (
            (result[residual_col].abs() >= residual_threshold * residual_std)
            .fillna(0)
            .astype(int)
        )
    else:
        result["label_residual"] = 0

    # ── Union Label ────────────────────────────────────────────────────────
    result["is_anomaly"] = (
        (result["label_threshold"] == 1)
        | (result["label_zscore"] == 1)
        | (result["label_residual"] == 1)
    ).astype(int)

    n_anomalies = result["is_anomaly"].sum()
    anomaly_rate = n_anomalies / len(result) * 100
    log.info(
        "Spike labelling complete: %d anomalies flagged (%.1f%% of data).",
        n_anomalies,
        anomaly_rate,
    )

    return result


# ===========================================================================
# SECTION 8 — MASTER PIPELINE FUNCTION (CORRIGÉE)
# ===========================================================================


def build_feature_matrix(
    raw_df: pd.DataFrame,
    sensor_locations: pd.DataFrame,
    pollutant: str = "pm25",
    stl_period: int = 24,
    n_spatial_neighbors: int = 3,
    rolling_window_hours: int = 24,
) -> pd.DataFrame:
    log.info(
        "Starting feature engineering pipeline for pollutant='%s', "
        "%d rows, %d sensors.",
        pollutant,
        len(raw_df),
        raw_df["location_id"].nunique(),
    )

    # ── Step 1: Sort ──────────────────────────────────────────────────────
    df = raw_df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # ── Step 2: Cyclical Time Features ────────────────────────────────────
    df = add_cyclical_time_features(df, timestamp_col="timestamp")
    log.info("Step 2 done: cyclical time features.")

    # ── Step 3: EPA AQI ───────────────────────────────────────────────────
    df["epa_aqi"] = df["value"].apply(lambda v: compute_aqi_for_pollutant(v, pollutant))
    df["aqi_category"] = df["epa_aqi"].apply(lambda v: aqi_category(v)[0])
    log.info("Step 3 done: EPA AQI computed.")

    # ── Step 4: Rolling Statistics ────────────────────────────────────────
    # Correction : Utilisation d'un dictionnaire temporaire pour éviter la fragmentation
    df = df.set_index("timestamp")
    rolling_parts = []

    for sensor_id, group in df.groupby("location_id"):
        group = group.sort_index()
        g = group["value"]
        new_cols = {}

        for window, label in [("1h", "1h"), ("6h", "6h"), ("24h", "24h")]:
            r = g.rolling(window, min_periods=1)
            new_cols[f"{pollutant}_mean_{label}"] = r.mean()
            new_cols[f"{pollutant}_std_{label}"] = r.std().fillna(0)
            new_cols[f"{pollutant}_min_{label}"] = r.min()
            new_cols[f"{pollutant}_max_{label}"] = r.max()

        new_cols_df = pd.DataFrame(new_cols, index=group.index)
        group = pd.concat([group, new_cols_df], axis=1)
        rolling_parts.append(group)

    df = pd.concat(rolling_parts).reset_index()
    log.info("Step 4 done: rolling statistics (1h / 6h / 24h).")

    # ── Step 5: STL Decomposition ─────────────────────────────────────────
    df = add_stl_features_all_sensors(
        df,
        sensor_col="location_id",
        value_col="value",
        timestamp_col="timestamp",
        period=stl_period,
    )
    log.info("Step 5 done: STL decomposition.")

    # ── Step 6: Spatial Features ──────────────────────────────────────────
    df = add_spatial_features(
        df,
        sensor_locations=sensor_locations,
        sensor_col="location_id",
        timestamp_col="timestamp",
        value_col="value",
        n_neighbors=n_spatial_neighbors,
    )
    log.info("Step 6 done: spatial correlation features.")

    # ── Step 7: Spike Labelling ───────────────────────────────────────────
    df = label_spike_events(
        df,
        value_col="value",
        pollutant=pollutant,
        residual_col="stl_residual",
    )
    log.info("Step 7 done: spike labelling. Feature matrix shape: %s", df.shape)

    return df


class AirQualityFeatureEngineer:
    """
    Wrapper stateful autour de build_feature_matrix, pour usage streaming
    (Kafka consumer). Reshape le DataFrame large (SensorSchema) vers le
    format long attendu par le pipeline (location_id, timestamp, value),
    accumule l'historique, puis applique le pipeline complet.

    Usage (consumer.py):
        engineer = AirQualityFeatureEngineer(
            sensor_locations_path="infrastructure/stations.json",
            pollutant="pm25",
        )
        enriched_df = engineer.extract_features(validated_df)
    """

    def __init__(
        self,
        sensor_locations_path: Optional[str] = None,
        pollutant: str = "pm25",
        stl_period: int = 24,
    ):
        self.pollutant = pollutant
        self.stl_period = stl_period
        self._history: Optional[pd.DataFrame] = None
        self.sensor_locations = self._load_sensor_locations(sensor_locations_path)

    def _load_sensor_locations(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path:
            log.warning(
                "No sensor_locations_path provided — spatial features will be skipped."
            )
            return None
        import json
        import os

        if not os.path.exists(path):
            log.warning(
                "stations.json not found at %s — spatial features will be skipped.",
                path,
            )
            return None
        with open(path, "r") as f:
            stations = json.load(f).get("stations", [])
        return pd.DataFrame(
            [
                {"location_id": s["id"], "latitude": s["lat"], "longitude": s["lon"]}
                for s in stations
            ]
        )

    def _reshape_for_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SensorSchema produit un format LARGE: sensor_id, pm25, pm10, co, no2, o3...
        build_feature_matrix attend un format LONG: location_id, value (un seul
        pollluant, celui configuré sur self.pollutant).
        """
        if self.pollutant not in df.columns:
            raise KeyError(
                f"Column '{self.pollutant}' not found in validated batch. "
                f"Available columns: {list(df.columns)}"
            )

        reshaped = df.rename(columns={"sensor_id": "location_id"}).copy()

        # Drop rows where this pollutant wasn't measured at this station (nullable=True in schema)
        reshaped = reshaped.dropna(subset=[self.pollutant])
        reshaped["value"] = reshaped[self.pollutant]

        return reshaped

    def extract_features(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        reshaped_batch = self._reshape_for_pipeline(batch_df)

        if self._history is None:
            self._history = reshaped_batch
        else:
            self._history = (
                pd.concat([self._history, reshaped_batch])
                .drop_duplicates(subset=["location_id", "timestamp"])
                .sort_values(["location_id", "timestamp"])
            )

        if self.sensor_locations is None:
            raise RuntimeError(
                "sensor_locations not loaded — pass sensor_locations_path to "
                "AirQualityFeatureEngineer() so spatial features can be computed."
            )

        enriched = build_feature_matrix(
            raw_df=self._history,
            sensor_locations=self.sensor_locations,
            pollutant=self.pollutant,
            stl_period=self.stl_period,
        )

        # Return only this batch's rows, not the whole accumulated history
        keys = set(zip(reshaped_batch["location_id"], reshaped_batch["timestamp"]))
        mask = list(zip(enriched["location_id"], enriched["timestamp"]))
        return enriched[[k in keys for k in mask]].reset_index(drop=True)


# ===========================================================================
# SECTION 9 — QUICK SMOKE TEST
# ===========================================================================
# Run this file directly to verify everything works with synthetic data:
#   python features.py
# ===========================================================================

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Running smoke test with synthetic IoT data...")
    log.info("=" * 60)

    # --- Synthetic sensor grid: 6 sensors in a city ---
    rng = np.random.default_rng(42)
    N_SENSORS = 6
    N_HOURS = 72  # 3 days of hourly readings

    sensor_ids = [f"SENSOR_{i:02d}" for i in range(N_SENSORS)]

    # Fake sensor coordinates (Tétouan-like bounding box)
    sensor_locations = pd.DataFrame(
        {
            "location_id": sensor_ids,
            "latitude": 35.57 + rng.uniform(-0.05, 0.05, N_SENSORS),
            "longitude": -5.37 + rng.uniform(-0.05, 0.05, N_SENSORS),
        }
    )

    # Build synthetic readings: baseline + diurnal pattern + noise + injected spikes
    records = []
    base_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    for sensor_id in sensor_ids:
        baseline = rng.uniform(15, 40)
        for h in range(N_HOURS):
            ts = base_time + pd.Timedelta(hours=h)
            # Diurnal pattern: peaks at 8am and 6pm (rush hours)
            diurnal = 15 * np.sin(2 * np.pi * h / 24) + 5 * np.sin(4 * np.pi * h / 24)
            noise = rng.normal(0, 3)
            value = max(0, baseline + diurnal + noise)

            # Inject 3 synthetic spikes into SENSOR_00 for testing labels
            if sensor_id == "SENSOR_00" and h in [10, 35, 60]:
                value = rng.uniform(160, 220)  # well into "Unhealthy" range

            records.append(
                {
                    "timestamp": ts,
                    "location_id": sensor_id,
                    "pollutant": "pm25",
                    "value": round(value, 2),
                }
            )

    raw_df = pd.DataFrame(records)
    log.info("Synthetic data created: %d rows, %d sensors.", len(raw_df), N_SENSORS)

    # --- Run the full pipeline ---
    features_df = build_feature_matrix(
        raw_df=raw_df,
        sensor_locations=sensor_locations,
        pollutant="pm25",
        stl_period=24,
        n_spatial_neighbors=3,
        rolling_window_hours=24,
    )

    # --- Print results ---
    log.info("-" * 60)
    log.info("Feature matrix shape : %s", features_df.shape)
    log.info("Columns              : %s", list(features_df.columns))
    log.info("Anomaly rate         : %.1f%%", features_df["is_anomaly"].mean() * 100)
    log.info(
        "AQI category counts  :\n%s",
        features_df["aqi_category"].value_counts().to_string(),
    )
    log.info("-" * 60)

    # Verify injected spikes were caught
    sensor_00 = features_df[features_df["location_id"] == "SENSOR_00"]
    spike_rows = sensor_00[sensor_00["is_anomaly"] == 1]
    log.info("SENSOR_00 — anomaly rows detected: %d (expected ≥ 3)", len(spike_rows))
    log.info(
        "Sample spike row:\n%s",
        spike_rows[
            [
                "timestamp",
                "value",
                "epa_aqi",
                "aqi_category",
                "is_anomaly",
                "label_threshold",
                "label_zscore",
            ]
        ]
        .head(3)
        .to_string(),
    )

    log.info("Smoke test PASSED. features.py is ready.")
