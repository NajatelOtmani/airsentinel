from pathlib import Path
from threading import Lock

import pandas as pd

CSV_PATH = Path("data/processed_sensor_data.csv")

_lock = Lock()
_cache = {"mtime": None, "df": None}


def load_data() -> pd.DataFrame:
    """Loads processed_sensor_data.csv, cached until file changes on disk."""
    with _lock:
        mtime = CSV_PATH.stat().st_mtime if CSV_PATH.exists() else None
        if _cache["mtime"] != mtime:
            df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            _cache["df"] = df
            _cache["mtime"] = mtime
        return _cache["df"]
