import pandas as pd
from fastapi import APIRouter, Depends

from src.api.auth import get_current_user
from src.api.data_store import load_data

router = APIRouter(prefix="/api/v1/sensors", tags=["sensors"])


def _clean_row(row: pd.Series) -> dict:
    """Converts a pandas Series to a JSON-safe dict (native Python types)."""
    result = {}
    for key, val in row.items():
        if pd.isna(val):
            result[key] = None
        elif isinstance(val, pd.Timestamp):
            result[key] = val.isoformat()
        elif hasattr(val, "item"):  # numpy scalar (int64, float64, etc.)
            result[key] = val.item()
        else:
            result[key] = val
    return result


@router.get("/")
async def list_sensors(user: str = Depends(get_current_user)):
    df = load_data()
    return {"sensors": sorted(df["location_id"].dropna().unique().tolist())}


@router.get("/{location_id}/latest")
async def latest_reading(location_id: str, user: str = Depends(get_current_user)):
    df = load_data()
    subset = df[df["location_id"] == location_id].sort_values(
        "timestamp", ascending=False
    )
    if subset.empty:
        return {"error": "no data for this location_id"}
    row = subset.iloc[0]
    return _clean_row(row)
