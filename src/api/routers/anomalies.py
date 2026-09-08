import pandas as pd
from fastapi import APIRouter, Depends

from src.api.auth import get_current_user
from src.api.data_store import load_data

router = APIRouter(prefix="/api/v1/anomalies", tags=["anomalies"])


def _clean_records(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        rec = {}
        for key, val in row.items():
            if pd.isna(val):
                rec[key] = None
            elif isinstance(val, pd.Timestamp):
                rec[key] = val.isoformat()
            elif hasattr(val, "item"):
                rec[key] = val.item()
            else:
                rec[key] = val
        records.append(rec)
    return records


@router.get("/{location_id}")
async def get_anomalies(
    location_id: str, limit: int = 50, user: str = Depends(get_current_user)
):
    df = load_data()
    subset = df[(df["location_id"] == location_id) & (df["is_anomaly"] == 1)]
    subset = subset.sort_values("timestamp", ascending=False).head(limit)
    return {"anomalies": _clean_records(subset)}
