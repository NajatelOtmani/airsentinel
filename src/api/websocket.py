import asyncio
import json

from fastapi import WebSocket
from loguru import logger

from src.api.data_store import load_data
from src.api.metrics import anomaly_events_total, websocket_connections_active


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        websocket_connections_active.inc()

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
            websocket_connections_active.dec()

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
_last_seen_ts = None


def pd_isna(val):
    import pandas as pd

    return pd.isna(val)


async def poll_anomalies_loop():
    """Polls the CSV every 1.5s for new is_anomaly==1 rows; pushes to WebSocket clients."""
    global _last_seen_ts
    while True:
        try:
            df = load_data()
            anomalies = df[df["is_anomaly"] == 1].sort_values("timestamp")
            if _last_seen_ts is not None:
                anomalies = anomalies[anomalies["timestamp"] > _last_seen_ts]
            for _, row in anomalies.iterrows():
                _last_seen_ts = row["timestamp"]
                anomaly_events_total.labels(location_id=row["location_id"]).inc()
                await manager.broadcast(
                    {
                        "type": "anomaly",
                        "location_id": row["location_id"],
                        "timestamp": str(row["timestamp"]),
                        "pm25": float(row["pm25"])
                        if not pd_isna(row["pm25"])
                        else None,
                        "epa_aqi": int(row["epa_aqi"])
                        if not pd_isna(row["epa_aqi"])
                        else None,
                    }
                )
        except Exception as e:
            logger.error(f"Anomaly poll error: {e}")
        await asyncio.sleep(1.5)
