import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
STANDALONE = os.getenv("STANDALONE_MODE", "").lower() == "true"

if STANDALONE:
    os.chdir(ROOT)  # tools and the report generator use paths relative to the repo root

CSV_PATH = ROOT / "data" / "processed_sensor_data.csv"


# ── Standalone mode: no HTTP, call the code directly ─────────────────────────


@st.cache_data(ttl=60)
def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _clean(row: pd.Series) -> dict:
    out = {}
    for key, val in row.items():
        if pd.isna(val):
            out[key] = None
        elif isinstance(val, pd.Timestamp):
            out[key] = val.isoformat()
        elif hasattr(val, "item"):
            out[key] = val.item()
        else:
            out[key] = val
    return out


def _local_get(path: str, params: dict = None):
    df = _load_df()
    if path == "/api/v1/sensors/":
        return {"sensors": sorted(df["location_id"].dropna().unique().tolist())}

    m = re.fullmatch(r"/api/v1/sensors/([^/]+)/latest", path)
    if m:
        sub = df[df["location_id"] == m.group(1)].sort_values(
            "timestamp", ascending=False
        )
        return (
            {"error": "no data for this location_id"}
            if sub.empty
            else _clean(sub.iloc[0])
        )

    m = re.fullmatch(r"/api/v1/anomalies/([^/]+)", path)
    if m:
        limit = int((params or {}).get("limit", 50))
        sub = df[(df["location_id"] == m.group(1)) & (df["is_anomaly"] == 1)]
        sub = sub.sort_values("timestamp", ascending=False).head(limit)
        return {"anomalies": [_clean(r) for _, r in sub.iterrows()]}

    raise ValueError(f"Unsupported path in standalone mode: {path}")


def _local_post(path: str, payload: dict = None):
    if path == "/api/v1/agent/":
        if "agent" not in st.session_state:  # one agent (and memory) per visitor
            from src.agents.react_agent import AirSentinelAgent

            st.session_state["agent"] = AirSentinelAgent(provider="groq")
        return {"response": st.session_state["agent"].run((payload or {})["query"])}

    m = re.fullmatch(r"/api/v1/reports/([^/]+)", path)
    if m:
        from src.agents.report_generator import AutoReportGenerator

        return AutoReportGenerator(provider="groq").generate_report(
            location_id=m.group(1)
        )

    raise ValueError(f"Unsupported path in standalone mode: {path}")


# ── HTTP mode (Docker / local API) ───────────────────────────────────────────


def login(username: str, password: str) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            st.session_state["token"] = resp.json()["access_token"]
            return True
        return False
    except Exception as e:
        st.error(f"Login request failed: {e}")
        return False


def _headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, **kwargs):
    resp = requests.request(
        method, f"{API_BASE}{path}", headers=_headers(), timeout=120, **kwargs
    )
    if resp.status_code == 401:
        st.session_state.pop("token", None)
        st.rerun()
    resp.raise_for_status()
    return resp.json()


def get(path: str, params: dict = None):
    if STANDALONE:
        return _local_get(path, params)
    return _request("GET", path, params=params)


def post(path: str, json: dict = None):
    if STANDALONE:
        return _local_post(path, json)
    return _request("POST", path, json=json)


def ensure_logged_in():
    """Call at the top of every page."""
    if STANDALONE or "token" in st.session_state:
        return True

    if os.getenv("DEMO_MODE", "").lower() == "true":
        if login(os.getenv("DEMO_USERNAME", ""), os.getenv("DEMO_PASSWORD", "")):
            return True

    st.title("AirSentinel Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if login(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials")
    return False
