import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def login(username: str, password: str) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            data={"username": username, "password": password},
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


def get(path: str, params: dict = None):
    resp = requests.get(f"{API_BASE}{path}", headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def post(path: str, json: dict = None):
    resp = requests.post(f"{API_BASE}{path}", headers=_headers(), json=json)
    resp.raise_for_status()
    return resp.json()


def ensure_logged_in():
    """Call at the top of every page. Shows a login form if not authenticated."""
    if "token" in st.session_state:
        return True

    st.title("AirSentinel Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if login(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials")
    return False
