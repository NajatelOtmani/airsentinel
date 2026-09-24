import streamlit as st
from api_client import ensure_logged_in, post

st.set_page_config(page_title="AirSentinel — AI Analyst", layout="wide")

if not ensure_logged_in():
    st.stop()
from styles import inject_css, page_header
inject_css()
page_header("AI Powered", "AI Analyst", "Ask about anomalies, forecasts, and health guidance")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col_chat, col_side = st.columns([3, 1])

with col_side:
    st.markdown("**Suggested Questions**")
    suggestions = [
        "Why did PM2.5 jump this morning?",
        "Which zone has the worst air quality?",
        "What's the 12h forecast for LONDON_HF1?",
    ]
    for s in suggestions:
        if st.button(s, key=s, use_container_width=True):
            st.session_state.pending_query = s

with col_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask the AirSentinel analyst...")
    pending = st.session_state.pop("pending_query", None)
    final_query = query or pending

    if final_query:
        st.session_state.chat_history.append({"role": "user", "content": final_query})
        with st.spinner("Analyzing..."):
            try:
                response = post("/api/v1/agent/", json={"query": final_query})
                answer = response.get("response", "No response received.")
            except Exception as e:
                answer = f"Request failed: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()