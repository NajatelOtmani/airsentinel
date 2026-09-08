import streamlit as st
from api_client import ensure_logged_in, post

st.set_page_config(page_title="AirSentinel — AI Analyst Chat", layout="wide")

if not ensure_logged_in():
    st.stop()

st.title("🤖 AI Analyst Chat")
st.caption("Ask questions about air quality, anomalies, and forecasts.")

# --- Session-state cached chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Render existing messages ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask the AirSentinel analyst...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = post("/api/v1/agent/", json={"query": user_input})
                answer = response.get("response", "No response received.")
            except Exception as e:
                answer = f"⚠️ Agent request failed: {e}"
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
