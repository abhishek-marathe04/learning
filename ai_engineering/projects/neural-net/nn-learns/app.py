"""How a Network Learns — a presenter-first Streamlit teaching app.

Run:  streamlit run app.py
Presenter notes: add ?presenter=1 to the URL.
"""

import streamlit as st

st.set_page_config(
    page_title="How a Network Learns",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from cached import warm_up          # noqa: E402  (after set_page_config)
from core.style import APP_CSS

st.markdown(APP_CSS, unsafe_allow_html=True)

# --- Warm-up: compute every default run once, so nothing trains on stage. ---
if "warmed_up" not in st.session_state:
    with st.spinner("Warming up: pre-computing every training run…"):
        warm_up()
    st.session_state["warmed_up"] = True

# --- Navigation ------------------------------------------------------------
pages = [
    st.Page("pages/1_scoreboard.py", title="1 · Keep score", icon="🎯"),
    st.Page("pages/2_downhill.py", title="2 · Downhill", icon="⛰️"),
    st.Page("pages/3_backprop.py", title="3 · Backprop", icon="🔗"),
    st.Page("pages/4_bends.py", title="4 · Bends", icon="〰️"),
    st.Page("pages/5_next_letter.py", title="5 · Next letter", icon="🔤"),
]
st.navigation(pages).run()
