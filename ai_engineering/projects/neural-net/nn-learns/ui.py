"""Small presenter-first UI helpers shared by every page."""

import streamlit as st

from core.style import PLOTLY_CONFIG


def is_presenter() -> bool:
    """Presenter notes are shown only when the URL has ?presenter=1."""
    val = st.query_params.get("presenter")
    return val in ("1", "true", "yes")


def presenter_notes(*points: str):
    """Bottom-of-page expander with talking points, hidden unless presenting."""
    if not is_presenter():
        return
    with st.expander("Presenter notes", expanded=False):
        for p in points:
            st.markdown(f"- {p}")


def reset_button(keys, label: str = "Reset", key: str | None = None):
    """A Reset button that clears this page's session_state keys on click."""
    if st.button(label, key=key):
        for k in keys:
            st.session_state.pop(k, None)
        st.rerun()


def chart(fig, **kwargs):
    """st.plotly_chart with the shared no-toolbar config."""
    st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True, **kwargs)
