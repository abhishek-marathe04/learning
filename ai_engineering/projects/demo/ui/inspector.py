"""Inspector panel: system prompt, tool definitions, and message history."""

import json
import streamlit as st

from agent.tools import TOOL_DEFINITIONS


def render_inspector(system_prompt: str, tools_enabled: bool) -> None:
    """
    Renders two expanders:
    1. System Prompt
    2. Tool Definitions
    """
    with st.expander("System Prompt", expanded=True):
        st.code(system_prompt, language="text")

    with st.expander("Tool Definitions", expanded=False):
        if tools_enabled:
            for tool_def in TOOL_DEFINITIONS:
                func = tool_def["function"]
                st.markdown(f"**`{func['name']}`**")
                st.caption(func["description"])
                st.json(func["parameters"])
                st.divider()
        else:
            st.info("Tools are disabled")

