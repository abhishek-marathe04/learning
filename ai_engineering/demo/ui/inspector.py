"""Inspector panel: system prompt, tool definitions, and message history."""

import json
import streamlit as st

from agent.tools import TOOL_DEFINITIONS


def render_inspector(system_prompt: str, tools_enabled: bool, messages: list[dict]) -> None:
    """
    Renders three expanders:
    1. System Prompt
    2. Tool Definitions
    3. Messages
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

    with st.expander("Messages", expanded=False):
        if messages:
            st.json(messages)
        else:
            st.caption("No messages yet — run the agent to see the conversation history.")
