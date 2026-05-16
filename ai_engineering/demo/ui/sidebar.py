"""Sidebar presenter controls."""

import streamlit as st


def render_sidebar() -> dict:
    """
    Renders sidebar controls and returns current state as a dict with keys:
    use_mock, tools_enabled, cot_enabled, memory_enabled,
    current_branch, previous_branch, run_clicked
    """
    with st.sidebar:
        st.header("Presenter Controls")

        # --- Mock toggle ---
        if "use_mock" not in st.session_state:
            st.session_state.use_mock = True
        use_mock = st.toggle(
            "USE MOCK DATA",
            value=st.session_state.use_mock,
            key="use_mock",
        )

        st.divider()
        st.subheader("Demo Mode")

        # --- Tools toggle ---
        if "tools_enabled" not in st.session_state:
            st.session_state.tools_enabled = False
        tools_enabled = st.toggle(
            "TOOLS ENABLED",
            value=st.session_state.tools_enabled,
            key="tools_enabled",
            help="Give the agent access to GitLab tools (get_branch_diff, get_pr_details, get_past_release_notes). Without tools the agent must rely only on what's in the prompt.",
        )

        # --- Chain of thought toggle ---
        if "cot_enabled" not in st.session_state:
            st.session_state.cot_enabled = False
        cot_enabled = st.toggle(
            "CHAIN OF THOUGHT",
            value=st.session_state.cot_enabled,
            key="cot_enabled",
            help="Instructs the agent to reason step-by-step before writing the release notes. Shows the model's internal reasoning process.",
        )

        # --- Memory toggle ---
        if "memory_enabled" not in st.session_state:
            st.session_state.memory_enabled = False
        memory_enabled = st.toggle(
            "MEMORY",
            value=st.session_state.memory_enabled,
            key="memory_enabled",
            help="Loads team preferences and past release context from agent_memory.json and injects it into the system prompt.",
        )

        st.divider()
        st.subheader("Task")

        current_branch = st.text_input(
            "Current Branch",
            value="release/v2.2.0",
            key="current_branch",
        )

        previous_branch = st.text_input(
            "Previous Branch",
            value="release/v2.1.0",
            key="previous_branch",
        )

        run_clicked = st.button("Run Agent", type="primary", use_container_width=True)

    return {
        "use_mock": use_mock,
        "tools_enabled": tools_enabled,
        "cot_enabled": cot_enabled,
        "memory_enabled": memory_enabled,
        "current_branch": current_branch,
        "previous_branch": previous_branch,
        "run_clicked": run_clicked,
    }
