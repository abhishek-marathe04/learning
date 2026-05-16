"""Release Notes Generator — Agent Demo (Streamlit entry point)."""

import streamlit as st

from agent.loop import AgentEvent, run as agent_run
from agent.prompts import build_system_prompt
from ui.sidebar import render_sidebar
from ui.inspector import render_inspector
from ui.trace import render_trace, EVENT_STYLES

st.set_page_config(layout="wide", page_title="Release Notes Agent")

# --- Session state initialisation ---
if "events" not in st.session_state:
    st.session_state.events = []

if "running" not in st.session_state:
    st.session_state.running = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = build_system_prompt(
        cot_enabled=False, memory_enabled=False
    )

# --- Sidebar ---
controls = render_sidebar()

# --- Title ---
st.title("Release Notes Generator — Agent Demo")

# --- Tabs ---
inspector_tab, trace_tab = st.tabs(["Inspector", "Agent Trace"])

# --- Update system prompt whenever controls change ---
system_prompt = build_system_prompt(
    cot_enabled=controls["cot_enabled"],
    memory_enabled=controls["memory_enabled"],
)
st.session_state.system_prompt = system_prompt

# --- Inspector tab (always rendered with current state) ---
with inspector_tab:
    render_inspector(
        system_prompt=st.session_state.system_prompt,
        tools_enabled=controls["tools_enabled"],
        messages=st.session_state.messages,
    )

# --- Agent Trace tab ---
with trace_tab:
    if controls["run_clicked"] and not st.session_state.running:
        # Reset state for a new run
        st.session_state.events = []
        st.session_state.messages = []
        st.session_state.running = True

        task = (
            f"Generate release notes for {controls['current_branch']} "
            f"compared to {controls['previous_branch']}."
        )

        # We'll render events live in this container
        trace_container = st.container()

        # Streaming state: track the placeholder for the current thought block
        streaming_state = {
            "thought_placeholder": None,
            "accumulated_thought": "",
            "last_was_thought": False,
        }

        def on_event(event: AgentEvent) -> None:
            """Callback called by the agent loop for each event."""
            if event.type == "thought":
                # Update or create the streaming thought placeholder
                if not streaming_state["last_was_thought"] or streaming_state["thought_placeholder"] is None:
                    # Start a new thought card placeholder
                    with trace_container:
                        streaming_state["thought_placeholder"] = st.empty()
                    streaming_state["accumulated_thought"] = ""

                streaming_state["accumulated_thought"] = event.content
                streaming_state["last_was_thought"] = True

                style = EVENT_STYLES["thought"]
                streaming_state["thought_placeholder"].markdown(
                    f"""<div style="
                        border-left: 4px solid {style['border_color']};
                        background-color: {style['bg_color']};
                        padding: 12px 16px;
                        margin-bottom: 12px;
                        border-radius: 4px;
                    ">
                        <div style="font-weight: 600; font-size: 0.8em; color: {style['border_color']}; margin-bottom: 6px;">
                            {style['label']}
                        </div>
                        <div style="white-space: pre-wrap; font-size: 0.9em;">{event.content}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # Record the latest thought in events (replace last if it was also a thought)
                if st.session_state.events and st.session_state.events[-1].type == "thought":
                    st.session_state.events[-1] = event
                else:
                    st.session_state.events.append(event)

            else:
                # Non-thought event: reset thought streaming state and render directly
                streaming_state["thought_placeholder"] = None
                streaming_state["accumulated_thought"] = ""
                streaming_state["last_was_thought"] = False

                st.session_state.events.append(event)

                style = EVENT_STYLES.get(event.type, EVENT_STYLES["thought"])
                with trace_container:
                    st.markdown(
                        f"""<div style="
                            border-left: 4px solid {style['border_color']};
                            background-color: {style['bg_color']};
                            padding: 12px 16px;
                            margin-bottom: 12px;
                            border-radius: 4px;
                        ">
                            <div style="font-weight: 600; font-size: 0.8em; color: {style['border_color']}; margin-bottom: 6px;">
                                {style['label']}
                            </div>
                            <div style="white-space: pre-wrap; font-size: 0.9em;">{event.content}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        try:
            agent_run(
                task=task,
                tools_enabled=controls["tools_enabled"],
                cot_enabled=controls["cot_enabled"],
                memory_enabled=controls["memory_enabled"],
                on_event=on_event,
            )
        except Exception as exc:
            error_event = AgentEvent(type="error", content=str(exc))
            on_event(error_event)
        finally:
            st.session_state.running = False

    elif st.session_state.events:
        # Re-render past events (e.g. after a rerun caused by sidebar toggle)
        render_trace(st.session_state.events)

    elif st.session_state.running:
        st.info("Agent is running...")

    else:
        st.caption("Configure the agent in the sidebar and click **Run Agent** to start.")
