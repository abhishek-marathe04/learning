"""Agent trace renderer — displays THOUGHT / ACTION / OBSERVATION / FINAL events."""

import streamlit as st

from agent.loop import AgentEvent

# Styling config per event type
EVENT_STYLES = {
    "thought": {
        "label": "🤔 THOUGHT",
        "border_color": "#1E90FF",
        "bg_color": "#EFF6FF",
    },
    "action": {
        "label": "⚡ ACTION",
        "border_color": "#F59E0B",
        "bg_color": "#FFFBEB",
    },
    "observation": {
        "label": "👁️ OBSERVATION",
        "border_color": "#9CA3AF",
        "bg_color": "#F9FAFB",
    },
    "final": {
        "label": "✅ FINAL",
        "border_color": "#10B981",
        "bg_color": "#ECFDF5",
    },
    "error": {
        "label": "❌ ERROR",
        "border_color": "#EF4444",
        "bg_color": "#FEF2F2",
    },
}


def _render_event_card(event: AgentEvent) -> None:
    """Renders a single event as a styled card."""
    style = EVENT_STYLES.get(event.type, EVENT_STYLES["thought"])

    st.markdown(
        f"""
        <div style="
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
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trace(events: list[AgentEvent]) -> None:
    """Renders the full list of agent events as styled cards."""
    if not events:
        st.caption("Agent trace will appear here when you run the agent.")
        return

    # Group consecutive thought events — show only the last (most complete) one
    # for the streaming case. For non-streaming display, show all non-thought events
    # plus the final thought state.
    rendered_thought_indices: set[int] = set()
    i = 0
    while i < len(events):
        event = events[i]
        if event.type == "thought":
            # Find the last consecutive thought event
            j = i
            while j + 1 < len(events) and events[j + 1].type == "thought":
                j += 1
            # Render only the last (most complete) thought in a run
            _render_event_card(events[j])
            i = j + 1
        else:
            _render_event_card(event)
            i += 1


def create_streaming_placeholder() -> st.delta_generator.DeltaGenerator:
    """Creates an empty placeholder for streaming thought content."""
    return st.empty()
