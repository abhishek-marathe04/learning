"""Page 1 — Keep score. Turn the knobs, watch the loss, run the wiggle test,
then let PyTorch produce the same slopes."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import neuron
from core.style import AMBER, MUTED, TEAL, callout, plotly_layout, stat_block
from ui import chart, presenter_notes, reset_button

X, Y = neuron.X_DEFAULT, neuron.Y_DEFAULT
KEYS = ["p1_w", "p1_b", "p1_history", "p1_panel"]

for k, default in [("p1_w", 1.0), ("p1_b", 0.0),
                   ("p1_history", [neuron.loss_of(1.0, 0.0)]), ("p1_panel", None)]:
    st.session_state.setdefault(k, default)

st.session_state.p1_w = neuron.clamp(float(st.session_state.p1_w), 0.0, 3.0)
st.session_state.p1_b = neuron.clamp(float(st.session_state.p1_b), -40.0, 80.0)


def do_nudge():
    w, b = st.session_state.p1_w, st.session_state.p1_b
    nw, nb, nloss = neuron.one_nudge(w, b)
    st.session_state.p1_w = neuron.clamp(nw, 0.0, 3.0)
    st.session_state.p1_b = neuron.clamp(nb, -40.0, 80.0)
    st.session_state.p1_history.append(nloss)
    st.session_state.p1_panel = "nudge"


st.title("Keep score")
st.caption("One neuron, one reading: 10°C should read 50°F. Turn the knobs; the loss keeps score.")

left, right = st.columns([1.1, 1])

with left:
    w = st.slider("w  (°C → °F scale)", 0.0, 3.0, key="p1_w", step=0.01)
    b = st.slider("b  (offset)", -40.0, 80.0, key="p1_b", step=0.01)
    dot = f"""
    digraph {{
        rankdir=LR; bgcolor="transparent";
        node [shape=box style=rounded fontcolor="#ECEEF8" color="#9AA2C8" fontsize=16];
        edge [color="#9AA2C8" fontcolor="#3CC6B0" fontsize=14];
        x   [label="°C = {X:g}" color="#ECEEF8"];
        wn  [label="× w ({w:.2f})" color="#3CC6B0" fontcolor="#3CC6B0"];
        bn  [label="+ b ({b:.2f})" color="#3CC6B0" fontcolor="#3CC6B0"];
        g   [label="guess = {w * X + b:.2f}" color="#ECEEF8"];
        x -> wn -> bn -> g;
    }}
    """
    st.graphviz_chart(dot, use_container_width=True)

with right:
    guess, miss, loss = neuron.forward(w, b)
    st.markdown(stat_block("input / truth", f"{X:g}°C → {Y:g}°F", "ink"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(stat_block("guess", f"{guess:,.2f}", "knob"), unsafe_allow_html=True)
    c2.markdown(stat_block("miss", f"{miss:,.2f}", "loss"), unsafe_allow_html=True)
    st.markdown(stat_block("LOSS", f"{loss:,.2f}", "loss"), unsafe_allow_html=True)

st.divider()
b1, b2, b3, b4 = st.columns(4)
if b1.button("🔍 Wiggle test", use_container_width=True):
    st.session_state.p1_panel = "wiggle"
if b2.button("⚡ Ask PyTorch", use_container_width=True):
    st.session_state.p1_panel = "pytorch"
b3.button("👇 Take one nudge", use_container_width=True, on_click=do_nudge)
with b4:
    reset_button(KEYS, key="p1_reset")

panel = st.session_state.p1_panel

if panel == "wiggle":
    st.subheader("Wiggle test — nudge each knob by +0.01, watch the loss")
    rows = neuron.wiggle_test(w, b)
    df = pd.DataFrame([{
        "knob": r["knob"],
        "new guess": f"{r['new_guess']:.2f}",
        "new loss": f"{r['new_loss']:.2f}",
        "change": f"{r['change']:.2f}",
        "slope = change ÷ 0.01": f"{r['slope']:.0f}",
    } for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Honest, but it costs one full model run per knob.")

elif panel == "pytorch":
    st.subheader("Ask PyTorch — one backward pass gives every slope")
    code = (
        f"w = torch.tensor({w:.2f}, requires_grad=True)\n"
        f"b = torch.tensor({b:.2f}, requires_grad=True)\n"
        f"loss = (w * {X:g} + b - {Y:g}) ** 2\n"
        f"loss.backward()\n"
        f"print(w.grad, b.grad)"
    )
    gw, gb = neuron.autograd_grads(w, b)
    st.code(code, language="python")
    st.code(f"tensor({gw:.1f}) tensor({gb:.1f})", language="text")
    wig = {r["knob"]: r["slope"] for r in neuron.wiggle_test(w, b)}
    st.markdown(callout(
        f"Wiggle test said ≈ {wig['w']:.0f} and ≈ {wig['b']:.0f}. "
        f"Autograd says exactly {gw:.0f} and {gb:.0f}.", "teal"),
        unsafe_allow_html=True)

if panel == "nudge" or len(st.session_state.p1_history) > 1:
    hist = st.session_state.p1_history
    if len(hist) > 1:
        st.subheader("Nudge history")
        st.caption(f"Step size fixed at {neuron.NUDGE_LR}. "
                   f"loss {hist[-2]:,.2f} → {hist[-1]:,.2f}")
    fig = go.Figure(go.Scatter(y=hist, mode="lines+markers",
                               line=dict(color=AMBER, width=3),
                               marker=dict(color=AMBER, size=7)))
    fig.update_layout(plotly_layout(height=220, showlegend=False,
                                    xaxis_title="nudge", yaxis_title="loss"))
    chart(fig)

presenter_notes(
    "Wiggle is honest but costs one model run per knob.",
    "Autograd gets every slope in one backward pass.",
    "Keep pressing nudge: watch the loss fall each time.",
)
