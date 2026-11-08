"""Page 5 — Next letter. A real (tiny) language model: one 27x27 matrix.
Same training loop; here the loss is surprise."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import bigram
from core.data import CHARS
from core.style import AMBER, INK, MUTED, TEAL, plotly_layout
from cached import cached_bigram
from ui import chart, presenter_notes, reset_button

PLAY_TICK = 0.35  # seconds between snapshots while playing (16 snapshots ≈ 6 s)

KEYS = ["p5_step", "p5_name", "p5_playing"]
HEAT_COLORS = [[0.0, "#151A33"], [0.5, "#3CC6B0"], [1.0, "#F2A541"]]

st.title("Next letter")
st.caption("27 tokens: '.' plus a–z. Predict the next letter. The loss is surprise, in nats.")

run = cached_bigram()
snaps = run["snaps"]
last = len(snaps) - 1
labels = [s["step"] for s in snaps]
steps = labels
losses = [s["loss"] for s in snaps]
# One colour scale for every step, so the untrained (flat) grid reads as dark
# and lights up as it learns, instead of rescaling each frame.
ZMAX = float(bigram.softmax_rows(snaps[-1]["W"]).max())
LOSS_RANGE = [min(losses + [run["optimum"]]) - 0.15, max(losses) + 0.15]

playing = st.session_state.get("p5_playing", False)
if st.button("▶ Play", key="p5_play_btn", disabled=playing,
             help="Watch training once, from untrained to the final step"):
    if st.session_state.get("p5_step", 0) >= last:
        st.session_state.p5_step = 0
    st.session_state.p5_playing = True
    playing = True


# While playing, this fragment reruns on a timer and advances one snapshot per
# tick, so the heatmap, invented names, surprise bars and loss marker all move
# together. It stops itself at the final step (one pass, no looping).
@st.fragment(run_every=PLAY_TICK if playing else None)
def stepper():
    if st.session_state.get("p5_playing", False):
        step = min(st.session_state.get("p5_step", 0) + 1, last)
        st.session_state.p5_step = step
        if step >= last:
            st.session_state.p5_playing = False
            st.rerun()  # full rerun so the timer is switched off
    elif "p5_step" not in st.session_state:
        st.session_state.p5_step = 0  # open on the untrained model

    i = st.select_slider("Training step", options=list(range(len(snaps))),
                         format_func=lambda k: f"step {labels[k]}", key="p5_step")
    cur = snaps[i]
    W = cur["W"]

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("What follows what")
        P = bigram.softmax_rows(W)
        hm = go.Figure(go.Heatmap(z=P, x=CHARS, y=CHARS, colorscale=HEAT_COLORS,
                                  showscale=False, zmin=0, zmax=ZMAX))
        hm.update_layout(plotly_layout(height=430, xaxis_title="next letter",
                                       yaxis_title="current letter"))
        hm.update_yaxes(autorange="reversed")
        chart(hm)

    with right:
        st.subheader("Names it invents right now")
        names = bigram.sample_names(W, n=10, seed=42)
        st.markdown("  \n".join(f"**{n or '(empty)'}**" for n in names))

        st.subheader("How surprised is it by this name?")
        name = st.text_input("Type a name", value="aarav", key="p5_name")
        rows, avg = bigram.surprise_of(W, name or "aarav")
        letters = [r["to"] for r in rows]
        fig = go.Figure(go.Bar(x=list(range(len(rows))), y=[r["surprise"] for r in rows],
                               marker_color=AMBER,
                               text=[f"{r['p'] * 100:.0f}%" for r in rows],
                               textposition="outside"))
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(rows))),
                         ticktext=[f"→{c}" for c in letters])
        fig.update_layout(plotly_layout(height=230, yaxis_title="surprise (−ln p)",
                                        showlegend=False))
        chart(fig)
        st.caption(f"Average surprise {avg:.2f}  (knew nothing = {run['uniform_loss']:.2f})")

    st.subheader("Loss = average surprise while training")
    lf = go.Figure(go.Scatter(x=steps, y=losses, mode="lines+markers",
                              line=dict(color=AMBER, width=3), marker=dict(size=6)))
    lf.add_vline(x=cur["step"], line=dict(color=TEAL, width=2, dash="dot"))
    lf.add_hline(y=run["uniform_loss"], line=dict(color=MUTED, dash="dash"),
                 annotation_text="knows nothing: 3.30", annotation_position="top right")
    lf.add_hline(y=run["optimum"], line=dict(color=TEAL, dash="dash"),
                 annotation_text=f"best one-letter memory: {run['optimum']:.2f}",
                 annotation_position="bottom right")
    lf.update_xaxes(range=[steps[0] - 5, steps[-1] + 5], autorange=False)
    lf.update_yaxes(range=LOSS_RANGE, autorange=False)
    lf.update_layout(plotly_layout(height=260, xaxis_title="step", yaxis_title="loss (nats)",
                                   showlegend=False))
    chart(lf)


stepper()

reset_button(KEYS, key="p5_reset")

presenter_notes(
    "Letters never seen (f, q, x, z) keep flat rows: no data, no nudges.",
    "Type an audience member's name and watch its surprise.",
    "An LLM is this, but with thousands of tokens of context through a transformer "
    "and ~10^11 knobs. Same loop.",
)
