"""Page 4 — Bends. One neuron draws a line; ReLU hinges summed trace any shape.
Too small a model can't fit, no matter how long it trains."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import mlp
from core.data import CURVES
from core.style import AMBER, INK, MUTED, PIECES, TEAL, plotly_layout
from cached import cached_mlp
from ui import chart, presenter_notes, reset_button

PLAY_TICK = 0.08  # seconds between snapshots while playing

KEYS = ["p4_data", "p4_H", "p4_pieces", "p4_step", "p4_playing"]


def _back_to_start():
    """Changing shape or H starts over from the initial snapshot, not playing."""
    st.session_state.p4_step = 0
    st.session_state.p4_playing = False


st.title("Bends")
st.caption("Each hidden neuron is one hinge. Enough hinges, summed, trace any curve.")

top = st.columns([2.2, 2.2, 1])
dataset = top[0].segmented_control("Shape", CURVES, default="Wave", key="p4_data",
                                   on_change=_back_to_start) or "Wave"
H = top[1].segmented_control("Hidden neurons (H)", mlp.HIDDEN_SIZES, default=4,
                             key="p4_H", on_change=_back_to_start) or 4
show_pieces = top[2].toggle("Show pieces", value=True, key="p4_pieces")

# Trained lazily: only the shape and H on screen are computed (cached after that).
with st.spinner("Training the network…"):
    run = cached_mlp(dataset, H)
snaps = run["snaps"]
last = len(snaps) - 1
xg = np.linspace(-3, 3, 200)


@st.cache_data(show_spinner=False)
def fixed_axes(dataset: str, H: int):
    """Axis limits that hold for every snapshot, so the scale never jumps while playing.

    Y covers the data, the network curve and each hinge at every step; the loss
    axis (log10) covers the full training run.
    """
    r = cached_mlp(dataset, H)
    xs = np.linspace(-3, 3, 200)
    ys = [np.ravel(r["y"])]
    for s in r["snaps"]:
        ys.append(mlp.eval_curve(s, xs))
        ys.append(np.ravel(mlp.neuron_pieces(s, xs)[0]))
    allv = np.concatenate(ys)
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.08 * (hi - lo) or 0.1
    losses = np.log10([s["loss"] for s in r["snaps"]])
    lpad = 0.08 * (losses.max() - losses.min()) or 0.1
    return ([lo - pad, hi + pad],
            [float(losses.min() - lpad), float(losses.max() + lpad)])


Y_RANGE, LOSS_RANGE = fixed_axes(dataset, H)

playing = st.session_state.get("p4_playing", False)
if st.button("▶ Play", key="p4_play_btn", disabled=playing,
             help="Watch training once, from the start to the final step"):
    if st.session_state.get("p4_step", 0) >= last:
        st.session_state.p4_step = 0
    st.session_state.p4_playing = True
    playing = True


# While playing, this fragment reruns on a timer and advances one snapshot per
# tick. It stops itself at the final step (one pass, no looping).
@st.fragment(run_every=PLAY_TICK if playing else None)
def stepper():
    step = st.session_state.get("p4_step", 0)
    if st.session_state.get("p4_playing", False):
        step = min(step + 1, last)
        st.session_state.p4_step = step
        if step >= last:
            st.session_state.p4_playing = False
            st.rerun()  # full rerun so the timer is switched off

    cur = snaps[step]

    main, side = st.columns([2, 1])

    with main:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=run["x"], y=run["y"], mode="markers", name="data",
                                 marker=dict(color=INK, size=7, opacity=0.8)))
        fig.add_trace(go.Scatter(x=xg, y=mlp.eval_curve(cur, xg), mode="lines",
                                 name="network", line=dict(color=AMBER, width=5)))
        pieces, bends = mlp.neuron_pieces(cur, xg)
        bx = [b for b in bends if b is not None and -3 <= b <= 3]
        by = [mlp.eval_curve(cur, np.array([b]))[0] for b in bx]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="bends",
                                 marker=dict(color=TEAL, size=11, symbol="diamond"),
                                 showlegend=False))
        if show_pieces:
            for j in range(H):
                fig.add_trace(go.Scatter(x=xg, y=pieces[j], mode="lines",
                                         line=dict(color=PIECES[j % len(PIECES)], width=1.2),
                                         opacity=0.7, showlegend=False))
        fig.update_layout(plotly_layout(height=430, xaxis_title="x", yaxis_title="y",
                                        legend=dict(orientation="h", y=1.12)))
        fig.update_xaxes(range=[-3.1, 3.1], autorange=False)
        fig.update_yaxes(range=Y_RANGE, autorange=False)
        chart(fig)

    with side:
        st.metric("loss", f"{cur['loss']:.4f}")
        c = st.columns(2)
        c[0].metric("step", f"{cur['step']:,}")
        c[1].metric("knobs (3H+1)", f"{run['n_knobs']}")
        losses = [s["loss"] for s in snaps]
        steps = [s["step"] for s in snaps]
        lf = go.Figure(go.Scatter(x=steps, y=losses, mode="lines",
                                  line=dict(color=AMBER, width=3)))
        lf.add_vline(x=cur["step"], line=dict(color=TEAL, width=2, dash="dot"))
        lf.update_yaxes(type="log", range=LOSS_RANGE, autorange=False)
        lf.update_xaxes(range=[steps[0], steps[-1]], autorange=False)
        lf.update_layout(plotly_layout(height=250, xaxis_title="step",
                                       yaxis_title="loss (log)", showlegend=False))
        chart(lf)
        reset_button(KEYS, key="p4_reset")


stepper()

if dataset == "Wave" and H <= 2:
    st.caption(f"H = {H}: too few hinges. The loss plateaus around {run['final_loss']:.2f} "
               "— more training won't fix a too-small model.")
elif dataset == "Wave" and H >= 8:
    st.caption(f"H = {H}: enough hinges. Final loss {run['final_loss']:.3f}. "
               "Nobody assigned jobs — each neuron found its own bend.")

presenter_notes(
    "One hinge can't make a wave; no amount of training fixes too-small a model.",
    "At H=8, nobody assigned jobs; each neuron found its bend because the gradient said so.",
    "Uses Adam, a smarter step rule — same gradients.",
)
