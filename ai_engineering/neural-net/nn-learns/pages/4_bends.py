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

KEYS = ["p4_data", "p4_H", "p4_pieces", "p4_step"]

st.title("Bends")
st.caption("Each hidden neuron is one hinge. Enough hinges, summed, trace any curve.")

top = st.columns([2.2, 2.2, 1])
dataset = top[0].segmented_control("Shape", CURVES, default="Wave", key="p4_data") or "Wave"
H = top[1].segmented_control("Hidden neurons (H)", mlp.HIDDEN_SIZES, default=4,
                             key="p4_H") or 4
show_pieces = top[2].toggle("Show pieces", value=True, key="p4_pieces")

run = cached_mlp(dataset, H)
snaps = run["snaps"]
i = st.slider("Training step", 0, len(snaps) - 1, len(snaps) - 1,
              format="snapshot %d", key="p4_step")
cur = snaps[i]

xg = np.linspace(-3, 3, 200)
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
    frames = []
    for k, s in enumerate(snaps):
        _, bd = mlp.neuron_pieces(s, xg)
        fbx = [b for b in bd if b is not None and -3 <= b <= 3]
        fby = [mlp.eval_curve(s, np.array([b]))[0] for b in fbx]
        frames.append(go.Frame(name=str(k), traces=[1, 2], data=[
            go.Scatter(x=xg, y=mlp.eval_curve(s, xg)),
            go.Scatter(x=fbx, y=fby),
        ]))
    fig.frames = frames
    fig.update_layout(plotly_layout(height=430, xaxis_title="x", yaxis_title="y",
                                    legend=dict(orientation="h", y=1.12)))
    fig.update_layout(updatemenus=[dict(
        type="buttons", showactive=False, x=0.0, y=1.22, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=60, redraw=True),
                                       transition=dict(duration=0), fromcurrent=True)])])])
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
    lf.update_yaxes(type="log")
    lf.update_layout(plotly_layout(height=250, xaxis_title="step",
                                   yaxis_title="loss (log)", showlegend=False))
    chart(lf)
    reset_button(KEYS, key="p4_reset")

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
