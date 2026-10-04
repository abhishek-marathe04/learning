"""Page 2 — Downhill. Two knobs discover F = 1.8·C + 32 from noisy readings.
Watch the ball roll to the valley floor, and watch a too-big step explode."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import neuron
from core.style import (AMBER, INK, MUTED, RED, TEAL, callout, plotly_layout)
from cached import cached_gd, cached_loss_surface, cached_temp_data
from ui import chart, presenter_notes, reset_button

KEYS = ["p2_lr", "p2_step", "p2_w0", "p2_b0"]
st.session_state.setdefault("p2_w0", neuron.W_START)
st.session_state.setdefault("p2_b0", neuron.B_START)

st.title("Downhill")
st.caption("Nobody tells it the formula. Gradient descent rolls the two knobs to the valley floor.")

top = st.columns([2, 1.4, 1])
lr_name = top[0].segmented_control(
    "Step size", list(neuron.LR_PRESETS),
    default="Good", key="p2_lr",
    format_func=lambda n: f"{n} {neuron.LR_PRESETS[n]:g}")
lr_name = lr_name or "Good"
w0 = top[1].number_input("Start w", 0.0, 3.2, key="p2_w0", step=0.1)
b0 = top[2].number_input("Start b", -40.0, 80.0, key="p2_b0", step=1.0)

x, y = cached_temp_data()
run = cached_gd(lr_name, w0, b0)
snaps = run["snaps"]
ws = np.array([s["w"] for s in snaps])
bs = np.array([s["b"] for s in snaps])

i = st.slider("Training step", 0, len(snaps) - 1, len(snaps) - 1,
              format="snapshot %d", key="p2_step")
cur = snaps[i]
gw_grid, bw_grid, Z = cached_loss_surface()

# --- Build the combined figure: left scatter, right loss landscape ----------
fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                    subplot_titles=("Data and guess", "Loss landscape"),
                    horizontal_spacing=0.09)

xline = np.linspace(x.min(), x.max(), 50)


def model_line(s):
    return s["w"] * xline + s["b"]


def miss_segments(s):
    px, py = [], []
    for xi, yi in zip(x, y):
        px += [xi, xi, None]
        py += [yi, s["w"] * xi + s["b"], None]
    return px, py


# Left: static data points, animated model line + miss lines.
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="readings",
                         marker=dict(color=INK, size=8)), row=1, col=1)
mpx, mpy = miss_segments(cur)
fig.add_trace(go.Scatter(x=mpx, y=mpy, mode="lines", name="miss",
                         line=dict(color=AMBER, width=1, dash="dot"),
                         showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=xline, y=model_line(cur), mode="lines", name="guess",
                         line=dict(color=TEAL, width=4)), row=1, col=1)

# Right: static contour, animated path + current dot + gradient arrow.
fig.add_trace(go.Contour(x=gw_grid, y=bw_grid, z=Z, showscale=False,
                         colorscale="Viridis", reversescale=False,
                         contours=dict(coloring="heatmap")), row=1, col=2)
fig.add_trace(go.Scatter(x=ws[:i + 1], y=bs[:i + 1], mode="lines",
                         line=dict(color="white", width=2), showlegend=False),
              row=1, col=2)
fig.add_trace(go.Scatter(x=[cur["w"]], y=[cur["b"]], mode="markers",
                         marker=dict(color="white", size=14,
                                     line=dict(color="black", width=1)),
                         showlegend=False), row=1, col=2)


def arrow(s):
    n = (s["gw"] ** 2 + s["gb"] ** 2) ** 0.5 or 1.0
    return ([s["w"], s["w"] - 0.25 * s["gw"] / n],
            [s["b"], s["b"] - 8.0 * s["gb"] / n])


ax, ay = arrow(cur)
fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines+markers",
                         line=dict(color=TEAL, width=4),
                         marker=dict(color=TEAL, size=[0, 12], symbol="arrow-up"),
                         showlegend=False), row=1, col=2)

# Animation frames update traces 1,2,4,5,6 (data points & contour stay put).
frames = []
for k, s in enumerate(snaps):
    fpx, fpy = miss_segments(s)
    fax, fay = arrow(s)
    frames.append(go.Frame(name=str(k), traces=[1, 2, 4, 5, 6], data=[
        go.Scatter(x=fpx, y=fpy),
        go.Scatter(x=xline, y=model_line(s)),
        go.Scatter(x=ws[:k + 1], y=bs[:k + 1]),
        go.Scatter(x=[s["w"]], y=[s["b"]]),
        go.Scatter(x=fax, y=fay),
    ]))
fig.frames = frames

fig.update_xaxes(title_text="°C", row=1, col=1)
fig.update_yaxes(title_text="°F", row=1, col=1)
fig.update_xaxes(title_text="w", row=1, col=2)
fig.update_yaxes(title_text="b", row=1, col=2)
fig.update_layout(plotly_layout(height=430, showlegend=False,
                                title=dict(text=f"guess = {cur['w']:.2f} × °C + {cur['b']:.1f}",
                                           x=0.25, font=dict(color=TEAL, size=18))))
fig.update_layout(updatemenus=[dict(
    type="buttons", showactive=False, x=0.0, y=1.18, xanchor="left",
    # Start the animation from the first frame so Streamlit's reruns or
    # the slider's current value (often the final snapshot) don't prevent
    # the Play button from showing the full precomputed run.
    buttons=[dict(label="▶ Play", method="animate",
                  args=[None, dict(frame=dict(duration=60, redraw=True),
                                   transition=dict(duration=0), fromcurrent=False)])])])
chart(fig)

# --- Metrics + status at the current snapshot ------------------------------
m = st.columns(4)
m[0].metric("w", f"{cur['w']:.3f}")
m[1].metric("b", f"{cur['b']:.2f}")
m[2].metric("loss", f"{cur['loss']:,.3f}")
m[3].metric("step", f"{cur['step']:,}")

if run["exploded"]:
    st.markdown(callout("Loss exploded: each step overshoots the valley and lands higher up.",
                        "red"), unsafe_allow_html=True)
elif run["converged"] and i == len(snaps) - 1:
    st.markdown(callout(f"Settled after {run['n_steps']:,} nudges: "
                        f"w ≈ {run['final_w']:.2f}, b ≈ {run['final_b']:.1f}. "
                        "Nobody told it the formula.", "teal"), unsafe_allow_html=True)
else:
    st.markdown(callout(f"Still rolling downhill — loss {cur['loss']:,.2f} at step {cur['step']:,}.",
                        "amber"), unsafe_allow_html=True)

reset_button(KEYS, key="p2_reset")

presenter_notes(
    "Ball drops to the valley floor fast, then slides along it.",
    "Now switch to Too big — watch it climb out of the valley.",
    "Change the start point: every start finds the same valley.",
)
