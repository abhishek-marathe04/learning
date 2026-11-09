"""Page 2 — Downhill. Two knobs discover F = 1.8·C + 32 from noisy readings.
Watch the ball roll to the valley floor, and watch a too-big step explode."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import base64
import io
import time
from PIL import Image, ImageFilter
from plotly.colors import sample_colorscale
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

xline = np.linspace(x.min(), x.max(), 50)


def model_line(s):
    return s["w"] * xline + s["b"]


def miss_segments(s):
    px, py = [], []
    for xi, yi in zip(x, y):
        px += [xi, xi, None]
        py += [yi, s["w"] * xi + s["b"], None]
    return px, py


def arrow(s):
    n = (s["gw"] ** 2 + s["gb"] ** 2) ** 0.5 or 1.0
    return ([s["w"], s["w"] - 0.25 * s["gw"] / n],
            [s["b"], s["b"] - 8.0 * s["gb"] / n])


@st.cache_data
def landscape_image(w_rng, b_rng, px=480, levels=12):
    """The loss landscape as a PNG data URI (Viridis fill + dark contour lines).

    It is a background image rather than a Contour trace on purpose: Plotly
    wipes contour/heatmap traces when Play runs an animation, while layout
    images are never touched.
    """
    ws = np.linspace(*w_rng, px)
    bs = np.linspace(*b_rng, px)[::-1]          # row 0 = top = highest b
    W, B = np.meshgrid(ws, bs)
    resid = W[..., None] * x + B[..., None] - y
    Z = np.log10(np.mean(resid ** 2, axis=-1) + 1e-9)

    t = (Z - Z.min()) / (Z.max() - Z.min())
    lut = np.array([[int(c) for c in rgb[4:-1].split(",")]
                    for rgb in sample_colorscale("Viridis", np.linspace(0, 1, 256))],
                   dtype=np.uint8)
    rgb = lut[(t * 255).astype(int)].astype(float)

    # Thin iso-lines: pixels within ~0.7 px of an evenly spaced level.
    step = (Z.max() - Z.min()) / levels
    gy, gx = np.gradient(Z)
    dist = np.abs(((Z - Z.min()) / step + 0.5) % 1 - 0.5) * step
    on_line = dist / (np.hypot(gx, gy) + 1e-12) < 0.7
    rgb[on_line] *= 0.35

    img = Image.fromarray(rgb.astype(np.uint8)).filter(ImageFilter.SMOOTH)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pick_frames(ws, bs, w_rng, b_rng, n_frames=60, max_jump=0.03):
    """Snapshot indices for the Play animation, spaced by on-screen distance.

    Snapshots are evenly spaced in training steps, but the ball's speed is
    not: it jumps to the valley floor, then crawls toward the minimum. Spacing
    frames evenly by step made long stretches where nothing visibly moved,
    which looked like the animation had paused. Instead, sample evenly along
    the path length (in axis-fraction units). Single jumps are capped at
    `max_jump` so one huge leap does not eat most of the frame budget.
    """
    n = len(ws)
    if n <= n_frames:
        return list(range(n))
    dw = np.diff(ws) / (w_rng[1] - w_rng[0])
    db = np.diff(bs) / (b_rng[1] - b_rng[0])
    seg = np.minimum(np.hypot(dw, db), max_jump)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], n_frames)
    idx = np.searchsorted(cum, targets, side="left")
    return sorted(set(int(i) for i in np.clip(idx, 0, n - 1)) | {0, n - 1})


@st.cache_data
def build_combined_figure(snaps_data):
    gw_grid, bw_grid, Z = cached_loss_surface()
    # Seed every animated trace with the first snapshot. Plotly cannot
    # animate from an empty trace, so empty starts made traces vanish.
    s0 = snaps_data[0]
    n0 = len(snaps_data)
    m0x, m0y = miss_segments(s0)
    a0x, a0y = arrow(s0)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=("Data and guess", "Loss landscape"),
                        horizontal_spacing=0.09)

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="readings",
                             marker=dict(color=INK, size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=m0x, y=m0y, mode="lines", name="miss",
                             line=dict(color=AMBER, width=1, dash="dot"),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xline, y=model_line(s0), mode="lines", name="guess",
                             line=dict(color=TEAL, width=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=[s0["w"]] * n0, y=[s0["b"]] * n0, mode="lines",
                             line=dict(color="white", width=2), showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[s0["w"]], y=[s0["b"]], mode="markers",
                             marker=dict(color="white", size=14,
                                         line=dict(color="black", width=1)),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=a0x, y=a0y, mode="lines+markers",
                             line=dict(color=TEAL, width=4),
                             marker=dict(color=TEAL, size=[0, 12], symbol="arrow-up"),
                             showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="°F", row=1, col=1)
    fig.update_xaxes(title_text="w", row=1, col=2)
    fig.update_yaxes(title_text="b", row=1, col=2)
    fig.update_layout(plotly_layout(height=430, showlegend=False))

    # Landscape as a background image pinned to the w/b axes (see
    # landscape_image for why it is not a trace).
    w_rng = (float(gw_grid.min()), float(gw_grid.max()))
    b_rng = (float(bw_grid.min()), float(bw_grid.max()))
    fig.update_xaxes(range=w_rng, autorange=False, showgrid=False,
                     zeroline=False, row=1, col=2)
    fig.update_yaxes(range=b_rng, autorange=False, showgrid=False,
                     zeroline=False, row=1, col=2)
    fig.add_layout_image(source=landscape_image(w_rng, b_rng),
                         xref="x2", yref="y2", x=w_rng[0], y=b_rng[1],
                         sizex=w_rng[1] - w_rng[0], sizey=b_rng[1] - b_rng[0],
                         sizing="stretch", layer="below")

    # Animation frames.
    # Smoothness choices (the old version blinked):
    #  * redraw=False: only the scatter traces move, nothing is re-rendered
    #    from scratch, so there is no flash between frames.
    #  * ~60 frames picked by distance travelled (pick_frames), so the ball
    #    moves at an even pace instead of idling where it barely moves.
    #  * The trail is padded to a fixed length so Plotly can interpolate it.
    n = len(snaps_data)
    snaps_ws = np.array([s["w"] for s in snaps_data])
    snaps_bs = np.array([s["b"] for s in snaps_data])
    frame_idx = pick_frames(snaps_ws, snaps_bs, w_rng, b_rng)

    frames = []
    for k in frame_idx:
        s = snaps_data[k]
        fpx, fpy = miss_segments(s)
        fax, fay = arrow(s)
        trail_w = np.concatenate([snaps_ws[:k + 1], np.full(n - k - 1, snaps_ws[k])])
        trail_b = np.concatenate([snaps_bs[:k + 1], np.full(n - k - 1, snaps_bs[k])])
        frames.append(go.Frame(name=str(k), traces=[1, 2, 3, 4, 5], data=[
            go.Scatter(x=fpx, y=fpy),
            go.Scatter(x=xline, y=model_line(s)),
            go.Scatter(x=trail_w, y=trail_b),
            go.Scatter(x=[s["w"]], y=[s["b"]]),
            go.Scatter(x=fax, y=fay),
        ]))
    fig.frames = frames

    play_button = dict(
        type="buttons", showactive=False, x=0.0, y=1.18, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=160, redraw=False),
                                       transition=dict(duration=160, easing="linear"),
                                       mode="immediate", fromcurrent=False)])])
    fig.update_layout(updatemenus=[play_button])
    return fig


fig = build_combined_figure(tuple(snaps))
final = snaps[-1]
fig.update_layout(title=dict(text=f"guess = {final['w']:.2f} × °C + {final['b']:.1f}",
                             x=0.25, font=dict(color=TEAL, size=18)))
chart(fig, key="downhill")

# Play animation
col1, col2 = st.columns([0.8, 10])
if col1.button("▶", help="Play"):
    for step in range(len(snaps)):
        st.session_state.p2_step = step
        time.sleep(0.05)

# Slider controls which snapshot to display
step_idx = st.slider("Training step", 0, len(snaps) - 1,
                     value=st.session_state.get("p2_step", 0),
                     format="snapshot %d", key="p2_step",
                     help="Drag to scrub through training steps")

# --- Metrics + status at the selected snapshot ------------------------------
cur = snaps[step_idx]
m = st.columns(4)
m[0].metric("w", f"{cur['w']:.3f}")
m[1].metric("b", f"{cur['b']:.2f}")
m[2].metric("loss", f"{cur['loss']:,.3f}")
m[3].metric("step", f"{cur['step']:,}")

if run["exploded"]:
    st.markdown(callout("Loss exploded: each step overshoots the valley and lands higher up.",
                        "red"), unsafe_allow_html=True)
elif run["converged"]:
    st.markdown(callout(f"Settled after {run['n_steps']:,} nudges: "
                        f"w ≈ {run['final_w']:.2f}, b ≈ {run['final_b']:.1f}. "
                        "Nobody told it the formula.", "teal"), unsafe_allow_html=True)

reset_button(KEYS, key="p2_reset")

presenter_notes(
    "Ball drops to the valley floor fast, then slides along it.",
    "Now switch to Too big — watch it climb out of the valley.",
    "Change the start point: every start finds the same valley.",
)
