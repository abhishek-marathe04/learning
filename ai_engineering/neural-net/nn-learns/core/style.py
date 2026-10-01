"""Shared palette, Plotly template and CSS for the whole app.

The colours here must match the slide deck. Two of them carry meaning that is
used consistently on every page:
  TEAL  -> "knobs": weights, the model's line, nudges, gradient arrows
  AMBER -> "loss": errors, misses, loss curves, surprise
"""

# --- Palette (section 4 of the spec) ---------------------------------------
NAVY = "#151A33"    # page background
PANEL = "#1D2344"   # cards, chart backgrounds
INK = "#ECEEF8"     # text, data points
MUTED = "#9AA2C8"   # labels, captions
TEAL = "#3CC6B0"    # knobs
AMBER = "#F2A541"   # loss
RED = "#EF6F6C"     # failure states only

# A small ordered palette for the per-neuron "pieces" on page 4.
PIECES = [
    "#3CC6B0", "#F2A541", "#8AB4F8", "#EF6F6C", "#B39DDB", "#7FD1AE",
    "#F6C177", "#9AA2C8", "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
]


def plotly_layout(**overrides):
    """Return a dict of Plotly layout settings for the dark theme.

    Transparent backgrounds, no grid clutter, readable fonts.
    """
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=15),
        margin=dict(l=50, r=20, t=40, b=45),
        xaxis=dict(gridcolor="rgba(154,162,200,0.15)", zerolinecolor="rgba(154,162,200,0.3)"),
        yaxis=dict(gridcolor="rgba(154,162,200,0.15)", zerolinecolor="rgba(154,162,200,0.3)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        colorway=PIECES,
    )
    layout.update(overrides)
    return layout


# Plotly config passed to st.plotly_chart on every chart.
PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}


APP_CSS = f"""
<style>
    /* Bigger, projector-friendly base type. */
    html, body, [class*="css"] {{
        font-size: 18px;
    }}
    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    /* Large custom stat blocks. */
    .stat {{
        background: {PANEL};
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 10px;
    }}
    .stat .label {{
        color: {MUTED};
        font-size: 15px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .stat .value {{
        font-size: 52px;
        font-weight: 700;
        line-height: 1.1;
    }}
    .stat .value.loss {{ color: {AMBER}; }}
    .stat .value.knob {{ color: {TEAL}; }}
    .stat .value.ink  {{ color: {INK}; }}
    .callout {{
        border-left: 4px solid {TEAL};
        background: rgba(60,198,176,0.10);
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 17px;
    }}
    .callout.amber {{ border-color: {AMBER}; background: rgba(242,165,65,0.10); }}
    .callout.red   {{ border-color: {RED};   background: rgba(239,111,108,0.12); }}
    /* Metric numbers a little larger. */
    [data-testid="stMetricValue"] {{ font-size: 40px; }}
</style>
"""


def stat_block(label: str, value: str, kind: str = "ink") -> str:
    """Return HTML for a large stat block. kind in {loss, knob, ink}."""
    return (
        f'<div class="stat"><div class="label">{label}</div>'
        f'<div class="value {kind}">{value}</div></div>'
    )


def callout(text: str, kind: str = "teal") -> str:
    """Return HTML for a coloured callout box. kind in {teal, amber, red}."""
    cls = "callout" if kind == "teal" else f"callout {kind}"
    return f'<div class="{cls}">{text}</div>'
