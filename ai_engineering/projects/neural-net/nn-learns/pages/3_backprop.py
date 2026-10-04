"""Page 3 — Backprop, one link at a time. Step backwards through the graph and
multiply local rates. It's exactly what autograd does."""

import streamlit as st

from core import graph, neuron
from core.style import AMBER, INK, MUTED, TEAL, callout
from ui import presenter_notes, reset_button

KEYS = ["p3_k", "p3_example", "p3_compare"]
st.session_state.setdefault("p3_k", 0)

st.title("Backprop, one link at a time")
st.caption("Each link only knows its own local rate. Multiply backwards — that is all backprop is.")

example = st.selectbox("Example", ["One neuron (page 1)", "Two layers (tanh)"],
                       key="p3_example")
if example.startswith("One"):
    nodes, steps = graph.build_graph(1.0, 0.0)
else:
    nodes, steps = graph.build_two_layer()

k = min(st.session_state.p3_k, len(steps))

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
if c1.button("◀ Backward one link", use_container_width=True):
    st.session_state.p3_k = min(k + 1, len(steps)); st.rerun()
if c2.button("⏪ Backward all", use_container_width=True):
    st.session_state.p3_k = len(steps); st.rerun()
with c3:
    reset_button(KEYS, key="p3_reset")
compare = c4.toggle("Compare with PyTorch", key="p3_compare")

# Which node gradients are revealed after k backward links.
grads = {"loss": nodes["loss"]["grad"]}
for s in steps[:k]:
    grads[s["sets_node"]] = s["grad"]
active_edge = steps[k - 1]["edge"] if k > 0 else None
edge_rates = {s["edge"]: s["rate_label"] for s in steps[:k]}


def fmt(v):
    return f"{v:g}"


def node_dot(name, spec):
    val = fmt(spec["value"])
    grad_line = ""
    if name in grads:
        grad_line = f'<br/><font color="{AMBER}"><b>grad {fmt(grads[name])}</b></font>'
    color = TEAL if name in ("w", "b", "w1", "b1", "w2") else INK
    label = f'<<b>{spec["label"]}</b><br/><font color="{MUTED}">{val}</font>{grad_line}>'
    return f'  {name} [label={label}, color="{color}"];'


lines = ['digraph {', '  rankdir=LR; bgcolor="transparent";',
         '  node [shape=box style=rounded fontname="Helvetica" fontcolor="#ECEEF8" fontsize=14];',
         '  edge [fontname="Helvetica" fontsize=13];']
for name, spec in nodes.items():
    lines.append(node_dot(name, spec))
for s in steps:
    frm, to = s["edge"]
    is_active = s["edge"] == active_edge
    revealed = s["edge"] in edge_rates
    color = AMBER if revealed else "#9AA2C8"
    penwidth = 3 if is_active else (2 if revealed else 1)
    lbl = edge_rates.get(s["edge"], s["link"])
    fontcolor = AMBER if revealed else "#9AA2C8"
    lines.append(f'  {frm} -> {to} [label="{lbl}", color="{color}", '
                 f'penwidth={penwidth}, fontcolor="{fontcolor}"];')
lines.append("}")
st.graphviz_chart("\n".join(lines), use_container_width=True)

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Running multiplication")
    if k == 0:
        st.caption("Press “Backward one link”. The loss seeds its own gradient at 1.")
    for s in steps[:k]:
        st.markdown(f"**{s['link']}** &nbsp; local rate {s['rate_label']} "
                    f"→ `{s['product']}`")
    if k == len(steps):
        st.markdown(callout("Backward walk complete — every knob now has its slope.",
                            "teal"), unsafe_allow_html=True)

with right:
    if compare:
        st.subheader("Compare with PyTorch")
        if example.startswith("One"):
            gw, gb = graph.torch_check(1.0, 0.0)
            wig = {r["knob"]: r["slope"] for r in neuron.wiggle_test(1.0, 0.0)}
            st.code(f"w.grad = {gw:.1f}\nb.grad = {gb:.1f}", language="text")
            st.markdown(callout(f"Wiggle test (page 1): w ≈ {wig['w']:.0f}, b ≈ {wig['b']:.0f}. "
                                "Same numbers, three ways.", "teal"), unsafe_allow_html=True)
        else:
            st.caption("Autograd matches the hand-walked chain above, link for link.")

presenter_notes(
    "Each link only knows its own local rate. Multiply backwards.",
    "That is all backprop is.",
    "Karpathy's micrograd is ~100 lines of exactly this.",
)
