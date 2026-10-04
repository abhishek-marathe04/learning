"""Page 3: build the computation graph and walk it backwards, link by link.

The graph is the one-neuron loss from page 1:

    w --x-->  guess = w*x + b  --(-y)-->  miss = guess - y  --square-->  loss
    b ---------^

Backward, each link multiplies its own local rate. Stepping through the links
reveals the chain rule as a running product.
"""

import math


def build_graph(w: float, b: float, x: float = 10.0, y: float = 50.0):
    """Return (nodes, backward_steps).

    nodes: name -> dict(value=..., grad=..., row=...). grad is the final value;
    the page reveals it progressively using backward_steps.

    backward_steps: ordered list, one per link, each a dict describing the link
    being processed and the node gradient it fills in.
    """
    guess = w * x + b
    miss = guess - y
    loss = miss ** 2

    nodes = {
        "w": {"value": w, "grad": None, "row": 0, "label": "w"},
        "b": {"value": b, "grad": None, "row": 1, "label": "b"},
        "guess": {"value": guess, "grad": None, "row": 0, "label": "guess = w·x + b"},
        "miss": {"value": miss, "grad": None, "row": 0, "label": "miss = guess − y"},
        "loss": {"value": loss, "grad": 1.0, "row": 0, "label": "loss = miss²"},
    }

    square_rate = 2 * miss          # d loss / d miss
    miny_rate = 1.0                 # d miss / d guess
    timesx_rate = x                 # d guess / d w
    plusb_rate = 1.0                # d guess / d b

    miss_grad = 1.0 * square_rate
    guess_grad = miss_grad * miny_rate
    w_grad = guess_grad * timesx_rate
    b_grad = guess_grad * plusb_rate

    def fmt(v):
        return f"{v:g}"

    steps = [
        {
            "link": "square",
            "edge": ("miss", "loss"),
            "rate": square_rate,
            "rate_label": f"2 × miss = {fmt(square_rate)}",
            "sets_node": "miss",
            "grad": miss_grad,
            "product": f"1 × {fmt(square_rate)} = {fmt(miss_grad)}",
        },
        {
            "link": "−y",
            "edge": ("guess", "miss"),
            "rate": miny_rate,
            "rate_label": "1",
            "sets_node": "guess",
            "grad": guess_grad,
            "product": f"{fmt(miss_grad)} × {fmt(miny_rate)} = {fmt(guess_grad)}",
        },
        {
            "link": "×x",
            "edge": ("w", "guess"),
            "rate": timesx_rate,
            "rate_label": f"x = {fmt(timesx_rate)}",
            "sets_node": "w",
            "grad": w_grad,
            "product": f"{fmt(square_rate)} × {fmt(miny_rate)} × {fmt(timesx_rate)} = {fmt(w_grad)}",
        },
        {
            "link": "+b",
            "edge": ("b", "guess"),
            "rate": plusb_rate,
            "rate_label": "1",
            "sets_node": "b",
            "grad": b_grad,
            "product": f"{fmt(square_rate)} × {fmt(miny_rate)} × {fmt(plusb_rate)} = {fmt(b_grad)}",
        },
    ]
    return nodes, steps


# --- Stretch goal: a two-layer chain --------------------------------------
def build_two_layer(w1=0.5, b1=0.2, w2=1.5, x=1.0, y=2.0):
    """loss = (tanh(w1*x + b1) * w2 - y)**2. Returns (nodes, backward_steps).

    Demonstrates the same backward walk on a longer chain (with a tanh link).
    """
    a = w1 * x + b1
    h = math.tanh(a)
    o = h * w2
    miss = o - y
    loss = miss ** 2

    def fmt(v):
        return f"{v:.4g}"

    nodes = {
        "w1": {"value": w1, "grad": None, "row": 0, "label": "w1"},
        "b1": {"value": b1, "grad": None, "row": 1, "label": "b1"},
        "a": {"value": a, "grad": None, "row": 0, "label": "a = w1·x + b1"},
        "h": {"value": h, "grad": None, "row": 0, "label": "h = tanh(a)"},
        "w2": {"value": w2, "grad": None, "row": 1, "label": "w2"},
        "o": {"value": o, "grad": None, "row": 0, "label": "o = h·w2"},
        "miss": {"value": miss, "grad": None, "row": 0, "label": "miss = o − y"},
        "loss": {"value": loss, "grad": 1.0, "row": 0, "label": "loss = miss²"},
    }

    r_square = 2 * miss                 # d loss / d miss
    r_miny = 1.0                        # d miss / d o
    r_h = w2                            # d o / d h
    r_w2 = h                            # d o / d w2
    r_tanh = 1 - h ** 2                 # d h / d a
    r_w1 = x                            # d a / d w1
    r_b1 = 1.0                          # d a / d b1

    g_miss = r_square
    g_o = g_miss * r_miny
    g_h = g_o * r_h
    g_w2 = g_o * r_w2
    g_a = g_h * r_tanh
    g_w1 = g_a * r_w1
    g_b1 = g_a * r_b1

    steps = [
        {"link": "square", "edge": ("miss", "loss"), "rate": r_square,
         "rate_label": f"2 × miss = {fmt(r_square)}", "sets_node": "miss",
         "grad": g_miss, "product": f"1 × {fmt(r_square)} = {fmt(g_miss)}"},
        {"link": "−y", "edge": ("o", "miss"), "rate": r_miny, "rate_label": "1",
         "sets_node": "o", "grad": g_o, "product": f"{fmt(g_miss)} × 1 = {fmt(g_o)}"},
        {"link": "·w2", "edge": ("h", "o"), "rate": r_h, "rate_label": f"w2 = {fmt(r_h)}",
         "sets_node": "h", "grad": g_h, "product": f"{fmt(g_o)} × {fmt(r_h)} = {fmt(g_h)}"},
        {"link": "·h", "edge": ("w2", "o"), "rate": r_w2, "rate_label": f"h = {fmt(r_w2)}",
         "sets_node": "w2", "grad": g_w2, "product": f"{fmt(g_o)} × {fmt(r_w2)} = {fmt(g_w2)}"},
        {"link": "tanh", "edge": ("a", "h"), "rate": r_tanh,
         "rate_label": f"1 − h² = {fmt(r_tanh)}", "sets_node": "a", "grad": g_a,
         "product": f"{fmt(g_h)} × {fmt(r_tanh)} = {fmt(g_a)}"},
        {"link": "×x", "edge": ("w1", "a"), "rate": r_w1, "rate_label": f"x = {fmt(r_w1)}",
         "sets_node": "w1", "grad": g_w1, "product": f"{fmt(g_a)} × {fmt(r_w1)} = {fmt(g_w1)}"},
        {"link": "+b1", "edge": ("b1", "a"), "rate": r_b1, "rate_label": "1",
         "sets_node": "b1", "grad": g_b1, "product": f"{fmt(g_a)} × 1 = {fmt(g_b1)}"},
    ]
    return nodes, steps


def torch_check(w: float, b: float, x: float = 10.0, y: float = 50.0):
    """Confirm the graph gradients with a real autograd pass."""
    import torch
    wt = torch.tensor(float(w), requires_grad=True)
    bt = torch.tensor(float(b), requires_grad=True)
    loss = (wt * x + bt - y) ** 2
    loss.backward()
    return wt.grad.item(), bt.grad.item()
