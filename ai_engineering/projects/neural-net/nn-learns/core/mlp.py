"""Page 4: a 1-H-1 ReLU MLP trained with Adam, stored as snapshots.

    y = sum_j v_j * ReLU(w_j * x + b_j) + c

One hidden unit draws one hinge. Enough hinges, summed, trace any shape.
"""

import numpy as np
import torch

HIDDEN_SIZES = [1, 2, 4, 8, 16, 32]
ADAM_LR = 0.02
STEPS = 3000
SNAP_EVERY = 25


def _init_params(H: int):
    """Deterministic init: torch seed 300 + H."""
    torch.manual_seed(300 + H)
    w = torch.randn(H)                                   # N(0, 1)
    b = torch.empty(H).uniform_(-2, 2)                  # U(-2, 2)
    v = torch.randn(H) * (0.5 / (H ** 0.5))            # N(0, 0.5/sqrt(H))
    c = torch.zeros(1)
    return w, b, v, c


def _forward(x, w, b, v, c):
    # x: (N,)  w,b,v: (H,)  ->  (N,)
    pre = x[:, None] * w[None, :] + b[None, :]          # (N, H)
    hidden = torch.relu(pre)
    return (hidden * v[None, :]).sum(dim=1) + c


def train_mlp(dataset: str, H: int, steps: int = STEPS, snap_every: int = SNAP_EVERY):
    """Train the MLP on a curve dataset. Returns a dict of snapshots.

    Each snapshot stores the parameters (numpy) and the loss, so the page can
    redraw the network output and every neuron's contribution at that step.
    """
    from core.data import curve_data

    xn, yn = curve_data(dataset)
    x = torch.tensor(xn, dtype=torch.float32)
    y = torch.tensor(yn, dtype=torch.float32)

    w, b, v, c = _init_params(H)
    for p in (w, b, v, c):
        p.requires_grad_(True)
    opt = torch.optim.Adam([w, b, v, c], lr=ADAM_LR)

    snaps = []

    def record(step):
        with torch.no_grad():
            loss = torch.mean((_forward(x, w, b, v, c) - y) ** 2).item()
        snaps.append({
            "step": step,
            "w": w.detach().numpy().copy(),
            "b": b.detach().numpy().copy(),
            "v": v.detach().numpy().copy(),
            "c": float(c.detach().item()),
            "loss": loss,
        })

    record(0)
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = torch.mean((_forward(x, w, b, v, c) - y) ** 2)
        loss.backward()
        opt.step()
        if step % snap_every == 0 or step == steps:
            record(step)

    return {
        "snaps": snaps,
        "H": H,
        "dataset": dataset,
        "n_knobs": 3 * H + 1,
        "final_loss": snaps[-1]["loss"],
        "x": xn,
        "y": yn,
    }


def eval_curve(snap, x_grid):
    """Network output over an x grid for one snapshot. Returns numpy array."""
    xg = np.asarray(x_grid)[:, None]
    pre = xg * snap["w"][None, :] + snap["b"][None, :]
    hidden = np.maximum(0.0, pre)
    return (hidden * snap["v"][None, :]).sum(axis=1) + snap["c"]


def neuron_pieces(snap, x_grid):
    """Each hidden neuron's contribution v_j*ReLU(w_j*x+b_j) over the grid.

    Returns (pieces, bends): pieces is (H, len(x_grid)); bends is the list of
    bend x-locations -b_j/w_j (None where w_j == 0).
    """
    xg = np.asarray(x_grid)[:, None]
    pre = xg * snap["w"][None, :] + snap["b"][None, :]
    hidden = np.maximum(0.0, pre)
    pieces = (hidden * snap["v"][None, :]).T            # (H, N)
    bends = []
    for wj, bj in zip(snap["w"], snap["b"]):
        bends.append(None if wj == 0 else -bj / wj)
    return pieces, bends
