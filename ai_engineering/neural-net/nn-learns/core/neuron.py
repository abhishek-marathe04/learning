"""Page 1: the one-neuron model, the wiggle test, and autograd slopes.

Also holds page 2's two-knob gradient-descent trajectory over the temperature
data (still a one-neuron model, just many data points and many steps).
"""

import numpy as np
import torch

# The single training example used on page 1.
X_DEFAULT = 10.0
Y_DEFAULT = 50.0
WIGGLE_EPS = 0.01
NUDGE_LR = 0.0005


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to the inclusive range [low, high]."""
    return min(max(value, low), high)


def forward(w: float, b: float, x: float = X_DEFAULT, y: float = Y_DEFAULT):
    """Return (guess, miss, loss) for the one-neuron model."""
    guess = w * x + b
    miss = guess - y
    loss = miss ** 2
    return guess, miss, loss


def loss_of(w: float, b: float, x: float = X_DEFAULT, y: float = Y_DEFAULT) -> float:
    return forward(w, b, x, y)[2]


def wiggle_test(w: float, b: float, x: float = X_DEFAULT, y: float = Y_DEFAULT,
                eps: float = WIGGLE_EPS):
    """Nudge each knob by +eps, measure the change in loss, estimate the slope.

    Returns a list of dicts, one per knob, with the numbers shown in the table.
    """
    base_loss = loss_of(w, b, x, y)
    rows = []
    for knob, (dw, db) in (("w", (eps, 0.0)), ("b", (0.0, eps))):
        new_guess, _, new_loss = forward(w + dw, b + db, x, y)
        change = new_loss - base_loss
        rows.append({
            "knob": knob,
            "new_guess": new_guess,
            "new_loss": new_loss,
            "change": change,
            "slope": change / eps,
        })
    return rows


def autograd_grads(w: float, b: float, x: float = X_DEFAULT, y: float = Y_DEFAULT):
    """Real PyTorch backward pass. Returns (w_grad, b_grad) as floats."""
    wt = torch.tensor(float(w), requires_grad=True)
    bt = torch.tensor(float(b), requires_grad=True)
    loss = (wt * x + bt - y) ** 2
    loss.backward()
    return wt.grad.item(), bt.grad.item()


def one_nudge(w: float, b: float, x: float = X_DEFAULT, y: float = Y_DEFAULT,
              lr: float = NUDGE_LR):
    """Take one gradient-descent step. Returns (new_w, new_b, new_loss)."""
    gw, gb = autograd_grads(w, b, x, y)
    new_w = clamp(w - lr * gw, 0.0, 3.0)
    new_b = clamp(b - lr * gb, -40.0, 80.0)
    return new_w, new_b, loss_of(new_w, new_b, x, y)


# --- Page 2: gradient descent over the temperature data --------------------
# Step-size presets shown via st.segmented_control.
LR_PRESETS = {"Tiny": 0.00015, "Good": 0.0012, "Too big": 0.0025}
W_START, B_START = 0.5, 0.0
MAX_STEPS = 8000
GRAD_STOP = 0.01
EXPLODE_LOSS = 1e6


def _mse_and_grads(w, b, x, y):
    pred = w * x + b
    resid = pred - y
    loss = float(np.mean(resid ** 2))
    gw = float(np.mean(2 * resid * x))
    gb = float(np.mean(2 * resid))
    return loss, gw, gb


def _snapshot_indices(n_recorded: int, dense: int = 30, spread: int = 150):
    """First `dense` steps individually, then ~`spread` points over the rest."""
    if n_recorded <= dense:
        return list(range(n_recorded))
    idx = list(range(dense))
    rest = np.linspace(dense, n_recorded - 1, spread)
    idx += [int(round(i)) for i in rest]
    # de-dup while keeping order
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def gd_trajectory(lr: float, x, y, max_steps: int = MAX_STEPS,
                  w0: float = W_START, b0: float = B_START):
    """Full-batch gradient descent. Returns a dict of snapshots and status.

    Stops early when the gradient norm falls below GRAD_STOP (converged) or the
    loss exceeds EXPLODE_LOSS (exploded).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ws, bs, losses, gws, gbs = [], [], [], [], []
    w, b = float(w0), float(b0)
    exploded = converged = False
    explode_step = None

    for step in range(max_steps + 1):
        loss, gw, gb = _mse_and_grads(w, b, x, y)
        ws.append(w); bs.append(b); losses.append(loss)
        gws.append(gw); gbs.append(gb)
        if loss > EXPLODE_LOSS or not np.isfinite(loss):
            exploded = True
            explode_step = step
            break
        if (gw ** 2 + gb ** 2) ** 0.5 < GRAD_STOP:
            converged = True
            break
        w -= lr * gw
        b -= lr * gb

    n = len(ws)
    snaps = []
    for i in _snapshot_indices(n):
        snaps.append({
            "step": i, "w": ws[i], "b": bs[i], "loss": losses[i],
            "gw": gws[i], "gb": gbs[i],
        })
    return {
        "snaps": snaps,
        "exploded": exploded,
        "converged": converged,
        "explode_step": explode_step,
        "n_steps": n - 1,
        "final_w": ws[-1], "final_b": bs[-1], "final_loss": losses[-1],
        "lr": lr,
    }


def loss_surface(x, y, w_range=(0.0, 3.2), b_range=(-40.0, 80.0), n=60):
    """Grid of log(MSE + eps) over (w, b) for the landscape contour."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ws = np.linspace(*w_range, n)
    bs = np.linspace(*b_range, n)
    W, B = np.meshgrid(ws, bs)
    Z = np.empty_like(W)
    for i in range(n):
        for j in range(n):
            pred = W[i, j] * x + B[i, j]
            Z[i, j] = np.mean((pred - y) ** 2)
    return ws, bs, np.log10(Z + 1e-9)
