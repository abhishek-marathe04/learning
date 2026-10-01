"""Page 5: a tiny bigram "language model" — one 27x27 weight matrix.

    logits = one_hot(current) @ W          # W[current] is the row of logits
    loss   = cross_entropy(logits, next)   # mean surprise, in nats

Same training loop as every other page; here the loss is surprise.
"""

import numpy as np
import torch
import torch.nn.functional as F

from core.data import CHARS, ITOS, STOI, bigram_pairs

VOCAB = len(CHARS)                     # 27
LR = 30.0
STEPS = 400
SNAP_STEPS = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 400]
UNIFORM_LOSS = float(np.log(VOCAB))    # 3.2958..., "knows nothing"


def train_bigram(steps: int = STEPS, snap_steps=None):
    """Full-batch gradient descent on the 27x27 matrix. Returns snapshots.

    Each snapshot stores W (numpy) and the loss at that step.
    """
    if snap_steps is None:
        snap_steps = SNAP_STEPS
    snap_set = set(snap_steps)

    xs_np, ys_np = bigram_pairs()
    xs = torch.tensor(xs_np, dtype=torch.long)
    ys = torch.tensor(ys_np, dtype=torch.long)

    W = torch.zeros((VOCAB, VOCAB), requires_grad=True)
    snaps = []

    for step in range(steps + 1):
        logits = W[xs]                                  # one_hot(xs) @ W
        loss = F.cross_entropy(logits, ys)
        if step in snap_set:
            snaps.append({
                "step": step,
                "W": W.detach().numpy().copy(),
                "loss": float(loss.item()),
            })
        if step == steps:
            break
        W.grad = None
        loss.backward()
        with torch.no_grad():
            W -= LR * W.grad

    return {"snaps": snaps, "uniform_loss": UNIFORM_LOSS,
            "optimum": bigram_optimum(), "final_loss": snaps[-1]["loss"]}


def softmax_rows(W: np.ndarray) -> np.ndarray:
    """Row-wise softmax of the weight matrix (probability of next letter)."""
    z = W - W.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sample_names(W: np.ndarray, n: int = 10, seed: int = 42, cap: int = 12):
    """Draw n names from the model at this snapshot. Fixed seed each call, so
    differences between snapshots reflect learning, not luck."""
    probs = softmax_rows(W)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = STOI["."]
        chars = []
        while True:
            idx = int(rng.choice(VOCAB, p=probs[idx]))
            if idx == STOI["."]:
                break
            chars.append(ITOS[idx])
            if len(chars) >= cap:
                chars.append("…")
                break
        out.append("".join(chars))
    return out


def surprise_of(W: np.ndarray, name: str):
    """Per-letter surprise (-ln p) for the transitions in '.name.'.

    Returns (rows, average) where each row is dict(to, p, surprise).
    """
    probs = softmax_rows(W)
    name = "".join(ch for ch in name.lower() if ch in STOI)
    chs = ["."] + list(name) + ["."]
    rows = []
    for a, b in zip(chs, chs[1:]):
        p = float(probs[STOI[a], STOI[b]])
        rows.append({"to": b, "p": p, "surprise": float(-np.log(max(p, 1e-12)))})
    avg = float(np.mean([r["surprise"] for r in rows])) if rows else 0.0
    return rows, avg


def bigram_optimum() -> float:
    """Exact best cross-entropy for a one-letter-memory model, from counts."""
    xs, ys = bigram_pairs()
    counts = np.zeros((VOCAB, VOCAB))
    for a, b in zip(xs, ys):
        counts[a, b] += 1
    row_tot = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_tot, out=np.zeros_like(counts), where=row_tot > 0)
    total = 0.0
    for a, b in zip(xs, ys):
        total += -np.log(max(probs[a, b], 1e-12))
    return float(total / len(xs))
