# Parameter Setup — All Weight Matrices

> **Tags:** `#parameters` `#weights` `#embeddings` `#attention` `#mlp` `#state-dict`

---

## Full initialization code

This runs **once, before any training**:

```python
n_embd = 16     # embedding dimension — size of every hidden vector
n_head = 4      # number of attention heads
n_layer = 1     # number of transformer layers
block_size = 16 # maximum sequence length
head_dim = n_embd // n_head  # 16 / 4 = 4 dims per head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),    # [27, 16]
    'wpe': matrix(block_size, n_embd),    # [16, 16]
    'lm_head': matrix(vocab_size, n_embd) # [27, 16]
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)      # [16, 16]
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)      # [16, 16]
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)      # [16, 16]
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)      # [16, 16]
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # [64, 16]
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # [16, 64]

params = [p for mat in state_dict.values() for row in mat for p in row]
# total: 4,192 parameters
```

Every matrix is initialized with small random numbers (`random.gauss(0, 0.08)`). Training will shape them into something meaningful. They all start knowing nothing — just noise.

---

## Acronym glossary — what every name means

| Name | Stands for | Shape | Role |
|------|-----------|-------|------|
| `wte` | **W**ord **T**oken **E**mbedding | [27, 16] | Lookup table: token id → 16-dim vector ("what am I?") |
| `wpe` | **W**ord **P**osition **E**mbedding | [16, 16] | Lookup table: position id → 16-dim vector ("where am I?") |
| `attn_wq` | **Attn** **W**eight **Q**uery | [16, 16] | Projects x into query space ("what am I looking for?") |
| `attn_wk` | **Attn** **W**eight **K**ey | [16, 16] | Projects x into key space ("what do I contain?") |
| `attn_wv` | **Attn** **W**eight **V**alue | [16, 16] | Projects x into value space ("what will I give if selected?") |
| `attn_wo` | **Attn** **W**eight **O**utput | [16, 16] | Mixes all heads' outputs back into one vector |
| `mlp_fc1` | **MLP** **F**ully **C**onnected layer **1** | [64, 16] | Expands 16 → 64 (wide thinking space) |
| `mlp_fc2` | **MLP** **F**ully **C**onnected layer **2** | [16, 64] | Compresses 64 → 16 (back to highway) |
| `lm_head` | **L**anguage **M**odel **Head** | [27, 16] | Final projection: 16-dim → 27 scores (one per token) |

---

## What is `fc` — Fully Connected?

`fc` = **Fully Connected**. Means every input neuron connects to every output neuron — nothing is skipped. In code, `linear(x, w)` IS a fully connected layer. `fc` and `linear` are the same thing, just different names from different traditions (PyTorch calls it `nn.Linear`, older literature calls it `fc`).

```
input  = [x0, x1, x2]           # 3 input neurons
output = [y0, y1, y2, y3]       # 4 output neurons

y0 = w00*x0 + w01*x1 + w02*x2
y1 = w10*x0 + w11*x1 + w12*x2  ← every input connects to every output
y2 = w20*x0 + w21*x1 + w22*x2
y3 = w30*x0 + w31*x1 + w32*x2
```

---

## Are `mlp_fc1` and `mlp_fc2` hidden layers?

Yes. The classic neural network picture maps directly:

```
input layer  →  hidden layer  →  output layer
x (16-dim)   →  fc1 (64-dim)  →  fc2 (16-dim)
```

The 64-dim middle is the **hidden layer** — "hidden" because you never directly see or interpret those 64 numbers. They are internal working space. After `fc1` expands to 64, ReLU kills roughly half those neurons (sets them to 0). Different inputs fire different neurons — this is how the network stores different "facts" in different places.

---

## Are fc1 and fc2 separate per transformer layer?

Yes — each transformer layer gets its **own** independent `fc1` and `fc2`:

```
layer0.mlp_fc1  [64, 16]  ← completely separate weights
layer0.mlp_fc2  [16, 64]
layer1.mlp_fc1  [64, 16]  ← different weights, different specialization
layer1.mlp_fc2  [16, 64]
...
```

In microgpt `n_layer=1` so there's only one set. But in GPT-2 (12 layers) you'd have 12 separate fc1 and 12 separate fc2. Each layer's MLP specializes in whatever gradient pressure shapes it into:

```
layer 0 MLP → might learn low-level patterns  ("q usually follows u")
layer 1 MLP → might learn mid-level patterns  ("this looks like a name ending")
layer 2 MLP → might learn high-level patterns ("this is a vowel-heavy context")
```

Same is true for `attn_wq/wk/wv/wo` — all separate per layer.

---

## All weights start as random noise

Every matrix — `wte`, `wpe`, `attn_wq/k/v/o`, `mlp_fc1/fc2`, `lm_head` — is identical in kind at init. All are just `random.gauss(0, 0.08)`. Training shapes them into completely different roles. The names reflect their **intended role after training**, not any difference in how they're initialized.
