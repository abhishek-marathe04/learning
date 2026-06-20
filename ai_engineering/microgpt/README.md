# microgpt — Transformer Internals Walkthrough

> **Source:** [microgpt by Andrej Karpathy](https://karpathy.github.io/2026/02/12/microgpt/)
> **Sessions:** June 17, 2026 (architecture) · June 20, 2026 (inference)
> **Tags:** `#microgpt` `#transformer` `#attention` `#autograd` `#karpathy` `#embeddings` `#kv-cache` `#multi-head-attention` `#inference` `#sampling` `#temperature` `#autoregression`

A 200-line pure Python GPT with no dependencies. Below is the full structure with key code excerpts. Each section links to detailed notes.

**Detailed notes:**
- [learning-micro-gpt.md](learning-micro-gpt.md) — autograd engine, helpers, parameters, full forward pass
- [inference-notes.md](inference-notes.md) — autoregressive sampling, temperature, hallucination

---

## 1. Autograd Engine — `Value`

→ **[Detailed notes: learning-micro-gpt.md](learning-micro-gpt.md#__slots__-on-the-value-class)**

The `Value` class is a scalar with built-in automatic differentiation. Every weight and intermediate result in microgpt is a `Value`.

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # ... log, exp, pow follow the same pattern
```

`__slots__` cuts memory 3–5× — tens of thousands of `Value` objects are created per forward pass.

Derivatives are stored **at the moment each operation runs** (the `(1, 1)` / `(other.data, self.data)` tuples). `backward()` just multiplies and accumulates:

```python
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad   # chain rule = multiplication
```

---

## 2. Helper Functions

→ **[Detailed notes: learning-micro-gpt.md](learning-micro-gpt.md#linear-matrix-vector-multiplication)**

Three pure functions used throughout the forward pass:

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

- `linear` — matrix-vector multiply (= fully connected / `nn.Linear`)
- `softmax` — scores → probabilities; subtracts max for numerical stability
- `rmsnorm` — volume normalization; preserves ratios, resets overall scale to 1

---

## 3. Parameter Setup

→ **[Detailed notes: learning-micro-gpt.md](learning-micro-gpt.md#parameter-setup--all-weight-matrices-explained)**

All weights initialized once before training. `n_embd=16`, `n_head=4`, `n_layer=1`, `vocab_size=27`:

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte':     matrix(vocab_size, n_embd),    # [27, 16] token embeddings
    'wpe':     matrix(block_size, n_embd),    # [16, 16] position embeddings
    'lm_head': matrix(vocab_size, n_embd),    # [27, 16] output projection
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

All matrices start as noise (`gauss(0, 0.08)`). Training shapes them into their roles. The names describe what they become, not what they start as.

---

## 4. Forward Pass — `gpt(token_id, pos_id)`

→ **[Detailed notes: learning-micro-gpt.md](learning-micro-gpt.md#stage-1-embeddings-in-gpt)**

### Stage 1: Embeddings

```python
tok_emb = state_dict['wte'][token_id]   # what am I?
pos_emb = state_dict['wpe'][pos_id]     # where am I?
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

Token + position are added (not concatenated) to fuse identity and location into one 16-dim vector.

### Stage 2: Multi-Head Attention

```python
for li in range(n_layer):
    x_residual = x
    x = rmsnorm(x)

    q = linear(x, state_dict[f'layer{li}.attn_wq'])   # "what am I looking for?"
    k = linear(x, state_dict[f'layer{li}.attn_wk'])   # "what do I contain?"
    v = linear(x, state_dict[f'layer{li}.attn_wv'])   # "what will I give if selected?"
    keys[li].append(k)
    values[li].append(v)

    x_attn = []
    for h in range(n_head):
        s, e = h * head_dim, (h + 1) * head_dim
        q_h = q[s:e]
        k_h = [k[s:e] for k in keys[li]]
        v_h = [v[s:e] for v in values[li]]

        attn_logits = [
            sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
            for t in range(len(k_h))
        ]
        attn_weights = softmax(attn_logits)
        head_out = [
            sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
            for j in range(head_dim)
        ]
        x_attn.extend(head_out)

    x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
    x = [a + b for a, b in zip(x, x_residual)]   # residual
```

4 heads run in parallel, each attending to different relationship types. KV cache (`keys[li]`, `values[li]`) accumulates all past positions. `attn_wo` mixes the heads' findings. Residual preserves the token's original state.

### Stage 3: MLP Block

```python
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # 16 → 64
    x = [xi.relu() for xi in x]
    x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # 64 → 16
    x = [a + b for a, b in zip(x, x_residual)]        # residual
```

Attention = communication (tokens looking at each other). MLP = computation (each token thinking alone). Expand to 64, ReLU kills ~half, compress back to 16.

### Stage 4: Output logits

```python
logits = linear(x, state_dict['lm_head'])
return logits   # 27 scores, one per vocabulary token
```

---

## 5. Inference Loop

→ **[Detailed notes: inference-notes.md](inference-notes.md)**

The same `gpt()` function runs unchanged. Three things differ from training: no target/loss/backward, the model's own sampled output feeds back as the next input (autoregression), and the KV cache resets per sample.

```python
temperature = 0.5
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

**Forward pass is fully deterministic.** Same input + frozen params → same logits → same probs, every time. All randomness is in `random.choices` — a weighted die roll over the probability distribution.

**The learned probabilities mirror training data frequencies.** If `'a'` started ~45% of names, the model emits `'a'` first ~45% of the time. At every position the model has learned `p(next | context)` — that's what cross-entropy optimized.

**Temperature** controls how sharp or flat the distribution is before sampling:

| temperature | effect |
|-------------|--------|
| → 0 | greedy decoding — always picks the single highest-probability token |
| 1.0 | raw learned distribution — no change |
| → ∞ | nearly uniform — ignores what was learned |

**BOS doubles as the stop signal** — training wrapped every document `[BOS, ...chars..., BOS]`, so the model predicts BOS when a name is complete.

**Hallucination is just sampling.** microgpt generating `karia` is mechanically identical to ChatGPT stating a plausible-but-wrong fact — both are sampling from a learned distribution with no concept of truth.

---

## 6. Pipeline at a glance

```
token_id, pos_id
      ↓
wte[token_id] + wpe[pos_id]    → 16-dim (what + where)
      ↓ rmsnorm
┌──── ATTENTION BLOCK ────────────────────────────┐
│  Q = linear(x, wq)                             │
│  K = linear(x, wk)  → KV cache                 │
│  V = linear(x, wv)  → KV cache                 │
│  4 heads: dot(Q,K)/√4 → softmax → weighted V   │
│  concat heads → linear(attn_wo) → + residual   │
└─────────────────────────────────────────────────┘
┌──── MLP BLOCK ──────────────────────────────────┐
│  fc1: 16 → 64 → relu → fc2: 64 → 16            │
│  + residual                                     │
└─────────────────────────────────────────────────┘
      ↓  (repeat n_layer times)
lm_head: 16 → 27 logits
      ↓
── TRAINING ──────────────────────────────────────
softmax(logits) → cross-entropy loss vs. target
loss.backward() → gradients → Adam update weights

── INFERENCE ─────────────────────────────────────
logits / temperature → softmax → random.choices()
      ↓
sampled token_id  ──────────────────────┐
      ↑                                 │ (feed back)
      └─────────────────────────────────┘
      stop when BOS sampled or block_size reached
```

---

## Open Questions / Next Steps

**Training loop (not yet covered):**
- How is cross-entropy loss computed from the 27 logits?
- How does `loss.backward()` flow all the way back through the KV cache to the weight matrices?
- Adam optimizer — what do the momentum buffers `m` and `v` actually do per parameter?
- How do the weight matrices actually change during a training step?

**Inference experiments to run:**
- Try `temperature` values (0.1, 0.5, 1.0, 1.5) — observe how name quality/diversity shifts.
- Fix `random.seed()` before sampling to confirm the forward pass is deterministic (same probs printed 20× in a row).
- Swap the dataset (cities, Pokémon, words) — same code, different conditional distribution.

---

## References

- [microgpt blog post](https://karpathy.github.io/2026/02/12/microgpt/)
- [microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [micrograd video (2.5 hrs)](https://www.youtube.com/watch?v=VMj-3S1tku0) — deep dive on the Value/autograd engine
- [Google Colab notebook](https://colab.research.google.com/drive/1vyN5zo6rqUp_dYNbT4Yrco66zuWCZKoN?usp=sharing)

**Session notes:**
- [learning-micro-gpt.md](learning-micro-gpt.md) — June 17 deep-dive: autograd, forward pass, all weight matrices
- [inference-notes.md](inference-notes.md) — June 20 deep-dive: sampling, temperature, autoregression, hallucination
