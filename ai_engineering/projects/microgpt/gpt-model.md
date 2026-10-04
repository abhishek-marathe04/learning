# GPT Model — Forward Pass Walkthrough

> **Tags:** `#gpt` `#transformer` `#attention` `#mlp` `#embeddings` `#kv-cache` `#multi-head-attention` `#residual`

---

## Helper functions

### `linear(x, w)` — matrix-vector multiplication

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

For each row `wo` in weight matrix `w`, compute one dot product with `x`. This is a learned linear transformation — the fundamental building block of neural networks.

**Concrete example:**
```
w = [[1, 0],    x = [3,
     [0, 1],         4]
     [2, 3]]

output[0] = 1*3 + 0*4 = 3
output[1] = 0*3 + 1*4 = 4
output[2] = 2*3 + 3*4 = 18
```

Input size 2 → output size 3. The weight matrix transforms the vector from one space to another.

---

### `softmax(logits)` — scores to probabilities

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Converts raw scores (any range) → probabilities (0 to 1, sum to 1). Core formula: `eˣⁱ / Σeˣ`

**Why `exp`?** Forces everything positive. Also amplifies differences — score 5 vs 3 becomes e⁵ vs e³ = 148 vs 20.

**Why subtract `max_val`?** Pure numerical safety. `exp(900)` overflows to infinity. `exp(900-900) = exp(0) = 1`. Mathematically identical result.

**Concrete example:**
```
logits = [2.0, 1.0, 0.5]

exp(2.0) = 7.39
exp(1.0) = 2.72
exp(0.5) = 1.65
total    = 11.76

probs = [0.63, 0.23, 0.14]   ← sums to 1.0
```

---

### `rmsnorm(x)` — volume normalization

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

**Mental model: volume normalization on a speaker.** Ask "how loud is this vector overall?" then turn the volume knob so loudness is always exactly 1.

Three steps:
1. Square every number, take the mean → `ms` (the "loudness")
2. `scale = 1 / √ms` (the volume knob)
3. Multiply every element by `scale`

**Concrete example:**
```
x = [3.0, 4.0, 0.0, 2.0]

Step 1: ms = (9 + 16 + 0 + 4) / 4 = 7.25
Step 2: scale = 1 / √7.25 = 0.371
Step 3: output = [1.11, 1.49, 0.0, 0.74]

Verify: (1.11² + 1.49² + 0.0² + 0.74²) / 4 = 1.0 ✓
```

**What stays the same:** the ratios between numbers (shape of the vector, i.e. the information).
**What changes:** only the overall scale.

**Why it matters:** Without it, activations can explode across layers (1.0 → 4.0 → 16.0 → 4096.0). RMSNorm resets the magnitude to ~1 after each layer so the network sees a consistent, stable signal.

---

## Stage 1: Embeddings

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
x = rmsnorm(x)
```

**`wte`** (word token embeddings) — shape `[27, 16]`. A lookup table: one 16-dim row per token. `wte[token_id]` is a simple row lookup.

**`wpe`** (word position embeddings) — shape `[16, 16]`. Same idea for positions 0–15.

Both are initialized randomly and **shaped by training**, just like `attn_wq/wk/wv`.

**Why add instead of concatenate?** The model needs to know two independent things:
- **What** is this token? → `wte`
- **Where** is it in the sequence? → `wpe`

Adding them fuses both into a single 16-dim vector:
```
tok_emb = [0.45,  0.11, -0.92, ...]   # "I am the letter e"
pos_emb = [0.03, -0.11,  0.77, ...]   # "I am at position 0"
x       = [0.48,  0.00, -0.15, ...]   # "I am e at position 0"
```

`rmsnorm` at the end stabilizes the result before it enters attention.

---

## Stage 2: Multi-Head Attention

```python
x_residual = x
x = rmsnorm(x)
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
keys[li].append(k)
values[li].append(v)
```

### Q, K, V — the library analogy

Same `x` (16-dim), three different weight matrices, three different outputs:
- **Q (Query)** — "what am I looking for?"
- **K (Key)** — "what do I contain / advertise?"
- **V (Value)** — "what will I give if selected?"

All three weight matrices start as random noise and are trained into their roles. Compare:
```
wte     → learns "what each token means"
attn_wq → learns "how to form a good query"
attn_wk → learns "how to form a good key"
attn_wv → learns "how to form a good value"
```

### KV Cache

Every processed position appends its k and v. For "emma" = `[BOS, e, m, m, a, BOS]`, at position 3 (second 'm'):

```
keys[0]   = [k_BOS, k_e, k_m, k_m]   ← positions 0, 1, 2, 3
values[0] = [v_BOS, v_e, v_m, v_m]
```

Note: NOT three m's. The sequence is BOS(0), e(1), m(2), m(3).

### Multi-head split

`n_head=4`, `head_dim=4` (16/4):
```
q = [q0,q1,q2,q3 | q4,q5,q6,q7 | q8,q9,q10,q11 | q12,q13,q14,q15]
     ← head 0 →   ← head 1 →    ←   head 2   →   ←    head 3    →
```

Each head independently learns different relationships:
```
head 0 → "look for vowels before me"
head 1 → "look for the start token"
head 2 → "look at immediately previous character"
head 3 → "look for repeated characters"
```

### Attention scores

The compact line:
```python
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
    for t in range(len(k_h))
]
```

Unrolled:
```python
attn_logits = []
for t in range(len(k_h)):         # loop over every past position
    dot = 0
    for j in range(head_dim):     # loop over every dimension (0,1,2,3)
        dot += q_h[j] * k_h[t][j]
    dot = dot / head_dim**0.5     # scale by √head_dim
    attn_logits.append(dot)
```

**Concrete example** (head 0, position 3):
```
q_h    = [0.5, 0.3, 0.8, 0.2]

t=0 (BOS): 0.5*0.9 + 0.3*0.1 + 0.8*0.2 + 0.2*0.4 = 0.72 / 2 = 0.36
t=1 ('e'): 0.5*0.6 + 0.3*0.2 + 0.8*0.3 + 0.2*0.1 = 0.62 / 2 = 0.31
t=2 ('m'): 0.5*0.3 + 0.3*0.7 + 0.8*0.5 + 0.2*0.1 = 0.78 / 2 = 0.39
t=3 ('m'): same as t=2                                         = 0.39

attn_logits = [0.36, 0.31, 0.39, 0.39]
```

Why divide by `√head_dim`? Without scaling, dot products grow large as `head_dim` grows, pushing softmax into near-zero gradient regions.

### Softmax → attention weights

```python
attn_weights = softmax(attn_logits)
# [0.36, 0.31, 0.39, 0.39] → [0.24, 0.23, 0.27, 0.27]
```

Now a probability distribution over past positions.

### Weighted sum of values

```python
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
```

Blend all past value vectors weighted by attention:
```
head_out = 0.24 * v_BOS
         + 0.23 * v_e
         + 0.27 * v_m (pos 2)
         + 0.27 * v_m (pos 3)
```

Result: a 4-dim vector carrying blended information from the entire past, weighted by relevance.

### Concatenate + output projection

```python
x_attn.extend(head_out)   # 4 heads × 4 dims = 16 dims
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
```

After the loop, `x_attn` is the concatenation of all 4 heads' outputs (4 dims each = 16 total):

```
head 0 output = [0.31, 0.28, 0.19, 0.22]
head 1 output = [0.11, 0.45, 0.33, 0.17]
head 2 output = [0.52, 0.08, 0.41, 0.29]
head 3 output = [0.24, 0.37, 0.15, 0.44]

x_attn = [0.31, 0.28, 0.19, 0.22, 0.11, 0.45, 0.33, 0.17, 0.52, 0.08, 0.41, 0.29, 0.24, 0.37, 0.15, 0.44]
```

The 4 heads worked in isolation — `attn_wo` is a learned `[16, 16]` mix that lets them combine insights:

```
head 0 found → "previous char was a vowel"
head 2 found → "this char is repeated"
attn_wo learns → "vowel + repeated = probably followed by a vowel"
```

### Residual connection (attention)

```python
x = [a + b for a, b in zip(x, x_residual)]
```

```
x_residual = [0.48, 0.00, -0.15, ...]   # "I am 'm' at position 3"  — saved at block entry
x (attn)   = [0.21, 0.33,  0.09, ...]   # "here's what I learned from the past"
x (new)    = [0.69, 0.33, -0.06, ...]   # both added together
```

Nothing is destroyed — the token keeps its own identity AND gains context from the past. Also critical for gradient flow during training — gradients can skip straight through the `+` to earlier layers.

---

## Stage 3: MLP Block

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # 16 → 64
x = [xi.relu() for xi in x]
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # 64 → 16
x = [a + b for a, b in zip(x, x_residual)]
```

**Attention = communication.** Tokens looking at each other.
**MLP = computation.** Thinking about what you just gathered, alone, at this position only.

### fc1 — expand (16 → 64)

`mlp_fc1` is shape `[64, 16]`. Expands the 16-dim vector to 64 dims (4× expansion).

**Why expand?** The 16-dim space is cramped. Complex intermediate features don't fit cleanly in 16 numbers. Expanding to 64 gives the network room to "think" — represent combinations that don't have a clean home in 16 dims.

### ReLU — kill negatives

```python
x = [xi.relu() for xi in x]
```

Every negative → 0. Every positive → unchanged.

```
before relu = [0.3, -0.8, 1.2, -0.1, 0.6, -0.4, ...]
after relu  = [0.3,  0.0, 1.2,  0.0, 0.6,  0.0, ...]
```

Two reasons this matters:

1. **Non-linearity** — without it, stacking linear layers is pointless (linear × linear = still linear). ReLU breaks that, letting the network learn complex non-linear patterns.
2. **Sparsity** — roughly half the neurons fire (positive), half are silent (zero). Different inputs activate different neurons. This is how the network stores different "facts" in different places.

### fc2 — compress back (64 → 16)

`mlp_fc2` is shape `[16, 64]`. Compresses back from 64 to 16.

The expand-think-compress pattern:
```
16 → 64 → 16
     ↑
  thinking happens here
  in the wide space, after ReLU kills half the neurons
```

### Residual connection (MLP)

```python
x = [a + b for a, b in zip(x, x_residual)]
```

Same pattern as attention — MLP's output is a delta, not a replacement. The token's state is enriched, not overwritten.

---

## Stage 4: Output logits

```python
logits = linear(x, state_dict['lm_head'])
return logits
```

`lm_head` is shape `[27, 16]`. Projects the final 16-dim hidden state to 27 scores — one per token in the vocabulary.

```
logits = [2.1, -0.3, 0.8, 1.4, ...]   # 27 numbers
           ↑
      score for 'a'   ...score for each possible next token
```

Higher logit = model thinks that token is more likely to come next. These raw scores get passed to softmax during training (to compute loss) and during inference (to sample the next token).

---

## Full Pipeline Diagram

```
token_id, pos_id
      ↓
wte[token_id] + wpe[pos_id]    → 16-dim vector (what + where)
      ↓
rmsnorm                         → stable magnitude
      ↓
┌─────────────────────────────────────────────────┐
│  ATTENTION BLOCK                                │
│  save x_residual ──────────────────────────┐   │
│  rmsnorm                                   │   │
│  x → Q (via attn_wq)                       │   │
│  x → K (via attn_wk) → append to KV cache  │   │
│  x → V (via attn_wv) → append to KV cache  │   │
│                                            │   │
│  split Q,K,V into 4 heads (4-dim each)     │   │
│  for each head:                            │   │
│    dot(q_h, each k_h) / √4 → scores       │   │
│    softmax → attn_weights                  │   │
│    weighted sum of v_h → head_out          │   │
│  concat 4 heads → 16-dim                  │   │
│  linear (attn_wo) → mix heads             │   │
│  + x_residual ←───────────────────────────┘   │
└─────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────┐
│  MLP BLOCK                                      │
│  save x_residual ──────────────────────────┐   │
│  rmsnorm                                   │   │
│  fc1: linear 16 → 64                       │   │
│  relu: kill negatives                       │   │
│  fc2: linear 64 → 16                       │   │
│  + x_residual ←───────────────────────────┘   │
└─────────────────────────────────────────────────┘
      ↓  (repeat n_layer times — just 1 in microgpt)
lm_head: linear 16 → 27
      ↓
27 logits  →  softmax  →  sample next token
```

---

## Key Takeaways

- **Q, K, V are three different projections of the same vector** via three independently learned weight matrices.
- **The KV cache is always conceptually present**, even during training. microgpt just makes it explicit because it processes one token at a time.
- **Multi-head attention = parallel independent lookups**. Each head can learn to attend to different kinds of relationships simultaneously.
- **Attention = communication, MLP = computation**. Attention lets tokens look at each other. MLP lets each token think alone in a wider space.
- **The 4× MLP expansion (16→64→16) gives the network room to think**. ReLU in the middle creates sparsity — different inputs fire different neurons.
- **Residual connections = additive enrichment**. Tokens keep their identity and add context. Gradients can also flow straight through, making deep training possible.
- **"16-dim" just means a list of 16 numbers** — no magic, just a vector.
- **`lm_head` is just one final linear layer** — 16 numbers in, 27 scores out, one per vocabulary token.
