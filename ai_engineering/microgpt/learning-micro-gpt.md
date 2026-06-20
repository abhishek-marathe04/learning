# microgpt — Transformer Internals Walkthrough
> **Session Date:** June 17, 2026  
> **Duration context:** Deep-dive  
> **Tags:** `#microgpt` `#transformer` `#attention` `#autograd` `#karpathy` `#embeddings` `#kv-cache` `#multi-head-attention`

---

## Overview

This session walked through Andrej Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) — a 200-line pure Python GPT implementation with no dependencies. The walkthrough was step-by-step, building intuition for each component before reading the code. Topics covered: `__slots__`, the `Value` autograd engine, where derivatives are computed, `linear`/`softmax`/`rmsnorm` helpers, and the complete `gpt()` forward pass — embeddings, multi-head attention (Q/K/V projections, KV cache, dot-product scoring, softmax, value retrieval, output projection, residual), and the MLP block (expand → ReLU → compress → residual).

---

## Core Concepts

### `__slots__` on the `Value` class

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
```

Normally every Python object carries a `__dict__` (a full dictionary) to store its attributes — flexible but memory-heavy (~200–400 bytes per object). `__slots__` tells Python: **this class will only ever have exactly these four attributes**. Python then allocates fixed-size slots instead of a dict, saving ~3–5x memory per object.

This matters in microgpt because the computation graph creates **tens of thousands of `Value` objects** — every scalar weight, every intermediate result. Without `__slots__`, memory usage would balloon significantly.

The four slots map exactly to what autograd needs:
- `data` — the scalar value computed in the forward pass
- `grad` — dL/d(this node), filled in by `backward()`
- `_children` — inputs to this operation (who created me?)
- `_local_grads` — ∂(this node)/∂(each child) — the local derivative

---

### Where derivatives are computed — at operation time

This is the key insight of the `Value` class. Derivatives are **not** computed during `backward()`. They are computed and stored **at the moment each operation executes**:

```python
def __add__(self, other):
    return Value(self.data + other.data, (self, other), (1, 1))
#                                                        ↑ ↑
#                                         ∂(a+b)/∂a=1   ∂(a+b)/∂b=1

def __mul__(self, other):
    return Value(self.data * other.data, (self, other), (other.data, self.data))
#                                                        ↑            ↑
#                                         ∂(a*b)/∂a=b   ∂(a*b)/∂b=a

def __pow__(self, other):
    return Value(self.data**other, (self,), (other * self.data**(other-1),))
#                                            ↑ power rule: n·aⁿ⁻¹

def log(self):
    return Value(math.log(self.data), (self,), (1/self.data,))
#                                               ↑ ∂ln(a)/∂a = 1/a

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))
#                                               ↑ ∂eˣ/∂x = eˣ

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
#                                             ↑ 1 if positive, 0 if negative
```

**Full table of local gradients:**

| Operation | Forward | Local gradient |
|-----------|---------|----------------|
| `a + b` | a + b | ∂/∂a = 1, ∂/∂b = 1 |
| `a * b` | a · b | ∂/∂a = b, ∂/∂b = a |
| `a ** n` | aⁿ | ∂/∂a = n·aⁿ⁻¹ |
| `log(a)` | ln(a) | ∂/∂a = 1/a |
| `exp(a)` | eᵃ | ∂/∂a = eᵃ |
| `relu(a)` | max(0,a) | ∂/∂a = 1 if a>0 else 0 |

`backward()` is just a **messenger** — it walks the graph in reverse topological order and multiplies stored local grads by the flowing gradient:

```python
for v in reversed(topo):
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad   # chain rule: just multiplication
```

The `+=` (accumulation) handles the case where a node is used in multiple places — gradients from all paths must be summed.

---

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

### Parameter Setup — All Weight Matrices, Explained

This is the full parameter initialization code that runs **once, before any training**:

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

#### Acronym glossary — what every name means

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

#### What is `fc` — Fully Connected?

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

#### Are `mlp_fc1` and `mlp_fc2` hidden layers?

Yes. The classic neural network picture maps directly:

```
input layer  →  hidden layer  →  output layer
x (16-dim)   →  fc1 (64-dim)  →  fc2 (16-dim)
```

The 64-dim middle is the **hidden layer** — "hidden" because you never directly see or interpret those 64 numbers. They are internal working space. After `fc1` expands to 64, ReLU kills roughly half those neurons (sets them to 0). Different inputs fire different neurons — this is how the network stores different "facts" in different places.

---

#### Are fc1 and fc2 separate per transformer layer?

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

### Stage 1: Embeddings in `gpt()`

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

### Stage 2: Multi-Head Attention

```python
x_residual = x
x = rmsnorm(x)
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
v = linear(x, state_dict[f'layer{li}.attn_wv'])
keys[li].append(k)
values[li].append(v)
```

#### Where are `attn_wq`, `attn_wk`, `attn_wv` built?

In the parameter setup section, before training:

```python
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # [16, 16]
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # [16, 16]
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # [16, 16]
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # [16, 16]
```

Same mechanism as embeddings — random init, trained into something meaningful:
```
wte     → learns "what each token means"
attn_wq → learns "how to form a good query"
attn_wk → learns "how to form a good key"
attn_wv → learns "how to form a good value"
```

#### Q, K, V — the library analogy

Same `x` (16-dim), three different weight matrices, three different outputs:
- **Q (Query)** — "what am I looking for?"
- **K (Key)** — "what do I contain / advertise?"
- **V (Value)** — "what will I give if selected?"

#### KV Cache

Every processed position appends its k and v. For "emma" = `[BOS, e, m, m, a, BOS]`, at position 3 (second 'm'):

```
keys[0]   = [k_BOS, k_e, k_m, k_m]   ← positions 0, 1, 2, 3
values[0] = [v_BOS, v_e, v_m, v_m]
```

Note: NOT three m's. The sequence is BOS(0), e(1), m(2), m(3).

#### Multi-head split

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

#### Attention scores (unpacked)

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

#### Softmax → attention weights

```python
attn_weights = softmax(attn_logits)
# [0.36, 0.31, 0.39, 0.39] → [0.24, 0.23, 0.27, 0.27]
```

Now a probability distribution over past positions.

#### Weighted sum of values

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

#### Concatenate + output projection

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

#### Residual connection (attention)

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

### Stage 3: MLP Block

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

#### fc1 — expand (16 → 64)

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
```

`mlp_fc1` is shape `[64, 16]`. Expands the 16-dim vector to 64 dims (4× expansion).

**Why expand?** The 16-dim space is cramped. Complex intermediate features don't fit cleanly in 16 numbers. Expanding to 64 gives the network room to "think" — represent combinations that don't have a clean home in 16 dims.

#### ReLU — kill negatives

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

#### fc2 — compress back (64 → 16)

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
```

`mlp_fc2` is shape `[16, 64]`. Compresses back from 64 to 16.

The expand-think-compress pattern:
```
16 → 64 → 16
     ↑
  thinking happens here
  in the wide space, after ReLU kills half the neurons
```

#### Residual connection (MLP)

```python
x = [a + b for a, b in zip(x, x_residual)]
```

Same pattern as attention — MLP's output is a delta, not a replacement. The token's state is enriched, not overwritten.

---

### Stage 4: Output logits

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

- **Derivatives live in the operation, not in `backward()`**. Each `__add__`, `__mul__` etc. stores its local gradient at the moment it runs. `backward()` just multiplies and accumulates.
- **`__slots__` is a memory optimization** that works perfectly here because `Value` never needs more than 4 attributes — the math defines exactly what's needed.
- **RMSNorm = volume control**. It preserves the shape (ratios between dimensions = information) and only normalizes the overall scale.
- **Q, K, V are three different projections of the same vector** via three independently learned weight matrices.
- **The KV cache is always conceptually present**, even during training. microgpt just makes it explicit because it processes one token at a time.
- **Multi-head attention = parallel independent lookups**. Each head can learn to attend to different kinds of relationships simultaneously.
- **Attention = communication, MLP = computation**. Attention lets tokens look at each other. MLP lets each token think alone in a wider space.
- **The 4× MLP expansion (16→64→16) gives the network room to think**. ReLU in the middle creates sparsity — different inputs fire different neurons.
- **Residual connections = additive enrichment**. Tokens keep their identity and add context. Gradients can also flow straight through, making deep training possible.
- **"16-dim" just means a list of 16 numbers** — no magic, just a vector.
- **`lm_head` is just one final linear layer** — 16 numbers in, 27 scores out, one per vocabulary token.
- **`fc` = Fully Connected = `linear()`** — every input neuron connects to every output neuron. `fc`, `linear`, and PyTorch's `nn.Linear` are all the same operation, just different names from different traditions.
- **`mlp_fc1`'s 64-dim output is the hidden layer** — internal working space never directly inspected. ReLU kills ~half the neurons, creating sparsity where different inputs fire different neurons.
- **Every weight matrix is per-layer and independent** — `layer0.attn_wq` and `layer1.attn_wq` are completely separate matrices that learn different specializations through training.
- **All weights start as random noise** — `random.gauss(0, 0.08)`. Every matrix (wte, wpe, attn_wq/k/v/o, mlp_fc1/fc2, lm_head) is identical in kind at init. Training shapes them into completely different roles.

---

## Open Questions / Next Steps

- Training loop — how is loss computed from the 27 logits?
- How does `loss.backward()` flow all the way back through the KV cache to the weight matrices?
- Adam optimizer — what do the momentum buffers `m` and `v` actually do per parameter?
- How do the weight matrices (`attn_wq` etc.) actually change during a training step?

---

## References

- [microgpt by Andrej Karpathy](https://karpathy.github.io/2026/02/12/microgpt/)
- [microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [micrograd video (2.5 hrs)](https://www.youtube.com/watch?v=VMj-3S1tku0) — deep dive on the Value/autograd engine
- [Google Colab notebook](https://colab.research.google.com/drive/1vyN5zo6rqUp_dYNbT4Yrco66zuWCZKoN?usp=sharing)
