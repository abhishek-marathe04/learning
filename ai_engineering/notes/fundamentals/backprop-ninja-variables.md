# Backprop Ninja — Full Variable Breakdown

## Part 1: The Network Forward Pass

```python
emb = C[Xb]                          # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact)
# Linear layer 2
logits = h @ W2 + b2
```

Let's use a tiny concrete example: batch size `n = 4`, context length `block_size = 3`, embedding dim = `2`, hidden size = `5`, vocab = `27`.

### `emb`
```python
emb = C[Xb]
```
`C` is the embedding table — shape `(27, 2)`, one 2-dimensional vector per character. `Xb` is the input batch — shape `(4, 3)`, i.e. 4 examples, each a context of 3 character indices.

Indexing `C` with `Xb` looks up a vector for each character in each context. Result shape: `(4, 3, 2)` — for each example, 3 characters, each represented by a 2-dim vector.

```
Xb = [[5, 12, 1], ...]   # example 0's context: chars 5, 12, 1
emb[0] = [C[5], C[12], C[1]]   # three 2-dim vectors stacked
```

### `embcat`
```python
embcat = emb.view(emb.shape[0], -1)
```
Flattens the last two dimensions. `(4, 3, 2)` → `(4, 6)`. Each example's 3 character-vectors (2 numbers each) get laid out side by side into one 6-number row. This is the actual input vector fed to the first linear layer.

### `hprebn`
```python
hprebn = embcat @ W1 + b1
```
First linear layer. `embcat` is `(4, 6)`, `W1` is `(6, 5)`, `b1` is `(5,)`. Result `hprebn` is `(4, 5)` — 4 examples, 5 hidden units each. "preBN" = pre-batchnorm, the raw pre-activation values before normalization.

### `bnmeani`
```python
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
```
Mean of `hprebn` across the batch dimension (dimension 0 — across the 4 examples), per hidden unit. Shape `(1, 5)`. This is "the average value this hidden unit takes across the current batch."

### `bndiff`
```python
bndiff = hprebn - bnmeani
```
Each value minus its column's mean (broadcast). Shape `(4, 5)`. This centers every hidden unit's activations around 0 for this batch.

### `bndiff2`
```python
bndiff2 = bndiff**2
```
Element-wise square of `bndiff`. Shape `(4, 5)`. Needed to compute variance — squaring makes all deviations positive so they don't cancel out when summed.

### `bnvar`
```python
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
```
Sum the squared deviations across the batch, divide by `n-1`. Shape `(1, 5)`. This is the **variance** of each hidden unit across the batch. Dividing by `n-1` instead of `n` is Bessel's correction — a standard unbiased-variance-estimate adjustment for sample statistics.

### `bnvar_inv`
```python
bnvar_inv = (bnvar + 1e-5)**-0.5
```
`1 / sqrt(variance + epsilon)`. Shape `(1, 5)`. The `1e-5` epsilon prevents division by zero if variance is 0. This is the reciprocal standard deviation — same "split division into multiply by inverse" trick seen in the loss section, so each step (`add eps`, `raise to -0.5`) has a simple separate derivative.

### `bnraw`
```python
bnraw = bndiff * bnvar_inv
```
`bndiff` (4,5) times `bnvar_inv` (1,5), broadcast. Shape `(4, 5)`. This is the actual normalization: `(x - mean) / std`. After this, each hidden unit has mean ≈ 0 and variance ≈ 1 across the batch.

### `hpreact`
```python
hpreact = bngain * bnraw + bnbias
```
Scale and shift the normalized values by learnable parameters `bngain` and `bnbias` (each shape `(1, 5)`). Shape `(4, 5)`. This lets the network learn to undo the normalization if needed — BatchNorm normalizes, then gives the network a learnable knob to re-adjust the scale and offset per hidden unit.

### `h`
```python
h = torch.tanh(hpreact)
```
Apply tanh element-wise. Shape `(4, 5)`. Squashes values into `(-1, 1)` — the actual hidden layer activations.

### `logits`
```python
logits = h @ W2 + b2
```
Second linear layer. `h` is `(4, 5)`, `W2` is `(5, 27)`, `b2` is `(27,)`. Result `logits` is `(4, 27)` — raw, unnormalized scores over the vocabulary for each example. This feeds directly into the loss pipeline (Part 2).

## Full Pipeline at a Glance

```
Xb (4,3)
  ↓ lookup
emb (4,3,2)
  ↓ flatten
embcat (4,6)
  ↓ linear (W1,b1)
hprebn (4,5)
  ↓ mean over batch        → bnmeani (1,5)
  ↓ subtract                → bndiff (4,5)
  ↓ square                   → bndiff2 (4,5)
  ↓ mean over batch          → bnvar (1,5)
  ↓ (+eps)^-0.5              → bnvar_inv (1,5)
  ↓ multiply                 → bnraw (4,5)
  ↓ scale+shift (bngain,bnbias) → hpreact (4,5)
  ↓ tanh
h (4,5)
  ↓ linear (W2,b2)
logits (4,27)
  ↓ ... (Part 2: loss pipeline)
```

---

# Part 2: Loss Variable Breakdown

Karpathy explodes `F.cross_entropy(logits, Yb)` into 8 separate variables so each one can be differentiated individually. Here's what each one holds, with a concrete example.

## Setup

Batch size = 3 examples, vocab size = 4 classes.

```
logits = [
  [2.0,  1.0,  0.5,  0.1],   # example 0 — correct answer is class 1
  [0.3,  2.5,  0.1,  0.8],   # example 1 — correct answer is class 2
  [1.0,  0.2,  3.0,  0.5],   # example 2 — correct answer is class 2
]

Yb = [1, 2, 2]   # correct class index per example
```

## The Pipeline

```
logits  (3×4)
   ↓  subtract row max
norm_logits  (3×4)
   ↓  exp()
counts  (3×4)
   ↓  sum across columns
counts_sum  (3×1)
   ↓  raise to -1
counts_sum_inv  (3×1)
   ↓  multiply (broadcast)
probs  (3×4)         ← actual softmax output
   ↓  log()
logprobs  (3×4)
   ↓  pick correct class, negate, mean
loss  (scalar)
```

## Variable by Variable

### 1. `logit_maxes`
```python
logit_maxes = logits.max(1, keepdim=True).values
```
The max logit per row. Shape `(3, 1)`.
```
logit_maxes = [[2.0], [2.5], [3.0]]
```
**Why:** pure numerical stability. Subtracting it before `exp()` prevents overflow. `softmax(x) = softmax(x - c)` for any constant `c`, so the math is unchanged.

### 2. `norm_logits`
```python
norm_logits = logits - logit_maxes
```
Logits with the row max subtracted (broadcast). Shape `(3, 4)`. Every row now has a max of 0.0.

### 3. `counts`
```python
counts = norm_logits.exp()
```
`e^(norm_logit)` for every element — the "unnormalized probabilities." Shape `(3, 4)`. All positive, but rows don't sum to 1 yet.

### 4. `counts_sum`
```python
counts_sum = counts.sum(1, keepdim=True)
```
Sum of each row in `counts`. Shape `(3, 1)`. This is the softmax denominator.

### 5. `counts_sum_inv`
```python
counts_sum_inv = counts_sum**-1
```
Reciprocal of `counts_sum`. Shape `(3, 1)`.

**Why invert instead of dividing directly?** `a * (1/b)` is the same as `a / b`, but splitting it into multiply + power(-1) gives two operations with simple, separately differentiable derivatives, instead of one harder-to-backprop division.

### 6. `probs`
```python
probs = counts * counts_sum_inv
```
Actual softmax probabilities — `counts` (3×4) broadcast-multiplied by `counts_sum_inv` (3×1). Shape `(3, 4)`. Every row now sums to 1.

### 7. `logprobs`
```python
logprobs = probs.log()
```
Natural log of every probability. Shape `(3, 4)`. Always negative — closer to 0.0 means probability close to 1 (good), more negative means probability close to 0 (bad).

### 8. `loss`
```python
loss = -logprobs[range(n), Yb].mean()
```

**`Yb`** is the ground-truth class index for each example — always within `[0, vocab_size-1]` since it comes from the same vocabulary as `logprobs`'s columns.

**`logprobs[range(n), Yb]`** is fancy indexing: pair row index `i` with column `Yb[i]`, picking only the logprob the model assigned to the *correct* class for each example. All wrong-class columns are ignored — cross-entropy only asks "how confident was the model about the right answer?"

**`.mean()`** averages across the batch.

**Negation** flips the sign so loss is positive and decreases as the model improves:
- Perfect prediction (prob → 1.0) → logprob → 0.0 → loss → 0.0
- Terrible prediction (prob → 0.0) → logprob → -∞ → loss → +∞

**One-line mental model:**
```
loss = -(log probability the model assigned to the correct answer), averaged over the batch
```

## On the `d` Prefix (Backward Pass Naming)

When Karpathy writes `dlogprobs`, `dprobs`, `dcounts`, etc., the `d` does **not** mean "derivative of the log" or anything specific to that variable's math. It always means:

```
dX = dLoss / dX
```

i.e. "how much does the loss change if I nudge any element of `X`?"

**Shape rule:** the gradient of any variable always has the *same shape* as the variable itself.
```
logprobs    (3,4)  →  dlogprobs    (3,4)
counts_sum  (3,1)  →  dcounts_sum  (3,1)
loss        scalar →  dloss        scalar
```

Read `dX` mentally as: "how angry is the loss at X."
