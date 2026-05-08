# Understanding Transformer Internals

> A deep-dive mental model for how LLMs actually work — covering Feed-Forward Networks, Attention, Multi-Head Attention, and Residual Connections. Built up from first principles using the example sentence "The bank approved my loan."

---

## Table of Contents

1. [The Feed-Forward Network (FFN)](#1-the-feed-forward-network-ffn)
2. [How Attention and FFN Work Together](#2-how-attention-and-ffn-work-together)
3. [What the FFN Actually Receives](#3-what-the-ffn-actually-receives)
4. [Multi-Head Attention](#4-multi-head-attention)
5. [Emergent Specialization (and Why It's Not Designed)](#5-emergent-specialization-and-why-its-not-designed)
6. [Residual Connections and the Residual Stream](#6-residual-connections-and-the-residual-stream)
7. [Putting It All Together](#7-putting-it-all-together)

---

## 1. The Feed-Forward Network (FFN)

The feed-forward network is one of the two core components inside every transformer block, sitting right after the self-attention layer. While attention often gets the spotlight, the **FFN is where the bulk of an LLM's parameters actually live** — typically around two-thirds of all weights in a transformer block.

### Where It Sits

A transformer block looks roughly like this:

```
Input → LayerNorm → Multi-Head Attention → Residual Add
      → LayerNorm → Feed-Forward Network → Residual Add → Output
```

- **Attention's job**: mix information *across* tokens — letting each position look at others and pull in relevant context.
- **FFN's job**: process each token *independently*, transforming its representation in place. No cross-token communication happens here.

This separation is deliberate. Attention handles *"what should I pay attention to?"* and FFN handles *"now that I have this information, what do I do with it?"*

### The Mechanics

In its classic form (from "Attention Is All You Need"), the FFN is just two linear layers with a non-linearity sandwiched between them:

```
FFN(x) = activation(x · W1 + b1) · W2 + b2
```

The key dimensional trick: if your model's hidden dimension is `d_model` (say 4096), the intermediate dimension is typically `4 × d_model` (so 16384). The first linear layer projects *up* to this larger space, the activation runs, then the second linear layer projects *back down*. This expand-then-contract pattern is sometimes called the **inverted bottleneck**.

For a model like Llama 3 70B, a single FFN might have:
- `W1`: 8192 × 28672 ≈ 235M parameters
- `W2`: 28672 × 8192 ≈ 235M parameters
- Per layer: ~470M parameters just in the FFN

Multiply by 80 layers and you see why FFNs dominate parameter counts.

### What It Actually Computes: Key-Value Memory

There's a well-known interpretation (Geva et al., *"Transformer Feed-Forward Layers Are Key-Value Memories"*) where the FFN acts as a **key-value memory store**:

- Each row of `W1` is a **key** — a pattern detector that fires when the input matches it
- Each corresponding column of `W2` is a **value** — what gets written back when that key fires
- The activation function gates which keys are active

So when a token's hidden state passes through the FFN, you can think of it as querying thousands of stored patterns simultaneously. The matching ones contribute their associated values. This is where a lot of the model's **factual knowledge and learned associations** are encoded — things like "Paris → France," verb conjugations, syntactic patterns, code idioms.

Mechanistic interpretability research has identified specific FFN neurons that fire for specific concepts: a "French" neuron, a "code indentation" neuron, a "negation" neuron, etc.

### Why the Non-Linearity Matters

Without the activation function, two stacked linear layers would collapse into one (matrix multiplication is associative). The non-linearity gives the FFN its computational power.

Activation evolution:
- **ReLU** — original transformer
- **GELU** — BERT, GPT-2/3
- **SwiGLU** — Llama, PaLM, Mistral, and most modern LLMs

SwiGLU uses a gated structure: `FFN(x) = (Swish(x · W1) ⊙ (x · W3)) · W2`. It introduces a third weight matrix and a multiplicative gating path, which empirically gives meaningful quality gains. This is why modern FFNs in Llama-style models actually have *three* matrices, not two.

### The Modern Twist: Mixture of Experts (MoE)

Because FFNs are so parameter-heavy, MoE asks: do we need *all* those parameters active for *every* token?

In an MoE layer, you replace the single FFN with N parallel FFNs ("experts"), and a router picks the top-k experts (typically k=1 or 2) for each token. Mixtral 8x7B and DeepSeek-V3 are well-known examples — same memory footprint, much less compute per token.

---

## 2. How Attention and FFN Work Together

A common confusion: if attention captures relationships between tokens, why doesn't the FFN also consider those relationships?

### The Subtle Point

When the FFN receives a token, that token is **no longer a "pure" word embedding**. It has already been contextualized by attention.

Take the sentence:

> "The bank approved my loan."

When "bank" enters the first attention layer, its embedding is generic — could mean financial institution or river bank. After attention, the representation of "bank" has absorbed signal from "approved" and "loan." Now the representation effectively encodes *bank-in-the-financial-sense*.

**Then** this contextualized representation enters the FFN.

So when we say "FFN processes each token independently," we mean *mechanically* — the FFN doesn't look at neighbors. But the token it's processing is already a rich, context-aware representation thanks to attention.

### Why This Division of Labor Is Brilliant

Think of it as a pipeline:

| Component | Job |
|-----------|-----|
| **Attention** | Gather the right context. For this token, what do I need to know from the rest of the sequence? |
| **FFN** | Given this contextualized representation, compute something useful. Apply transformations, look up stored knowledge, refine meaning. |

The FFN doesn't need to consider relationships between tokens because **attention already baked those relationships into each token's representation**.

### Why Process Tokens Independently in FFN?

1. **Computational efficiency.** Attention is O(n²) in sequence length. If FFN also did cross-token mixing, you'd have another expensive operation. Per-token FFN is O(n) and trivially parallelizable.
2. **Specialization of roles.** Attention is good at "routing" but limited at *transforming*. FFN is good at non-linear transformation and storing knowledge but can't move information between positions. They complement each other.
3. **Stacking creates depth.** FFN output → next layer's attention → mixes again → next FFN → transforms again. Over many layers, each token's representation becomes increasingly rich.

### Mental Model: The Restaurant Analogy

> The waiter (attention) gathers orders and brings them to the chef (FFN). The chef just needs to cook based on the order ticket — they don't need to walk out and talk to each table.

---

## 3. What the FFN Actually Receives

After attention processes "The bank approved my loan", you have a **matrix** of shape `[sequence_length, hidden_dim]`.

For this 5-token sentence with hidden_dim = 4096:

```
                    hidden_dim (4096)
                    ←──────────────────→
       ┌────────────────────────────────┐
"The"  │ [0.2, -0.5, 0.8, ..., 0.1]    │  ← vector for "The"
"bank" │ [0.7, 0.3, -0.2, ..., 0.9]    │  ← vector for "bank"
"app." │ [-0.1, 0.6, 0.4, ..., -0.3]   │  ← vector for "approved"
"my"   │ [0.5, -0.8, 0.1, ..., 0.2]    │  ← vector for "my"
"loan" │ [0.4, 0.2, -0.6, ..., 0.7]    │  ← vector for "loan"
       └────────────────────────────────┘

Shape: [5, 4096]
```

Each **row** is one token's contextualized representation.

### How the FFN Processes This

The FFN has fixed weights (`W1`, `W2`, activation). The same weights are applied to every token's vector independently:

```
"The"      →  FFN(W1, W2)  →  new vector for "The"
"bank"     →  FFN(W1, W2)  →  new vector for "bank"
"approved" →  FFN(W1, W2)  →  new vector for "approved"
"my"       →  FFN(W1, W2)  →  new vector for "my"
"loan"     →  FFN(W1, W2)  →  new vector for "loan"
```

### Implementation

You don't loop over tokens. You batch the whole `[5, 4096]` matrix through the FFN:

```python
# Input: X has shape [5, 4096]
hidden = activation(X @ W1)   # [5, 4096] @ [4096, 16384] → [5, 16384]
output = hidden @ W2          # [5, 16384] @ [16384, 4096] → [5, 4096]
```

Critically:
- Row 0 of the output depends only on row 0 of X
- Row 1 of the output depends only on row 1 of X
- **No mixing between rows**

The matrix multiply is just a parallel way of doing the same per-token computation 5 times at once. It's an implementation efficiency, not a conceptual change.

Compare to attention, where `Q @ K.T` explicitly creates a `[5, 5]` matrix of cross-token scores — *that's* where mixing happens. The FFN never produces such a cross-token matrix.

### Conceptual vs Computational

| Lens | Description |
|------|-------------|
| **Conceptually** | Each token's vector is processed independently by the FFN. |
| **Computationally** | The whole `[seq_len, hidden_dim]` matrix goes through at once for GPU efficiency. |

The batching doesn't create cross-token interaction.

### Why This Matters

1. **Parallelism during training**: All tokens (and across many sequences) can be FFN'd in parallel.
2. **KV-cache efficiency during inference**: When generating token by token, you only run FFN on the *new* token. Old tokens' FFN outputs are already computed.
3. **Where compute goes**: For sequence length N, FFN does N independent computations, attention does N². For long contexts attention dominates; for short sequences FFN dominates (because of its huge intermediate dimension).

---

## 4. Multi-Head Attention

Multi-head attention is the same core concept as attention, but with a critical twist.

### The Core Idea

Single-head attention computes one set of attention patterns. Multi-head attention does this **multiple times in parallel**, with different learned projections, then combines the results.

A single attention pattern can only capture one "type" of relationship at a time. Real language has many simultaneous relationships:

- **Syntactic**: "approved" → "bank" (subject-verb)
- **Semantic**: "loan" → "bank" (topical association)
- **Positional**: "the" → "bank" (adjacent determiner)
- **Coreference**: "my" → "I" (possessive reference)

Multi-head attention is how the model tracks all of these at once.

### How It Works Mechanically

Say `hidden_dim = 4096` and we have 32 heads. Each head works in a `4096 / 32 = 128`-dim subspace.

```
Input: [seq_len, 4096]
       ↓
Project to Q, K, V → each is [seq_len, 4096]
       ↓
Reshape into 32 heads → [32, seq_len, 128]
       ↓
Each head computes its own attention independently:
   Head 1:  Q1, K1, V1 → attention pattern 1 → output 1 [seq_len, 128]
   Head 2:  Q2, K2, V2 → attention pattern 2 → output 2 [seq_len, 128]
   ...
   Head 32: Q32, K32, V32 → attention pattern 32 → output 32 [seq_len, 128]
       ↓
Concatenate all 32 outputs → [seq_len, 4096]
       ↓
Final linear projection W_O → [seq_len, 4096]
```

Each head:
1. Operates in its own smaller subspace (128 dims)
2. Has its own learned Q, K, V projection matrices
3. Computes its own attention scores and weighted sum
4. Produces its own output, which gets concatenated

Parameter count is roughly the same as single-head attention with the full dimension — you're reorganizing the same compute into parallel slices.

### Concrete Example with "The bank approved my loan"

When processing "bank" with multi-head attention:

- One head might attend strongly to "loan" → financial-context disambiguation
- Another head attends to "approved" → what's happening to the bank
- Another head attends to "The" → determiner-noun relationship
- Another head attends to "bank" itself → preserves identity information
- Most other heads contribute smaller, more diffuse signals

All these get concatenated and projected through `W_O`. The result: a "bank" representation that simultaneously encodes financial context, the action being performed, and grammatical role — all in one vector.

With one head, the attention pattern would have to compromise. Multi-head lets the model have its cake and eat it too.

### Why Smaller Subspaces?

1. **Compute efficiency.** 32 heads at 128 dims = 1 head at 4096 dims, because attention compute scales with `seq_len² × head_dim`. Splitting doesn't add cost.
2. **Forced specialization.** With only 128 dimensions, a head can't try to encode everything. It's pressured to focus on a specific kind of pattern.

### Modern Variants

| Variant | Description | Used In |
|---------|-------------|---------|
| **Multi-Query (MQA)** | All heads share K and V; each has own Q. Reduces KV cache memory at some quality cost. | PaLM, earlier models |
| **Grouped-Query (GQA)** | Compromise — e.g. 32 Q heads, 8 K/V heads. Big inference speedup, minimal quality loss. | Llama 2 70B, Llama 3, Mistral |
| **Multi-Latent (MLA)** | Compresses K/V into smaller latent space, dramatically shrinking KV cache. | DeepSeek-V3 |

These are all variations on "how do I structure the heads?" The fundamental concept stays the same.

---

## 5. Emergent Specialization (and Why It's Not Designed)

A crucial realization: when we say "head 1 does syntax, head 2 does coreference" — **that's our post-hoc interpretation, not a designed property.**

### What We Actually Do

1. Create N parallel heads with **identical structure** but **different random initializations**
2. Train the whole network end-to-end on next-token prediction
3. *Hope* that gradient descent finds a configuration where heads do different useful things

That's it. Nothing in the architecture, loss function, or training procedure assigns roles to specific heads.

### Why Does Specialization Happen?

1. **Different random initializations.** Each head starts slightly different; gradient descent amplifies the differences.
2. **Redundancy is wasteful.** If two heads computed identical patterns, they'd contribute the same signal — gradient pressure pushes them apart.
3. **The output projection W_O.** If two heads produce similar outputs, W_O has to do extra work. Differentiation at the head level makes downstream computation easier.
4. **The loss function rewards diversity *implicitly*.** Better next-token prediction requires capturing many kinds of patterns.

But none of this is **guaranteed**. It's emergent, not designed.

### Reality Check

Studies of trained models find:

- **Some heads specialize cleanly.** Induction heads, previous-token heads, name-mover heads have been identified.
- **Many heads are polysemantic.** They do different things in different contexts.
- **Some heads are redundant.** You can ablate 20-40% of heads from a trained transformer with little degradation.
- **Specialization is uneven across layers.** Early layers tend toward "mechanical" heads (positional, syntactic). Later layers toward semantic ones.

### The General Principle

> **We design the architecture to allow useful structure to emerge. We don't design the structure itself.**

This applies broadly:

- Multi-head attention (we hope heads specialize)
- Convolutional filters (we hope they learn different visual features)
- FFN neurons (we hope they encode useful concepts)
- Mixture of Experts (we hope different experts handle different inputs)
- Word embeddings (we hope semantic structure emerges)

In every case, we set up the *capacity* for differentiation and let optimization find a use for it. Sometimes it works beautifully. Sometimes it fails (mode collapse, dead neurons, redundant heads).

---

## 6. Residual Connections and the Residual Stream

Residual connections look like a tiny detail (a "+" in a diagram) but are absolutely critical to making deep networks trainable.

### The Problem They Solve

Before residuals, deeper networks often performed *worse* than shallower ones — even on training data. Gradients struggled to flow back through many layers. Early layers couldn't learn properly.

In 2015, the ResNet paper introduced residual connections. The same idea is what makes 80-layer transformers like Llama 70B trainable.

### What a Residual Connection Is

The simplest possible description:

```
output = input + layer(input)
```

The original input is preserved as a "skip connection" that bypasses the layer.

In a transformer block, you have two residuals:

```
x = x + Attention(LayerNorm(x))    ← residual #1
x = x + FFN(LayerNorm(x))           ← residual #2
```

Each sublayer's output is **added** to what came in, not *replacing* it.

### The Full Block Diagram

```
        Input x
           │
           ├──────────────────┐  (residual path — carries x forward)
           │                  │
        LayerNorm             │
           │                  │
        Attention             │
           │                  │
           └──────► + ◄───────┘
                    │
                    │  ← x + Attention(LayerNorm(x))
                    │
                    ├──────────────────┐  (another residual path)
                    │                  │
                 LayerNorm             │
                    │                  │
                  FFN                  │
                    │                  │
                    └──────► + ◄───────┘
                             │
                          Output
```

### The Bank Example, Layer by Layer

**Initial token embedding** — the vector for "bank" is the generic "bank" embedding. Could mean financial institution, river bank, blood bank, anything.

```
x₀ = [generic "bank" embedding]
```

**Layer 1, after attention residual:**

Attention figures out "this bank is associated with financial concepts" — call this signal `a₁`.

```
x₁ = x₀ + a₁
   = [generic "bank"] + [financial-context signal]
   = [bank-leaning-financial]
```

The original meaning isn't *replaced*. It's *augmented*.

**Layer 1, after FFN residual:**

FFN activates a "financial institution" knowledge pattern stored in its weights — call this `f₁`.

```
x₁' = x₁ + f₁
    = [bank-leaning-financial] + [financial-institution knowledge]
    = [bank-as-financial-institution, slightly stronger]
```

**Layer 2, after attention residual:**

Attention runs again on this updated representation. It picks up additional context — maybe linking "bank" with "approved" to encode "the bank is the agent doing the approving."

```
x₂ = x₁' + a₂
   = [bank-as-financial-institution] + [agent-of-approving]
   = [bank as financial institution acting as approver]
```

And so on through 80 layers. Each layer **adds** new information without overwriting the old.

### The Residual Stream

This is the crucial mental model. Anthropic's interpretability team popularized the term **residual stream**.

Think of residual connections as a continuous "highway" running through the network from input to output. Each token has its own residual stream — a vector evolving from layer to layer.

```
Token "bank":

x₀ ──► [+a₁] ──► [+f₁] ──► [+a₂] ──► [+f₂] ──► ... ──► [+a₈₀] ──► [+f₈₀] ──► final
│                                                                              │
└──────────────────── this is the residual stream ────────────────────────────┘
```

Each layer (attention or FFN) **reads from** the stream, computes something, and **writes back** by adding to it. The stream is a workspace all layers share.

This gives a beautiful interpretation:

- **Attention reads from the stream** at multiple positions, mixes information, writes to the current position
- **FFN reads from the stream** at the current position, looks up knowledge, writes results back
- **The stream itself** accumulates a richer representation of the token's role

### Why Addition Specifically?

1. **Gradient highway.** During backpropagation, residuals provide a direct path for gradients to flow from output back to early layers. Even if a layer's gradient is tiny, the identity path keeps the signal alive. **This is what enables training 80+ layer networks.**

2. **Layers can choose to do nothing.** If a layer's contribution would hurt, it can output near-zero, and the residual passes the input through. Adding more layers can never hurt (in principle) — extra layers degrade gracefully into identity functions.

3. **Shape preservation.** Addition requires matching shapes. Every layer reads and writes to the same `[seq_len, hidden_dim]` workspace.

### Layers as "Updaters"

Because of residuals, each layer doesn't compute the *full* representation from scratch. It only computes the **delta** — what to add to what's already there.

This is why layers can specialize cleanly. An attention head doesn't need to encode "everything about bank." It just needs to compute "the financial-context update for bank" and write that into the stream.

This compositional structure is part of why mechanistic interpretability is tractable. When researchers find an "induction head" that copies tokens, they're identifying a specific *update* it writes — not the head's entire computation.

### Without vs With Residuals

Without residuals:
```
x₀ → Layer1 → x₁ → Layer2 → x₂ → ...
(each xᵢ is a full new representation; original info likely lost)
```

With residuals:
```
x₀ ─┬─ Layer1 ─┐
    │          + → x₁ ─┬─ Layer2 ─┐
    └──────────┘       │          + → x₂ → ...
                       └──────────┘
(each xᵢ is xᵢ₋₁ plus a delta; original info preserved and enriched)
```

### Practical Implications

- **Interpretability.** "What layer X does" really means "what layer X writes to the residual stream."
- **LoRA and fine-tuning.** LoRA adds small low-rank updates — additional deltas the residual stream picks up. Works because of how residual streams compose.
- **Inference optimization.** Layer skipping, early exit, speculative decoding exploit the fact that residual streams sometimes "stabilize" before the final layer. Only possible because of the additive structure.

---

## 7. Putting It All Together

Here's the complete mental model for processing "The bank approved my loan":

### The Pipeline

1. **Tokenization & Embedding** — each word becomes a vector via the embedding table. These are generic, context-free representations.

2. **For each of L transformer layers:**

   a. **Multi-Head Attention** (with residual) — N heads run in parallel, each attending to different relationship patterns. Outputs concatenate. The result is *added* to the residual stream.

   b. **Feed-Forward Network** (with residual) — each token's vector is independently transformed: knowledge lookup, non-linear refinement. The result is *added* to the residual stream.

3. **Final projection** — the last layer's residual stream output is projected to vocabulary logits to predict the next token.

### What Each Component Contributes

| Component | What It Does | Where the Magic Lives |
|-----------|-------------|----------------------|
| **Embeddings** | Convert tokens to vectors | Embedding table |
| **Attention** | Mix information across tokens | Q, K, V projections; learned attention patterns |
| **Multi-Head** | Capture multiple simultaneous relationships | Different heads emerge with different specializations |
| **FFN** | Per-token transformation, knowledge storage | Most of the model's parameters; key-value memory of facts |
| **Residual Stream** | Accumulate refinements without losing prior information | The "highway" connecting all layers |

### The Big Picture

> **Attention decides what information to gather. FFN decides what to do with it. Residual connections preserve everything along the way. Stacking creates depth. And specialization emerges from training, not from design.**

That's the transformer in one sentence.

---

## Further Reading

- *Attention Is All You Need* (Vaswani et al., 2017) — the original transformer paper
- *Deep Residual Learning for Image Recognition* (He et al., 2015) — introduced residual connections
- *Transformer Feed-Forward Layers Are Key-Value Memories* (Geva et al., 2020)
- *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021) — the residual stream framing
- *In-context Learning and Induction Heads* (Anthropic, 2022) — emergent specialization in attention heads

---

*These notes were built up through a conversation, working from concrete examples to general principles. The "bank approved my loan" example threads through every section to keep the math grounded in something tangible.*
