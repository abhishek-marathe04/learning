/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * TypeScript port of @karpathy's microgpt.py
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

// ─── Seeded RNG (replaces Python's random.seed(42)) ─────────────────────────
// JavaScript's Math.random() can't be seeded, so we use a simple LCG (Linear
// Congruential Generator) that gives deterministic results given seed=42.
let rngState = 42;
function randNext(): number {
  rngState = (rngState * 1664525 + 1013904223) & 0xffffffff;
  return (rngState >>> 0) / 0xffffffff;
}
function randGauss(mean = 0, std = 1): number {
  // Box-Muller transform: converts two uniform randoms into a normal distribution
  const u = randNext();
  const v = randNext();
  return mean + std * Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
}
function randChoices(n: number, weights: number[]): number {
  // Weighted random sampling: pick index proportional to weights
  const total = weights.reduce((a, b) => a + b, 0);
  let r = randNext() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}
function shuffle<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(randNext() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ─── Dataset ─────────────────────────────────────────────────────────────────
// Download the names dataset if it doesn't exist
const inputPath = path.join(process.cwd(), "input.txt");
if (!fs.existsSync(inputPath)) {
  console.log("Downloading input.txt...");
  const url =
    "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
  // Synchronous download via child_process for simplicity (mirrors Python's urlretrieve)
  execSync(`curl -s -o "${inputPath}" "${url}"`);
}

const rawText = fs.readFileSync(inputPath, "utf-8");
const docs: string[] = rawText
  .split("\n")
  .map((l) => l.trim())
  .filter((l) => l.length > 0);
shuffle(docs);
console.log(`num docs: ${docs.length}`);

// ─── Tokenizer ───────────────────────────────────────────────────────────────
// Build vocabulary from the unique characters in the dataset.
// Each char gets an integer id (0..n-1). BOS is a special token at id n.
const allChars = docs.join("");
const ucharSet = new Set(allChars.split(""));
const uchars: string[] = [...ucharSet].sort();
const BOS: number = uchars.length;         // e.g. if 26 letters, BOS = 26
const vocab_size: number = uchars.length + 1; // +1 for BOS
console.log(`vocab size: ${vocab_size}`);

// Helper: char → token id
const charToId = new Map<string, number>(uchars.map((c, i) => [c, i]));

// ─── Autograd: Value class ───────────────────────────────────────────────────
// Every scalar in our computation (weights, activations, loss) is a Value node.
// Forward pass computes .data; backward pass accumulates .grad via chain rule.
class Value {
  data: number;
  grad: number;
  _children: Value[];
  _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
  }

  // Arithmetic operators — each records children + their local derivatives
  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    // d/d(self) (self+other) = 1,  d/d(other) (self+other) = 1
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    // d/d(self) (self*other) = other.data,  d/d(other) = self.data
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(exp: number): Value {
    // d/d(self) (self^n) = n * self^(n-1)
    return new Value(
      Math.pow(this.data, exp),
      [this],
      [exp * Math.pow(this.data, exp - 1)]
    );
  }

  log(): Value {
    // d/d(self) ln(self) = 1/self
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    // d/d(self) e^self = e^self
    const ev = Math.exp(this.data);
    return new Value(ev, [this], [ev]);
  }

  relu(): Value {
    // d/d(self) max(0, self) = 1 if self>0, else 0
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0]
    );
  }

  neg(): Value { return this.mul(-1); }
  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }
  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  // Backward pass: topological sort then reverse-mode differentiation
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    function buildTopo(v: Value): void {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) buildTopo(child);
        topo.push(v);
      }
    }

    buildTopo(this);
    this.grad = 1; // dL/dL = 1 (seed the gradient at the output)

    // Walk backwards: for each node, accumulate gradient into its children
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        // chain rule: child.grad += localGrad * this.grad
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// ─── Model hyperparameters ───────────────────────────────────────────────────
const n_layer = 1;
const n_embd = 16;
const block_size = 16;
const n_head = 4;
const head_dim = n_embd / n_head; // 4

// ─── Parameter initialization ────────────────────────────────────────────────
// A "matrix" is a 2D array of Value nodes, filled with small random Gaussians.
// Small std (0.08) prevents exploding activations at startup.
function makeMatrix(nout: number, nin: number, std = 0.08): Value[][] {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(randGauss(0, std)))
  );
}

// StateDict holds all learnable parameters, keyed by name (mirrors PyTorch).
type Matrix = Value[][];
const stateDict: Record<string, Matrix> = {
  wte: makeMatrix(vocab_size, n_embd),   // token embedding table
  wpe: makeMatrix(block_size, n_embd),   // position embedding table
  lm_head: makeMatrix(vocab_size, n_embd), // language model head (output projection)
};
for (let i = 0; i < n_layer; i++) {
  stateDict[`layer${i}.attn_wq`] = makeMatrix(n_embd, n_embd); // query projection
  stateDict[`layer${i}.attn_wk`] = makeMatrix(n_embd, n_embd); // key projection
  stateDict[`layer${i}.attn_wv`] = makeMatrix(n_embd, n_embd); // value projection
  stateDict[`layer${i}.attn_wo`] = makeMatrix(n_embd, n_embd); // output projection
  stateDict[`layer${i}.mlp_fc1`] = makeMatrix(4 * n_embd, n_embd); // MLP expand
  stateDict[`layer${i}.mlp_fc2`] = makeMatrix(n_embd, 4 * n_embd); // MLP contract
}

// Flatten all parameters into a single list for the optimizer
const params: Value[] = Object.values(stateDict).flatMap((mat) =>
  mat.flatMap((row) => row)
);
console.log(`num params: ${params.length}`);

// ─── Model architecture functions ────────────────────────────────────────────

// Linear layer: y = W @ x  (matrix-vector multiply)
// Each output neuron is the dot product of one row of W with x.
function linear(x: Value[], w: Matrix): Value[] {
  return w.map((wo) =>
    wo.reduce((acc, wi, i) => acc.add(wi.mul(x[i])), new Value(0))
  );
}

// Softmax: converts raw logits into a probability distribution.
// Subtracting max_val first (numerically stable — prevents exp() overflow).
function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((a, b) => a.add(b), new Value(0));
  return exps.map((e) => e.div(total));
}

// RMSNorm: normalizes x by its root-mean-square.
// Keeps activations well-scaled through many layers (replaces LayerNorm here).
function rmsnorm(x: Value[]): Value[] {
  const ms = x.reduce((acc, xi) => acc.add(xi.mul(xi)), new Value(0)).div(x.length);
  const scale = ms.add(1e-5).pow(-0.5); // 1 / sqrt(mean_sq + eps)
  return x.map((xi) => xi.mul(scale));
}

// GPT forward pass: given one token at position pos_id, return logits over vocab.
// keys/values are the KV cache — they accumulate across positions in a sequence.
function gpt(
  token_id: number,
  pos_id: number,
  keys: Value[][][],
  values: Value[][][]
): Value[] {
  // 1. Embed: look up token + position embeddings and add them
  const tok_emb = stateDict["wte"][token_id];
  const pos_emb = stateDict["wpe"][pos_id];
  let x: Value[] = tok_emb.map((t, i) => t.add(pos_emb[i]));
  x = rmsnorm(x);

  for (let li = 0; li < n_layer; li++) {
    // ── Attention block ──────────────────────────────────────────────────────
    const x_residual = x;
    x = rmsnorm(x);

    // Project x into query, key, value spaces
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);

    // Append this position's k,v to the cache (so future tokens can attend to it)
    keys[li].push(k);
    values[li].push(v);

    const x_attn: Value[] = [];
    for (let h = 0; h < n_head; h++) {
      const hs = h * head_dim;

      // Slice out this head's portion of q, k, v
      const q_h = q.slice(hs, hs + head_dim);
      const k_h = keys[li].map((ki) => ki.slice(hs, hs + head_dim));
      const v_h = values[li].map((vi) => vi.slice(hs, hs + head_dim));

      // Attention scores: dot(q, k_t) / sqrt(head_dim)  for each past position t
      const attn_logits = k_h.map((kt) =>
        kt
          .reduce((acc, kj, j) => acc.add(q_h[j].mul(kj)), new Value(0))
          .div(Math.sqrt(head_dim))
      );

      // Softmax over scores → attention weights
      const attn_weights = softmax(attn_logits);

      // Weighted sum of values
      const head_out = Array.from({ length: head_dim }, (_, j) =>
        attn_weights.reduce(
          (acc, wt, t) => acc.add(wt.mul(v_h[t][j])),
          new Value(0)
        )
      );
      x_attn.push(...head_out);
    }

    // Project concatenated head outputs back to n_embd, then add residual
    x = linear(x_attn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((xi, i) => xi.add(x_residual[i]));

    // ── MLP block ────────────────────────────────────────────────────────────
    const x_residual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]); // expand: n_embd → 4*n_embd
    x = x.map((xi) => xi.relu());                    // non-linearity
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]); // contract: 4*n_embd → n_embd
    x = x.map((xi, i) => xi.add(x_residual2[i]));
  }

  // Final projection to vocabulary logits
  return linear(x, stateDict["lm_head"]);
}

// ─── Adam optimizer state ────────────────────────────────────────────────────
const learning_rate = 0.01;
const beta1 = 0.85;
const beta2 = 0.99;
const eps_adam = 1e-8;
const m_buf = new Array(params.length).fill(0); // first moment (mean of grads)
const v_buf = new Array(params.length).fill(0); // second moment (mean of grad^2)

// ─── Training loop ───────────────────────────────────────────────────────────
const num_steps = 1000;
for (let step = 0; step < num_steps; step++) {
  // Pick the next document (cycle through dataset)
  const doc = docs[step % docs.length];

  // Tokenize: wrap with BOS on both sides so the model learns to start/end names
  const tokens = [BOS, ...doc.split("").map((ch) => charToId.get(ch)!), BOS];
  const n = Math.min(block_size, tokens.length - 1);

  // Forward pass: process token-by-token, accumulate loss
  const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
  const values_cache: Value[][][] = Array.from({ length: n_layer }, () => []);
  const losses: Value[] = [];

  for (let pos_id = 0; pos_id < n; pos_id++) {
    const token_id = tokens[pos_id];
    const target_id = tokens[pos_id + 1];
    const logits = gpt(token_id, pos_id, keys, values_cache);
    const probs = softmax(logits);
    // Cross-entropy loss: -log(probability of the correct next token)
    losses.push(probs[target_id].log().neg());
  }

  // Average loss across all positions
  const loss = losses
    .reduce((a, b) => a.add(b), new Value(0))
    .div(n);

  // Backward pass: compute gradients for all parameters
  loss.backward();

  // Adam update
  const lr_t = learning_rate * (1 - step / num_steps); // linear LR decay
  for (let i = 0; i < params.length; i++) {
    const g = params[i].grad;
    m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * g;
    v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * g * g;
    const m_hat = m_buf[i] / (1 - Math.pow(beta1, step + 1)); // bias correction
    const v_hat = v_buf[i] / (1 - Math.pow(beta2, step + 1));
    params[i].data -= lr_t * m_hat / (Math.sqrt(v_hat) + eps_adam);
    params[i].grad = 0; // reset gradient for next step
  }

  process.stdout.write(
    `\rstep ${String(step + 1).padStart(4)} / ${num_steps} | loss ${loss.data.toFixed(4)}`
  );
}

// ─── Inference ───────────────────────────────────────────────────────────────
const temperature = 0.5; // lower = more focused, higher = more random
console.log("\n--- inference (new, hallucinated names) ---");
for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
  const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
  const values_cache: Value[][][] = Array.from({ length: n_layer }, () => []);
  let token_id = BOS;
  const sample: string[] = [];

  for (let pos_id = 0; pos_id < block_size; pos_id++) {
    const logits = gpt(token_id, pos_id, keys, values_cache);
    // Divide logits by temperature before softmax to control randomness
    const scaled = logits.map((l) => l.div(temperature));
    const probs = softmax(scaled);
    token_id = randChoices(vocab_size, probs.map((p) => p.data));
    if (token_id === BOS) break; // BOS also acts as EOS
    sample.push(uchars[token_id]);
  }

  console.log(`sample ${String(sampleIdx + 1).padStart(2)}: ${sample.join("")}`);
}
