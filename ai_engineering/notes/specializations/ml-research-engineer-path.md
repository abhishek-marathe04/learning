# ML/AI Research Engineer — Personal Training Path

> A first-principles, Karpathy-style path from senior software engineer to ML/AI research engineer.
> Built around **5–8 hrs/week**, **<$200 budget**, and the goal of training/fine-tuning LLMs and contributing to OSS — not certificate chasing.

**Estimated duration:** 24–30 months at 5–8 hrs/week.
**Owner:** Abhishek
**Last updated:** May 2026

---

## The Philosophy First

Three rules before the curriculum:

1. **Build before you abstract.** Don't touch PyTorch's `nn.Transformer` until you've written attention from scratch in NumPy. Don't use HuggingFace `Trainer` until you've written a training loop manually.
2. **One paper per week, reproduced.** Reading without reproducing is theatre. Reproducing forces real understanding.
3. **Public learning loop.** Every module ends in a writeup or repo. This *is* your evaluation.

---

## The Path — 6 Phases

### Phase 0: Math Foundations (8–10 weeks)

You can't skip this for a research engineer goal. But you don't need a math degree either — you need *working fluency* in three areas.

**Linear Algebra**
- 3Blue1Brown's *Essence of Linear Algebra* (free, YouTube) — for intuition
- Gilbert Strang's MIT 18.06 (free, MIT OCW) — for rigor
- Focus: vectors, matrices, eigenvalues/eigenvectors, SVD, matrix calculus
- Skip: determinants beyond intuition

**Calculus & Matrix Calculus**
- 3Blue1Brown's *Essence of Calculus* (free, YouTube)
- *Matrix Cookbook* (free PDF) as reference
- The thing that actually matters: chain rule on tensors. Most people fail at backprop because they never internalized this.

**Probability & Statistics**
- Harvard's Stat 110 by Joe Blitzstein (free, YouTube + free textbook)
- Focus: distributions, expectation, Bayes, MLE, KL divergence, entropy

**Self-evaluation for Phase 0**
- Can you derive the gradient of softmax cross-entropy by hand?
- Can you explain why we use log-likelihood instead of likelihood?
- Can you compute the SVD of a 3x3 matrix on paper?

**Cost:** $0

---

### Phase 1: Classical ML (6–8 weeks)

Skipping this is the #1 mistake LLM-era learners make. Concepts like regularization, bias/variance, gradient descent, and kernels have nothing to do with deep learning and everything to do with *thinking like an ML person*.

**Resources**
- Andrew Ng's **CS229 (Stanford, free YouTube)** — the full Stanford lectures, *not* the Coursera version
- *Pattern Recognition and Machine Learning* by Bishop — canonical book, free PDF from Microsoft Research; read selectively

**Build**
- Implement linear regression, logistic regression, k-means, a decision tree, and a small SVM **from scratch in NumPy**
- Compare against scikit-learn
- Push to GitHub

**Self-evaluation**
- Can you explain bias-variance tradeoff to a junior dev without ML jargon?
- Can you derive the closed-form solution for linear regression?
- Do you understand why logistic regression is "linear"?

**Cost:** $0

---

### Phase 2: Deep Learning Foundations — The Karpathy Track (10–12 weeks)

This is the gold standard. Double down on what you already started.

**Resources**
- **Karpathy's *Neural Networks: Zero to Hero* (free, YouTube)** — finish all of it, in order, *typing every line yourself*. No copy-paste. The micrograd → makemore → GPT progression is the best ML pedagogy that exists.
- **Fast.ai Part 1 (free)** — totally different angle (top-down vs Karpathy's bottom-up). The combination is unreasonably effective.
- *Deep Learning* by Goodfellow, Bengio, Courville (free online) — reference, not linear reading

**Build**
- Finish the Karpathy series end-to-end with your own annotations
- Train a small CNN on CIFAR-10 from scratch
- Train an RNN/LSTM on text from scratch (don't skip this — understanding why transformers won requires understanding what they replaced)
- **Capstone:** GPT-2 small (~124M params) trained from scratch on a small corpus on the Mac Mini M4 Pro using MLX. This is the rite of passage.

**Self-evaluation**
- Can you implement attention from scratch in NumPy, no PyTorch?
- Can you explain why we need positional encodings?
- Can you debug a training loop where loss goes to NaN?

**Note:** Do not open Raschka's *Build an LLM From Scratch* yet. It belongs in Phase 3 — opening it now skips the bottom-up intuition Karpathy is building.

**Cost:** $0

---

### Phase 3: Transformers & LLM Internals (12–16 weeks)

Going *much* deeper than the "Tokens to Trained Weights" deck.
**Spine of this phase:** Sebastian Raschka — *Build a Large Language Model (From Scratch)*. Already owned. Use it as the daily driver, with CS336 as supplementary depth.

**Important sequencing note:** Do *not* open Raschka before finishing Karpathy's Zero to Hero in Phase 2. Karpathy builds intuition from scratch in plain Python/NumPy; Raschka uses `torch.nn` from chapter 2. Reading Raschka first means skipping the "why." Reading him *after* Karpathy means every shortcut he takes is one you understand.

**Three-pass plan with Raschka**

*Pass 1 — Build (weeks 1–8): Code along, type every line yourself.*
- Chapters 1–7: tokenization → attention → full GPT → pretraining → fine-tuning
- After every chapter, close the book and re-implement the key component from memory in a blank file
- Specific checkpoint: after chapter 4 (multi-head attention), implement MHA from memory before moving on. If you can't, redo the chapter.
- Output: a working GPT-2-style model in your own repo, trained on a small corpus

*Pass 2 — Modernize (weeks 9–10): Rewrite Raschka's model with modern components.*
- Replace LayerNorm → RMSNorm
- Replace learned positional embeddings → RoPE
- Replace GELU MLP → SwiGLU
- Replace MHA → GQA (Grouped Query Attention)
- This forces you to read the LLaMA paper and translate it into working code. Single best exercise of Phase 3.

*Pass 3 — Extend (weeks 11–16): Add real-world features and reproduce papers.*
- Add KV cache for inference
- Implement LoRA fine-tuning manually (read the LoRA paper, no PEFT library)
- Implement a simple DPO loop
- Each is a 1–2 week project that produces a portfolio piece

**Supplementary depth (run alongside, not instead)**
- **Stanford CS336** — Language Modeling from Scratch (2025, free YouTube). Use for what Raschka glosses: data curation, scaling laws, evals, training dynamics. Watch ~2 lectures/week from week 11 onward.
- HuggingFace's free LLM course — light reference

**Dropped from original plan**
- CS224N — good but redundant once you have Karpathy + Raschka + CS336. Keep as optional NLP reference.

**Papers to reproduce (one per 2 weeks, integrated into passes 2–3)**
1. *Attention is All You Need* (Vaswani 2017) — covered in Pass 1
2. *GPT-2* (Radford 2019) — covered in Pass 1
3. *LLaMA* (Touvron 2023) — Pass 2 modernization
4. *LoRA* (Hu 2021) — Pass 3
5. *RLHF / InstructGPT* (Ouyang 2022) — Pass 3
6. *DPO* (Rafailov 2023) — Pass 3
7. *Mixture of Experts* (Switch Transformer / Mixtral)
8. *FlashAttention* (Dao 2022)

For each: read → take notes → modify your Raschka-based codebase → write a blog post or LinkedIn breakdown. Modifying *your own* code (instead of nanoGPT) means you understand every line you touch. This is your portfolio.

**Trap to avoid:** Do not read Raschka passively like a textbook. The code is the lesson. Reading on the couch without typing produces the illusion of understanding. The whole value is in the typing.

**Self-evaluation**
- Can you explain why FlashAttention is faster *without* being faster mathematically?
- Can you implement multi-head attention with KV cache from scratch?
- Can you explain rotary position embeddings to a colleague?

**Cost:** $0

---

### Phase 4: Training Infrastructure & Systems (8–10 weeks)

The most underrated phase. Separates "I understand transformers" from "I can train models."

**Resources**
- *How to Scale Your Model* (DeepMind, free online book — released 2024)
- Stas Bekman's *ML Engineering Book* (free on GitHub)
- HuggingFace's *Ultra-Scale Playbook* (free)
- *Programming Massively Parallel Processors* (Hwu/Kirk) — used copy
- Horace He's blog posts and talks (PyTorch internals)

**Build**
- Multi-GPU training with DDP on a rented GPU (vast.ai or runpod.io — $0.30–1/hr for an A100)
- Implement a custom CUDA kernel (start with vector add, work up to a fused operation)
- Profile a training run, identify bottleneck, optimize it
- Implement gradient checkpointing manually
- Train a 1B+ param model on rented compute (~$50–100 budget)

**Self-evaluation**
- Can you explain why ZeRO-3 trades compute for memory?
- Can you debug a deadlock in distributed training?
- Can you read an Nsight profile and identify the bottleneck?

**Cost:** ~$100–150 (cloud compute + maybe one used book) — this is where most of the $200 budget goes.

---

### Phase 5: Research Skills & OSS Contribution (Ongoing, 6+ months overlap)

This meta-phase runs alongside Phases 3 and 4.

**Research skill building**
- Follow arxiv-sanity and paperswithcode.com
- Sergey Karayev's *How to Read a Paper* (free, online)
- Subscribe to: Sebastian Raschka's *Ahead of AI*, *Interconnects* (Nathan Lambert), *The Gradient*
- Watch Yannic Kilcher's paper breakdowns
- **Goal: 1 paper read deeply per week, with notes**

**OSS contribution path**
- Start: LangGraph/LangChain (already in motion)
- Widen to: HuggingFace Transformers, vLLM, llama.cpp, MLX, PyTorch
- Path: read issues → fix small bugs → add features → propose RFCs
- **Goal by month 18:** meaningful contributions in 2–3 major repos

**Public knowledge loop**
- Keep the Obsidian wiki going
- Convert to public blog posts on meatier topics
- Pivot LinkedIn/X content from "AI engineer" → "research engineer thinking out loud"

**Capstone projects (pick 2–3 over the journey)**
- Reproduce a recent paper (Mamba, DeepSeek-V2 architecture, an RLHF variant)
- Train a small but actually-good domain LLM (Cricket-LLM? Hindi/Marathi LLM?)
- Build a benchmark/eval harness that gets used by others

**Cost:** $0 (compute already in Phase 4)

---

## How to Evaluate Yourself — The Real Tests

Forget certificates. These are the actual signals.

### Tier 1 — Foundational fluency (after Phase 2, ~6 months in)
- Implement a Transformer from scratch in 200 lines, with training loop, in under 4 hours
- Read a recent ML paper and explain it correctly to a peer
- Debug a training run that's not converging

### Tier 2 — Practitioner level (after Phase 3, ~12 months in)
- Reproduce a paper end-to-end given the paper alone
- Give technically correct answers to questions like "why does attention scale with √d_k?"
- Fine-tune an open model and beat baseline on a custom task

### Tier 3 — Research engineer level (after Phase 4–5, ~18–24 months in)
- Train a model from scratch on rented GPUs and ship it
- Non-trivial code contributed to a major OSS project
- A real research engineer can read your code and not cringe
- You've written something — paper, blog, repo — that other people in the field have engaged with

### Honest external benchmarks
- Apply to **ML Collective** or **Cohere For AI** open research programs (free, by application — *the* legit communities for this goal)
- Eleuther AI Discord — hang out, contribute to projects
- Apply to Anthropic/DeepMind/etc. residency programs as a forcing function (even prep teaches you a lot)

---

## Total Budget Breakdown

| Item | Cost |
|---|---|
| All courses (Karpathy, CS229, CS336, Fast.ai) | $0 |
| All free textbooks (Bishop, Goodfellow, etc.) | $0 |
| Sebastian Raschka — *Build a Large Language Model (From Scratch)* | **Owned ✓** |
| Cloud GPU compute (Phase 4) | ~$100–150 |
| Optional: one CUDA / ML Engineering book, used | ~$30–40 |
| **Total remaining** | **~$130–190** |

The expensive courses (Coursera specializations, DeepLearning.AI) are *not* what gets you to research engineer. They're great for credentialing applied roles. This goal needs deeper material, and almost all of it is free.

---

## This Week — Concrete Starting Actions

1. Drop the current DeepLearning.AI plan — it's optimized for a different goal.
2. Start Phase 0 math (3Blue1Brown linear algebra series) — 1 video/day, 30 min each.
3. In parallel, continue Karpathy's Zero to Hero (Phase 2 but most motivating).
4. Set up a public learning repo on GitHub: `ml-from-scratch-journey` or similar. Commit something every week. This becomes your evidence.

---

## Progress Log

> Use this section to track weekly progress. Date, phase, what was studied, what was built, what's next.

| Date | Phase | Studied | Built / Output | Next |
|---|---|---|---|---|
|   |   |   |   |   |
