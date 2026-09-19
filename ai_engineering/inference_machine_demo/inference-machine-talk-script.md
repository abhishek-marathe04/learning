# The Inference Machine — Presenter Script

**Devocracy · ~20 min · slides + live demo (`index.html`)**

## How to read this script

- Normal paragraphs = **words to say** (a spine, not a cage — deviate freely).
- *Italic in brackets* = **stage directions** (what you do / what the audience sees).
- `[SLIDE n]` = advance the deck. `[DEMO: view]` = alt-tab to the browser app.
- `⌨ key` = a keyboard press in the demo (matches the app's shortcuts).
- `(m:ss)` = running clock, so you can feel the pace.

## Pre-flight (do this before you're mic'd)

- Deck open, presenter view on, **slide 1 up**.
- `index.html` open in the browser, **full-screen**, on **view 1 (Fit)**, freshly **Reset** (`R`).
- Alt-tab (or a second display) rehearsed so slide→demo→slide is one motion, not a fumble.
- Seed left at default (so every demo runs exactly as you rehearsed).
- Backup: if a demo ever misbehaves live, press `R` and keep talking — never debug on stage.

---

## 1 · The hook  (0:00)

`[SLIDE 1 — Inside the Inference Machine]`

Quick show of hands — who's typed something into an AI chatbot in the last day or so?

*(let hands go up)*

Keep it up if you've got a clear picture of what actually happened on the other side — between you hitting enter, and the words showing up.

*(most hands drop)*

Right. That gap — the machinery between "enter" and the answer — that's the whole talk. And it's not the machine-learning part. It's *our* part: the infrastructure.

`[SLIDE 2 — AI is an infrastructure problem]`

Here's the reframe I want you to leave with. Training a model is like baking a cake — you do it once, it's expensive, it's a science project. **Inference** is the restaurant: you have to serve that cake hot, thousands of times a second, without going bankrupt. The hyperscalers are pouring trillions into GPUs for the restaurant, not the bakery. And running the restaurant — keeping GPUs fed, cached, and balanced — is a DevOps and SRE problem wearing a new costume. That's us.

`[SLIDE 3 — Where we're going]`

So here's the plan. We'll look at what a model even *is*, build up the hardware mental model, then walk one request through the machine — and that's where the counter-intuitive punchline lands. Then we make it fast, and scale it to a fleet. And I'm going to *show* you most of it running, not just tell you.

---

## 2 · Anatomy and hardware  (2:00)

`[SLIDE 4 — A model is two things]`

A model is two things. There's the **transformer** — a few hundred lines of fixed math, the recipe. It's identical whether it's a 1-billion or a 400-billion parameter model, and it knows *nothing*. And there are the **weights** — billions of numbers, frozen after training, shipped as a giant file. All the knowledge lives in the weights. Rule of thumb to burn in: in half-precision, it's two bytes per parameter. So a 70-billion model is a **140-gigabyte** blob. Hold onto that number.

`[SLIDE 5 — Why a CPU can't, and a GPU can]`

Why not just run that on a CPU? A CPU is a handful of Formula-1 cars — a few very fast cores, brilliant for serial work. A forward pass isn't one urgent parcel; it's a *million* parcels — huge matrix multiplies where every element is independent. For that you don't want F1 cars, you want ten thousand delivery bikes. That's a GPU. Inference is embarrassingly parallel — that's the whole reason it lives on GPUs.

`[SLIDE 6 — Where the weights actually live]`

But before three numbers, one picture — because it makes those numbers obvious. Think of memory as a hierarchy, from big-and-slow up to small-and-fast. At the bottom, **disk** — where the weights file rests. Above it, **system RAM** — roomy, but it talks to the GPU across a slow-ish bus. Then, on the GPU board itself, right next to the cores, **VRAM** — this is the fast, expensive memory, and it's *small*. And at the top, the **cores** that do the math.

Two things to notice, and only two. First: the weights get loaded up this ladder — disk to RAM to VRAM — exactly **once**, at startup. That's the grey arrow. Second, and this is the one that matters: the trip that happens on *every single token* is this one — VRAM up to the cores. *(point to the green arrow)* The cores can only work on data that's physically reached them, and the speed of that trip is what we call **bandwidth**.

Why does VRAM even exist? Because system RAM is far too slow to feed the cores — so the GPU carries its own fast memory right beside them. The price you pay is that it's tiny and expensive. And that — *(point)* — is exactly the capacity limit that runs you out of memory.

`[SLIDE 7 — The GPU trinity]`

So the trinity is really just three questions about that picture. **Compute** — how fast the cores do math. **Capacity** — how much that little VRAM can hold. And **bandwidth** — how fast VRAM feeds the cores. Most people fixate on compute, the big TFLOPs number on the spec sheet. By the end of the next ten minutes you'll know why that's usually the *wrong* thing to fixate on.

Let me show you the first two live.

---

## 3 · DEMO 1 — "Will it fit?"  (3:30)

`[DEMO: view 1 — Fit]`  *(press ⌨ `1`; confirm it's Reset)*

This is a VRAM budget. Weights in orange, KV cache in cyan, overhead in grey — against the card's capacity line. Right now: Llama-70B, half-precision, on an 80-gig H100.

*(gesture at the bar already over the line)*

Notice we're already red. The **weights alone** — 140 gig — don't fit on an 80-gig card. Before a single user shows up. That one fact drives a huge amount of infra design.

So what do we do? Two options. One —

*(⌨ `T` to flip quantization toward INT4)*

— quantize. Store the numbers in fewer bits. Watch the orange collapse... and now it fits. We traded a little accuracy for a lot of room.

But here's the sneaky one. Watch the cyan.

*(⌨ `B` — concurrency ramps; KV cache climbs and crosses the line into red / OOM)*

That's the KV cache. It grows with every user and every token of context — and it grows *linearly*, right past the line, into out-of-memory. So your capacity planning isn't "does the model fit." It's "does the model fit **plus** every conversation my users are holding at once." Capacity is what pages you at 3am.

*(press ⌨ `R` to reset before leaving)*

---

## 4 · The serving stack, and one request  (5:30)

`[SLIDE 9 — From a file on disk to an API]`

Quick tour of the stack. **PyTorch** is the engine that runs the math. **vLLM** is the server that turns that weights file into a running web service — it loads the model, manages the KV cache, batches requests. And it hands you an ordinary API. The thing to internalise: that server is a *heavy beast*. Every replica keeps the entire model resident in VRAM for its whole life. So "just add more pods" is an expensive sentence. Scaling inference is nothing like scaling a stateless web app.

`[SLIDE 10 — Two very different phases]`

Now let's walk one request through it. Every request has two phases, and they could not be more different. **Prefill** reads your whole prompt and produces the first token — we measure it as *time to first token*. **Decode** then writes the answer one token at a time, looping — we measure *time per output token*.

`[SLIDE 11 — Prefill is compute-bound]`

Prefill is the model *reading*. Your whole prompt goes through in one big parallel pass — one enormous matrix multiply. Every core is busy. This phase is **compute-bound**: double the prompt, roughly double the work, and your time-to-first-token goes up. Long context and RAG make this heavy.

`[SLIDE 12 — Decode is memory-bandwidth-bound]`

And this — this is the slide. If you remember one thing, remember this one.

Decode is the model *writing*, one token at a time. And here's the trick: to generate **one** token, the GPU has to stream **every single weight** out of VRAM — all 140 gigabytes — do a tiny sliver of math, emit one token... and then do it *all over again* for the next token. The math units sit there idle, waiting for bytes. So the speed limit isn't compute. It's **memory bandwidth**.

I know that sounds abstract. Let me prove it to you.

---

## 5 · DEMO 2 — the money moment  (7:30)

`[DEMO: view 2 — Speed]`  *(press ⌨ `2`)*

Two big readouts: tokens per second, and compute utilization. Same 70-billion model on an H100. Look — about 24 tokens a second, and compute utilization is... **zero-point-three percent**. The most expensive part of the chip is doing basically nothing.

Now. Conventional wisdom says: want it faster? Buy more compute. Let's test that. I'm going to crank the compute slider all the way to the max — B200 territory, more than double the FLOPs.

*(⌨ `B` — compute sweeps min→max)*

*(let the silence sit)*

...Nothing. Tokens per second didn't move. We threw a fortune of extra compute at it and got nothing, because compute was never the bottleneck.

Now watch what happens when I touch the *other* slider — bandwidth.

*(⌨ `→` a few times to raise bandwidth)*

There it goes. Same everything, more bandwidth, and the token rate climbs. That's the entire thesis of this talk in one dial: **generation speed tracks memory bandwidth, not compute.** It changes which GPUs you buy and how you read your dashboards.

*(optional, if time — ⌨ `T` to show batching preview)* And there's one honest escape from this: read the weights once, and serve *many* users on that single read. That's batching — which is exactly where we're going.

*(⌨ `R` to reset)*

---

## 6 · KV cache and throughput  (10:00)

`[SLIDE 13 — KV cache]`

First, the thing that makes decode even possible: the **KV cache**. Attention needs the keys and values of every previous token. Without a cache you'd recompute the whole history on every single step — quadratic, absurd. So instead you compute each token's keys and values once and keep them. Decode drops to linear. The catch — and we saw it — it lives in VRAM and grows with every token and every user. Speed you buy with memory you were counting on.

`[SLIDE 14 — Two more levers: reuse and batch]`

Two levers fall right out of this. **Prefix caching**: if a thousand requests share the same giant system prompt, compute its keys and values *once* and reuse them — time-to-first-token collapses. And **continuous batching**: since a token step re-reads all the weights anyway, pack many users into that one read. "Continuous" means requests join and leave the batch mid-flight, so the GPU never idles. Let me show you why that's the single biggest throughput win we have.

---

## 7 · DEMO 3 — static vs continuous batching  (11:00)

`[DEMO: view 3 — Batching]`  *(press ⌨ `3`)*

Same GPU, same requests, two scheduling strategies, side by side. On the left, **static** batching: form a batch, and everyone waits for the *slowest* request in it to finish before anyone new gets in. On the right, **continuous**: the moment a slot frees up, the next request slots in.

*(⌨ `Space` to run both)*

Watch the utilization numbers. Left side — lots of idle slots, sitting around 40, 50 percent, because one long request holds the whole batch hostage.

*(optional ⌨ `B` to inject a long request into the static batch)*

See that? One long generation, and its entire batch stalls behind it — head-of-line blocking.

Right side — slots refill instantly, utilization pinned up near 100. And look at the throughput counters: continuous is running two-to-three times the tokens per second, on identical hardware. That's not a new chip. That's just *scheduling*. That's the vLLM magic.

*(⌨ `R` to reset)*

---

## 8 · Metrics, and scaling out  (13:00)

`[SLIDE 15 — Sharding]`  *(keep this one quick)*

One more single-GPU trick: when the model genuinely doesn't fit — 140 gig on an 80-gig card — you **shard** it, splitting the model across GPUs so their memory adds up. The cost is communication: the cards have to talk on every layer, so you want fast interconnect. Sharding buys you capacity, not free speed.

`[SLIDE 16 — The numbers on your dashboard]`

So these are the three numbers you'll actually get paged about. Time-to-first-token — how snappy it feels. Time-per-output-token — how fast it streams. And throughput — tokens per second, which is basically your cost per token. And there's a permanent tension: cram more into the batch and throughput soars, but each user's first token comes a little later. Tuning inference is picking a point on that curve — and it's different for autocomplete than for a batch summariser.

`[SLIDE 17 — Scale out, and round-robin betrays you]`

Now the last move: go from one replica to many. And the moment you do, your trusty load balancer betrays you. A normal load balancer assumes every backend is identical and stateless — perfect for web servers. But an LLM pod is *not* stateless — it's carrying a warm KV cache. Round-robin ignores that: it'll send a request whose prefix is already warm on pod A over to cold pod B, and pay for the whole prompt again. It ignores memory pressure, so it piles onto a pod that's about to OOM. It can't tell a 32-thousand-token prefill from a one-word reply. Let me show you what that costs — and how we fix it.

---

## 9 · DEMO 4 — the climax: smart routing  (15:00)

`[SLIDE 18 — llm-d]` *(leave it up for one beat, then to the demo)*

The fix is a Kubernetes-native router — this is what a project like **llm-d** does. Let me put it under load.

`[DEMO: view 4 — Fleet]`  *(press ⌨ `4`)*

Six replica pods, real traffic, and most requests share one of a few big system prompts. We're on plain **round-robin** right now.

*(⌨ `Space` to run; let it reach steady state)*

Watch the dashboard. Cache-hit rate — low, down around 20 percent. p95 time-to-first-token — high, spiky. And look at the pods: some are thrashing and evicting, some are sitting idle. It's chaos, and it's *evenly distributed* chaos.

Now — same traffic, I don't reset anything — I just flip the router to **cache-aware**.

*(⌨ `T`)*

*(let the dashboard visibly move)*

Watch it flip. Cache hits climbing — 70, 80 percent — because requests now go to the pod that already has their prefix warm. p95 latency dropping off a cliff. Memory balancing out across the fleet. Same hardware, same requests — we just made the router *understand* that these servers have state. That's the difference between treating LLMs like web apps and treating them like what they are.

*(⌨ `R` to reset)*

---

## 10 · Land it  (18:00)

`[SLIDE 19 — The mental model to walk out with]`

So, five things to walk out with. Knowledge is in the weights, and the weights are big enough to define the whole problem. Think in the trinity — compute, capacity, bandwidth. **Decode is memory-bound** — more FLOPs won't save you, batching will. The KV cache is the hidden cost that quietly eats your VRAM. And serving LLMs is not serving web apps — it's stateful, so your routing has to be inference-aware.

`[SLIDE 20 — Thanks / questions]`

That's the machine. None of it was magic — it was bytes, bandwidth, and scheduling. Everything you just watched is a couple hundred lines of simulation; the concepts are the same ones running the trillion-dollar restaurants.

I'll leave the demo up. And I mean the invitation on the slide — come argue with me about whether your GPU is compute-bound or memory-bound. Thank you.

*(open the floor)*

---

## Q&A — likely questions and quick answers

- **"Isn't prefix caching just prompt caching from the API providers?"** Same idea, one layer down — this is the server-side mechanism that makes the feature possible.
- **"How does this compare to NVIDIA Dynamo?"** Same disaggregation and cache-aware ideas; Dynamo sits above vLLM and is more tied to NVIDIA's stack, llm-d is Kubernetes-native via the Gateway API. Pick based on where you already live.
- **"What's PagedAttention?"** The vLLM trick that stops the KV cache from fragmenting VRAM — think virtual memory paging, for the cache.
- **"Why FP16 = 2 bytes, does quantization hurt quality?"** A little, and it depends; 8-bit is usually free-ish, 4-bit is where you start testing carefully. The demo lets you feel the memory trade instantly.
- **"Real numbers?"** They're approximate and version-dependent on purpose — I'm teaching scale, not spec sheets.
- **If you don't know:** "Good question — I don't want to guess on stage; grab me after and we'll dig in." (Always better than bluffing.)

## Timing map and safety valves

Total is now ~21 min with the memory-hierarchy slide added. To hold a hard 20-min cap, see the cut list.

| Section | Ends around |
|---|---|
| Hook + framing (1–3) | 2:00 |
| Anatomy + hardware (4–7, incl. memory hierarchy) | 4:15 |
| **Demo 1 · Fit** (covers slide 8) | 6:15 |
| Stack + prefill (9–11) | 8:15 |
| **Demo 2 · Speed** (money) | 10:45 |
| KV + levers (13–14) | 11:45 |
| **Demo 3 · Batching** | 13:45 |
| Sharding + metrics + round-robin (15–17) | 15:45 |
| **Demo 4 · Fleet** (climax) | 18:45 |
| Takeaways + close (19–20) | 20:45 |

**If you're running long, cut in this order:** (1) the `B` long-request injection in Demo 3; (2) compress the memory-hierarchy slide (6) to two spoken sentences on slide 5 and skip the slide; (3) the whole Batching demo — describe it from slide 14 instead; (4) the sharding slide to two sentences. **Never cut Demo 2 or Demo 4** — those are the two beats people remember.

**If you have extra time:** linger on Demo 2 (sweep compute twice), or in Demo 4 add the `least-loaded` router to show even "smart" load balancing isn't enough without cache-awareness.
