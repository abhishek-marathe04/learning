# Build Spec — "The Inference Machine" live demo app

A single-file, offline, browser-based interactive simulator to run **live** during a 20-minute
tech talk on LLM inference. It teaches — by letting the presenter *break things on stage* — why
decode is memory-bandwidth-bound, what the KV cache costs, how batching wins, and why cache-aware
routing beats round-robin.

There is **no real model**. Everything is simulated with correct first-order math. That is a
feature: the simulator exposes the mechanics a real model hides.

---

## 0. Hard constraints (do not violate)

- **One file.** A single `index.html` with inline `<style>` and `<script>`. No build step.
- **Zero dependencies, zero network.** No npm, no CDN, no external fonts, no API calls. Must run
  by double-clicking the file on a locked-down corporate laptop, fully offline. Use vanilla JS,
  the system font stack, hand-drawn SVG/DOM for all visuals. (This rules out Chart.js, React, etc.)
- **Projector-first.** Target 1920×1080, 16:9. Dark background, very high contrast, large type
  (big readouts 48–96px). Assume the back row is 20 metres away.
- **Deterministic.** All randomness comes from a seeded PRNG. Same seed ⇒ identical run every time.
- **Live-safe.** Every view has a one-key **Reset** to a known-good state. Never crash; clamp all
  inputs; if a number would be absurd, cap it and show the cap.

---

## 1. Global design system

### Palette (CSS custom properties — matches the companion slide deck)
```
--bg:      #0B1120   /* app background */
--card:    #15203A   /* panels/cards */
--card2:   #1C2A4A   /* raised/secondary */
--border:  #30416A
--text:    #EAF1FB
--muted:   #94A7C6
--compute: #FF8C42   /* ORANGE = compute-bound  (use everywhere compute is the theme) */
--memory:  #2DD4E8   /* CYAN   = memory/bandwidth-bound */
--green:   #3DDC84   /* throughput / good / cache-hit */
--purple:  #A78BFA   /* routing / orchestration */
--red:     #F4707E   /* danger / OOM / cache-miss / p95 spike */
```
**Colour is semantic and consistent with the talk:** orange = compute, cyan = memory/bandwidth,
green = throughput/hits, red = OOM/miss, purple = routing. Never use colour decoratively in a way
that fights this.

### Typography
System stack: `-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif`.
Monospace stack (for numbers, formulas, code-like labels): `ui-monospace, "SF Mono", Menlo, Consolas, monospace`.
Big metric readouts in monospace so digits don't jitter as they animate.

### Layout shell
- Fixed top bar: app title left; a 4-button nav center — **`1 Fit` · `2 Speed` · `3 Batching` · `4 Fleet`**;
  a small "Act I — The single GPU / Act II — The fleet" label; seed box + global Reset right.
- One view visible at a time (the others are hidden, state preserved). Switching views must be instant.
- Each view: a **left control rail** (sliders, toggles, presets — big hit targets) and a **main stage**
  (the visualization + the big readouts). Controls never overlap the stage.

### Keyboard map (essential — the presenter drives with this)
| Key | Action |
|-----|--------|
| `1`/`2`/`3`/`4` | Jump directly to Fit / Speed / Batching / Fleet |
| `Space` | Play/Pause the running simulation (Batching, Fleet) |
| `R` | Reset the **current** view to its known-good default |
| `→` / `←` | Nudge the "primary slider" of the current view up/down (see each view) |
| `B` | Fire the view's **"break it"** action (defined per view) |
| `T` | Toggle the view's **primary toggle** (quant / bound / batch-mode / router) |
| `+` / `-` | Sim speed up / down (Batching, Fleet) |
Show a small "?" overlay listing these (toggled by `?`).

### Simulation engine (shared)
- A single loop via `requestAnimationFrame` with a **fixed-timestep accumulator** so speed control
  and determinism both work: accumulate real ms × speedMultiplier, consume in fixed `TICK_MS` steps.
- One decode **tick** = one decode step for every active sequence.
- **Seeded PRNG:** implement `mulberry32(seed)`. All arrival times, prompt/output lengths, and
  prefix choices draw from it. The Batching view's two modes and the Fleet view's two routers must
  consume the **same seeded stream** so comparisons are provably fair (identical request sequence).
- Decouple sim state from render; render reads state each frame.

### Units convention (be consistent everywhere)
- Bytes are bytes. `1 GB = 1e9 B` (decimal — gives clean teaching numbers like 70B→140GB).
- Bandwidth in `TB/s = 1e12 B/s`. Compute in `TFLOP/s = 1e12 FLOP/s`.
- Display GB with one decimal; tok/s as integers; percentages with one decimal.

---

## 2. Shared reference data

### GPU registry
| id | label | VRAM (GB) | Bandwidth (TB/s) | Compute FP16 (TFLOP/s) |
|----|-------|-----------|------------------|------------------------|
| a100 | A100 80GB | 80 | 2.0 | 312 |
| h100 | H100 80GB | 80 | 3.35 | 990 |
| h200 | H200 141GB | 141 | 4.8 | 990 |
| b200 | B200 192GB | 192 | 8.0 | 2250 |

(Numbers are approximate/marketing-round on purpose — we teach scale, not spec-sheet precision.
Add a one-line disclaimer in the UI footer.)

### Model registry
| id | label | params (B) | layers | hidden | heads | kv_heads | head_dim |
|----|-------|-----------|--------|--------|-------|----------|----------|
| l3_8b | Llama-3 8B | 8 | 32 | 4096 | 32 | 8 | 128 |
| l3_70b | Llama-3 70B | 70 | 80 | 8192 | 64 | 8 | 128 |
| l31_405b | Llama-3.1 405B | 405 | 126 | 16384 | 128 | 8 | 128 |
| custom | Custom | slider 1–500 | — | — | — | 8 | 128 |
For **custom**, estimate `layers ≈ round(9 * params_B**0.33) ` clamped to [16, 140] (rough, only for
the shape of the curve; label it "estimated"). kv_heads=8, head_dim=128 fixed.

### Core formulas (implement exactly)
```
bytesPerParam(quant):  FP16 -> 2,  INT8 -> 1,  INT4 -> 0.5
weightsBytes         = params_B * 1e9 * bytesPerParam(quant)

# KV cache uses FP16 (2 B) regardless of weight quant (default; keep it simple + honest)
kvBytesPerToken      = 2 /*K&V*/ * layers * kv_heads * head_dim * 2 /*FP16*/
kvBytes(L, B)        = kvBytesPerToken * L /*context*/ * B /*concurrent seqs*/
overheadBytes        = 0.15 * (weightsBytes + kvBytes)          # activations + fragmentation
totalBytes           = weightsBytes + kvBytes + overheadBytes
capacityBytes(gpu, N)= gpu.VRAM_GB * 1e9 * N                    # N = number of GPUs (sharding)

# --- Decode roofline (per decode step, batch size b, context L) ---
bytesReadPerStep     = weightsBytes + kvBytesPerToken * L * b   # stream weights + read KV
flopsPerStep         = 2 * params_B*1e9 * b                     # ~2 FLOPs/param/token
t_mem                = bytesReadPerStep / (gpu.BW_TBs * 1e12)
t_compute            = flopsPerStep     / (gpu.TFLOPs  * 1e12)
tStep                = max(t_mem, t_compute)
boundBy              = (t_mem >= t_compute) ? "MEMORY" : "COMPUTE"
tokPerSec_total      = b / tStep
tokPerSec_user       = 1 / tStep
arithmeticIntensity  = flopsPerStep / bytesReadPerStep          # FLOPs per byte  (~= b at low L)
computeUtilPct       = 100 * (t_compute / tStep)                # math-unit busy fraction
ridgePoint(gpu)      = (gpu.TFLOPs*1e12) / (gpu.BW_TBs*1e12)    # AI where mem-bound flips to compute-bound
```

### Sanity checks (use these to verify the build is correct)
- Llama-3 8B: `kvBytesPerToken = 2*32*8*128*2 = 131072 B ≈ 128 KB/token`. 8K context ⇒ ~1.05 GB per user. Weights FP16 = 16 GB.
- Llama-3 70B: `kvBytesPerToken ≈ 320 KB/token`. Weights FP16 = **140 GB** ⇒ does **not** fit on an 80 GB card even at zero context. INT4 ⇒ 35 GB (fits).
- Roofline, **H100 + 70B FP16, b=1, L=2048**: `t_mem ≈ 0.042 s` ⇒ **~24 tok/s**, `computeUtil ≈ 0.3%`, `AI ≈ 1.0`, `ridge ≈ 295`. Cranking compute to B200's 2250 TFLOP/s leaves tok/s **unchanged**. Raising bandwidth to 8 TB/s ⇒ **~57 tok/s**. This is the money beat — verify it lands.

---

## 3. ACT I — "The single GPU"

### View 1 — `Fit`: "Will the model even fit?"  (anchors slides 6–7, 12)

**Purpose:** Make "the model doesn't fit" a thing they *watch happen*, and show KV cache growth is
the sneaky killer.

**Controls (left rail):**
- Model preset dropdown (registry) + params slider when Custom.
- Quantization segmented control: `FP16 | INT8 | INT4`  (**primary toggle**, key `T`).
- Context length slider `L`: 512 → 128,000 (log scale, snap to nice values).
- Concurrent sequences slider `B`: 1 → 256  (**primary slider**, keys `←`/`→`).
- GPU dropdown; GPU **count `N`** stepper 1–8 (for the sharding story).

**Stage:**
- A big horizontal **stacked bar**: `weights` (orange) + `KV cache` (cyan) + `overhead` (grey),
  drawn against a **capacity line** at `capacityBytes(gpu, N)`. Animate segment widths on change.
- If `totalBytes > capacity`: the bar turns red past the line, a large **`OUT OF MEMORY`** flag
  appears with "over by X GB", and the stage border pulses red once.
- Big readouts (monospace): `Weights`, `KV cache`, `Total / Capacity`, `KV per token`, and two
  derived helpers: **"Max users that fit @ this context"** and **"Max context that fits @ this users"**.

**"Break it live" (`B`):** ramps `B` (concurrency) upward until OOM, ~1.5s animated, so the KV
segment visibly grows and crosses the line. Follow-up beat: press `T` to INT4 → weights segment
collapses → fits again (the fix).

**Acceptance:** 70B/FP16/H100/N=1 shows OOM at the weights alone (140 > 80). INT4 fits. Pushing
context or users grows only the KV segment, linearly. `N=2` doubles the capacity line.

---

### View 2 — `Speed`: "The decode speed dial"  (THE money demo — slide 11)

**Purpose:** Prove, live, that generation speed tracks **memory bandwidth**, not compute.

**Controls (left rail):**
- GPU preset (sets compute + bandwidth) **and** two manual override sliders:
  - `Compute` (TFLOP/s): 100 → 2500.
  - `Bandwidth` (TB/s): 1 → 9   (**primary slider**, keys `←`/`→`).
- Model preset (sets weights). Batch size `b`: 1 → 512 (default **1**). Context `L` default 2048.
- A `bound-by` note is derived, not set.

**Stage — two things side by side:**
1. **Big readouts:** `tokens/sec` (per user, huge), `aggregate tok/s`, `Arithmetic Intensity`,
   `Compute utilization %`, and a **`BOUND BY: MEMORY|COMPUTE`** badge (cyan/orange).
2. **Interactive roofline chart** (hand-drawn SVG, log-log): x = arithmetic intensity, y = achievable
   throughput. Draw the diagonal **memory-bound** line (slope set by bandwidth) and the flat
   **compute-bound ceiling** (height set by compute); mark the **ridge point**; plot a dot at the
   current `(AI, throughput)`. The dot must sit ON the binding constraint.

**"Break it live" (`B`):** sweeps the Compute slider min→max over ~2s. Watch: `tok/s` barely moves,
the compute ceiling rises but the dot (low AI) stays pinned on the diagonal. Then the presenter
manually raises Bandwidth → the diagonal tilts up → the dot rises → `tok/s` climbs. `T` toggles a
"show what batching does" helper that bumps `b` and shows AI marching toward the ridge (hands off to View 3).

**Acceptance:** matches the sanity check above (H100/70B/b1 ⇒ ~24 tok/s, ~0.3% util, MEMORY).
Compute sweep leaves tok/s flat; bandwidth changes move it proportionally.

---

### View 3 — `Batching`: "Static vs continuous"  (slides 13, 15)

**Purpose:** Show the vLLM throughput win as a scheduling animation — the systems "aha".

**Model:** discrete ticks. `G` batch slots (default 8). A seeded request stream: Poisson arrivals
(rate slider), each request has `promptLen` and `outputLen` drawn from a **high-variance mixture**
(≈60% short outputs 5–25 tokens, ≈40% long 80–300) so the "wait for the slowest" pain is real.
Both modes below consume the **same seeded stream**.

- **Static batching:** admit up to `G` queued requests as a batch; run them; **all** slots stay
  occupied until the **longest** output in the batch finishes; finished slots idle (greyed) until
  then; only then admit the next batch.
- **Continuous batching:** keep up to `G` sequences active; each tick every active seq emits one
  token; the instant a seq finishes, free its slot and admit the next queued request immediately.

**Stage:** two panels side by side (`STATIC` | `CONTINUOUS`), each showing `G` slot rows with a
per-sequence progress bar (coloured per request), a greyed state for idle slots, and a left-side
queue of waiting request chips. Under each panel, live: **utilization %**, **throughput (tok/s)**,
**completed reqs**, **avg wait**. A shared clock.

**Controls:** arrival rate (**primary slider**), `G`, speed (`+`/`-`), `Space` play/pause, a
**Step** button (advance one tick for close narration).

**"Break it live":** just run it. Static settles ~40–55% util; continuous ~90–100%; continuous
throughput counter reaches ~2–3× static within ~20s. `B` injects one very long request into the
static batch to dramatize head-of-line blocking (its whole batch stalls behind the long one).

**Acceptance:** identical arrival stream in both panels; continuous strictly dominates on util and
throughput; the gap widens with output-length variance.

---

## 4. ACT II — "The fleet"

### View 4 — `Fleet`: round-robin vs cache-aware routing  (the climax — slides 16–17)

**Purpose:** Turn the payoff slide into a live load test. Flip the router mid-load and watch the
whole dashboard flip with it.

**Model:**
- `N` replica pods (default 6). Each pod holds: a set of **cached prefixes** (id → size, with LRU
  eviction), a **VRAM budget** (weights + active KV + cached prefixes; can fill and evict/OOM),
  **active load** (sequences currently decoding, up to a per-pod slot cap), and a memory bar.
- **Request stream** (seeded): each request picks a `prefixId` from a small set — **4 shared "system
  prompts" + a tail of unique ones — with a skewed (Zipf-ish) distribution** so a few prefixes
  dominate (realistic: everyone shares the big system prompt). Plus `promptLen`, `outputLen`, arrival.
- **Routers (primary toggle `T`):**
  - `round-robin`: next pod in rotation. Ignores cache and load.
  - `least-loaded` (optional 3rd): fewest active seqs. Better, still cache-blind — nice to show
    "even ordinary load balancing isn't enough".
  - `cache-aware` (llm-d style): score each pod = `+big if prefix already cached` `+ freeMemory`
    `- activeLoad`; route to max score.
- **Per-request cost:**
  - **Cache hit** (routed pod already has `prefixId`): skip prefill ⇒ low **TTFT** (`≈ base`), flash green.
  - **Cache miss:** full prefill ⇒ `TTFT ≈ base + promptLen * prefillPerTok (+ queueWait if pod busy)`;
    the pod now caches that prefix (may evict LRU; if it can't fit ⇒ OOM/drop, flash red).
  - Decode consumes a slot for `outputLen` ticks.

**Stage:**
- A grid of **pod cards**: pod name, memory bar (weights + KV + cached prefixes, red when near full),
  active load, a tiny list of cached prefix ids, and a hit/miss flash on each incoming request.
- **Global metric wall:** big **cache-hit %**, **p50 / p95 TTFT**, **throughput**, **drops/OOM**,
  and a rolling **TTFT sparkline** (SVG) that visibly steps when the router is toggled.

**Controls:** QPS (**primary slider**), `N` pods, per-pod slot cap, speed, `Space`, router toggle `T`.

**"Break it live":** run under load on `round-robin` → hit % low (~15–25%), p95 TTFT high, memory
uneven (some pods thrash/evict, some idle), occasional drops. Press `T` → `cache-aware` **without
resetting the stream** → hit % climbs (~70–85%), p95 TTFT drops sharply, memory balances, throughput
up; the sparkline steps down at the toggle. That step-change is the gasp.

**Acceptance:** same seeded stream across routers; toggling live (no reset) produces the divergence
above; numbers are stable/repeatable for a given seed so it can be rehearsed.

---

## 5. Build order (do it in milestones, verify each)

1. **Shell + design system + nav + keyboard + seeded PRNG + fixed-timestep loop + Reset plumbing.**
2. **View 1 `Fit`** (pure math, no animation loop) — easiest, validates formulas + sanity checks.
3. **View 2 `Speed`** — reuse formulas; build the SVG roofline; nail the compute-sweep beat.
4. **View 3 `Batching`** — the scheduler + slot animation; verify same-stream fairness.
5. **View 4 `Fleet`** — pods, routers, metric wall, sparkline; verify live toggle divergence.
6. **Polish pass:** big-type readouts, `?` help overlay, footer disclaimer, reduced-motion guard,
   make sure every view Resets cleanly and nothing overflows at 1920×1080.

---

## 6. Definition of done

- Opens offline by double-click; no console errors; no network requests.
- All four views reachable by keys `1`–`4`; every view Resets with `R`; `Space`/`+`/`-` control sims.
- Sanity-check numbers in §2 reproduce exactly.
- Each view's "break it live" beat works from the keyboard alone (`B` / `T`).
- Same seed ⇒ identical runs (rehearsable). Legible from across a room at 1080p.

## 7. Stretch goals (only after DoD)

- Fleet **disaggregation** toggle: split pods into a prefill pool and a decode pool with KV handoff
  between them (illustrates slide 17's disaggregated serving).
- A `prefix-cache` mini-view: same shared system prompt reused across requests, TTFT with/without.
- A "presenter caption" bar that shows the current teaching line per view (toggle with `C`).
- URL hash encodes seed + view + control state, so a bookmark restores an exact demo state.

---

## 8. Kickoff prompt for Claude Code (paste this first, then paste this whole spec)

> Build a single self-contained `index.html` (inline `<style>`+`<script>`, **vanilla JS, no
> dependencies, no CDN, no network, no build step**) that runs offline by double-click and is meant
> to be projected at 1920×1080 during a live talk. It's an interactive LLM-inference simulator with
> four keyboard-switchable views (`1`–`4`). Follow the attached build spec exactly — especially the
> formulas in §2 and the sanity checks (verify them as you go). Prioritise: deterministic seeded
> runs, a one-key Reset per view, and the specific "break it live" interactions. Implement in the
> milestone order in §5 and confirm each milestone before moving on. Use the exact colour palette
> (colour is semantic: orange=compute, cyan=memory, green=throughput, red=OOM/miss, purple=routing).
