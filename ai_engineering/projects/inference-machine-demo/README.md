# The Inference Machine — Demo README

Single-file offline simulator for the ~20-min Devocracy talk.
Full presenter script: [`inference-machine-talk-script.md`](inference-machine-talk-script.md)

---

## Quick start

```
open index.html        # macOS — double-click also works
```

No build step, no server, no network. Works on a locked-down corporate laptop.
Target resolution: **1920×1080, browser full-screened** (`F11` / `⌘ Shift F`).

---

## Keyboard map (memorise these)

| Key | Action |
|-----|--------|
| `1` `2` `3` `4` | Jump to Fit / Speed / Batching / Fleet |
| `R` | Reset current view to known-good defaults |
| `B` | "Break it live" — the dramatic action per view |
| `T` | Toggle: quantization (Fit) · batch helper (Speed) · router (Fleet) |
| `→` / `←` | Nudge primary slider up / down |
| `Space` | Play / Pause simulation (Batching, Fleet) |
| `+` / `-` | Sim speed up / down |
| `?` | Show / hide keyboard cheat sheet |

---

## Per-demo cheat sheet

### View 1 — Fit  `⌨ 1`  _(talk script §3, ~3:30)_

Default loads **70B FP16 on H100** — already OOM (weights alone = 140 GB > 80 GB).

| Moment | Key | What the audience sees |
|--------|-----|------------------------|
| "watch the orange collapse" | `T` | Quantization cycles FP16 → INT8 → INT4; bar shrinks to fit |
| "watch the cyan" | `B` | Concurrency ramps up, KV cache grows and crosses the OOM line |
| Before leaving | `R` | Clean state |

**Teaching beat:** weights OOM → `T` (INT4 fits) → `B` (KV cache is the sneaky killer).

---

### View 2 — Speed  `⌨ 2`  _(talk script §5, ~7:30) — never cut this one_

Default: **70B on H100, b=1, L=2048** → ~24 tok/s, ~0.3% compute util.

| Moment | Key | What the audience sees |
|--------|-----|------------------------|
| "let's crank compute" | `B` | Compute slider sweeps min→max; tok/s doesn't move |
| "watch what happens to bandwidth" | `→` (a few times) | Bandwidth rises; tok/s climbs proportionally |
| Optional: show batching escape | `T` | Batch size ramps; AI dot walks toward the ridge |
| Before leaving | `R` | Clean state |

**Teaching beat:** compute sweep → silence → bandwidth nudge → number climbs. Let the silence sit.

**Sanity numbers** (these are exact, verify before stage):
- Tok/s ≈ 24, Compute util ≈ 0.3%, Arithmetic Intensity ≈ 1.0, Ridge ≈ 296

---

### View 3 — Batching  `⌨ 3`  _(talk script §7, ~11:00)_

Default: 8 slots, 2 req/s arrival rate, both panels receive identical request stream.

| Moment | Key | What the audience sees |
|--------|-----|------------------------|
| Start both panels | `Space` | Both schedulers run; static util ~40–55%, continuous ~90%+ |
| Optional: dramatise head-of-line | `B` | Long red request injected into static queue; whole batch stalls |
| Before leaving | `R` | Clean state |

**Teaching beat:** let it run ~20s and let the throughput counters diverge on their own.
Cut the `B` injection if running long (it's noted in the script as optional).

---

### View 4 — Fleet  `⌨ 4`  _(talk script §9, ~15:00) — never cut this one_

Default: 6 pods, 4 QPS, **round-robin** router. Most requests share one of 4 system prompts.

| Moment | Key | What the audience sees |
|--------|-----|------------------------|
| Start under load | `Space` | Runs; hit % low (~15–25%), p95 TTFT high, pods uneven |
| **The gasp** — flip router | `T` | Switches to least-loaded, then cache-aware on next `T`; hit % climbs, p95 drops, sparkline steps down |
| If time: show least-loaded isn't enough | `T` once (stays on least-loaded) | Better, but cache-blind |
| Before leaving | `R` | Clean state |

**Teaching beat:** do NOT reset between round-robin and cache-aware — the live divergence is the point.
If you have extra time, pause on least-loaded to show "even good load balancing isn't enough."

---

## Pre-flight checklist

Run this before you're mic'd:

- [ ] `index.html` open in browser, **full-screened**
- [ ] On **View 1 (Fit)**, freshly Reset (`R`)
- [ ] Seed left at `42` (default) — same seed = identical run every time
- [ ] Alt-tab or second display rehearsed so slide→demo→slide is one motion
- [ ] Run through each `B` beat once so the pacing feels comfortable

**Recovery:** if anything misbehaves on stage, press `R` and keep talking. Never debug live.

---

## Seed and determinism

The Seed box (top right) controls all randomness. Default `42` gives the exact run described in the script. Change it to rehearse different traffic patterns. Same seed always produces identical arrivals, output lengths, and prefix draws — so the demo is fully rehearsable.

---

## Approximate timing

| View | Ends around |
|------|-------------|
| 1 · Fit | 5:30 |
| 2 · Speed | 10:00 |
| 3 · Batching | 13:00 |
| 4 · Fleet | 18:00 |
| Q&A | 20:00+ |

**Running long?** Cut (in order): `B` injection in Batching → whole Batching demo → sharding slide.
**Never cut** Demo 2 (Speed) or Demo 4 (Fleet).
