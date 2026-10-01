# Spec: "How a Network Learns" — Streamlit teaching app

A live, presenter-first Streamlit app for a 20-minute Devocracy talk on how neural networks learn. Audience: experienced software engineers who use AI assistants daily. The app replaces the three browser demos in the deck and adds two "real PyTorch" moments the HTML demo can't do.

This document is the brief for Claude Code. Build exactly this; where something is unspecified, prefer the simpler option.

---

## 1. Why Streamlit, and the one rule that makes it work

**What Streamlit gives us over the HTML demo**

- **Real autograd on screen.** The audience sees actual PyTorch compute `w.grad = -800.0`, the same number they just derived by hand on the slides.
- **Scrubbable training.** Every training run is computed once and stored as snapshots. The presenter drags a "training step" slider backwards and forwards: pause at step 3, rewind to 0, jump to the end.
- **Readable code.** Each page can show the exact Python that produced what's on screen. Engineers trust what they can read.

**The rule: compute in Python, animate in the browser.**

- Never animate by looping `st.rerun()` or `time.sleep()` in the script. That is choppy and fragile on stage.
- For every training run: compute the full trajectory once in PyTorch → cache it with `@st.cache_data` keyed on the settings → render it with a Plotly figure that has **animation frames and a play button**, plus a Streamlit slider to scrub to any snapshot.
- Everything is **seeded and deterministic**. Every run on stage matches rehearsal exactly.

---

## 2. Scope

**In scope:** 5 pages (below), shared dark theme, presenter notes toggle, cached precomputation, pytest checks for the key numbers.

**Non-goals:** GPUs, real LLMs, network or API calls, databases, user accounts, deployment. It runs locally with `streamlit run app.py` on a laptop, fully offline.

---

## 3. Tech stack and project layout

- Python 3.11+
- Recent `streamlit` (1.40 or newer, for `st.navigation`, `st.fragment`, `st.pills`/`st.segmented_control`)
- CPU-only `torch`, `plotly`, `numpy`
- `pytest` (dev only)
- Graphs: `st.graphviz_chart` with a DOT string. It renders in the browser, so the system `graphviz` binary is not required.

```
nn-learns/
  app.py                  # st.navigation, page config, theme CSS, presenter toggle
  .streamlit/config.toml  # dark theme + minimal toolbar
  requirements.txt
  core/
    data.py               # datasets (temperature, curves, names) — seeded
    neuron.py             # 1-neuron model, wiggle test, GD trajectory
    graph.py              # computation-graph builder for the backprop page
    mlp.py                # 1-H-1 ReLU MLP + Adam trajectory with snapshots
    bigram.py             # 27x27 bigram NN, snapshots, sampling, surprise
    style.py              # palette constants, Plotly layout template, CSS
  pages/
    1_scoreboard.py
    2_downhill.py
    3_backprop.py
    4_bends.py
    5_next_letter.py
  tests/
    test_numbers.py       # the acceptance numbers in section 7
```

`.streamlit/config.toml`:

```toml
[theme]
base = "dark"
backgroundColor = "#151A33"
secondaryBackgroundColor = "#1D2344"
textColor = "#ECEEF8"
primaryColor = "#3CC6B0"

[client]
toolbarMode = "minimal"
showSidebarNavigation = true
```

---

## 4. Visual language (must match the slide deck)

| Token | Hex | Meaning — use consistently everywhere |
|---|---|---|
| Navy background | `#151A33` | page background |
| Panel | `#1D2344` | cards, chart backgrounds |
| Ink | `#ECEEF8` | text, data points |
| Muted | `#9AA2C8` | labels, captions |
| **Teal** | `#3CC6B0` | **knobs**: weights, the model's line, nudges, gradient arrows |
| **Amber** | `#F2A541` | **loss**: errors, misses, loss curves, surprise |
| Red | `#EF6F6C` | failure states only (exploding loss) |

Presenter-first rules:

- `st.set_page_config(layout="wide", initial_sidebar_state="collapsed")`.
- Big type: body 18px+, key numbers 40–64px.
- One idea per page.
- At most ~4 controls visible on a page.
- Must be readable at 1280×720 on a projector.
- Key numbers are shown as `st.metric` or large custom HTML stat blocks, amber for loss and teal for knobs.
- Plotly: transparent background, no grid clutter, no mode bar (`config={"displayModeBar": False}`), fonts ≥ 14px.
- Keep Streamlit's default widgets. Style them with the theme, not with heavy custom CSS.

**Presenter notes:** if the URL has `?presenter=1`, each page shows an expander at the bottom with talking points (text given per page below). It is hidden by default so the projector view stays clean.

**Reset:** every page has a "Reset" button that clears that page's `st.session_state` keys.

---

## 5. Pages

The talk flow is: slides → page 1 → slides → page 2 → slides → page 3 → slides → page 4 → slides → page 5 → closing slides.

### Page 1 — "Keep score" (after slide 7; replaces slides 8–9 live)

**Purpose:** do the running example live. The audience turns the knobs, sees the loss, runs the wiggle test, then sees PyTorch produce the same slopes.

**Layout:**

```
┌──────────────────────────────┬───────────────────────────────┐
│  w  [ slider 0..3 ]          │   input 10°C   truth 50°F     │
│  b  [ slider -40..80 ]       │   guess  10.00                │
│                              │   miss  -40.00                │
│  (chain diagram, graphviz)   │   LOSS  1,600.00  (big amber) │
│  °C ─×w─▶ ─+b─▶ guess        │                               │
├──────────────────────────────┴───────────────────────────────┤
│ [ Wiggle test ]   [ Ask PyTorch ]   [ Take one nudge ]  [Reset]│
│  results table / code panel appear below as each is pressed    │
└──────────────────────────────────────────────────────────────┘
```

**Behaviour:**

- Defaults: `w = 1.0`, `b = 0.0`, one training example (x = 10, y = 50).
- Live readout: guess = w·x + b, miss = guess − y, loss = miss².
- **Wiggle test** button: shows a table. For each knob, nudge it by +0.01 and show the new guess, new loss, change, and slope = change ÷ 0.01. At defaults this must show w: 1,592.01, −7.99, ≈ −799 and b: 1,599.20, −0.80, ≈ −80.
- **Ask PyTorch** button: runs this and shows both the code (`st.code`) and its real output:
  ```python
  w = torch.tensor(1.0, requires_grad=True)
  b = torch.tensor(0.0, requires_grad=True)
  loss = (w * 10 + b - 50) ** 2
  loss.backward()
  print(w.grad, b.grad)   # tensor(-800.) tensor(-80.)
  ```
  The code must use the current slider values, not hard-coded ones. Show a teal callout: "Wiggle test said ≈ −799 and ≈ −80. Autograd says exactly −800 and −80."
- **Take one nudge** button: step size fixed at 0.0005 (shown on screen). Apply `new = old − 0.0005 × grad`, update the sliders via session state, and show "loss 1,600.00 → 1,293.12".
- Pressing it repeatedly keeps nudging. Show a small loss history sparkline in amber.

**Presenter notes:** "Wiggle is honest but costs one model run per knob. Autograd gets every slope in one backward pass. Keep pressing nudge: watch the loss fall each time."

### Page 2 — "Downhill" (slide 11: Demo 1)

**Purpose:** two knobs discover F = 1.8·C + 32 from noisy data. Show the loss landscape, the learning rate, and an explosion.

**Data:** 24 points, x evenly spaced from −10 to 40 °C with jitter U(−0.75, 0.75), y = 1.8x + 32 + N(0, 1.6), numpy seed 7.

**Model and training:**

- Plain gradient descent on mean squared error, full batch, start at w = 0.5, b = 0.
- Step-size choice via `st.segmented_control`: **Tiny 0.00015 · Good 0.0012 · Too big 0.0025**.
- Compute up to 8,000 steps.
- Stop early when the gradient norm is below 0.01, or when the loss exceeds 1e6 (mark the run "exploded").
- Store snapshots with dense early sampling: every step for the first 30, then about 150 snapshots spread over the rest.

**Layout:** two equal columns.

- **Left, "Data and guess":** scatter of readings (ink), the model line (teal), and dashed vertical miss lines from each point to the line (amber). Title shows `guess = 1.81 × °C + 32.3` for the current snapshot.
- **Right, "Loss landscape":**
  - Plotly contour or heatmap of log(loss) over w ∈ [0, 3.2] and b ∈ [−40, 80] (dark = low).
  - The path so far as a white line, with the current point as a large white dot.
  - A teal arrow showing the direction of the next step.
- Below: a **training step** slider (scrub through snapshots) and a Plotly play button that animates through the same frames. The metrics row shows w, b, loss and step count.
- Status line at the current snapshot:
  - Converged: "Settled after N nudges: w ≈ 1.81, b ≈ 32.3. Nobody told it the formula." (teal)
  - Exploded: "Loss exploded: each step overshoots the valley and lands higher up." (red)

**Extra:** a "Start from here" option: two number inputs, or clicking the landscape if simple (Plotly click events are optional; number inputs are fine). This shows every starting point finds the same valley.

**Presenter notes:** "Ball drops to the valley floor fast, then slides along it. Now switch to Too big."

### Page 3 — "Backprop, one link at a time" (slides 12–13)

**Purpose:** make the chain rule visible as gears. Step backwards through the computation graph and multiply local rates. Then show it's exactly what autograd does.

**Graph** (graphviz DOT, left to right):

```
w (1.0) ──×x(10)──▶ guess (10) ──−y──▶ miss (−40) ──square──▶ loss (1600)
b (0.0) ──────────────────▲
```

**Behaviour:**

- Each node shows its **forward value**.
- Controls: **Backward one link**, **Backward all**, **Reset**.
- Stepping backward from the loss fills in each node's **gradient** (amber badge) and labels the edge being processed with its local rate: square → 2 × miss = −80; −y → 1; ×x → 10; +b → 1.
- Final state shows w.grad = −80 × 1 × 10 = −800 and b.grad = −80 × 1 × 1 = −80. A side panel shows the running multiplication as text, e.g. `−80 × 1 × 10 = −800`.
- Toggle **"Compare with PyTorch"**: shows the same graph's `torch` result, plus the grad from the page-1 wiggle test.
- **Stretch goal:** a dropdown with a second, slightly bigger example, e.g. `loss = (tanh(w1·x + b1)·w2 − y)²`. This shows the same backward walk on a two-layer chain. Only build it if the first graph is solid.

**Presenter notes:** "Each link only knows its own local rate. Multiply backwards. That is all backprop is. Karpathy's micrograd is ~100 lines of this."

### Page 4 — "Bends" (slides 15–16: Demo 2)

**Purpose:** one neuron can only draw a line. ReLU hinges, summed, trace any shape. Show "too small a model can't fit."

**Data:** 60 points, x evenly spaced on [−3, 3], noise N(0, 0.04). Datasets, via `st.segmented_control`:

- **Wave:** `sin(1.7x)·0.9 + 0.15x`
- **Hockey stick:** `0.2 + 0.35·max(0, x − 0.8)²`
- **Two bumps:** `1.1·exp(−3(x+1.3)²) − 0.8·exp(−4(x−1.2)²)`

**Model:** y = Σⱼ vⱼ·ReLU(wⱼ·x + bⱼ) + c, with hidden size H ∈ {1, 2, 4, 8, 16, 32} via `st.segmented_control`.

- Init with torch seed `300 + H`: wⱼ ~ N(0, 1), bⱼ ~ U(−2, 2), vⱼ ~ N(0, 0.5/√H), c = 0.
- Adam, lr 0.02, full batch, 3,000 steps.
- Snapshot every 25 steps.

**Layout:**

- Main chart:
  - data points (ink)
  - network output (thick amber)
  - each hidden neuron's contribution vⱼ·ReLU(wⱼx + bⱼ) as a thin line in its own colour (toggle "Show pieces")
  - a dot on the curve at each neuron's bend point x = −bⱼ/wⱼ, in that neuron's colour
- Side: loss curve (log y, amber), metrics for loss, step, and knob count (3H + 1).
- Training-step scrub slider plus a play button.

**Must demonstrate:** Wave with H = 1 or 2 plateaus around loss ≈ 0.38. Wave with H = 8 gets below 0.02.

**Presenter notes:** "One hinge can't make a wave; no amount of training fixes too-small a model. At 8, nobody assigned jobs; each neuron found its bend because the gradient said so. Uses Adam, a smarter step rule; same gradients."

### Page 5 — "Next letter" (slides 17–18: Demo 3)

**Purpose:** a real (tiny) language model. Same loop; the loss is surprise.

**Data:** the names list in Appendix A. Characters are `.` plus a–z (27 total); `.` marks the start and end of a name. Training pairs are every (current, next) character in `.name.`, about 1,160 pairs.

**Model:**

- W is a 27×27 matrix of zeros (729 knobs).
- logits = one_hot(current) @ W; loss = `F.cross_entropy` (mean).
- Full-batch gradient descent, lr 30, 400 steps.
- Snapshots at steps 0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300 and 400.

**Layout:**

- **Left: heatmap** of softmax(W) rows (row = current letter, column = next letter), colorscale navy → teal → amber, axis labels are the letters.
- **Right top: "Names it invents right now."** 10 samples drawn at the current snapshot with a fixed sampling seed (42), so changes reflect learning, not luck. Cap at 12 characters and add "…".
- **Right bottom: "How surprised is it by this name?"** A text input (default `aarav`) and a bar per letter: height = surprise = −ln p, amber, with p% under each letter. Show "average surprise X (knew nothing = 3.30)".
- **Bottom:** loss curve with two dashed reference lines:
  - "knows nothing: 3.30" = ln 27
  - "best possible with one-letter memory", computed exactly from the pair counts (≈ 2.05)
- Training-step scrub slider plus a play button (a Plotly animation for the heatmap and loss; the samples update with the slider).

**Presenter notes:** "Letters never seen (f, q, x) stay flat and unsure. Type an audience member's name. An LLM is this, but it sees thousands of tokens of context through a transformer, with ~10¹¹ knobs. Same loop."

### Optional page 6 — "Behind the curtain"

A single page with `st.code` showing the core training loop from each of `neuron.py`, `mlp.py` and `bigram.py`, side by side with the 5-line loop from slide 14. Useful for Q&A; skip if short on time.

---

## 6. State, caching and performance

- All trajectories come from pure functions in `core/`, wrapped with `@st.cache_data`. Arguments are plain values (lr, H, dataset name, seed); return values are numpy arrays or dicts.
- Pages never train inside a widget callback loop. Changing a control reads from the cache, computing on first use.
- **Warm-up:** `app.py` calls every cached function for the default settings at startup, inside `st.spinner`, so nothing computes live on stage. Target: under 10 s total on a normal laptop CPU.
- Use `st.session_state` for: page-1 knob values and history, page-3 backward-step index, and each page's current snapshot index.
- Plotly animations: build frames from the snapshots and use `transition={"duration":0}` with a frame duration of about 60 ms, so playback is smooth client-side.

---

## 7. Acceptance numbers (turn these into `tests/test_numbers.py`)

| Check | Expected |
|---|---|
| Page 1 defaults: loss | 1600.0 |
| Page 1 wiggle slopes | w ≈ −799 (±1), b ≈ −80 (±0.1) |
| Page 1 autograd | w.grad = −800.0, b.grad = −80.0 exactly |
| Page 1 one nudge (lr 0.0005) | w = 1.40, b = 0.04, loss ≈ 1293.12 |
| Page 3 chain | −80 × 1 × 10 = −800; −80 × 1 × 1 = −80 |
| Page 2, Good lr | converges; final w, b within 0.05 / 0.5 of `np.polyfit(x, y, 1)` |
| Page 2, Too big | flagged "exploded" within 200 steps |
| Page 2, Tiny | not converged at 8,000 steps; loss still decreasing |
| Page 4, Wave H=1 | final loss > 0.3 |
| Page 4, Wave H=8 | final loss < 0.02 |
| Page 5, step 0 | loss = ln 27 ≈ 3.296 |
| Page 5, step 400 | loss within 0.05 of the exact bigram optimum (≈ 2.05) |

---

## 8. Stage safety

- Runs fully offline. `requirements.txt` is installed in a venv the day before; verify with the Wi-Fi off.
- Launch with `streamlit run app.py --server.headless true`. Open the browser, press F11, and visit every page once to warm up.
- Test at 1280×720 and 1920×1080 browser sizes; nothing important below the fold on any page.
- No randomness anywhere without an explicit seed.
- If anything misbehaves: press Reset on that page and keep talking. **Backup:** `index.html` (the existing single-file demo) is open in a second browser tab.

---

## 9. Build order for Claude Code

1. `core/` functions and `tests/test_numbers.py`. Get all acceptance numbers passing before any UI.
2. `app.py` shell, theme, navigation, presenter toggle, warm-up.
3. Page 1, then 2, then 4, then 5. Check each at 1280×720 before moving on.
4. Page 3 (the graph).
5. Stretch items: page 3's second example, page 6.

Keep each page file under ~200 lines. Put plotting helpers in `core/style.py`.

---

## Appendix A — names corpus

```
aarav aditya arjun vihaan reyansh ayaan krishna ishaan shaurya atharv advik pranav advait dhruv kabir ritvik aarush kayaan darsh veer rudra arnav yash rohan rahul amit sachin rajesh suresh mahesh ganesh ramesh dinesh nitin vivek anil sunil vikram varun karan nikhil akash abhishek siddharth harsh manish sanjay vijay ajay deepak gaurav saurabh tushar omkar prasad chinmay tejas aniket sagar swapnil mayur ninad parth sahil soham vedant yogesh mangesh shreyas anand ashok bharat chetan girish hemant jayant kunal lalit mohan naresh pankaj rakesh sameer tarun uday vinay aadhya ananya diya saanvi aarohi anika navya pari myra sara kiara ira riya aanya siya avni prisha anvi kavya isha neha pooja priya sneha shruti swati anjali divya kirti madhuri meera nisha pallavi radha rashmi sakshi sonali tanvi trisha vaishnavi gauri ketaki mrunal rutuja sayali shravani tejaswini aishwarya bhavana chaitra deepa gayatri jyoti kalyani lata manasi namrata padma rekha sunita usha varsha yamini aparna archana asmita harshada janhavi kasturi leena mitali prajakta rasika shivani smita supriya vrushali amruta aditi ishita juhi komal lavanya mansi nandini rupali snehal tanaya
```

## Appendix B — how the app maps to the deck

| Slide | App |
|---|---|
| 1–7 | slides only |
| 8–9 (wiggle, update rule) | present the slides, then **Page 1** to do it live (or use Page 1 instead of the slides) |
| 10–11 (landscape, Demo 1) | **Page 2** |
| 12–13 (backprop) | slide 12, then **Page 3** instead of slide 13 |
| 14 (training loop) | slide (or page 6) |
| 15–16 (bends, Demo 2) | slide 15, then **Page 4** |
| 17–18 (surprise, Demo 3) | slide 17, then **Page 5** |
| 19–22 | slides only |

Adding pages 1 and 3 costs about 2 extra minutes. To stay at 20, drop slides 8, 9 and 13 in favour of the live versions, or cut the page-2 explosion.
