# How a Network Learns

A presenter-first Streamlit app for a ~20-minute talk on how neural networks
learn. Five pages take an audience from "keep score with a loss" to a real
(tiny) language model, showing **actual PyTorch autograd** on screen the whole
way. Every training run is computed once, cached, and scrubbed with a slider —
so nothing trains live on stage.

Built to the spec in [`streamlit-app-spec.md`](../streamlit-app-spec.md).

## The five pages

| Page | Idea | What you do live |
|---|---|---|
| **1 · Keep score** | loss, wiggle test, autograd | Turn `w`/`b`, run the wiggle test, then let PyTorch print the same slopes. Press "nudge" to watch the loss fall. |
| **2 · Downhill** | gradient descent + loss landscape | Two knobs discover `F = 1.8·C + 32` from noisy data. Try step sizes Tiny / Good / Too big and watch one explode. |
| **3 · Backprop** | the chain rule as gears | Step backward through the computation graph link by link; each link multiplies its own local rate. |
| **4 · Bends** | ReLU hinges fit any curve | Fit Wave / Hockey stick / Two bumps with `H ∈ {1,2,4,8,16,32}` neurons. See that too small a model can't fit. |
| **5 · Next letter** | a real bigram language model | One 27×27 matrix predicts the next letter; the loss is *surprise*. Type any name and see how surprised the model is. |

## Requirements

- **Python 3.11+** (this project was built and tested on 3.12)
- The packages in [`requirements.txt`](requirements.txt): `streamlit`, `torch`
  (CPU is fine), `plotly`, `numpy`, `pandas`, `pytest`. No GPU, no network.

## Run it

From this `nn-learns/` folder:

```bash
# 1. Create a virtual environment (once)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (once)
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Streamlit opens `http://localhost:8501` in your browser. On first load it spends
a second or two "warming up" — pre-computing every default training run so the
talk never waits on compute.

Use the **sidebar** (or the arrows) to move between the five pages. Each chart
has a **▶ Play** button to animate training, and a **Training step** slider to
scrub to any moment. Every page has a **Reset** button.

### Presenter notes

Add `?presenter=1` to the URL, e.g. `http://localhost:8501/?presenter=1`, to show
a "Presenter notes" expander with talking points at the bottom of each page. It
is hidden by default so the projected view stays clean.

### Running fully offline (for the stage)

```bash
streamlit run app.py --server.headless true
```

Everything runs locally with no network or API calls. Install the dependencies
into the venv the day before and verify with Wi-Fi off. Then open the browser,
press F11 for full screen, and visit every page once to warm the cache.

## Tests

The acceptance numbers from the spec (loss values, gradients, convergence
behaviour) are encoded as tests, alongside a smoke test that runs every page:

```bash
pytest -q
```

- `tests/test_numbers.py` — the 12 acceptance numbers (e.g. default loss `1600`,
  autograd `w.grad = -800`, Wave H=1 underfits, H=8 fits, bigram reaches its
  exact optimum).
- `tests/test_smoke.py` — loads each of the five pages and asserts no runtime
  errors.

## Project layout

```
nn-learns/
  app.py                 # navigation, theme, presenter toggle, warm-up
  cached.py              # @st.cache_data wrappers around core/ (compute once)
  ui.py                  # presenter notes, reset button, chart helper
  .streamlit/config.toml # dark theme + minimal toolbar
  core/
    data.py              # seeded datasets (temperatures, curves, names)
    neuron.py            # 1-neuron model, wiggle test, page-2 gradient descent
    graph.py             # computation-graph builder for the backprop page
    mlp.py               # 1-H-1 ReLU MLP trained with Adam, as snapshots
    bigram.py            # 27x27 bigram model, sampling, surprise
    style.py             # palette, Plotly template, CSS
  pages/                 # 1_scoreboard … 5_next_letter
  tests/                 # test_numbers.py, test_smoke.py
```

### How it stays smooth on stage

The one rule: **compute in Python, animate in the browser.** Pure functions in
`core/` produce full training trajectories as snapshots; `cached.py` wraps them
in `@st.cache_data` so each run is computed exactly once. Pages never train
inside a widget loop — moving a slider just reads the cache and redraws a Plotly
figure. Everything is seeded and deterministic, so every run matches rehearsal.
```
