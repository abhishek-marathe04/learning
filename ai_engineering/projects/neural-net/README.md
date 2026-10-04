# How a Network Learns — Neural Network Visualization

A Streamlit teaching app that visualizes how neural networks learn in real-time with interactive controls.

## Overview

An interactive educational tool designed for a 20-minute talk to software engineers on neural network fundamentals. Shows actual PyTorch computation, real-time training visualization, and step-by-step backpropagation.

**Key Features:**
- Real autograd on screen (actual PyTorch gradients)
- Scrubbable training visualization (play/pause/seek through training)
- Interactive demonstrations of neural network learning
- Readable code examples for audience verification
- Fully deterministic and seeded for reproducibility

## Project Structure

```
nn-learns/
  app.py                  # Main Streamlit app with navigation
  .streamlit/config.toml  # Dark theme configuration
  requirements.txt        # Dependencies
  ui.py                   # UI components
  cached.py               # Caching helpers
  core/
    data.py              # Datasets (temperature, curves, names)
    neuron.py            # Single neuron model and GD trajectory
    graph.py             # Computation graph builder
    mlp.py               # 1-H-1 ReLU MLP with Adam training
```

## Setup

### Prerequisites
- Python 3.11+
- Streamlit 1.40+
- PyTorch (CPU)
- Plotly, NumPy

### Install

```bash
cd nn-learns
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Pages

The app contains 5 interactive pages demonstrating different neural network concepts:

1. **Gradient Descent** — Single neuron learning
2. **Backpropagation** — Computation graph visualization
3. **Neural Networks** — Multi-layer network training
4. **Training Dynamics** — Visualization of learning over time
5. **Reflection** — Key takeaways and next steps

Each page includes:
- Interactive sliders to explore different parameter values
- Live visualization of the network's behavior
- Toggle for presenter notes
- Deterministic, seeded computation for on-stage reliability

## Using the App

### Presentation Mode
- Use the sidebar toggle for presenter notes
- All computations are precomputed and cached
- Use sliders to explore different configurations
- Results are fully reproducible with same seed

### The Slider Paradigm
Instead of live recomputation (slow, fragile on stage):
1. Full training trajectory is computed once
2. Cached with settings as the key
3. Streamlit slider scrubs through snapshots
4. Plotly animation with play/pause control

## Design Principles

- **Compute in Python, animate in browser** — no `st.rerun()` loops
- **Real numbers, real code** — audience sees actual PyTorch output
- **Seeded and deterministic** — every run matches rehearsal
- **Readable source** — engineers can verify the math

## Topics Covered

- Gradient descent and learning rates
- Backpropagation and automatic differentiation
- Multi-layer networks and ReLU activations
- Adam optimizer and training dynamics
- Visualization of loss surfaces and weight evolution

---

← [Back to Projects](../README.md)
← [Back to Main Hub](../../README.md)
