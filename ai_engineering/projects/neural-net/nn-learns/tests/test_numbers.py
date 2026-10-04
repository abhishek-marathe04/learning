"""Acceptance numbers from section 7 of the spec. Run: pytest -q"""

import numpy as np
import pytest

from core import bigram, graph, mlp, neuron
from core.data import temperature_data


# --- Page 1 ----------------------------------------------------------------
def test_page1_default_loss():
    _, _, loss = neuron.forward(1.0, 0.0)
    assert loss == pytest.approx(1600.0)


def test_page1_wiggle_slopes():
    rows = {r["knob"]: r for r in neuron.wiggle_test(1.0, 0.0)}
    assert rows["w"]["slope"] == pytest.approx(-799.0, abs=1.0)
    assert rows["b"]["slope"] == pytest.approx(-80.0, abs=0.1)
    # exact intermediate readouts quoted in the spec
    assert rows["w"]["new_loss"] == pytest.approx(1592.01, abs=0.01)
    assert rows["b"]["new_loss"] == pytest.approx(1599.20, abs=0.01)


def test_page1_autograd():
    gw, gb = neuron.autograd_grads(1.0, 0.0)
    assert gw == -800.0
    assert gb == -80.0


def test_page1_one_nudge():
    w, b, loss = neuron.one_nudge(1.0, 0.0, lr=0.0005)
    assert w == pytest.approx(1.40, abs=1e-6)
    assert b == pytest.approx(0.04, abs=1e-6)
    assert loss == pytest.approx(1293.12, abs=0.01)


def test_page1_knob_clamps_to_slider_bounds():
    assert neuron.clamp(3.0808348999023436, 0.0, 3.0) == pytest.approx(3.0)
    assert neuron.clamp(-50.0, -40.0, 80.0) == pytest.approx(-40.0)
    assert neuron.clamp(20.0, -40.0, 80.0) == pytest.approx(20.0)


# --- Page 3 ----------------------------------------------------------------
def test_page3_chain():
    _, steps = graph.build_graph(1.0, 0.0)
    grads = {s["sets_node"]: s["grad"] for s in steps}
    assert grads["w"] == pytest.approx(-800.0)
    assert grads["b"] == pytest.approx(-80.0)
    # matches a real autograd pass
    gw, gb = graph.torch_check(1.0, 0.0)
    assert (gw, gb) == (-800.0, -80.0)


# --- Page 2 ----------------------------------------------------------------
def test_page2_good_converges():
    x, y = temperature_data()
    run = neuron.gd_trajectory(neuron.LR_PRESETS["Good"], x, y)
    assert not run["exploded"]
    m, c = np.polyfit(x, y, 1)
    assert run["final_w"] == pytest.approx(m, abs=0.05)
    assert run["final_b"] == pytest.approx(c, abs=0.5)


def test_page2_too_big_explodes():
    x, y = temperature_data()
    run = neuron.gd_trajectory(neuron.LR_PRESETS["Too big"], x, y)
    assert run["exploded"]
    assert run["explode_step"] is not None and run["explode_step"] <= 200


def test_page2_tiny_not_converged():
    x, y = temperature_data()
    run = neuron.gd_trajectory(neuron.LR_PRESETS["Tiny"], x, y)
    assert not run["converged"]
    assert run["n_steps"] == neuron.MAX_STEPS
    losses = [s["loss"] for s in run["snaps"]]
    assert losses[-1] < losses[-2]  # still decreasing


# --- Page 4 ----------------------------------------------------------------
def test_page4_wave_h1_underfits():
    run = mlp.train_mlp("Wave", 1)
    assert run["final_loss"] > 0.3


def test_page4_wave_h8_fits():
    run = mlp.train_mlp("Wave", 8)
    assert run["final_loss"] < 0.02


# --- Page 5 ----------------------------------------------------------------
def test_page5_step0_uniform():
    run = bigram.train_bigram()
    step0 = next(s for s in run["snaps"] if s["step"] == 0)
    assert step0["loss"] == pytest.approx(np.log(27), abs=1e-3)


def test_page5_step400_near_optimum():
    run = bigram.train_bigram()
    final = next(s for s in run["snaps"] if s["step"] == 400)
    assert final["loss"] == pytest.approx(run["optimum"], abs=0.05)
    assert run["optimum"] == pytest.approx(2.05, abs=0.1)
