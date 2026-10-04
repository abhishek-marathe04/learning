"""Streamlit-cached wrappers around the pure functions in core/.

Kept separate from core/ so the core modules stay import-clean for pytest.
Every training run is computed once and reused from cache; pages and the
start-up warm-up both go through here.
"""

import streamlit as st

from core import bigram, mlp, neuron
from core.data import curve_data, temperature_data


@st.cache_data(show_spinner=False)
def cached_temp_data():
    return temperature_data()


@st.cache_data(show_spinner=False)
def cached_gd(lr_name: str, w0: float = neuron.W_START, b0: float = neuron.B_START):
    x, y = temperature_data()
    return neuron.gd_trajectory(neuron.LR_PRESETS[lr_name], x, y, w0=w0, b0=b0)


@st.cache_data(show_spinner=False)
def cached_loss_surface():
    x, y = temperature_data()
    return neuron.loss_surface(x, y)


@st.cache_data(show_spinner=False)
def cached_curve(name: str):
    return curve_data(name)


@st.cache_data(show_spinner=False)
def cached_mlp(dataset: str, H: int):
    return mlp.train_mlp(dataset, H)


@st.cache_data(show_spinner=False)
def cached_bigram():
    return bigram.train_bigram()


def warm_up():
    """Precompute the default runs for every page so nothing trains on stage."""
    cached_temp_data()
    cached_loss_surface()
    for lr_name in neuron.LR_PRESETS:
        cached_gd(lr_name)
    cached_mlp("Wave", 4)
    cached_mlp("Wave", 8)
    cached_bigram()
