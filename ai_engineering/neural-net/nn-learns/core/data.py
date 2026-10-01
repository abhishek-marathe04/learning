"""Seeded datasets for every page. All randomness has an explicit seed."""

import numpy as np

# --- Page 2: temperature readings ------------------------------------------
# True law: F = 1.8 * C + 32, hidden in noisy readings.


def temperature_data():
    """24 (celsius, fahrenheit) readings. Deterministic (numpy seed 7)."""
    rng = np.random.default_rng(7)
    x = np.linspace(-10, 40, 24)
    x = x + rng.uniform(-0.75, 0.75, size=x.shape)
    y = 1.8 * x + 32 + rng.normal(0, 1.6, size=x.shape)
    return x.astype(np.float64), y.astype(np.float64)


# --- Page 4: curves the MLP tries to trace ---------------------------------
CURVES = ["Wave", "Hockey stick", "Two bumps"]


def _curve_fn(name: str):
    if name == "Wave":
        return lambda x: np.sin(1.7 * x) * 0.9 + 0.15 * x
    if name == "Hockey stick":
        return lambda x: 0.2 + 0.35 * np.maximum(0, x - 0.8) ** 2
    if name == "Two bumps":
        return lambda x: 1.1 * np.exp(-3 * (x + 1.3) ** 2) - 0.8 * np.exp(-4 * (x - 1.2) ** 2)
    raise ValueError(f"unknown curve {name!r}")


def curve_data(name: str):
    """60 (x, y) points on [-3, 3] with noise N(0, 0.04). Seeded per curve."""
    seed = {"Wave": 11, "Hockey stick": 12, "Two bumps": 13}[name]
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, 60)
    y = _curve_fn(name)(x) + rng.normal(0, 0.04, size=x.shape)
    return x.astype(np.float64), y.astype(np.float64)


# --- Page 5: names corpus (Appendix A) -------------------------------------
NAMES_RAW = (
    "aarav aditya arjun vihaan reyansh ayaan krishna ishaan shaurya atharv advik "
    "pranav advait dhruv kabir ritvik aarush kayaan darsh veer rudra arnav yash "
    "rohan rahul amit sachin rajesh suresh mahesh ganesh ramesh dinesh nitin vivek "
    "anil sunil vikram varun karan nikhil akash abhishek siddharth harsh manish "
    "sanjay vijay ajay deepak gaurav saurabh tushar omkar prasad chinmay tejas "
    "aniket sagar swapnil mayur ninad parth sahil soham vedant yogesh mangesh "
    "shreyas anand ashok bharat chetan girish hemant jayant kunal lalit mohan "
    "naresh pankaj rakesh sameer tarun uday vinay aadhya ananya diya saanvi aarohi "
    "anika navya pari myra sara kiara ira riya aanya siya avni prisha anvi kavya "
    "isha neha pooja priya sneha shruti swati anjali divya kirti madhuri meera "
    "nisha pallavi radha rashmi sakshi sonali tanvi trisha vaishnavi gauri ketaki "
    "mrunal rutuja sayali shravani tejaswini aishwarya bhavana chaitra deepa "
    "gayatri jyoti kalyani lata manasi namrata padma rekha sunita usha varsha "
    "yamini aparna archana asmita harshada janhavi kasturi leena mitali prajakta "
    "rasika shivani smita supriya vrushali amruta aditi ishita juhi komal lavanya "
    "mansi nandini rupali snehal tanaya"
)

# Vocabulary: '.' (start/end marker) plus a-z  ->  27 tokens.
CHARS = ["."] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for ch, i in STOI.items()}


def names_list():
    return NAMES_RAW.split()


def bigram_pairs():
    """Return (current, next) index arrays over every '.name.' pair."""
    xs, ys = [], []
    for name in names_list():
        chs = ["."] + list(name) + ["."]
        for a, b in zip(chs, chs[1:]):
            xs.append(STOI[a])
            ys.append(STOI[b])
    return np.array(xs), np.array(ys)
