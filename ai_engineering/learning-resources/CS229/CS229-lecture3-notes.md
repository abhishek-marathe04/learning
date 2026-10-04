# CS229 Lecture 3 — Probabilistic Interpretation of Least Squares (Deep Dive)

> **Session Date:** 2026-07-07
> **Duration context:** Deep-dive (extended multi-turn, first-principles derivation with numeric worked examples)
> **Tags:** `#cs229` `#linear-regression` `#probabilistic-interpretation` `#mle` `#gaussian-distribution` `#likelihood-vs-probability` `#least-squares`

## Overview

This session is a slow, first-principles walkthrough of the **probabilistic interpretation of linear regression**, as taught in Stanford CS229 Autumn 2018, Lecture 3 (Locally Weighted & Logistic Regression). The core question being answered: *why is minimizing squared error the "correct" thing to do, rather than just a convenient heuristic?* The session builds up, piece by piece: the error term ε, why it's modeled as Gaussian with mean 0, what "density" means for continuous random variables (with a fully worked numeric example), how density becomes **likelihood** once you fix the data and let θ vary, and finally the conceptual distinction between **probability** and **likelihood** — the same formula, asked as two different questions. The session deliberately stayed slow and concrete, revisiting and re-deriving numbers whenever the reasoning wasn't fully clear, per the user's explicit request to "go slow."

## Core Concepts

### 1. Where does ε (epsilon) come from?

Linear regression assumes:

$$y^{(i)} = \theta^T x^{(i)} + \varepsilon^{(i)}$$

θᵀx⁽ⁱ⁾ is what the linear model *can* capture — the part of y that's a linear function of the features you have. ε⁽ⁱ⁾ is defined as whatever's left over:

$$\varepsilon^{(i)} = y^{(i)} - \theta^T x^{(i)}$$

ε⁽ⁱ⁾ captures two things:
- **Unmodeled effects** — real factors influencing price that aren't in x (e.g., whether the seller was in a hurry, school quality, buyer's mood)
- **Random noise** — measurement error, idiosyncratic randomness

ε is not a mystical add-on — it's just "the residual," defined by construction. The interesting modeling choice is what *distribution* to assume for it.

### 2. Why is ε assumed Gaussian?

This follows from a **Central Limit Theorem (CLT)** argument. ε⁽ⁱ⁾ bundles together many small, independent, unmodeled factors (school quality, seller mood, weather, neighbor's lawn, etc.), none of which dominates. CLT says: **the sum of many small, independent random contributions converges to a Gaussian**, regardless of the individual contributions' own distributions. This is a reasonable **modeling assumption** justified by CLT-style intuition — not a rigorously derived fact about reality.

### 3. Why mean 0?

This is a **free choice**, not a discovery. If the true mean of the errors were some nonzero constant c, you could rewrite:

$$y^{(i)} = \theta^T x^{(i)} + c + \varepsilon'^{(i)}, \quad \text{where } \varepsilon'^{(i)} = \varepsilon^{(i)} - c \text{ has mean } 0$$

That constant c is just an intercept — and if the feature vector x includes a bias term x₀=1 (which it does, by convention), c gets absorbed automatically into θ₀ during fitting. So **assuming E[ε]=0 costs nothing in generality**, as long as the model has an intercept term. This is why it's "WLOG" (without loss of generality).

### 4. Why call it "variance" σ²?

σ² is just notation for "how spread out the noise is." Its actual value is unknown, doesn't need to be known in advance, and — as later steps in the derivation show — turns out **not to matter** for finding the best θ.

$$\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

with the ε's assumed **i.i.d.** across training examples:
- **Independent** — one house's noise doesn't affect another's
- **Identically distributed** — same σ² for every example (homoscedasticity — noise spread doesn't grow/shrink with house size, etc.)

### 5. From ε's distribution to y's distribution

Since ε⁽ⁱ⁾ = y⁽ⁱ⁾ − θᵀx⁽ⁱ⁾ and ε⁽ⁱ⁾ ~ N(0, σ²), a shift-of-a-Gaussian argument gives:

$$y^{(i)} \mid x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$

$$p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

This is the critical object — a genuine probability statement, not just "minimize squared error because it seems reasonable."

### 6. What does "density" actually mean?

For **discrete** random variables (e.g., a die roll), P(X=3) = 1/6 is a real probability of an exact value.

For **continuous** random variables (e.g., house price), P(y = exactly 4,200,000.000...) is **exactly zero** — there are infinitely many possible real-valued outcomes, so no single exact value can carry positive probability (they'd sum to infinity, not 1). So probability isn't the right tool for continuous variables — we need **density** instead.

**Density is not probability. It's probability per unit length** — like physical density (kg/m³): a chunk of iron has density at every point, but an infinitesimal point has zero mass. To get actual probability (a real, positive quantity), you integrate density over an **interval**:

$$P(a \le y \le b) = \int_a^b p(y)\, dy$$

Key properties of a density curve:
- Tallest at the mean (μ) — most "crowded" there
- Tapers off moving away from μ
- **Total area under the whole curve = 1**
- The curve's **height can exceed 1** — density isn't bounded like probability is. A narrow, tall spike (small σ) is completely valid.

### 7. σ vs σ² — standard deviation vs variance (a correction made mid-session)

- **σ (standard deviation)** — the "spread" of the noise, in the *same units* as y (e.g., lakh ₹). This is the intuitive "±50 lakh" type of statement.
- **σ² (variance)** — σ squared, in *squared units* (e.g., lakh²) — not something you interpret directly; it's a mathematical intermediate.

Why the formula uses σ in the front constant but σ² in the exponent:
- The front constant `1/(√(2π)·σ)` sets the **scale/height** of the curve — naturally expressed in the same units as y, hence σ (not σ²).
- The exponent involves `(y−μ)²` (squared units from squaring), so it needs `σ²` in the denominator to cancel units and produce a clean dimensionless ratio.

### 8. What does the "height" of the density curve (e.g., 0.00798) actually mean?

It is simply **the height of the bell curve at a specific x-position**, when plotted with price on the x-axis and p(y) on the y-axis. It is **not** a probability, **not** a percentage — just an "altitude reading" on the graph.

**Layman analogy:** picture a sand mound shaped like the bell curve. Standing at x=500 (the peak), the sand under your feet is 0.00798 units tall. Walk to x=600, the sand is only 0.00108 units tall — much shorter. The number alone means little in isolation; what matters is:
1. **Comparing heights** across positions (500 is ~7.4× taller than 600)
2. **Using it as an ingredient to compute area** (height × width ≈ probability for a narrow interval) — this is what turns it into something real-world interpretable.

**Why is the height so small (<1)?** Because σ=50 is fairly wide — the "mound" is spread over a large range, so to keep total area under the whole curve equal to exactly 1, any single point's height must be small. A wide, flat hill has a low peak; a narrow, spiky hill has a very tall peak (demonstrated with σ=0.1 giving a peak height of ~3.99 — confirming density can exceed 1).

### 9. Likelihood vs. Probability — same formula, two different questions

This is the pivotal conceptual distinction of the session.

$$p(y \mid x; \theta) \quad \text{vs} \quad L(\theta) = p(y \mid x; \theta)$$

**Literally the identical mathematical expression.** The difference is entirely about **what's held fixed and what's allowed to vary**.

**Probability view — θ fixed, y varies:**
Treat θ as known/given. Ask: "across different possible values of y, how likely is each one?"
> Example: "If θ=45 is true, what's the probability my next house (size x=10) sells between 495 and 505 lakh?" — y is the unknown, varying thing (hasn't happened yet); θ=45 is locked in as fact.

**Likelihood view — y fixed, θ varies:**
Treat y as known/fixed (it already happened, it's in the data, unchangeable). Ask: "across different possible values of θ, which one best explains this y I already observed?"
> Example: House 1 (x=10) definitely sold for y=500 — this is history, frozen. Ask: "Given that this house sold for 500, how plausible is θ=45? How about θ=47? Or θ=50?" — y=500 is frozen, θ is what's moving.

**Side-by-side:**

| | What's fixed | What varies | Question being asked |
|---|---|---|---|
| **Probability** | θ = 45 | y (500, 550, 600, ...) | "Given θ=45 is true, how likely are different possible prices?" |
| **Likelihood** | y = 500 (actual, observed) | θ (45, 47, 50, ...) | "Given price=500 actually happened, how plausible is each candidate θ?" |

**Critical practical distinction:** In the probability view, integrating P(y) over *all* possible y gives exactly 1 (a genuine probability distribution). In the likelihood view, integrating/summing L(θ) over *all* possible θ does **not** give 1 or any bounded meaningful number — **L(θ) is not a probability distribution over θ**. It's a **relative scoring function**: bigger L(θ) = better-supported guess, with no "these all add up to 100%" property.

**Classic coin-flip analogy used to reinforce this:**
Let p = P(heads), the unknown parameter.
- **Probability question** (p fixed, outcome varies): "If p=0.5 (fair coin), what's the chance of getting exactly 7 heads in 10 flips?"
- **Likelihood question** (outcome fixed, p varies): You flip a coin 10 times and *observe* 7 heads (fixed, already happened). "Is p=0.5 a good explanation? Or p=0.7? Or p=0.9?" — turns out p=0.7 maximizes this, matching intuition (observed frequency).

**Why this matters in ML:** You almost never know θ — that's the whole point of training. What you *do* have is fixed, already-collected data. So training is always "likelihood mode" — data is locked in, and you search over θ-space to find the value that makes your fixed, already-happened data look as plausible as possible. This is precisely **Maximum Likelihood Estimation (MLE)**.

### 10. Log-likelihood, and why maximizing it collapses into minimizing squared error

**Why take the log at all?** With m training examples, L(θ) is a product of m small numbers (each roughly 0.001–0.008 in our toy example) — multiplying hundreds of these together is numerically painful, and products are messy to differentiate (would require the product rule across all m terms). The fix: log is **monotonically increasing**, so whatever θ maximizes L(θ) also maximizes log(L(θ)) — the location of the peak doesn't move, only the shape of the curve changes. This new function is the **log-likelihood**:

$$\ell(\theta) = \log L(\theta)$$

**Step 1 — product becomes a sum.** Using log(a×b×c) = log a + log b + log c:

$$\ell(\theta) = \log \prod_{i=1}^{m} p(y^{(i)}|x^{(i)};\theta) = \sum_{i=1}^{m} \log p(y^{(i)}|x^{(i)};\theta)$$

Sums are far easier to differentiate term-by-term than products (derivative of a sum = sum of derivatives, no product rule needed).

**Step 2 — expand the log of the Gaussian density.** Substitute the actual density formula, then split using log(A×B) = log A + log B:

$$\ell(\theta) = \sum_{i=1}^{m} \left[ \log\frac{1}{\sqrt{2\pi}\,\sigma} + \log\exp\left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right) \right]$$

Since log(exp(x)) = x (log and exp are inverse operations), the second piece simplifies directly:

$$\ell(\theta) = \sum_{i=1}^{m} \left[ \log\frac{1}{\sqrt{2\pi}\,\sigma} - \frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2} \right]$$

**Step 3 — split into two separate sums.** The first piece, `log(1/(√(2π)σ))`, doesn't involve i at all (σ is fixed) — summing the same constant m times is just m× that constant. The second piece's `1/(2σ²)` is also constant w.r.t. i, so it pulls outside the sum:

$$\ell(\theta) = \underbrace{m \log\frac{1}{\sqrt{2\pi}\,\sigma}}_{\text{Term 1 — constant}} \;-\; \underbrace{\frac{1}{2\sigma^2}\sum_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2}_{\text{Term 2 — depends on }\theta}$$

**Step 4 — Term 1 is irrelevant to finding the best θ.** Term 1 depends only on m and σ (both fixed) — it contains **no θ anywhere**. No matter what θ is plugged in, Term 1 never changes.

*Analogy:* if the entire road you're trying to find the highest point on sits on a platform uniformly raised 100m above sea level, that flat offset doesn't change *where* the highest point on the road is — it just shifts every point up equally. Term 1 is exactly that flat platform: part of the total ℓ(θ) value, but irrelevant to *which* θ maximizes it. So **maximizing ℓ(θ) over θ reduces to maximizing just Term 2.**

**Step 5 — the negative sign flips maximize into minimize.** Term 2 is:

$$-\frac{1}{2\sigma^2}\sum_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2$$

`1/(2σ²)` is a positive constant (σ² is always positive), and `Σ(y⁽ⁱ⁾−θᵀx⁽ⁱ⁾)²` is a sum of squares — always ≥ 0. So Term 2 = −(positive constant)×(something ≥ 0), meaning **Term 2 is always ≤ 0.**

To make ℓ(θ) = Term1 + Term2 as large as possible, and since Term 1 is fixed, Term 2 must be made **as large as possible** — i.e., as close to zero as it can get (its maximum achievable value, since it can never be positive).

*Concrete numeric intuition:* if Σ(errors)² = 100, Term 2 = −(constant)×100, fairly negative. If Σ(errors)² = 10, Term 2 = −(constant)×10, less negative (bigger). If Σ(errors)² = 0 (a hypothetical perfect fit), Term 2 = 0, its ceiling. So **smaller sum-of-squared-errors → larger (better) Term 2 → larger (better) ℓ(θ)**. The negative sign is what flips "maximize the sum of squares" into "minimize the sum of squares."

**Step 6 — arriving at J(θ).** The entire MLE problem collapses to:

$$\arg\max_\theta \, \ell(\theta) \;=\; \arg\min_\theta \, \sum_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2$$

(The positive constant `1/(2σ²)` scaling the sum doesn't change *where* the minimum is located — same reasoning as Term 1 being irrelevant.) Ng defines this minimization target — with a conventional ½ factor included purely so that later differentiation cleanly cancels the 2 that comes down from differentiating a square — as the **cost function**:

$$J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(y^{(i)}-\theta^Tx^{(i)})^2$$

**The punchline:** ordinary least-squares regression — "minimize sum of squared errors," which can look like an arbitrary, "seems reasonable" choice when first encountered — turns out to be **mathematically forced** the moment you assume Gaussian, i.i.d., zero-mean noise. Maximizing likelihood under that assumption *is* minimizing squared error. Least squares isn't a heuristic; it's the MLE solution to an explicit probabilistic model.

**Footnote — σ² never mattered for finding θ.** Throughout this derivation, σ² only ever appeared as a positive scaling constant (in Term 1, and multiplying the sum in Term 2) — it never influenced *which* θ is optimal. This confirms the earlier point: **you don't need to know the true noise level to find the best-fit θ**; whatever σ² actually is, the MLE θ comes out identical. σ² would only matter later for reasoning about *confidence* in the estimate, not for finding θ itself.

## Worked Examples

### Example 1 — Computing density at specific points (single house, hand-derivation)

**Setup:** Neighborhood with model-predicted mean price μ = 500 (lakh ₹), noise spread σ = 50 (lakh ₹).

$$p(y) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)$$

**Step 1 — the constant part** (front of the formula, depends only on σ, not on y):

$$\frac{1}{\sqrt{2\pi} \times 50} = \frac{1}{2.5066 \times 50} = \frac{1}{125.33} \approx 0.00798$$

This value stays **fixed for every calculation** as long as σ=50 doesn't change — it doesn't matter what y is, or later, what θ is being tested. It's purely a scaling constant that ensures the whole curve integrates to 1.

**Step 2 — density at three points:**

| y (lakh ₹) | (y−μ)/σ | exponent = −(y−μ)²/(2σ²) | p(y) |
|---|---|---|---|
| 500 (= μ) | 0 | 0 | 0.00798 × e⁰ = **0.00798** |
| 550 (1σ away) | 1 | −(50)²/(2×2500) = −0.5 | 0.00798 × e⁻⁰·⁵ = 0.00798 × 0.6065 ≈ **0.00484** |
| 600 (2σ away) | 2 | −(100)²/(2×2500) = −2.0 | 0.00798 × e⁻² = 0.00798 × 0.1353 ≈ **0.00108** |

**"1σ away" explained:** this is just (error)/σ — how many "typical spread units" the point is from the center. Error=50, σ=50 → 50/50 = 1, i.e., exactly 1 standard deviation away. Whenever error is exactly 1σ, the exponent always works out to −0.5 regardless of σ's actual value, because (1σ)²/(2σ²) = σ²/(2σ²) = 1/2 — a useful shortcut to recognize.

**Step 3 — turning density into an actual probability** (interval 495 to 505, width=10, centered at μ):

*Approximate way* (density × width, valid for narrow intervals):
$$P \approx p(500) \times 10 = 0.00798 \times 10 = 0.0798 \rightarrow 7.98\%$$

*Exact way* (Z-table / standard normal CDF):
$$z_1 = \frac{495-500}{50} = -0.1, \quad z_2 = \frac{505-500}{50} = 0.1$$
$$P = \Phi(0.1) - \Phi(-0.1) = 2\Phi(0.1) - 1 = 2(0.5398) - 1 = 0.0796 \rightarrow 7.96\%$$

The two methods (7.98% vs 7.96%) are nearly identical — concrete proof that "density × width ≈ probability" for narrow intervals, becoming exact in the limit (which is what the integral ∫p(y)dy formalizes).

**Step 4 — sanity-checking total area = 1** using the empirical (68–95–99.7) rule:
- P(μ−σ ≤ y ≤ μ+σ) ≈ 68%
- P(μ−2σ ≤ y ≤ μ+2σ) ≈ 95%
- P(μ−3σ ≤ y ≤ μ+3σ) ≈ 99.7%

As the interval widens to cover all real numbers, probability approaches 100% = 1, confirming all the density everywhere sums to exactly 1 unit of total probability.

**Step 5 — demonstrating density can exceed 1** (tighter noise, σ=0.1):

$$p(\mu) = \frac{1}{\sqrt{2\pi} \times 0.1} = \frac{1}{0.2507} \approx 3.99$$

A density of ~3.99 (>1) is completely valid — the narrower the curve, the taller its peak must be to keep total area fixed at 1.

### Example 2 — From density to likelihood (2-house toy dataset)

**Setup:** Model without intercept: y = θx. Two houses:

| House | x (size, ×100 sq ft) | y (actual price, lakh ₹) |
|---|---|---|
| 1 | 10 | 500 |
| 2 | 20 | 900 |

Noise assumed fixed: σ = 50.

**Testing θ = 45:**

Predictions: House 1 → θx = 45×10 = 450 (actual 500, error 50 = 1σ); House 2 → θx = 45×20 = 900 (actual 900, error 0 = 0σ).

- House 1 density: center=450, error=50 → exponent=−0.5 → p(y⁽¹⁾) = 0.00798 × e⁻⁰·⁵ ≈ **0.00484**
- House 2 density: center=900, error=0 → exponent=0 → p(y⁽²⁾) = 0.00798 × e⁰ = **0.00798**

**Multiplying — the joint plausibility:**

Because the two houses' noise terms are assumed independent (i.i.d. assumption), the joint plausibility of observing *both* outcomes together, given this θ, is the **product** of their individual densities (same logic as P(heads) × P(rolling a 6) = P(both) for independent events):

$$L(\theta=45) = p(y^{(1)}) \times p(y^{(2)}) = 0.00484 \times 0.00798 \approx 0.0000386$$

**Interpretation of this number:** "If θ=45 is the true price-vs-size relationship, the probability-density of the universe handing me *exactly this dataset* (House 1 at 500, House 2 at 900) is 0.0000386." It's a relative score attached to the whole hypothesis θ=45 — not to any single house.

**Testing θ = 47** (to compare):

Predictions: House 1 → 47×10=470 (actual 500, error 30 → 0.6σ); House 2 → 47×20=940 (actual 900, error 40 → 0.8σ).

- House 1: exponent = −(30)²/(2×2500) = −900/5000 = −0.18 → e⁻⁰·¹⁸=0.8353 → p = 0.00798×0.8353 ≈ **0.00667**
- House 2: exponent = −(40)²/(2×2500) = −1600/5000 = −0.32 → e⁻⁰·³²=0.7261 → p = 0.00798×0.7261 ≈ **0.00580**

$$L(\theta=47) = 0.00667 \times 0.00580 \approx 0.0000387$$

**Comparison table:**

| θ | L(θ) |
|---|---|
| 45 | 0.0000386 |
| 47 | 0.0000387 |

θ=47 is marginally better-supported by the data than θ=45. This is the **Maximum Likelihood Estimate (MLE)** search process: try candidate θ's, compute this joint-density product for each, and find the θ that maximizes it. Repeating for many more θ values (46, 46.5, 46.8, ...) and plotting L(θ) vs θ would trace out a smooth curve with a single peak — that peak's location is the MLE.

**Detective analogy used for intuition:** two suspects (θ=45 vs θ=47), fixed evidence (observed data: 500 and 900 — never changes). For each suspect, ask: "if this suspect (this θ) were the real culprit, how likely is it the crime scene looks exactly like this?" Whichever suspect makes the evidence look most expected wins.

**Why this connects to least squares (preview, not yet fully derived):** The term driving each density is `e^{-(\text{error})^2/(2\sigma^2)}`. Maximizing the *product* of these terms requires making each individual **error² as small as possible** (smaller error² → less negative exponent → larger e^(...) → larger density). So "maximize the product of densities" collapses into "minimize the sum of squared errors" — i.e., ordinary least squares, now justified as the MLE solution under Gaussian noise, rather than an arbitrary heuristic.

### Example 3 — Decoding the product notation ∏

$$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta)$$

**Symbol-by-symbol breakdown:**

- **m** — total number of training examples (2 in the toy example; could be 500, 10,000, etc. in a real dataset)
- **i** — an index/counter, like a spreadsheet row number. i=1 → House 1, i=2 → House 2, etc. The **(i)** superscript in parentheses is *not* an exponent — it's a label meaning "the i-th training example." (Important recurring CS229 notational convention.)
- **∏ (capital Pi)** — "product notation," the multiplication analogue of Σ (summation). Just as Σ adds up terms across an index range, ∏ multiplies them:
  $$\prod_{i=1}^{3} i = 1 \times 2 \times 3 = 6$$
- Reading the full expression: "Multiply together the term p(y⁽ⁱ⁾|x⁽ⁱ⁾;θ), for every i from 1 to m."

**Unrolled (no notation), general m:**

$$L(\theta) = p(y^{(1)}|x^{(1)};\theta) \times p(y^{(2)}|x^{(2)};\theta) \times \cdots \times p(y^{(m)}|x^{(m)};\theta)$$

**Matching to the 2-house example (m=2):**

$$L(\theta) = \prod_{i=1}^{2} p(y^{(i)}|x^{(i)};\theta) = p(500 \mid 10; \theta) \times p(900 \mid 20; \theta)$$

— exactly the hand-computed calculation from Example 2. The ∏ notation is purely a compact stand-in for "multiply this across every training example," useful once m is large (e.g., 500 houses — nobody wants to write "×" 500 times by hand).

**The semicolon convention:** in `p(y⁽ⁱ⁾|x⁽ⁱ⁾;θ)`, the vertical bar `|` means "given" (conditioning on an actual random variable, x⁽ⁱ⁾), while the semicolon `;` before θ signals that **θ is treated as a parameter, not a random variable** — a fixed (if unknown) quantity being chosen/tested, not something with its own probability distribution. This distinction becomes more significant later when contrasting frequentist vs. Bayesian treatments of θ.

**What L(θ) is as an object:** a function that takes a candidate θ, runs it through every training example, computes each one's density, multiplies them all together, and returns one score representing "how plausible is my entire dataset, if this θ were the true relationship." MLE = search over all θ to maximize this score.

## Key Takeaways

- **ε is the residual by definition** (y − θᵀx), not an independently existing physical quantity — modeling it as Gaussian(0, σ²) is a CLT-motivated assumption, not a derived fact.
- **Mean-zero assumption on ε is free** as long as the model has an intercept term (x₀=1) — any nonzero mean gets absorbed into θ₀.
- **σ² (variance) is a mathematical convenience** — its actual value turns out not to matter for finding the MLE θ.
- **Density ≠ probability.** Density is "probability per unit length" (a height/rate); actual probability requires integrating density over an interval. Density can exceed 1; probability never can.
- **Density × width ≈ probability** for narrow intervals — this approximation becomes exact as interval width → 0, which is what the integral formalizes.
- **σ is standard deviation (same units as y), σ² is variance (squared units)** — don't casually call σ "variance," this is a common and consequential mix-up.
- **Likelihood and probability are the same formula, asked with different variables held fixed:** probability fixes θ and varies y (asking about future/unknown outcomes); likelihood fixes y (observed data) and varies θ (asking which hypothesis best explains what already happened).
- **L(θ) is a relative score, not a probability distribution over θ** — it does not integrate to 1 over all θ. Only compare L(θ) values against each other; don't read a single L(θ) value as a "percentage chance θ is correct."
- **Training an ML model = always operating in "likelihood mode"** — data is fixed and given, θ is the unknown being searched over to maximize plausibility of that fixed data.
- **∏ (product notation) is to multiplication what Σ is to addition** — a compact way to write "repeat this operation across an index range."
- **Taking the log converts a product into a sum** (log(a×b×c) = log a + log b + log c), making the likelihood numerically tractable and easy to differentiate — and since log is monotonically increasing, the θ that maximizes L(θ) also maximizes ℓ(θ) = log L(θ); the peak's location never moves.
- **The log-likelihood splits into two terms:** a constant term (independent of θ, so irrelevant to the argmax) and a term with a negative sign in front of a sum-of-squared-errors. That negative sign is precisely what flips "maximize" into "minimize."
- **The big reveal — fully derived:** maximizing ℓ(θ) under Gaussian, i.i.d., zero-mean noise reduces exactly to minimizing Σ(y⁽ⁱ⁾−θᵀx⁽ⁱ⁾)², i.e., minimizing J(θ) = ½Σ(y⁽ⁱ⁾−θᵀx⁽ⁱ⁾)². Least-squares linear regression is the MLE solution under this explicit probabilistic model — not an arbitrary heuristic.
- **σ² never affects which θ is optimal** — it only ever appears as a positive scaling constant throughout the derivation (confirmed algebraically, not just asserted). It would matter later for reasoning about *confidence* in the estimate, but not for finding θ itself.

## Open Questions / Next Steps

- **Next natural topic in the lecture arc:** having established the probabilistic justification for least-squares (linear regression, "Regression 1" in Ng's framing), the lecture moves to **locally weighted regression (LWR)** — a non-parametric method that fits a separate weighted regression around each query point rather than one global θ. Worth covering with the same first-principles, numeric-example approach.
- Also upcoming per Ng's lecture arc: the transition from linear regression's probabilistic setup to **logistic regression** — swapping the Gaussian assumption on y for a Bernoulli assumption (since y becomes binary), and re-deriving a new cost function via the same MLE machinery just built here. This is a strong candidate for direct comparison against today's derivation, since the mechanics (likelihood → log-likelihood → argmax) repeat almost identically.
- Possible future thread: frequentist vs. Bayesian treatment of θ (hinted at via the semicolon-vs-bar notation discussion earlier in this session).

## References

- Stanford CS229: Machine Learning, Autumn 2018 — Lecture 3, "Locally Weighted & Logistic Regression" (Andrew Ng), probabilistic interpretation section.
