# CS229 Lecture 2 — Linear Regression, Gradient Descent & Normal Equation

> **Session Dates:** 2026-07-02 (part 1) · 2026-07-06 (part 2)
> **Duration context:** Deep-dive across two sessions
> **Tags:** `#linear-regression` `#gradient-descent` `#cost-function` `#partial-derivatives` `#chain-rule` `#contour-plots` `#convexity` `#global-optima` `#gradient-notation` `#trace-trick` `#matrix-calculus` `#normal-equation` `#cs229` `#andrew-ng` `#ml-fundamentals`

---

## Overview

**Session 1 (2026-07-02)** walked through Andrew Ng's CS229 Lecture 2 (Autumn 2018) from minute 8 through minute 35: the hypothesis function, dummy feature trick, cost function J(θ), gradient descent update rule, full chain rule derivation of (h(x)−y)·xⱼ, why J(θ) is always a quadratic bowl, global vs local optima, and contour plots. Every design decision was explained from first principles with analogies rooted in Indian real estate and cricket.

**Session 2 (2026-07-06)** picked up the same lecture at the 1-hour mark, where Andrew Ng shifts from scalar partial-derivative notation to vector/matrix notation using the gradient symbol ∇ (nabla). From there, the session worked through the trace trick — the mechanism Andrew uses to differentiate matrix expressions — and derived the normal equation θ = (XᵀX)⁻¹Xᵀy from scratch, verifying every step against plain scalar calculus on a toy 2-house dataset.

---

## Core Concepts

### 1. The Hypothesis Function

**Formula:**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂
```

Or in compact sum form:
```
h(x) = Σ θⱼxⱼ   (j from 0 to n)
```

**What it means in plain English:**
> "Base price + (price per sq ft × area) + (price per bedroom × bedrooms)"

This is exactly the mental math you do when evaluating a flat in Pimple Saudagar.

| Symbol | What it is | Example (house price) |
|--------|-----------|----------------------|
| `h(x)` | Model's prediction | ₹54L predicted price |
| `x₁, x₂` | Features / inputs from your data | sq ft, number of bedrooms |
| `θ₀` | Base price (bias/intercept) | ₹10L — even a 0 sqft plot costs something |
| `θ₁` | Weight for x₁ | ₹500 per sq ft |
| `θ₂` | Weight for x₂ | ₹2L per bedroom |

**Key insight:** The x values come from your dataset. The θ values are what the model *learns* via gradient descent.

---

### 2. The Dummy Feature x₀ = 1

Andrew defines x₀ = 1 so that θ₀ also has an x next to it:

```
h(x) = θ₀x₀ + θ₁x₁ + θ₂x₂     (x₀ = 1, so θ₀×1 = θ₀)
```

**Why?** Without it, θ₀ is a special case with no feature — you can't write the whole thing as a clean sum `Σ θⱼxⱼ`. With x₀ = 1, every term has the same shape, so the feature vector becomes `[1, x₁, x₂]` and theta becomes `[θ₀, θ₁, θ₂]`, and their dot product gives h(x) in one operation.

**Cricket analogy:** If you want to say "total = extras + (runs per over × overs)", you could treat extras as a special case — or you could say "extras = extras × 1" and now everything is in the same `(rate × count)` format. Same result, cleaner math.

x₀ = 1 is called the **bias term** and appears in almost every ML model including neural networks.

---

### 3. The Cost Function J(θ)

**Formula:**
```
J(θ) = ½ · Σ ( h(xⁱ) − yⁱ )²
```

**Why each part is there:**

**Why square the difference?**
Three reasons working together:
1. **Kills the sign** — +10 and −10 errors cancel if you sum raw differences. Squaring makes both positive.
2. **Punishes big mistakes harder** — error of 10 → cost 100; error of 20 → cost 400. Twice as wrong = four times the penalty.
3. **Smooth and differentiable everywhere** — gives a clean parabola (bowl shape) that gradient descent can roll down without getting stuck.

**Why not use absolute value |error|?**
The absolute value function has a **sharp corner at zero** — the slope jumps instantly from −1 to +1 with no smooth transition. At exactly that point, two valid slopes exist (−1 and +1) so the derivative is undefined. Gradient descent needs to ask "which way is downhill?" at every point, and at the corner there is no clear answer. (This variant is called MAE — Mean Absolute Error — and is used in practice but needs special handling.)

![Absolute value vs squared error — the sharp corner problem](assets/absolute-vs-squared-error.svg)

**Why the ½?**
Pure cleanup. When you differentiate (h−y)², the power rule brings down a 2. Multiplying by ½ in advance cancels it:
```
d/dθ [½·(h−y)²] = (h−y)·...
```
The ½ doesn't change where the minimum is — it just keeps the gradient formula clean. Andrew placed it there knowing the derivative was coming.

---

### 4. Partial Derivatives vs Regular Derivatives

**Regular derivative** — one variable:
```
J(θ) = θ²   →   dJ/dθ = 2θ
```

**Partial derivative** — multiple variables (θ₀, θ₁, θ₂):
```
∂J/∂θ₁ = "how does J change if I nudge only θ₁, holding θ₀ and θ₂ frozen?"
```

The ∂ symbol (instead of d) signals: "there are other variables around, but I'm ignoring them right now."

**Cricket analogy:** "How much does the team total change if Virat scores one more run?" — you're holding Rohit and extras fixed and only watching Virat. That's a partial derivative.

The mechanics of computing a partial derivative are **identical** to regular derivatives — same power rule, same chain rule — you just mentally cross out the other variables while you work.

---

### 5. The Gradient Descent Update Rule

**Formula:**
```
θⱼ := θⱼ − α · ∂J(θ)/∂θⱼ
```

![Gradient descent update rule — decoded symbol by symbol](assets/gradient-descent-update-rule.svg)

| Part | Meaning |
|------|---------|
| `θⱼ :=` | Replace θⱼ with a new value |
| `θⱼ` (RHS) | Current value — start from where you are |
| `α` | Learning rate — how big a step to take |
| `∂J/∂θⱼ` | Slope of the loss at your current position (which way is uphill, how steep) |
| `− sign` | Subtract the slope → walk **downhill** |

**The entire algorithm in one sentence:** Take a small step in the direction that reduces loss.

**The − sign is the genius of gradient descent.** The derivative tells you which way is uphill. Subtracting it walks you downhill. One character, entire algorithm.

**Learning rate α sensitivity:**
- Too large → overshoot the minimum, bounce around
- Too small → crawl and take forever to converge
- Just right → smooth convergence to the bottom of the bowl

**Connection to Karpathy / PyTorch:**
- `∂J/∂θⱼ` is what `.backward()` computes (backprop fills in the gradients)
- `θⱼ := θⱼ − α · gradient` is what `optimizer.step()` applies
- Andrew is building the same thing from scratch, without the library

---

### 6. Full Derivation — How Andrew Reaches (h(x) − y) · xⱼ

Starting from:
```
J(θ) = ½ · Σ ( h(x) − y )²
```

We want: `∂J/∂θⱼ`

**Step 1 — Power rule on the square (chain rule outer layer):**
```
∂J/∂θⱼ = ½ · 2 · (h(x) − y) · ∂/∂θⱼ (h(x) − y)
```

**Step 2 — ½ × 2 cancel:**
```
∂J/∂θⱼ = (h(x) − y) · ∂/∂θⱼ (h(x) − y)
```

**Step 3 — Expand h(x) and apply partial derivative:**
```
h(x) = θ₀x₀ + θ₁x₁ + θ₂x₂ + ...

∂/∂θⱼ (h(x) − y) = ∂/∂θⱼ (Σ θₖxₖ − y)
```

Freeze all θ's except θⱼ:
- Every term `θₖxₖ` where k ≠ j → treated as constant → derivative = 0
- The term `θⱼxⱼ` → derivative = **xⱼ**
- `y` is a data label (constant) → derivative = 0

**Step 4 — Final result:**
```
∂J/∂θⱼ = (h(x) − y) · xⱼ
```

So the full update rule becomes:
```
θⱼ := θⱼ − α · (h(x) − y) · xⱼ
```

![Chain rule derivation step by step](assets/chain-rule-derivation.svg)

**Three moves total:**
1. Power rule → brings down 2, cancels with ½
2. Chain rule → differentiate the inside too
3. Partial derivative of h(x) w.r.t. θⱼ → all other terms die, only xⱼ survives

**Intuition behind (h(x)−y)·xⱼ:**

| Factor | What it captures |
|--------|-----------------|
| `(h(x) − y)` | How wrong were you, and in which direction? Positive = over-predicted, negative = under-predicted, zero = perfect |
| `xⱼ` | How much did feature j contribute? Large xⱼ = θⱼ had big influence = deserves bigger correction |

**Cricket analogy:** You predicted Virat would score 80, he scored 50. Error = 30. If the "home ground" feature (x₁) was large (strong home advantage), that feature gets blamed more and θ₁ gets corrected harder. A feature that wasn't active (xⱼ ≈ 0) gets no blame.

---

### 7. Why J(θ) Is a Quadratic Bowl — The Algebra

**User question that triggered this:** "Linear regression is quadratic — but why? And why bowl?"

The chain of logic:

1. `h(x) = θ·x` is **linear** in θ — just a straight line as θ varies
2. `(h(x) − y) = (θ·x − y)` is still linear in θ
3. Squaring a linear expression always gives a **quadratic**: `(θ·x − y)² = aθ² + bθ + c`
4. A quadratic with positive leading coefficient (a > 0) **always opens upward** — that's a bowl
5. Summing over many training examples: sum of quadratics = still quadratic = still a bowl

```
Concrete example (x=1, y=3):
J(θ) = ½·(θ−3)² = ½·(θ²−6θ+9) = 0.5θ² − 3θ + 4.5
                    ↑
               aθ² + bθ + c form. a=0.5 > 0 → opens upward.
```

![Why J(theta) is a quadratic bowl](assets/why-j-theta-is-a-bowl.svg)

**The progression by number of weights:**

| Weights | Shape | Name |
|---------|-------|------|
| 1 (θ) | Parabola in 2D | Curve with one bottom |
| 2 (θ₀, θ₁) | Paraboloid in 3D | Cereal bowl / satellite dish |
| n (θ₀…θₙ) | n-dim paraboloid | Same guarantee, unvisualizable |

**Key point:** In all cases, there is exactly one lowest point. The bowl shape makes it mathematically impossible to have two different bottoms.

---

### 8. Global Optima — Why Linear Regression Has No Local Traps

Andrew makes a big deal of this at ~minute 30–35 because it's a rare guarantee.

**Convex function:** A function where drawing a line between any two points on the curve, that line stays above (or on) the curve. A bowl satisfies this. A bumpy mountain range doesn't.

**Consequence:** A convex function has **only one lowest point — the global minimum.** No valley inside a valley, no place where gradient descent can get stuck.

![Global optima vs local optima](assets/global-vs-local-optima.svg)

| Linear Regression (convex) | Neural Nets etc. (non-convex) |
|---------------------------|-------------------------------|
| Single perfect bowl | Bumpy surface, many valleys |
| One global minimum | Many local minima |
| GD always finds the best answer | GD can get stuck |
| Guaranteed convergence | No guarantee — depends on init, architecture |

**Why this matters:** Most ML models (deep networks, etc.) are non-convex. Training them is hard and involves tricks (random restarts, momentum, careful initialization). Linear regression doesn't need any of that — the bowl shape makes it bulletproof.

---

### 9. Contour Plots — The Bird's-Eye View

**What is a contour?**
Imagine pouring water into the 3D bowl at different heights and photographing each waterline from directly above. Each waterline is a **contour** — a ring where J(θ) has the same value all the way around. Like altitude lines on a hiking / topo map.

- Innermost ring = lowest loss
- Outer rings = higher loss
- Center point = global minimum

**Why Andrew draws contours instead of 3D bowls:**
- Easier to draw on a blackboard
- Easier to show gradient descent steps as arrows
- Richer information visible (shape of ellipses reveals learning dynamics)

**Gradient descent on contour plots:**
Each GD step moves **perpendicular to the nearest contour ring** — that's the steepest downhill direction. The path spirals inward, crossing rings as it goes, until it reaches the center.

![Contour plot with gradient descent path](assets/contour-plot-gradient-descent.svg)

**What the ellipse shape tells you:**

The shape of the contours is not just aesthetic — it reveals how the loss behaves in each direction.

![Contour shape and feature scaling](assets/contour-shape-feature-scaling.svg)

| Contour shape | Cause | Consequence |
|---------------|-------|-------------|
| Narrow / elongated ellipses | Features at very different scales (sq ft vs bedrooms) | GD zigzags slowly, inefficient |
| Circular / round ellipses | Features normalized to same range | GD goes straight to center, fast convergence |

This is exactly why **feature scaling / normalization** matters — it rounds out the ellipses and makes gradient descent converge much faster.

---

### 10. The Gradient Symbol ∇ (Nabla) — "Reverse Triangle" Notation

The upside-down triangle Andrew draws is **not a new operator to derive** — it's shorthand for stacking all the partial derivatives you already know how to compute into one vector.

Given the previously-derived scalar result:

```
∂J/∂θⱼ = (h(x) − y)·xⱼ
```

Doing this separately for every θ₀, θ₁, ..., θₙ and stacking the results gives:

```
∇_θ J(θ) =
[ ∂J/∂θ₀ ]
[ ∂J/∂θ₁ ]
[   ...   ]
[ ∂J/∂θₙ ]
```

**Key rule:** ∇_θ J(θ) always has the *same shape* as θ itself — the same "gradient shape matches variable shape" rule from the backprop-ninja sessions applies here too.

This collapses the per-component update rule:
```
θⱼ := θⱼ − α · ∂J/∂θⱼ   (repeat for every j)
```
into a single vector equation:
```
θ := θ − α · ∇_θ J(θ)
```
Same math, just written for the whole parameter vector at once.

---

### 11. Trace of a Matrix — the Basics

**tr(A)** = sum of the diagonal entries of a square matrix A.

Example:
```
A = [[1, 2], [3, 4]]
tr(A) = 1 + 4 = 5
```

**Key fact:** if a is a scalar (1×1 matrix), tr(a) = a. This is what lets Andrew rewrite the scalar J(θ) as tr(J(θ)), opening the door to applying trace identities to it.

---

### 12. The Cyclic Property — tr(AB) = tr(BA)

Matrix multiplication is not commutative (AB ≠ BA in general, and dimensions often don't even match both ways), but their **traces** are always equal when both products are defined.

**Verification with real numbers:**
```
A (2×3) = [[1,2,3],[4,5,6]]
B (3×2) = [[1,0],[0,1],[1,1]]

AB (2×2) = [[4,5],[10,11]]  →  tr(AB) = 4 + 11 = 15

BA (3×3) = [[1,2,3],[4,5,6],[5,7,9]]  →  tr(BA) = 1 + 5 + 9 = 15
```
Same trace, even though AB and BA are different sizes. This is the **cyclic property**: you can rotate matrices inside a trace, but not arbitrarily reorder them.
```
tr(AB) = tr(BA)
tr(ABC) = tr(CAB) = tr(BCA)   (cycling allowed, arbitrary swaps like tr(ACB) are NOT equal in general)
```

---

### 13. Trace Derivative Identities (the Toolbox)

Memorized building blocks, analogous to memorizing d/dx(x²) = 2x once and reusing it:

```
1. ∇_A tr(AB) = Bᵀ
2. ∇_A tr(ABAᵀC) = CAB + CᵀABᵀ
3. ∇_A tr(AᵀB) = B
```

The cyclic property is how you massage an expression into a form matching one of these templates before applying the identity.

---

### 14. Why the Transpose Appears in J(θ) = ½(Xθ − y)ᵀ(Xθ − y)

Original scalar cost function:
```
J(θ) = ½ Σᵢ (h(xᵢ) − yᵢ)²
```

**Step 1 — Xθ is all predictions at once.** Stack every training example as a row in matrix X, θ as a column vector. Then Xθ is a column vector of every prediction:
```
X = [[1,1000,2],[1,1500,3],[1,800,1]]   θ = [θ₀,θ₁,θ₂]ᵀ

Xθ = [h(x₁), h(x₂), h(x₃)]ᵀ
```
So `e = Xθ − y` is the vector of all errors at once.

**Step 2 — the transpose produces sum-of-squares.** You need Σ(errors)², i.e. the dot product of e with itself: `eᵀe`.
```
e = [2, -1, 3]ᵀ
eᵀe = [2,-1,3]·[2,-1,3]ᵀ = 4 + 1 + 9 = 14 = Σ(errors)²
```
The transpose exists purely to flip the first copy of e from a column into a row, so "row × column" matrix multiplication produces the sum of squares.

**Step 3 — putting it together:**
```
Σ(h(xᵢ) − yᵢ)² = eᵀe = (Xθ − y)ᵀ(Xθ − y)
J(θ) = ½(Xθ − y)ᵀ(Xθ − y)
```
No new concept — same Σ(error)² from day one, rewritten as one matrix expression instead of a summation with a for-loop.

---

### 15. Expanding (Xθ − y)ᵀ(Xθ − y) into Four Terms

Two transpose rules needed:
```
Rule 1: (A − B)ᵀ = Aᵀ − Bᵀ
Rule 2: (AB)ᵀ = BᵀAᵀ   (order reverses — forced by matrix dimensions, not optional)
```

Let a = Xθ. Expand (a − y)ᵀ(a − y) = aᵀa − aᵀy − yᵀa + yᵀy.

Substitute a = Xθ back in, using Rule 2 for aᵀ = (Xθ)ᵀ = θᵀXᵀ:
```
aᵀa = θᵀXᵀXθ
aᵀy = θᵀXᵀy
yᵀa = yᵀXθ
yᵀy = yᵀy
```
So:
```
(Xθ − y)ᵀ(Xθ − y) = θᵀXᵀXθ − θᵀXᵀy − yᵀXθ + yᵀy
J(θ) = ½(θᵀXᵀXθ − θᵀXᵀy − yᵀXθ + yᵀy)
```

**Merging the middle terms:** θᵀXᵀy and yᵀXθ are both scalars (1×1), and a scalar equals its own transpose:
```
(θᵀXᵀy)ᵀ = yᵀ(Xᵀ)ᵀθ = yᵀXθ  ⟹  θᵀXᵀy = yᵀXθ
```
So they combine into a single −2θᵀXᵀy term:
```
J(θ) = ½(θᵀXᵀXθ − 2θᵀXᵀy + yᵀy)
```

---

### 16. Differentiating Term-by-Term to Reach ∇_θ J(θ)

**Term 1 — ∇_θ(θᵀXᵀXθ):** matches identity `∇_A tr(ABAᵀC) = CAB + CᵀABᵀ` with A=θ, B=I, C=XᵀX.
```
CAB + CᵀABᵀ = (XᵀX)θ + (XᵀX)ᵀθ = XᵀXθ + XᵀXθ = 2XᵀXθ
```
(uses the fact that XᵀX is always symmetric, so (XᵀX)ᵀ = XᵀX)

**Term 2 — ∇_θ(−2θᵀXᵀy):** matches identity `∇_A tr(AᵀB) = B` with A=θ, B=Xᵀy.
```
∇_θ(θᵀXᵀy) = Xᵀy  ⟹  ∇_θ(−2θᵀXᵀy) = −2Xᵀy
```

**Term 3 — ∇_θ(yᵀy):** no θ present → gradient = 0.

**Combining:**
```
∇_θ J(θ) = ½(2XᵀXθ − 2Xᵀy + 0) = XᵀXθ − Xᵀy
```

---

### 17. The Normal Equation

Set the gradient to zero (minimum ⟺ slope = 0, same logic as 1D calculus, now in vector form):
```
XᵀXθ − Xᵀy = 0
XᵀXθ = Xᵀy
θ = (XᵀX)⁻¹Xᵀy
```
A closed-form solution for θ — no iteration, no learning rate, no convergence loop, provided XᵀX is invertible.

| Property | Gradient Descent | Normal Equation |
|----------|-----------------|-----------------|
| Requires learning rate α | Yes | No |
| Iterative | Yes | No (one-shot) |
| Scales with feature count | Well (O(n) per step) | Poorly (O(n³) inversion) |
| Requires XᵀX invertible | No | Yes |
| Handles very large datasets | Better | Worse |

---

## Worked Examples

### Verifying ∇_θ J(θ) = XᵀXθ − Xᵀy Against Plain Scalar Calculus

Toy dataset — 2 houses, 1 feature plus bias:
```
X = [[1, 2], [1, 3]]     y = [5, 8]ᵀ
```
(Row 1: x₀=1, x₁=2, price=5. Row 2: x₀=1, x₁=3, price=8.)

**Direct way (plain partial derivatives):**
```
h₁ = θ₀ + 2θ₁,   h₂ = θ₀ + 3θ₁
J(θ) = ½[(θ₀+2θ₁−5)² + (θ₀+3θ₁−8)²]

∂J/∂θ₀ = (θ₀+2θ₁−5) + (θ₀+3θ₁−8) = 2θ₀ + 5θ₁ − 13
∂J/∂θ₁ = 2(θ₀+2θ₁−5) + 3(θ₀+3θ₁−8) = 5θ₀ + 13θ₁ − 34
```

**Matrix formula way — compute XᵀXθ − Xᵀy:**
```
Xᵀ = [[1,1],[2,3]]
XᵀX = [[1,1],[2,3]] × [[1,2],[1,3]] = [[2,5],[5,13]]
XᵀXθ = [2θ₀+5θ₁, 5θ₀+13θ₁]ᵀ

Xᵀy = [[1,1],[2,3]] × [5,8]ᵀ = [13, 34]ᵀ

XᵀXθ − Xᵀy = [2θ₀+5θ₁−13, 5θ₀+13θ₁−34]ᵀ
```

**Result:** row 1 gives `2θ₀+5θ₁−13`, row 2 gives `5θ₀+13θ₁−34` — exactly matching the two partial derivatives computed the slow way, term for term. The matrix formula is not a different technique; it packs both scalar partial-derivative equations into one vector equation.

### Verifying tr(AB) = tr(BA) with Real Numbers

```
A (2×3) = [[1,2,3],[4,5,6]]
B (3×2) = [[1,0],[0,1],[1,1]]

AB = [[4,5],[10,11]]  →  tr(AB) = 15
BA = [[1,2,3],[4,5,6],[5,7,9]]  →  tr(BA) = 15
```
Confirms the cyclic property even though AB and BA are different-sized matrices.

---

## Diagrams

### Absolute value vs squared error
![Sharp corner problem — why |error| breaks gradient descent](assets/absolute-vs-squared-error.svg)

Shows why the V-shape of |error| has an undefined slope at zero (sharp corner), while error² gives a smooth bowl with a well-defined slope everywhere.

### Gradient descent update rule decoded
![Gradient descent update rule](assets/gradient-descent-update-rule.svg)

Every symbol in `θⱼ := θⱼ − α · ∂J/∂θⱼ` annotated, plus the final expanded form `θⱼ := θⱼ − α·(h(x)−y)·xⱼ`.

### Chain rule derivation
![Full 4-step derivation from J(theta) to (h(x)-y)*xj](assets/chain-rule-derivation.svg)

The complete step-by-step derivation: power rule → ½ cancels 2 → chain rule → partial of h(x) gives xⱼ.

### Global vs local optima
![Convex bowl vs bumpy non-convex surface](assets/global-vs-local-optima.svg)

Side-by-side comparison: the perfect bowl of linear regression vs a bumpy non-convex surface with multiple local minima.

### Contour plot with gradient descent path
![Contour rings with GD spiral path inward](assets/contour-plot-gradient-descent.svg)

Left: what contours are (bird's-eye view of bowl, rings = same J value). Right: gradient descent steps spiraling inward toward the global minimum.

### Contour shape and feature scaling
![Narrow vs round contour ellipses](assets/contour-shape-feature-scaling.svg)

How elongated contours (unscaled features) cause GD to zigzag, while circular contours (normalized features) let GD converge directly.

---

## Comparisons & Tradeoffs

### Absolute Error vs Squared Error

| Property | \|error\| (MAE) | error² (MSE) |
|----------|----------------|-------------|
| Kills sign problem | Yes | Yes |
| Differentiable everywhere | No (corner at 0) | Yes (smooth bowl) |
| Punishes outliers | Linearly | Quadratically |
| Works with gradient descent cleanly | Needs special handling | Yes, naturally |
| Used in practice | Yes (robust tasks) | Yes (standard regression) |

### Convex vs Non-Convex Loss Surfaces

| Property | Convex (linear regression) | Non-convex (deep learning) |
|----------|---------------------------|---------------------------|
| Shape | Perfect bowl | Bumpy, many valleys |
| Local minima | None | Many |
| GD guarantee | Always finds global minimum | No guarantee |
| Starting point matters | No | Yes, critically |
| Example | Linear regression, logistic regression | Neural networks |

### Gradient Descent vs Normal Equation

| Property | Gradient Descent | Normal Equation |
|----------|-----------------|-----------------|
| Requires learning rate | Yes | No |
| Iterative | Yes | No (one-shot) |
| Scales with features | O(kn) per iteration | O(n³) inversion |
| Requires invertible XᵀX | No | Yes |
| Handles large datasets | Better | Worse |
| Practical threshold | Default choice | Feasible up to ~n=10,000 features |

---

## Key Takeaways

- `h(x) = Σ θⱼxⱼ` is just a weighted sum of features — the θ's are what the model learns, the x's are your data
- `x₀ = 1` (dummy feature) exists purely so θ₀ fits the same `θⱼxⱼ` pattern — allows compact matrix math and uniform treatment
- The ½ in J(θ) is placed intentionally to cancel the 2 that appears when you differentiate the square
- Squaring the error is chosen because it is smooth (differentiable everywhere), sign-agnostic, and outlier-penalizing — unlike |error| which has an undefined derivative at 0
- The − sign in gradient descent is the whole algorithm: derivative tells you uphill, subtracting walks you downhill
- `∂J/∂θⱼ = (h(x)−y)·xⱼ` comes from three mechanical steps: power rule → chain rule → partial derivative of h(x) gives xⱼ
- J(θ) is quadratic (bowl-shaped) because squaring a linear-in-θ expression always gives aθ² + bθ + c with a > 0
- Quadratic = only one bottom = guaranteed single global minimum — this is what "convex" means
- Contour plots are the bird's-eye view of the 3D bowl — each ring = same loss value, innermost ring = lowest loss
- Elongated contour ellipses mean features are at very different scales → GD zigzags slowly; round ellipses mean features are normalized → GD converges directly
- Feature scaling / normalization rounds out the contour ellipses and makes gradient descent converge faster
- The gradient symbol ∇ is not a new operator — it's a vector stacking already-known partial derivatives, and it always matches the shape of θ
- The trace trick exists solely to make matrix-expression differentiation mechanical, the same way memorizing d/dx(x²)=2x avoids re-deriving it from limits every time
- tr(AB) = tr(BA) even when AB and BA are different sizes — this cyclic property is what lets you rearrange expressions to match a known trace-derivative identity
- The transpose in (Xθ − y)ᵀ(Xθ − y) has one job: converting a vector of errors into a scalar sum-of-squares via a dot product (row × column)
- θᵀXᵀy and yᵀXθ are equal because both are scalars, and a scalar always equals its own transpose — this is why they merge into −2θᵀXᵀy
- The normal equation θ = (XᵀX)⁻¹Xᵀy is the exact, closed-form counterpart to iterative gradient descent — same objective, no learning rate, but requires XᵀX to be invertible and scales as O(n³)
- Practical calibration: gradient descent intuition, the existence and tradeoffs of the normal equation, and general matrix-notation fluency transfer directly to real ML engineering. The trace-identity algebra is worth doing once for the "verify matrix calculus against scalar calculus" muscle, not for memorization.

---

## Open Questions / Next Steps

- Why does XᵀX need to be invertible for the normal equation to work, and what happens (multicollinearity) when it isn't?
- What is the difference between batch, stochastic, and mini-batch gradient descent?
- When does gradient descent fail to converge — what does the loss curve look like when α is too large?
- How does **feature scaling / normalization** work in practice (min-max, z-score)?
- What happens to the convexity guarantee when we add regularization (L1/L2)?

---

## References

- **Lecture:** Stanford CS229 Machine Learning — Lecture 2 (Autumn 2018), Andrew Ng
- **Related:** Andrej Karpathy "Neural Networks: Zero to Hero" series (micrograd, backprop ninja) — same math, library-free; shared "gradient shape matches variable shape" rule
- **Concepts covered:** Linear regression, hypothesis function, cost function (MSE), gradient descent, partial derivatives, chain rule, convexity, global optima, contour plots, feature scaling, gradient notation (∇), trace trick, matrix calculus identities, normal equation derivation
