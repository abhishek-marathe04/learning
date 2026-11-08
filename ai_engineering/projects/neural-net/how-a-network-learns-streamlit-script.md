# How a Neural Network Learns — Presenter Script (Streamlit edition)

**Devocracy · ~20 min · slides + Streamlit app (`streamlit run app.py`)**

## How to read this script

- Normal paragraphs = **words to say** (a spine, not a cage).
- *Italic in brackets* = **stage directions**.
- `[SLIDE n]` = advance the deck. `[APP: Page n]` = switch to that browser tab.
- **Bold button names** = click that control in the app.
- `(m:ss)` = running clock.

## What changed from the original slide demos

- **Page 1 (Keep score)** replaces slides 8–9. You do the wiggle test and the nudge live, and PyTorch confirms the numbers.
- **Page 3 (Backprop)** replaces slide 13. The backward pass happens one click at a time.
- Pages 2, 4 and 5 are the three original demos, now running live in the app. Each training run is precomputed, so you can replay it step by step.
- **Skip slides 8, 9 and 13** (hide them in PowerPoint). Everything else in the deck stays.

## Pre-flight (30 minutes before)

1. Wi-Fi off. In the venv: `streamlit run app.py --server.headless true`. Wait for the warm-up spinner to finish.
2. Open **five browser tabs, one per page, in order** (Page 1 … Page 5). Switching is then just `Ctrl+Tab`, with no sidebar hunting on stage.
3. Browser full-screen (F11). Zoom so each page fits the projector with nothing below the fold.
4. In every tab press **Reset**. Check:
   - Page 1 reads loss **1,600.00**.
   - Page 2 is on **Good**.
   - Page 4 is on **Wave, 1 neuron**.
   - Page 5 is at step **0**.
5. Deck on slide 1, presenter view on, slides 8, 9 and 13 hidden.
6. Backup: if a page misbehaves, press **Reset** on that page and keep talking.

## The numbers you must know cold

| | |
|---|---|
| Example | 10 °C in, truth 50 °F, knobs start w = 1, b = 0 |
| Forward pass | guess 10 → miss −40 → loss **1,600** |
| Wiggle test | w: loss 1,592.01, slope ≈ −799 · b: loss 1,599.20, slope ≈ −80 |
| Autograd | w.grad = **−800**, b.grad = **−80** |
| One nudge (step 0.0005) | w 1.40, b 0.04, loss **1,293.12** (19% better) |
| Next nudges | loss ≈ 1,045 → 845 → … each nudge ~19% less wrong |
| Chain rule | −80 × 1 × 10 = −800 · −80 × 1 × 1 = −80 |

---

## 1 · Hook  (0:00)

`[SLIDE 1 — What does it mean for a machine to learn?]`

Quick show of hands — who used an AI assistant today? Copilot, ChatGPT, Claude, anything.

*(hands up)*

Keep it up if you could explain, mechanically, what "the model was trained" means. Not "it read the internet." What actually *happened* to it.

*(most hands drop)*

That's this talk. Last time we treated a model's weights as a frozen 140-gig blob and asked how to serve it. Today: where did those numbers come from?

`[SLIDE 2 — Nobody wrote the rules]`

The software we write: we write the rules, feed in data, get answers. Software that learns flips it. We give it data *and the answers*, and training produces the rules. Except the rules are numbers, not code.

Your coding assistant has no if-statement for Python syntax. Nobody wrote a rule for what a race condition is. Every behaviour you rely on was *found*, not written. So how does finding work?

`[SLIDE 3 — Where we're going]`

One tiny example, built by hand, then scaled all the way to an LLM. And everything you'll see computed today is real PyTorch, running live on this laptop. Nothing is a mock-up.

---

## 2 · Knobs and a scoreboard  (1:45)

`[SLIDE 4 — A model is a function with knobs]`

A model is two things: a fixed recipe of multiplies and adds, and a pile of knobs. People call them weights or parameters; same thing. **Training never edits the recipe.** It only turns the knobs. Our first model has two knobs. Llama 3.1 has 405 billion. Same idea.

`[SLIDE 5 — Meet the smallest neural network]`

Celsius to Fahrenheit. You know the formula; the model doesn't, and we're not allowed to tell it. Multiply by a knob w, add a knob b. That's one neuron. Start with a bad guess: w = 1, b = 0. The right answer is 1.8 and 32.

`[SLIDE 6 — Step 1 of learning: keep score]`

To improve, it needs a scoreboard. Feed in 10 degrees. Guess 10. Truth 50. Missed by 40. Square it: **1,600**. That's the loss. Squaring means too high and too low both count, and big misses hurt more.

`[SLIDE 7 — Learning = turning the knobs so the loss goes down]`

*(pause, let it sit)*

That's the definition. Learning is turning the knobs so that one number goes down. Everything else is doing it *efficiently*. Let's do it, live.

---

## 3 · APP Page 1 — Keep score  (3:45)

`[APP: Page 1 — Keep score]` *(Ctrl+Tab)*

This is exactly the slide, but live. w is 1, b is 0. Input 10, truth 50. Guess 10, miss minus 40, loss 1,600.

**The forward pass** — *(drag w up slowly to ~2, then back)*

Watch the numbers as I turn the w knob. Guess goes up, the miss shrinks, the loss drops... and back. Everything you just saw — input, multiply, add, compare, square — is the **forward pass**. Run the recipe with the current knobs, get a score.

*(press **Reset**)*

**The wiggle test** — So which way should each knob turn? The dumbest honest way: wiggle it and look.

*(click **Wiggle test**)*

Nudge w by 0.01: loss drops to 1,592, down about 8. Slope: roughly minus 800. Nudge b by 0.01: down 0.8, slope minus 80.

The **sign** is the direction: negative means "turn me up." The **size** is how much that knob matters right now. w matters ten times more, because it multiplies the input, 10. That slope, for every knob, is the **gradient**.

**Ask PyTorch** — Now let's ask the real thing.

*(click **Ask PyTorch**)*

Four lines of PyTorch: make w and b, compute the loss, call `loss.backward()`, print the gradients. *(point)* **Minus 800, minus 80.** Exactly. Our wiggle test was an estimate; this is exact. And notice: it didn't wiggle anything. It got both slopes in one backward pass. Hold that thought.

**The nudge** — Now the whole update rule: new knob = old knob minus step size times slope. Step size 0.0005.

*(click **Take one nudge**)*

w goes to 1.4, b to 0.04. Loss: 1,600 to **1,293**. Nineteen percent less wrong.

*(click **Take one nudge** 3–4 more times, slowly)*

1,045... 845... every nudge, about a fifth less wrong. Look at the sparkline. That's training. There is nothing else.

> **Optional beat, only if on time:** keep nudging until the loss is near zero. "Perfect score! And the formula is... w ≈ 5, b ≈ 0.4. Wrong! With only one example, lots of knob settings fit perfectly. That's why models need lots of data. So let's give it 24 readings."
> *Needs the w slider range to go to at least 6 — check this in rehearsal.*

*(press **Reset**)*

---

## 4 · Downhill  (7:00)

`[SLIDE 10 — Picture it: a ball rolling downhill]`

Zoom out. Every combination of knob values has a loss, so the knobs define a landscape. Training is a ball rolling downhill.

The step size has a famous name: the **learning rate**. Too small: right direction, takes forever. Too big: overshoot the valley, land higher, and it can explode.

`[SLIDE 11 — Live demo 1]` *(one beat)*

## 5 · APP Page 2 — Downhill  (7:45)

`[APP: Page 2 — Downhill]`

Now 24 real, noisy thermometer readings. Left: the data, the model's line in teal, and every miss dashed in amber. Right: the landscape — every possible w and b, coloured by loss. Dark is good. The white dot is where our knobs are.

*(press the **play** button)*

Watch the ball. It drops fast to the valley floor... then slides along the valley... and settles. *(read the status line)* w about 1.8, b about 32. Nobody told it the formula. It found it from noisy readings.

> *Read the exact numbers from the status line; they depend on the seeded data. Expect roughly 1.8 and 32.*

*(drag the **training step** slider back to about step 3)*

And because this is a recording of real training, I can rewind. Step 3: the ball has just dropped. Step 0: the starting guess. *(drag to the end)* Done.

Now let's get greedy.

*(select **Too big**, press **play**)*

Same data, bigger step. It overshoots... bounces to the other wall, higher each time... *(status goes red)* Exploded. One number set wrong and training is garbage. This is why people babysit learning rates.

*(select **Good** again)*

---

## 6 · Backprop  (10:00)

`[SLIDE 12 — Wiggling works. It just doesn't scale.]`

Back to that "one backward pass" thought. The wiggle test costs one extra model run per knob, per nudge. With 405 billion knobs, that's 405 billion runs for one step. Never finishes.

Backprop gives every slope in one forward and one backward pass — roughly three times the cost of one forward run, whatever the number of knobs. That's the trick that makes deep learning possible. Let's open it up.

## 7 · APP Page 3 — Backprop, one link at a time  (10:45)

`[APP: Page 3 — Backprop]`

This is our neuron as a chain of operations. Left to right, the forward pass we've already done: w times 10 gives the guess, minus 50 gives the miss, squared gives the loss, 1,600. Each box shows its forward value.

Now think gears. If gear A turns B three times as fast, and B turns C twice as fast, A turns C six times as fast. Rates multiply along a chain. Backprop walks the chain backwards, multiplying.

*(click **Backward one link**)*

Start at the loss. Squaring: at a miss of minus 40, the loss changes 2 × −40 = **minus 80** per unit of miss. So the miss's gradient is minus 80.

*(click **Backward one link**)*

Miss is guess minus 50: moves one-for-one. Times 1. The guess's gradient: minus 80.

*(click **Backward one link**)*

Guess is w times 10. Local rate: 10. So w's gradient: minus 80 times 1 times 10 — **minus 800**.

*(click **Backward one link**)*

And b just adds, rate 1: **minus 80**.

*(turn on **Compare with PyTorch**)*

Minus 800 and minus 80. The same numbers as the wiggle test and `loss.backward()` on page 1. That's all backprop is: each link knows only its own tiny local rate, and you multiply backwards. Chain a billion links and it's still just this. Karpathy's micrograd does it in about a hundred lines.

*(press **Reset**)*

`[SLIDE 14 — The training loop]`

In real code it's five lines. `model(x)` is the forward pass you watched. The loss line is the scoreboard. `loss.backward()` is the walk you just clicked through. `optimizer.step()` is the nudge. In essence, that loop trained every model you use.

---

## 8 · Bends  (12:30)

`[SLIDE 15 — Straight lines aren't enough, so add a bend]`

One problem: multiply-and-add can only ever draw a straight line. The world is curvy. The fix: after the multiply-and-add, if the result is negative, output zero. That's ReLU — a hinge. Many hinges, added up, trace any shape. That's all hidden neurons and layers are. Same training loop.

`[SLIDE 16 — Live demo 2]`

## 9 · APP Page 4 — Bends  (13:30)

`[APP: Page 4 — Bends]`

A wavy curve, and a network with one hidden neuron: four knobs.

*(press **play**)*

It finds the best it can with one bend... and it's stuck. The loss flattens around 0.38. No amount of training fixes a model that's too small.

*(select **8** hidden neurons, press **play**)*

Eight neurons. The orange curve snaps into shape. The thin coloured lines are each neuron's contribution; each dot is where that neuron bends.

Nobody assigned the jobs. Nobody told neuron 3 to handle the left dip. Each neuron found its role because the gradient kept saying "you — move a bit this way."

*(optional: press **Play** again to replay the bends moving into place; toggle **Show pieces** off to show the clean fit)*

---

## 10 · The next token  (15:15)

`[SLIDE 17 — An LLM learns the same way. The loss is surprise.]`

The leap. An LLM is trained on one task: predict the next token. Given `for i in range(` it puts a probability on every token it knows. We reveal the real one and score its surprise. 100% on the right token: zero. A random guess across 100,000 tokens: 11.5. Then: backprop, nudge, repeat, on trillions of tokens.

`[SLIDE 18 — Live demo 3]`

## 11 · APP Page 5 — Next letter  (16:00)

### Prep: read this before the talk (not spoken)

**What the model actually is.** A 27 × 27 grid of numbers, W. The 27 tokens are `.` plus a–z. `.` marks both the start and the end of a name, so `aarav` is read as `. a a r a v .`. Row = the letter it is looking at now. Column = the letter it guesses next. To predict, it takes the row for the current letter and turns those 27 numbers into probabilities that sum to 100% (softmax). That's the whole model. It has no memory beyond the one previous letter.

**What training does.** The data is 1,162 (current, next) letter pairs cut from 174 Indian first names. Each step it scores its surprise on every pair, backprop gives the slope for all 729 knobs, and it nudges them all at once. It's plain gradient descent with learning rate 30, run for 400 steps. This is the same loop as pages 1–2. Only the score changed, from squared miss to surprise.

**Surprise, in one line.** Surprise = −ln(p), where p is the probability it gave the letter that really came next.

| p it gave the right letter | surprise |
|---|---|
| 100% | 0 |
| 72% (`y` → `a`) | 0.33 |
| 1 in 27 = 3.7% (pure guess) | 3.30 |
| ~0% | very large |

Loss = average surprise over all pairs. Lower = less surprised = better predictions.

**What each panel shows.**

- **Heatmap ("What follows what").** It shows each row's probabilities. Dark navy is near 0%, teal is in the middle, and amber is the most likely next letter. The colour scale is fixed across all steps, so step 0 looks uniformly dark and the grid "lights up" as it learns.
  - **Top row (`.`)** shows how names *start*. The top picks are a 18%, s 16%, r 9%, then m and k.
  - **Left column (`.`)** shows how names *end*. Names often end after l, i, h, a.
  - **Brightest cell: `c` → `h` at 98%.** Almost every c in the data is followed by h, as in sachin, chinmay, chetan, chaitra. It's a good cell to point at.
  - **Rows f, q, x, z stay perfectly flat at 3.7% each.** Those letters never appear in any name, so those rows are never used and never get a single nudge. They stay exactly at their starting zeros, which means "no idea".
  - **Columns f, q, x, z go dark** in every *other* row. They are never the right answer, so every nudge pushes them down. The model learned "these letters never come next" but has no idea what follows them.
- **"Names it invents right now."** It starts at `.`, draws a letter from the probabilities, and repeats until it draws `.` again. Names are capped at 12 letters plus "…". The random seed is fixed, so the same dice rolls are used at every step. Any change you see is learning, not luck.
- **"How surprised is it by this name?"** It shows one bar per transition in `.name.`, including the final "→." (the name ends here). Bar height = surprise. The % under each bar is the probability it gave that letter.
- **Loss curve.** The amber line is the loss over training. The teal dotted line is the current step. The two dashed lines are the "knows nothing" ceiling and the floor.
  - **Ceiling: 3.30 = ln 27.** That's guessing uniformly among 27 letters, and it's exactly where step 0 sits.
  - **Floor: 2.05.** This is the best any model that only sees one previous letter can ever do on this data. It's computed exactly from counting the pairs. Training ends at 2.06, essentially on the floor. **It cannot go lower.** The model isn't broken; one letter of context just isn't enough. That's the bridge to LLMs, which have more context.

**What Play does.** It starts at step 0 and walks through the 16 saved snapshots: steps 0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 400. One pass takes about 6 seconds and stops at 400. *Everything* moves together: the heatmap, the names, the surprise bars, and the dotted line on the loss curve. The early snapshots are close together because that's where all the change happens. Pressing Play at the end restarts from 0.

**The story in numbers (what the audience will see).**

| step | loss | sample names on screen |
|---|---|---|
| 0 | 3.30 | tkwrbztucljy…, qvkfnavqtizx…, ueladrtzhjle… |
| 5 | 2.53 | shsjayptalhw…, shba, rilshfoh, shltila |
| 20 | 2.23 | shni, pashi, wwtama, arkkpera |
| 400 | 2.06 | shni, vit, mavitaan, shi, vrsha, rershana |

- **Steps 0–5:** the loss falls fastest here. It learns letter frequency first: common letters (a, s, r, h, n) get brighter everywhere.
- **Steps 5–30:** pairs appear: `sh`, `an`, `ch`, names ending in a / i. Names get short and pronounceable.
- **After ~75:** the names barely change. It has hit the one-letter floor. More training doesn't help, just as on page 4 a too-small model plateaued.
- **Be honest about the result.** "vrsha" and "mavitaan" are *name-shaped*, not real names. That's the point. Each letter only looks at the one before it.

**Names to type, with what you'll get (final step).**

| name | average surprise | why |
|---|---|---|
| anita | 1.58 | every pair is common, so short bars |
| priya | 1.73 | `y` → `a` is 72%, a very short bar |
| aarav (default) | 2.15 | `a` → `a` is only 4%, one tall bar |
| rahul | 2.66 | `a` → `h` → `u` is rare in this data |
| farhan | 3.59 | `f` never seen: a huge first bar, then a flat 3.7% guess after it |
| zoya | 3.77 | same story with `z` |

- If the audience member's name has **f, q, x or z**, celebrate it. It's the best demo: a giant bar exactly where the unseen letter is.
- At **step 0**, *every* name scores 3.30, because every letter is a 3.7% guess. If someone types a name before you press Play, that's a nice "it knows nothing yet" moment.
- Tall bar ≠ "wrong name". It means "this pair was rare in what I read". That's the setup for slide 20's "trained to be unsurprising, not correct."

**Gotchas.**

- Pre-flight: press **Reset** on this page so it opens at step 0 with gibberish. If it opens on real-looking names, you're at the end. Press **Reset**.
- Don't drag the slider *while* it's playing. Let the pass finish, then rewind.
- The name box is case-insensitive and silently drops anything outside a–z, such as spaces, hyphens and accented letters. "Aarav" and "aarav" score the same.
- Samples are fixed by the seed, so they're identical in rehearsal and on stage. You can memorise them.

**Likely questions on this page.**

- **"Why does it stop at 2.05 and not 0?"** With one letter of memory, after `a` the next letter genuinely could be n, r, v, `.`… It can't know which. Only more context lowers the floor. That's what LLMs buy with thousands of tokens.
- **"Why are f/q/x/z rows flat if it's trained?"** Those letters never appear in the data, so their rows never get a gradient. No data means no nudges, so the model stays at "no idea".
- **"Why learning rate 30? Page 2 used tiny steps."** Here the gradients are tiny, because each knob affects only a few pairs out of 1,162. A big step size compensates. It's the same rule (knob −= lr × slope).
- **"Is this how GPT works?"** It's the same training loop and the same loss (cross-entropy, i.e. surprise). The difference is that GPT predicts tokens from thousands of tokens of context through a transformer, with ~10¹¹ knobs instead of 729.

### On stage

`[APP: Page 5 — Next letter]`

A real language model — tiny. It learns Indian first names, letter by letter, looking only at the previous letter. An LLM looks at thousands of tokens, but the learning is identical.

*(point at the heatmap)*

This grid is the whole model: 27 by 27, so 729 knobs, all starting at zero. The row is the letter it just saw. The column is what it guesses comes next. Right now it's completely flat: every letter equally likely, a 1-in-27 guess.

*(point at the names)*

Here are names it invents right now. *(read two: "tkwrbztucljy", "qvkfnavqtizx")* Nonsense. And the loss down here: 3.30 — the "knows nothing" line.

The score is new: **surprise**. When it guesses the next letter, we check how much probability it gave the real one. Gave it 100%: zero surprise. Gave it a 1-in-27 guess: 3.30. Loss is just the average surprise. Same loop as before — backprop, nudge, repeat.

*(press **▶ Play**)*

Watch it learn. First it figures out which letters are common at all... then which letters go together. *(names start shortening)* The names stop being noise... and become name-shaped. And the loss slides down toward the green line.

*(when it stops, point at the `c` row)*

Look at this bright cell: after `c` it's 98% sure of `h`. Sachin, chinmay, chetan — it worked that out from nudges alone. Nobody told it.

*(point at rows f, q, x, z)*

And these rows stayed flat. No name in the data has an f, q, x or z, so those knobs never got a single nudge. No data, no learning. It just stays unsure.

*(point at the green dashed line)*

And it stops here, about 2.05, the best *any* model can do when it only sees one letter back. After an "a", the next letter could be n, r, v or the end of the name — one letter of context can't tell. That's exactly why LLMs look at thousands of tokens.

*(drag the **training step** slider back to about step 5, then forward to 400)*

Rewind: gibberish. Forward: names. Every one of those 729 numbers got its value from nudges exactly like the one we did by hand on page 1.

*(click the name box, ask for a volunteer's name, type it)*

How surprised is it by your name? One bar per letter. Short bar: "I expected that." Tall bar: "whoa, didn't see that coming."

*(if the name has f, q, x or z, point at the giant bar: "Never seen that letter in its life." Otherwise try **farhan** for the same effect.)*

A tall bar doesn't mean the name is wrong. It just means it didn't see that pattern much. Hold that thought — it's the first lesson for Monday.

---

## 12 · Land it  (18:00)

`[SLIDE 19 — Same loop. Wildly different scale.]`

Our neuron: two knobs, 24 readings, milliseconds. The names model: 729 knobs, seconds. Llama 3.1: about 405 billion knobs, 15 trillion tokens, months on around 16,000 GPUs. The ideas didn't change. The engineering did. Those 405 billion numbers are the weights file from last time. Now you know where every one came from.

`[SLIDE 20 — What this means for your day job]`

Three things for Monday:

1. It was trained to be *unsurprising*, not *correct*. A confident wrong answer is just a low-surprise guess, so verify.
2. It doesn't learn from your chat. The weights are frozen; next session, same knobs.
3. It's best at what it saw most. Our internal code got zero nudges, so give it context.

`[SLIDE 21 — The mental model to walk out with]`

Knobs. One score. The gradient says which way. Backprop makes it cheap. An LLM is this loop, enormous, with surprise as the loss.

`[SLIDE 22 — Thank you / Questions]`

None of it was magic: multiplication, a scoreboard, and patience. Karpathy's "Zero to Hero" builds all of this from scratch. The app stays up for questions. Thank you.

---

## Q&A — likely questions

- **"Is that really PyTorch?"** Yes. Open `nn-learns/` and show the training loop. The demos precompute each run once and replay it, so what you see is the real training history.
- **"Doesn't it get stuck in local minima?"** In 2-D, it can. With billions of knobs, almost every flat-looking spot still has some direction going downhill, so real training rarely gets truly stuck.
- **"How does it not just memorise?"** It can — that's overfitting (page 1's "w ≈ 5" beat is a mini example). We watch loss on held-out data it never trains on.
- **"Page 4 learned fast — plain gradient descent?"** No, it uses Adam, a smarter per-knob step rule. The gradients come from the same backprop. Pages 1, 2 and 5 use plain gradient descent.
- **"Then why is ChatGPT helpful, not just autocomplete?"** Pre-training is next-token prediction. A second phase (fine-tuning, RLHF) runs the same loop with a different score: human preference.
- **"Fine-tuning / LoRA?"** Same loop, starting from trained knobs. LoRA trains a small add-on set of knobs.
- **"Where do transformers fit?"** They're the recipe: they decide what the knobs are wired to, not how they learn.
- **"What are the coloured lines / hidden neurons on Bends?"** Each hidden neuron is one hinge: `v · ReLU(w·x + b)`, flat at 0 until its bend point `x = −b/w`, then sloped. Each coloured line is one neuron's contribution (**Show pieces**). The orange curve is their sum plus a constant offset, so the network is literally the sum of its pieces. The dots mark where each neuron bends.
- **"Why change the number of hidden neurons?"** It sets capacity: how many bends the curve can have. Too few (H = 1) can't fit a wave no matter how long you train, because the loss flattens. Enough (H = 8) and gradient descent places the bends itself. Each neuron adds three knobs (w, b, v), so the knob count (3H + 1) grows with H.
- **Don't know?** "Good question — grab me after, I don't want to guess on stage."

## Timing map

| Section | Ends around |
|---|---|
| Slides 1–7 | 3:45 |
| **Page 1 · Keep score** | 7:00 |
| Slides 10–11 | 7:45 |
| **Page 2 · Downhill** | 10:00 |
| Slide 12 | 10:45 |
| **Page 3 · Backprop** + slide 14 | 12:30 |
| Slides 15–16 | 13:30 |
| **Page 4 · Bends** | 15:15 |
| Slides 17–18 | 16:00 |
| **Page 5 · Next letter** | 18:00 |
| Slides 19–22 | 20:00 |

**If running long, cut in this order:**

1. The optional "w ≈ 5" beat on page 1.
2. The slider-dragging at the start of page 1 — go straight to **Wiggle test**.
3. The **Too big** explosion on page 2 (slide 10 covers it).
4. Slide 14, reduced to one sentence at the end of page 3.
5. The rewind on page 5.

**Never cut:** page 1's **Ask PyTorch** → −800 moment, page 2's "settled near 1.8 and 32", and page 5's gibberish-to-names.

**If you have spare time:** on page 2 use **Start from here** with a far-off starting point (every start finds the same valley). On page 4 switch to **Hockey stick** with 1 neuron — one bend is enough when the data has one bend.
