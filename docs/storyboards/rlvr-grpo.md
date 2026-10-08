# RLVR and GRPO (`rlvr-grpo`)

Track: training · Section: recipe · Prereqs: rlhf-dpo
Next: `agentic-rl` (the one slug whose `prereqs` list `rlvr-grpo`, from `shared/concepts.json`)
Status: approved (expert review)
Sources: 02 §4.1, §4.3, §4.4, §4.5, §4.6, §4.7, §5, §7; 05 §1.1, §2

Running example: prompt `7 × 8 = ?`, a group of 8 sampled answers, verifier = "is the last token `56`?". `7 × 8` stands in for a hard competition problem; a real 2026 model gets it right 8 times out of 8, which is exactly the no-signal case of frame 6. Proposed for reuse by `agentic-rl` (same group table in its async/mismatch frames).

## 1. Learning objective
After this page you can explain how a model learns to reason from a checker that only says right or wrong: a group of sampled answers is scored, each answer's advantage is its reward measured against its own group (mean and spread), every token of an answer is pushed by that advantage, and the 2026 patches: clip-higher (stops entropy collapse, where good but unlikely tokens can't grow and the model stops exploring), token-level loss (long wrong answers no longer pay less per token) and dynamic sampling (no rollouts wasted on groups with no spread), plus two you'll meet again in `agentic-rl`: little or no KL penalty, and importance-sampling corrections.

## 2. Misconceptions to correct
- **Misconception:** "The verifier grades the reasoning, so the model is told which step was wrong." → **Reality:** the verifier reads only the final answer and returns one number per answer (binary 0/1 in practice). Every token in that answer gets the same advantage, including `7 + 8 = 56`, which is wrong working with a right final token. Credit assignment happens statistically, across many groups and many prompts. Longer, self-checking reasoning emerges because it raises the chance of a right final token, not because anyone showed the model a reasoning trace. Corrected by frame 2 (row 5 gets a 1) and frame 5 (the whole row is pushed). (source: 02 §4.3)
- **Misconception:** "GRPO still needs a value model like PPO; it just hides it." → **Reality:** there is no critic. The baseline is the mean reward of the group sampled for that prompt; the advantage is the reward standardized within the group. PPO keeps four models in memory (policy, reference, reward model, critic); GRPO-with-RLVR keeps the policy, a checker program, and, when a KL term is used at all, a reference copy. Corrected by frames 3 and 7. (source: 02 §4.1, §4.4)
- **Misconception:** "A right answer always gets the same positive push." → **Reality:** an answer's advantage depends on its siblings. The same `7 × 8 = 56` earns +1.73 when 2 of 8 answers are right, +1.00 when 4 of 8 are, and exactly 0 when all 8 are. A group with no spread is a wasted rollout, which is why DAPO-style dynamic sampling drops it and samples again. Corrected by frame 6 and try-this 1. (source: 02 §4.4, §7)
- **Misconception:** "Clipping makes a token's update smaller." → **Reality:** once a token's ratio has moved past the band in the direction its advantage pushes (above 1 + ε for a good answer, below 1 − ε for a bad one), the clipped term wins the `min`. That term doesn't depend on the parameters, so the token's gradient is zero. Moving the other way is never clipped. Clipping switches the update off; it does not shrink it. That is why clip-higher widens only the upper bound (rare good tokens may keep growing) and why CISPO clips the weight but keeps the gradient. Corrected by frames 8–9 and try-this 3. (source: 02 §4.4, §7)
- **Misconception:** "RL teaches the model reasoning it could never produce." → **Reality:** RL can only reinforce answers the model already sometimes samples. At k = 0 there is no signal at all. That is why labs start RL from an SFT cold start and filter prompts to ones the model solves sometimes but not always. Corrected by frame 6 (all-wrong is the same picture as all-right) and try-this 1 at k = 0. (source: first principles; 02 §4.3 difficulty filtering; 02 §5 step 3)

## 3. Hook and intuition (final wording)
**Hook:** How can a model get better at reasoning when the only feedback is "right" or "wrong" on its final answer?

Reinforcement learning with verifiable rewards (RLVR) replaces the learned reward model with a program: a math checker, a unit test, a compiler, a format check. The program reads an answer and returns a reward, usually just 1 or 0. That is cheap, precise, and much harder to game than a learned reward model, and it is available for exactly the tasks where 2026 models made the biggest gains: math, code, science, tool use.

A 0/1 reward alone is not enough, because "right" means nothing without knowing how hard the prompt was. GRPO solves this with a group. For one prompt the policy (the model being trained) samples several answers (8 on this page; GLM-5 uses 32). The checker scores each one. Each answer's advantage, how much better it did than its siblings, is its reward minus the group's mean, divided by the group's spread. The mean is the baseline: the score an answer has to beat. It is grading on a curve: a right answer in a group where everyone is right earns nothing; a right answer in a group of failures earns a lot. That baseline is what PPO had to learn with a separate critic network. The price is sampling: every prompt costs G full generations (forward passes only), so generation, not the update step (a forward and a backward pass), is where RL time goes; long rollouts leave GPUs idle unless they run asynchronously (02 §4.7), and the total is no longer small (DeepSeek-V3.2 reports its post-training compute at more than 10% of its pretraining compute, in the report's own accounting).

The advantage then flows back onto every token of its answer as a push: how strongly the update raises or lowers that token's probability. Each token in a right answer becomes more likely, each token in a wrong answer less likely, through the same kind of clipped ratio PPO uses. The trainer usually takes several update steps on one batch of answers, and with asynchronous rollouts the answers may come from a slightly older model. So by the time a token is used, its probability has already moved; the clip limits how far. Under an outcome reward, every token in an answer shares that answer's fate, including the shared prefix `7 × 8 =`. Across many groups, a token that doesn't change the odds of a right ending is pushed up about as often as down, so its net push is near zero; tokens that do change the odds keep a net push. In this group, `7 × 8 =` starts one right answer and three wrong ones: +1.732 − 3 × 0.577 ≈ 0 (exactly zero before rounding).

Because the reward only checks the final answer, any token sequence that ends in the right answer is rewarded. RL can only reinforce what the model already sometimes samples, so training starts from an SFT model that sometimes reasons its way to the answer. From there, decomposing the problem, checking, and backtracking all raise the chance of a right ending, so the policy learns to spend more tokens on them: that is why a right/wrong signal on the final answer is enough to grow reasoning. The original 2024 recipe was unstable at scale, and most 2026 open-model reports ship some mix of the same patches: a higher upper clip bound, a token-level loss, a KL term (a penalty for drifting from the starting model) that is tiny or gone, filtering of groups with no spread, and corrections for the gap between the engine that sampled the answers and the trainer that updates on them.

## 4. Visual metaphor
Glyphs used (from spec §5.1 and the built library): `token` (every answer is a row of token chips; `fill` = `valueColor(advantage)`, `hatched` = gradient switched off), `vector` (the reward column R and the advantage column A, `NUMBER_CELL` size so the numbers print), `cell` (the two probability cells in frame 8, printed numbers), `block` (policy, checker, and the PPO shelf of policy / reference / reward model / critic; `active` vs `dim`), `flow` (carry `token` from the policy into each row; carry `gradient` from the A column back into the chips), `verdict` (✓/✗ badge per row), `clipLine` (frames 8–9), `text` for plain labeled marks.

Plain labeled marks (no glyph): the baseline in frame 3 (a horizontal rule labeled `mean 0.25`), the `Σ A = 0.00` readout, the `what if all 8 were right?` branch label and the `no signal: filtered` label in frame 6, the `prevents entropy collapse` label in frame 9, the verifier footnote under the checker block.

New glyphs proposed: none (`verdict`, `clipLine` and the `token` `fill`/`hatched` options were proposed in the draft and now exist in `shared/glyphs/`).

Terms introduced on this page (one per frame, defined where first shown): group of G samples (frame 1), checker / verifiable reward (frame 2), baseline (frame 3), advantage (frame 4), push (frame 5), dynamic sampling (frame 6), none new (frame 7, the cost contrast), none new (frame 8: recap of the ratio r from `rlhf-dpo` frame 5, plus why it drifts), clip-higher ε_high (frame 9).
Terms assumed from `rlhf-dpo` (its §4 contract): policy, reward model, critic (value model), reference model, KL term (penalty), PPO's clip range ε and the ratio r = now / when sampled. Frame 1's label `policy: the model being trained` is the same recap label `rlhf-dpo` frame 3 uses, not a new definition.

Encoding (one definition per quantity): chip fill always encodes the answer's advantage A (`valueColor(A, 2.65)`); the per-token push A · w and the per-answer push A · w · L are printed numbers only (inspector and row end), never a color. Hatch has one meaning: excluded / doesn't count (a clipped token's gradient, or a filtered group).

Stage budget (≈ 580 × 366): header band 22 px (prompt chips left, `R` / `A` column headers right) + 8 rows × 43 px (`NUMBER_CELL` cells for R and A) = 366 px. The token chips, verdict badge, R and A columns use ≈ 360 px of width; the readouts `mean`, `std`, `Σ A`, `Σ|A|` stack in the right-hand margin (≈ 200 px). Frames 7–9 draw far less. The toy's table is a non-stage SVG that grows with its content.

Indexing: rows and slots are numbered 1–8 on screen; there are no 0-based addresses on this page.

## 5. Animation script
Group shown in frames 1–5 and the toy's default (k = 2 of G = 8 right; see §6 for how the group is built):

| Slot | Tokens | Verdict | R | A |
|---|---|---|---|---|
| 1 | `7` `×` `8` `=` `56` | ✓ | 1 | +1.73 |
| 2 | `7` `×` `8` `=` `54` | ✗ | 0 | −0.58 |
| 3 | `7` `+` `8` `=` `15` | ✗ | 0 | −0.58 |
| 4 | `7` `×` `8` `=` `48` `,` `so` `48` | ✗ | 0 | −0.58 |
| 5 | `7` `+` `8` `=` `56` | ✓ | 1 | +1.73 |
| 6 | `63` | ✗ | 0 | −0.58 |
| 7 | `7` `×` `8` `=` `58` | ✗ | 0 | −0.58 |
| 8 | `8` `×` `8` `=` `64` | ✗ | 0 | −0.58 |

Thread order: frames 1–5 tell the advantage story; frame 6 is a labeled what-if branch (no spread); frame 7 contrasts with PPO; frames 8–9 zoom on one token for the clip.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Prompt chips `7 × 8 = ?` at the top. Policy `block` (active) at the left, labeled `policy: the model being trained`. Eight empty rows below. | A `flow` dot (carry `token`) leaves the policy for each row in turn; the row's token chips appear left to right as the dot passes. Rows fill top to bottom. | One prompt, eight sampled answers from the same model. Only the random sampling differs. | `G = 8`; row index 1–8 |
| 2 | Same rows. Checker `block` (label `checker: last token = 56?`) at the right, drawn as a small program block, not a model, with a visible one-line footnote under it: `Real checkers parse a boxed answer and compare it symbolically; this one reads the last token.` Column header `R`. | The checker lights `active`; a `verdict` badge stamps onto each row top to bottom; the `R` `vector` fills cell by cell with the semantic fill option, `fill: 'ok'` (`--sem-ok`) for 1 and `fill: 'bad'` (`--sem-bad`) for 0, each cell still printing its number. | A program checks each final answer: 1 if it is 56, 0 if not. Row 5 has wrong working and a right final token, and still gets a 1. | `R = [1, 0, 0, 0, 1, 0, 0, 0]` |
| 3 | The `R` column with a horizontal rule across it labeled `mean 0.25` (plain labeled mark). Column header becomes `R − mean`. | The rule slides from 0 up to 0.25. Each cell's number counts from `R` to `R − mean`. Nothing else moves. | The group's own mean is the baseline, the score an answer has to beat: 2 of 8 right gives 0.25. Right answers sit 0.75 above it, wrong ones 0.25 below. | `mean = 2/8 = 0.25`; `+0.75` (rows 1, 5); `−0.25` (others) |
| 4 | The column, now headed `A (advantage)`, colored by the value scale. A `std 0.43` label beside it. | Each cell's number scales from `+0.75 / −0.25` to `+1.73 / −0.58` while its fill goes from `--val-zero` to its value color. A `Σ A = 0.00` readout stays at zero. | Divide by the group's spread, 0.43, and you have each answer's advantage: +1.73 for right, −0.58 for wrong. Within a group they always sum to zero. | `std = √(0.25 · 0.75) = 0.433`; `0.75 / 0.433 = +1.73`; `−0.25 / 0.433 = −0.58`; `Σ A = 0` |
| 5 | **Key frame.** Rows of chips at the left, the `A` column at the right. | A `flow` dot (carry `gradient`) leaves each `A` cell and travels right to left along its row; every chip it passes takes the row's value color. Right rows warm, wrong rows cool. | Every token in a right answer gets +1.73 and is pushed up; every token in a wrong one gets −0.58 and is pushed down, shared prefix included. | per-chip `A`: `+1.73` on 10 chips, `−0.58` on 29 chips |
| 6 | Branch label at the top: `what if all 8 were right?` The full group with all eight rows ending in `56` (k = 8, slots filled from the correct pool). `R`, `A` columns. | All eight `verdict` badges stamp ✓; the `R` column fills with 1s; the mean rule rises to 1.0; the `A` column fills with `--val-zero` cells reading `0.00`; every chip gets the hatch (excluded: the group leaves the batch, and its gradient is zero anyway); the label `no signal: filtered (dynamic sampling)` appears under the group. | All right, or all wrong: no spread, so every advantage is 0 and nothing moves. Dynamic sampling drops such groups and samples again until the batch has signal. | `R = [1 × 8]`; `mean = 1.0`; `std = 0`; `A = [0 × 8]`; `Σ|A| = 0` |
| 7 | Back on the main thread. Two shelves side by side. Left, `GRPO + RLVR`: policy `block` + checker `block`. Right, `PPO`: four `block`s labeled `policy`, `reference`, `reward model`, `critic`, all `active`. | On the right the `critic` block fades to `dim` and a label `replaced by the group mean` appears under it; the `reward model` block fades to `dim` with `replaced by the checker`. The left shelf does not move. | PPO keeps four models in memory, including a critic network that estimates the baseline. GRPO takes the baseline from the group, and RLVR swaps the reward model for a checker. | `PPO: 4 models in memory` → `GRPO + RLVR: 1 (+ reference only if a KL term is kept)` |
| 8 | Zoom on row 1's `56` chip (hatched). Two `cell`s at `NUMBER_CELL` size with printed numbers (`maxAbs: 1`, so both read near-neutral on the value scale), labeled `when sampled 0.20` and `now 0.25`. A `clipLine` from 0.6 to 1.6, band `[0.8, 1.2]` shaded, marker at `r = 1.25`, label `ratio r = now / when sampled` (identical settings to `rlhf-dpo` frame 5, so the learner recognizes it). | The `now` cell counts up from 0.20 to 0.25; the marker slides from 1.0 to 1.25 and crosses the band's right edge; the chip's hatch appears when it does. | The trainer updates on the same answers several times, so 56 is already 25% more likely than when it was sampled. Past 1.2 the clip switches its gradient off. | `r = 0.25 / 0.20 = 1.25`; `A = +1.73`; objective `min(1.25·1.73, 1.20·1.73) = 2.08`, clipped |
| 9 | Same zoom. Label `clip-higher: ε_high 0.20 → 0.28` and, under the band, the plain mark `prevents entropy collapse: unlikely good tokens can keep growing`. | The band's right edge slides from 1.2 to 1.28 and passes the marker; the chip's hatch lifts; the objective readout changes from `2.08 (clipped)` to `2.17`. | Clip-higher raises only the upper edge, to 1.28 (DAPO, GLM-5). A good token that started out unlikely can keep growing, so the model keeps exploring. | `ε_high 0.20 → 0.28`; objective `2.08 → 2.17`; gradient on |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Frames 1–5 draw `buildGroup(2)` + `groupAdvantages()`; frame 6 is `buildGroup(8)`; frames 8–9 use `clippedSurrogate(1.25, 1.7321, { epsHigh })`. Caption check (script-counted at review, 2026-10-07): each caption is at most two sentences and 30 words and contains no formulas. Counter: `node -e 'const md=require("fs").readFileSync("docs/storyboards/rlvr-grpo.md","utf8"); for (const r of md.split("\n").filter(l=>/^\| [1-9] \| /.test(l))) { const cap=r.split(" | ")[3].replace(/`/g,""); console.log(r.slice(2,3), cap.trim().split(/\s+/).length, cap.split(/[.?!](\s|$)/).filter(s=>s.trim()).length); }'` → frames 1–9: 14/2, 29/2, 30/2, 26/2, 27/1, 28/2, 30/2, 29/2, 25/2 (words/sentences).

## 6. Toy
"Grade a group." One prompt, eight seeded sampled answers, live rewards, advantages, and per-token pushes.

**Seeded data** (hand-authored constants, not drawn from the RNG, so every number is reproducible from this file). The pools, the slot order and the target are `GROUP_TOY = { correct: [C1…C8], wrong: [W1…W8], slotOrder: [1, 5, 3, 7, 2, 6, 4, 8], target: '56' }`, imported from `math/grpo.js` (ruling P3-R5; `agentic-rl` and `distillation` import the same constant); the per-token ratios below stay in `training/concepts/rlvr-grpo.js`. A visible line under the toy says: `The eight answers and the ratios are hand-picked stand-ins; a real policy samples them and a real trainer measures the ratios.`

Correct pool (final token `56`), in order:
`C1 = 7 × 8 = 56` · `C2 = 7 + 8 = 56` · `C3 = 56` · `C4 = 8 × 7 = 56` · `C5 = 49 + 7 = 56` · `C6 = 70 − 14 = 56` · `C7 = so 56` · `C8 = 7 eights are 56`

Wrong pool, in order:
`W1 = 7 × 8 = 54` · `W2 = 7 + 8 = 15` · `W3 = 7 × 8 = 48 , so 48` · `W4 = 63` · `W5 = 7 × 8 = 58` · `W6 = 8 × 8 = 64` · `W7 = 7 × 8 = 42` · `W8 = 7 × 8 = 65`

Tokens are whitespace-split. Slot order `[1, 5, 3, 7, 2, 6, 4, 8]`: the first `k` entries name which slots hold correct answers. Walking slots 1..8 in ascending order, a correct slot takes the next unused `C`, any other slot takes the next unused `W`. So `k = 2` gives the table in §5 (`C1` in slot 1, `C2` in slot 5), and `k = 3` puts `C2` in slot 3 and `C3` (`56`) in slot 5.

Importance ratios `r = π_θ / π_old` per token (1.00 everywhere except): `C1 = [1.02, 0.98, 1.05, 1.00, 1.25]`, `C2 = [1.00, 1.00, 1.00, 1.00, 1.35]`, `W3 = [1.00, 1.00, 1.00, 1.00, 0.75, 1.00, 1.00, 1.10]`. Stand-in for the probability pair: every token was sampled at `π_old = 0.20`, so the inspector shows `when sampled 0.200 → now 0.200 · r` to 3 decimals (r = 1.25 → 0.250, 1.35 → 0.270, 0.75 → 0.150, 1.10 → 0.220, 1.02 → 0.204, 0.98 → 0.196, 1.05 → 0.210, 1.00 → 0.200). Frame 8's `0.20 → 0.25` is the r = 1.25 case of the same rule.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `k` | Right answers out of 8 (how hard the prompt is) | Slider | 0–8, step 1 (readout `k / 8`) | 2 | chips `0`, `2`, `4`, `8` |
| `norm` | Divide by the group's std | Toggle | on (GRPO) / off (Dr.GRPO, DeepSeek-V3.2) | on | – |
| `agg` | Loss aggregation | Toggle | `sample` (mean per answer, then per group: original GRPO) / `token` (one mean over all tokens: DAPO) | `sample` | – |
| `epsHigh` | Upper clip bound ε_high (ε_low fixed at 0.2) | Preset chips | `PPO 0.20` / `clip-higher 0.28` (the 0.28 chip loads from `data/models.json` `glm-5.rl_clip_eps_high`) | 0.20 | the two chips are the control |

**Selection, not hover.** A chip is selected by click, tap, or arrow keys (visible focus ring = "the token we're following"); the default selection is row 1's `56`. A visible **token inspector** panel under the table always shows the selected token's row, `A`, weight `w`, push `A · w`, ratio (`when sampled` → `now`), objective and `clipped yes/no`. Every number a try-this prompt refers to is in the inspector or printed in the table; nothing is behind hover.

**Check my work:** none (the token inspector is this page's check: it prints every step behind the selected chip's push; ruling P3-R16).

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Verdict badge and `R` per row | `verifyFinalAnswer(tokens, '56')` | ✓/✗, `1`/`0` |
| `mean`, `std` (printed under the table) | `groupStats(rewards)` (population std, divides by G) | 3 decimals |
| `A` per row (printed, `NUMBER_CELL`) | `groupAdvantages(rewards, { normalizeStd: norm })`; all zeros when `std = 0` | signed, 2 decimals, value-scale cell |
| Signal banner (visible) | `hasSignal(rewards)`; false → banner `No spread: every A = 0. Dynamic sampling would drop this group.` | text |
| Total push `Σ|A|` (printed) | `totalPush(advantages)` | 2 decimals |
| Chip fill per token | `valueColor(A_i, 2.65)` (maxAbs = the largest advantage reachable at G = 8, which is k = 1) | color; the number is the row's printed `A` |
| `push per answer` = `A · w · L` (printed at row end) | `tokenWeights(lengths, { aggregation: agg })`, summed over the row | 4 decimals |
| Inspector: `w`, `push per token` = `A · w`, ratio, objective, clipped | `tokenWeights(...)`, `clippedSurrogate(r, A, { epsLow: 0.2, epsHigh })` | 4 decimals; `objective` 3 decimals; yes/no |
| Chip hatch (gradient off) | `clippedSurrogate(...).clipped` | hatch |

**Try this** (each leads to a named insight)
1. Predict first: if 4 of 8 answers are right instead of 2, does `7 × 8 = 56` get a bigger or smaller push? Slide `k` 2 → 4 → 8. `A` for a right answer goes +1.73 → +1.00 → 0.00; `Σ|A|` goes 6.93 → 8.00 → 0. Now slide to 1: +2.65 for the lone right answer, −0.38 for each of the seven wrong ones; and to 0: all zeros, the same banner as at 8. Then switch `norm` off and repeat: +0.75 → +0.50 → 0, and the lone right answer gets only +0.875. → **Insight: advantages are relative, not absolute.** The rarer the outcome inside its group, the bigger its push; dividing by the std amplifies that unevenly. Compare `Σ|A|` with and without the division: 5.29 vs 1.75 at k = 1 (3.02×), 6.93 vs 3.00 at k = 2 (2.31×), 8.00 vs 4.00 at k = 4 (2×). The total push still peaks at k = 4, but the division boosts near-impossible and near-solved prompts the most (the difficulty bias), which is Dr.GRPO's argument for dropping it, as DeepSeek-V3.2 does. A group with no spread teaches nothing, which is why dynamic sampling throws it away and why GLM-5 keeps only prompts its previous model solves rarely but can solve (§8).
2. With `k = 2`, select the `48` in row 4 (`7 × 8 = 48 , so 48`, 8 tokens) and then the `63` in row 6 (1 token). Under `sample` aggregation the inspector shows a `push per token` of −0.0090 for row 4 and −0.0722 for row 6: both wrong, but the long answer is punished 8× less per token, and both rows print the same `push per answer`, −0.0722. Flip `agg` to `token`: every token now pays −0.0148, so the long wrong answer's `push per answer` becomes −0.1184 and the short one's −0.0148. → **Insight: sample-level averaging shields long wrong answers (and dilutes long right ones); token-level loss charges every token the same.** That is DAPO's token-level loss, part of the 2026 consensus recipe (§8: Olmo 3; brief 02 §4.4).
3. With `k = 2` and `epsHigh = 0.20`, three chips are hatched: `56` in row 1 (r = 1.25), `56` in row 5 (r = 1.35), and the first `48` in row 4 (r = 0.75). Select row 1's `56`: objective 2.078, clipped. Press the `clip-higher 0.28` chip: row 1's `56` loses its hatch (objective 2.165, gradient on); row 5's `56` stays hatched (1.35 > 1.28); row 4's `48` stays hatched (0.75 < 0.8, the lower bound did not move). → **Insight: clipping switches a token's gradient off, it does not shrink it, and clip-higher widens only the upward side** so a good token that was improbable when sampled can keep gaining probability (no entropy collapse), while the lower bound still stops collapse the other way.

## 7. Show me the math
```tex
\htmlClass{hl-r}{R_i} = \text{checker}(o_i) \in \{0, 1\}, \quad i = 1..G
```
```tex
\htmlClass{hl-a}{A_i} = \frac{\htmlClass{hl-r}{R_i} - \operatorname{mean}(R)}{\operatorname{std}(R)}
\qquad A_i = 0 \text{ when } \operatorname{std}(R) = 0
\qquad
\text{(Dr.GRPO / DeepSeek-V3.2: } A_i = R_i - \operatorname{mean}(R)\text{)}
```
```tex
\htmlClass{hl-ratio}{r_{i,t}} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}
```
```tex
\mathcal{J}_{\text{sample}} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\!\Big( \htmlClass{hl-ratio}{r_{i,t}}\,\htmlClass{hl-a}{A_i},\;
\operatorname{clip}(\htmlClass{hl-ratio}{r_{i,t}},\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}})\,\htmlClass{hl-a}{A_i} \Big)
- \beta\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
```
```tex
\mathcal{J}_{\text{token}} = \frac{1}{\sum_i |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min(\cdots)
\qquad \text{(DAPO token-level; 2026 consensus: } \beta \approx 0,\; \varepsilon_{\text{high}} > \varepsilon_{\text{low}}\text{)}
```
Shapes: `R, A ∈ ℝ^G` (G = 8 on this page); `r_{i,t}` is one scalar per token, `Σ|o_i| = 39` tokens in the default group; the policy's logits are `[|o_i| × vocab]` per answer, never shown. Population std (divides by G): `std = √(p(1−p))` for 0/1 rewards with pass rate `p`, so `Σ|A_i| = 2G·√(p(1−p))`, which peaks at `p = 0.5` and is 0 at `p ∈ {0, 1}`.
Notes shown in the panel: (a) Trainers such as TRL and verl divide by G − 1, which makes the std 1.07× larger and every advantage 1.07× smaller (+1.62 instead of +1.73), and add a small ε instead of returning zeros. (b) Dr.GRPO also drops the 1/|oᵢ| length term (compare the `agg` toggle); the `norm` toggle shows only the std change. (c) `π_old` is the policy that produced the sample; it differs from `π_θ` because the trainer takes several steps per batch and rollouts may be asynchronous (frame 8).
Color links to the animation: `hl-r` = the R column (frames 2–3, `--sem-ok`/`--sem-bad`); `hl-a` = the A column and the chip fills (frames 4–5, value scale); `hl-ratio` = the marker on the `clipLine` (frames 8–9).

## 8. In today's models (Oct 2026)
All keys below are proposed (the task-4 key list has no RL keys; accepted in §13, to be added in the data-extension pass). Entry ids name the exact report the fact comes from: `glm-5`, `deepseek-v3.2`, `deepseek-v4-pro`, `nemotron-3-super`, `kimi-k3`, `gpt-oss-120b` are single checkpoints; `minimax-m2`, `olmo-3`, `qwen3-2507` and `magistral` are series-level reports, and their entries carry a `note` naming the report (the brief gives no per-checkpoint RL recipe).

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GLM-5 reasoning RL: IcePop-GRPO, ε_low 0.2 / ε_high 0.28, KL term removed, group size 32, batch 32, fully on-policy | `models.json` `glm-5.rl_algorithm` = "IcePop-GRPO", `glm-5.rl_clip_eps_low` = 0.2, `glm-5.rl_clip_eps_high` = 0.28, `glm-5.rl_kl_coeff` = 0, `glm-5.rl_group_size` = 32 (confirmed) | 02 §4.4 (GLM-5 report, arXiv 2602.15763) |
| GLM-5 uses binary outcome rewards for math, science, code and tool-integrated reasoning; prompts filtered to ones the previous model solves rarely but stronger teachers can solve | `glm-5.rl_reward_type` = "binary outcome" (confirmed) | 02 §4.3 |
| DeepSeek-V3.2 defines the advantage as `R − mean(R)` with no std division | `deepseek-v3.2.rl_advantage_norm` = "none" (confirmed) | 02 §4.4 (arXiv 2512.02556) |
| DeepSeek-V3.2 post-training compute exceeded 10% of pretraining compute (the report's own accounting); 1,827 synthetic environments / 4,417 tasks | `deepseek-v3.2.post_training_compute_share` = ">10% of pretraining compute", `deepseek-v3.2.rl_environments` = 1827 (confirmed) | 02 §4.3, §4.6, §5 |
| DeepSeek-V4 specialists are still trained with SFT + GRPO ("hyper-parameters closely aligned with prior research"); only the mixed-RL stage was replaced by on-policy distillation | `deepseek-v4-pro.rl_algorithm` = "GRPO" (confirmed) | 02 §4.4, §4.8 |
| Nemotron 3 Super: asynchronous GRPO over 21 environments / 37 RL datasets | `nemotron-3-super.rl_algorithm` = "asynchronous GRPO", `nemotron-3-super.rl_environments` = 21 (confirmed) | 02 §4.3, §4.4 |
| MiniMax-M2 series: CISPO, importance weight clipped to `[0, 1+ε_high]` with stop-gradient so every token keeps a gradient | `minimax-m2.rl_algorithm` = "CISPO" (confirmed) | 02 §4.4 |
| Length and effort are controlled through the reward: Kimi K3's judge gives over-long answers an automatic loss under a verbosity budget; MiniMax-M2 adds a completion-time reward | `kimi-k3.rl_length_control` = "verbosity budget (auto-lose)", `minimax-m2.rl_length_control` = "completion-time reward" (confirmed) | 02 §4.5 |
| Olmo 3: clip-higher, token-level loss, no KL, zero-gradient (no-spread) group filtering | `olmo-3.rl_patches` = "clip-higher, token-level loss, no KL, zero-gradient filtering" (reported) | 02 §4.4 |
| Magistral (Mistral's published RL recipe): KL-free, ε_high 0.26–0.28, non-diverse group filtering; Mistral Large 4's own algorithm is not disclosed | `magistral.rl_clip_eps_high` = "0.26–0.28" (reported) | 02 §4.4, §6 |
| Qwen3 2507 models: GSPO, a sequence-level importance ratio and clip | `qwen3-2507.rl_algorithm` = "GSPO" (reported) | 02 §4.4 |
| Proprietary labs do not disclose their RL algorithm; gpt-oss says only "similar CoT RL techniques as OpenAI o3" | `gpt-oss-120b.rl_algorithm` = "undisclosed (similar to o3)" (confirmed) | 02 §4.4 |
| Why ratios drift: rollout engines and trainers compute different probabilities for the same tokens; async rollouts make data stale. Fixes: IcePop masking (GLM-5), truncated IS, Keep Routing (DeepSeek), detailed on `agentic-rl` | `glm-5.rl_mismatch_fix` = "IcePop mask, ρ ∈ [1/2, 2]" (confirmed) | 02 §4.4, §4.7 |

Mechanics (GRPO's advantage formula, PPO's clipped surrogate, DAPO's clip-higher / dynamic sampling / token-level loss, Dr.GRPO's normalization change, CISPO's weight clip) are written from first principles per spec §7 and cite arXiv 2402.03300, 2503.14476, 2503.20783, 2506.13585; they are `[BG]` in the brief and are not dated facts.

## 9. Takeaways
1. RLVR scores an answer with a program, not a learned reward model, and the score is usually just 1 or 0 on the final answer; every token in the answer shares that score, and reasoning grows because it raises the odds of a right ending in a model that already sometimes gets there.
2. GRPO's advantage is the reward standardized within its own group of samples: relative, zero-mean, and zero when all answers agree. That group mean is the baseline PPO needed a critic for; the price is G full generations per prompt.
3. The 2026 recipe is GRPO plus patches: clip-higher (no entropy collapse: unlikely good tokens can keep growing), token-level loss (long wrong answers pay per token), dynamic sampling (no rollouts wasted on groups with no spread), and two covered in `agentic-rl`: little or no KL penalty, and importance-sampling corrections for the sampler-vs-trainer gap.

## 10. Next and go deeper
Next: `agentic-rl` (multi-turn rollouts, async off-policy data, the training/inference mismatch and its IS fixes). After that, `distillation` shows how the RL specialists are merged. Back: `rlhf-dpo` for the reward model, PPO and the KL term this page drops.
Go deeper: UNIPO, interactive GRPO-family explorer with per-token coloring (https://poloclub.github.io/unipo/) · Sebastian Raschka, *The State of Reinforcement Learning for LLM Reasoning* (https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training) · Jay Alammar, *The Illustrated DeepSeek-R1* (https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)

## 11. Key-frame sketch
Frame 5 (advantages flow back onto the tokens), desktop width. `[+]` chips are value-positive, `[-]` value-negative, `*` is the gradient dot. Checked against the reproducer below (R, A, Σ A, Σ|A|, 10 right / 29 wrong chips).

```text
 prompt: [7][×][8][=][?]                      R    A
 ┌──────────────────────────────────────────┬────┬──────┐
 │ 1 [7+][×+][8+][=+][56+]        ✓         │ 1  │+1.73 │
 │ 2 [7-][×-][8-][=-][54-]        ✗         │ 0  │−0.58 │
 │ 3 [7-][+-][8-][=-][15-]        ✗         │ 0  │−0.58 │
 │ 4 [7-][×-][8-][=-][48-][,-][so-][48-] ✗  │ 0  │−0.58 │
 │ 5 [7+][++][8+][=+][56+]   <*** ✓         │ 1  │+1.73 │
 │ 6 [63-]                        ✗         │ 0  │−0.58 │
 │ 7 [7-][×-][8-][=-][58-]        ✗         │ 0  │−0.58 │
 │ 8 [8-][×-][8-][=-][64-]        ✗         │ 0  │−0.58 │
 └──────────────────────────────────────────┴────┴──────┘
   mean 0.25   std 0.433   Σ A = 0.00   Σ|A| = 6.93
 Every token in a right answer gets +1.73 and is pushed
 up; every token in a wrong one gets −0.58 and is pushed
 down, shared prefix included.
 [Back] [Pause] [Next]  ━━━━━●━━━━  5 / 9   1×
```

**Reproducer** (every number in §5, §6 and this sketch; run 2026-10-07 against the draft module whose behavior is specified below, to be re-run against `math/grpo.js` once built):
```sh
node -e '
import("./math/grpo.js").then((m) => {
  const C = [["7","×","8","=","56"],["7","+","8","=","56"],["56"],["8","×","7","=","56"],["49","+","7","=","56"],["70","−","14","=","56"],["so","56"],["7","eights","are","56"]];
  const W = [["7","×","8","=","54"],["7","+","8","=","15"],["7","×","8","=","48",",","so","48"],["63"],["7","×","8","=","58"],["8","×","8","=","64"],["7","×","8","=","42"],["7","×","8","=","65"]];
  const slotOrder = [1,5,3,7,2,6,4,8], f = (x) => Number(x.toFixed(4));
  for (const k of [0,1,2,4,8]) { const g = m.buildGroup(k,{correct:C,wrong:W,slotOrder}); const R = g.map((t)=>m.verifyFinalAnswer(t,"56"));
    const A = m.groupAdvantages(R), D = m.groupAdvantages(R,{normalizeStd:false}), s = m.groupStats(R);
    console.log(k, R.join(""), f(s.mean), f(s.std), m.hasSignal(R), A.map(f), f(m.totalPush(A)), D.map(f)); }
  const A = m.groupAdvantages([1,0,0,0,1,0,0,0]), L = [5,5,5,8,5,1,5,5];
  const ws = m.tokenWeights(L), wt = m.tokenWeights(L,{aggregation:"token"});
  console.log("row4", f(A[3]*ws[3][0]), f(A[3]*wt[3][0]), f(A[3]*ws[3][0]*8), f(A[3]*wt[3][0]*8), "row6", f(A[5]*ws[5][0]), f(A[5]*wt[5][0]));
  console.log("prefix net", A[0]+A[1]+A[3]+A[6], "chips", L.filter((_,i)=>A[i]>0).reduce((a,b)=>a+b,0), L.reduce((a,b)=>a+b,0));
  for (const eh of [0.2,0.28]) console.log(eh, m.clippedSurrogate(1.25,A[0],{epsHigh:eh}), m.clippedSurrogate(1.35,A[4],{epsHigh:eh}), m.clippedSurrogate(0.75,A[3],{epsHigh:eh}), m.clippedSurrogate(1.5,A[3],{epsHigh:eh}));
  console.log("unbiased", f(Math.sqrt(8/7)), f(0.75/Math.sqrt(0.1875*8/7)));
});'
```
Output (2026-10-07): k = 0 → std 0, no signal, all 0; k = 1 → +2.6458 / −0.3780, Σ|A| 5.2915, Dr.GRPO +0.875 / −0.125; k = 2 → +1.7321 / −0.5774, Σ|A| 6.9282, Dr.GRPO +0.75 / −0.25; k = 4 → ±1, Σ|A| 8; k = 8 → all 0, no signal. Row 4: −0.0090 / −0.0148 per token, −0.0722 / −0.1184 per answer; row 6: −0.0722 / −0.0148. Prefix net −2.2e−16 (zero before rounding); 10 of 39 chips positive. Clip: (1.25, +1.7321) → 2.0785 clipped at 0.20, 2.1651 not clipped at 0.28; (1.35) → clipped at both; (0.75, −0.5774) → −0.4619 clipped at both; (1.50, −0.5774) → −0.8661 not clipped. Unbiased-std factor 1.069, +1.6202.

Catch-up addition (2026-10-07), the std-amplification numbers in try-this 1: `node -e 'import("./math/grpo.js").then((m) => { for (const R of [[1,0,0,0,0,0,0,0],[1,0,0,0,1,0,0,0],[1,0,1,0,1,0,1,0]]) { const a = m.totalPush(m.groupAdvantages(R)), d = m.totalPush(m.groupAdvantages(R, { normalizeStd: false })); console.log(R.filter(Boolean).length, a.toFixed(4), d.toFixed(4), (a / d).toFixed(4)); } })'` → `1 5.2915 1.7500 3.0237` · `2 6.9282 3.0000 2.3094` · `4 8.0000 4.0000 2.0000` (the ratio equals 1/std).

**`math/grpo.js` signatures** (pure, no DOM; tests first). All worked examples are quoted to 4 decimals from the reproducer.

```js
// 0/1 outcome reward: the last token is the final answer.
verifyFinalAnswer(tokens: string[], target: string) → 0 | 1
//   (['7','×','8','=','56'], '56') → 1
//   (['7','+','8','=','56'], '56') → 1      // wrong working, right final token
//   (['7','×','8','=','48',',','so','48'], '56') → 0

// Group mean and population std (divides by G). Throws on an empty group.
groupStats(rewards: number[]) → { mean: number, std: number }
//   [1,0,0,0,1,0,0,0] → { mean: 0.25, std: 0.4330 }
//   [1,1,0]           → { mean: 0.6667, std: 0.4714 }
//   [1,1,1,1]         → { mean: 1, std: 0 }

// Group-relative advantages; every entry is 0 when std is 0 (no signal).
groupAdvantages(rewards: number[], { normalizeStd = true } = {}) → number[]
//   [1,0,0,0,1,0,0,0] → [1.7321, −0.5774, −0.5774, −0.5774, 1.7321, −0.5774, −0.5774, −0.5774]  (sums to 0)
//   [1,0,0,1]         → [1, −1, −1, 1]
//   [1,1,0], { normalizeStd: false } → [0.3333, 0.3333, −0.6667]
//   [0,0,0]           → [0, 0, 0]

hasSignal(rewards: number[]) → boolean
//   [1,0,0,0,1,0,0,0] → true · [1,1,1,1,1,1,1,1] → false · [0,0] → false

// Σ|A_i|: how much total push a group delivers.
totalPush(advantages: number[]) → number
//   groupAdvantages([1,0,0,0,1,0,0,0]) → 6.9282 · [1,−1,1,−1] → 4

// Per-token loss weights. 'sample': 1/(G·|o_i|). 'token': 1/Σ|o_i|.
tokenWeights(lengths: number[], { aggregation = 'sample' } = {}) → number[][]
//   [5, 8, 1]                        → [[0.0667 ×5], [0.0417 ×8], [0.3333]]
//   [5, 8, 1], { aggregation: 'token' } → [[0.0714 ×5], [0.0714 ×8], [0.0714]]   (1/14)
//   default group lengths [5,5,5,8,5,1,5,5] → sample: 0.025 / 0.0156 / 0.125 per token; token: 0.0256 (1/39)

// PPO/GRPO clipped surrogate for one token. `clipped` = true means the gradient is zero:
// A > 0 and r > 1+ε_high, or A < 0 and r < 1−ε_low. Moving the other way is never clipped.
clippedSurrogate(ratio: number, advantage: number, { epsLow = 0.2, epsHigh = 0.2 } = {}) → { objective: number, clipped: boolean }
//   (1.25, 1.7321)                    → { objective: 2.0785, clipped: true }
//   (1.25, 1.7321, { epsHigh: 0.28 }) → { objective: 2.1651, clipped: false }
//   (0.75, −0.5774)                   → { objective: −0.4619, clipped: true }
//   (1.50, −0.5774)                   → { objective: −0.8661, clipped: false }   // pushing down past the band is allowed

// Deterministic group assembly from the pools (pools: `GROUP_TOY`, exported by this module, ruling P3-R5).
buildGroup(k: number, { correct: string[][], wrong: string[][], slotOrder: number[] }) → string[][]
//   k = 2 → [C1, W1, W2, W3, C2, W4, W5, W6]  (the §5 table)
//   k = 3 → [C1, W1, C2, W2, C3, W3, W4, W5]
//   k = 8 → [C1, C2, C3, C4, C5, C6, C7, C8]  (every slot is correct; pools are consumed in ascending slot order)
//   throws on k ∉ {0..slotOrder.length} or non-integer k
```

Test cases the builder should include: advantages sum to zero for every k in 0..8 (spec §8); `std = 0` returns all zeros and `hasSignal` false; `tokenWeights` rows sum to 1/G under 'sample' and the whole table sums to 1 under both modes; `clippedSurrogate` gradient-off conditions (`A > 0 && r > 1+ε_high`, `A < 0 && r < 1−ε_low`) and nothing else; `buildGroup` is a pure function (same input, same output) and never mutates the pools.

## 12. Open questions for the reviewer
Resolved by the main-session and expert reviews: see §13. Still open:
- **Data entries.** The proposed RL keys and the entries `glm-5`, `deepseek-v3.2`, `nemotron-3-super`, `minimax-m2`, `olmo-3`, `magistral`, `qwen3-2507` (plus `kimi-k3.rl_length_control`, `minimax-m2.rl_length_control` added in this revision) are accepted but not yet in `data/models.json`; the `clip-higher 0.28` chip and the §8 table depend on the data-extension pass after Task 12.
- **Running example handoffs.** `agentic-rl` reusing this group table, and `rlhf-dpo` reusing `clipLine` and the four-model shelf, are confirmed for Task 12 (§13); the Task 12 agents should be told explicitly.

## 13. Reviewer rulings (main session)
Worked numbers hand-checked: k = 1/2/4 advantages, Σ|A|, per-token pushes under both aggregations, all four `clippedSurrogate` cases, 39 tokens (10 right / 29 wrong).
- Std: keep the population std; the math panel adds one line noting that trainers often use the unbiased std plus an ε. **Expert review 2026-10-07:** wording fixed to "divide by G − 1, which makes the std 1.07× larger and every advantage 1.07× smaller (+1.62 instead of +1.73)".
- Data: accept the proposed RL keys and entries (`glm-5`, `deepseek-v3.2`, `nemotron-3-super`, `minimax-m2`, `olmo-3`, `magistral`, `qwen3-2507`). They are added in the data-extension pass after Task 12. Magistral's ε_high is stored as the string "0.26–0.28" (not a conflict range).
- BG mechanics are treated as timeless (spec §7); only adoption facts are dated.
- Kimi K3 effort control (the untagged −1 override) stays off the page until tagged. **Expert review 2026-10-07:** a generic length/effort-control row was added to §8 using only the CONFIRMED §4.5 facts (K3 verbosity budget, MiniMax-M2 completion-time reward).
- Verifier: the simplification note is a one-line footnote under the checker block (frame 2), visible on touch and keyboard (expert review 2026-10-07). No fifth control.
- Hand-authored ratios are fine (toy constants, documented). **Expert review 2026-10-07:** their origin (several trainer steps per batch, async rollouts) is now explained where the ratio first appears (frame 8, §3 ¶3, math note c), and a visible stand-in line sits under the toy.
- Frames: nine, no summary card; the takeaways carry the patch checklist (expert review 2026-10-07: the clip frame is split into 8, ratio and clip, and 9, clip-higher). Order: 1–5 (advantage), 6 (no-spread what-if, labeled as a branch), 7 (PPO shelf), 8–9 (clip).
- Glyphs: `verdict`, `clipLine` and the `token` `fill`/`hatched` options are accepted and now exist in `shared/glyphs/`. **Expert review 2026-10-07:** the frame-8 probability "bars" are replaced by two `cell`s with printed numbers; small marks (baseline rule, branch label, filtered label, entropy-collapse label, verifier footnote) are plain labeled text marks.
- Running-example handoffs to `agentic-rl` and `rlhf-dpo` are confirmed for Task 12.
- `epsHigh` is built as two preset chips, not a slider.
- **Expert review 2026-10-07, further rulings applied:** the learning objective and takeaway 3 promise only the three patches the page demonstrates (clip-higher, token-level loss, dynamic sampling) and name the two deferred to `agentic-rl`; misconception 4's clipping wording is direction-aware; "every 2026 report" became "most 2026 open-model reports ship some mix of the same patches"; frame 5's caption is the reviewer's wording trimmed to the 30-word rule ("shared prefix included"); the credit-assignment sentence is now conditional on whether a token changes the odds, with the `7 × 8 =` check (+1.732 − 3 × 0.577 = 0); the hook is one question; terms (policy, advantage, baseline, KL term, push) are defined on first use and listed in §4; a fifth misconception ("RL teaches reasoning it could never produce") was added; the `k` slider is labeled as prompt difficulty; all try-this numbers moved from hover to a selection-driven inspector (README lesson 5); `Next:` lists only `agentic-rl` (lesson 1).
- **Catch-up pass 2026-10-07 (README lessons 16–28 + settlements):**
  - Terms (coordinator note): `policy` and the ratio `r` are defined by `rlhf-dpo` (its frames 3 and 5) and are now listed as assumed; frame 1 keeps the identical recap label `policy: the model being trained`; frame 1 introduces "group of G samples", frame 8 introduces nothing new (recap of r plus why it drifts). `KL term (penalty)` wording kept.
  - Lesson 16: one definition per quantity. Chip fill encodes the advantage A only; `push per token` = A · w and `push per answer` = A · w · L are printed numbers, never colors (§4 encoding line; §6 output names).
  - Lesson 17: try-this 1's "difficulty bias" insight now states the toy's own numbers: Σ|A| with vs without the std division is 5.29 vs 1.75 at k = 1 (3.02×), 6.93 vs 3.00 at k = 2 (2.31×), 8.00 vs 4.00 at k = 4 (2×); the total still peaks at k = 4, so the bias is in the amplification, not the total. Reproducer call recorded in §11.
  - Lesson 18: stage budget stated in §4: 22 px header + 8 × 43 px = 366 px; readouts in the right margin; the toy table is a non-stage SVG.
  - Lessons 19, 22, 25, 27, 28 and the byte-unit, `math/memory.js`, chip-memory, FORMATS and `--carry-activation` settlements: no change (no share bars, byte sizes, bandwidth ladders, back-solved presets or activation flows on this page; slider stops 0–8 and ε_high 0.20/0.28 are inside the model).
  - Lesson 20: the two struck-through alternatives in this section were deleted and restated as settled rulings.
  - Lesson 21: the two push quantities are named distinctly (`push per token`, `push per answer`).
  - Lessons 23 and 26: the DeepSeek-V3.2 compute anchor says "compute, in the report's own accounting"; §3 ¶2 says which pass (generation = forward only; the update = forward + backward).
  - Lesson 24 / hatch settlement: hatch = excluded everywhere on the page; frame 6 now hatches the filtered group's chips (excluded from the batch); zero advantages keep the neutral `--val-zero` fill.
  - Data-id settlement: §8 says which ids are single checkpoints and which are series-level reports carrying a `note`.

## 14. Plan 3 rulings applied (S3, 2026-10-08)
- P3-R19: the header carries `Next: agentic-rl` (from `shared/concepts.json`).
- P3-R5: the pools, slot order and target are `GROUP_TOY` in `math/grpo.js` (§6 seeded data, `buildGroup`
  comment); the ratios stay in the concept module.
- X-3: "(2.00×)" → "(2×)" (try-this 1, §13); "1.069×" → "1.07×" (math note a, §12). Every ratio prints
  through `formatRatio` (3.02×, 2.31×, 2×; README lesson 35).
- S3 glyph option: the R column uses the `vector` semantic fill (`fill: 'ok' | 'bad'`), frame 2 and §7.
- P3-R16: no "Check my work" (the token inspector is the check).
