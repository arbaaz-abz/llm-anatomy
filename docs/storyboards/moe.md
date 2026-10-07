# Mixture of Experts (`moe`)

Track: architecture · Section: architecture · Prereqs: decoder-recap
Next: `model-card`, `parallelism` (the two slugs whose `prereqs` list `moe` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1 ("MLP" row), §5 (core ideas, table, active ratios), §7 · 04 §1.1 (per-expert batch), §1.4 (MoE at small vs large batch) · 05 §1.1 (MoE: "Both"), §1.2. Nothing beyond the briefs.

Opens `decoder-anatomy` frame 6's branch and keeps its toy exactly: d_model 8, 8 experts of hidden 8 (192
parameters each, so two active experts equal the dense MLP of hidden 16), top-2, router 8 × 8 per block,
"sat" routed to E3 and E6. All parameter counts come from `math/params.js` (`paramBreakdown`, `mlpParams`),
not redefined here; the one definition of *active* is `decoder-anatomy`'s, quoted verbatim in §6. New
functions (routing and balancing) go in `math/moe.js`.

## 1. Learning objective
After this page you can follow one token through a Mixture-of-Experts layer: router scores, top-k choice,
gate weights, weighted sum (frames 1–3); say why total parameters grow with the number of experts while
active parameters barely move (frames 5–7, toy A); explain fine-grained and shared experts (frames 6–7);
and explain why experts drift out of balance and how a selection-only bias evens them out without an extra
loss (frames 8–10, toy B).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "A Mixture of Experts makes the model cheaper to store." → **Reality:** every
  expert must sit in memory; only the work per token shrinks. The toy's MoE holds 4,008 parameters against
  the dense model's 1,576, while active goes 1,448 → 1,576; DeepSeek-V4-Pro keeps 1.6T parameters to use
  49B per token. (01 §5: "MoE decouples total params (knowledge) from active params (cost)") · corrected by
  frame 5, try-this 1
- **Misconception:** "Each expert is a specialist in a topic, like 'math' or 'French'." → **Reality:** the
  router is a learned matrix that scores each token's vector; nothing assigns topics, and in the toy the four
  words of one sentence go to seven different experts. The page shows routing, not meaning, and makes no
  claim about what experts learn. (first principles; 01 §5 core ideas) · corrected by frame 4
- **Misconception:** "More experts per token is always better, so top-k should be large." → **Reality:**
  active parameters (and compute) grow with k. Fine-grained designs instead cut each expert smaller and
  pick more of them, keeping active work fixed while the number of possible combinations explodes: 28
  with 8 experts top-2, 1,820 with 16 half-size experts top-4. (01 §5 "Fine-grained experts … more
  combinatorial flexibility") · corrected by frame 6, try-this 2
- **Misconception:** "Load balancing needs an extra loss term." → **Reality:** DeepSeek's aux-loss-free
  method adds a per-expert bias to the router scores *for choosing only*, nudged up for idle experts and
  down for busy ones; the gate weights still use the raw scores. In the toy the busiest expert falls from
  3.00× its fair share to 1.13× in six steps. (01 §5 "Aux-loss-free balancing") · corrected by frames
  9–10, try-this 3

## 3. Hook and intuition (final wording)
**Hook:** DeepSeek-V4-Pro has 1.6 trillion parameters but uses only 49 billion for each token. Who decides
which 3% to use, and what stops a few experts from doing all the work?

In a dense model, every token goes through the same MLP. A Mixture of Experts swaps that MLP for many small
MLPs, the experts, plus a router: a small matrix that gives each expert a score for this token. The token
goes only to the top few, and their outputs are blended by weights computed from the same scores. The rest
of the experts sit idle for this token, though another token may choose them.

That is how total and active parameters come apart. Add experts and the model can store more, because
every expert is a full set of weights in memory; the work per token stays the same as long as the number
chosen, and their size, stays the same. 2026 models push this hard: hundreds of small experts, a handful
chosen, 3–5% of the parameters active per token. Cutting experts smaller and choosing more of them keeps the
work fixed and gives the router many more combinations to pick from. One or two shared experts that every
token uses hold what all tokens need, so the routed ones can differ.

The cost is in the routing. A router that learns to favor a few experts overloads them while others sit
idle, and in serving and training the busiest expert sets the pace. DeepSeek's fix is a small per-expert
bias added to the scores only when choosing: raise it for idle experts, lower it for busy ones, a little
every step. The weights in memory, the communication between GPUs that holds different experts
(`parallelism`), and the fact that a busy server ends up touching every expert anyway are the rest of the
bill.

## 4. Visual metaphor
**Toy** (`decoder-anatomy` §4, unchanged): d_model 8, 8 experts E1–E8 of hidden 8 (SwiGLU, 192 parameters
each), top-2, router W_router [8 × 8], 2 blocks. Router scores for the four tokens are hand-picked stand-ins
on the quarter grid, chosen so that "sat" picks E3 and E6 as in `decoder-anatomy` frame 6:

| token | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | picks |
|---|---|---|---|---|---|---|---|---|---|
| The | 0.5 | 1.5 | 0 | −0.5 | 1.0 | 0 | −1 | 0.25 | E2, E5 |
| cat | −0.5 | 0 | 0.5 | 1.75 | 0 | 1.25 | 0.25 | −1 | E4, E6 |
| sat | 0 | −0.5 | 2.0 | 0.25 | −1 | 1.5 | 0.5 | 0 | E3, E6 |
| down | 1.25 | 0 | 0.5 | −0.25 | 0.75 | 0 | 1.0 | −0.5 | E1, E7 |

The balancing frames use a seeded batch: 128 tokens per step, scores `randomMatrix(128, 8, 100 + step)`
plus a fixed popularity offset [1, 0.5, 0, 0, 0, 0, 0, 0] (stand-in for a router that has drifted toward
E1 and E2). A visible line under the stage (README lesson 10): "Router scores are hand-picked for the four
words and seeded random numbers for the batch; parameter counts are exact."

**Terms introduced, one per frame** (README lesson 3): 1 router · 2 top-k · 3 gate weight · 4 load
(tokens per expert) · 5 none (active vs total, defined in `decoder-anatomy`, reused) · 6 fine-grained
experts · 7 shared expert · 8 imbalance (busiest load ÷ fair share) · 9 balancing bias · 10 none (the bias
at work). **Terms assumed from prereqs:** MLP, SwiGLU, expert, total vs active, residual stream, block
(`decoder-anatomy`, `decoder-recap`). **Named and deferred:** expert parallelism and all-to-all
(`parallelism`), wide expert parallelism in serving (`disaggregation`), MoE decode at large batch
(`prefill-decode`).

**Indexing:** experts and positions are 1-based on screen (E1 … E8, The₁ … down₄). No addresses.

**Layout** (stage ≈ 580 × 366):
- Left: the lifted row "sat" (selection outline) as a row `vector` at 18 px (its 8 numbers were read on
  `decoder-anatomy`; hover-only here).
- Center: `block` "router [8 × 8]" → a row `vector` of 8 scores at `NUMBER_CELL` (8 × 43 = 344 px).
- Right/below: 8 small `block`s E1–E8 (active fill for the chosen two, `idle` otherwise), an `adder`
  where the two weighted outputs join the stream.
- Frame 4: a `heatmap` [4 × 8] at `NUMBER_CELL` (344 × 172 px) of router scores; unchosen cells hatched
  (the library's "excluded" mark), row "sat" outlined; a `bars` load histogram (new glyph, below) under it.
- Frames 5–7: text readouts beside the expert blocks; frame 6 splits each block in two.
- Frames 8–10: `bars` with 8 bars (loads), a dashed "fair share" line, and the imbalance readout labeled
  "busiest ÷ fair share". Same explicit `max` (96) in frames 8 and 10, so the flattening is visible as motion.
- Stage budget, frame 4 (README lesson 18): the 172 px heatmap plus the `bars` must fit 366 px; mock at
  580 × 366 before build; if it overflows, the bars drop to 80 px tall with the loads printed above them.
- Builder note (README lesson 32): `activation` dots are now the Architecture accent, the same as the
  `active` expert fill. In frames 1 and 3 a dot entering E3 or E6 carries a thin `--bg` ring (halo), or stops
  at the block's edge and the block pulses instead.

Glyphs used (from spec §5.1 and the accepted `decoder-anatomy` proposals): vector, block, heatmap (hatched
cells = not chosen), flow (carry `activation`), adder, token.
New glyph proposed:
- `bars(parent, { x, y, w, h, values, labels, max, reference, format })`: vertical bars, one per label,
  each with its value printed above (length plus printed number is a required label, as for `shareBar`),
  and an optional dashed `reference` line with its own label ("fair share 32"). Fill is a neutral
  `--fg`-tint (an amount, never an accent; outlines stay selection-only). Why: `memBar` and `shareBar` are
  parts of one whole; loads and counts are separate quantities to compare against a reference, and a
  histogram is the clearest picture of "flattening". Reused by `sampling` (counts of 20 sampled tokens) and
  open to `serving-calculator` and `disaggregation`. **Accepted by the expert review with conditions:** (a)
  each bar prints its value above it; (b) neutral `--fg` tint fill, never the value scale or the accent, no
  outline except the selection outline on a followed bar; (c) dashed reference line with its own printed
  label; (d) fixed bar width and gap so 8 bars fit 344 px at `NUMBER_CELL` pitch; (e) explicit `max` so the
  y-scale holds between frames; (f) the numbered-cell fallback (an 8-cell `vector` with printed counts) is
  documented but not needed. Added to the gallery.

Color: router scores and gate weights on the value scale (maxAbs 2 for scores, 1 for weights). Chosen
experts use the Architecture accent `active` state; idle experts the `idle` state (not a selection mark,
per the `decoder-anatomy` ruling). The followed token "sat" carries the selection outline in every frame it
appears; in frames 8–10 (a batch) nothing is followed, and the caption says so.

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Row "sat" (outlined) enters a `block` "router [8 × 8]"; out comes a row of 8 scores at `NUMBER_CELL`, one per expert E1–E8 (blocks drawn idle beside). | A `flow` dot carries the row into the router; the eight score cells fill left to right. | In a Mixture-of-Experts layer, a router gives each expert a score for this token. It is one small learned matrix, 8 by 8 here. | scores for sat: [0, −0.5, 2.0, 0.25, −1, 1.5, 0.5, 0] · router 8 × 8 = 64 parameters per block |
| 2 | The two highest cells (E3 2.0, E6 1.5) stay; the other six hatch over. Blocks E3 and E6 switch to `active`. | Six cells hatch; two blocks fill. | Only the top two experts run for this token: top-k with k equal to 2. The other six are skipped, not computed. | top-2: E3 (2.0), E6 (1.5) · 6 experts skipped |
| 3 | The two scores pass through a small softmax readout: 0.622 and 0.378; each expert's output (drawn as a plain `vector` at 18 px) is scaled and both meet at an `adder` on the stream. | Two `flow` dots run through E3 and E6; their outputs shrink by their weights and merge into one row at the ⊕. | The two chosen scores become gate weights that add to 1. The layer's output is the two experts' outputs, blended by those weights. | softmax([2.0, 1.5]) = [0.622, 0.378] · output = 0.622 · E3(x) + 0.378 · E6(x) (in Numbers only) · visible line: "softmax over the chosen two, as Mixtral and Qwen3 do; DeepSeek normalizes sigmoid scores instead" |
| 4 | The 4 × 8 score `heatmap`; each row's two chosen cells stay, the rest hatched; `bars` under it with loads per expert. Row "sat" outlined. | Rows fill top to bottom; each chosen cell drops a token into its expert's bar. | Every token is routed on its own. These four words land on seven different experts, and one expert gets nothing. | The → E2, E5 · cat → E4, E6 · sat → E3, E6 · down → E1, E7 · loads [1, 1, 1, 1, 1, 2, 1, 0] · visible line: "the router scores vectors, not topics; this page makes no claim about what each expert learns" |
| 5 | Plain text readout beside the 8 expert blocks: "all 8: 8 × 192 = 1,536 · used by this token: 2 × 192 = 384 = one dense MLP"; whole-toy line under the stage. | The readout counts up; the two active blocks pulse. | Every expert is stored, but each token runs only two. Total parameters grow with the expert count; the work per token does not. | per block: 1,536 total, 384 active (dense MLP: 384) · whole toy (2 blocks): dense 1,576 total / 1,448 active · 8 experts 4,008 / 1,576 · 16 experts 7,208 / 1,704 |
| 6 | Each expert block splits into two half-size blocks (E1a, E1b, …, 16 in all); four of them light for "sat". A plain readout: "choices: 28 → 1,820". | Blocks split; the router row widens to 16 cells (18 px, hover-only); four picks light. | Fine-grained experts cut each expert in half and pick twice as many. The work per token stays the same, but the router has far more combinations to choose from. | 16 experts of hidden 4 (96 params), top-4 → 4 × 96 = 384 · C(8, 2) = 28 · C(16, 4) = 1,820 · DeepSeek-V4-Pro: C(384, 6) ≈ 4.3 × 10¹² (exact 4,281,625,192,384 on hover) |
| 7 | Back to 8 routed experts; a ninth block "shared" sits on the stream, always active; the router now picks top-1. | The shared block fills and stays filled; one routed expert lights per token. | A shared expert runs for every token and holds what all tokens need. Here one shared plus one routed expert does the same work as before. | shared 192 + top-1 routed 192 = 384 active per block · whole toy 4,392 total / 1,576 active · real models: 1 or 2 shared |
| 8 | Branch label "a batch of 128 tokens, not one word". `bars` of 8 loads, dashed "fair share 32" line; E1 at 96. Readout "busiest ÷ fair share: 3.00". | Bars rise from 0 to their loads. | Routers drift: across a batch, a few experts get most tokens. Here the busiest expert has three times its fair share, and in practice the slowest expert sets the pace. | step 0 loads [96, 51, 12, 16, 25, 9, 21, 26] · fair share 128 × 2 ÷ 8 = 32 · imbalance 3.00 · the slowest expert sets the pace on the GPU that holds it (`parallelism`) |
| 9 | Back to "sat" (outlined): a second row of 8 cells "bias" [0, 0, 0, 0, 0, −0.75, 0.5, 0] is added to the scores for choosing; the picks become E3, E7; the gate readout still uses the raw scores. | The bias row slides under the score row; the sum row appears; E6 dims, E7 lights; the gate cells retype. | DeepSeek's fix adds a per-expert bias to the scores, only for choosing. The gate weights still come from the raw scores. | biased: E3 2.0, E7 0.5 + 0.5 = 1.0, E6 1.5 − 0.75 = 0.75 · picks E3, E7 · gates softmax([2.0, 0.5]) = [0.818, 0.182] |
| 10 | The batch `bars` again, with a step counter 0 → 6; under each bar its bias. Each step nudges bias by 0.1 up for experts below fair share and down for those above. | Bars move each step; E1's bar sinks, the short ones rise; the imbalance readout falls. | After each batch, idle experts' biases go up a little and busy ones' go down. Within six steps the loads are close to even. | imbalance by step: 3.00 · 2.31 · 2.03 · 1.75 · 1.44 · 1.44 · 1.13 · step 6 loads [32, 35, 36, 33, 28, 28, 30, 34] · biases [−0.6, −0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4] · visible line under the bars: "Each step routes a fresh batch of 128 tokens, so the number wobbles; the bias keeps it near 1." |

Determinism: every frame is a pure function of (step, progress); the batch is seeded. Reduced motion shows
each frame's end state. The router row keeps its position in frames 1–3 and 9; the `bars` keep their
position in frames 4, 8 and 10.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 24 · 22 · 23 · 20 · 23 ·
29 · 26 · 30 · 21 · 24.

Absolutes checked (README lesson 7): frame 2's "not computed" is true for routed experts (the point of
top-k). Frame 5's "the work per token does not" holds at fixed k and expert size, which frame 5 shows; the
router's small growth (64 → 128 per block for 16 experts) is printed in the toy. Frame 8's "the slowest
expert sets the pace" is worded "in practice" and is developed on `parallelism` (01 §5, 04 §1.1).

Branches (README lesson 13): frame 8 leaves the four-word sentence for a batch and is labeled on screen;
frame 9 returns to "sat" with its outline; frame 10 returns to the batch, labeled.

## 6. Toy
Title on page: "Route, count, rebalance." Two panels, one state object, one `render()`.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `routed` | Routed experts | Slider (snapped) | [0 (dense), 2, 4, 8, 16, 32] | 8 | — |
| `split` | Split each expert into | segmented | 1 · 2 · 4 (hidden 8 / 4 / 2, top-k × split) | 1 | — |
| `shared` | Shared expert | toggle | off / on (one expert of the routed size; top-k halves when on, so active stays 384 per block) | off | — |
| `real` | Compare a real model | preset chips | DeepSeek-V4-Pro · Kimi K3 · Qwen3.8 · GLM-5.3 · MiniMax-M3 · gpt-oss-120b | — | published totals, actives, expert counts from `data/models.json` |
| `gamma` | Balancing step | Slider (snapped) | [0 (off), 0.05, 0.1, 0.2] | 0.1 | — |
| `step` | Batch | Slider | 0–9 | 0 | — |

Panel A uses `routed`, `split`, `shared`, `real`; panel B uses `gamma`, `step`. Panel A's routing table for
the four words appears only at 8 experts, split 1 (the hand-picked scores exist only there; a visible line
says so).

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Total, active, active share (whole toy) | `paramBreakdown({ ...PRESETS.toy, moe: { routed: routed·split, shared, topK, hidden: 8 / split, denseLayers: 0 } })` from `math/params.js` | counts; % |
| The active definition, printed under it verbatim from `decoder-anatomy` §6 (README lesson 16): "Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication." | — | text |
| Expert parameters per token per block | `(topK + shared) · mlpParams({ kind: 'swiglu', hidden }, 8)` | count (384 at every default) |
| Router parameters per block | `paramBreakdown(…).parts.router / layers` | count |
| Possible expert combinations | `expertCombinations(routed · split, topK)` | exact up to 10¹⁵, then "≈ 4.3 × 10¹²" |
| Real chip: published total, active, active share, routed used per token | `models.<id>.total_params`, `.active_params`; `experts_active / experts_total` | 3 s.f.; % |
| Panel A routing table (default only) | `routeTopK(ROUTER_TOY[i], 2)`, `gateWeights(ROUTER_TOY[i], picks)`, `expertLoads(ROUTER_TOY, 2)` | `heatmap` + `bars` |
| Panel B loads, imbalance, biases at `step` | `simulateBalancing({ tokens: 128, experts: 8, k: 2, gamma, steps: step + 1, seed: 100, popularity: [1, 0.5, 0, 0, 0, 0, 0, 0] }).at(-1)` | `bars`; imbalance 2 d.p.; biases 2 d.p. |
| Panel B wobble note (visible) | "Each step routes a fresh batch of 128 tokens, so the number wobbles; the bias keeps it near 1." | text |

**Try this** (each leads to a named insight)
1. Panel A, slide **routed** 0 → 8 → 16: total 1,576 → 4,008 → 7,208; active 1,448 → 1,576 → 1,704 (the
   only growth is the router: 0 → 64 → 128 parameters per block). Tap **DeepSeek-V4-Pro**: 1.6T total, 49B active,
   3.1%, 6 of 384 routed experts per token (1.6%). → **Insight: experts buy capacity in memory, not work per
   token.**
2. Set **split** to 2, then 4: 16 experts of hidden 4 top-4, then 32 of hidden 2 top-8. Expert parameters
   per token stay 384; combinations go 28 → 1,820 → 10,518,300; the router grows to 128 and 256 per block,
   so active ticks up to 1,704 and 1,960. → **Insight: fine-grained experts keep the work fixed and multiply
   the router's choices; the price is a bigger router and more, smaller pieces to move between GPUs**
   (`parallelism` frame 10 draws exactly that).
3. Panel B, **gamma 0**: scrub the batch slider; imbalance stays near 3 (2.50–3.00; step 9: 2.91). **Gamma
   0.1**: 3.00 → 1.13 by step 6. **Gamma 0.2**: 1.22 by step 2, then it swings back up to 1.84 at step 5. →
   **Insight: a small selection-only bias balances the experts without touching the gate weights; too big a
   nudge overshoots.**

**`math/moe.js`** (pure, no DOM, inputs never mutated; tests first; imports `softmax`, `randomMatrix` from
`./core.js`). Parameter counts are **not** here; they come from `math/params.js`.

```js
export const ROUTER_TOY = [ /* the 4 × 8 table in §4: The, cat, sat, down */ ];

// Indices (0-based) of the k highest scores after adding bias (bias only affects the choice). Ties → lower index.
routeTopK(scores, k, bias = zeros) → number[]
//   (ROUTER_TOY[2], 2)                                 → [2, 5]   (E3, E6)
//   (ROUTER_TOY[2], 2, [0, 0, 0, 0, 0, −0.75, 0.5, 0]) → [2, 6]   (E3, E7)
//   (ROUTER_TOY[0], 2) → [1, 4] · (ROUTER_TOY[1], 2) → [3, 5] · (ROUTER_TOY[3], 2) → [0, 6]

// Softmax over the chosen experts' RAW scores (never the biased ones).
gateWeights(scores, picks) → number[]
//   (ROUTER_TOY[2], [2, 5]) → [0.622, 0.378]
//   (ROUTER_TOY[2], [2, 6]) → [0.818, 0.182]
//   (ROUTER_TOY[3], [0, 6]) → [0.562, 0.438]

// Tokens per expert for a batch of score rows.
expertLoads(rows, k, bias = zeros) → number[]
//   (ROUTER_TOY, 2) → [1, 1, 1, 1, 1, 2, 1, 0]

// Busiest load ÷ fair share (tokens · k / experts). The one definition of imbalance (README lesson 16).
imbalance(loads, { tokens, k }) → number
//   ([96, 51, 12, 16, 25, 9, 21, 26], { tokens: 128, k: 2 }) → 3.00
//   ([32, 35, 36, 33, 28, 28, 30, 34], { tokens: 128, k: 2 }) → 1.13 (36 ÷ 32 = 1.125)

// Aux-loss-free update: bias_e += gamma · sign(fairShare − load_e).
balanceStep(bias, loads, { gamma, tokens, k }) → number[]
//   (zeros, [96, 51, 12, 16, 25, 9, 21, 26], { gamma: 0.1, tokens: 128, k: 2 }) → [−0.1, −0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

// Each step routes a fresh seeded batch: randomMatrix(tokens, experts, seed + step) + popularity, then updates the bias.
simulateBalancing({ tokens, experts, k, gamma, steps, seed, popularity }) → [{ step, loads, imbalance, bias }]
//   ({ tokens: 128, experts: 8, k: 2, gamma: 0.1, steps: 10, seed: 100, popularity: [1, 0.5, 0, 0, 0, 0, 0, 0] })
//     imbalance by step: 3.00, 2.31, 2.03, 1.75, 1.44, 1.44, 1.13, 1.44, 1.13, 1.44
//     step 0 loads [96, 51, 12, 16, 25, 9, 21, 26] · step 3 [56, 35, 35, 30, 26, 26, 27, 21]
//     step 6 loads [32, 35, 36, 33, 28, 28, 30, 34] · bias [−0.6, −0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
//   gamma 0:   3.00, 2.53, 2.72, 2.69, 2.88, 2.50, 2.88, 2.69, 2.59, 2.91
//   gamma 0.2: 3.00, 1.94, 1.22, 1.28, 1.31, 1.84, 1.28, 1.47, 1.44, 1.75

// Number of ways to choose k of n experts (BigInt, exact).
expertCombinations(n, k) → bigint
//   (8, 2) → 28n · (16, 4) → 1820n · (32, 8) → 10518300n · (384, 6) → 4281625192384n · (256, 8) → 409663695276000n
```

Worked examples from `math/params.js` (as specified in `decoder-anatomy` §6, toy preset plus the `moe`
field): dense → total 1,576, active 1,448 · `{ routed: 8, topK: 2, hidden: 8 }` → 4,008 / 1,576 ·
`{ routed: 16, topK: 2, hidden: 8 }` → 7,208 / 1,704 · `{ routed: 16, topK: 4, hidden: 4 }` → 4,136 / 1,704 ·
`{ routed: 32, topK: 8, hidden: 2 }` → 4,392 / 1,960 · `{ routed: 8, shared: 1, topK: 1, hidden: 8 }` →
4,392 / 1,576. `mlpParams({ kind: 'swiglu', hidden: 8 }, 8)` = 192; hidden 4 → 96.

Tests to write first: the examples above; `gateWeights` sums to 1 and ignores the bias (the frame 9 claim,
README lesson 16); `routeTopK` with a zero bias equals routing without one; `expertLoads` sums to rows · k;
two experts of hidden 8 equal one dense MLP of hidden 16 (reuse `decoder-anatomy`'s test, lesson 17); expert
parameters per token are 384 for every split at the default (lesson 17: "fine-grained keeps the work fixed");
`simulateBalancing` with `gamma: 0` never changes the bias; inputs not mutated.

**Reproducer** (run from the repo root on 2026-10-07 with these functions and `decoder-anatomy`'s parameter
formulas inlined; output matched every number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({softmax,randomMatrix})=>{
const r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):Number(x.toFixed(d));
const R=[[0.5,1.5,0,-0.5,1.0,0,-1,0.25],[-0.5,0,0.5,1.75,0,1.25,0.25,-1],[0,-0.5,2.0,0.25,-1,1.5,0.5,0],[1.25,0,0.5,-0.25,0.75,0,1.0,-0.5]];
const topk=(s,k,b=Array(s.length).fill(0))=>s.map((x,i)=>({i,v:x+b[i]})).sort((a,c)=>c.v-a.v||a.i-c.i).slice(0,k).map(o=>o.i);
R.forEach((s,t)=>{const p=topk(s,2);console.log(t,p,r(softmax(p.map(i=>s[i]))));});
const L=Array(8).fill(0);R.forEach(s=>topk(s,2).forEach(i=>L[i]++));console.log("loads",L);
const pb=topk(R[2],2,[0,0,0,0,0,-0.75,0.5,0]);console.log("biased",pb,r(softmax(pb.map(i=>R[2][i]))));
const gqa=(d,nH,nKV,dh)=>d*nH*dh+2*d*nKV*dh+nH*dh*d,mlp=(d,h)=>3*d*h;
const bd=(m)=>{const d=8,Lr=2,ex=m?mlp(d,m.h):0,p={emb:128,attn:Lr*gqa(d,2,2,4),mlp:m?0:Lr*mlp(d,16),experts:m?Lr*(m.n+m.s)*ex:0,router:m?Lr*d*m.n:0,norms:Lr*2*d+d,head:128};
 const total=Object.values(p).reduce((s,v)=>s+v,0);return [total,total-p.experts+(m?Lr*(m.k+m.s)*ex:0)-p.emb,p.router/Lr];};
for(const m of [null,{n:8,s:0,k:2,h:8},{n:16,s:0,k:2,h:8},{n:16,s:0,k:4,h:4},{n:32,s:0,k:8,h:2},{n:8,s:1,k:1,h:8}])console.log(JSON.stringify(m),bd(m));
const C=(n,k)=>{let x=1n;for(let i=0n;i<BigInt(k);i++)x=x*(BigInt(n)-i)/(i+1n);return x;};
console.log("C",[[8,2],[16,4],[32,8],[384,6],[256,8]].map(([n,k])=>C(n,k).toString()).join(" "));
const sim=(g)=>{let b=Array(8).fill(0);const out=[];for(let s=0;s<10;s++){
 const S=randomMatrix(128,8,100+s,1).map(row=>row.map((x,i)=>x+[1,0.5,0,0,0,0,0,0][i]));const Lo=Array(8).fill(0);
 S.forEach(row=>topk(row,2,b).forEach(i=>Lo[i]++));out.push({Lo,ratio:(Math.max(...Lo)/32).toFixed(2),b:b.map(x=>+x.toFixed(2))});
 b=b.map((x,i)=>x+g*Math.sign(32-Lo[i]));}return out;};
for(const g of [0,0.1,0.2]){const o=sim(g);console.log("gamma",g,o.map(x=>x.ratio).join(" "),"s0",JSON.stringify(o[0].Lo),"s3",JSON.stringify(o[3].Lo),"s6",JSON.stringify(o[6].Lo),JSON.stringify(o[6].b));}
})'
```
Output on 2026-10-07: `0 [1,4] [0.622,0.378]` · `1 [3,5] [0.622,0.378]` · `2 [2,5] [0.622,0.378]` ·
`3 [0,6] [0.562,0.438]` · `loads [1,1,1,1,1,2,1,0]` · `biased [2,6] [0.818,0.182]` ·
`null [1576,1448,0]` · `{8,top-2,h8} [4008,1576,64]` · `{16,top-2,h8} [7208,1704,128]` ·
`{16,top-4,h4} [4136,1704,128]` · `{32,top-8,h2} [4392,1960,256]` · `{8+1 shared,top-1,h8} [4392,1576,64]` ·
`C 28 1820 10518300 4281625192384 409663695276000` ·
`gamma 0 3.00 2.53 2.72 2.69 2.88 2.50 2.88 2.69 2.59 2.91 s0 [96,51,12,16,25,9,21,26] …` ·
`gamma 0.1 3.00 2.31 2.03 1.75 1.44 1.44 1.13 1.44 1.13 1.44 s0 [96,51,12,16,25,9,21,26] s3
[56,35,35,30,26,26,27,21] s6 [32,35,36,33,28,28,30,34] [-0.6,-0.1,0.4,0.4,0.4,0.4,0.4,0.4]` ·
`gamma 0.2 3.00 1.94 1.22 1.28 1.31 1.84 1.28 1.47 1.44 1.75 …`.

## 7. Show me the math
```tex
\htmlClass{hl-score}{s} = h\, W_{\text{router}} \in \mathbb{R}^{E},\qquad
\mathcal{T} = \operatorname{top\text{-}k}\big(\htmlClass{hl-score}{s} + \htmlClass{hl-bias}{b}\big),\qquad
\htmlClass{hl-gate}{g_e} = \frac{e^{s_e}}{\sum_{j \in \mathcal{T}} e^{s_j}}\ \ (e \in \mathcal{T})
```
```tex
\operatorname{MoE}(h) = \sum_{e \in \mathcal{T}} \htmlClass{hl-gate}{g_e}\, \operatorname{MLP}_e(h)
\;+\; \sum_{\text{shared}} \operatorname{MLP}_{s}(h)
\qquad \text{worked: } 0.622\, E_3(h) + 0.378\, E_6(h)
```
```tex
\text{per block: total} = (E + S)\cdot 3\,d\,h_e + d\,E,\qquad
\text{active experts} = (k + S)\cdot 3\,d\,h_e
\qquad \text{toy: } 8 \cdot 192 + 64 \ \text{vs}\ 2 \cdot 192 = 384
```
```tex
\htmlClass{hl-bias}{b_e} \leftarrow b_e + \gamma\, \operatorname{sign}\!\Big(\tfrac{T k}{E} - \text{load}_e\Big),
\qquad \text{imbalance} = \frac{\max_e \text{load}_e}{T k / E}
```
```tex
\text{combinations} = \binom{E}{k}:\quad \binom{8}{2} = 28,\ \ \binom{16}{4} = 1{,}820
```
Shapes (toy): h [d_model] (8); W_router [d_model × E] (8 × 8); each expert W_in, W_gate [8 × h_e], W_out
[h_e × 8] with h_e = 8 (192 parameters). The gate here is a softmax over the chosen experts, as in Mixtral
and Qwen3; DeepSeek-V3 uses sigmoid affinities normalized over the chosen ones, and DeepSeek-V4 a
"sqrt-softplus" affinity (01 §5). Color links: `hl-score` → the router row; `hl-bias` → the bias row
(frame 9); `hl-gate` → the gate cells (frame 3). KaTeX with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Every frontier open model on this list is a MoE with many experts and 3–5%
of its parameters active per token; most are fine-grained with one or two shared experts. Where they differ is the router and
the balancing method."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4-Pro (2026): 1.6T / 49B (3.1%), 384 routed + 1 shared, top-6, expert hidden 3,072 | `models.deepseek-v4-pro.total_params`, `.active_params`, `.experts_total`, `.experts_active` (existing, confirmed); `.expert_hidden` = 3072, `.experts_shared` = 1 (proposed, `decoder-anatomy` §8) | 01 §5 table [C] |
| Kimi K3 (2026): 2.78T / 104.2B (3.7%), 896 routed + 2 shared, top-16; experts work in a smaller "latent" space (3,584, half the model width) to cut traffic between GPUs | `models.kimi-k3.*` (existing, confirmed); `.experts_shared` = 2, `.moe_latent_dim` = 3584 (proposed) | 01 §5 table, core ideas [C] |
| Qwen3.8 (2026): 2.4T / 95B (4.0%), 512 routed + 1 shared, top-10 | `models.qwen3.8.*` (existing, confirmed) | 01 §5 table [C] |
| GLM-5.3 (2026): 753B / 40B (5.3%, the active count carried over from GLM-5), 256 routed + 1 shared, top-8 | `models.glm-5.3.total_params` (confirmed), `.active_params` (**reported**), `.experts_total` = 256, `.experts_active` = 8 (proposed) | 01 §5 table [C]/[R] |
| MiniMax-M3 (2026): about 428B / 23B (5.4%), 128 routed + 1 shared, top-4, first 3 layers dense | `models.minimax-m3.*` (existing, confirmed); `.experts_total` = 128, `.experts_active` = 4, `.experts_shared` = 1, `.dense_layers` = 3 (proposed) | 01 §5 table [C] |
| gpt-oss-120b (2025): 116.8B / 5.1B (4.4%), 128 experts top-4, no shared expert | `models.gpt-oss-120b.*` (existing, confirmed) | 01 §5 table [C] |
| Mistral Large 4 (preview Oct 2026): 1.05T / 49B routed-active (52B counting embeddings), "granular MoE", expert count unpublished | `models.mistral-large-4.total_params`, `.active_params` + its note (existing, \`reported\`: the blog says 52B active, the briefs 49B) | 01 §5 table, §7 verdicts |
| The trend: about 3–5% active in 2026, down from 9–28% in 2023–25 (Mixtral 8x7B, Dec 2023, about 28%; Qwen3-235B, 2025, 9%) | derived from the rows above (no new key); the two background ratios as text with year | 01 §5 "Active ratios" |
| Routers: softmax top-k (Mixtral, Qwen3); sigmoid plus bias (DeepSeek-V3); "sqrt-softplus" scores and a fixed hash of the token id for the first 3 MoE layers (DeepSeek-V4) | `models.deepseek-v4-pro.router` = "sqrt-softplus; hash routing in first 3 MoE layers" (proposed) | 01 §5 core ideas [C for V4] |
| Balancing: aux-loss-free bias (DeepSeek, 2024); V4 adds only a small sequence-level term; Kimi K3 uses "Quantile Balancing" instead | `models.deepseek-v4-pro.balancing` = "aux-loss-free bias + small sequence-level term", `models.kimi-k3.balancing` = "Quantile Balancing" (proposed) | 01 §5 core ideas [C] |
| The serving cost: at large batch nearly every expert is touched every step, so the whole model is read; spreading experts over many GPUs (expert parallelism) gathers enough tokens per expert | — (mechanism; numbers live on `parallelism` and `disaggregation`) | 04 §1.1, §1.4 |

Rendered with `renderFact`; GLM-5.3's active count carries its "reported" chip.

## 9. Takeaways
1. A router scores every expert for each token, the top k run, and their outputs are blended by gate weights
   from the same scores; the rest are skipped (frames 1–4).
2. Every expert is stored, but each token runs only k of them: total grows with the expert count while the
   work per token stays put. Fine-grained experts keep the work fixed and multiply the choices; shared
   experts run for every token (frames 5–7, try-this 1–2).
3. Routers drift toward a few experts and the busiest sets the pace. A per-expert bias, used only for
   choosing and nudged each step, evens the load without an extra loss; too large a nudge overshoots
   (frames 8–10, try-this 3).

## 10. Next and go deeper
Next: `model-card` (reading "384 + 1 experts, top-6" on a spec sheet) and `parallelism` (expert parallelism
and the all-to-all that moves tokens to experts); both list `moe` in `shared/concepts.json`. In-page links:
`decoder-anatomy` (where the expert branch first appeared), `disaggregation` (wide expert parallelism in
serving).

Go deeper (brief 05 §1.1–1.2; 01 §5): Maarten Grootendorst, *A Visual Guide to Mixture of Experts*
(https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) · Dai et al.,
*DeepSeekMoE* (https://arxiv.org/abs/2401.06066), fine-grained and shared experts · Wang et al.,
*Auxiliary-Loss-Free Load Balancing* (https://arxiv.org/abs/2408.15664).

## 11. Key-frame sketch
Frame 9 (the bias changes the choice, not the weights), desktop width. Numbers from the §6 reproducer
(`biased [2,6] [0.818,0.182]`).
```text
┌──────────────────────────────────────────────────────────┐
│ ►sat ▪▪▪▪▪▪▪▪ → [router 8×8]                             │
│          E1    E2    E3    E4    E5    E6    E7    E8    │
│ scores │  0  │−0.5 │ 2.0 │0.25 │ −1  │ 1.5 │ 0.5 │  0  │ │
│ bias   │  0  │  0  │  0  │  0  │  0  │−0.75│ 0.5 │  0  │ │
│ choose │  0  │−0.5 │ 2.0 │0.25 │ −1  │0.75 │ 1.0 │  0  │ │
│                     ▲                       ▲            │
│  [E1][E2][■E3][E4][E5][E6][■E7][E8]                      │
│  gates from raw scores: E3 0.818 · E7 0.182              │
├──────────────────────────────────────────────────────────┤
│ DeepSeek's fix adds a per-expert bias to the scores,     │
│ only for choosing. The gate weights still come from the  │
│ raw scores.                                              │
│ [◄] [Pause] [►]  ━━━━━━━━●━━━  9 / 10  speed [1×]        │
└──────────────────────────────────────────────────────────┘
```
"►" is the selection outline on the followed token "sat"; ■ marks an expert in the `active` state. The
three score rows are 8 cells at `NUMBER_CELL` (344 px). At 400 px the stage scrolls inside its container.

## 12. Open questions for the reviewer
**Data-pass keys** (*new* unless noted): `experts_shared` (kimi-k3 2, minimax-m3 1; deepseek-v4-pro 1 as
proposed by `decoder-anatomy`), `experts_total` / `experts_active` for glm-5.3 (256 / 8) and minimax-m3
(128 / 4), `dense_layers` minimax-m3 = 3, `moe_latent_dim` kimi-k3 = 3584, `router` and `balancing`
strings for deepseek-v4-pro and kimi-k3 (01 §5 [C]). Mixtral 8x7B and Qwen3-235B ratios are background with
no entry; shown as dated text, or add entries with their papers?

**Graph changes:** none.

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (0 Must). Status is now "approved (expert review)". No number changed; the §6
reproducer was re-run and matches.

Rulings applied (lesson 20): hand-picked router scores accepted ("sat → E3, E6" holds); fresh batch every
step kept, with the wobble note; `bars` accepted with conditions (a)–(f), no fallback needed; "the slowest
expert sets the pace" kept, with the pointer to `parallelism`.

Should (8/8, no rebuttals): §8 framing ("a MoE with many experts … most are fine-grained"); the wobble line
under frame 10's bars and in panel B; frame 8 points to `parallelism`; frame 4's "router scores vectors, not
topics" line; try-this 2 links to `parallelism` frame 10; frame 8's readout labeled "busiest ÷ fair share";
the halo builder note for frames 1 and 3 (README lesson 32); the frame 4 stage-budget mock note.
Nice (2/2): frame 3 names the softmax-over-chosen convention on stage; C(384, 6) exact on hover.
- Data pass 2026-10-07: Mistral Large 4 total and active parameters are \`reported\` (blog says 52B active, briefs 49B); \`expert_hidden\` cited; no numbers changed.
