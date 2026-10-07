# Scaling laws (`scaling-laws`)

Track: training · Section: recipe · Prereqs: pretraining
Next: `scale-reliability` (GPUs & scale; the one slug whose `prereqs` list `scaling-laws`)
Status: draft
Sources: 02 §0 items 1–2, §1.3, §1.5, §1.8, §7 (scaling-laws and Muon rows) · 05 §1.1, §1.2. Beyond the
briefs: the fitted constants of the Chinchilla scaling law as re-estimated by Besiroglu et al. (Epoch AI,
2024, arXiv 2404.10102). Re-verified by the recipe storyboard author (Opus 5.5 sub-agent) on 2026-10-07 by
reading the paper's HTML version: "L(N,D) = 1.8172 + 482.01/N^0.3478 + 2085.43/D^0.3658", and "our fitted
model implies an optimal ratio of around 20 tokens per parameter, which is consistent with both how the
Chinchilla model was trained and the findings from Approaches 1 and 2." The original 2022 paper's own
fitted constants are not used anywhere on this page: the re-estimate's authors show they imply about 70
tokens per parameter, which contradicts the 20 the original paper itself trained with.

Scope decision (recorded as a judgment call in §12 and on `midtraining`): spec §3.2 lists LR schedules
(cosine vs WSD) here. They move to `midtraining`, which needs the decay phase to explain annealing and does
not have this page as a prereq. This page keeps the compute split, over-training, and the optimizer
(AdamW → Muon).

Definition used everywhere on this page (README lesson 16): **tokens per parameter = pretraining tokens ÷
active parameters** (for a dense model active = total). A visible line under frame 8 and the toy states
it and shows the per-total figure beside it.

## 1. Learning objective
After this page you can estimate a pretraining run's compute as 6 × parameters × tokens, explain why for a
fixed budget the loss is lowest when parameters and tokens grow together (about 20 tokens per parameter,
the Chinchilla rule), explain why 2026 labs deliberately train much smaller models on far more tokens
(serving cost per token scales with parameters, so over a model's lifetime a smaller, longer-trained model
is cheaper for the same loss), read "tokens per active parameter" off a model card, and say what the Muon
optimizer does differently from AdamW.

## 2. Misconceptions to correct
- **Misconception:** "Bigger models are always the better use of compute." → **Reality:** at a fixed
  budget, a model that is too big sees too few tokens. At 10²⁴ FLOPs a 1T-parameter model reaches a fitted
  loss of 2.013, a 96B one 1.960. The 2020 Kaplan laws favored size; the 2022 Chinchilla work corrected
  this to about 20 tokens per parameter. Corrected by frames 3–4 and try-this 1. (source: 02 §1.3 `[BG]`;
  the fit above)
- **Misconception:** "Compute-optimal is the right way to train a model you will ship." → **Reality:**
  compute-optimal minimizes *training* compute only. Every served token costs about 2 FLOPs per parameter,
  so for a model that will serve 100T tokens the cheapest way to reach the same loss is a 29B model on
  14.4T tokens (495 tokens per parameter), 58.5% less total compute. Corrected by frames 6–7 and try-this 2.
  (source: 02 §1.3 "inference cost, not training cost, dominates the lifetime bill")
- **Misconception:** "2026 models are wildly over-trained, by every measure." → **Reality:** it depends on
  the denominator. Per *active* parameter DeepSeek-V4-Pro read 673 tokens; per *total* parameter, 21, right
  at Chinchilla. Active parameters set the compute per token, so they are the denominator for the cost
  argument; the page says which one it uses. Corrected by frame 8. (source: 02 §1.3 "arguably active
  params")
- **Misconception:** "The optimizer is a detail; only data and size matter." → **Reality:** the optimizer
  changes how much each token teaches. Muon, now used by DeepSeek-V4, GLM-5 and Kimi K3, rescales each
  weight update so that no single direction dominates it. Corrected by frames 9–10. (source: 02 §1.5)

## 3. Hook and intuition (final wording)
**Hook:** With a fixed compute budget, should you train a bigger model on fewer tokens or a smaller one on
more? And why do 2026 labs pick "far more tokens" anyway?

Training compute is easy to estimate: each token passes through every active parameter once forward
(about 2 FLOPs per parameter) and twice as much backward, so a run costs about **6 × parameters × tokens**.
DeepSeek-V4-Pro's 49B active parameters on 33T tokens come to about 10²⁵ FLOPs. For a fixed budget you can
spend it on a bigger model that sees fewer tokens or a smaller one that sees more.

Scaling laws answer which split gives the lowest loss. Researchers trained hundreds of small models, fitted
loss as a smooth function of parameters and tokens, and read off the minimum. The 2020 laws said "mostly
make it bigger"; the 2022 Chinchilla work found that parameters and tokens should grow together, at about
20 tokens per parameter. Too big and the model is under-fed; too small and it cannot hold what the data
offers.

But compute-optimal only minimizes the cost of *training*. A deployed model also costs about 2 FLOPs per
parameter for every token it serves, and a popular model serves far more tokens than it was trained on.
So labs **over-train**: they pick a smaller model and feed it many more tokens to reach the same loss. They
pay more for training and save much more on serving. That is why 2026 open models sit at hundreds to
thousands of tokens per active parameter.

The other lever is getting more out of each token. Muon, the optimizer behind DeepSeek-V4, GLM-5 and Kimi
K3, replaces AdamW for most weight matrices. It adjusts each update so that every direction in it moves by
about the same amount, rather than letting one dominant direction take the whole step. The cost is a few
extra matrix multiplies per step.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `block` (three model sizes in frame 2, `active` vs `dim`;
optimizer blocks in frames 9–10), `vector` (`NUMBER_CELL` cells: the two singular values in frames 9–10,
printed), `cell` (single printed readouts), plain labeled text marks (formulas in "numbers shown", the
tokens-per-parameter table in frame 8, the definition line).

Glyph reused from a parallel proposal: **`curvePlot`**, proposed on `prefill-decode` (Serving):
`curvePlot(parent, { x, y, w, h, xAxis: { label, log, ticks }, yAxis: { label, log, ticks }, series: [{
points, label, style }], markers: [{ x, y, label, followed }] })`. This page and `midtraining` need one
extension, proposed here: **`bands: [{ from, to, label }]`**, shaded x-ranges drawn under the series (the
warmup / stable / decay phases of a learning-rate schedule; a "same loss" guide line is a one-point-wide
band or a plain labeled rule). No library glyph draws a continuous function (`vector` and `heatmap` are
discrete cells, `clipLine` is one number line). Every point a learner must read is also printed as text
(the marker label or a readout), so the curve itself is never load-bearing. A marker with `followed: true`
carries the selection accent.

New glyphs proposed: none beyond the `bands` option on `curvePlot`.

Stand-ins (visible line under the stage, frames 6–7 and the toy): "The loss values come from a published
fit (Epoch AI's 2024 re-estimate of Chinchilla). Serving volumes are what-ifs, not any model's real
traffic. Muon's update in frames 9–10 is a stand-in with two directions."

Terms introduced (one per frame): FLOPs and the compute budget (1) · tokens per parameter (2) · scaling-law
fit (3) · compute-optimal (4) · none new (5) · serving (inference) compute (6) · over-training (7) · active
vs total as the denominator (none new; 8) · singular values (9) · orthogonalized update, Muon (10).
Terms assumed from prereqs: pretraining loss, tokens (`pretraining`); parameters, active vs total parameters
(`decoder-anatomy`, `moe` via `decoder-recap`).

Indexing: no positions or addresses on this page.

Selection: the followed item is the compute-optimal point (frames 3–7, the `curvePlot` marker), then the larger
singular value (frames 9–10).

## 5. Animation script
Thread order: 1–5 the compute split; 6–8 over-training; 9–10 the optimizer.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Three plain readouts in a row: `parameters 49B (active)` · `tokens 33T` · `× 6`, combining into `compute ≈ 9.7 × 10²⁴ FLOPs` (DeepSeek-V4-Pro). | The three factors slide together; the product counts up. | Training costs about six floating-point operations per parameter per token: two forward, four backward. DeepSeek-V4-Pro's run comes to about ten trillion trillion. | `6 × 49e9 × 33e12 = 9.70 × 10²⁴` · Llama 3.1 405B (2024): `6 × 405e9 × 15.6e12 = 3.79 × 10²⁵` |
| 2 | A fixed budget label `10²⁴ FLOPs`. Three `block`s of increasing width: `1B`, `96B`, `1T`, each with a token readout below. | The blocks grow in turn while their token readouts shrink. | The same budget can buy a small model reading many tokens or a big one reading few. Tokens per parameter measures the split. | `1B → 167T tokens (166,667 per param)` · `96B → 1.74T (18)` · `1T → 167B (0.17)` |
| 3 | A `curvePlot`: x = parameters (log, 10⁹–10¹²), y = fitted loss (1.94–2.20). Three points from frame 2 drop on, then the full curve draws through them. | Points fall in; the line draws left to right. | A scaling law is a smooth fit of loss against parameters and tokens, from hundreds of smaller runs. Along a fixed budget, it is a valley. | `1B: 2.187` · `96B: 1.960` · `1T: 2.013` · also `10B: 2.008`, `30B: 1.972`, `300B: 1.972` |
| 4 | Same `curvePlot`; the marker settles at the minimum with label `96B · 1.74T tokens · 18 per param`. A plain note: `2020 (Kaplan): mostly bigger · 2022 (Chinchilla): about 20 tokens per parameter`. | The marker slides along the curve to the bottom. | The bottom of the valley is the compute-optimal split: about 20 tokens per parameter, the 2022 Chinchilla rule. Earlier 2020 laws had favored bigger models. | `N* = 9.59 × 10¹⁰`, `D* = 1.74 × 10¹²`, `18.1 tokens/param`, loss `1.960` |
| 5 | A plain table of the optimum at four budgets. | Rows type in. | Bigger budgets move the optimum along both axes together: model and data grow at about the same rate, and the ratio stays near 20. | `10²² → 9.0B, 184B tokens (20.4)` · `10²⁴ → 96B, 1.74T (18.1)` · `10²⁶ → 1.0T, 16.4T (16.1)` |
| 6 | A second readout row under the curve: `serving: 2 FLOPs per parameter per token`, and a chip `served: 100T tokens` (what-if). Two stacked readouts for the compute-optimal model: `train 1.0 × 10²⁴` + `serve 1.9 × 10²⁵` = `lifetime 2.0 × 10²⁵`. | The serve readout counts up and dwarfs the train readout. | Serving costs about two operations per parameter for every token generated. For a popular model, serving can dwarf training. | compute-optimal 96B serving 100T tokens: `2 × 9.59e10 × 1e14 = 1.92 × 10²⁵` · lifetime `2.02 × 10²⁵` |
| 7 | **Key frame.** Same curve and loss line `1.960`. A second marker (selection style) at `29B` with label `14.4T tokens · 495 per param`. Readouts side by side: compute-optimal lifetime `2.02 × 10²⁵` vs over-trained lifetime `8.36 × 10²⁴`, and `58.5% less`. | The second marker slides left from 96B to 29B; the readouts count. | Over-training: a smaller model fed more tokens reaches the same loss. Training costs more, serving much less, so over a model's life it is far cheaper. | over-trained: `N = 2.92 × 10¹⁰`, `D = 1.44 × 10¹³`, train `2.53 × 10²⁴`, serve `5.84 × 10²⁴`, total `8.36 × 10²⁴` · saving `58.5%` |
| 8 | A plain table `tokens per active parameter`: `Chinchilla rule 20 · Llama 3.1 405B (2024) 39 · DeepSeek-V4-Pro 673 · Nemotron 3 Super 2,083 · DeepSeek-V4-Flash 2,462`, with a dim second column `per total parameter: 39 · 21 · 208 · 113`. Definition line visible. | Rows type in; the second column fades in last. | 2026 open models train at hundreds to thousands of tokens per active parameter, far past Chinchilla. Counted per total parameter, the gap is much smaller. | `33T / 49B = 673` · `25T / 12B = 2,083` · `32T / 13B = 2,462` · `15.6T / 405B = 39` · per total: `33T / 1.6T = 21` · `25T / 120B = 208` · `32T / 284B = 113` |
| 9 | Optimizer thread. A `block` "update for one weight matrix" with a two-cell `vector` `[3.00, 0.30]` labeled `stretch per direction (singular values)`; after normalizing, `[0.995, 0.100]`. A plain mark: `AdamW: steps mostly along the first direction`. | The two cells fill; then both scale down together to their normalized values. | A weight update stretches some directions far more than others; the stretch factors are its singular values. Here one direction is ten times stronger than the other. | `[3.0, 0.3] / √9.09 = [0.995, 0.100]` |
| 10 | Same `vector`, labeled `Muon: Newton–Schulz iterations`. A step counter `0 → 10`. | The two cells step through the iterations: the small one jumps up, both oscillate near 1, and the last two DeepSeek steps settle them at 1.00. | Muon repeats a cheap polynomial step that pulls every singular value toward one, so every direction moves equally. DeepSeek-V4's last two steps land exactly on one. | Muon coefficients (3.4445, −4.7750, 2.0315): after 1 step `[0.705, 0.338]`, 2 `[1.109, 0.989]`, 8 `[1.092, 1.103]` (between 0.70 and 1.11 throughout) · then (2, −1.5, 0.5) twice: `[1.007, 1.009]`, `[1.000, 1.000]` |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 1–8 draw `trainingFlops` (imported from `math/scale.js`), `isoFlopLoss`, `computeOptimal`, `lifetimeFlops`,
`inferenceAwareOptimum`, `tokensPerParam`; frames 9–10 draw `newtonSchulzSingular`. Caption counts (words/sentences, `rlvr-grpo`'s counter, run 2026-10-07): 22/2, 23/2, 26/2, 25/2, 24/1, 19/2,
26/2, 25/2, 27/2, 26/2.

## 6. Toy
"Spend a compute budget." A fixed budget, a model-size slider along the valley, and a serving-volume
what-if that moves the cheapest choice.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `C` | Training budget | Preset chips | `10²²` / `10²³` / `10²⁴` / `10²⁵` / `10²⁶` FLOPs | `10²⁴` | the chips |
| `logN` | Model size (active parameters) | Slider, log scale | 10⁹–10¹², step 0.05 decade; readout in B | the compute-optimal size for `C` | – |
| `Dinf` | Tokens the model will serve over its life (what-if) | Preset chips | `0` / `1T` / `10T` / `100T` / `1,000T` | `0` | – |
| reset | Reset | Button | – | – | – |

**Live outputs** (printed; the `curvePlot` marker mirrors them)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Tokens D at this size | `isoFlopLoss(C, N).D` | T tokens, 3 significant figures |
| Tokens per parameter | `tokensPerParam(D, N)` (the page's one definition) | 1 decimal |
| Fitted loss at this size | `isoFlopLoss(C, N).loss` | 3 decimals |
| Compute-optimal point | `computeOptimal(C)` → N*, D*, ratio, loss | as above |
| Cheapest model for the same loss, given `Dinf` | `inferenceAwareOptimum({ targetLoss: computeOptimal(C).loss, inferenceTokens: Dinf })` → N, D, tokens/param | as above |
| Lifetime compute: compute-optimal vs cheapest, and the saving | `lifetimeFlops(N*, D*, Dinf)`, `.total`, `1 − total / that` | scientific, 3 s.f.; percent, 1 decimal |

**Try this** (each leads to a named insight)
1. Predict first: at 10²⁴ FLOPs, is a 1T-parameter model better than a 96B one? Drag `logN` from 96B to 1T:
   loss 1.960 → 2.013, tokens 1.74T → 167B. Drag down to 1B: 2.187. → **Insight: there is a valley.** Too
   big and the model sees too few tokens; too small and it cannot use them. The bottom sits near 18 tokens
   per parameter at this budget, the Chinchilla rule of about 20.
2. Leave `logN` at the optimum and step `Dinf`: `1T` → the cheapest model is 79B at 27 tokens per parameter
   (1.6% saving); `10T` → 48B at 89 (24.0%); `100T` → 29B at 495 (58.5%); `1,000T` → 21B at 3,005 (74.1%). →
   **Insight: the more a model will be used, the smaller and longer-trained it should be.** At heavy use the
   best ratio lands in the hundreds to thousands, where 2026 models actually are (frame 8).
3. Switch `C` from `10²²` to `10²⁶` with `Dinf = 0`: the optimum goes 9.0B / 184B tokens → 1.0T / 16.4T, and
   the ratio only drifts 20.4 → 16.1. → **Insight: compute-optimal scales parameters and tokens together.**
   Ten thousand times the budget buys about a hundred times more of each.

Lesson-17 check: try-this 2's over-training shift is visible across the chip range (18 → 27 → 89 → 495 →
3,005 tokens per parameter); with `Dinf = 0` the cheapest model equals the compute-optimal one (18.1, saving
0), so the toy cannot show over-training where it should not exist.

## 7. Show me the math
```tex
C \approx 6\,\htmlClass{hl-n}{N}\,\htmlClass{hl-d}{D},\qquad
L(N, D) = E + \frac{A}{\htmlClass{hl-n}{N}^{\alpha}} + \frac{B}{\htmlClass{hl-d}{D}^{\beta}}
\quad (E = 1.8172,\ A = 482.01,\ B = 2085.43,\ \alpha = 0.3478,\ \beta = 0.3658)
```
```tex
N^*(C) = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}},\quad
G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\quad
D^* = \frac{C}{6N^*}
```
```tex
C_{\text{life}}(N) = 6\,N\,D(N) + 2\,N\,D_{\text{serve}},\qquad
D(N) = \Big(\frac{B}{L^* - E - A N^{-\alpha}}\Big)^{1/\beta}
```
```tex
\text{Muon (per weight matrix):}\quad X_0 = \frac{M}{\lVert M \rVert_F},\quad
X_{k+1} = a X_k + b\,(X_k X_k^\top) X_k + c\,(X_k X_k^\top)^2 X_k
\;\Rightarrow\; \sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5
```
Shapes: N, D, C scalars; M is one weight matrix's momentum [d_out × d_in]; the polynomial acts on each
singular value σ separately, which is why the page can show it on two numbers. Panel notes: (a) the fit is
for dense models; using active parameters for MoE is the page's stated convention, not part of the fit.
(b) 6ND ignores attention's extra cost at long context. (c) DeepSeek-V4: 10 iterations, 8 with (3.4445,
−4.7750, 2.0315) then 2 with (2, −1.5, 0.5); momentum 0.95, weight decay 0.1, update RMS rescaled to 0.18;
AdamW is kept for the embedding, the output head and norm weights. Color links: `hl-n` = model size (x axis,
`logN` slider); `hl-d` = tokens (readout).

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4-Pro: 49B active, 33T tokens (673 tokens per active parameter; 21 per total) | `models.deepseek-v4-pro.active_params`, `.total_params`, `.pretrain_tokens` (existing, confirmed) | 02 §1.3, §1.8 |
| DeepSeek-V4-Flash: 13B active / 284B total, 32T tokens (2,462 per active) | `models.deepseek-v4-flash.active_params` = 13e9, `.total_params` = 284e9, `.pretrain_tokens` = 32e12 (proposed, confirmed) | 02 §1.3, §1.8 |
| Nemotron 3 Super: 12B active / 120B total, 25T tokens (2,083 per active) | `models.nemotron-3-super.active_params` = 12e9, `.total_params` = 120e9, `.pretrain_tokens` (proposed by `pretraining`) (confirmed) | 02 §1.3, §1.8 |
| Llama 3.1 405B (2024): dense, 15.6T tokens (39 per parameter) | `models.llama-3-405b.total_params`, `.pretrain_tokens` (existing, confirmed) | data file |
| Olmo 3: ~5.9T tokens for 7B and 32B dense (185–840 per parameter) | `models.olmo-3.pretrain_tokens` = 5.9e12 (reported) | 02 §1.3, §1.8 |
| Kimi K3 ran its own scaling-law studies (batch size, LR, tokens per parameter, shape) and claims about 2.5× scaling efficiency over K2 from architecture, data and recipe together | `models.kimi-k3.scaling_efficiency_vs_k2` = 2.5 (confirmed) | 02 §1.3 |
| Muon (or a variant) in DeepSeek-V4 (with AdamW for embedding, head and norms), GLM-5 ("Muon Split"), Kimi K3 ("Per-Head Muon"); AdamW in MiMo-V2-Flash and Nemotron 3 Super | `models.deepseek-v4-pro.optimizer` (existing, confirmed); `models.glm-5.optimizer` = "Muon Split", `models.kimi-k3.optimizer` = "Per-Head Muon", `models.mimo-v2-flash.optimizer` = "AdamW", `models.nemotron-3-super.optimizer` = "AdamW" (proposed, confirmed) | 02 §0 item 2, §1.5 |
| DeepSeek-V4's Newton–Schulz schedule: 8 steps (3.4445, −4.7750, 2.0315) then 2 steps (2, −1.5, 0.5) | `models.deepseek-v4-pro.muon_ns_schedule` (proposed, confirmed) | 02 §1.5 |

Mechanics (6ND, Kaplan 2020, Chinchilla 2022 "about 20 tokens per parameter", Muon's Newton–Schulz
iteration) are `[BG]` in brief 02 and timeless per spec §7; the fit constants carry the re-verification
record in this file's header.

## 9. Takeaways
1. Training compute ≈ 6 × parameters × tokens; for a fixed budget the loss is lowest near 20 tokens per
   parameter, with model and data scaled together.
2. Serving costs about 2 FLOPs per parameter per token, so a model that will be used heavily should be
   smaller and trained on far more tokens: 2026 open models sit at hundreds to thousands of tokens per
   active parameter.
3. Muon replaces AdamW for most weight matrices at DeepSeek, GLM and Kimi: it equalizes each update's
   directions with a few cheap polynomial steps, at the cost of extra matrix multiplies per step.

## 10. Next and go deeper
Next: `scale-reliability` (what 10²⁵ FLOPs means in GPUs, days and failures).
Related in the recipe: `midtraining` (the learning-rate schedule and the end of the run).
Go deeper: Google, *How to Scale Your Model* (https://jax-ml.github.io/scaling-book/) · Hugging Face, *The
Smol Training Playbook* (https://huggingfacetb-smol-training-playbook.hf.space/)

## 11. Key-frame sketch
Frame 7 (over-training), desktop width. `●` = compute-optimal marker, `◆` = over-trained marker. Checked
against the reproducer below.

```text
 budget 10²⁴ FLOPs            fitted loss
 2.20 ┤•
      │  •                                       serving 100T tokens
 2.00 ┤     •                     •              ─────────────────────
      │        •   ◆  •   ●   •                   96B  ● lifetime 2.02e25
 1.96 ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (same loss)   29B  ◆ lifetime 8.36e24
      └──┬──────┬──────┬──────┬──                      58.5% less
        1B     10B    100B    1T   parameters      ◆ 14.4T tokens, 495/param
 Over-training: a smaller model fed more tokens reaches the same
 loss. Training costs more, serving much less, so over a model's
 life it is far cheaper.
 [Back] [Pause] [Next]  ━━━━━━●━━━  7 / 10   1×
```
(The ◆ sits on the same-loss line, not on the 10²⁴ valley: it is trained with more compute, 2.53 × 10²⁴.)

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against scratchpad drafts of
`math/scaling.js` and of `math/scale.js`'s `trainingFlops` as specified on `scale-reliability`):
```sh
node -e '
Promise.all(["./math/scaling.js","./math/scale.js"].map((p)=>import(p))).then(([m,sc])=>{
  const e=(x)=>x.toExponential(3), f=(x)=>Number(x.toFixed(4));
  console.log(e(sc.trainingFlops({params:49e9,tokens:33e12})), e(sc.trainingFlops({params:405e9,tokens:15.6e12})));
  console.log([[33e12,49e9],[32e12,13e9],[25e12,12e9],[15.6e12,405e9],[33e12,1.6e12],[25e12,120e9],[32e12,284e9]].map(([d,n])=>f(m.tokensPerParam(d,n))));
  for (const C of [1e22,1e24,1e26]) { const o=m.computeOptimal(C); console.log("opt",e(C),e(o.N),e(o.D),f(o.tokensPerParam),f(o.loss)); }
  for (const N of [1e9,1e10,3e10,9.586e10,3e11,1e12]) { const r=m.isoFlopLoss(1e24,N); console.log("iso",e(N),e(r.D),f(r.tokensPerParam),f(r.loss)); }
  const L=m.computeOptimal(1e24);
  for (const s of [0,1e12,1e13,1e14,1e15]) { const b=m.inferenceAwareOptimum({targetLoss:L.loss,inferenceTokens:s}), co=m.lifetimeFlops(L.N,L.D,s);
    console.log("serve",e(s),e(b.N),e(b.D),f(b.tokensPerParam),e(b.train),e(b.serve),e(b.total),e(co),f(1-b.total/co)); }
  const ds=[...Array(8).fill([3.4445,-4.775,2.0315]),...Array(2).fill([2,-1.5,0.5])];
  m.newtonSchulzSingular([3,0.3],ds).forEach((s,i)=>console.log("ns",i,s.map(f)));
});'
```
Output (2026-10-07): 9.702e24, 3.791e25 · tokens/param 673.47, 2461.54, 2083.33, 38.52, 20.63, 208.33,
112.68 · optimum 10²² → 9.045e9 / 1.843e11 / 20.37 / 2.1411; 10²⁴ → 9.586e10 / 1.739e12 / 18.14 / 1.9597;
10²⁶ → 1.016e12 / 1.641e13 / 16.15 / 1.8799 · iso at 10²⁴: 1B 2.1874, 10B 2.0079, 30B 1.9720, 96B 1.9597,
300B 1.9718, 1T 2.0133 (D = 1.667e14 at 1B, 1.667e11 at 1T) · serving 0 → 9.605e10, 18.07, saving 0; 1T →
7.898e10, 27.11, 1.6%; 10T → 4.842e10, 88.83, 24.0%; 100T → 2.917e10 / 1.444e13 / 494.93, train 2.528e24,
serve 5.835e24, total 8.362e24 vs 2.017e25, 58.5%; 1,000T → 2.101e10, 3005.49, 74.1% · Newton–Schulz trace,
steps 0–10 (step 0 is the normalized input): [0.995, 0.0995], [0.7047, 0.3381], [1.1093, 0.9889], [0.7153,
0.7097], [1.0967, 1.1034], [0.7020, 0.7087], [1.1125, 1.1047], [0.7192, 0.7100], [1.0918, 1.1031], [1.0071,
1.0094], [1.0000, 1.0000]. Frame 10's step counter shows the trace index.

**`math/scaling.js` signatures** (owner: `scaling-laws`; pure, no DOM; tests first). 6ND is **not** defined
here: per the coordinator's ruling (2026-10-07) its single exporter is `trainingFlops({ params, tokens })` in
`math/scale.js` (owner `scale-reliability`), which this module imports for `lifetimeFlops` and the page
imports for frame 1 (README lesson 16).
```js
export const CHINCHILLA_FIT = { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 } // Epoch 2024

// imported, not defined: trainingFlops({ params, tokens }) from math/scale.js (owner scale-reliability)
//   ({ params: 49e9, tokens: 33e12 }) → 9.702e24 (this page's frame 1) · ({ params: 405e9, tokens: 15.6e12 }) → 3.791e25 (both pages)

tokensPerParam(D: number, N: number) → number         // D / N; callers pass active params for MoE
//   (33e12, 49e9) → 673.47 · (33e12, 1.6e12) → 20.63

fittedLoss(N, D, fit = CHINCHILLA_FIT) → number
//   (9.586e10, 1.739e12) → 1.9597

isoFlopLoss(C, N, fit) → { D, tokensPerParam, loss }   // D = C / 6N
//   (1e24, 1e12) → { 1.667e11, 0.1667, 2.0133 } · (1e24, 1e9) → { 1.667e14, 166666.67, 2.1874 }

computeOptimal(C, fit) → { N, D, tokensPerParam, loss }   // closed form
//   1e24 → { 9.586e10, 1.739e12, 18.14, 1.9597 } · 1e22 → { 9.045e9, 1.843e11, 20.37, 2.1411 }

tokensForLoss(N, targetLoss, fit) → number            // Infinity when N alone cannot reach the loss

lifetimeFlops(N, D, inferenceTokens) → number         // 6ND + 2N·inferenceTokens
//   (9.586e10, 1.739e12, 1e14) → 2.017e25

inferenceAwareOptimum({ targetLoss, inferenceTokens, fit, logNMin = 8, logNMax = 13, steps = 2000 })
  → { N, D, tokensPerParam, train, serve, total }      // grid search over log10 N
//   (1.9597, 1e14) → { 2.917e10, 1.444e13, 494.93, 2.528e24, 5.835e24, 8.362e24 }
//   (1.9597, 0)    → N ≈ 9.6e10 (the compute-optimal size within grid resolution)

// Muon's Newton–Schulz polynomial applied to each singular value of the Frobenius-normalized update.
newtonSchulzSingular(singularValues: number[], schedule: [a, b, c][]) → number[][]   // trace incl. step 0
//   ([3, 0.3], DeepSeek-V4's 8 + 2 schedule) → step 0 [0.995, 0.0995] … step 10 [1, 1]
```
Test cases for the builder: `computeOptimal(C).loss ≤ isoFlopLoss(C, N).loss` for N on a grid around N*;
`tokensForLoss(N, fittedLoss(N, D)) = D`; `inferenceAwareOptimum` with 0 serving tokens returns N within
one grid step of `computeOptimal`; its `tokensPerParam` is non-decreasing as `inferenceTokens` grows;
`newtonSchulzSingular` with (2, −1.5, 0.5) has fixed point 1; inputs are not mutated.

## 12. Open questions for the reviewer
**Data-pass keys** (new): `deepseek-v4-flash.active_params` = 13e9, `.total_params` = 284e9 ·
`nemotron-3-super.active_params` = 12e9, `.total_params` = 120e9, `.optimizer` = "AdamW" ·
`olmo-3.pretrain_tokens` = 5.9e12 (reported) · `kimi-k3.scaling_efficiency_vs_k2` = 2.5 ·
`kimi-k3.optimizer`, `glm-5.optimizer`, `mimo-v2-flash.optimizer` · `deepseek-v4-pro.muon_ns_schedule`. A
non-model fact: the Chinchilla re-fit constants (proposed `papers.chinchilla-refit-2024`, arXiv 2404.10102,
confirmed by the author's read on 2026-10-07; same "paper-level fact" question raised on `agentic-rl`).

**Cross-track reconciliation (resolved by the coordinator, 2026-10-07):** 6ND has one exporter,
`trainingFlops({ params, tokens })` in `math/scale.js`, owned by `scale-reliability`. This page imports it
(frame 1, `lifetimeFlops`) and defines no copy. The two pages' worked examples agree: 405B × 15.6T →
3.7908e25. Note the dependency direction: a Recipe module imports a GPUs-track module although
`scale-reliability` comes later in the graph; the learner meets 6ND here first.

**Graph changes:** none. Adding `scaling-laws` to `midtraining`'s prereqs is not proposed, because the LR
schedule now lives on `midtraining`; see below.

**Judgment calls:**
- **LR schedules moved to `midtraining`.** Spec §3.2 puts "LR schedules (cosine vs WSD)" here. `midtraining`
  needs the decay phase to explain annealing and does not list this page as a prereq, so it owns schedules;
  this page does not mention them. The reviewer may instead add `scaling-laws` to `midtraining`'s prereqs and
  move them back.
- **Beyond-brief source.** The fitted constants come from arXiv 2404.10102, re-verified as recorded in the
  header. The alternative is a labeled stand-in fit tuned to 20 tokens per parameter.
- **Active-parameter denominator for MoE.** The fit is for dense models; the page applies it with active
  parameters and shows the per-total column beside it (frame 8), so the learner sees both.
- **Serving what-ifs.** `Dinf` chips are hypothetical volumes, labeled as such; the briefs give no serving
  volumes per model.
