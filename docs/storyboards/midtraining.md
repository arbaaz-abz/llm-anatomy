# Mid-training (`midtraining`)

Track: training · Section: recipe · Prereqs: pretraining, rope
Next: none in the graph (no slug lists `midtraining` as a prereq; ruled, see §12). The recipe continues at
`sft`, whose only prereq, `pretraining`, the learner already has.
Status: approved (expert review)
Sources: 02 §0 item 3, §1.4, §1.8, §2, §5 step 2, §7 (LR schedules and mid-training rows) · 05 §1.1, §1.2

Ruling (main session, 2026-10-07): LR schedules live on `midtraining`; spec §3.2 to be amended.
`scaling-laws` does not mention schedules.

Running examples: a learning-rate curve over one run, drawn as a fraction of the peak (most labs do not
publish their peak), with Nemotron 3 Super's fully published schedule as the absolute example; and GLM-5's
four context stages, `4K → 32K → 128K → 200K`, with their token budgets.

## 1. Learning objective
After this page you can draw the two schedule families 2026 labs use (cosine; warmup-stable-decay, of
which DeepSeek-V4's constant-then-cosine-tail is a variant), explain why the decay at the end is where labs put their best data
(annealing) and why warmup-stable-decay lets them choose that moment late, say what goes into the named
mid-training stage, and explain why context is extended in a few short stages near the end (GLM-5: 4K to
200K with 1.55T of its 28.5T tokens) rather than from the start.

## 2. Misconceptions to correct
- **Misconception:** "Mid-training is just fine-tuning." → **Reality:** it is still next-token pretraining
  on documents, with no chat template and no answers to imitate; what changes is the mix (more
  high-quality, reasoning, code and synthetic agent data), the sequence length, and that it usually runs in
  the learning-rate decay. Corrected by frames 4–5. (source: 02 §2 definition)
- **Misconception:** "Long-context models are trained on long documents from the start." → **Reality:**
  most of the run uses 4K–8K sequences; context is stretched in a few late stages on a small slice of
  tokens. GLM-5's 200K stage is 50B tokens, 0.18% of its run. Corrected by frames 6–7 and try-this 3.
  (source: 02 §2 GLM-5, MiniMax-M2, Kimi K3, DeepSeek-V4)
- **Misconception:** "The learning-rate schedule is a fixed recipe; cosine is standard." → **Reality:**
  it is unsettled. Nemotron 3 Super uses warmup-stable-decay, DeepSeek-V4 a constant rate with a cosine
  tail, and Kimi K3 found cosine beat WSD when each was tuned separately. WSD's draw is that you can branch
  a decay off any plateau checkpoint. Corrected by frames 2–3 and try-this 1. (source: 02 §1.4 table and
  takeaway)
- **Misconception:** "Longer context needs no changes besides longer documents." → **Reality:** a model
  using RoPE has never seen the larger position offsets, so the rotation is rescaled (`rope` covers how);
  Kimi K3's 24 full-attention layers have no positional encoding (its linear-attention layers carry order
  another way), so it skips the rescaling. Corrected by frame 8. (source: 02 §2 Kimi K3; `rope`)

## 3. Hook and intuition (final wording)
**Hook:** Why do labs save their best data for the end of pretraining, and their longest documents for its
last few percent?

The **learning rate** sets how big each weight update is. Runs warm it up from near zero, hold it high
while the model learns broadly, and decay it at the end. With a **cosine** schedule the rate falls
smoothly over the whole run, so the end has to be fixed on day one. With **warmup-stable-decay** (WSD) the
rate stays flat for most of the run and falls only in a final stretch; any checkpoint on the plateau can be
branched into its own short decay. That makes the end of the run a separate, cheaper decision.

The decay matters because, as the steps shrink, the model stops bouncing around and settles into what it
is reading at that moment. So labs put their best data there: this is **annealing**, and in 2026 it is a
named stage, **mid-training**. The mix shifts toward high-quality text, reasoning, code and synthetic agent
trajectories. It is still plain next-token prediction on documents: no chat template, no answers to
imitate.

Mid-training is also where context grows. Most of pretraining runs on 4K–8K-token sequences, because
long documents are scarce and every token in a long sequence costs more attention work: at 200K each new
token is compared with 50 times as many earlier tokens as at 4K. So labs stretch the window in a few short
stages near the end. GLM-5 goes 4K → 32K (1T tokens) → 128K (500B) → 200K (50B): the 200K stage is 0.18%
of the run (GLM-5's 28.5T; its published stages sum to 28.55T). Models that use RoPE also rescale their rotations so the new, larger offsets look familiar;
Kimi K3 avoids that: its 24 full-attention layers have no positional encoding (its linear-attention
layers carry order another way). The cost of all this is
choice: a bad anneal mix is baked in at exactly the moment the model is settling, and the briefs show labs
still disagree on the schedule.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `curvePlot` (built in S3, ruling P3-R10: the learning-rate
schedule, x = share of the run with `domain: [0, 1]` (or tokens for the Nemotron preset), y = fraction of
peak with `domain: [0, 1]` (or absolute LR); `bands: [{ from, to, label }]` for the warmup, stable and decay
phases; series `style: 'solid' | 'muted'` (`labelAt: 'mid'` for the `branch` label), never dashed; in the
toy the "stop or branch here" point is the one marker with `followed: true`, which draws its own
`G.selectionMark`; in frames 3–5 the page draws `G.selectionMark` around the decay band, since curvePlot
outlines only a followed marker),
`shareBar` (token budget by context length; unpublished counts passed as `{ value: null }`, drawn in the
neutral fill at the fixed width outside the scale, labeled `not published`, never hatched; a bar with no
published counts is labeled `no published shares`, ruling P3-R6; frame 6 uses its zoomed tail with
`tailBasis: 'tail'` and `tailLabel: 'last 5%'`, ruling P3-R7), `dial` (proposed on `rope`, which lists this
page as a reuser: frame 8), `block` (the data mixture's
sources, `active` vs `dim`), `flow` (carry `token` from the mixture into the run), `token` (document chips
of different lengths in frames 6–7), plain labeled text marks (phase labels, facts, the per-model table).

New glyphs proposed: none.

Stand-ins (visible line under the stage): "Curves are drawn as a fraction of the peak learning rate
because most labs do not publish their peak; Nemotron 3 Super's numbers are its published ones. The decay
curve's shape is not published for every lab and is drawn linear unless stated."

Terms introduced (one per frame; "learning rate" is defined in §3 and on frame 1's axis label): warmup (1) · cosine schedule (2) · warmup-stable-decay
(3) · annealing (4) · mid-training (5) · context extension (6) · none new (7, the cost per token) · none
new (8, RoPE rescaling links to `rope`) · none new (9).
Terms assumed from prereqs: pretraining loss, data mixture, quality classifier (`pretraining`); RoPE,
rotation, position offset (`rope`); attention over earlier tokens (`attention`).

Indexing: no positions or addresses are numbered on this page.

Selection: the followed item is the decay window (frames 3–5) and then GLM-5's 200K stage (frames 6–7),
each outlined where it appears.

## 5. Animation script
Thread order: 1–4 the schedule and why its end matters; 5 the named stage; 6–8 context extension; 9 the
2026 table.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Empty `curvePlot`: x = share of the run (0–100%), y = learning rate as a fraction of peak (0–1). A short warmup ramp draws from 0 to 1. | The line climbs from 0 to 1 over the first few percent. | The learning rate sets how big each weight update is. Runs start it near zero and raise it over a short warmup, so early updates do not wreck the weights. | Nemotron 3 Super: warmup over the first 200B of 25T tokens to 4.5 × 10⁻⁴ |
| 2 | The cosine schedule draws from the peak down to 0 across the whole run. A plain mark at 60%: `0.35 of peak`. | The curve sweeps down; the 60% mark drops in. | A cosine schedule lowers the rate smoothly over the whole run. The finish line has to be chosen on day one, because the curve depends on it. | cosine at 25% `0.854`, 50% `0.500`, 60% `0.345`, 75% `0.146`, 100% `0` |
| 3 | The WSD schedule beside it: flat at 1.0 to 80%, then a decay to 0. Bands `warmup · stable · decay`. A branch leaves the plateau at 60% with its own short decay, drawn as a solid `muted` series labeled `branch` (no dashes, ruling P3-R11). A plain mark: `DeepSeek-V4: constant, then a cosine tail to 10% (start not published)`. | The plateau draws; then the decay; then the main decay dims while the muted branch peels off and draws. | Warmup-stable-decay keeps the rate flat and drops it only at the end. Any checkpoint on the plateau can be branched into its own short decay. | WSD: 1.0 until 80%, linear decay `0.5` at 90%, `0` at 100% · branch from 60% |
| 4 | The decay band of the WSD curve outlined. Above it, the data mixture `block`s switch: `web` dims while `high-quality · reasoning · code · agent traces` light; a legend under the stage names both groups (`dims: web` · `lights: high-quality · reasoning · code · agent traces`). | The mix blocks change state as the curve enters the decay band. | As the steps shrink, the model settles into what it reads last. So labs put their best data in the decay: this is annealing. | Nemotron 3 Super decay: last 5T of 25T (20%) · MiniMax-M2 decay: 9.3T of 29.2T (31.8%) |
| 5 | A plain stage label `mid-training` spans the decay band; a fact list: `GLM-5: 1.55T tokens, long documents and synthetic agent trajectories upsampled` · `Olmo 3: ~100B tokens (reported)` · `DeepSeek-V4: agentic data injected here`. | The label stretches across the band; facts type in. | In 2026 this end phase is a named stage, mid-training. It is still next-token prediction on documents, with a richer mix and longer sequences. | GLM-5 mid-training `1.55T of 28.55T = 5.4%` (28.5T reported; its published stages sum to 28.55T) |
| 6 | **Key frame.** One `shareBar` (560 px) with its zoomed tail (`tail: 'zoom'`, `tailBasis: 'tail'`, `tailLabel: 'last 5%'`, ruling P3-R7). Main bar: GLM-5's whole run by context length, `4K 94.57%` printed in its segment; 32K (17.6 px drawn), 128K and 200K fold into the tail, bracketed `last 5%`. The zoomed tail bar (560 px) states its basis in its label, `the last 5%, zoomed (1.55T): shares of the tail`: `32K 64.5% · 128K 32.3% · 200K 3.2%` (shares of the tail, through `sharePct`), the 200K segment 18.1 px wide, under the 30 px print threshold, so its share prints in the tail's legend as `200K · 3.2% of last 5%` (no leader line; the page draws `G.selectionMark` around the segment). Document `token` chips above the zoomed segments get longer. | The upper bar slides in; the bracket opens; the zoomed bar slides in segment by segment; the zoomed 200K segment takes the selection outline. | Context extension grows the sequence length in a few short stages. GLM-5 goes from 4K to 200K tokens using the last 5% of its run. | `27T @ 4K · 1T @ 32K · 500B @ 128K · 50B @ 200K` · 28.5T (its published stages sum to 28.55T) · zoomed: `1 / 1.55 = 64.5%`, `0.5 / 1.55 = 32.3%`, `0.05 / 1.55 = 3.2%` · 200K is 0.18% of the whole run |
| 7 | Beside each segment, a plain readout of attention work per new token in the forward pass, relative to 4K: `×8 · ×32 · ×50`. A plain fact: `long documents are scarce`. A visible line under the readouts: `attention only: the rest of the forward pass costs the same per token (Reaching 1M tokens shows how 2026 models cut this)`. | The readouts type in under the zoomed bar. | Late and short, because each long-context token is costly: at 200K a new token attends to 50 times as many earlier tokens as at 4K. Long documents are scarce too. | `32K / 4K = 8` · `128K / 4K = 32` · `200K / 4K = 50` |
| 8 | Two `dial`s (from `rope`) for the slowest-rotating pair, labeled `offset 4K` and `offset 200K`. Each has the pale `seen` sector "angles seen in training" from 0 to 0.015 turn (the angle at offset 4K). The left hand sits at 0.015 turn, inside the sector; the right hand sweeps to ¾ turn, far outside it. No degrees are printed (dial condition e, ruling P3-R18); the dial's quarter-turn ticks carry the reading. Plain label `RoPE: rescale (see the RoPE lesson)`; a second label `Kimi K3: 24 full-attention layers with no positional encoding; nothing to rescale`. Stand-in line: `pair speed chosen so 200K is ¾ turn`. | The right dial's hand sweeps from 0.015 turn to ¾ turn past the sector edge. | A model using RoPE has never seen rotations this large, so its rotations are rescaled to fit. Kimi K3's full-attention layers have no positional encoding and skip this step. | slowest pair, stand-in speed: `¾ turn × 4K / 200K = 0.015 turn` at 4K · `¾ turn` at 200K · trained to 4K · extended to 200K (GLM-5) · Kimi K3 256K → 1M in its cooldown |
| 9 | A plain table of 2026 runs: GLM-5 `4K → 32K → 128K → 200K`, MiniMax-M2 `8K → 32K → 192K inside its 9.3T decay`, DeepSeek-V4 `4K → 16K → 64K → 1M`, Kimi K3 `8K → 64K in pretraining, 256K → 1M in cooldown`. | Rows type in. | All four 2026 reports here stage their context this way, ending near 200K or 1M. Kimi K3 says why: the extension is cheap because it is short. | as listed |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 1–4 draw `lrAt`; frame 6 draws `budgetShares` (`math/pipeline.js`, owned by
`training-pipeline`); frame 7 draws `attentionCostRatio`. Caption counts (words/sentences, `rlvr-grpo`'s counter, run 2026-10-07): 30/2, 27/2, 25/2, 24/2, 24/2, 25/2, 30/2,
29/2, 27/2.

## 6. Toy
"Plan the end of a run." A schedule you can switch, a "stop here" point, and presets for how 2026 runs
spend their last tokens.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `schedule` | Learning-rate schedule | Preset chips | `cosine` / `WSD` (linear decay) / `WSD minus-sqrt` (Nemotron's shape) | `WSD` | – |
| `decayFrac` | Share of the run spent decaying (WSD) | Slider | 5%–40%, step 5% | 20% | chips `Nemotron 3 Super 20%` · `MiniMax-M2 31.8%` |
| `stopAt` | Where you decide to stop or branch | Slider | 50%–100%, step 5% | 60% | – |
| `run` | Context stages | Preset chips | `GLM-5` / `MiniMax-M2` / `DeepSeek-V4` / `Kimi K3` | `GLM-5` | the chips |

**Live outputs** (printed)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| LR at `stopAt`, as a fraction of peak, under the chosen schedule | `lrAt(stopAt, { kind, total: 1, decayStart: 1 − decayFrac, decayShape })` | 3 decimals |
| The same for the other schedule (side by side) | `lrAt(...)` | 3 decimals |
| Nemotron 3 Super absolute LR at the marker (when its chip is on) | `lrAt(t, { kind: 'wsd', peak: 4.5e-4, floor: 4.5e-6, warmup: 0.2, total: 25, decayStart: 20, decayShape: 'minus-sqrt' })` (t in T tokens) | scientific, 3 s.f. |
| Context `shareBar` with its zoomed tail (any segment under 18 px folds into the tail; `tailBasis: 'tail'`) | `budgetShares(stages)` for the main bar, `budgetShares(tailStages)` for the tail's printed shares; unpublished counts passed as `{ value: null }` (neutral fill, `not published`); a run with no published counts (DeepSeek-V4, Kimi K3, MiniMax-M2) draws the all-unknown bar labeled `no published shares` | `sharePct`: percent, 2 decimals (tail: 1 decimal) |
| Attention work per new token vs the first stage | `attentionCostRatio(ctx, firstCtx)` per stage | `×n` |

**Try this** (each leads to a named insight)
1. Predict first: you planned a full run but must stop at 60%. Under `cosine` the LR there is 0.345 of
   peak and still falling, so the checkpoint never got its decay. Switch to `WSD`: 1.000, on the plateau,
   ready to branch into a short decay of its own. → **Insight: WSD lets you choose the end late.** You
   can branch anneals off one long run, with different mixes or lengths, instead of committing on day one.
   Kimi K3 still found cosine better when both were tuned separately, so this is a trade, not a rule.
2. Press `Nemotron 3 Super 20%`, then `MiniMax-M2 31.8%`, and watch the decay band widen. → **Insight:
   the decay window where the best data goes is a fifth to a third of these runs**: 5T of 25T, 9.3T of
   29.2T.
3. With `GLM-5`, read both bars: 4K 94.57% of the run; zoomed, 32K 64.5%, 128K 32.3%, 200K 3.2% of the last
   1.55T (0.18% of the run), and the attention-only readouts ×8, ×32, ×50. Switch to `Kimi K3` or
   `DeepSeek-V4`: their per-stage token counts are not published, so the bar shows only neutral segments
   labeled `no published shares`. → **Insight: context is extended on a thin slice of tokens,** precisely because each long token
   costs much more attention work and long documents are scarce.

**Check my work** (default state: `WSD`, decay 20%, stop at 60%; templated from `lrAt` for any state, the
plateau, decay and cosine lines chosen by where `stopAt` falls; mono, `aria-live="polite"`; this exact text
appears on the page):
```text
WSD: the decay starts at 100% − 20% = 80% of the run
stop at 60%: before 80%, on the plateau → 1.000 of peak
cosine at 60%: ½ × (1 + cos(π × 0.60)) = ½ × (1 − 0.309) = 0.345 of peak
```
(Inside the decay the WSD line reads, e.g. at 90%: `1 − (90% − 80%) ÷ 20% = 0.500 of peak`.)

Lesson-17 check: the readouts give tokens per stage and per-token attention work separately and never a
"cheap" total; the page's "cheap" claim is Kimi K3's, quoted, and the toy does not compute total compute
(that would need each model's dimensions). Try-this 1's numbers are `lrAt` at 0.6 under each kind.

## 7. Show me the math
```tex
\text{cosine:}\ \eta(t) = \eta_{\min} + (\eta_{\max} - \eta_{\min})\,\tfrac12\Big(1 + \cos \frac{\pi (t - t_w)}{T - t_w}\Big)
```
```tex
\text{WSD:}\ \eta(t) = \begin{cases}
\eta_{\max}\, t / t_w & t < t_w \\
\eta_{\max} & t_w \le t \le t_d \\
\eta_{\min} + (\eta_{\max} - \eta_{\min})\, s\!\big(\tfrac{t - t_d}{T - t_d}\big) & t > t_d
\end{cases}
\qquad s(p) = 1 - p \ \text{(linear)},\ \ 1 - \sqrt{p}\ \text{(minus-sqrt)}
```
```tex
\text{attention work per new token} \propto \text{context length } L \quad\Rightarrow\quad \frac{L_{200K}}{L_{4K}} = 50
```
Shapes: η and t are scalars; t in tokens. Panel notes: (a) DeepSeek-V4 uses a 2,000-step warmup, a
constant peak (2.0 × 10⁻⁴ for Pro) for most of training, then a cosine decay to 10% of peak "near the end";
its decay start is not published, so it has no `decayFrac` chip. (b) Nemotron 3 Super's decay is called
"minus-sqrt" in its report; the page's `s(p) = 1 − √p` is the standard form of that shape, stated here as
the page's assumption. (c) Batch size also ramps during DeepSeek-V4's run (to 94.4M tokens for Pro).
Color links: the decay band (page-drawn `G.selectionMark` in frames 3–5) and the `decayFrac` slider share the
selection accent; the `shareBar` segment for the selected stage carries the selection outline (page-drawn).

## 8. In today's models (Oct 2026)
Every key below is in `data/models.json` (data passes 2026-10-07 and 2026-10-08); entry ids as accepted in `rlvr-grpo` §13.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Nemotron 3 Super: WSD, 200B-token warmup to 4.5e-4, long plateau, minus-sqrt decay over the final 5T of 25T to 4.5e-6 | `models.nemotron-3-super.lr_schedule` (the string, printed) and the numeric keys the toy's preset reads: `.lr_schedule_kind` = "wsd", `.lr_peak` = 4.5e-4, `.lr_floor` = 4.5e-6, `.lr_warmup_tokens` = 200e9, `.lr_decay_tokens` = 5e12, `.lr_decay_shape` = "minus-sqrt", with `.pretrain_tokens` = 25e12 (confirmed) | 02 §1.4 |
| DeepSeek-V4: 2000-step warmup, constant peak (Pro 2.0e-4, Flash 2.7e-4), cosine decay to 10% of peak near the end; sequence length 4K → 16K → 64K → 1M; batch ramped to 94.4M (Pro) / 75.5M (Flash) tokens | `models.deepseek-v4-pro.lr_schedule`, `.context_stages` = ["4K", "16K", "64K", "1M"]* (confirmed) | 02 §1.4, §2 |
| Kimi K3: cosine with 1% warmup; its scaling study found cosine beat WSD when each was tuned separately; context 8K → 64K in pretraining and 256K → 1M in the cooldown; NoPE | `models.kimi-k3.lr_schedule` = "cosine, 1% warmup", `.context_stages` (confirmed) | 02 §1.4, §2 |
| MiniMax-M2: 19.9T constant phase + 9.3T decay phase; context 8K → 32K → 192K during the decay | `models.minimax-m2.lr_schedule` (string), `.constant_phase_tokens` = 19.9e12, `.decay_phase_tokens` = 9.3e12 (the 31.8% chip: `sharePct(9.3e12, 29.2e12)`), `.context_stages` (confirmed) | 02 §1.4, §2 |
| GLM-5: cosine decay; context 32K (1T tokens) → 128K (500B) → 200K (50B) after a 27T base at 4K; long documents and synthetic agent trajectories upsampled late; 160B tokens of SWE data | `models.glm-5.lr_schedule` = "cosine", `.context_stages` (string, printed), and the numeric stage counts the bars read: `.base_tokens` = 27e12, `.stage_32k_tokens` = 1e12, `.stage_128k_tokens` = 0.5e12, `.stage_200k_tokens` = 0.05e12; `.midtrain_data` = "long documents and synthetic agent trajectories upsampled at later stages"; `.swe_data_tokens` (confirmed) | 02 §1.4, §2 |
| MiMo-V2-Flash: AdamW, two-stage cosine (to 1e-4 over 10T, then to 3e-5 over 4T) | `models.mimo-v2-flash.lr_schedule` (confirmed) | 02 §1.4 |
| Olmo 3: 100B-token Dolmino mid-training, then 65K context extension | `models.olmo-3.midtrain_tokens` = 100e9, `.context_extension` = "65K" (reported) | 02 §2 |
| DeepSeek-V4: agentic data injected in mid-training (frame 5) | `models.deepseek-v4-pro.midtrain_data` = "agentic data in mid-training" (confirmed) | 02 §2 |
| Kimi K3 states that context extension is cheap because it is concentrated in a small fraction of the budget | `models.kimi-k3.context_extension_note` (confirmed) | 02 §2 |

\* "4K", "32K" etc. are the briefs' nominal labels; the data pass should store the exact token counts from
each report, or the labels as strings, and the page prints the labels.

## 9. Takeaways
1. Runs warm the learning rate up, hold it, and decay it; cosine fixes the end on day one, while
   warmup-stable-decay lets you branch a decay off any plateau checkpoint, and labs still disagree on which
   is better.
2. The decay is where the model settles, so labs anneal on their best data there; in 2026 that end phase
   is the named mid-training stage (still next-token prediction on documents).
3. Context grows late and in a few short stages (GLM-5: 4K to 200K on 5% of its tokens) because long
   tokens cost more attention work and long documents are scarce; RoPE models rescale positions to match.

## 10. Next and go deeper
Next: none in the graph. Continue the recipe at `sft` (prereq `pretraining`, which you have). Related:
`rope` for how position scaling works, `long-context-attention` for how 2026 models make 1M tokens
affordable.
Go deeper: Hugging Face, *The Smol Training Playbook* (schedules, annealing and ablations;
https://huggingfacetb-smol-training-playbook.hf.space/) · Andrej Karpathy, *nanochat*, which has a
separate midtrain stage (https://github.com/karpathy/nanochat)

## 11. Key-frame sketch
Frame 6 (GLM-5's context stages), desktop width. `▒` = selected segment. Checked against the reproducer.

```text
 GLM-5: 28.5T (its published stages sum to 28.55T)
 ┌───────────────────────────────────────────────┬─┬┬┐
 │ 4K                                     94.57% │ │││  ← last 5%
 └───────────────────────────────────────────────┴─┴┴┘
                                                  ╲___╱
 the last 5%, zoomed (1.55T): shares of the tail
 ┌──────────────────────────────┬───────────────┬▒┐
 │ 32K 64.5%                    │ 128K 32.3%    │▒│
 └──────────────────────────────┴───────────────┴▒┘
   1T                             500B            50B ◀ followed
 legend: … · 200K · 3.2% of last 5%
 [doc────] [doc──────────] [doc───────────────]
 Context extension grows the sequence length in a few
 short stages. GLM-5 goes from 4K to 200K tokens using
 the last 5% of its run.
 [Back] [Pause] [Next]  ━━━━━●━━━━  6 / 9   1×
```

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against scratchpad drafts of
`math/schedule.js` and `math/pipeline.js` matching the signatures):
```sh
node -e '
Promise.all(["./math/schedule.js","./math/pipeline.js"].map((x)=>import(x))).then(([s,p])=>{
  const f=(x)=>Number(x.toFixed(4)), e=(x)=>x.toExponential(3);
  const nem={kind:"wsd",peak:4.5e-4,floor:4.5e-6,warmup:0.2,total:25,decayStart:20,decayShape:"minus-sqrt"};
  for (const t of [0.1,0.2,20,21.25,22.5,25]) console.log("nem",t,e(s.lrAt(t,nem)));
  for (const t of [0.25,0.5,0.6,0.75,0.9,1]) console.log(t,"cos",f(s.lrAt(t,{kind:"cosine",total:1})),"wsd",f(s.lrAt(t,{kind:"wsd",total:1,decayStart:0.8})));
  const g=p.budgetShares([{name:"4K",value:27e12},{name:"32K",value:1e12},{name:"128K",value:0.5e12},{name:"200K",value:0.05e12}]);
  console.log(e(g.knownTotal),g.parts.map((q)=>(100*q.share).toFixed(2)),(100*1.55e12/g.knownTotal).toFixed(2));
  console.log([32,128,200].map((k)=>s.attentionCostRatio(k,4)), p.budgetShares([{name:"c",value:19.9e12},{name:"d",value:9.3e12}]).parts.map((q)=>(100*q.share).toFixed(1)));
  const z=p.budgetShares([{name:"32K",value:1e12},{name:"128K",value:0.5e12},{name:"200K",value:0.05e12}]);
  console.log("zoom",e(z.knownTotal),z.parts.map((q)=>(100*q.share).toFixed(2)),"px at 560",z.parts.map((q)=>(560*q.share).toFixed(1)),"full-bar px",g.parts.map((q)=>(560*q.share).toFixed(1)),"dial",270*4/200);
});'
```
Output (2026-10-07): Nemotron LR at 0.1T 2.250e-4, 0.2T 4.500e-4, 20T 4.500e-4, 21.25T 2.272e-4, 22.5T
1.350e-4, 25T 4.500e-6 · cosine 0.8536, 0.5, 0.3455, 0.1464, 0.0245, 0 at 25/50/60/75/90/100% · WSD (decay
from 80%) 1, 1, 1, 1, 0.5, 0 · GLM-5 total 2.855e13, shares 94.57 / 3.50 / 1.75 / 0.18%, mid-training 5.43% ·
attention ratios 8, 32, 50 · MiniMax-M2 constant 68.2% / decay 31.8% · zoomed bar (1.55e12): 64.52 / 32.26 / 3.23%,
361.3 / 180.6 / 18.1 px at 560 px · full bar at 560 px: 529.6 / 19.6 / 9.8 / 1.0 px · dial 5.4°.

**`math/schedule.js` signatures** (owner: `midtraining`; pure, no DOM; tests first):
```js
// Learning rate at position t (tokens or a 0–1 fraction; any consistent unit).
lrAt(t: number, { kind: 'cosine' | 'wsd', peak = 1, floor = 0, warmup = 0, total: number,
                  decayStart?: number, decayShape = 'linear' | 'minus-sqrt' | 'cosine' }) → number
//   (0.6, { kind: 'cosine', total: 1 })                     → 0.3455
//   (0.6, { kind: 'wsd', total: 1, decayStart: 0.8 })       → 1
//   (0.9, { kind: 'wsd', total: 1, decayStart: 0.8 })       → 0.5
//   (21.25, Nemotron preset)                                → 2.272e-4
//   throws RangeError unless 0 ≤ t ≤ total, and for 'wsd' unless warmup ≤ decayStart < total

attentionCostRatio(ctx: number, baseCtx: number) → number   // ctx / baseCtx
//   (200, 4) → 50 · (32, 4) → 8
```
Reuses `budgetShares(parts)` from `math/pipeline.js` (owner `training-pipeline`):
`[27e12, 1e12, 0.5e12, 0.05e12]` → shares 0.9457 / 0.0350 / 0.0175 / 0.0018, `knownTotal` 2.855e13.

Test cases for the builder: `lrAt` is continuous at warmup end and at `decayStart`; it equals `floor` at
`total` for every shape; WSD is constant on the plateau; cosine is non-increasing after warmup; inputs are
not mutated.

## 12. Open questions for the reviewer
**Data-pass keys** (new): `nemotron-3-super.lr_schedule` · `deepseek-v4-pro.lr_schedule`,
`.context_stages` · `kimi-k3.lr_schedule`, `.context_stages`, `.context_extension_note` ·
`minimax-m2.lr_schedule`, `.context_stages` · `glm-5.lr_schedule`, `.context_stages` ·
`mimo-v2-flash.lr_schedule` · `olmo-3.midtrain_tokens` (reported). The exact token counts behind "4K",
"32K", "200K" should come from each report; the page prints the nominal labels.

**Graph changes:** none. Ruling (main session, 2026-10-07): `sft` does not list `midtraining`; this page
stays a leaf with "continue at `sft`" in §10.

**Judgment calls:**
- **LR schedules:** Ruling (main session, 2026-10-07): LR schedules live on `midtraining`; spec §3.2 to be
  amended.
- **Minus-sqrt shape.** The brief names it without a formula; the page uses 1 − √p and says so (math note b).
- **GLM-5's base stage at "4K".** The brief says GLM-5 extends "from 4K to 200K" after a 27T base; the page
  draws the base at 4K. GLM-5's total: ruled (expert review) — `glm-5.pretrain_tokens` holds the reported
  28.5e12, `glm-5.context_stages` the four stage counts; every page prints "28.5T (its published stages sum
  to 28.55T)" and computes shares from the stage counts (mid-training 5.4%).
- **Frame 8** uses `rope`'s `dial` (ruled by the expert review); the pair's speed is a stand-in, labeled.
- **"All four 2026 reports here"** (frame 9) is scoped to the four reports in the table (02 §2) to satisfy
  README lesson 7.

## 13. Reviewer rulings (expert review, 2026-10-07)
Applied from `track-review-recipe.md` §4 (change log: `fix-recipe-review.md`):
- Must 1: frame 6 adds a zoomed second bar for the last 1.55T (32K 64.5% · 128K 32.3% · 200K 3.2%, 18.1 px at 560 px) with a bracket; the selection outline sits on the zoomed 200K segment; §6 outputs and the §11 sketch show both bars (README lesson 19).
- Must 2: frame 8 uses `rope`'s `dial` (two dials, stand-in pair speed: 5.4° at 4K, 270° at 200K).
- Must 3: "not published" segments use the neutral fill, never hatch (§4, §6, try-this 3).
- Must 4: the objective names the two schedule families shown; frame 3 carries the DeepSeek-V4 constant-then-cosine-tail mark.
- Must 5: frame 7 carries a visible "attention only" line.
- Should: hook separates "end of pretraining" (best data) from "last few percent" (longest documents); frame 1's term is warmup ("learning rate" defined in §3 and on the axis); Kimi K3 NoPE wording matches `rope`; GLM-5 printed as "28.5T (its published stages sum to 28.55T)"; the LR-schedule and `sft`-prereq alternatives are deleted and the rulings stated (lesson 20); frame 3 dims the main decay while the branch draws; try-this 1's Kimi caveat is part of the insight.
- Nice: frame 4 has a legend naming the groups that dim and light.
- Data pass 2026-10-07: `context_schedule` renamed `context_stages` for deepseek-v4-pro, kimi-k3 and minimax-m2; no numbers changed.

## 14. Plan 3 rulings applied (S3, 2026-10-08)
- P3-R10 / P3-R11: `curvePlot` named with its built API; frame 3's dashed branch is a solid `muted` series
  labeled `branch`.
- P3-R7: frame 6 is one `shareBar` with `tailBasis: 'tail'`, `tailLabel: 'last 5%'`; the tail prints shares of
  the tail (64.5 / 32.3 / 3.2%) and says so in its label. §6's output row matches.
- S3-C final API (reconciled 2026-10-08): axes carry `domain: [0, 1]`; the decay band's outline is a
  page-drawn `G.selectionMark` (curvePlot outlines only a followed marker); frame 6's 200K share prints in
  the tail's legend, "200K · 3.2% of last 5%", with no leader line (the segment is under the print threshold).
- P3-R6: runs with no published stage counts draw the all-unknown bar labeled `no published shares` (§6,
  try-this 3).
- P3-R18: frame 8 prints turns, not degrees (0.015 turn at 4K, ¾ turn at 200K).
- P3-R14 (data gaps 13–17): numeric keys added beside the strings: `nemotron-3-super.lr_schedule_kind`,
  `.lr_peak`, `.lr_floor`, `.lr_warmup_tokens`, `.lr_decay_tokens`, `.lr_decay_shape`;
  `minimax-m2.constant_phase_tokens`, `.decay_phase_tokens`; `glm-5.stage_32k_tokens`, `.stage_128k_tokens`,
  `.stage_200k_tokens`, `.midtrain_data`; `deepseek-v4-pro.midtrain_data`; `olmo-3.context_extension`. The
  string keys stay (printed in §8 rows).
- X-2 / P3-R16: "Check my work" added (§6), from `lrAt`.
