# Mid-training (`midtraining`)

Track: training · Section: recipe · Prereqs: pretraining, rope
Next: none in the graph (no slug lists `midtraining` as a prereq; see §12). The recipe continues at `sft`,
whose only prereq, `pretraining`, the learner already has.
Status: draft
Sources: 02 §0 item 3, §1.4, §1.8, §2, §5 step 2, §7 (LR schedules and mid-training rows) · 05 §1.1, §1.2

Scope decision (recorded on both pages): spec §3.2 lists LR schedules under `scaling-laws`. This page owns
them instead, because annealing is the decay phase of the schedule and this page does not have
`scaling-laws` as a prereq. `scaling-laws` does not mention schedules.

Running examples: a learning-rate curve over one run, drawn as a fraction of the peak (most labs do not
publish their peak), with Nemotron 3 Super's fully published schedule as the absolute example; and GLM-5's
four context stages, `4K → 32K → 128K → 200K`, with their token budgets.

## 1. Learning objective
After this page you can draw the three learning-rate schedules 2026 labs use (cosine, warmup-stable-decay,
and constant with a short decay tail), explain why the decay at the end is where labs put their best data
(annealing) and why warmup-stable-decay lets them choose that moment late, say what goes into the named
mid-training stage, and explain why context is extended in a few short stages near the end (GLM-5: 4K to
200K with 1.55T of 28.55T tokens) rather than from the start.

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
  Kimi K3 uses no positional encoding in its attention layers (NoPE), so it needs no rescaling. Corrected by
  frame 8. (source: 02 §2 Kimi K3; `rope`)

## 3. Hook and intuition (final wording)
**Hook:** Why do labs save their best data and their longest documents for the last few percent of
pretraining?

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
of the run. Models that use RoPE also rescale their rotations so the new, larger offsets look familiar;
Kimi K3 avoids that by using no positional encoding in its attention layers. The cost of all this is
choice: a bad anneal mix is baked in at exactly the moment the model is settling, and the briefs show labs
still disagree on the schedule.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `curvePlot` (proposed on `prefill-decode`, with the `bands` option proposed on `scaling-laws`: the learning-rate
schedule, x = tokens, y = fraction of peak or absolute LR; bands for warmup, stable and decay phases; a
selection-style marker for the "stop or branch here" point), `shareBar` (proposed on `decoder-anatomy`:
token budget by context length, with hatched `not published` segments), `block` (the data mixture's
sources, `active` vs `dim`), `flow` (carry `token` from the mixture into the run), `token` (document chips
of different lengths in frames 6–7), plain labeled text marks (phase labels, facts, the per-model table).

New glyphs proposed: none on this page (`curvePlot` and its `bands` option are proposed elsewhere).

Stand-ins (visible line under the stage): "Curves are drawn as a fraction of the peak learning rate
because most labs do not publish their peak; Nemotron 3 Super's numbers are its published ones. The decay
curve's shape is not published for every lab and is drawn linear unless stated."

Terms introduced (one per frame): learning rate and warmup (1) · cosine schedule (2) · warmup-stable-decay
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
| 3 | The WSD schedule beside it: flat at 1.0 to 80%, then a decay to 0. Bands `warmup · stable · decay`. A dashed branch leaves the plateau at 60% with its own short decay (plain label `branch`). | The plateau draws; then the decay; then the dashed branch peels off. | Warmup-stable-decay keeps the rate flat and drops it only at the end. Any checkpoint on the plateau can be branched into its own short decay. | WSD: 1.0 until 80%, linear decay `0.5` at 90%, `0` at 100% · branch from 60% |
| 4 | The decay band of the WSD curve outlined. Above it, the data mixture `block`s switch: `web` dims while `high-quality · reasoning · code · agent traces` light. | The mix blocks change state as the curve enters the decay band. | As the steps shrink, the model settles into what it reads last. So labs put their best data in the decay: this is annealing. | Nemotron 3 Super decay: last 5T of 25T (20%) · MiniMax-M2 decay: 9.3T of 29.2T (31.8%) |
| 5 | A plain stage label `mid-training` spans the decay band; a fact list: `GLM-5: 1.55T tokens, long documents and synthetic agent trajectories upsampled` · `Olmo 3: ~100B tokens (reported)` · `DeepSeek-V4: agentic data injected here`. | The label stretches across the band; facts type in. | In 2026 this end phase is a named stage, mid-training. It is still next-token prediction on documents, with a richer mix and longer sequences. | GLM-5 mid-training `1.55T of 28.55T = 5.4%` |
| 6 | **Key frame.** A `shareBar` of GLM-5's run by context length: `4K 94.57% · 32K 3.50% · 128K 1.75% · 200K 0.18%`, with document `token` chips above each segment getting longer. | Segments slide in left to right; the 200K segment takes the selection outline. | Context extension grows the sequence length in a few short stages. GLM-5 goes from 4K to 200K tokens using the last 5% of its run. | `27T @ 4K · 1T @ 32K · 500B @ 128K · 50B @ 200K` · total 28.55T (sum of the stages; GLM-5 reports 28.5T) |
| 7 | Beside each segment, a plain readout of attention work per new token relative to 4K: `×8 · ×32 · ×50`. A plain fact: `long documents are scarce`. | The readouts type in under the bar. | Late and short, because each long-context token is costly: at 200K a new token attends to 50 times as many earlier tokens as at 4K. Long documents are scarce too. | `32K / 4K = 8` · `128K / 4K = 32` · `200K / 4K = 50` |
| 8 | A small rotation dial (drawn as a `curvePlot` of angle vs offset) with the trained range shaded and the new offsets beyond it; plain label `RoPE: rescale (see rope)`; a second label `Kimi K3: no positional encoding (NoPE), no rescaling`. | The offset marker slides past the shaded range; the label appears. | A model using RoPE has never seen offsets this large, so its rotations are rescaled to fit. Kimi K3 has no positional encoding in attention and skips this step. | trained to 4K · extended to 200K (GLM-5) · Kimi K3 256K → 1M in its cooldown |
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
| Context `shareBar` | `budgetShares(stages)`; unpublished token counts are hatched `not published` | percent, 2 decimals |
| Attention work per new token vs the first stage | `attentionCostRatio(ctx, firstCtx)` per stage | `×n` |

**Try this** (each leads to a named insight)
1. Predict first: you planned a full run but must stop at 60%. Under `cosine` the LR there is 0.345 of
   peak and still falling, so the checkpoint never got its decay. Switch to `WSD`: 1.000, on the plateau,
   ready to branch into a short decay of its own. → **Insight: WSD lets you choose the end late.** You
   can branch anneals off one long run, with different mixes or lengths, instead of committing on day one.
   (Kimi K3 still found cosine better when both were tuned separately, so it is a trade, not a rule.)
2. Press `Nemotron 3 Super 20%`, then `MiniMax-M2 31.8%`, and watch the decay band widen. → **Insight:
   the decay window where the best data goes is a fifth to a third of these runs**: 5T of 25T, 9.3T of
   29.2T.
3. With `GLM-5`, read the bar: 4K 94.57%, 32K 3.50%, 128K 1.75%, 200K 0.18%, and the attention readouts ×8,
   ×32, ×50. Switch to `Kimi K3` or `DeepSeek-V4`: their per-stage token counts are not published and show
   hatched. → **Insight: context is extended on a thin slice of tokens,** precisely because each long token
   costs much more attention work and long documents are scarce.

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
Color links: the `curvePlot`'s decay band and the `decayFrac` slider share the selection accent; the `shareBar`
segment for the selected stage carries the selection outline.

## 8. In today's models (Oct 2026)
All keys proposed for the data-extension pass; entry ids as accepted in `rlvr-grpo` §13.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Nemotron 3 Super: WSD, 200B-token warmup to 4.5e-4, long plateau, minus-sqrt decay over the final 5T of 25T to 4.5e-6 | `models.nemotron-3-super.lr_schedule` = { kind: "wsd", peak: 4.5e-4, warmup_tokens: 200e9, decay_tokens: 5e12, floor: 4.5e-6, shape: "minus-sqrt" } (confirmed) | 02 §1.4 |
| DeepSeek-V4: 2000-step warmup, constant peak (Pro 2.0e-4, Flash 2.7e-4), cosine decay to 10% of peak near the end; sequence length 4K → 16K → 64K → 1M; batch ramped to 94.4M (Pro) / 75.5M (Flash) tokens | `models.deepseek-v4-pro.lr_schedule`, `.context_schedule` = ["4K", "16K", "64K", "1M"]* (confirmed) | 02 §1.4, §2 |
| Kimi K3: cosine with 1% warmup; its scaling study found cosine beat WSD when each was tuned separately; context 8K → 64K in pretraining and 256K → 1M in the cooldown; NoPE | `models.kimi-k3.lr_schedule` = "cosine, 1% warmup", `.context_schedule` (confirmed) | 02 §1.4, §2 |
| MiniMax-M2: 19.9T constant phase + 9.3T decay phase; context 8K → 32K → 192K during the decay | `models.minimax-m2.lr_schedule` = { constant: 19.9e12, decay: 9.3e12 }, `.context_schedule` (confirmed) | 02 §1.4, §2 |
| GLM-5: cosine decay; context 32K (1T tokens) → 128K (500B) → 200K (50B) after a 27T base at 4K; long documents and synthetic agent trajectories upsampled late; 160B tokens of SWE data | `models.glm-5.lr_schedule` = "cosine", `.context_stages` = [["4K", 27e12], ["32K", 1e12], ["128K", 0.5e12], ["200K", 0.05e12]] (confirmed) | 02 §1.4, §2 |
| MiMo-V2-Flash: AdamW, two-stage cosine (to 1e-4 over 10T, then to 3e-5 over 4T) | `models.mimo-v2-flash.lr_schedule` (confirmed) | 02 §1.4 |
| Olmo 3: 100B-token Dolmino mid-training, then 65K context extension | `models.olmo-3.midtrain_tokens` = 100e9 (reported) | 02 §2 |
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
 GLM-5 pretraining run by context length (28.55T tokens)
 ┌───────────────────────────────────────────────┬─┬┬▒┐
 │ 4K                                     94.57% │ │││▒│
 └───────────────────────────────────────────────┴─┴┴▒┘
   27T @ 4K        32K 3.50%  128K 1.75%  200K 0.18% ◀ followed
                   1T          500B        50B
 [doc] [doc────] [doc──────────] [doc───────────────]
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
});'
```
Output (2026-10-07): Nemotron LR at 0.1T 2.250e-4, 0.2T 4.500e-4, 20T 4.500e-4, 21.25T 2.272e-4, 22.5T
1.350e-4, 25T 4.500e-6 · cosine 0.8536, 0.5, 0.3455, 0.1464, 0.0245, 0 at 25/50/60/75/90/100% · WSD (decay
from 80%) 1, 1, 1, 1, 0.5, 0 · GLM-5 total 2.855e13, shares 94.57 / 3.50 / 1.75 / 0.18%, mid-training 5.43% ·
attention ratios 8, 32, 50 · MiniMax-M2 constant 68.2% / decay 31.8%.

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
`.context_schedule` · `kimi-k3.lr_schedule`, `.context_schedule`, `.context_extension_note` ·
`minimax-m2.lr_schedule`, `.context_schedule` · `glm-5.lr_schedule`, `.context_stages` ·
`mimo-v2-flash.lr_schedule` · `olmo-3.midtrain_tokens` (reported). The exact token counts behind "4K",
"32K", "200K" should come from each report; the page prints the nominal labels.

**Graph changes (proposal):**
- `midtraining` is a leaf. In the curriculum it sits between `pretraining` and `sft`, and `training-pipeline`
  names it as a stop in that order. Option A: add `midtraining` to `sft`'s prereqs, so the recipe reads as
  one line (this would also make `rope` a transitive prereq of `sft`, which it does not need). Option B:
  leave the graph and keep the "continue at `sft`" line in §10. This draft assumes B.
- If the reviewer prefers LR schedules on `scaling-laws` as in spec §3.2, add `scaling-laws` to
  `midtraining`'s prereqs and move frames 1–3 there; frames 4–9 stay.

**Judgment calls:**
- **LR schedules live here, not on `scaling-laws`** (see the header). Recorded on both pages.
- **Minus-sqrt shape.** The brief names it without a formula; the page uses 1 − √p and says so (math note b).
- **GLM-5's base stage at "4K".** The brief says GLM-5 extends "from 4K to 200K" after a 27T base; the page
  draws the base at 4K. The 28.55T total is the sum of the stages; the brief rounds it to 28.5T, and
  `pretraining` prints 28.5T from the same brief line. Lesson 16 asks for one definition: the reviewer
  should pick one (the data entry's value) and both pages print it.
- **Frame 8's rotation drawing** reuses `curvePlot` for angle vs offset; if `rope` (Architecture) proposes a
  clock-hand glyph, frame 8 should use that instead.
- **"All four 2026 reports here"** (frame 9) is scoped to the four reports in the table (02 §2) to satisfy
  README lesson 7.
