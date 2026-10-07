# The 2026 training pipeline (`training-pipeline`)

Track: training · Section: recipe · Prereqs: decoder-recap
Next: `pretraining` (the one slug whose `prereqs` list `training-pipeline`)
Status: draft
Sources: 02 §0, §1.8, §2, §3, §4.1, §4.2, §4.6, §4.8, §5 (the canonical pipeline), §7 ("whole pipeline"
row) · 05 §1.1, §2 (pattern 9: overview → detail ladder)

This page is the map of the Recipe section. Each frame lights one stage, says in one line what it adds,
and hands off to the lesson that teaches it; no frame explains a mechanism. Frame → stop: 2 →
`pretraining`, `scaling-laws` · 3 → `midtraining` · 4 → `sft` · 5 → `rlhf-dpo`, `rlvr-grpo`, `agentic-rl` ·
6 → `distillation` · 7 → `rlhf-dpo` · 8–9 → the GPUs & scale section (`gpu-primer`, `training-memory`,
`scale-reliability`) for what the compute is spent on. The model it trains is the decoder from
`decoder-anatomy` and `decoder-recap`; this page never changes the architecture.

Running example: one prompt, `What is 7 × 8?`, the one the RL lessons grade. After each stage the
checkpoint's reply changes; those replies are illustrative and labeled so on screen.

## 1. Learning objective
After this page you can name the six stages of a 2026 training pipeline in order (pretraining,
mid-training, supervised fine-tuning, specialist reinforcement learning, merging by on-policy distillation,
final polish), say in one sentence what each adds and which lesson teaches it, and explain why pretraining
is nearly all of the tokens while post-training is no longer a small share of the compute.

## 2. Misconceptions to correct
- **Misconception:** "A model is trained once, end to end, on one objective." → **Reality:** six stages
  with different data and objectives, each starting from the previous stage's weights; in the middle the
  model is even copied into several specialists and merged back. Corrected by frames 1 and 5–6. (source: 02
  §5)
- **Misconception:** "RL is a small finishing touch after pretraining." → **Reality:** in 2026,
  specialist RL is where most reasoning and agent ability comes from, and post-training compute exceeded
  10% of pretraining for DeepSeek-V3.2. Corrected by frames 5 and 9. (source: 02 §0 item 6; §5 step 4)
- **Misconception:** "The share of tokens tells you the share of effort." → **Reality:** pretraining is
  94.6% of GLM-5's published tokens, yet RL is measured as a share of compute because each RL token costs
  far more: it must be generated, scored, and often run through a sandbox first. Corrected by frames 8–9
  and try-this 2. (source: 02 §5 "Token/compute budget shape")

## 3. Hook and intuition (final wording)
**Hook:** What happens between a pile of 30 trillion tokens and a model that thinks before it answers,
calls tools, and holds a conversation?

A 2026 model is built in six stages, each starting from the weights the last one left. **Pretraining**
reads tens of trillions of tokens and learns to predict the next one; the result knows a great deal but
only continues text. **Mid-training** is the end of that run, on the best data and with a longer context
window. **Supervised fine-tuning** shows it worked conversations, so it learns to answer in turns and to
reason between think tags.

Then the model is copied. Each copy, a **specialist**, is trained with reinforcement learning on one kind
of task: math and code with answers a program can check, agent tasks in sandboxes, conversation judged by
another model. This is where most 2026 gains in reasoning and tool use come from. **On-policy distillation**
then merges the specialists back into one model that keeps each one's peak. A **final polish** tunes
style and safety, and the weights are prepared for serving.

The stages are very unequal. Pretraining is nearly all of the tokens: 94.6% of GLM-5's published budget.
But a post-training token is far more expensive than a pretraining one, since it has to be generated,
scored and often run in a sandbox first, so post-training is no longer cheap: DeepSeek-V3.2 spent more
than 10% of its pretraining compute on it. The rest of this section opens each stage in turn.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `block` (the six stage blocks in one row, `active` vs
`dim`; the checkpoint; specialist copies in frames 5–6), `flow` (carry `activation` for the weights handed
from stage to stage; carry `token` for the prompt and replies), `token` (prompt and reply chips),
`blockStack` (proposed on `decoder-anatomy`; drawn once, small, in frame 1 as "the model" so the learner
sees it is the same decoder), `shareBar` (proposed on `decoder-anatomy`; token budget by stage with
`not published` hatched segments), plain labeled text marks (the "adds:" line per stage, the stop links,
the compute-share line, `illustrative`).

`shareBar` usage note for its builder: a segment with `unknown: true` is drawn at a fixed 12% of the bar
width, hatched, outside the percentage scale, and labeled `not published`; known segments share the rest
in proportion.

New glyphs proposed: none.

Stand-ins (visible line under the stage): "The replies to `What is 7 × 8?` are illustrative, not real
model outputs. Token counts are published figures."

Terms introduced (one per frame; each is defined in one clause and taught on its stop): stage and
checkpoint (1) · base model (2) · mid-training (3) · supervised fine-tuning (4) · specialist (5) · merging
by distillation (6) · polish (7) · token share (8) · compute share (9).
Terms assumed from prereqs: decoder, block, parameters, total vs active (`decoder-anatomy`,
`decoder-recap`); next-token prediction (named here, taught on `pretraining`).

Indexing: stages are numbered 1–6 on screen; no 0-based addresses.

Selection: the stage being visited carries the selection outline in every frame; the checkpoint chip
travels along the row.

## 5. Animation script
Thread order: frames 1–7 walk the six stages left to right; 8–9 compare their sizes.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Six `block`s in a row, all `dim`: `1 pretrain · 2 mid-train · 3 SFT · 4 specialist RL · 5 merge · 6 polish`. At the left, a small `blockStack` labeled `the decoder (random weights)` and a reply chip row `zq mat ,, ,,` (`illustrative`). | The row draws left to right; the decoder fades in. | A 2026 model is trained in six stages. Each one starts from the weights the previous stage left, its checkpoint, and adds one thing. | 6 stages |
| 2 | Stage 1 `active` (outlined). A `flow` of document chips into it from a label `25–33 trillion tokens of text`. The reply becomes `What is 9 × 6? What is 4 × 7?`. Stop links: `pretraining · scaling-laws`. | Chips stream in; the reply retypes. | Pretraining predicts the next token over tens of trillions of tokens. The base model it produces knows a lot but only continues text. | 25T–33T tokens (2026 open models) · adds: knowledge, language, code |
| 3 | Stage 2 `active`. A label `best data · longer context: 4K → 200K (GLM-5)`. Stop link: `midtraining`. | The stage block widens slightly as the context label grows. | Mid-training is the end of that run: the best data, and a context window stretched from thousands to hundreds of thousands of tokens. | GLM-5: 1.55T tokens, 4K → 200K · adds: long context, reasoning-heavy data |
| 4 | Stage 3 `active`. The reply becomes `<think> 7 × 8 = 56 </think> 56`. Stop link: `sft`. | The reply retypes inside a chat template. | Supervised fine-tuning shows it worked conversations. Now it answers in turns, thinks between tags, and writes tool calls. | DeepSeek-R1: ~800K examples · Olmo 3: ~2.3M traces (reported) · adds: format, readable reasoning |
| 5 | Stage 4 `active`; the checkpoint splits into three specialist `block`s: `math & code`, `agents`, `chat`. Labels under each: `checked answers`, `sandbox tasks`, `judge model`. Stop links: `rlhf-dpo · rlvr-grpo · agentic-rl`. | The checkpoint copies into three; each specialist lights in turn. | The model is copied into specialists, each trained with reinforcement learning on its own tasks. This is where most 2026 reasoning and agent skill comes from. | Kimi K3: 9 specialists · GLM-5: >10K software environments |
| 6 | Stage 5 `active`; the three specialists flow into one block. Stop link: `distillation`. | Three `flow`s converge into one model. | On-policy distillation merges the specialists back into one model that keeps each one's best skills. | DeepSeek-V4: > 10 teachers · Kimi K3: 9 |
| 7 | Stage 6 `active`. Labels `style and safety (judge-model RL)` and `prepared for serving`. Stop links: `rlhf-dpo` (polish), `quantization` (serving). | The block lights; the labels type in. | A final polish tunes style and safety, often with RL against a judge model, and the weights are prepared for serving. | Nemotron 3 Super: separate final RLHF stage · DeepSeek-V4, Kimi K3: FP4-aware training in post-training |
| 8 | **Key frame.** A `shareBar` under the row for GLM-5: `pretrain 94.6% · mid-train 5.4% · post-training not published` (hatched). | Segments slide in under their stages. | By tokens, pretraining is nearly everything: 94.6% of GLM-5's published budget. Post-training token counts are mostly not published. | `27T / 28.55T = 94.6%` · `1.55T / 28.55T = 5.4%` |
| 9 | Same bar; a plain line beside the post-training segment: `DeepSeek-V3.2: post-training > 10% of pretraining compute`. Stop links: `gpu-primer`, `scale-reliability`. | The line types in; the hatched segment pulses once. | By compute, post-training is no longer small: over a tenth of pretraining for DeepSeek-V3.2. Each RL token is generated and scored before it teaches anything. | `> 10%` (DeepSeek-V3.2) · Mistral: ~33B RL tokens per day on ~3k GPUs |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frame 8 draws `budgetShares` (`math/pipeline.js`). Caption counts (words/sentences, `rlvr-grpo`'s counter, run 2026-10-07): 24/2, 23/2, 23/1, 18/2, 26/2, 15/1,
21/1, 18/2, 25/2.

## 6. Toy
"Read a recipe." Pick a 2026 model; see which of the six stages its report describes, what it says about
each in its own units, and its token budget by stage.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `model` | Whose recipe | Preset chips | `GLM-5` / `DeepSeek-V4` / `Kimi K3` / `Nemotron 3 Super` / `Olmo 3 (reported)` | `GLM-5` | the chips |
| `stage` | Inspect a stage | Selection: click, tap or arrow keys on a stage block | stages 1–6 | 1 | – |

**Live outputs** (printed; nothing behind hover)
| Output | Formula / source | Units / format |
|---|---|---|
| Stage strip: lit if the report describes the stage, `dim` with `not described` if not | preset data (§8 keys) | – |
| Inspector for the selected stage: the published figure in its own unit, and the stop link | preset data | tokens, examples, pairs, prompts, environments or teachers, as published |
| Token `shareBar` | `budgetShares(parts)` | percent, 1 decimal; hatched `not published` |
| Known tokens total | `budgetShares(parts).knownTotal` | T tokens |

Preset contents (from §8; `—` = not described in the brief):
- GLM-5: pretrain 27T · mid-train 1.55T (4K → 200K) · SFT (rejection sampling, masked errors) · RL
  reasoning → agentic (>10K SWE environments) → general · merge: cross-stage distillation from its own
  earlier checkpoints · polish: general RL with rule + ORM + GRM rewards.
- DeepSeek-V4: pretrain 33T with the context schedule inside it (mid-train not separately counted) ·
  SFT · specialist RL with GRPO · merge: > 10 teachers · polish: FP4-aware training.
- Kimi K3: pretrain not disclosed · long-context cooldown 256K → 1M · SFT cold start · 9 specialists ·
  merge: MOPD · FP4-aware training from SFT on.
- Nemotron 3 Super: pretrain 25T · RLVR (21 environments) → SWE-RL → RLHF with a judge model · merge: —.
- Olmo 3 (reported): pretrain ~5.9T · mid-train 100B · SFT ~2.3M traces · DPO ~200K pairs · RLVR ~105K
  prompts · merge: —.

**Try this** (each leads to a named insight)
1. Step through the five presets and watch the `merge` stage: lit for GLM-5, DeepSeek-V4 and Kimi K3, `not
   described` for Nemotron 3 Super and Olmo 3. → **Insight: merging specialists is the 2026 frontier
   step,** used by the largest open MoE labs; smaller and fully open pipelines often stop at one RL stage
   (`distillation` explains the merge).
2. With `GLM-5`, read the bar: pretrain 94.6%, mid-train 5.4%, post-training hatched. Then select stage 4
   and read its unit: environments, not tokens. Switch to `Olmo 3`: 98.3% / 1.7%, and its post-training
   stages are counted in traces, pairs and prompts. → **Insight: tokens are the wrong ruler for
   post-training.** It is reported in examples, environments and compute, because each of its tokens costs
   far more than a pretraining token.
3. Select `Kimi K3`: the bar is entirely hatched, yet every stage is lit. → **Insight: a recipe can be
   fully described without its budget.** Labs often publish the method but not the amounts, which is why
   the course labels unknowns rather than guessing.

Lesson-17 check: the token shares are `budgetShares` on published figures only; the toy never estimates an
unpublished post-training token count, so it cannot make post-training look smaller or larger than it is.

## 7. Show me the math
```tex
\text{share}_i = \frac{T_i}{\sum_{j\,\in\,\text{published}} T_j}
\qquad
\text{GLM-5: } \frac{27}{27 + 1.55} = 94.6\%,\ \ \frac{1.55}{28.55} = 5.4\%
```
```tex
\text{training compute} \approx 6\,N_{\text{active}}\,D \quad (\text{taught on } \texttt{scaling-laws})
```
Panel notes: (a) shares are of the *published* total only; an unpublished stage is shown, hatched, never
estimated. (b) Why an RL token costs more than a pretraining token: it is first generated one token at a
time (decode, `prefill-decode`), then scored by a checker, sandbox or judge, and only then trained on.
No formula is given because the ratio depends on the run. Color links: none (no live equation terms).

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Canonical 2026 pipeline: pretrain → mid-train → SFT (cold start) → specialist RL → merge by OPD → final polish | mechanics, from the brief's synthesis (no data key) | 02 §5 |
| 25–33T pretraining tokens for open frontier MoEs; DeepSeek-V4-Pro 33T | `models.deepseek-v4-pro.pretrain_tokens` (existing); others proposed on `pretraining` | 02 §0, §1.8 |
| GLM-5: 27T base + 1.55T mid-training (4K → 200K) | `models.glm-5.context_stages` (proposed on `midtraining`) | 02 §1.8, §2 |
| DeepSeek-R1 ~800K SFT examples; Olmo 3 ~2.3M traces (reported) | proposed on `distillation` / `sft` | 02 §3 |
| Kimi K3: 9 specialists; GLM-5: > 10K SWE environments | proposed on `agentic-rl` | 02 §4.6 |
| DeepSeek-V4 > 10 teachers; Kimi K3 9; GLM-5 cross-stage | proposed on `distillation` | 02 §4.8 |
| Nemotron 3 Super: RLVR (21 environments) → SWE-RL → separate RLHF stage with a judge model | proposed on `rlhf-dpo` / `agentic-rl` | 02 §4.1, §4.6 |
| DeepSeek-V4 and Kimi K3 use FP4 quantization-aware training in post-training | `models.deepseek-v4-pro.post_training_qat` = "MXFP4 (experts, indexer QK)", `models.kimi-k3.post_training_qat` = "MXFP4 weights + MXFP8 activations, from SFT" (proposed, confirmed) | 02 §1.7 item 2 |
| DeepSeek-V3.2: post-training compute above 10% of pretraining | `models.deepseek-v3.2.post_training_compute_share` (proposed by `rlvr-grpo`) | 02 §4.3, §5 |
| Mistral: ~33B RL tokens per day (~16B trainable) at ~3k GPUs | proposed on `agentic-rl` | 02 §4.7, §6 |
| Olmo 3 (reported): ~5.9T pretrain, 100B mid-train, DPO ~200K pairs, RLVR ~105K prompts | proposed on `scaling-laws`, `midtraining`, `rlhf-dpo` | 02 §1.8, §2, §4.2 |

Not on this page, by design: number formats and memory (`gpu-primer`, `training-memory`); which Nemotron 3
model pretrained in NVFP4 (a known conflict, spec §7).

## 9. Takeaways
1. A 2026 model is trained in six stages, each starting from the last one's weights: pretraining,
   mid-training, SFT, specialist RL, merging by on-policy distillation, and a final polish.
2. Pretraining gives knowledge; SFT gives format; specialist RL gives most of the reasoning and agent skill;
   distillation merges the specialists into one model.
3. Pretraining is nearly all the tokens, but post-training is no longer cheap in compute (over 10% of
   pretraining for DeepSeek-V3.2), because every RL token is generated and scored before it teaches.

## 10. Next and go deeper
Next: `pretraining`. Then, in order: `scaling-laws`, `midtraining`, `sft`, `rlhf-dpo`, `rlvr-grpo`,
`agentic-rl`, `distillation`.
Go deeper: Andrej Karpathy, *nanochat*, a runnable tokenizer → pretrain → midtrain → SFT → RL pipeline
(https://github.com/karpathy/nanochat) · Hugging Face, *The Smol Training Playbook*
(https://huggingfacetb-smol-training-playbook.hf.space/)

## 11. Key-frame sketch
Frame 8 (token shares), desktop width. Checked against the reproducer below.

```text
 [1 pretrain]→[2 mid-train]→[3 SFT]→[4 RL ×3]→[5 merge]→[6 polish]
   ▲ outlined
 GLM-5 published tokens (28.55T)
 ┌──────────────────────────────────────────┬──┬░░░░░░┐
 │ pretrain 94.6%                           │5.4│░n/p░│
 └──────────────────────────────────────────┴──┴░░░░░░┘
   27T                                   1.55T  post-training
                                                 not published
 By tokens, pretraining is nearly everything: 94.6% of GLM-5's
 published budget. Post-training token counts are mostly not
 published.
 [Back] [Pause] [Next]  ━━━━━━━●━━  8 / 9   1×
```

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against a scratchpad draft of
`math/pipeline.js` matching the signature):
```sh
node -e '
import("./math/pipeline.js").then((p)=>{
  const P={"glm-5":[["pretrain",27e12],["mid-train",1.55e12],["post-training",null]],"olmo-3":[["pretrain",5.9e12],["mid-train",0.1e12],["post-training",null]],
    "deepseek-v4":[["pretrain incl. mid-train",33e12],["post-training",null]],"nemotron-3-super":[["pretrain",25e12],["post-training",null]],"kimi-k3":[["pretrain",null],["mid-train",null],["post-training",null]]};
  for (const [k,v] of Object.entries(P)) { const r=p.budgetShares(v.map(([name,value])=>({name,value})));
    console.log(k,r.knownTotal,r.unknownCount,r.parts.map((q)=>q.name+":"+(q.share==null?"n/p":(100*q.share).toFixed(1)+"%")).join(" ")); }
});'
```
Output (2026-10-07): GLM-5 28.55e12, 1 unknown, pretrain 94.6%, mid-train 5.4% · Olmo 3 6.0e12, 98.3% /
1.7% · DeepSeek-V4 33e12, 100% (mid-training inside) · Nemotron 3 Super 25e12, 100% · Kimi K3 knownTotal 0,
3 unknown, all shares null.

**`math/pipeline.js` signature** (owner: `training-pipeline`; pure, no DOM; tests first; reused by
`midtraining` for context stages):
```js
// Shares of a budget among named parts; value null = not published. Shares are of the known total only;
// every share is null when nothing is published.
budgetShares(parts: { name: string, value: number | null }[]) →
  { knownTotal: number, unknownCount: number, parts: { name, value, unknown: boolean, share: number | null }[] }
//   [27e12, 1.55e12, null]            → knownTotal 2.855e13, shares [0.9457, 0.0543, null]
//   [27e12, 1e12, 0.5e12, 0.05e12]    → shares [0.9457, 0.0350, 0.0175, 0.0018]   (midtraining)
//   [null, null, null]                → knownTotal 0, unknownCount 3, shares all null
//   throws RangeError on an empty list or a negative value
```
Test cases for the builder: known shares sum to 1 whenever `knownTotal > 0`; unknown parts never change the
known shares; the input array is not mutated.

## 12. Open questions for the reviewer
**Data-pass keys** (new): `deepseek-v4-pro.post_training_qat`, `kimi-k3.post_training_qat` (confirmed, 02
§1.7). All other keys on this page are proposed on the stage pages and listed there.

**Graph changes:** none on this page. (`midtraining` is a leaf; its §12 discusses whether `sft` should list
it.)

**Judgment calls:**
- **Stops into other tracks.** Frame 7 links `quantization` (Serving) and frame 9 links `gpu-primer` and
  `scale-reliability` (GPUs & scale) as "see also", not as `Next:`; the header's `Next:` stays
  `pretraining` per README lesson 1.
- **`blockStack` in frame 1** depends on `decoder-anatomy`'s glyph proposal; if it is not accepted, frame 1
  uses a single `block` "the decoder".
- **`shareBar` unknown-segment width** (fixed 12%, outside the scale) is a usage rule this page needs; it
  should be agreed with `decoder-anatomy`'s author, who proposed the glyph.
- **Illustrative replies** to `What is 7 × 8?` are not model outputs; labeled on screen.
