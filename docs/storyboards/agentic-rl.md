# Agentic RL (`agentic-rl`)

Track: training · Section: recipe · Prereqs: rlvr-grpo
Next: `distillation` (the one slug whose `prereqs` list `agentic-rl`)
Status: approved (expert review)
Sources: 02 §0 items 6–8, §3 (loss masking), §4.3, §4.4 (IcePop, DeepSeek-V3.2 patches, consensus list),
§4.5, §4.6, §4.7, §5 step 4, §6 · 05 §1.1, §1.2

Running example, continued from `rlvr-grpo`: prompt `7 × 8 = ?`, a group of G = 8, the checker "is the
last token `56`?", and the default group k = 2 built with `rlvr-grpo`'s pools and slot order
`[1, 5, 3, 7, 2, 6, 4, 8]`, so rows 1 and 5 are right and the advantages are +1.73 / −0.58 (39 tokens).
Frames 1–3 replay row 1 as a tiny tool-using episode drawn beside the table (the table itself stays
`rlvr-grpo`'s eight answers, 39 tokens, everywhere on this page); frames 5–8 and the toy reuse the table
row for row. A visible line under the stage says: "Real agent episodes fix code, run terminals or search
the web over hundreds of tool calls; this one stays tiny so every token fits on screen."

Contract with `rlvr-grpo` (its §1, §9 takeaway 3 and §8 last row): this page demonstrates the two patches
deferred to it, **importance-sampling corrections for the sampler-vs-trainer gap** (frames 7–8, try-this
1–2) and **little or no KL penalty** (frame 9). It imports `math/grpo.js` unchanged and adds nothing to it.
`r = now / when sampled` (the policy-change ratio of `rlvr-grpo` frame 8) and this page's `ρ = trainer /
engine` (the mismatch ratio) are different quantities; frame 7 says so on screen.

## 1. Learning objective
After this page you can explain how RL trains a model to act over many steps: an episode alternates the
model's own tokens with observations from a sandbox, only the model's tokens are trained, and the reward
comes from tests or, when nothing can be tested, from a judge model with a rubric; why such episodes force
asynchronous training (waiting for the longest episode leaves most generators idle), and why that creates
off-policy data plus a gap between the engine that sampled each token and the trainer that updates it; how
importance-sampling corrections handle that gap (full reweighting, truncation, or IcePop's masking); and why
most 2026 open-model reports run with little or no KL term.

## 2. Misconceptions to correct
- **Misconception:** "The model is trained on everything in the episode, tool outputs included." →
  **Reality:** observation tokens come from the environment, not the policy, so they are masked out of
  the loss, exactly as prompts are in SFT; here 8 of 11 tokens are trained. Corrected by frame 2.
  (source: 02 §3 "environment/tool outputs are also excluded from the loss in agentic RL", GLM-5,
  confirmed)
- **Misconception:** "If two programs run the same weights, they assign the same probability to a token."
  → **Reality:** the rollout engine (vLLM/SGLang) and the trainer (Megatron/FSDP) use different kernels,
  rounding and MoE routing, so for the same token they disagree: in the toy, 0.050 vs 0.160 for row 4's
  `48` (ρ = 3.2). Corrected by frame 7. (source: 02 §4.7 "compute different probabilities for the same
  tokens")
- **Misconception:** "Importance sampling fixes the mismatch exactly, so you should always reweight by
  the full ratio." → **Reality:** full reweighting is unbiased but lets one token carry 3.2× its push.
  Truncated IS caps the weight (at 2 in this toy); IcePop drops tokens whose two probabilities differ more than
  twofold. Each trades a little bias for much less variance. Corrected by try-this 1. (source: 02 §4.4
  IcePop; §4.7 TIS/MIS)
- **Misconception:** "Dropping the KL term means nothing limits how far the policy moves." → **Reality:**
  each step is still bounded by the clip and the IS masks; what disappears is the pull back toward the SFT
  starting point. A checker is much harder to game than a learned reward model, so that pull matters less,
  and dropping it frees the reference copy from memory. The cost: if the checker has a hole, nothing pulls
  the policy back. Corrected by frame 9. (source: 02 §4.4 GLM-5 "KL term removed"; §4.4 consensus "no or
  tiny KL"; §7 KL row "many 2026 recipes set β ≈ 0 and rely on clipping/masking")
- **Misconception:** "Agentic RL needs a new algorithm." → **Reality:** the update is still GRPO-family:
  the same group, the same +1.73 / −0.58 advantages (frame 3). What changes is everything around it:
  environments, judges, asynchronous rollouts and the corrections they force. Corrected by frames 3 and
  5–8. (source: 02 §4.4 adoption table; §4.6)

## 3. Hook and intuition (final wording)
**Hook:** A coding agent works for an hour, makes 200 tool calls, and then the tests pass or fail. How do
you train on that, and what breaks when you try?

An agentic episode is a long conversation between the model and a sandbox. The model thinks, writes a
tool call, the sandbox runs it (a test suite, a shell command, a web search), and the result comes back as
an **observation**: new tokens in the context that the model did not write. That repeats until the task
ends, and then the environment scores the outcome, for example "the failing test now passes". The model's
own tokens are trained; the observations are masked out, because the model should learn what to *do*,
not to predict what the sandbox will print. When nothing can be tested (a research summary, a slide
deck), a **generative reward model** steps in: a judge model writes a rubric for the task and scores each
attempt against it.

The update itself is the GRPO you already know: a group of episodes per task, each one's advantage
measured against its siblings. The trouble is time. Episodes in a group can take minutes or hours, and a
synchronous trainer waits for the slowest one while the other generators sit idle. So 2026 systems run
**asynchronously**: the trainer updates as soon as enough episodes are done, and the rest finish later
under newer weights. Those late tokens are **off-policy**, sampled by a slightly older model.

There is a second gap even without staleness. The engine that generates the tokens and the trainer that
updates on them are different programs with different kernels, rounding and expert routing, so they
disagree about a token's probability. Training as if they agreed adds bias, and a few tokens where they
disagree badly can dominate an update. The fix is an **importance-sampling correction**: weight each
token by how much likelier the trainer finds it than the engine did, cap that weight (truncated IS), or
drop the token when the two differ by more than a factor of two (IcePop, GLM-5). With the clip and these
masks already bounding every step, and a checker that is hard to game, most 2026 open-model reports set the KL term
to zero or near it, which also frees the reference model's memory.

This is where most RL compute goes in 2026: GLM-5 trains on more than 10,000 verifiable software-engineering
environments, DeepSeek-V3.2 synthesized 1,827 environments, and Kimi K3 trains nine separate experts. The
next page, `distillation`, shows how those experts are merged back into one model.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `token` (episode and answer chips; `fill` =
`valueColor(advantage)`, `hatched` = masked out of the loss or gradient off), `block` (policy, sandbox
environment, checker, judge, trainer, engine, reference; `active` vs `dim`), `flow` (carry `token` from
policy to sandbox and back; carry `gradient` from the A column into chips), `vector` (`NUMBER_CELL` columns
R and A for the group, as on `rlvr-grpo`), `cell` (engine and trainer probabilities, printed), `verdict`
(✓/✗ per row), `request` (one bar per episode on a shared time axis: the prefill segment is the prompt, the
decode ticks are generation time; used here as an episode timeline), `clipLine` (frame 8, band
[0.5, 2.0]), plain labeled text marks (`idle`, `cut here: 6 of 8 done`, `observation: masked`, rubric
lines, `β = 0`).

New glyphs proposed: none. Note: `request` is reused as an episode timeline. Its two segments mean
"prompt" and "generated tokens"; on this page the prefill segment is drawn at width 0 and each tick is one
minute of the episode. Ruled (expert review): reuse `request`; no `rollout` alias.

Stand-ins (visible line under the stage, frames 4–8): "Episode durations, rubric scores and the engine
and trainer probabilities are hand-picked stand-ins; the arithmetic on them is exact."

Terms introduced (one per frame): environment / sandbox (1) · observation (2) · none new, the group
returns (3) · generative reward model and rubric (4) · synchronous rollout (5) · off-policy data (6) ·
training/inference mismatch, ratio ρ (7) · importance-sampling correction (8) · none new (9: β ≈ 0 for
the KL term) · none new (10).
Terms assumed from prereqs: policy, reward model, reference model, KL term (`rlhf-dpo`); group,
advantage, checker, push, ratio r and clip (`rlvr-grpo`); loss masking (`sft`).

Indexing: rows 1–8 and token positions are 1-based on screen; no 0-based addresses on this page.

Selection: row 4's second `48` is the followed token in frames 7–8 and the toy's default selection; it
carries the selection outline wherever it appears. In frames 1–2 the followed item is the episode's final
`56`.

## 5. Animation script
Thread order: frames 1–4 what an agentic episode is and how it is scored; 5–8 why training goes
asynchronous and the correction that requires (patch 1); 9 the KL term (patch 2); 10 scale and cost.

Episode used in frames 1–3 (row 1 of the group, as a tool episode; whitespace-split):
policy turn 1 `<think> use python </think> <call> print(7*8) </call>` (7 tokens) · observation
`<obs> 56 </obs>` (3 tokens) · policy turn 2 `56` (1 token). 8 policy tokens, 3 observation tokens.

Episode durations for frames 5–6 and the toy (rows 1–8, minutes): `[3, 2, 4, 16, 5, 3, 9, 6]`.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Prompt chips `7 × 8 = ?`. Policy `block` at the left, a `block` "sandbox (runs tools)" at the right. Turn 1 chips type in left to right. | A `flow` (carry `token`) carries the `<call> print(7*8) </call>` chips to the sandbox; the sandbox lights; a `flow` returns with `<obs> 56 </obs>`; turn 2 `56` types in (selection outline). | In agentic RL the model acts inside an environment. It writes a tool call, a sandbox runs it, and the result comes back into its context. | 2 policy turns · 1 tool call |
| 2 | The same 11 chips in one row. The three observation chips take the hatch with the plain label `observation: masked`. Readout `trained 8 of 11`. | The hatch sweeps across the three observation chips; the readout counts. | The sandbox's reply is an observation: the model did not write it, so it is masked out of the loss. Only the model's own 8 tokens will be trained. | 8 policy tokens · 3 observation tokens · trained 8 / 11 |
| 3 | The group table from `rlvr-grpo`, unchanged (8 rows, 39 tokens, `verdict`s, R and A columns). The episode strip from frame 2 sits above it with the plain label `an agent version of row 1: same final 56`. | The checker stamps ✓/✗ down the rows; R fills; A fills +1.73 / −0.58; a `flow` (carry `gradient`) runs from row 1's A cell up into the episode's 8 policy chips, skipping the hatched observation chips. | Episode or plain answer, each of the eight is scored and compared exactly as on the previous page. The update is still GRPO; everything around it changes. | `R = [1, 0, 0, 0, 1, 0, 0, 0]` · `A = +1.73` (rows 1, 5) / `−0.58` · Σ A = 0 |
| 4 | Branch label `when nothing can be tested`. A `block` "judge model" writes three plain rubric lines: `cites a source`, `answers the question asked`, `under the length budget`. Two attempt rows with scores `3 / 3` and `1 / 3`. | Rubric lines type in; each attempt row gets a tick per line met; the scores print. | When nothing can be tested, a generative reward model writes a rubric for the task and scores each attempt against it. DeepSeek-V4 uses the policy itself as judge. | rubric 3 items · attempt A `3 / 3` · attempt B `1 / 3` (stand-ins) · visible note: "Kimi K3's judge makes an answer over its length budget lose automatically" |
| 5 | Back on the main line. A time axis 0–16 min. Eight `request` bars (rows 1–8), lengths 3, 2, 4, 16, 5, 3, 9, 6. A `block` "trainer" at the right, `dim`. Plain label `idle` on the empty space after each bar. | Bars grow left to right together; each one stops at its length and its remaining space greys out as `idle`; the trainer lights only at minute 16. | Synchronous training waits for the slowest episode before it updates. Here one 16-minute episode leaves the other generators idle most of the time. | iteration 16 min · busy 48 of 128 slot-minutes · utilization `48 / (8 × 16) = 37.5%` · idle 80 |
| 6 | Same axis. A vertical plain mark `cut here: 6 of 8 done` at minute 6. Bars 4 and 7 continue past the cut, each with the plain label `finishes under newer weights`. | The cut line drops at minute 6; the trainer lights there; bars 4 and 7 keep growing past the line into the next iteration. | Asynchronous training updates once most episodes are done; the rest finish under newer weights. Their tokens are off-policy: sampled by a slightly older model than the one being trained. | cut at 6 of 8 (λ = 0.75) · iteration 6 min · busy 35 of 48 · utilization `72.9%` · carried: rows 4 and 7 (off-policy: the part of each sampled before the cut) |
| 7 | Zoom on row 4's second `48` (selection outline). Two `block`s: `engine (samples)` and `trainer (updates)`, each with a printed `cell`: `0.050` and `0.160`. Readout `ρ = trainer / engine = 3.2`. A plain note: `ρ is not the r of the last page: same weights, different programs.` | The engine cell fills first, then the trainer cell; ρ types in. | Even with identical weights, the sampling engine and the trainer compute different probabilities for the same token. Here the trainer finds this 48 more than three times likelier. | engine `0.050` · trainer `0.160` · `ρ = 3.20` · visible note: `most tokens agree within a few percent (34 of 39 here); a few outliers like this one are what the corrections are for` |
| 8 | **Key frame.** A `clipLine` from 0 to 3.5, band [0.5, 2.0] shaded, label `ρ = trainer / engine`, marker at 3.20, and under the band the plain line `same factor either way: ½ and 2`. Below it, collapsed per README lesson 18: rows 4, 6 and 7 in full with their R and A, three chips hatched (row 4's second `48` 3.20, row 6's `63` 0.40, row 7's `58` 2.40), then one plain line `5 other rows: ρ within [½, 2], unmasked`. Readout `masked 3 of 39`. Layout at 580 × 366: clipLine block 56 px + 3 rows × 34 px + collapsed line 20 px + readout 20 px ≈ 200 px tall; the longest row (row 4, 8 chips + R + A) ≈ 430 px wide. | The marker lands at 3.20 outside the band; the chip's hatch appears; then the two other out-of-band chips hatch in turn; the readout counts to 3. | An importance-sampling correction reweights each token by that ratio, caps it, or drops it. IcePop, used by GLM-5, masks any token whose two probabilities differ more than twofold. | band `[1/2, 2]` (β = 2) · row 4's `48`: full IS weight `3.20`, truncated at 2 `2.00`, IcePop `masked` · push `−0.58 → −1.85 / −1.15 / 0` · masked 3 of 39 |
| 9 | The two shelves from `rlvr-grpo` frame 7, left one only: `policy` + `checker` (a program) + `reference` drawn `dim` with the plain label `KL term β = 0: not loaded`. A readout `models in memory 2 → 1`. | The reference block fades from `active` to `dim`; the readout counts down. | Most 2026 open-model reports drop the KL term or make it tiny. Clip and IS masks already bound each step, checkers are hard to game, and the reference leaves memory. | `β = 0` (GLM-5, Olmo 3) · "weak or zero for math" (DeepSeek-V3.2) · models in memory 2 → 1 |
| 10 | Plain text table: `GLM-5: > 10,000 SWE environments, 9 languages` · `DeepSeek-V3.2: 1,827 environments, 4,417 tasks` · `Kimi K3: 9 experts (3 domains: general, general agents, coding agents × low / high / max effort)` · `Nemotron 3 Super: 21 environments` · `Mistral: ~33B tokens produced per day by one run on ~3k GPUs, ~16B of them trainable completion tokens`. | The rows type in top to bottom. | Agentic RL is now where most RL compute goes: thousands of sandboxes, long episodes, and separate specialist models. Distillation, next, merges those specialists into one. | as listed (§8) |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 3 and 8 draw `buildGroup(2)` + `groupAdvantages()` from `math/grpo.js`; frames 5–6 draw
`rolloutSchedule(durations, { mode })`; frame 8 draws `isCorrection(ρ, { mode })`. Caption counts
(words/sentences, `rlvr-grpo`'s counter widened to two-digit frames, run 2026-10-07): 26/2, 29/2, 27/2, 28/2, 23/2, 29/2, 28/2, 28/2, 30/2, 25/2 (re-run after review: unchanged).

## 6. Toy
"Fix the mismatch." The `rlvr-grpo` default group (k = 2), with the sampling engine's and the trainer's
probability for every token, and the eight episodes on a time axis. One state, two linked panels: the group
table (with a token inspector) and the rollout timeline.

**Seeded data** (hand-authored constants in `training/concepts/agentic-rl.js`). Every token was sampled by
the engine at its probability and the trainer agrees (ρ = 1.00) except five tokens, given as (engine →
trainer under BF16):
row 1 `56`: 0.200 → 0.220 (ρ 1.10) · row 5 `56`: 0.200 → 0.180 (0.90) · row 4 second `48`: 0.050 → 0.160
(3.20) · row 6 `63`: 0.200 → 0.080 (0.40) · row 7 `58`: 0.050 → 0.120 (2.40).
Under FP16 the toy shrinks each log ρ by 8× (`mismatchRatio`), because FP16 rounds about 8× finer than
BF16 (number formats are taught on `gpu-primer`). Visible line under the toy: "The probabilities are
hand-picked stand-ins. The FP16 setting models rounding only; mismatch from MoE routing does not shrink
with precision (DeepSeek's fix for that is Keep Routing)."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `correction` | How the trainer treats the mismatch | Preset chips | `ignore` (weight 1) / `full IS` (weight ρ) / `truncated IS` (weight min(ρ, 2)) / `IcePop` (mask outside [½, 2]) | `ignore` | the chips are the control |
| `precision` | Rollout and trainer number format | Toggle | `BF16` / `FP16` (the chip itself reads `FP16: toy model of rounding only (reported fix)`) | `BF16` | – |
| `lambda` | Update when this share of episodes is done | Preset chips | `all 8 (sync)` / `6 of 8` / `4 of 8` | `all 8` | – |

Selection, not hover: a token is selected by click, tap or arrow keys; default row 4's second `48`. A
visible **inspector** shows the selected token's row, A, engine p, trainer p, ρ, IS weight, masked yes/no,
and its push A × weight.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| R, A per row | `groupAdvantages(buildGroup(2) → verifyFinalAnswer)` (`math/grpo.js`, unchanged) | as on `rlvr-grpo` |
| ρ per token (inspector) | `mismatchRatio(ρ_bf16, { precision })` | 3 decimals |
| IS weight, masked | `isCorrection(ρ, { mode: correction, cap: 2, band: [0.5, 2] })` | 3 decimals; hatch when masked |
| Push per token (inspector) | `A × weight` | signed, 3 decimals |
| Masked tokens | count of `masked` over the 39 tokens | `n of 39` |
| Timeline: iteration length, utilization, carried rows | `rolloutSchedule(durations, { mode, lambda })`, with utilization = busy slot-time ÷ (8 × iteration length), the page's one definition | minutes; percent, 1 decimal; row numbers |

**Try this** (each leads to a named insight)
1. Predict first: should the trainer fully trust the ratio? With row 4's `48` selected, step through
   `correction`: `ignore` → weight 1, push −0.577; `full IS` → weight 3.200, push −1.848; `truncated IS`
   → 2.000, −1.155; `IcePop` → masked, 0, and the readout says 3 of 39 masked. → **Insight: corrections
   trade bias for variance.** Ignoring the gap is biased; full reweighting is unbiased but lets one
   token push 3.2× as hard; truncation caps it (at 2 in this toy), and IcePop simply drops tokens where engine and trainer
   disagree more than twofold.
2. Keep `IcePop` and switch `precision` to FP16. Row 4's ρ falls 3.20 → 1.157, row 7's 2.40 → 1.116, row
   6's 0.40 → 0.892, and the readout drops to 0 of 39 masked. → **Insight: part of the mismatch is
   rounding,** and a finer number format shrinks it before any correction is needed (a reported 2025 fix).
   The rest, such as MoE routing that differs between engine and trainer, needs its own fix (Keep Routing).
3. Step `lambda` from `all 8` to `6 of 8` to `4 of 8`: utilization 37.5% → 72.9% → 87.5%, while the
   rows that finish under newer weights go from none to rows 4 and 7 to rows 4, 5, 7 and 8. → **Insight:
   async is a trade.** The less the trainer waits, the more of each batch is off-policy, and the more work
   the corrections from try-this 1 have to do. (Kimi K3's partial rollouts pause at a fraction λ of
   finished episodes and resume the rest next iteration.)

Lesson-17 check: all three insights read directly off `isCorrection`, `mismatchRatio` and `rolloutSchedule`
at the stated settings (reproducer below). The FP16 insight would reverse only if routing mismatch were
modeled, which the visible line says it is not.

## 7. Show me the math
```tex
\htmlClass{hl-rho}{\rho_{i,t}} = \frac{\pi^{\text{train}}_{\text{old}}(o_{i,t} \mid \cdot)}{\pi^{\text{engine}}_{\text{old}}(o_{i,t} \mid \cdot)}
\qquad
w_{i,t} = \begin{cases}
1 & \text{ignore} \\
\rho_{i,t} & \text{full IS} \\
\min(\rho_{i,t}, C) & \text{truncated IS (TIS)} \\
\mathbb{1}\big[\tfrac{1}{\beta} \le \rho_{i,t} \le \beta\big] & \text{IcePop } (\beta = 2)
\end{cases}
```
```tex
\mathcal{J} = \frac{1}{\sum_i |o_i|_{\text{policy}}} \sum_{i=1}^{G} \sum_{t \in \text{policy tokens}}
w_{i,t}\, \min\!\big( r_{i,t} \htmlClass{hl-a}{A_i},\ \operatorname{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\, \htmlClass{hl-a}{A_i} \big)
\;-\; \underbrace{\beta_{\text{KL}}}_{\approx\, 0}\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
```
```tex
\text{utilization} = \frac{\sum_i \min(\ell_i, T)}{G \cdot T},\qquad
T = \ell_{(\lceil \lambda G \rceil)} \;(\text{the } \lceil \lambda G\rceil\text{-th shortest episode})
```
Shapes: one ρ and one w per policy token (39 in the default group; observation tokens have no term at all,
which is the masking of frame 2); A ∈ ℝ^G; ℓ ∈ ℝ^G episode durations. Panel notes: (a) r (policy change,
`rlvr-grpo`) and ρ (engine vs trainer) multiply in practice; this page holds r = 1 to isolate ρ. (b) GLM-5's
agentic RL uses "direct double-sided importance sampling" (a token-level clip on rollout log-probs) and a
token-in-token-out gateway so the trainer sees exactly the engine's tokens; DeepSeek-V3.2 adds off-policy
sequence masking, Keep Routing and Keep Sampling Mask. (c) Precision formats themselves are taught on
`gpu-primer`. Color links: `hl-rho` = the clipLine marker and the inspector's ρ (frames 7–8); `hl-a` = the A
column and chip fills (frame 3).

## 8. In today's models (Oct 2026)
All keys are proposed for the data-extension pass unless marked existing; entry ids as accepted in
`rlvr-grpo` §13.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GLM-5: fully async, decoupled rollouts and training (slime); TITO gateway; direct double-sided importance sampling; more than 10K verifiable SWE environments across 9 languages, thousands of Docker terminal tasks, multi-hop search over >2M pages | `models.glm-5.rl_infra` = "async decoupled (slime), TITO, double-sided IS", `.agentic_envs_swe` = ">10,000 (9 languages)" (confirmed) | 02 §4.6 |
| GLM-5 masks tokens whose train/infer ratio leaves [1/2, 2] (IcePop); KL term removed | `models.glm-5.rl_mismatch_fix` = "IcePop mask, ρ ∈ [1/2, 2]", `.rl_kl_coeff` = 0 (both proposed by `rlvr-grpo`, confirmed) | 02 §4.4 |
| DeepSeek-V3.2: weak or zero KL for math with an unbiased KL estimator; off-policy sequence masking; Keep Routing; Keep Sampling Mask; 1,827 environments / 4,417 tasks; post-training above 10% of pretraining compute | `models.deepseek-v3.2.rl_patches` = "weak/zero KL (math), off-policy sequence masking, Keep Routing, Keep Sampling Mask" (confirmed); `.rl_environments` = 1827, `.post_training_compute_share` (proposed by `rlvr-grpo`) | 02 §4.4, §4.3, §4.6 |
| DeepSeek-V4: preemptible rollout service with a token-level write-ahead log, because regenerating interrupted requests from scratch biases toward short responses; the policy itself is the generative judge | `models.deepseek-v4-pro.rl_infra` = "preemptible rollouts, token-granular WAL", `.reward_model` = "GRM, actor as judge" (confirmed) | 02 §4.6, §4.5 |
| Kimi K3: 9 experts (3 domains × 3 effort levels); partial rollouts that pause at a finished fraction λ and resume next iteration in persistent microVM sandboxes; agentic GRM with a generated rubric and a verbosity budget | `models.kimi-k3.rl_experts` = 9, `.rl_rollouts` = "partial rollouts (λ), AgentEnv microVMs", `.reward_model` = "agentic GRM: rubric + scorepad, verbosity budget" (confirmed) | 02 §4.6, §4.5 |
| Nemotron 3 Super: asynchronous GRPO over 21 environments; a separate SWE-RL stage because SWE rollouts are slow and long | `models.nemotron-3-super.rl_algorithm`, `.rl_environments` (proposed by `rlvr-grpo`); `.swe_rl_stage` = true (confirmed) | 02 §4.4, §4.6 |
| Olmo 3: no KL term (with clip-higher, token-level loss, zero-gradient filtering) | `models.olmo-3.rl_patches` (proposed by `rlvr-grpo`, reported) | 02 §4.4 |
| Mistral Large 4: async RL, ~33B tokens per day (~16B trainable completion tokens) at ~3k GPUs | `models.mistral-large-4.rl_tokens_per_day` = 33e9, `.rl_trainable_tokens_per_day` = 16e9, `.rl_gpus` = 3000 (confirmed, Mistral blog) | 02 §4.7, §6 |
| MiniMax-M2: dense process rewards (language mixing, tool-format errors) and a completion-time reward; "hundreds of thousands of real-world environments" is marketing for M2.5 | `models.minimax-m2.rl_rewards` = "process + completion-time + task" (confirmed); `.agentic_envs` = "hundreds of thousands (M2.5 marketing)" (reported) | 02 §4.5, §4.6 |
| Using FP16 instead of BF16 shrinks the rounding mismatch | `papers.fp16-mismatch-2025` (arXiv 2510.26788) (reported) — no model entry | 02 §4.7 |

## 9. Takeaways
1. An agentic episode alternates the model's tokens with sandbox observations; only the model's tokens
   are trained, the reward comes from tests or from a judge model with a rubric, and the update is still
   GRPO over a group of episodes.
2. Long, uneven episodes force asynchronous training, which makes some data off-policy; and the sampling
   engine and the trainer disagree about token probabilities even with the same weights. Importance-sampling
   corrections (full, truncated, or IcePop's mask) trade a little bias for much less variance.
3. Most 2026 open-model reports set the KL term to zero or near it: the clip and IS masks bound each step, checkers are
   hard to game, and the reference copy leaves memory. The cost: nothing pulls the policy back if a checker
   has a hole.

## 10. Next and go deeper
Next: `distillation` (merging the specialists this page trains into one model).
Back: `rlvr-grpo` (the group, the advantage and the clip), `sft` (loss masking).
Go deeper: Sebastian Raschka, *The State of Reinforcement Learning for LLM Reasoning*
(https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training) · Nathan Lambert,
*RLHF Book* (https://rlhfbook.com/)

## 11. Key-frame sketch
Frame 8 (IcePop masks three tokens), desktop width. `[x]` = hatched (masked) chip. Checked against the
reproducer below (ρ 3.20 / 0.40 / 2.40 masked; weights 3.20 / 2.00 / 0; 3 of 39).

```text
 ρ = trainer / engine
 0.00 ──────[░░░░░░ 0.50 ─ 2.00 ░░]──────────▼3.20── 3.50
             same factor either way: ½ and 2   row 4's 48
                                     R    A
 4 [7][×][8][=][48][,][so][x48]  ✗   0  −0.58
 6 [x63]                         ✗   0  −0.58
 7 [7][×][8][=][x58]             ✗   0  −0.58
 5 other rows: ρ within [½, 2], unmasked
 masked 3 of 39   full IS 3.20 · truncated 2.00 · IcePop 0
 [Back] [Pause] [Next]  ━━━━━━━●━━  8 / 10   1×
```

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against scratchpad drafts of
`math/agentic.js` and of `math/grpo.js` built to `rlvr-grpo` §11; re-run against the built modules):
```sh
node -e '
Promise.all(["./math/agentic.js","./math/grpo.js"].map((p)=>import(p))).then(([m,g])=>{
  const f=(x)=>Number(x.toFixed(4)), L=[3,2,4,16,5,3,9,6];
  for (const [mode,lambda] of [["sync",1],["partial",0.75],["partial",0.5]]) { const s=m.rolloutSchedule(L,{mode,lambda});
    console.log(mode,lambda,s.iterationTime,s.busy,f(s.utilization),s.idle,s.carried.map((c,i)=>c?i+1:0).filter(Boolean)); }
  const A=g.groupAdvantages([1,0,0,0,1,0,0,0]);
  for (const [name,row,pe,pt] of [["r1 56",0,0.2,0.22],["r5 56",4,0.2,0.18],["r4 48",3,0.05,0.16],["r6 63",5,0.2,0.08],["r7 58",6,0.05,0.12]]) {
    const out=[name]; for (const prec of ["bf16","fp16"]) { const r=m.mismatchRatio(pt/pe,{precision:prec}); out.push(prec,f(r));
      for (const mode of ["none","full","tis","icepop"]) { const c=m.isCorrection(r,{mode}); out.push(mode+":"+f(c.weight)+(c.masked?"M":"")+"/"+f(A[row]*c.weight)); } }
    console.log(out.join(" ")); }
});'
```
Output (2026-10-07): sync → iteration 16, busy 48, utilization 0.375, idle 80, carried none · partial 0.75
→ 6, 35, 0.7292, 13, carried rows 4 and 7 · partial 0.5 → 4, 28, 0.875, 4, carried rows 4, 5, 7, 8.
BF16 / FP16 ρ: row 1 `56` 1.10 / 1.012 · row 5 `56` 0.90 / 0.9869 · row 4 `48` 3.20 / 1.1565 · row 6 `63`
0.40 / 0.8918 · row 7 `58` 2.40 / 1.1156. Row 4 `48` under BF16: ignore 1 → push −0.5774; full 3.2 →
−1.8475; TIS 2 → −1.1547; IcePop masked → 0. Row 6 `63` and row 7 `58` are masked under BF16 IcePop and
unmasked under FP16; rows 1 and 5 are never masked. Group: 39 tokens, A = +1.7321 / −0.5774 (from
`rlvr-grpo`'s reproducer).

**`math/agentic.js` signatures** (owner: `agentic-rl`; pure, no DOM; never mutates inputs; tests first):
```js
// One iteration of rollout scheduling. 'sync' waits for every episode; 'partial' ends the iteration when
// ceil(lambda·G) episodes have finished and carries the rest into the next iteration.
// utilization = busy ÷ (G · iterationTime): the page's single definition (README lesson 16).
rolloutSchedule(durations: number[], { mode = 'sync', lambda = 1 } = {}) →
  { iterationTime: number, busy: number, idle: number, utilization: number, carried: boolean[] }
//   [3,2,4,16,5,3,9,6]                               → { 16, 48, 80, 0.375, all false }
//   [3,2,4,16,5,3,9,6], { mode: 'partial', lambda: 0.75 } → { 6, 35, 13, 0.7292, rows 4 and 7 true }
//   [3,2,4,16,5,3,9,6], { mode: 'partial', lambda: 0.5 }  → { 4, 28, 4, 0.875, rows 4, 5, 7, 8 true }
//   throws on an empty array, a non-positive duration, or lambda ∉ (0, 1]

// Toy model of rounding mismatch: FP16 rounds about 8× finer than BF16, so log ρ shrinks 8×.
mismatchRatio(rhoBf16: number, { precision = 'bf16' } = {}) → number
//   (3.2) → 3.2 · (3.2, { precision: 'fp16' }) → 1.1565 · (0.4, fp16) → 0.8918 · (1, any) → 1
//   throws on rho ≤ 0 or an unknown precision

// Importance-sampling treatment of one token with ratio ρ = p_trainer / p_engine.
isCorrection(rho: number, { mode = 'none', cap = 2, band = [0.5, 2] } = {}) → { weight: number, masked: boolean }
//   (3.2, none) → { 1, false } · (3.2, full) → { 3.2, false } · (3.2, tis) → { 2, false }
//   (3.2, icepop) → { 0, true } · (0.4, icepop) → { 0, true } · (1.1, icepop) → { 1, false }
//   throws on rho ≤ 0 or an unknown mode
```
Reuses from `math/grpo.js` (owner `rlvr-grpo`, unchanged): `buildGroup`, `verifyFinalAnswer`,
`groupAdvantages`.

Test cases for the builder: `rolloutSchedule` with lambda = 1 equals 'sync'; utilization is 1 when all
durations are equal; `busy + idle = G · iterationTime`; `carried` is false for every episode with duration ≤
iterationTime; `mismatchRatio` preserves ρ = 1 and the side of 1 (ρ > 1 stays > 1); `isCorrection` IcePop
masks exactly ρ < 1/2 or ρ > 2 (the band edges are kept); inputs are never mutated.

## 12. Open questions for the reviewer
**Data-pass keys** (new; entry ids as accepted in `rlvr-grpo` §13): `glm-5.rl_infra`,
`glm-5.agentic_envs_swe` · `deepseek-v3.2.rl_patches` · `deepseek-v4-pro.rl_infra`,
`deepseek-v4-pro.reward_model` · `kimi-k3.rl_experts` = 9, `kimi-k3.rl_rollouts`, `kimi-k3.reward_model` ·
`nemotron-3-super.swe_rl_stage` · `mistral-large-4.rl_tokens_per_day` = 33e9,
`.rl_trainable_tokens_per_day` = 16e9, `.rl_gpus` = 3000 · `minimax-m2.rl_rewards`,
`minimax-m2.agentic_envs` (reported) · a non-model fact `papers.fp16-mismatch-2025` (arXiv 2510.26788,
reported), in `data/papers.json` (ruled).

**Graph changes:** none.

**Judgment calls:**
- **`request` reused as an episode timeline** (§4): accepted (expert review), no alias.
- **The FP16 ×8 model.** It is a first-principles toy (unit roundoff 2⁻¹¹ vs 2⁻⁸, owned by `gpu-primer`), not a measurement; the
  page says so visibly, on the chip itself. The brief's FP16 fix is REPORTED. Accepted (expert review).
- **Episode durations attached to `rlvr-grpo`'s answer rows.** A one-token answer `63` "taking 3 minutes"
  is a stand-in; the stage line says so. Accepted (expert review).
- **Frame 4 is a branch** (judges for untestable tasks) inside the "how episodes are scored" thread; it is
  labeled as a branch on screen and frame 5 returns to the main line.

## 13. Reviewer rulings (expert review, 2026-10-07)
Applied from `track-review-recipe.md` §7 (change log: `fix-recipe-review.md`):
- Must 1: frame 8 collapsed per README lesson 18 (rows 4, 6, 7 in full plus `5 other rows: ρ within [½, 2], unmasked`), with a 580 × 366 layout check; §11 sketch redrawn to match.
- Should: `same factor either way: ½ and 2` printed under the linear band (lesson 21); truncated IS "caps the weight (at 2 in this toy)"; "most 2026 open-model reports" in §1, §3, frame 9 and takeaway 3; frame 7 note "most tokens agree within a few percent (34 of 39 here)"; frame 6 numbers say only the part sampled before the cut is off-policy; the FP16 chip itself reads "toy model of rounding only"; §3 ¶4 dated "in 2026".
- Nice (declined in part): frame 10 prints Kimi K3's actual domains (general, general agents, coding agents) × low / high / max, not the track's illustrative `math & code · agents · chat`, because brief 02 §4.6 names K3's three domains and relabeling them would misstate the report.
