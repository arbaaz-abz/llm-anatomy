# RLHF and DPO (`rlhf-dpo`)

Track: training · Section: recipe · Prereqs: sft
Next: `rlvr-grpo` (the one slug whose `prereqs` list `rlhf-dpo`)
Status: draft
Sources: 02 §4.1, §4.2, §4.5, §5, §6, §7 (RLHF/PPO clip, KL penalty and DPO toy rows) · 05 §1.1, §1.2

Running example: one prompt, `Why is the sky blue? (asked by a 7-year-old)`, and two answers.
A = `Air scatters blue light the most` (6 tokens). B = `Rayleigh scattering , λ⁻⁴` (4 tokens). A person picks A.
The same pair is reused for the reward model (frames 1–2) and for DPO (frame 10), so the learner sees one
piece of data feed two different methods.

Contract with `rlvr-grpo` (its §4 "terms assumed from `rlhf-dpo`"): this page defines **policy, reward
model, critic (value model), reference model, KL term, and PPO's clip range ε**, one per frame. It draws
the same four-model shelf that `rlvr-grpo` frame 7 dims, and the same `clipLine` (band [0.8, 1.2],
ε = 0.2, marker 1.25, label `ratio r = now / when sampled`) that `rlvr-grpo` frames 8–9 zoom into. This
page stops at "past the band the gradient switches off"; why that is not a shrink, and clip-higher, stay
on `rlvr-grpo`.

## 1. Learning objective
After this page you can explain how a model is trained toward answers people prefer when no program can
check them: a reward model learns to score the picked answer of a pair higher, PPO samples answers from
the policy and pushes them by how much they beat a critic's prediction, the clip range caps each step,
and a KL term against a frozen reference model stops the policy from gaming the reward model; and how DPO
gets the same preference signal from the pairs directly, with no reward model and no sampling, by widening
the gap between the chosen and rejected answers relative to the reference. You can also say where each
still appears in 2026 pipelines.

## 2. Misconceptions to correct
- **Misconception:** "The reward model knows what a good answer is." → **Reality:** it is a learned
  stand-in, trained on a finite set of pairs. The policy is optimized against it and finds answers it
  overrates: here, opening with `Great` scores 1.3 against the honest answer's 1.0. This is reward hacking,
  and it is the reason RLHF needs an anchor. Corrected by frame 7 (labeled branch). (source: 02 §4.1 step 2
  "to prevent reward hacking"; §4.5 "ORMs … hackable")
- **Misconception:** "The KL term is an optional regularizer that only smooths training." → **Reality:**
  in RLHF it is what keeps the policy near answers the reward model was trained on. With β = 0 the
  flattering policy wins (1.3 > 1.0); with β = 0.5 the honest one wins (0.908 > 0.248), because the
  flattering policy moved much further from the reference (KL 2.10 vs 0.18). Corrected by frame 8. 2026
  reasoning RL drops or shrinks this term because its checkers are harder to game; you'll see why on
  `rlvr-grpo` and `agentic-rl`. (source: 02 §4.1; §7 "KL penalty" row: "many 2026 recipes set β ≈ 0")
- **Misconception:** "DPO makes the chosen answer more likely." → **Reality:** DPO's loss depends only on
  the gap between the two answers' log-probability changes. If the chosen answer drops by 1 and the
  rejected by 3 (both less likely than before), the loss still falls, 0.693 → 0.598. Corrected by
  try-this 2. (source: first principles on the DPO loss, arXiv 2305.18290 `[BG]` via 02 §4.2)
- **Misconception:** "RLHF is how 2026 models learn to reason." → **Reality:** reasoning and agent
  gains come from RL on verifiable rewards (`rlvr-grpo`, `agentic-rl`). RLHF survives as a late polish
  stage for helpfulness, style and safety, usually with a judge model as the reward; DPO survives as a
  cheap preference stage in smaller, fully open pipelines. Corrected by frame 11. (source: 02 §4.1 item 4;
  §4.2)

## 3. Hook and intuition (final wording)
**Hook:** How do you train a model toward the answers people prefer, when "better" is something no
program can check?

For a question like "why is the sky blue?" asked by a child, there is no unit test. What you can get is
a judgment: show a person two answers and ask which is better. RLHF (reinforcement learning from human
feedback) turns many such judgments into a training signal in two steps. First it trains a **reward
model**: a copy of the language model with a score head, adjusted until it scores the picked answer of
each pair above the other. Then it runs reinforcement learning against that score.

The model being trained is called the **policy**. In PPO, the classic algorithm, the policy writes fresh
answers, the reward model scores them, and a fourth network, the **critic**, predicts what score this
prompt usually earns. Each answer's tokens are pushed up or down by how much the answer beat that
prediction. Two brakes keep the steps safe. The **clip range** ε stops pushing a token once its
probability has moved more than 20% from where it was when the answer was sampled. The **KL term**
subtracts a penalty for drifting away from a frozen **reference model**, the SFT model the run started
from. The KL term matters because the reward model is only an approximation: left alone, the policy
finds answers it overrates, such as flattery, and the score rises while real quality falls.

That is four models in memory (policy, reference, reward model, critic) and a sampling loop on every
step. **DPO** (direct preference optimization) skips both the reward model and the sampling. It works
straight from the pairs: for each pair it measures how much more likely the policy made each answer than
the reference does, and widens the gap between the chosen and rejected answers. The price is that it
only ever learns from the fixed pairs it was given, and it optimizes a gap, not the chosen answer itself.

In 2026 both still exist, but in narrower roles. Reasoning comes from RL with checkable rewards. RLHF,
now usually with a judge model as the reward, polishes helpfulness, style and safety at the end of the
pipeline (Nemotron 3 Super runs it as a separate final stage). DPO is the cheap preference stage of
smaller, fully open pipelines (Olmo 3 runs it between SFT and RL).

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `token` (the two answers as chip rows), `verdict` (✓ on
the picked answer, ✗ on the other; the person's choice is a correct/incorrect judgment for this pair),
`block` (policy, reward model, critic, reference; `active` vs `dim`), `flow` (carry `token` from the
policy into a new answer; carry `gradient` back into the chips), `vector` (`NUMBER_CELL` rows of 4
next-token probabilities, printed; 4 × 43 = 172 px, well inside the stage), `cell` (single printed
scores), `clipLine` (frame 5, identical settings to `rlvr-grpo` frame 8), plain labeled text marks for
readouts (`P(A preferred) 0.818`, `β = 0.5`, `total`), and the branch label `what if nothing anchors the
policy?`.

New glyphs proposed: none.

The four next-token cells are the candidates for an answer's first token: `Air`, `The`, `Rayleigh`,
`Great`. Their probabilities, the reward scores (1.2, −0.3, 1.0, 1.3), the critic's 0.8 and the
log-probability changes in frame 10 are hand-picked stand-ins. A visible line under the stage says: "The
scores and probabilities are hand-picked stand-ins; the arithmetic on them is exact. Real KL terms sum
over every token of an answer; this page shows one position."

Terms introduced (one per frame, defined on screen where first shown): preference pair (1) · reward
model (2) · policy (3) · critic, also called the value model (4) · clip range ε (5) · reference model (6)
· reward hacking (7, branch) · KL term (8) · none new (9, the cost) · DPO (10) · none new (11).
Terms assumed from prereqs: SFT model and chat answers (`sft`); token, next-token probability,
cross-entropy (`pretraining`).

Indexing: no positions or addresses are numbered on this page.

Selection: the answer being followed (A) carries the selection outline in frames 1, 2 and 10; in frames
6–8 the followed row is the policy row. Every quantity is encoded once: probabilities are printed cell
numbers on the value scale (maxAbs 1); scores are printed cells; no bar lengths.

## 5. Animation script
Thread order: frames 1–2 build the reward model; 3–5 the PPO update; 6–8 the anchor (6 reference,
7 a labeled branch, 8 the KL term back on the main line); 9 the cost; 10 DPO; 11 where each lives in 2026.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Prompt line `Why is the sky blue? (asked by a 7-year-old)`. Two chip rows: A `Air scatters blue light the most`, B `Rayleigh scattering , λ⁻⁴`. A small `block` "person" at the right. | The rows type in; a `verdict` ✓ stamps beside A and ✗ beside B. A gets the selection outline. | A person reads two answers to the same prompt and picks the better one. That choice, a preference pair, is the only signal here: no program could check this. | `pair: A ≻ B` |
| 2 | A `block` "reward model (SFT copy + score head)". Two `cell`s at its right: `r(A)`, `r(B)`. Readout `P(A preferred)`. | Each row flows into the block; the two cells fill; the readout types in. | The reward model is a copy of the model that outputs one score. It is trained until the picked answer scores higher: 1.2 against −0.3 here. | `r(A) = 1.2` · `r(B) = −0.3` · `P(A preferred) = σ(1.5) = 0.818` · loss `−ln 0.818 = 0.201` (if it had the order wrong: 1.701) |
| 3 | Left: `block` "policy: the model being trained" (active). A new chip row appears from it: `Air bends blue light most` (stand-in). The reward model block scores it. | A `flow` (carry `token`) runs policy → row; the row flows into the reward model; a `cell` `r = 1.0` fills. | The policy, the model being trained, writes a fresh answer. The reward model scores it, and that score is what reinforcement learning tries to raise. | `r = 1.0` |
| 4 | A third `block` "critic (value model)" under the policy, with a `cell` `0.8` labeled "usual score for this prompt". Readout `beat it by +0.2`. | The critic's cell fills; the readout subtracts and types `+0.2`; a `flow` (carry `gradient`) runs back into the answer's chips, which warm slightly. | A critic, a second trained network, predicts the score this prompt usually earns. Each token is pushed by how much the answer beat that prediction. | `value = 0.8` · `1.0 − 0.8 = +0.2` |
| 5 | Zoom on the answer's last chip. `clipLine` 0.6 → 1.6, band [0.8, 1.2] shaded, label `ratio r = now / when sampled`, marker at 1.0. | The marker slides 1.0 → 1.25 and crosses the band's right edge; the chip gets the hatch (gradient off). | PPO updates on each batch several times, so probabilities drift. Once a token is 20% more likely than when sampled, the clip range switches its update off. | `ε = 0.2` · band `[0.8, 1.2]` · `r = 1.25`, `beat it by +0.2` · objective `min(1.25 · 0.2, 1.2 · 0.2) = 0.24`, clipped · at `r = 1.10`: `0.22`, not clipped |
| 6 | A dim `block` "reference (frozen SFT copy)" beside the policy. Two `vector` rows of 4 printed cells, headers `Air · The · Rayleigh · Great`: `reference` and `policy`. | The reference row fills first; then the policy row's cells count from the reference values to the drifted values. | The reference model is a frozen copy of where the policy started. Comparing their probabilities shows how far training has moved the policy. | reference `[0.40, 0.30, 0.20, 0.10]` · policy `[0.70, 0.15, 0.10, 0.05]` |
| 7 | Branch label at the top: `what if nothing anchors the policy?` A third row `hacked policy` `[0.01, 0.01, 0.01, 0.97]` and its answer `Great question ! Air scatters blue …`. Reward cells: honest 1.0, hacked 1.3. | The hacked row's `Great` cell fills to 0.97; the reward cell counts to 1.3 and the label `reward model overrates flattery` appears. | Branch: with no anchor, the policy finds answers the reward model overrates, like opening with flattery. This is reward hacking: the score rises while real quality falls. | `r(honest) = 1.0` · `r(hacked) = 1.3` |
| 8 | **Key frame.** Back on the main line. A plain three-row readout table under the rows: `reward · KL from reference · total` for honest and hacked; a `β = 0.5` label. | The KL column fills (0.184, 2.103); the total column counts down from the reward to reward minus penalty; the honest row's total ends larger. | The KL term subtracts a penalty for drifting from the reference. Flattery drifted far, so with the penalty the honest answer scores higher. | KL(honest ‖ ref) `= 0.184` · KL(hacked ‖ ref) `= 2.103` · β = 0: `1.000` vs `1.300` · β = 0.5: `1.0 − 0.5 · 0.184 = 0.908` vs `1.3 − 0.5 · 2.103 = 0.248` |
| 9 | The four-model shelf, all `active`: `policy`, `reference`, `reward model`, `critic`, with a plain label `in memory at once`; a loop arrow (`flow`, carry `token`) policy → answers → reward model labeled `sample every step`. | The four blocks light one after another; the loop arrow runs once. | Classic RLHF keeps four models in memory and generates fresh answers at every step. That cost is why simpler methods took over most of this job. | `4 models` · `1 sampling loop per step` |
| 10 | The pair from frame 1 again (A selected). Under each answer, a `cell` with its log-probability change versus the reference; a `β = 0.1` label; readouts `gap`, `loss`. The reward model and critic blocks fade to `dim` with the label `not needed`. | The two change cells fill (+0.5, −0.2); the gap readout types `0.07`; the loss readout types `0.659`. | DPO trains on the pairs directly: no reward model, no sampling. Measured against the reference, it widens the gap between the chosen and the rejected answer. | log π − log π_ref: A `+0.5`, B `−0.2` · β = 0.1 · implicit rewards `0.05`, `−0.02` · gap `0.07` · loss `−ln σ(0.07) = 0.659` (`ln 2 = 0.693` at zero gap) · update weight `σ(−0.07) = 0.483` |
| 11 | A six-stage pipeline strip (the `training-pipeline` strip, dim) with two stages lit: a `DPO` tag between SFT and RL labeled `Olmo 3 (reported)`, and an `RLHF` tag on the final stage labeled `Nemotron 3 Super: judge-model reward`. | The two tags drop onto the strip one after the other. | In 2026, DPO is a cheap preference stage in smaller open pipelines. RLHF, now with a judge model as the reward, polishes style and safety at the end. | Olmo 3: SFT → DPO (~200K pairs) → RLVR (~105K prompts), reported · Nemotron 3 Super: RLVR → SWE-RL → RLHF with a principle-following judge, confirmed |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 2, 8 and 10 draw from `bradleyTerry`, `klDivergence` + `klPenalizedReward` and `dpoLoss`;
frame 5 from `clippedSurrogate` in `math/grpo.js` (owned by `rlvr-grpo`, reused, not redefined).
Caption counts (words/sentences, `rlvr-grpo`'s counter with the row pattern widened to two-digit frames,
run 2026-10-07): 29/2, 26/2, 25/2, 25/2, 27/2, 23/2, 27/2, 23/2, 26/2, 26/2, 28/2.

## 6. Toy
"Tune a DPO pair." The pair from frame 1, its two log-probability changes, and β, with live DPO numbers.
A visible line under the toy: "The log-probability changes are what training would produce; here you set
them by hand."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `dChosen` | Change in log-probability of A (chosen) vs the reference | Slider | −10 to 10, step 0.1 | 0.5 | – |
| `dRejected` | Change in log-probability of B (rejected) vs the reference | Slider | −10 to 10, step 0.1 | −0.2 | – |
| `beta` | β (how tightly DPO is tied to the reference) | Preset chips | `0.1` / `0.5` | 0.1 | the two chips are the control |
| reset | Reset | Button | – | – | back to (0.5, −0.2, 0.1) |

**Live outputs** (all printed; nothing behind hover)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Implicit reward of A, of B | `dpoLoss({ dChosen, dRejected, beta }).rewardChosen / .rewardRejected` | signed, 3 decimals, `NUMBER_CELL` cells on the value scale (maxAbs 1) |
| Gap (margin) | `.margin` | signed, 3 decimals |
| P(A preferred), as the loss sees it | `sigmoid(margin)` | 3 decimals |
| Loss | `.loss` | 3 decimals; a fixed reference mark `0.693 = ln 2 (no preference yet)` printed beside it |
| Update weight (how hard DPO still pushes this pair) | `.weight` | 3 decimals |

**Try this** (each leads to a named insight)
1. Predict first: if A keeps gaining and B keeps losing, does DPO keep pushing just as hard? Drag `dChosen`
   to 5 and `dRejected` to −5: loss 0.313, update weight 0.269. Keep going to 10 and −10: loss 0.127, weight 0.119. → **Insight: the update fades as a pair is learned.**
   DPO stops spending effort on pairs it already ranks correctly, the way a reward model's loss does
   (frame 2), but with no reward model and no sampling.
2. Set `dChosen = −1` and `dRejected = −3`. Both answers are now less likely than under the reference, yet
   the loss is 0.598, lower than the 0.693 at (0, 0). → **Insight: DPO optimizes the gap, not the chosen
   answer.** The chosen answer can become less likely during DPO; this is a known side effect and one
   reason labs check likelihoods during preference training.
3. At (0.5, −0.2), switch β from 0.1 to 0.5: gap 0.07 → 0.35, loss 0.659 → 0.533. Then, at β = 0.1, find
   the same loss: you need `dChosen = 3.5` with `dRejected = 0`, against `0.7` at β = 0.5. → **Insight: β
   is the leash.** A larger β reaches the same loss with five times less drift from the reference. It plays
   the role the KL term's β plays in RLHF (frame 8), because DPO is derived from that same KL-anchored
   objective (§7).

Lesson-17 check: all three insights read straight off `dpoLoss` outputs at the stated settings (see
§11 reproducer); none depends on the stand-in reward scores.

## 7. Show me the math
```tex
\text{Reward model (Bradley–Terry):}\quad
P(\htmlClass{hl-a}{A} \succ B) = \sigma\big(r_\phi(x, \htmlClass{hl-a}{A}) - r_\phi(x, B)\big),
\qquad \mathcal{L}_{\text{RM}} = -\ln \sigma\big(r_\phi(x, A) - r_\phi(x, B)\big)
```
```tex
\text{RLHF objective:}\quad
\max_\theta\; \mathbb{E}_{y \sim \htmlClass{hl-pol}{\pi_\theta}}\big[\, r_\phi(x, y) \,\big]
- \htmlClass{hl-beta}{\beta}\, \mathbb{D}_{\text{KL}}\big(\htmlClass{hl-pol}{\pi_\theta}(\cdot \mid x) \,\|\, \htmlClass{hl-ref}{\pi_{\text{ref}}}(\cdot \mid x)\big),
\qquad \mathbb{D}_{\text{KL}}(p \,\|\, q) = \sum_v p_v \ln \frac{p_v}{q_v}
```
```tex
\text{PPO step:}\quad
\hat A_t = r - V_\psi(x)\ \ (\text{GAE in practice}),\qquad
\min\!\Big( r_t \hat A_t,\ \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\, \hat A_t \Big),\quad
r_t = \frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{old}}(y_t \mid \cdot)}
```
```tex
\text{DPO:}\quad
\mathcal{L}_{\text{DPO}} = -\ln \sigma\Big( \htmlClass{hl-beta}{\beta}\big[ \underbrace{\ln \tfrac{\pi_\theta(A)}{\pi_{\text{ref}}(A)}}_{d_A} - \underbrace{\ln \tfrac{\pi_\theta(B)}{\pi_{\text{ref}}(B)}}_{d_B} \big] \Big)
```
Shapes: r_φ outputs one scalar per (prompt, answer); the probability rows on screen are [4] slices of a
[vocab] distribution at one position; d_A, d_B are sums of per-token log-probability changes over each
whole answer (scalars). Panel notes: (a) DPO is derived by solving the RLHF objective above in closed
form (the optimal policy is the reference reweighted by exp(r/β)) and substituting it into the
Bradley–Terry loss, which is why the same β appears in both. (b) Real PPO computes per-token advantages
with GAE and a per-token KL; this page uses one advantage and one position. (c) Variants named in brief 02
§4.2: SimPO (no reference), KTO (single good/bad labels), ORPO (SFT and preference in one stage), APO.
Color links: `hl-a` = the chosen answer A (frames 1, 2, 10); `hl-pol` = the policy row; `hl-ref` = the
reference row (frames 6–8); `hl-beta` = the `β` label (frames 8, 10).

## 8. In today's models (Oct 2026)
All keys are proposed for the data-extension pass (entry ids follow the ids accepted in `rlvr-grpo` §13).

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Nemotron 3 Super runs RLVR (21 environments), then SWE-RL, then a separate RLHF stage with a principle-following generative reward model initialized from Qwen3-235B-A22B-Thinking-2507 and trained on HelpSteer 3 | `models.nemotron-3-super.rlhf_stage` = "GenRM (principle-following, from Qwen3-235B-A22B-Thinking-2507, HelpSteer 3)" (confirmed); `.rl_environments` = 21 (proposed by `rlvr-grpo`) | 02 §4.1 item 4 |
| GLM-5's general RL mixes rule rewards, outcome reward models and generative reward models, and anchors on human-written responses to avoid style drift | `models.glm-5.general_rl_rewards` = "rule + ORM + GRM; human-written anchors" (confirmed) | 02 §4.1 item 4; §4.5 |
| Mistral Large 4 verifies with "reward models, unit tests, LLM judges, and static checks" | `models.mistral-large-4.rl_verifiers` (confirmed; Mistral blog) | 02 §4.5, §6 |
| Olmo 3: SFT → DPO on ~200K "Delta Learning" pairs (chosen from a stronger model, rejected from a weaker one) → RLVR on ~105K prompts | `models.olmo-3.dpo_pairs` = 200000, `.rlvr_prompts` = 105000 (reported) | 02 §4.2 |
| SmolLM3 used APO, a DPO-family method | `models.smollm3.preference_method` = "APO" (reported) | 02 §4.2 |
| DPO is not listed as a main stage in the DeepSeek-V4, GLM-5, Kimi K3, MiniMax-M2 or MiMo-V2-Flash reports (absent from the reports read, not proof of non-use) | `models.<id>.dpo_stage` = "not listed" for `deepseek-v4-pro`, `glm-5`, `kimi-k3`, `minimax-m2`, `mimo-v2-flash` (confirmed absent) | 02 §4.2; §8 |
| PPO keeps four models in memory: policy, reference, reward model, critic | mechanics `[BG]` (InstructGPT, arXiv 2203.02155), not a dated fact | 02 §4.1 item 3 |

Mechanics (Bradley–Terry reward model, PPO's clipped surrogate, KL penalty, DPO's loss) are written from
first principles per spec §7 and cite arXiv 2203.02155 and 2305.18290.

## 9. Takeaways
1. RLHF learns a reward model from preference pairs, then runs PPO against it: four models in memory
   (policy, reference, reward model, critic) and fresh samples every step.
2. The reward model is an approximation the policy can game; the KL term against the frozen reference
   model is the anchor that stops it, and the clip range caps each step.
3. DPO gets the preference signal straight from the pairs (no reward model, no sampling) by widening the
   chosen-vs-rejected gap relative to the reference; in 2026 it is a cheap stage for open pipelines, while
   RLHF with a judge model polishes style and safety at the end.

## 10. Next and go deeper
Next: `rlvr-grpo` (swap the reward model for a checker and the critic for a group mean; the KL term
shrinks or disappears).
Back: `sft` (where the reference model comes from).
Go deeper: Nathan Lambert, *RLHF Book* (reward modeling, policy gradients, direct alignment;
https://rlhfbook.com/) · UNIPO, interactive comparison of REINFORCE, PPO and GRPO-family objectives with
per-token coloring (https://poloclub.github.io/unipo/)

## 11. Key-frame sketch
Frame 8 (the KL term), desktop width. Checked against the reproducer output below (KL 0.184 / 2.103;
totals 0.908 / 0.248 at β = 0.5; 1.000 / 1.300 at β = 0).

```text
 prompt: Why is the sky blue? (asked by a 7-year-old)
            Air    The   Rayleigh Great
 reference [0.40] [0.30] [0.20]  [0.10]   (frozen SFT copy)
 policy    [0.70] [0.15] [0.10]  [0.05]   ← followed
 hacked    [0.01] [0.01] [0.01]  [0.97]   (branch, frame 7)

            reward   KL from ref   total (β = 0.5)
 honest     1.000      0.184        0.908
 hacked     1.300      2.103        0.248
 The KL term subtracts a penalty for drifting from the
 reference. Flattery drifted far, so with the penalty
 the honest answer scores higher.
 [Back] [Pause] [Next]  ━━━━━━━●━━━  8 / 11   1×
```

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against scratchpad drafts of
`math/lm.js`, `math/preference.js` and `math/grpo.js` that follow the signatures below; re-run against the
built modules):
```sh
node -e '
Promise.all(["./math/lm.js","./math/preference.js","./math/grpo.js"].map((p)=>import(p))).then(([lm,pr,g])=>{
  const f=(x)=>Number(x.toFixed(4));
  const bt=pr.bradleyTerry(1.2,-0.3); console.log("BT",f(bt.pChosen),f(bt.loss),"flip",f(pr.bradleyTerry(-0.3,1.2).loss));
  console.log("clip",g.clippedSurrogate(1.25,0.2),g.clippedSurrogate(1.1,0.2));
  const ref=[0.40,0.30,0.20,0.10], honest=[0.70,0.15,0.10,0.05], hacked=[0.01,0.01,0.01,0.97];
  const kh=lm.klDivergence(honest,ref), kx=lm.klDivergence(hacked,ref); console.log("KL",f(kh),f(kx));
  for (const b of [0,0.1,0.2,0.5]) console.log("beta",b,f(pr.klPenalizedReward(1.0,kh,b)),f(pr.klPenalizedReward(1.3,kx,b)));
  for (const [dc,dr,b] of [[0,0,0.1],[0.5,-0.2,0.1],[5,-5,0.1],[10,-10,0.1],[-1,-3,0.1],[0.5,-0.2,0.5],[3.5,0,0.1],[0.7,0,0.5]]) {
    const r=pr.dpoLoss({dChosen:dc,dRejected:dr,beta:b});
    console.log("dpo",dc,dr,b,f(r.rewardChosen),f(r.rewardRejected),f(r.margin),f(pr.sigmoid(r.margin)),f(r.loss),f(r.weight)); }
});'
```
Output (2026-10-07): BT 0.8176 / 0.2014, flipped 1.7014 · clip (1.25, +0.2) → 0.24 clipped; (1.10, +0.2)
→ 0.22 not clipped · KL 0.1838 / 2.1031 · β 0 → 1 / 1.3; β 0.1 → 0.9816 / 1.0897; β 0.2 → 0.9632 /
0.8794; β 0.5 → 0.9081 / 0.2484 · DPO (0, 0, 0.1) → loss 0.6931, weight 0.5; (0.5, −0.2, 0.1) → rewards
0.05 / −0.02, gap 0.07, σ 0.5175, loss 0.6588, weight 0.4825; (5, −5) → gap 1, loss 0.3133, weight 0.2689;
(10, −10) → gap 2, loss 0.1269, weight 0.1192; (−1, −3) → gap 0.2, loss 0.5981, weight 0.4502;
(0.5, −0.2, 0.5) → gap 0.35, σ 0.5866, loss 0.5334, weight 0.4134; (3.5, 0, 0.1) and (0.7, 0, 0.5) → the
same gap 0.35 and loss 0.5334.

Note from the run: the flip from "hacked wins" to "honest wins" happens between β = 0.1 and β = 0.2
(crossover β = 0.3 / 1.919 = 0.156); frame 8 shows β = 0.5 so the gap is visible at a glance.

**`math/preference.js` signatures** (owner: `rlhf-dpo`; pure, no DOM; tests first):
```js
sigmoid(x: number) → number
//   0 → 0.5 · 1.5 → 0.8176 · −0.07 → 0.4825

// Bradley–Terry pairwise reward-model loss.
bradleyTerry(rChosen: number, rRejected: number) → { pChosen: number, loss: number }
//   (1.2, −0.3) → { pChosen: 0.8176, loss: 0.2014 } · (−0.3, 1.2) → loss 1.7014 · (0, 0) → { 0.5, 0.6931 }

// DPO loss from the summed log-probability changes vs the reference.
dpoLoss({ dChosen: number, dRejected: number, beta = 0.1 }) →
  { rewardChosen, rewardRejected, margin, loss, weight }   // weight = σ(−margin), the gradient scale
//   (0.5, −0.2, 0.1)  → { 0.05, −0.02, 0.07, 0.6588, 0.4825 }
//   (−1, −3, 0.1)     → { −0.1, −0.3, 0.2, 0.5981, 0.4502 }
//   (0.5, −0.2, 0.5)  → { 0.25, −0.1, 0.35, 0.5334, 0.4134 }
//   throws RangeError when beta ≤ 0 or an input is not finite

// RLHF's per-sample objective with the KL penalty.
klPenalizedReward(reward: number, kl: number, beta: number) → number
//   (1.0, 0.1838, 0.5) → 0.9081 · (1.3, 2.1031, 0.5) → 0.2484 · (1.3, 2.1031, 0) → 1.3
```
Uses `klDivergence(p, q)` from `math/lm.js` (owner: `pretraining`, signature on that page):
`([0.70,0.15,0.10,0.05], [0.40,0.30,0.20,0.10]) → 0.1838` · `([0.01,0.01,0.01,0.97], same) → 2.1031` ·
`(p, p) → 0`. Uses `clippedSurrogate` from `math/grpo.js` unchanged.

Test cases for the builder: `dpoLoss` at zero margin equals ln 2 for any β; loss is strictly decreasing
in `dChosen` and strictly increasing in `dRejected`; `weight + sigmoid(margin) = 1`; `bradleyTerry(a, b).loss
+ ln(pChosen) = 0`; `klPenalizedReward` with β = 0 returns the reward unchanged.

## 12. Open questions for the reviewer
**Data-pass keys** (all new; entry ids as accepted in `rlvr-grpo` §13):
- `nemotron-3-super.rlhf_stage` (confirmed, 02 §4.1) · `glm-5.general_rl_rewards` (confirmed, 02 §4.1/§4.5)
  · `mistral-large-4.rl_verifiers` (confirmed, 02 §6; source https://mistral.ai/news/mistral-large-4/) ·
  `olmo-3.dpo_pairs` = 200000 and `olmo-3.rlvr_prompts` = 105000 (reported, 02 §4.2; source arXiv
  2512.13961) · `smollm3.preference_method` = "APO" (reported, 02 §4.2; new entry `smollm3`) ·
  `<id>.dpo_stage` = "not listed" for five entries (confirmed absent, 02 §4.2). `mimo-v2-flash` is a new
  entry id (the data file's `mimo-v2.6-pro` is a different checkpoint).

**Graph changes:** none.

**Judgment calls:**
- **Frame count (11).** One more than the pilots, because the term contract with `rlvr-grpo` needs six
  terms on screen, one per frame. Frames 9 and 11 introduce no term and could merge if the reviewer
  prefers ten.
- **Flattery as the hack.** A cartoon of reward hacking (a reward model that overrates an opener) is
  illustrative, not a documented case from the briefs; the page labels it as a stand-in branch. Brief 02
  supports only the general claim (KL "to prevent reward hacking"; ORMs "hackable").
- **One-position KL.** Real KL penalties sum per-token KLs over the answer; the page says so in its
  stand-in line and math note (b).
- **`verdict` for a preference.** ✓/✗ marks "picked / not picked" here, which reads as a correctness
  judgment for this pair. If the reviewer finds it misleading next to `rlvr-grpo`'s checker ✓/✗, swap to
  a plain text mark `picked`.
