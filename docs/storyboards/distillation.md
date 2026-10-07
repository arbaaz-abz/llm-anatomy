# Distillation (`distillation`)

Track: training · Section: recipe · Prereqs: agentic-rl
Next: none (no slug lists `distillation` as a prereq; it closes the Recipe section)
Status: draft
Sources: 02 §0 item 5, §3 (R1 distilled models, Olmo 3 traces), §4.8, §5 steps 5–6, §7 (Distillation / OPD
row) · 05 §1.1, §1.2

Running example, continued from `rlvr-grpo` and `agentic-rl`: the prompt `7 × 8 =` and the next token.
Four candidates `56 · 54 · 48 · 63` (the final tokens of `rlvr-grpo`'s rows 1, 2, 4 and 6). A **teacher**
(the math specialist that RL produced) and a **student** each give a probability to each:
teacher `[0.90, 0.05, 0.03, 0.02]`, student `[0.40, 0.30, 0.20, 0.10]`. Frame 6 reuses `rlvr-grpo`'s row 2,
`7 × 8 = 54`, whose GRPO advantage was −0.58 on every token. A visible line under the stage says: "The
teacher and student probabilities are hand-picked stand-ins; the losses and rewards computed from them are
exact."

## 1. Learning objective
After this page you can compare the three ways a student model learns from a teacher (fine-tuning on the
teacher's sampled text, matching the teacher's full probabilities on fixed text, and on-policy distillation,
where the student writes and the teacher grades every token), compute each one's loss or reward for one
token, explain why grading the student's own samples teaches it to handle its own mistakes and gives a
per-token signal that RL's one-number-per-answer does not, and describe how 2026 labs use multi-teacher
on-policy distillation to merge their RL specialists into one model.

## 2. Misconceptions to correct
- **Misconception:** "Distillation means fine-tuning a small model on a big model's outputs." → **Reality:**
  that is one of three kinds, and the least informative per token: it only uses the one token the teacher
  happened to write (loss 0.916 here). Logit distillation uses the teacher's whole row (KL 0.551), and
  on-policy distillation grades the student's own tokens. Corrected by frames 2–5. (source: 02 §4.8)
- **Misconception:** "If the student copies the teacher's text perfectly, it will behave like the
  teacher." → **Reality:** it only ever trained on prefixes the teacher wrote. The first time it samples
  its own mistake, `54`, it is somewhere it never practised. On-policy distillation trains exactly there:
  the teacher grades `54` at −1.79. Corrected by frames 4–5. (source: 02 §4.8 item 3; first principles)
- **Misconception:** "Distillation and RL are different worlds." → **Reality:** on-policy distillation
  is an RL loop with a different reward: the student samples, and each token's reward is the log of the
  teacher's probability over the student's. Where GRPO gave `7 × 8 = 54` −0.58 on every token, the teacher
  puts −1.79 on `54` and about 0 on `7 × 8 =`. Corrected by frame 6. (source: 02 §4.8 item 3 "equivalent
  to using sg(log π_T − log π_S) as a dense per-token reward")
- **Misconception:** "DeepSeek-V4 replaced RL with distillation." → **Reality:** its specialists are still
  trained with SFT and GRPO; only the final mixed-RL stage was replaced by multi-teacher on-policy
  distillation. A popular blog summary got this wrong; the paper is clear. Corrected by frame 7 and §8.
  (source: 02 §4.8 "Correction to a popular summary")

## 3. Hook and intuition (final wording)
**Hook:** A lab has trained a math expert, a coding agent and a chat model with RL. How does it ship one
model that is as good as each of them at its own job?

Distillation trains a **student** to match a **teacher**. The oldest way is to let the teacher write
answers and fine-tune the student on that text, exactly like SFT: for each position, only the one token
the teacher wrote counts. A richer way, logit distillation, uses the teacher's whole probability row at
each position: not just "56" but "56 at 0.90, 54 at 0.05". Both have the same blind spot. The student only
ever practises on text the teacher wrote, so the first time it samples its own mistake it is somewhere it
has never been, and nothing taught it what to do next.

**On-policy distillation** flips who writes. The student samples its own answer, mistakes included, and
the teacher scores every token the student wrote: the reward is how much likelier the teacher finds that
token than the student did. A good token earns a positive reward, the student's own `54` a strongly
negative one. That is an RL loop, with two differences that make it efficient: the reward is dense (one
per token, not one per answer, so the blame lands on `54` rather than on all of `7 × 8 = 54`), and it
needs no checker, so it works for any task a teacher can do.

That is why it became the 2026 way to merge specialists. Labs train separate experts with RL (math, code,
agents, chat, sometimes at several effort levels), then distill all of them into one student: each prompt
is graded by the teacher for its domain, or by a weighted mix. DeepSeek-V4 uses more than ten teachers,
Kimi K3 nine, and MiMo-V2-Flash reports that the merged model keeps each teacher's peak. The cost: every
teacher must be run (or cached) during training, and a student can at best match its teachers, not beat
them.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `vector` (`NUMBER_CELL` rows of 4 printed probabilities,
labeled `56 · 54 · 48 · 63`, value scale maxAbs 1; 4 × 43 = 172 px), `cell` (printed losses and rewards),
`token` (the student's sampled answer chips; `fill` = `valueColor(reward, 2)`), `block` (teacher, student,
specialists; `active` vs `dim`), `flow` (carry `token` for the sampled answer; carry `gradient` from the
teacher's grades into the chips), plain labeled text marks (`teacher wrote 56`, `student's own sample`,
`dense`, the per-model table).

New glyphs proposed: none.

Terms introduced (one per frame): teacher and student (1, one pair) · sequence-level (trace) distillation
(2) · logit distillation (3) · none new: off-policy from `agentic-rl`, applied to the teacher's text (4,
branch) · on-policy distillation (5) · dense reward (6) · multi-teacher merge (7) · catastrophic forgetting
(8) · none new (9).
Terms assumed from prereqs: policy, sampling, advantage, push, group (`rlvr-grpo`); off-policy, specialist
experts, generative judges (`agentic-rl`); KL divergence (`rlhf-dpo`); cross-entropy loss (`pretraining`);
SFT (`sft`).

Indexing: no positions or addresses are numbered on screen beyond `rlvr-grpo`'s row label "row 2".

Selection: the followed item is the student's `54` (frames 4–6), outlined wherever it appears; in frames
1–3 the followed item is the `56` cell.

## 5. Animation script
Thread order: frames 1–3 learning from the teacher's text; 4 (branch) the blind spot; 5–6 on-policy
distillation; 7–8 merging specialists; 9 small models.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Prompt chips `7 × 8 =`. Two `block`s: `teacher: math specialist` and `student`. Under each, a 4-cell `vector` headed `56 · 54 · 48 · 63`. | The teacher row fills, then the student row. The `56` cells take the selection outline. | A teacher, here the math specialist RL produced, and a student both give probabilities for the next token. The student is less sure. | teacher `[0.90, 0.05, 0.03, 0.02]` · student `[0.40, 0.30, 0.20, 0.10]` |
| 2 | The teacher writes `56` (a `token` chip with the plain label `teacher wrote 56`). The student's `56` cell is the only one lit; readout `loss 0.916`. | The chip flows to the student; three of the student's four cells dim. | The oldest way: fine-tune the student on text the teacher wrote, exactly like SFT. Only the one token the teacher picked counts. | `−ln 0.40 = 0.916` |
| 3 | The whole teacher row slides under the student row; readout `KL(teacher ‖ student) = 0.551`. | All four student cells light; the four pairwise terms type in under them, then sum. | Logit distillation uses the teacher's whole row as the target, so the student also learns how unlikely 54 and 48 are. | terms `0.90 ln 2.25 = 0.730` · `0.05 ln (1/6) = −0.090` · `0.03 ln 0.15 = −0.057` · `0.02 ln 0.2 = −0.032` · sum `0.551` |
| 4 | Branch label `what if the student writes its own answer?` The student samples `54` (chip outlined, plain label `student's own sample`). A plain mark: `never in its training text`. | A `flow` (carry `token`) runs from the student to the `54` chip. | Branch: both methods only trained on text the teacher wrote. When the student samples its own mistake, it is somewhere it never practised. | student's `p(54) = 0.30` |
| 5 | Back on the main line. The teacher `block` lights and grades the student's sample: a `cell` under `54` reads `−1.79`; beside it, plain reference marks for the other choices: `56 +0.81 · 48 −1.90 · 63 −1.61`. | A `flow` (carry `gradient`) runs from the teacher into the `54` chip, which cools to the value color for −1.79. | On-policy distillation lets the student write and the teacher grade every token it wrote. 54 earns minus 1.79: the teacher finds it six times less likely. | `ln 0.05 − ln 0.30 = −1.792` · `ln 0.90 − ln 0.40 = +0.811` · student's expected reward `−0.754` (= minus the reverse KL) |
| 6 | **Key frame.** Two rows of the answer `7 × 8 = 54`. Upper, labeled `GRPO (rlvr-grpo)`: every chip at −0.58. Lower, labeled `on-policy distillation`: per-chip rewards `0.00 · +0.05 · +0.03 · 0.00 · −1.79`. | The lower row's chips take their colors one by one; the `54` chip goes deep cool while the prefix stays near neutral. | GRPO gave every token of this wrong answer the same minus 0.58. The teacher's grades are dense: the blame lands on 54, not on the correct steps before it. | GRPO `−0.58 × 5` · OPD `7: 0.000 · ×: +0.054 · 8: +0.031 · =: 0.000 · 54: −1.792` |
| 7 | Three specialist `block`s from `agentic-rl` (`math`, `coding agent`, `chat`) above one `student`. Prompts of three kinds flow in; each is graded by its domain's teacher (a `flow` from the matching specialist). | Prompts drop in one by one; for each, the matching specialist lights and its `flow` reaches the student. | To merge RL specialists, each prompt the student answers is graded by the teacher for its domain. One student learns every specialist's skill. | DeepSeek-V4: > 10 teachers · Kimi K3: 9 (3 domains × 3 effort levels) · visible note: "DeepSeek-V4's specialists are still trained with SFT and GRPO; only the final mixed-RL stage was replaced" |
| 8 | A plain two-row comparison: `RL one domain after another` with a plain mark `earlier skills fade` vs `distill from all specialists at once` with `each teacher's peak kept (MiMo-V2-Flash)`. A plain line: `GLM-5 distills from its own earlier checkpoints to undo this.` | The first row's earlier skill labels dim one by one; the second row's stay lit. | Training one skill after another tends to erase earlier ones: catastrophic forgetting. Distilling from all specialists at once keeps them, which is why 2026 pipelines end this way. | facts per §8 |
| 9 | A four-step strip: `pretrain small` → `SFT on teacher traces` → `on-policy distillation` → `short RLVR (optional)`. A plain fact line: `DeepSeek-R1 (2025): ~800K samples distilled into small Qwen and Llama models`. | The four steps light in order. | Small models are made the same way: pretrain, fine-tune on a big teacher's traces, then distill on-policy. The cost: a student rarely surpasses its teachers. | facts per §8 |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 2–3 draw `tokenLoss` and `klDivergence` (`math/lm.js`); frames 5–6 draw `opdTokenReward`
(`math/distill.js`); frame 6's upper row is `rlvr-grpo`'s `groupAdvantages` (k = 2) value for row 2.
Caption counts (words/sentences, `rlvr-grpo`'s counter, run 2026-10-07): 23/2, 22/2, 21/1, 23/2, 26/2,
29/2, 23/2, 28/2, 25/2.

## 6. Toy
"Grade the student." The `7 × 8 =` position with a teacher row and a student row; switch how the student
learns, which token it sampled, and which teacher grades it.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `method` | How the student learns | Preset chips | `teacher's text` / `teacher's probabilities` / `on-policy` | `on-policy` | the chips are the control |
| `student` | Student's probabilities | Preset chips | `unsure [0.40, 0.30, 0.20, 0.10]` / `confident and wrong [0.10, 0.70, 0.10, 0.10]` / `close to the math teacher [0.85, 0.07, 0.05, 0.03]` | `unsure` | the chips |
| `sampled` | Which token the student sampled (on-policy only) | Selection: click, tap or arrow keys on a student cell | `56` / `54` / `48` / `63` | `54` | – |
| `teacher` | Who grades | Preset chips | `math teacher [0.90, 0.05, 0.03, 0.02]` / `chat teacher [0.60, 0.20, 0.10, 0.10]` / `50/50 mix` | `math` | – |

**Live outputs** (printed)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Teacher's text: loss on the teacher's token `56` | `tokenLoss(student[0])` | 3 decimals |
| Teacher's probabilities: forward KL | `klDivergence(teacher, student)` | 3 decimals |
| On-policy: reward for the sampled token, and for each other choice (4 cells, value scale maxAbs 3) | `opdTokenReward(teacher[i], student[i])` | signed, 3 decimals |
| On-policy: student's expected reward | `−klDivergence(student, teacher)` | signed, 3 decimals |
| Multi-teacher loss | `multiTeacherLoss(student, teachers, weights)` with weights `[1]`, `[1]` or `[0.5, 0.5]` | 3 decimals |

**Try this** (each leads to a named insight)
1. With `on-policy`, `unsure` and the math teacher, select each sampled token: `56` +0.811, `54` −1.792, `48`
   −1.897, `63` −1.609; the expected reward is −0.754. → **Insight: the teacher grades the student's own
   choices, wrong ones included.** Every token the student might write gets its own signed grade, which is
   what fixes the blind spot of frame 4.
2. Switch `student` to `close to the math teacher`: every reward shrinks toward 0 (`56` +0.057, `54`
   −0.337) and the expected reward is −0.013. Then switch to `confident and wrong`: `54` falls to −2.639 and
   the expected reward to −1.909. → **Insight: the signal is largest where the student is most wrong and
   vanishes as it matches the teacher.** On-policy distillation can bring a student up to its teacher but
   gives it no reason to go beyond.
3. Keep `close to the math teacher` and switch `teacher`: math 0.013, chat 0.152, 50/50 mix 0.082. →
   **Insight: averaging teachers pulls the student toward a blend.** A student that already matches the math
   specialist is still pushed away from it by the mix, which is why Kimi K3 picks one teacher per prompt by
   domain and effort level, and DeepSeek-V4 weights its teachers.

Lesson-17 check: the three insights are read straight off `opdTokenReward`, `klDivergence` and
`multiTeacherLoss` at the stated presets (reproducer below); none depends on more than the four-candidate
simplification, which the stage line names.

## 7. Show me the math
```tex
\text{trace (SFT) distillation:}\quad \mathcal{L} = -\ln \pi_S(y^{T}_t \mid y^{T}_{<t}),\quad y^{T} \sim \htmlClass{hl-t}{\pi_T}
```
```tex
\text{logit distillation:}\quad \mathcal{L} = \mathbb{D}_{\text{KL}}\big(\htmlClass{hl-t}{\pi_T}(\cdot \mid y_{<t}) \,\|\, \htmlClass{hl-s}{\pi_S}(\cdot \mid y_{<t})\big)
\quad\text{on fixed text}
```
```tex
\text{on-policy:}\quad y \sim \htmlClass{hl-s}{\pi_S},\qquad
\htmlClass{hl-r}{r_t} = \operatorname{sg}\big(\ln \htmlClass{hl-t}{\pi_T}(y_t \mid y_{<t}) - \ln \htmlClass{hl-s}{\pi_S}(y_t \mid y_{<t})\big),
\qquad \mathbb{E}_{y_t \sim \pi_S}[r_t] = -\mathbb{D}_{\text{KL}}(\pi_S \,\|\, \pi_T)
```
```tex
\text{DeepSeek-V4:}\ \mathcal{L} = \sum_i w_i\, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{E_i})\ (\text{full vocabulary})
\qquad
\text{Kimi K3:}\ r_t = \operatorname{clip}\big(\operatorname{sg}\ln \tfrac{\pi_T}{\pi_\theta}, \pm R_{\max}\big)
\qquad
\text{MiMo-V2-Flash:}\ A = \operatorname{sg}\ln \tfrac{\pi_T}{\pi_\theta} + \alpha A_{\text{ORM}}
```
Shapes: π_T, π_S are [|V|] rows at each position; the page shows a [4] slice (four candidates), so its KLs
are over those four entries only. sg = stop-gradient: the reward is a number, not something to
differentiate through. Panel notes: (a) reverse KL (student ‖ teacher) is "mode-seeking": it punishes the
student for putting probability where the teacher puts little; forward KL is "mean-covering". (b) DeepSeek
chose full-vocabulary KL because sampled-token estimates had high variance, and caches each teacher's
last-layer hidden states to rebuild logits on the fly. (c) Kimi K3 found top-k logit variants gave no gain.
Color links: `hl-t` = the teacher row; `hl-s` = the student row and the sampled chip; `hl-r` = the reward
cells (frames 5–6).

## 8. In today's models (Oct 2026)
All keys proposed for the data-extension pass; entry ids as accepted in `rlvr-grpo` §13 plus
`mimo-v2-flash`.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4: multi-teacher on-policy distillation, more than 10 teachers, full-vocabulary reverse KL; replaces only the mixed-RL stage, specialists still trained with SFT + GRPO | `models.deepseek-v4-pro.opd` = "multi-teacher, >10 teachers, full-vocab reverse KL", `.rl_algorithm` = "GRPO" (proposed by `rlvr-grpo`) (confirmed) | 02 §4.8 |
| Kimi K3: MOPD; per-token reward from one of 9 experts chosen by (domain, effort), clipped | `models.kimi-k3.opd` = "per-token reward from 1 of 9 experts (domain × effort), clipped", `.rl_experts` = 9 (proposed by `agentic-rl`) (confirmed) | 02 §4.8, §4.6 |
| MiMo-V2-Flash: MOPD adds a GRPO outcome advantage; reports that each teacher's peak is preserved without the usual trade-off | `models.mimo-v2-flash.opd` = "teacher log-ratio + α·ORM advantage; keeps each teacher's peak" (confirmed) | 02 §4.8 |
| GLM-5: on-policy cross-stage distillation as the final stage, with its own earlier checkpoints (SFT, reasoning RL, general RL) as teachers, to undo catastrophic forgetting | `models.glm-5.opd` = "cross-stage, earlier checkpoints as teachers" (confirmed) | 02 §4.8 |
| On-policy distillation was popularized by Thinking Machines and Qwen3; claims of large compute savings versus RL (about an order of magnitude) | `papers.thinking-machines-opd` (reported) — no model entry | 02 §4.8 item 3 |
| DeepSeek-R1 (2025): ~800K SFT samples, reused to distill small Qwen and Llama models | `models.deepseek-r1.sft_samples` = 800000 (`[BG]`, arXiv 2501.12948) | 02 §3, §4.8 item 1 |
| Olmo 3's SFT used ~2.3M reasoning traces distilled from QwQ-32B and DeepSeek-R1 | `models.olmo-3.sft_traces` = 2.3e6 (reported) | 02 §3 |
| Mistral says Large 4 "leaned less on distillation" than peers | `models.mistral-large-4.distillation_note` (reported; not in Mistral's blog) | 02 §4.8, §6 |

## 9. Takeaways
1. A student can learn from a teacher's text (one token per position), from its full probabilities on fixed
   text, or on-policy: the student writes and the teacher grades every token it wrote.
2. On-policy distillation is RL with a dense reward (the log of the teacher's probability over the student's):
   it trains the student on its own mistakes and puts the blame on the token that caused them.
3. 2026 labs train RL specialists, then merge them with multi-teacher on-policy distillation (DeepSeek-V4,
   Kimi K3, MiMo-V2-Flash, GLM-5 across its own stages); the cost is running every teacher, and a student can
   at best match them.

## 10. Next and go deeper
Next: none in the graph; this closes the Recipe section. Back: `agentic-rl` (where the specialists come
from), `training-pipeline` (the whole map).
Go deeper: Thinking Machines, *On-Policy Distillation* (https://thinkingmachines.ai/blog/on-policy-distillation/)
· Nathan Lambert, *RLHF Book*, chapter on on-policy distillation (https://rlhfbook.com/)

## 11. Key-frame sketch
Frame 6 (dense vs one-number grades), desktop width. Checked against the reproducer below.

```text
 prompt 7 × 8 =        56     54     48     63
 teacher (math)      [0.90] [0.05] [0.03] [0.02]
 student             [0.40] [0.30] [0.20] [0.10]

 row 2: 7 × 8 = 54        7      ×      8      =     54
 GRPO (rlvr-grpo)     [−0.58][−0.58][−0.58][−0.58][−0.58]
 on-policy distill.   [ 0.00][+0.05][+0.03][ 0.00][−1.79]
                                                  ▲ followed
 GRPO gave every token of this wrong answer the same minus
 0.58. The teacher's grades are dense: the blame lands on
 54, not on the correct steps before it.
 [Back] [Pause] [Next]  ━━━━━━●━━━  6 / 9   1×
```
The per-token teacher/student pairs for row 2 (stand-ins): `7` 0.50/0.50 · `×` 0.95/0.90 · `8` 0.98/0.95
· `=` 0.97/0.97 · `54` 0.05/0.30.

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against scratchpad drafts of
`math/lm.js` and `math/distill.js` matching the signatures):
```sh
node -e '
Promise.all(["./math/lm.js","./math/distill.js"].map((p)=>import(p))).then(([lm,d])=>{
  const f=(x)=>Number(x.toFixed(4)), T=[0.90,0.05,0.03,0.02], C=[0.60,0.20,0.10,0.10];
  const S={unsure:[0.40,0.30,0.20,0.10],wrong:[0.10,0.70,0.10,0.10],near:[0.85,0.07,0.05,0.03]};
  for (const [k,s] of Object.entries(S)) console.log(k,"traces",f(lm.tokenLoss(s[0])),"fwd",f(lm.klDivergence(T,s)),"rev",f(lm.klDivergence(s,T)),
    "rewards",s.map((p,i)=>f(d.opdTokenReward(T[i],p))),"chat",f(d.multiTeacherLoss(s,[C],[1])),"math",f(d.multiTeacherLoss(s,[T],[1])),"mix",f(d.multiTeacherLoss(s,[T,C],[0.5,0.5])));
  console.log("row2",[[0.5,0.5],[0.95,0.9],[0.98,0.95],[0.97,0.97],[0.05,0.3]].map(([t,s])=>f(d.opdTokenReward(t,s))));
});'
```
Output (2026-10-07): unsure → traces 0.9163, forward 0.5511, reverse 0.7535, rewards [+0.8109, −1.7918,
−1.8971, −1.6094], math 0.7535, chat 0.0981, mix 0.4258 · confident and wrong → 2.3026, 1.7772, 1.9090,
[+2.1972, −2.6391, −1.2040, −1.6094], math 1.9090, chat 0.6978, mix 1.3034 · close → 0.1625, 0.0112,
0.0127, [+0.0572, −0.3365, −0.5108, −0.4055], math 0.0127, chat 0.1518, mix 0.0822 · row 2 → [0, 0.0541,
0.0311, 0, −1.7918]. GRPO's −0.5774 for row 2 is from `rlvr-grpo`'s reproducer.

**`math/distill.js` signatures** (owner: `distillation`; pure, no DOM; never mutates inputs; tests first):
```js
// On-policy distillation reward for one sampled token: ln pTeacher − ln pStudent, clipped to ±clip
// (Kimi K3 clips; default no clip).
opdTokenReward(pTeacher: number, pStudent: number, { clip = Infinity } = {}) → number
//   (0.90, 0.40) → 0.8109 · (0.05, 0.30) → −1.7918 · (0.97, 0.97) → 0 · (0.05, 0.30, { clip: 1 }) → −1
//   throws RangeError unless both probabilities are > 0

// Σ w_i · KL(student ‖ teacher_i), the DeepSeek-V4 form. Weights ≥ 0 and summing to 1.
multiTeacherLoss(student: number[], teachers: number[][], weights: number[]) → number
//   ([0.85,0.07,0.05,0.03], [[0.90,0.05,0.03,0.02]], [1])                        → 0.0127
//   ([0.85,0.07,0.05,0.03], [[0.90,0.05,0.03,0.02],[0.60,0.20,0.10,0.10]], [0.5,0.5]) → 0.0822
//   throws on a weight/teacher count mismatch, a negative weight, or weights not summing to 1
```
Reuses `tokenLoss` and `klDivergence` from `math/lm.js` (owner `pretraining`; this page's worked examples
0.9163, 0.5511 and 0.7535 are listed there).

Test cases for the builder: the expected `opdTokenReward` under the student's own distribution equals
`−klDivergence(student, teacher)` (0.7535 here); `opdTokenReward` is 0 when the two probabilities agree;
`multiTeacherLoss` with one teacher and weight 1 equals `klDivergence(student, teacher)`; it is 0 only when
the student equals every teacher with positive weight.

## 12. Open questions for the reviewer
**Data-pass keys** (new): `deepseek-v4-pro.opd` · `kimi-k3.opd` · `mimo-v2-flash` (new entry) `.opd` ·
`glm-5.opd` · `deepseek-r1` (new entry) `.sft_samples` = 800000 (`[BG]`; the reviewer should decide whether
a BG count may become a data entry, or stay as "about 800K" with the arXiv link) · `olmo-3.sft_traces` =
2.3e6 (reported) · `mistral-large-4.distillation_note` (reported) · `papers.thinking-machines-opd`
(reported; same "paper-level fact" question as `agentic-rl`'s FP16 row).

**Graph changes:**
- `distillation` is a leaf. Spec §3.4 lists `rlvr-grpo → agentic-rl → distillation` and nothing after it, so
  this is consistent; no change proposed.

**Judgment calls:**
- **Four-candidate KL.** All KLs on the page are over four candidates, not a full vocabulary; the stage line
  and math shapes say so.
- **Frame 6's prefix rewards** (+0.05, +0.03) are stand-ins chosen so teacher and student nearly agree on
  the correct steps; the point (blame concentrates on `54`) holds for any prefix where they agree.
- **"Student can at best match its teachers"** (frame 9, takeaway 3) is a first-principles reading of the
  reward (it is maximized when the student equals the teacher, try-this 2), not a brief quote. If the
  reviewer wants a source line, the alternative wording is "the reward stops pushing once the student
  matches the teacher".
