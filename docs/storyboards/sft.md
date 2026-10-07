# Supervised fine-tuning (`sft`)

Track: training · Section: recipe · Prereqs: pretraining
Next: `rlhf-dpo` (the one slug whose `prereqs` list `sft`)
Status: draft
Sources: 02 §3 (all bullets), §4.3 (R1 cold start), §4.6 (observations excluded from the loss), §5 step 3,
§7 (SFT loss masking and chat-template rows) · 05 §1.1, §1.2

Running example: the course's RL prompt, `What is 7 × 8 ?`, now as one short training conversation with
a chat template, a thinking block that contains a mistake and its correction, one tool call, the tool's
reply and a final answer. 26 tokens, whitespace-split, in eight segments (§5). The same prompt is graded
by a checker on `rlvr-grpo`; this page shows how the model first learns the format it will be graded in.
A visible line under the stage says: "The template tags here are generic stand-ins; real ones differ per
lab (§8). The base model's continuation in frame 1 is illustrative."

## 1. Learning objective
After this page you can explain how supervised fine-tuning turns a base model that only continues text
into one that answers in turns: conversations are wrapped in a chat template with special tokens for each
role, reasoning goes between `<think>` tags, and the training loss is the same next-token cross-entropy as
pretraining but counted only on the assistant's tokens (prompts, template tags and tool outputs masked;
GLM-5 also masks the erroneous part of a trace it keeps); and why this stage is the "cold start" that RL
needs.

## 2. Misconceptions to correct
- **Misconception:** "The model learns from the whole conversation." → **Reality:** the loss counts
  only the assistant's tokens. The user's words and the template tags are context, not targets: here 15 of
  26 tokens are trained. Train on all 26 and the model also learns to write the user's side. Corrected by
  frames 4–5 and try-this 1. (source: 02 §3 "standard SFT computes loss only on assistant tokens")
- **Misconception:** "Showing a mistake in training data teaches the model to make it." → **Reality:**
  only if the mistake is trained. GLM-5 keeps erroneous segments in its agent traces but masks them out of
  the loss, so the model reads the error and the recovery but is only trained to produce the recovery: 15 →
  10 trained tokens here. Corrected by frame 7 and try-this 3. (source: 02 §3 GLM-5, confirmed)
- **Misconception:** "`<think>` is a special part of the architecture." → **Reality:** it is a pair of
  ordinary tokens in the vocabulary. The model learns to open them, reason, close them and then answer,
  because the SFT data does exactly that; products then switch thinking modes by template and training, not
  by a different network. Corrected by frame 3. (source: 02 §3 chat templates, thinking modes)
- **Misconception:** "SFT is where the model learns to reason well." → **Reality:** SFT mostly teaches
  format and gives a readable starting point; it can only imitate its data. The big reasoning gains come
  from RL, which in turn needs SFT's "cold start": RL can only reinforce answers the model already
  sometimes produces. Corrected by frame 9. (source: 02 §3 cold start, Kimi K3 quote; `rlvr-grpo`
  misconception 5)

## 3. Hook and intuition (final wording)
**Hook:** A pretrained model, asked "What is 7 × 8?", might just write another question. How do a few
hundred thousand examples teach it to answer, to think first, and to call tools?

A base model has read trillions of tokens, so it knows a great deal, but it only knows how to continue
text. Supervised fine-tuning (SFT) shows it what the next part of a *conversation* looks like. Every
example is wrapped in a **chat template**: special tokens that mark who is speaking (system, user,
assistant, tool), so the model can tell a question it was asked from an answer it should write. Reasoning
models add a `<think>` … `</think>` block before the answer. These are ordinary tokens; the model learns to
use them because every example does.

The loss is the same next-token cross-entropy as in pretraining, with one change: a **loss mask**. Only the
assistant's tokens count. The prompt, the template tags and anything a tool returned are there as context
but are never targets, so the model learns to write answers and tool calls, not to imitate the user or to
invent tool results. GLM-5 adds a refinement for agent traces: when an example contains a mistake followed
by a recovery, the mistake stays in the text but is masked, so the model sees how to recover without being
trained to make the error.

The data is the hard part. In 2026 it comes from earlier specialist models, from rejection sampling (keep
only the samples that pass a check), and from runs in real tool environments, often with long reasoning
traces. DeepSeek-R1 used about 800K examples; Olmo 3 about 2.3 million reasoning traces. The result is a
model that answers in the right format and sometimes reasons its way to the right answer. That "sometimes"
is the point: it is the **cold start** that RL needs, because RL can only reinforce what the model already
occasionally does. The cost is that SFT only imitates: it is bounded by its data, and wrong traces that are
not filtered or masked are learned like right ones.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `token` (every token of the transcript as a chip;
`state: 'active'` = trained, and `hatched` = masked out of the loss; template tags drawn as chips with the
`dim` state so they read as markup), `block` (base model, SFT model, tool sandbox; `active` vs `dim`),
`flow` (carry `token` from the model to the call and from the tool back), plain labeled text marks (role
labels `user`, `assistant`, `tool` at line starts; readouts `trained 15 of 26`; the per-lab template table).

Layout: the 26 chips wrap into one line per segment group, like a chat transcript (5 lines, the longest
line 9 chips); no line exceeds the stage width.

New glyphs proposed: none.

Encoding: "trained vs masked" is encoded once, by the hatch (masked) vs plain (trained) chip; color is not
used for it. The followed token is the final `56` (selection outline in frames 3–9).

Terms introduced (one per frame): base model (assumed, recalled in 1) · chat template (2) · `<think>` tags
(3) · loss mask (4) · none new (5) · tool call and observation (6) · masked error segment (7) · rejection
sampling (8) · cold start (9).
Terms assumed from prereqs: next-token cross-entropy, token, vocabulary (`pretraining`); the six pipeline
stages (`training-pipeline`).

Indexing: tokens are counted, not numbered; no 0-based addresses on this page.

## 5. Animation script
The transcript (kind → tokens):
1. template `<user>` · 2. user `What is 7 × 8 ?` (6) · 3. template `<assistant>` · 4. assistant `<think>` ·
5. error `7 × 8 = 54` (5) · 6. assistant `wait , check </think> <call> calc(7*8) </call>` (7) ·
7. observation `<obs> 56 </obs>` (3) · 8. assistant `56 <end>` (2). Total 26.

Thread order: frames 1–3 the format; 4–7 the loss mask; 8–9 where the data comes from and what it is for.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | A `block` "base model". Prompt chips `What is 7 × 8 ?`. The model's continuation types in: `What is 9 × 6 ? What is 4 × 7 ?` (plain label `illustrative`). | The continuation chips appear left to right. | A pretrained base model only continues text. Asked a question, it may write more questions, because worksheets full of questions are common text. | – |
| 2 | The prompt again, now wrapped: `<user>` before it and `<assistant>` after it, both `dim` chips. Role labels `user` and `assistant` at the line starts. | The two tag chips slide in from the sides. | A chat template wraps each turn in special tokens that mark who is speaking. The model can now tell a question it was asked from an answer it should write. | 2 template tokens |
| 3 | The assistant line fills: `<think> 7 × 8 = 54 wait , check </think> … 56 <end>` (tool chips shown dim for now). | Chips type in; the `<think>` and `</think>` chips briefly lift. | Reasoning models put their working between think tags before the answer. The tags are ordinary tokens; the model uses them because every example does. | – |
| 4 | The whole transcript (5 lines). The 8 template and user chips take the hatch, with the plain label `context only`. | The hatch sweeps over `<user> What is 7 × 8 ? <assistant>`. | A loss mask decides which tokens are targets. The prompt and the template tags are context only, never targets. | masked 8 (template 2, user 6) |
| 5 | Same transcript; the trained chips are plain. A readout `trained 15 of 26` and a plain line `same loss as pretraining, on these tokens only`. | The readout counts up chip by chip over the unhatched tokens. | The loss is the same next-token cross-entropy as pretraining, counted only on the assistant's tokens. | trained 15 · masked 11 · 15 / 26 = 57.7% |
| 6 | The tool lane: `<call> calc(7*8) </call>` flows to a `block` "tool"; `<obs> 56 </obs>` flows back and takes the hatch, with the plain label `tool output: masked`. | `flow` (carry `token`) out and back; the observation chips hatch. | The model's tool call is trained. The tool's reply is masked, so the model learns to call tools without learning to invent their results. | call 3 tokens trained · observation 3 masked (already counted in the 11) |
| 7 | The error segment `7 × 8 = 54` (5 chips) highlighted with the plain label `mistake kept in the text`. Readout changes `trained 15 → 10`. | The five chips take the hatch; the readout counts down. | GLM-5 keeps mistakes in its agent traces but masks them. The model reads the error and the recovery, and is trained only on the recovery. | trained 10 of 26 = 38.5% · masked 16 |
| 8 | A `block` "earlier specialist" producing 4 candidate traces; a checker stamps ✓ (`verdict`) on two and ✗ on two; only the ✓ traces flow into a pile labeled `SFT data`. | Traces appear; verdicts stamp; the ✗ traces fade. | Most 2026 SFT data comes from other models. Rejection sampling keeps only the traces that pass a check, often on problems the previous model found hard. | 2 of 4 kept (illustrative) · DeepSeek-R1 ~800K examples · Olmo 3 ~2.3M traces (reported) |
| 9 | **Key frame.** The SFT model `block` with the transcript beside it, followed `56` outlined; an arrow (`flow`, carry `token`) to a dim pipeline strip where the `RL` stage lights. A plain line: `sometimes right → RL has something to reinforce`. | The arrow draws; the RL stage lights. | SFT gives RL its cold start: a model that answers in format and sometimes reasons its way to the right answer. RL can only reinforce what it already does sometimes. | – (the k = 0 case of `rlvr-grpo` frame 6: no right answers, no signal) |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 4–7 draw `lossMask` / `maskSummary` (`math/sft.js`) on the transcript above.
Caption counts (words/sentences, `rlvr-grpo`'s counter, run 2026-10-07): 23/2, 30/2, 24/2, 19/2, 15/1,
24/2, 25/2, 26/2, 30/2.

## 6. Toy
"Which tokens teach?" The 26-token transcript with three mask switches and live counts. A visible line
under the toy: "Tags are generic stand-ins; each lab's template differs."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `maskPrompt` | Mask the prompt and template tags | Toggle | on / off | on | – |
| `maskObservation` | Mask the tool's reply | Toggle | on / off | on | – |
| `maskError` | Mask the mistake in the trace (GLM-5) | Toggle | on / off | off | – |
| `template` | Show the template as | Preset chips | `generic` / `DeepSeek-V4 <think>` / `gpt-oss harmony roles` | `generic` | label-only: changes the tag text, never the counts |

Selection: tapping or arrow-keying to a chip shows, in a visible panel under the transcript, its segment
kind and whether it is trained; nothing is hover-only.

**Live outputs** (printed)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Chip hatch per token | `lossMask(segments, { maskPrompt, maskObservation, maskError })` | hatch = masked |
| Trained / masked counts and share | `maskSummary(...)` → `trained`, `masked`, `trained / total` | integers; percent, 1 decimal |
| Per-kind breakdown | `maskSummary(...).byKind` | `template 0/2 · user 0/6 · assistant 10/10 · error 5/5 · observation 0/3` (trained/total) |

**Try this** (each leads to a named insight)
1. Turn `maskPrompt` off: trained 15 → 23 of 26 (88.5%), and the 6 user tokens plus 2 tags are now
   targets. → **Insight: without the mask, the model is also trained to write the user's side,** so it
   spends capacity imitating questions instead of answering them.
2. Turn `maskObservation` off (prompt mask back on): trained 15 → 18, with `<obs> 56 </obs>` now a target.
   → **Insight: training on tool outputs teaches the model to predict results it should have waited
   for.** That is how a model learns to invent a tool's answer; GLM-5 and others exclude observations,
   and agentic RL does the same (`agentic-rl`).
3. Turn `maskError` on: trained 15 → 10 (38.5%); the five mistake chips stay visible but hatched. →
   **Insight: masking is how you show a mistake without teaching it.** The model still reads `7 × 8 = 54`
   as context and is trained to write `wait , check` and the tool call after it.

Lesson-17 check: the three insights follow from the counts and which kinds flip (reproducer); the transcript
is a stand-in but the masking rules are the documented ones.

## 7. Show me the math
```tex
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t \htmlClass{hl-m}{m_t}} \sum_{t} \htmlClass{hl-m}{m_t} \ln p_\theta(x_t \mid x_{<t}),
\qquad
\htmlClass{hl-m}{m_t} = \begin{cases} 1 & x_t \text{ written by the assistant} \\ 0 & \text{prompt, template, tool output, masked error} \end{cases}
```
Shapes: x [T] (T = 26 here); m [T] boolean; the denominator counts trained tokens (15, or 10 with the
error masked). Panel notes: (a) everything else is pretraining's loss (`pretraining`, §7); the mask is the
only change. (b) Some labs average per example instead of per token; the page uses the token mean.
(c) Thinking modes (DeepSeek-V4 Non-think / Think High / Think Max; Kimi K3 low / high / max; gpt-oss low / medium /
high) are selected by template and trained with different length budgets later, in RL. Color links: `hl-m` =
the plain (trained) chips vs hatched (masked) chips.

## 8. In today's models (Oct 2026)
All keys proposed for the data-extension pass; entry ids as accepted in `rlvr-grpo` §13.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4: `<think>…</think>` plus an XML-style `|DSML|` tool-call schema (XML reduced escaping errors) | `models.deepseek-v4-pro.chat_template` = "<think> tags; |DSML| XML tool calls" (confirmed) | 02 §3 |
| gpt-oss "harmony" format: roles System > Developer > User > Assistant > Tool | `models.gpt-oss-120b.chat_template` = "harmony (System > Developer > User > Assistant > Tool)" (confirmed) | 02 §3 |
| Kimi K3 XTML template; GLM-5 template with interleaved, preserved and turn-level thinking | `models.kimi-k3.chat_template` = "XTML", `models.glm-5.chat_template` = "interleaved / preserved / turn-level thinking" (confirmed) | 02 §3 |
| GLM-5 retains erroneous segments in agent trajectories but masks them out of the loss; environment/tool outputs are excluded from the loss | `models.glm-5.sft_masking` = "erroneous segments masked; tool outputs excluded" (confirmed) | 02 §3 |
| Kimi K3: "SFT stage establishes a high-quality cold-start policy for the subsequent RL stage"; trajectories from earlier Kimi specialists plus multi-stage verification and human annotation | `models.kimi-k3.sft_data` (confirmed) | 02 §3 |
| GLM-5 builds SFT data by rejection sampling, filtering to problems its previous model finds hard | `models.glm-5.sft_data` = "rejection sampling on hard problems" (confirmed) | 02 §3 |
| DeepSeek-R1 (2025): ~800K SFT samples (600K reasoning + 200K non-reasoning) | `models.deepseek-r1.sft_samples` = 800000 (`[BG]`, arXiv 2501.12948; proposed by `distillation`) | 02 §3 |
| Olmo 3: ~2.3M reasoning traces distilled from QwQ-32B and DeepSeek-R1 | `models.olmo-3.sft_traces` = 2.3e6 (reported; proposed by `distillation`) | 02 §3 |
| Thinking modes: DeepSeek-V4 Non-think / Think High / Think Max; Kimi K3 low / high / max; gpt-oss low / medium / high | `models.deepseek-v4-pro.thinking_modes`, `models.kimi-k3.thinking_modes`, `models.gpt-oss-120b.thinking_modes` (confirmed) | 02 §3 |
| DeepSeek-V3.2 uses a cold start to merge reasoning with tool use ("thinking in tool-use"); Qwen3: long-CoT cold start → reasoning RL → thinking-mode fusion → general RL | `models.deepseek-v3.2.sft_cold_start` (confirmed); `models.qwen3.post_training` (reported) | 02 §3 |

## 9. Takeaways
1. SFT wraps conversations in a chat template (role tokens, `<think>` tags, tool-call tags) and trains with
   pretraining's own loss, counted only on the assistant's tokens.
2. The loss mask is the whole trick: prompts, tags and tool outputs are context, never targets, and GLM-5
   also masks a kept mistake so the model learns the recovery, not the error.
3. SFT data mostly comes from other models and rejection sampling; the stage teaches format and gives RL its
   cold start, and it can only imitate what its data shows.

## 10. Next and go deeper
Next: `rlhf-dpo` (learning from preferences, starting from this SFT model as the reference).
Later on the same line: `rlvr-grpo` (where the cold start is used).
Go deeper: Andrej Karpathy, *nanochat*, whose SFT stage shows a real chat template and loss mask in code
(https://github.com/karpathy/nanochat) · Nathan Lambert, *RLHF Book*, instruction-tuning chapter
(https://rlhfbook.com/)

## 11. Key-frame sketch
Frame 7 (the mistake is kept and masked), desktop width. `[/x/]` = hatched (masked) chip; `{x}` = dim
template chip. Checked against the reproducer below (trained 10 of 26 = 38.5%).

```text
 user       {<user>}[/What/][/is/][/7/][/×/][/8/][/?/]
 assistant  {<assistant>}[<think>]
            [/7/][/×/][/8/][/=/][/54/]   ← mistake kept in the text
            [wait][,][check][</think>][<call>][calc(7*8)][</call>]
 tool       [/<obs>/][/56/][/</obs>/]
 assistant  [56][<end>]
                ▲ followed
 trained 10 of 26 (38.5%)   masked 16
 GLM-5 keeps mistakes in its agent traces but masks them.
 The model reads the error and the recovery, and is
 trained only on the recovery.
 [Back] [Pause] [Next]  ━━━━━━●━━  7 / 9   1×
```
(`{<user>}` and `{<assistant>}` are template chips and are also hatched; the sketch abbreviates.)

**Reproducer** (every count in §2, §5, §6 and this sketch; run 2026-10-07 against a scratchpad draft of
`math/sft.js` matching the signatures):
```sh
node -e '
import("./math/sft.js").then((s)=>{
  const SEG=[{kind:"template",tokens:["<user>"]},{kind:"user",tokens:["What","is","7","×","8","?"]},{kind:"template",tokens:["<assistant>"]},
    {kind:"assistant",tokens:["<think>"]},{kind:"error",tokens:["7","×","8","=","54"]},
    {kind:"assistant",tokens:["wait",",","check","</think>","<call>","calc(7*8)","</call>"]},{kind:"observation",tokens:["<obs>","56","</obs>"]},{kind:"assistant",tokens:["56","<end>"]}];
  for (const o of [{},{maskPrompt:false},{maskObservation:false},{maskError:true}]) { const r=s.maskSummary(SEG,o);
    console.log(JSON.stringify(o),r.total,r.trained,r.masked,(100*r.trained/r.total).toFixed(1)); }
  console.log(s.lossMask(SEG).map((b)=>b?1:0).join(""));
});'
```
Output (2026-10-07): default → 26 total, 15 trained, 11 masked, 57.7% · maskPrompt off → 23, 3, 88.5% ·
maskObservation off → 18, 8, 69.2% · maskError on → 10, 16, 38.5% · mask string
`00000000111111111111100011`.

**`math/sft.js` signatures** (owner: `sft`; pure, no DOM; never mutates inputs; tests first):
```js
// Segment kinds: 'template' | 'user' | 'assistant' | 'error' | 'observation'.
// assistant: always trained · template, user: trained only if maskPrompt is false ·
// observation: trained only if maskObservation is false · error: trained unless maskError.
lossMask(segments: { kind: string, tokens: string[] }[],
         { maskPrompt = true, maskObservation = true, maskError = false } = {}) → boolean[]
//   the §5 transcript → 26 entries, 15 true: 00000000111111111111100011
//   [{ kind: 'user', tokens: ['hi'] }, { kind: 'assistant', tokens: ['hello'] }] → [false, true]
//   throws RangeError on an unknown kind

maskSummary(segments, options) → { total, trained, masked, byKind: { [kind]: { trained, masked } } }
//   §5 transcript, defaults        → { 26, 15, 11, … assistant { 10, 0 }, error { 5, 0 } }
//   §5 transcript, maskError: true → { 26, 10, 16, … error { 0, 5 } }
```
The loss itself is `meanLoss(probs, lossMask(...))` from `math/lm.js` (owner `pretraining`).

Test cases for the builder: `lossMask` length equals the total token count; assistant tokens are true under
every option; `maskSummary.trained + masked = total`; turning any option from masked to unmasked never
decreases `trained`; inputs are not mutated.

## 12. Open questions for the reviewer
**Data-pass keys** (new): `deepseek-v4-pro.chat_template`, `.thinking_modes` · `gpt-oss-120b.chat_template`,
`.thinking_modes` · `kimi-k3.chat_template`, `.sft_data`, `.thinking_modes` · `glm-5.chat_template`,
`.sft_masking`, `.sft_data` · `deepseek-v3.2.sft_cold_start` · `qwen3.post_training` (reported; entry
`qwen3` proposed by `pretraining`). `deepseek-r1.sft_samples` and `olmo-3.sft_traces` are proposed on
`distillation`.

**Graph changes:**
- Curriculum order puts `midtraining` before `sft`, but `sft` lists only `pretraining`. That is fine: SFT
  does not depend on context extension. No change proposed (see `midtraining` §12 for the leaf question).

**Judgment calls:**
- **Frame 1's base-model continuation** is illustrative, labeled as such, and not from a brief; the claim it
  supports ("a base model only continues text") is first principles.
- **Generic tags.** The page's `<user>`/`<assistant>`/`<call>`/`<obs>` tags are stand-ins; the `template`
  chips switch only the tag text to the documented DeepSeek-V4 and gpt-oss forms, never the counts.
- **Template tags counted as masked.** Some recipes train the closing end-of-turn token; this page trains
  `<end>` (it is the assistant's) and masks the opening role tags.
