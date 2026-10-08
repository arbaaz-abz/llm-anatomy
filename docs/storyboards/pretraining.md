# Pretraining (`pretraining`)

Track: training · Section: recipe · Prereqs: training-pipeline
Next: `scaling-laws`, `midtraining`, `sft` (the three slugs whose `prereqs` list `pretraining`)
Status: approved (expert review)
Sources: 02 §0 item 1, §1.1, §1.2, §1.3, §1.8, §5 step 1, §7 (cross-entropy and tokenization rows) · 05
§1.1, §1.2 · spec §12 (tokenizer: one section of this page, no deep dive)

Running example: the course-wide sentence from `attention` and `decoder-anatomy`, extended by four
tokens: `The cat sat down on the mat .` (8 tokens, positions 1–8). Position 4's prediction ("down" → next
is "on") uses `decoder-anatomy` frame 8's exact numbers (logits on 2.0 · "." 1.5 · and 0.5 · the 0.0 · 12
others −1.0 → p(on) = 0.390). The other six probabilities are hand-picked stand-ins.

This page owns `math/lm.js` (per-token loss, mean loss, perplexity, KL divergence), which `sft`,
`rlhf-dpo` and `distillation` reuse.

## 1. Learning objective
After this page you can say what a model is graded on during pretraining (minus the log of the
probability it gave the true next token, averaged over every position of every sequence), compute that
loss and its perplexity for a short sentence by hand, explain why one 8-token sequence yields 7 graded
predictions in one pass, name the steps that turn raw crawl into a training mix (rule and classifier
filters, deduplication, a reweighted mixture, rephrased data), and place 2026 budgets of 25–33 trillion
tokens.

## 2. Misconceptions to correct
- **Misconception:** "The loss only checks whether the model's top guess is right." → **Reality:** the
  loss is −ln p of the true token. "on" is the model's top guess at 0.390 and still costs 0.941; had it
  been 0.99 it would cost 0.01. A confident miss costs far more: at p = 0.01 one token's loss is 4.61.
  Corrected by frame 3 and try-this 1–2. (source: 02 §1.1 objective)
- **Misconception:** "Each document is one training example." → **Reality:** with a causal mask, every
  position predicts its next token in the same forward pass, so 8 tokens give 7 graded predictions, and a
  33-trillion-token run grades about 33 trillion predictions. Corrected by frame 4. (source: first
  principles; causal mask from `attention`)
- **Misconception:** "More data just means crawling more of the web." → **Reality:** most of the work is
  removing and reshaping: rule and classifier filters, deduplication, reweighting domains (code, math and
  STEM upsampled), rephrasing knowledge and math text, and filtering templated machine-made text out.
  Corrected by frames 6–9. (source: 02 §1.2)
- **Misconception:** "A token is a word." → **Reality:** a tokenizer splits text into pieces from a fixed
  vocabulary of about 129,000–201,000 entries; common words are one piece, rare words several. A bigger vocabulary
  means fewer tokens per document but a larger embedding table: 163,840 × 7,168 ≈ 1.17B parameters
  in Kimi K3. Corrected by frame 1. (source: 02 §1.1 vocab table; `models.kimi-k3.vocab_size`, `.d_model`)

## 3. Hook and intuition (final wording)
**Hook:** A pretrained model is only ever asked to guess the next token. Why does that, repeated over 30
trillion tokens, produce something that knows chemistry, grammar and Python?

Pretraining shows the model text and, at every position, asks for a probability for every possible next
token. The grade is simple: take the probability it gave the token that actually came next, and score
minus its logarithm. A confident right guess costs almost nothing, an unsure one costs a little, and a
confident wrong one costs a lot. The average of that score over all positions is the **cross-entropy
loss**, and training nudges every weight to lower it.

Nothing in that grade mentions facts or grammar. But text is full of them, and the cheapest way to
assign high probability to the next token of a chemistry paper, a legal contract or a Python file is to
have absorbed how chemistry, law and Python work. Across tens of trillions of tokens there is no shortcut
left: lowering the loss further requires knowing more. That is why the objective is still plain
next-token cross-entropy in 2026, with one common addition: several labs also predict a second token
ahead (multi-token prediction) as an extra loss.

Two things make this affordable. Every position in a sequence is graded in the same forward pass, so one
8-token sentence is 7 training examples. And the text is chosen with great care: raw crawl is filtered by
rules and by quality classifiers, near-duplicates are removed, domains like code and math are upsampled,
and some knowledge and math text is rephrased by other models. The open frontier MoEs reported in 2026 read 25–33
trillion tokens this way (Kimi K3 did not say; smaller open models read far less, Olmo 3 about 5.9T, reported). The cost is that this stage is by far the largest in tokens and usually in
compute; how to split that compute between model size and tokens is the next page, `scaling-laws`.

## 4. Visual metaphor
Glyphs used (spec §5.1 and the built library): `token` (the 8 chips; `fill` = `valueColor(loss)` on the
loss strip), `vector` (`NUMBER_CELL` rows: the 5-cell probability row of frames 2–3 and the 7-cell loss
row of frames 4–5; 7 × 43 = 301 px), `cell` (single printed numbers), `block` (the model, and the data
pipeline stages in frames 6–9, `active` vs `dim`), `flow` (carry `token` through the pipeline; carry
`activation` from chips to the model), `heatmap` (frame 4, the 8 × 8 causal mask with masked cells hatched,
reused from `attention`; numbers are not printed in it, they are in the loss row below), plain labeled
text marks (`mean`, `perplexity`, pipeline facts, the budget table).

Lesson 18: `decoder-anatomy`'s 16-word probability row (16 × 43 = 688 px) does not fit; frames 2–3 show
five named cells `on · "." · and · the · 12 others (0.019 each)`.

New glyphs proposed: none.

Stand-ins (visible line under the stage): "Position 4's probabilities are the exact softmax from the
`decoder-anatomy` page; the other six probabilities are hand-picked stand-ins. The losses are exact for
those numbers."

Terms introduced (one per frame): tokenizer and its vocabulary (1) · next-token target (2) · cross-entropy
loss (3) · none new: the causal mask returns from `attention` (4) · perplexity (5) · quality classifier (6) ·
deduplication (7) · data mixture (8) · rephrased (synthetic) data (9) · token budget (10).
Terms assumed from prereqs: token, embedding, logits, softmax, residual stream, block (`decoder-anatomy`);
causal mask (`attention`); the six pipeline stages (`training-pipeline`).

Indexing: token positions are 1-based on screen (The₁ … .₈); there are no 0-based addresses.

Selection: the followed prediction is position 4 → "on" (frames 2–5); it carries the selection outline
wherever it appears. In the toy the selected position carries it.

## 5. Animation script
Thread order: frame 1 the input; frames 2–5 the objective; frames 6–9 the data; frame 10 the budget.

Per-position targets and probabilities (frames 4–5, toy default), positions 1–7 predict 2–8:
`cat 0.10 · sat 0.25 · down 0.30 · on 0.390 · the 0.60 · mat 0.45 · . 0.80`.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | A plain text line `The cat sat down on the mat.` above; below it, 8 `token` chips with subscripts 1–8. A plain mark `vocabulary: 163,840 pieces (Kimi K3)`. A plain line under the stage: `Tiktokenizer (go deeper) shows real BPE boundaries on your own text.` | The text line splits at the piece boundaries and each piece drops into a chip; the period becomes its own chip. | A tokenizer cuts text into pieces from a fixed vocabulary: whole common words, fragments of rare ones. Kimi K3's vocabulary has 163,840 pieces. | 8 tokens · vocabularies 129K–201K (DeepSeek-V4 to gpt-oss; printed from `models.deepseek-v4-pro.vocab_size` and `models.gpt-oss-120b.vocab_size` through the `count` placeholder format, ruling P3-R18) · embedding table Kimi K3 `≈ 163,840 × 7,168 ≈ 1.17B` params · visible note: "Bigger vocabularies mean fewer tokens per document but a bigger table." |
| 2 | Chips 1–4 lit, 5–8 dim. A `block` "model" over them. Out of position 4 comes a `vector` of 5 printed cells: `on 0.390 · . 0.237 · and 0.087 · the 0.053 · 12 others 0.019 each`. Chip 5 `on` gets a plain label `true next token`. | A `flow` (carry `activation`) runs from chip 4 into the model and out to the probability row; the `on` cell and chip 5 take the selection outline. | At each position the model gives a probability to every possible next token. Here, after "down", the true next token "on" got 0.390. | probabilities from `decoder-anatomy` frame 8 · Σ = 1.000 |
| 3 | The `on` cell alone, enlarged, with a readout `loss = −ln 0.3903 = 0.941` (the probability to four places, so the readout checks by hand). Beside it two plain reference marks: `if 0.99: 0.010` and `if 0.01: 4.605`. | The readout types in; then the two reference marks fade in. | The grade is the cross-entropy loss: minus the log of the probability given to the true token. Unsure costs a little; confident and wrong costs a lot. | `−ln 0.3903 = 0.941` · `−ln 0.99 = 0.010` · `−ln 0.01 = 4.605` |
| 4 | All 8 chips lit. The 8 × 8 causal mask `heatmap` (future hatched; 16 px cells, nothing printed in it) at the left. Layout check at 580 × 366: chip row on top; heatmap 8 × 17 = 136 px wide beside the 7-cell loss row, 7 × 43 = 301 px; 136 + 16 + 301 = 453 px wide, 24 + 12 + 136 px tall. Under the chips, a 7-cell loss `vector` (positions 1–7, value-colored, printed). | The mask's rows light one after another; each lit row drops one loss cell into the strip under its position. | One pass grades every position at once: the causal mask lets each position see only the past. Eight tokens give seven graded predictions. | losses `2.303 · 1.386 · 1.204 · 0.941 · 0.511 · 0.799 · 0.223` |
| 5 | The loss strip with a plain readout `mean 1.052` and `perplexity 2.86`; a plain mark `uniform guess over 16 words: perplexity 16`. | The seven cells slide together into the mean readout; the perplexity readout types in. | Training lowers the mean loss over all positions. Its exponential, perplexity, is how many equally likely tokens the model is effectively choosing between: about 2.9 here. | `mean = 7.366 / 7 = 1.052` · `perplexity = e^1.052 = 2.86` · `ln 16 = 2.773 → 16` |
| 6 | A pipeline row of `block`s: `crawl` → `rules` → `quality classifier` → `dedup` → `mixture` → `rephrase` → `packed sequences`, with only `crawl`, `rules` and `quality classifier` active. Plain fact line: `Kimi K3: rule-based heuristics, classifier quality scoring, dedup, per domain`. | A stream of document `token` chips flows in from `crawl`; chips that fail `rules` or score low at the classifier drop out, dimmed. | Raw crawl is mostly not worth training on. Rules remove the obvious junk, and a quality classifier, itself a small model, scores the rest. | facts per §8 (no published keep ratios: "the reports give recipes, not ratios", brief 02 §1.2) |
| 7 | Same row, `dedup` active. Two identical document chips meet at the block. | One copy passes; the duplicate fades out. | Deduplication removes exact and near-copies, so the model does not spend its budget reading the same page many times. | no published counts |
| 8 | Same row, `mixture` active. Four plain labeled lanes `web · code · math · knowledge` leave it, code and math lanes drawn with more chips. Fact line: `MiniMax-M2: code, mathematics and STEM significantly upsampled`. | Chips sort into lanes; the code and math lanes fill faster. | The surviving data is mixed by domain, with code and math upsampled. Kimi K3 set its per-domain rates with small-model experiments. | facts per §8 · `GLM-5: ~10M issue–PR pairs, ~160B tokens, for software engineering` |
| 9 | Same row, `rephrase` active. A knowledge-domain chip enters and leaves as two differently worded chips. A plain fact line: `DeepSeek-V4 filters out batched auto-generated and templated text`. | One chip splits into two rephrased chips; a separate templated chip is dropped at the classifier. | Some knowledge and math text is rephrased by another model, checked against its source, and added back. Mass-produced templated text is filtered out instead. | facts per §8 (Kimi K3 rephrasing with fidelity checks; DeepSeek-V4 templated-content filter) |
| 10 | **Key frame.** A plain text table: `DeepSeek-V4-Pro 33T · DeepSeek-V4-Flash 32T · MiniMax-M2 29.2T · GLM-5 28.5T · MiMo-V2-Flash 27T · Nemotron 3 Super 25T · Kimi K3 not disclosed`, with `Llama 3.1 405B (2024) 15.6T` dim at the bottom. | Rows type in top to bottom. | 2026 open frontier models pretrain on 25 to 33 trillion tokens. It is the biggest stage by far, and how to split that budget is the next page's question. | as listed · range 25T–33T |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 2 and 4–5 draw `softmax` (`math/core.js`) and `tokenLoss`, `meanLoss`, `perplexity`
(`math/lm.js`). Caption counts (words/sentences, `rlvr-grpo`'s counter widened to two-digit frames, run
2026-10-07, re-run after review): 24/2, 23/2, 27/2, 23/2, 26/2, 24/2, 19/1, 21/2, 24/2, 29/2.

## 6. Toy
"Grade a sentence." The 8-token sentence, its 7 per-position probabilities, the per-token losses, mean loss
and perplexity, live.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `pos` | Which prediction (select a target chip 2–8) | Selection by click, tap or arrow keys: each target chip a focusable `<g role="button" tabindex="0" aria-label="position n: token">`, Enter / Space selects, arrow keys move, visible focus, selection drawn with `G.selectionMark` (ruling P3-R17; this page builds the pattern that `training-pipeline` and `distillation` reuse) | positions 2–8 | 6 (`mat`) | – |
| `p` | Probability the model gave the selected true token | Slider | 0.01–1.00, step 0.01 | the stand-in for `pos` | chips `confident miss 0.01` · `uniform 1/16` · `certain 1.00` |
| `allUniform` | Make every position a uniform guess over 16 words | Button | – | – | sets all seven to 0.0625 |
| reset | Reset | Button | – | – | the stand-ins |

**Live outputs** (all printed)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Per-token loss strip (7 cells, value-colored) | `tokenLoss(p_i)` | 3 decimals |
| Mean loss | `meanLoss(probs)` | 3 decimals |
| Perplexity | `perplexity(meanLoss(probs))` | 2 decimals |
| Reference marks (fixed, visible) | `uniformLoss(16) = 2.773` · `uniformLoss(163840) = 12.007` (Kimi K3's vocabulary) | 3 decimals |

**Try this** (each leads to a named insight)
1. Predict first: select `mat` (0.45, loss 0.799). Which moves the mean more: making it certain, or making
   it a confident miss? Slide to 1.00: mean 1.052 → 0.938 (down 0.114). Slide to 0.01: its loss is 4.605
   and the mean jumps to 1.596; perplexity 2.86 → 4.93. → **Insight: the log punishes confident mistakes
   far more than it rewards certainty.** One badly wrong token moves the average almost five times as
   much (+0.544) as one perfect one does (−0.114).
2. Select `on`, the model's top guess at 0.390 (against 0.237 for the runner-up `.`). Its loss is 0.941, not 0. → **Insight: there is no
   credit for being top-1, only for probability.** The model keeps learning even on tokens it already
   ranks first, by making them more likely.
3. Press `uniform guess over 16`: every loss becomes 2.773, the mean is 2.773 and perplexity is exactly
   16. Now read the reference mark: a model that knows nothing about Kimi K3's vocabulary of 163,840 pieces
   would start near 12.01. → **Insight: perplexity is the effective number of choices.** Training a real
   model is the long walk from about ln(vocabulary size) down toward the loss of the text itself.

**Check my work** (default state; templated from `tokenLoss`, `meanLoss`, `perplexity` for any state: the
selected position's line first, then the seven losses at 4 decimals; mono, `aria-live="polite"`; this exact
text appears on the page):
```text
selected: mat   −ln 0.45 = 0.7985
sum   = 2.3026 + 1.3863 + 1.2040 + 0.9410 + 0.5108 + 0.7985 + 0.2231 = 7.3663
mean  = 7.3663 ÷ 7 = 1.0523 → 1.052
perplexity = e^1.0523 = 2.864 → 2.86
```

Lesson-17 check: each insight is read off `tokenLoss`/`meanLoss`/`perplexity` at the stated values (§11
reproducer). The 16-word "uniform" reference is exact for a 16-word vocabulary; the 163,840 mark is only a
starting reference, stated as "near".

## 7. Show me the math
```tex
\mathcal{L} = -\frac{1}{T-1}\sum_{t=2}^{T} \ln \htmlClass{hl-p}{p_\theta(x_t \mid x_{<t})},
\qquad
\text{PPL} = e^{\mathcal{L}}
```
```tex
\htmlClass{hl-p}{p_\theta(\cdot \mid x_{<t})} = \operatorname{softmax}(z_t),\quad z_t = W_U\, \operatorname{norm}(h_t) \in \mathbb{R}^{|V|}
\qquad
\text{uniform: } p = \tfrac{1}{|V|} \Rightarrow \mathcal{L} = \ln |V|
```
```tex
\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda_{\text{MTP}}\, \mathcal{L}_{\text{MTP}}
\qquad (\text{DeepSeek-V4: depth 1},\ \lambda = 0.3 \to 0.1 \text{ when LR decay starts})
```
Shapes (toy in parentheses): x [T] token ids (8); h_t [d_model]; W_U [d_model × |V|] (|V| = 16 here,
163,840 in Kimi K3); z_t, p [|V|]; the loss strip has T − 1 = 7 entries. The causal mask lets all T − 1
predictions come from one forward pass. Panel notes: (a) the brief writes the mean over T positions; with
a target for every position but the first, it is T − 1 = 7 here. (b) DeepSeek-V4 packs documents into
sequences and masks attention between them, so the loss never asks a document to predict its neighbor.
Color links: `hl-p` → the probability row (frames 2–3) and the loss strip (frames 4–5).

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4-Pro: 33T pretraining tokens | `models.deepseek-v4-pro.pretrain_tokens` (existing, confirmed) | 02 §0, §1.8 |
| DeepSeek-V4-Flash 32T · MiniMax-M2 29.2T · GLM-5 28.5T (27T base + mid-training) · MiMo-V2-Flash 27T · Nemotron 3 Super 25T | `models.deepseek-v4-flash.pretrain_tokens` = 32e12, `models.minimax-m2.pretrain_tokens` = 29.2e12, `models.glm-5.pretrain_tokens` = 28.5e12, `models.mimo-v2-flash.pretrain_tokens` = 27e12, `models.nemotron-3-super.pretrain_tokens` = 25e12 (proposed, confirmed) | 02 §0, §1.8 |
| Kimi K3 did not disclose its token count | `models.kimi-k3.pretrain_tokens` = null with note "not disclosed" (proposed, confirmed absent) | 02 §0, §1.8, §8 |
| Llama 3.1 405B (2024): 15.6T tokens | `models.llama-3.1-405b.pretrain_tokens` (existing, confirmed); the year from `models.llama-3.1-405b.release_date` = "2024-07-23" (confirmed, Meta's release post) | data entry only (source arXiv 2407.21783; not in brief 02) |
| Vocabularies: DeepSeek-V4 129,280 · GLM-5 151,552–154,880 (two columns in its report; shown as a range) · Kimi K3 163,840 · MiniMax-M2 200,064 · gpt-oss ~201K (`o200k_harmony`) | `models.deepseek-v4-pro.vocab_size` = 129280, `models.glm-5.vocab_size` = [151552, 154880], `models.kimi-k3.vocab_size` = 163840, `models.minimax-m2.vocab_size` = 200064, `models.gpt-oss-120b.vocab_size` = 201088 (config-verified) | 02 §1.1 table |
| Kimi K3 embedding table ≈ 163,840 × 7,168 ≈ 1.17B parameters (the briefs said "160K" and 1.15B; the config says 163,840) | derived from `kimi-k3.vocab_size` × `kimi-k3.d_model` | 02 §1.1; 01 §5 |
| Multi-token prediction as an auxiliary loss: DeepSeek-V4 (depth 1, weight 0.3, then 0.1 when LR decay starts), GLM-5, MiniMax-M2, Nemotron 3 Super, Kimi K3 | `models.deepseek-v4-pro.mtp_loss` = "depth 1, weight 0.3 → 0.1" (confirmed); the `mtp` flag is true on `glm-5`, `minimax-m2`, `nemotron-3-super` and `kimi-k3` (confirmed) | 02 §1.1 |
| Kimi K3: four text domains (Web, Code, Math, Knowledge), each filtered by heuristics, classifier quality scoring and dedup; per-domain rates from small-model ablations; knowledge and math rephrased with fidelity checks | `models.kimi-k3.data_pipeline` (confirmed) | 02 §1.2 |
| MiniMax-M2: model-based quality scoring; code, mathematics and STEM significantly upsampled | `models.minimax-m2.data_pipeline` (confirmed) | 02 §1.2 |
| DeepSeek-V4: filters batched auto-generated and templated content to avoid model collapse; long-document emphasis; packing with sample-level attention masking | `models.deepseek-v4-pro.data_pipeline` (confirmed) | 02 §1.2 |
| GLM-5: ~10M issue–PR pairs (~160B unique tokens) for software engineering | `models.glm-5.swe_data_tokens` = 160e9, `models.glm-5.swe_issue_pr_pairs` = 10e6 (confirmed) | 02 §1.2 |
| Qwen3: 36T tokens over 119 languages; Mistral Large 4: 160+ languages | `models.qwen3.pretrain_tokens` (reported); `models.qwen3.languages` = 119 (confirmed, Qwen3 release post); `models.mistral-large-4.languages` = "160+" (confirmed) | 02 §1.2 |

Not on this page, by design: which Nemotron 3 models pretrained in NVFP4 (both Super and Ultra did; no conflict) and all
precision detail (`gpu-primer`, `training-memory`).

## 9. Takeaways
1. Pretraining grades one thing: minus the log of the probability given to the true next token,
   averaged over every position. Confident mistakes cost far more than certainty earns, and perplexity
   (its exponential) is the effective number of choices.
2. Every position is graded in the same pass, and the text is curated hard: filters, a quality
   classifier, deduplication, an upsampled code-and-math mixture, and rephrased knowledge.
3. The open frontier MoEs reported in 2026 read 25–33 trillion tokens; it is the largest stage, and the next page asks
   how to split that compute between model size and tokens.

## 10. Next and go deeper
Next: `scaling-laws` (how big a model for how many tokens), `midtraining` (the end of pretraining: the best
data and longer context), `sft` (from continuing text to answering).
Go deeper: Hugging Face, *The Smol Training Playbook* (https://huggingfacetb-smol-training-playbook.hf.space/)
· Andrej Karpathy, *nanochat*, a runnable tokenizer → pretrain → midtrain → SFT pipeline
(https://github.com/karpathy/nanochat) · *Tiktokenizer*, type text and see the tokens
(https://tiktokenizer.vercel.app/)

## 11. Key-frame sketch
Frame 5 (mean loss and perplexity), desktop width. Checked against the reproducer below.

```text
  [The]₁ [cat]₂ [sat]₃ [down]₄ [on]₅ [the]₆ [mat]₇ [.]₈
  predicts:  cat     sat     down    on     the    mat     .
  p        0.10    0.25    0.30   0.390   0.60   0.45   0.80
  loss    [2.303] [1.386] [1.204] [0.941] [0.511] [0.799] [0.223]
                                  ▲ followed
  mean loss 1.052        perplexity 2.86
  uniform over 16 words: loss 2.773, perplexity 16
  Training lowers the mean loss over all positions. Its
  exponential, perplexity, is how many equally likely tokens
  the model is effectively choosing between: about 2.9 here.
  [Back] [Pause] [Next]  ━━━━●━━━━━  5 / 10   1×
```

**Reproducer** (every number in §2, §5, §6 and this sketch; run 2026-10-07 against a scratchpad draft of
`math/lm.js` matching the signatures below, plus the real `math/core.js`):
```sh
node -e '
Promise.all(["./math/lm.js","./math/core.js"].map((p)=>import(p))).then(([lm,core])=>{
  const f=(x)=>Number(x.toFixed(4));
  const p=core.softmax([2.0,1.5,0.5,0.0,...Array(12).fill(-1.0)]); console.log("probs",p.slice(0,5).map(f));
  const P=[0.10,0.25,0.30,p[0],0.60,0.45,0.80];
  console.log("losses",P.map((x)=>f(lm.tokenLoss(x))),"mean",f(lm.meanLoss(P)),"ppl",f(lm.perplexity(lm.meanLoss(P))));
  for (const v of [0.01,1.0,0.0625]) { const Q=P.map((x,j)=>j===5?v:x); console.log("mat",v,f(lm.tokenLoss(v)),f(lm.meanLoss(Q)),f(lm.perplexity(lm.meanLoss(Q)))); }
  const U=P.map(()=>1/16); console.log("uniform",f(lm.meanLoss(U)),f(lm.perplexity(lm.meanLoss(U))),f(lm.uniformLoss(163840)));
  console.log("emb",163840*7168, "tokenLoss(0.99)",f(lm.tokenLoss(0.99)));
});'
```
Output (2026-10-07): probs on 0.3903 · "." 0.2367 · and 0.0871 · the 0.0528 · each other 0.0194 · losses
2.3026, 1.3863, 1.2040, 0.9410, 0.5108, 0.7985, 0.2231 · mean 1.0523 · perplexity 2.8643 · `mat` at 0.01 →
loss 4.6052, mean 1.5961, perplexity 4.9339; at 1.00 → 0, 0.9383, 2.5555; at 0.0625 → 2.7726, 1.3343,
3.7975 · uniform → 2.7726, 16 · ln 163840 = 12.0066 · embedding 1,174,405,120 (was ln 160000 = 11.9829, 1,146,880,000) · −ln 0.99 = 0.0101.

**`math/lm.js` signatures** (owner: `pretraining`; pure, no DOM; never mutates inputs; tests first). This is
the single spec of the module; the worked examples quoted by `rlhf-dpo`, `sft` and `distillation` are listed
here.
```js
// Loss for one position: minus the natural log of the probability given to the true token.
tokenLoss(p: number) → number
//   0.390253 (exact softmax p from decoder-anatomy's logits) → 0.9410 · 0.3903 → 0.9408 · 0.390 → 0.9416
//   0.01 → 4.6052 · 1 → 0 · 0.0625 → 2.7726
//   throws RangeError unless 0 < p ≤ 1

// Mean loss over positions; an optional boolean mask keeps only positions where mask[i] is true.
meanLoss(probs: number[], mask: boolean[] | null = null) → number
//   [0.10, 0.25, 0.30, 0.3903, 0.60, 0.45, 0.80] → 1.0523
//   [0.5, 0.25, 0.8], [false, true, true] → 0.8047     // the masked form; sft builds its mask with lossMask
//   throws when mask length differs or no position is kept

perplexity(loss: number) → number            // e^loss
//   1.0523 → 2.8642 · 2.7726 → 16.0002 · ln 16 → 16 (exactly) · the unrounded mean loss → 2.8643

uniformLoss(vocabSize: number) → number      // ln |V|, the loss of a uniform guess
//   16 → 2.7726 · 163840 → 12.0066 · 201088 → 12.2115 · throws unless an integer ≥ 1

// KL divergence D(p ‖ q) = Σ p ln(p/q); terms with p = 0 contribute 0.
klDivergence(p: number[], q: number[]) → number
//   ([0.70,0.15,0.10,0.05], [0.40,0.30,0.20,0.10]) → 0.1838      // rlhf-dpo frame 8, honest policy
//   ([0.01,0.01,0.01,0.97], [0.40,0.30,0.20,0.10]) → 2.1031      // rlhf-dpo frame 7, hacked policy
//   ([0.90,0.05,0.03,0.02], [0.40,0.30,0.20,0.10]) → 0.5511      // distillation, forward KL(teacher ‖ student)
//   ([0.40,0.30,0.20,0.10], [0.90,0.05,0.03,0.02]) → 0.7535      // distillation, reverse KL(student ‖ teacher)
//   (p, p) → 0 · throws on length mismatch or q_i = 0 where p_i > 0
```
Test cases for the builder: `tokenLoss` is strictly decreasing on (0, 1]; `meanLoss` of a constant array
equals `tokenLoss` of the constant; `perplexity(uniformLoss(V)) = V`; `klDivergence ≥ 0` with equality only
when p = q; KL is not symmetric (the two distillation examples differ); no input array is mutated.

## 12. Open questions for the reviewer
**Data-pass keys** (all now in `data/models.json`; verified by `scripts/check-data-refs.js`): the
pretraining token counts for `deepseek-v4-flash`, `minimax-m2`, `glm-5`, `mimo-v2-flash`, `nemotron-3-super` and
`qwen3` (reported); `kimi-k3.pretrain_tokens` = "not disclosed"; the vocabulary sizes for `deepseek-v4-pro` (129,280),
`glm-5` (a range), `kimi-k3` (163,840) and `minimax-m2` (200,064); the `data_pipeline` strings for `kimi-k3`,
`minimax-m2` and `deepseek-v4-pro`; the `mtp` flags on `glm-5`, `minimax-m2`, `nemotron-3-super` and `kimi-k3`;
`glm-5.swe_data_tokens` = 160e9; `mistral-large-4.languages` = "160+". The newer checkpoints DeepSeek-V4.1-Flash
and GLM-5.3 are separate data entries; this page does not use them. All from 02 §1.1, §1.2, §1.8.
- Kimi K3's config gives a vocabulary of 163,840 (the briefs said "160K"), so the page prints 163,840 and
  "≈ 1.17B" (163,840 × 7,168 = 1,174,405,120).
- The Llama 3.1 405B entry id is `llama-3.1-405b` (renamed in the data pass; this page cites the new id).

**Graph changes:** none.

**Judgment calls:**
- **T − 1 vs T.** The brief's formula averages over T; the page averages over the T − 1 positions that
  have a target and says so in math note (a).
- **No tokenizer toy.** Spec §12 limits tokenization to one section; frame 1 and its numbers are that
  section. A live BPE toy is linked out (Tiktokenizer).
- **No keep ratios in frames 6–9.** The briefs say the reports give recipes, not ratios, so these frames
  show stages and confirmed facts only, with no stand-in percentages.

## 13. Reviewer rulings (expert review, 2026-10-07)
Applied from `track-review-recipe.md` §2 (change log: `fix-recipe-review.md`):
- Must 1: try-this 1 now says "almost five times as much (+0.544) as one perfect one does (−0.114)"; ratio 4.77 from the reproducer's means (1.0523 → 1.5961 and → 0.9383).
- Must 2: Kimi K3's vocabulary printed as 163,840 and the table as "≈ 1.17B" (frame 1 caption and numbers, misconception 4, toy reference, §8); corrected in the 2026-10-07 data pass from "about 160,000" and "≈ 1.15B".
- Should: §3 ¶3 scopes 25–33T to the open frontier MoEs reported in 2026, notes Kimi K3 did not say and smaller open models read 5–12T (takeaway 3 matches); frame 4 carries a 580 × 366 layout check with a 16 px heatmap; try-this 2 prints the runner-up (0.237).
- Nice: frame 1 carries the Tiktokenizer line.
- Data pass 2026-10-07: Kimi K3 vocabulary 163,840 (was about 160,000), embedding table ≈ 1.17B (was 1.15B), uniform loss 12.007 (was 11.983); DeepSeek-V4 vocabulary 129,280; Llama 3.1 405B id renamed; Nemotron NVFP4 has no conflict.

## 14. Plan 3 rulings applied (S3, 2026-10-08)
- P3-R18: frame 1 prints the data's vocabulary range "129K–201K" (was "128K–200K"), through `{…|count}`.
- P3-R17: the `pos` selection is the course's stage-selection pattern (focusable buttons, `G.selectionMark`).
- P3-R14 (data gaps 9, 10, 12): `qwen3.languages` = 119 and `glm-5.swe_issue_pr_pairs` = 10e6 added; §3's
  "Olmo 3 read 5–12T" (no source for 12T) becomes "Olmo 3 about 5.9T, reported" (`olmo-3.pretrain_tokens`).
- X-1: "Llama 3.1 405B (2024)" stays: `llama-3.1-405b.release_date` added (confirmed).
- X-2 / P3-R16: "Check my work" added (§6), from `tokenLoss`, `meanLoss`, `perplexity`.
- S3-A findings: frame 3 prints `−ln 0.3903 = 0.941` (−ln 0.390 is 0.942); the `tokenLoss` and
  `perplexity` examples in §11 now hold for their printed inputs (0.390 → 0.9416; 1.0523 → 2.8642; 2.7726 →
  16.0002).
