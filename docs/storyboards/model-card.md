# Reading a model card (`model-card`)

Track: architecture · Section: architecture · Prereqs: long-context-attention, moe, multimodal
Next: none (the track's capstone; no slug lists `model-card` as a prerequisite in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1, §2, §3, §4, §5, §6, §7 (spec sheet and verdicts), "Open uncertainties" · 02 §1 (items 2 and 9, for the resolved Nemotron question only) · 03 §1 (NVFP4 row, same) · 04 §8 (V4-Pro config and KV estimate), §10 item 3 · 05 §1.2. Spec §1 (the goal: read "granular MoE, 1T total / 49B active, MLA, 1M context") and §7 (the conflicts kept open). Nothing beyond the briefs.

The capstone. It reads real 2026 spec sheets field by field, straight from `data/models.json`, and sends
each field back to the lesson that explains it. It computes nothing new about architecture: every derived
number comes from a function an earlier page specified (`math/params.js`, `math/memory.js`), and every
fact keeps its "reported" chip and its range where sources disagree. Three of the spec's four §7 conflicts stay
open on this page: GLM-5.3's 78 vs 80 layers, Mistral Large 4's 1M vs about 512K context and DeepSeek-V4-Pro's
KV estimate (shown as ranges from the data file). The fourth, which Nemotron 3 model pretrained in NVFP4, is
resolved: both Super and Ultra did (§12).

## 1. Learning objective
After this page you can take a 2026 spec sheet such as "1.6T total / 49B active, 61 layers, 384 + 1
experts top-6, 1 KV head × 512 with compressed attention, 1M context, text-only" and say what each field
means, which part of the decoder it describes and what it costs (frames 1–9); spot fields that sources
disagree on and read them as ranges (frame 10, try-this 3); and compare two models' cards on active share,
cache per token and context (toy).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "The bigger total number is the faster or more expensive model to run." → **Reality:**
  per-token work follows *active* parameters; total sets memory. Kimi K3 has 2.78T total and 104.2B active,
  DeepSeek-V4-Pro 1.6T and 49B. (01 §5; `decoder-anatomy`, `moe`) · corrected by frame 2, try-this 1
- **Misconception:** "'1M context' means the same thing on every card." → **Reality:** it says how many
  positions the model accepts, not how well it uses them or what they cost. Mistral claims 1M for Large 4
  while evaluators reportedly measure about 512K; and the cache for 1M tokens is about 4–12 GB in
  DeepSeek-V4-Pro (an estimate) but about 123 GB in MiniMax-M3 (derived from its config), both at
  1,000,000 tokens. (01 §3, §7; 01 §4)
  · corrected by frames 6–7, try-this 2
- **Misconception:** "A number on a card is a fact." → **Reality:** cards and papers disagree, and labs
  count differently: GLM-5.3's config says 78 layers and the GLM-5 paper 80; Mistral quotes 49B active
  ("routed") and 52B (with embeddings). The page shows both, with sources. (01 §7, "Open uncertainties") ·
  corrected by frame 10, try-this 3
- **Misconception:** "A 2026 frontier model reads images." → **Reality:** the modalities field says; several
  leaders are text-only (DeepSeek-V4-Pro, GLM-5.3, Qwen3.8, gpt-oss), and some secondary sites claim V4 is
  multimodal against its own paper. (01 §6 [C]) · corrected by frame 8

## 3. Hook and intuition (final wording)
**Hook:** "1.6T total / 49B active, 61 layers, 384 + 1 experts, top-6, compressed attention with one 512-wide
KV head, 1M context, text-only." You have now seen every part this line describes. Can you read it, and can
you tell which parts to trust?

A model card is the decoder you have been taking apart, written as numbers. Total and active parameters
are all the experts versus the few each token uses. Layers count blocks. The experts line says how the MLP
was split and how many pieces each token runs. The attention line says how many key/value sets are stored,
whether they are compressed or windowed, and so how big the cache per token is; together with the context
length, that decides how much memory a long conversation needs. The modalities field says whether there is
a vision encoder in front at all.

Two habits make a card readable. First, turn each field into its cost: active parameters into compute per
token, the attention line into bytes per token, the context into gigabytes per conversation. Second, read
the source: a field from a config file is checkable, a field from a blog is not, and the same lab can count
"active" two ways. When sources disagree, keep both numbers and say so. This page does exactly that.

## 4. Visual metaphor
The stage holds two things side by side: on the left the card, as a column of `token`-style field chips
("1.6T total", "49B active", "61 layers", …) read from `data/models.json`; on the right the decoder diagram
from `decoder-anatomy` (`blockStack` with attention and MLP halves, the expert row, a `kvStack`, the input lane
with an optional `patch`). Each frame lights one field chip (selection outline) and the part of the diagram it
describes (Architecture accent `active` state), with a plain text line "→ lesson" naming where it was taught.
Under the stage, as page text, each field's one-line gloss stays visible (README lesson 5: nothing behind
hover).

**The followed card:** DeepSeek-V4-Pro, because every field it needs is confirmed in the data file except
the cache estimate, which is the conflict this page wants to show. Frame 10 brings in GLM-5.3 and Mistral
Large 4 for their conflicts.

**Terms introduced, one per frame** (README lesson 3): 1 model card / config · 2 none (total vs active) ·
3 none (layers) · 4 none (experts) · 5 none (attention line) · 6 none (cache per token) · 7 none (context) ·
8 modalities · 9 none (beyond the block) · 10 conflicting sources / range. This page introduces almost no new
terms; that is the point of a capstone. **Terms assumed from prereqs:** everything from `decoder-anatomy`,
`moe`, `kv-cache`, `kv-compression`, `rope`, `long-context-attention`, `multimodal`; MTP from `sampling`.
**Named and deferred:** optimizer (`scaling-laws`), pretraining tokens (`pretraining`), precision
(`quantization`), serving cost (`serving-calculator`).

**Layout** (stage ≈ 580 × 366): card column 200 px wide (10 chips of 24 px), diagram 340 px. At 400 px the
card stacks above the diagram.

Glyphs used (from spec §5.1 and the accepted `decoder-anatomy` proposals): token (field chips), blockStack,
block, kvStack, patch, flow (from chip to part, carry `token`), `shareBar` (frame 2: active vs total on a ×10 bar over the first 10 % of the total, README lesson 19).
New glyphs proposed: none. Small marks (README lesson 15): "→ lesson" lines, the "reported" chip (the
library's `renderFact` chip), range labels ("78–80") and every gloss are plain text.

Color: the lit part uses the `active` state; field chips use the default token style; "reported" facts carry
`renderFact`'s chip; ranges print both ends and never use a hatch (hatch means "excluded"). The followed field
chip carries the selection outline.

## 5. Animation script
Every number is the data file's value (§8) or computed from it by an earlier page's function.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | The DeepSeek-V4-Pro card as 10 field chips on the left; the decoder diagram on the right, all parts `idle`. | The chips type in top to bottom. | A model card lists a model's shape in numbers. Each field below points at one part of the decoder you have already taken apart. | DeepSeek-V4-Pro (preview Apr 24, 2026) · 1.6T total · 49B active · 61 layers · 384 + 1 experts, top-6 · 1 KV head × 512, compressed + window 128 · 1M context · text · Muon · 33T tokens · MIT |
| 2 | Chips "1.6T total" and "49B active" lit; a ×10 `shareBar` over the first 10 % of the total, active 3.06 % (104 px) vs not used, with the line 'the other 90 % is not used by this token either'; the expert row lights. "→ `decoder-anatomy`, `moe`". | The bar fills; the expert row flickers to show 6 lit. | Total counts every expert; active counts what one token runs. This model stores 1.6 trillion parameters and uses 49 billion per token, about 3%. | 49B ÷ 1.6T = 3.06% · per-token work follows the 49B |
| 3 | Chip "61 layers" lit; the `blockStack` reads "× 61". "→ `decoder-anatomy`". | The stack's count label types 61. | "Layers" on a card counts blocks: each one is attention plus an MLP or experts, added onto the stream. Here there are 61. | 61 blocks |
| 4 | Chip "384 + 1 experts, top-6" lit; the expert row shows 384 small boxes collapsed to "384", one "shared" box, 6 lit. "→ `moe`". | Six routed boxes and the shared box light. | Each token's router picks 6 of 384 small experts, plus one shared expert that every token uses. | 6 ÷ 384 = 1.56% of routed experts per token · expert hidden 3,072 · 384 × 3,072-wide experts per MoE layer |
| 5 | Chip "1 KV head × 512, compressed + window 128" lit; the attention half opens: one wide `kvStack`, a "merge 4 / merge 128" label, a 128-token window. "→ `kv-compression`, `long-context-attention`". | The kvStack tiles fuse in fours (CSA) and in 128s (HCA) on alternating layers; a window band appears. | The attention line says how much each token stores: one shared 512-wide key/value set, merged across tokens, plus a short window. | 1 KV head, 512 wide · CSA: merge 4, read top 1,024 · HCA: merge 128 · window 128 on every compressed layer |
| 6 | A readout "cache per token: about 4–12 kB (estimate)" with the "reported" chip; under it "at 1,000,000 tokens: about 4–12 GB"; beside it, for scale, "Llama-3.1-70B style GQA: 328 GB". "→ `kv-cache`". | A `gpu` glyph (80 GB) fills to the low end (5%), with "to 15% at the high end" printed beside it. | Turn the attention line into memory. Its paper gives only a ratio, so this is an estimate: about 4 to 12 kB per token. | 4,000–12,000 B per token (reported, formula-derived; layer mix uncertain) · × 1,000,000 = 4.0–12.0 GB · Llama-3.1-70B: 327,680 B × 1,000,000 = 328 GB · GPU share `sharePct(bytes, 80e9)` from `math/memory.js`: 5.0% and 15.0% · the paper's ratio: about 10% of V3.2's KV at 1M |
| 7 | Chip "1M context" lit (the card's own fields come first in the column; the chips added in frames 4, 7 and 9 are labeled "from the paper / config"); the input lane stretches to "1,000,000 positions"; small labels "RoPE base 10,000 (160,000 for compressed streams), YaRN factor 16" and "trained in stages 4K → 16K → 64K → 1M". "→ `rope`, `long-context-attention`, `midtraining`". | The lane stretches; the labels appear in order. | "1M context" is how many positions the model accepts. Reaching it took a stretched position encoding, staged training and cheap attention. | 1,000,000 positions (the card's "1M") · base 10,000 · YaRN ×16 · 4K → 16K → 64K → 1M |
| 8 | Chip "text" lit; the input lane's `patch` slot stays empty and dims, labeled "no vision encoder". Small text: "some secondary sites call V4 multimodal; its paper lists multimodality as future work". "→ `multimodal`". | The patch slot fades. | The modalities field says what can enter the stream. DeepSeek-V4-Pro takes text only; its paper names images as future work. | text-only (confirmed) · Kimi K3, MiniMax-M3: native multimodal |
| 9 | Chips "Muon", "33T tokens", and two from the brief's config ("MTP depth 1", "experts FP4, the rest FP8") lit; they point off the diagram to labeled exits. "→ `scaling-laws`, `pretraining`, `sampling`, `quantization`". | Four `flow`s run from the chips to the stage edge. | Some fields describe training or storage, not the block. They lead to the training and serving lessons. | Muon optimizer · 33T pretraining tokens · MTP depth 1 · FP4 experts, FP8 elsewhere |
| 10 | **Key frame.** Three conflict chips from other cards replace the column: "GLM-5.3 layers 78–80", "Mistral Large 4 context 512K–1M", "DeepSeek-V4-Pro cache 4–12 kB per token"; each shows both sources. A fourth line: "Mistral counts active two ways: 49B routed, 52B with embeddings". | Each chip splits into its two numbers with their source labels. | Sources disagree, and labs count differently. When they do, this course shows both numbers and where each came from. | GLM-5.3: config 78, GLM-5 paper 80 · Mistral Large 4: card 1M, evaluators about 512K (both reported, via one secondary source) · V4-Pro: 1 : 1 or 3 : 1 layer mix → 4–12 kB · Mistral 49B (4.67%) vs 52B (4.95%) |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state.
The card column and the diagram keep their positions in frames 1–9.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 24 · 24 · 23 · 17 · 21 · 24 ·
21 · 20 · 17 · 19.

Absolutes checked (README lesson 7): frame 4's "every token uses" the shared expert is the definition of a
shared expert (01 §5). Frame 6's "no per-token number" in its paper is 04 §8.2/§10 (the brief derives it).
Frame 7's three ingredients are 01 §3's mainstream recipe for V4 ([C]).

## 6. Toy
Title on page: "Decode a card." Two columns so two cards can be compared. One state object, one `render()`.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `left` | Card | preset chips | DeepSeek-V4-Pro · Kimi K3 · Qwen3.8 · GLM-5.3 · MiniMax-M3 · Mistral Large 4 · gpt-oss-120b | DeepSeek-V4-Pro | from `data/models.json` |
| `right` | Compare with | preset chips | same list + "none" | Kimi K3 | — |
| `context` | Conversation length for the cache line | segmented | the card's own context · 131,072 | own | — |

**Live outputs** (one row per data key; every row shows value, confidence chip, source link, a visible
one-line gloss and the lesson link)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Field rows | `decodeCard(entry)` from `math/card.js` with `FIELD_GUIDE` | text; ranges as "78–80" |
| Active share (one definition, README lesson 16: published active ÷ published total; ranges or missing → "—") | `activeShare(entry)` | % (2 d.p.), drawn as a ×10 `shareBar` over the first 10 % of the total (README lesson 19) |
| GPU share of one conversation's cache | `sharePct(bytes, 80e9)` from `math/memory.js`, both ends of a range | % (1 d.p.) |
| Routed experts used per token | `experts_active / experts_total` | %; "—" when not published |
| Cache per token | `cachePerToken(entry)`: `'derived'` when the config keys give it (plain GQA, or a full : window mix) via `math/memory.js` (`kvBytesPerToken`, `kvBytesPerTokenMla`, `stackKvBytes`), else `'reported'` from `kv_bytes_per_token`, else `'not in our data'` | exact bytes and `formatBytes` (decimal); a range for reported ranges |
| Cache for one conversation | `kvCacheBytes({ bytesPerToken, tokens })` (or `stackKvBytes(...).total`) with tokens = the context control | decimal GB; range when either input is a range |
| "Labs differ" notes | the entry's `note` strings (e.g. Mistral's 52B) | text |

**Try this** (each leads to a named insight)
1. **DeepSeek-V4-Pro** vs **Kimi K3**: 1.6T / 49B (3.06%) against 2.78T / 104.2B (3.75%); routed experts per
   token 1.56% vs 1.79% (lower than active share, because shared experts, attention and the head run for every
   token too). → **Insight: compare active parameters for per-token cost and total for memory; the
   bigger total is not the bigger bill per token.**
2. **DeepSeek-V4-Pro** vs **MiniMax-M3**, own context (1,000,000 for V4-Pro, 1,048,576 for MiniMax-M3): about 4–12 GB
   (reported estimate) vs 129 GB (123 GB at exactly one million tokens; derived: 60 layers × 4 KV heads × 128 × 2 × 2 B = 122,880 B per token).
   Switch to **gpt-oss-120b**: 36,864 B per token + 4.7 MB fixed, 4.84 GB at its 131,072. → **Insight: the
   attention line, not the context number, decides what a long conversation costs.**
3. **GLM-5.3** vs **Mistral Large 4**: layers "78–80" (config vs paper), context "512K–1M" (claimed vs
   measured, reported), Mistral active "49B (52B with embeddings)", GLM active "reported". → **Insight: a card
   is a set of claims with sources; when they disagree, keep the range and the sources, and prefer the config
   file.**

**`math/card.js`** (pure, no DOM, inputs never mutated; tests first). It maps data keys to meaning and
reuses earlier modules for every number; it adds no architecture formula.
```js
// One entry per data key the page shows: label, a one-line gloss (always visible), the lesson(s) that explain it
// (`lesson` is a slug, a list of slugs, or null).
export const FIELD_GUIDE = {
  total_params:       { label: 'Total parameters', gloss: 'Every weight stored, all experts included.', lesson: 'moe' },
  active_params:      { label: 'Active parameters', gloss: 'Weights one token multiplies; sets compute per token.', lesson: 'decoder-anatomy' },
  layers:             { label: 'Layers', gloss: 'Blocks: attention plus MLP or experts, repeated.', lesson: 'decoder-anatomy' },
  experts_total:      { label: 'Routed experts', gloss: 'Small MLPs the router chooses from.', lesson: 'moe' },
  experts_active:     { label: 'Experts per token', gloss: 'How many routed experts each token runs.', lesson: 'moe' },
  attention:          { label: 'Attention', gloss: 'How keys and values are stored and read.', lesson: 'kv-compression' },
  kv_bytes_per_token: { label: 'Cache per token', gloss: 'Bytes of keys and values kept for each token.', lesson: 'kv-cache' },
  context_length:     { label: 'Context', gloss: 'Positions the model accepts, not how well it uses them.', lesson: ['long-context-attention', 'rope'] },
  modalities:         { label: 'Inputs', gloss: 'What can enter the stream.', lesson: 'multimodal' },
  optimizer:          { label: 'Optimizer', gloss: 'How the weights were trained.', lesson: 'scaling-laws' },
  pretrain_tokens:    { label: 'Pretraining tokens', gloss: 'Text seen in pretraining.', lesson: 'pretraining' },
  license:            { label: 'License', gloss: 'What you may do with the weights.', lesson: null },
  release_date:       { label: 'Released', gloss: 'When the weights or preview appeared.', lesson: null },
};

// Rows in FIELD_GUIDE order for the keys the entry has; ranges keep both ends.
decodeCard(entry) → [{ key, label, gloss, lesson, value, display, confidence, isRange, sourceUrl, note }]
//   (models['deepseek-v4-pro']) → 13 rows; layers display "61"; total "1.6T"; kv_bytes_per_token display
//     "4–12 kB" isRange true confidence 'reported'
//   (models['glm-5.3']).find(r => r.key === 'layers') → { display: '78–80', isRange: true, confidence: 'confirmed', note: '…config.json says 78; the GLM-5 paper says 80.' }
//   (models['mistral-large-4']).find(r => r.key === 'context_length') → { display: '512K–1M', isRange: true, confidence: 'reported' }

// Published active ÷ published total; null if either is missing or a range.
activeShare(entry) → number | null
//   deepseek-v4-pro → 0.0306 · kimi-k3 → 0.0375 · qwen3.8 → 0.0396 · glm-5.3 → 0.0531 · minimax-m3 → 0.0537
//   mistral-large-4 → 0.0467 · gpt-oss-120b → 0.0439

// Cache per token with its provenance.
cachePerToken(entry) → { kind: 'derived' | 'reported' | 'not in our data', bytes?: number | [number, number], fixed?: number }
//   minimax-m3 (config keys n_kv_heads 4, head_dim 128, layers 60) → { kind: 'derived', bytes: 122_880 }
//   gpt-oss-120b (stack: 18 full + 18 window-128 layers, 8 KV × 64) → { kind: 'derived', bytes: 36_864, fixed: 4_718_592 }
//   deepseek-v4-pro → { kind: 'reported', bytes: [4000, 12000] }
//   kimi-k3, glm-5.3, qwen3.8, mistral-large-4 → { kind: 'not in our data' }
```
Worked cache per conversation (from `kvCacheBytes` / `stackKvBytes`, `math/memory.js`): MiniMax-M3 122,880 ×
1,000,000 = 122,880,000,000 B (123 GB) · DeepSeek-V4-Pro [4,000, 12,000] × 1,000,000 = 4.0–12.0 GB ·
gpt-oss-120b 36,864 × 131,072 + 4,718,592 = 4,836,556,800 B (4.84 GB) · Llama-3.1-70B for scale 327,680 ×
1,000,000 = 327.68 GB.

Tests to write first: every FIELD_GUIDE lesson slug exists in `shared/concepts.json` (or is null);
`decodeCard` keeps both ends of every range value in `data/models.json` and never averages them (spec §7: the
conflicts stay open; README lesson 16); `activeShare` returns null for ranges; `cachePerToken` returns
`'reported'` only where the config keys give no derivation and `kv_bytes_per_token` is in the data; the `'derived'` values equal
`kv-cache`'s and `long-context-attention`'s worked examples; inputs not mutated (frozen data).

**Reproducer** (run from the repo root on 2026-10-07 against the shipped `math/core.js` and
`data/models.json` values; output matched every number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({formatBytes,formatCount})=>{
const M={"deepseek-v4-pro":[1.6e12,49e9,384,6],"kimi-k3":[2.78e12,104.2e9,896,16],"qwen3.8":[2.4e12,95e9,512,10],"glm-5.3":[753e9,40e9],"minimax-m3":[428e9,23e9],"mistral-large-4":[1.05e12,49e9],"gpt-oss-120b":[116.8e9,5.1e9,128,4]};
for(const [k,[t,a,e,ea]] of Object.entries(M))console.log(k,formatCount(t),formatCount(a),(100*a/t).toFixed(2)+"%",e?(100*ea/e).toFixed(2)+"%":"-");
console.log("mistral 52B",(100*52e9/1.05e12).toFixed(2)+"%");
const m3=2*60*4*128*2;console.log("m3",m3,m3*1e6,formatBytes(m3*1e6),"v4",formatBytes(4000*1e6),formatBytes(12000*1e6),"gptoss",36864*131072+4718592,formatBytes(36864*131072+4718592),"llama",327680*1e6/1e9);
})'
```
Output on 2026-10-07: `deepseek-v4-pro 1.6T 49B 3.06% 1.56%` · `kimi-k3 2.78T 104B 3.75% 1.79%` · `qwen3.8 2.4T 95B
3.96% 1.95%` · `glm-5.3 753B 40B 5.31% -` · `minimax-m3 428B 23B 5.37% -` · `mistral-large-4 1.05T 49B 4.67% -` ·
`gpt-oss-120b 117B 5.1B 4.37% 3.13%` · `mistral 52B 4.95%` · `m3 122880 122880000000 123 GB v4 4 GB 12 GB gptoss
4836556800 4.84 GB llama 327.68`. (`formatCount` prints Kimi K3's 104.2B as "104B"; the card row prints the data
value 104.2B.)

## 7. Show me the math
```tex
\text{active share} = \frac{\htmlClass{hl-act}{\text{active}}}{\text{total}} = \frac{49\text{B}}{1.6\text{T}} = 3.06\%,
\qquad \text{routed used} = \frac{k}{E} = \frac{6}{384} = 1.56\%
```
```tex
\text{cache per conversation} = \htmlClass{hl-kv}{\text{bytes per token}} \times \text{tokens}
\;(+\ \text{fixed window or state bytes})
\qquad \text{MiniMax-M3: } 2 \cdot 60 \cdot 4 \cdot 128 \cdot 2 \times 10^6 = 123\ \text{GB}
```
```tex
\text{a range stays a range: } [a, b] \times n = [a n,\ b n],
\qquad [4{,}000,\ 12{,}000]\ \text{B} \times 10^6 = [4.0,\ 12.0]\ \text{GB}
```
Every formula is an earlier page's (`decoder-anatomy`, `moe`, `kv-cache`, `long-context-attention`); this panel
links to each. Color links: `hl-act` → the active field chip and `shareBar`; `hl-kv` → the cache readout. KaTeX
with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
The whole page is this section: the cards render from `data/models.json` with `renderFact`. Rows the frames and
toy read:

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4-Pro: 1.6T / 49B, 61 layers, 384 routed (+1 shared) top-6, CSA (merge 4, top 1,024) + HCA (merge 128) + window 128 with 1 KV head of 512, 1M context, text-only, Muon, 33T tokens, MIT, preview 2026-04-24 | `models.deepseek-v4-pro.*` (existing: total_params, active_params, layers, experts_total, experts_active, attention, context_length, modalities, optimizer, pretrain_tokens, license, release_date; all confirmed) | 01 §5, §7 [C]; 02 §1 (tokens) |
| DeepSeek-V4-Pro cache: about 4–12 kB per token, formula-derived, layer mix uncertain | `models.deepseek-v4-pro.kv_bytes_per_token` = [4000, 12000] (existing, **reported**) | 04 §8.2, §10 item 3 (spec §7 conflict) |
| DeepSeek-V4-Pro extras for frames 4, 7, 9: expert hidden 3,072; RoPE base 10,000 (160,000 compressed), YaRN 16; staged 4K → 16K → 64K → 1M; MTP depth 1; FP4 experts, FP8 elsewhere | `.expert_hidden` (proposed, `decoder-anatomy`), `.rope_theta`, `.rope_theta_compressed`, `.yarn_factor` (proposed, `rope`), `.context_stages` = "4K→16K→64K→1M" (proposed *new*), `.mtp_depth` (proposed, `sampling`), `.pretrain_precision` (in data, `decoder-recap`) | 01 §3, §5, §1 [C]; 04 §8 |
| Kimi K3: 2.78T / 104.2B, 93 layers (69 linear + 24 MLA), 896 + 2 experts top-16, 1M, native multimodal | `models.kimi-k3.*` (existing, confirmed; release date reported) | 01 §5, §7 [C] |
| Qwen3.8: 2.4T / 95B, 92 layers, 512 + 1 experts top-10, 262K native context, text-only | `models.qwen3.8.*` (existing, confirmed) | 01 §5, §7 [C] |
| GLM-5.3: 753B / 40B (active carried over from GLM-5), **78–80 layers** (config 78, GLM-5 paper 80), MLA + DSA, 1M, text-only | `models.glm-5.3.total_params` (confirmed), `.active_params` (reported), `.layers` = [78, 80] (existing, with the two-source note) | 01 §5, §7, "Open uncertainties" (spec §7 conflict) |
| MiniMax-M3: about 428B / 23B, 60 layers, GQA 64 Q / 4 KV + sparse attention, 1M, native multimodal; 122,880 B per token derived | `models.minimax-m3.*` (existing, confirmed); `.n_kv_heads`, `.head_dim` (proposed, `kv-compression`) | 01 §2, §4, §7 [C] |
| Mistral Large 4 (preview Oct 6, 2026): 1.05T / 49B routed-active (52B with embeddings), **context 512K–1M** (claimed 1M, measured about 512K, both reported), multimodal input, weights not yet out | `models.mistral-large-4.*` (existing; `total_params` and `active_params` are `reported`: the sources differ; `context_length` = [512000, 1000000] reported) | 01 §7 verdicts (spec §7 conflict) |
| gpt-oss-120b: 116.8B / 5.1B, 36 layers, 128 experts top-4, alternating full / window-128 layers, 131,072 context | `models.gpt-oss-120b.*` (existing, confirmed); `.context_length` = 131072 (proposed, `rope`) | 01 §5, §7 [C] |

Not shown: Nemotron 3 (both Super and Ultra were pretrained in NVFP4; §12 item 1); MiMo-V2.6-Pro and DeepSeek-V4.1-Flash
(reported only; available as chips if the reviewer wants reported-only cards); Llama "5" (01 §7 [U]).

## 9. Takeaways
1. A model card is the decoder in numbers: total vs active is experts stored vs used, layers count blocks, the
   experts line is the router's menu, the attention line sets the cache per token, and modalities says what
   can enter (frames 1–8).
2. Turn fields into costs: active parameters into compute per token, the attention line times the context
   into memory per conversation (frame 6, try-this 1–2).
3. Read the source: config files beat blogs, labs count "active" differently, and where sources disagree
   keep both numbers (frame 10, try-this 3).

## 10. Next and go deeper
Next: none in the graph; this page closes the Architecture track. Where to go from here, as in-page links:
`training-pipeline` (how a model like this is trained), `serving-calculator` (what serving one costs),
`prefill-decode` (why active parameters and the cache set speed).

Go deeper (brief 05 §1.2; 01 §8): Sebastian Raschka, LLM Architecture Gallery
(https://sebastianraschka.com/llm-architecture-gallery/), 100+ cards side by side with a diff tool · DeepSeek-AI,
*DeepSeek-V4* (https://arxiv.org/pdf/2606.19348), the card this page reads · the DeepSeek-V4-Pro `config.json`
(https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json), to read the raw fields yourself.

## 11. Key-frame sketch
Frame 10 (conflicts as ranges), desktop width; values from `data/models.json` and the §6 reproducer.
```text
┌──────────────────────────────────────────────────────────┐
│ Sources disagree                                         │
│                                                          │
│ ►GLM-5.3 · layers        78 ── 80                        │
│            config.json 78 · GLM-5 paper 80               │
│  Mistral Large 4 · context  512K ── 1M  [both reported]  │
│            evaluators ~512K · Mistral card 1M            │
│  DeepSeek-V4-Pro · cache/token  4 ── 12 kB [reported]    │
│            1:1 layer mix (config) · 3:1 (blog summaries) │
│  Mistral Large 4 · active  49B routed · 52B with embed.  │
│            4.67% · 4.95% of 1.05T                        │
├──────────────────────────────────────────────────────────┤
│ Sources disagree, and labs count differently. When they  │
│ do, this course shows both numbers and where each came   │
│ from.                                                    │
│ [◄] [Pause] [►]  ━━━━━━━━━━●  10 / 10  speed [1×]        │
└──────────────────────────────────────────────────────────┘
```
"►" is the selection outline on the first conflict chip; the stepper walks the outline down the four. At
400 px the source lines wrap under each value.

## 12. Open questions for the reviewer
**Data-pass keys:**
1. **Nemotron 3 NVFP4 (spec §7 question, resolved in the data pass).** Both `nemotron-3-super` and
   `nemotron-3-ultra` have `pretrain_precision` = "NVFP4" (confirmed by arXiv 2604.12374 and 2606.15007).
   The briefs named one model each, so there is no conflict and no fifth conflict chip; the page still
   shows neither card.
2. *New:* `deepseek-v4-pro.context_stages`. Reused: every key proposed on the other eight Architecture pages
   (this page reads them; it adds none of its own beyond item 1 and this one).
3. Ruled: reported-only cards (MiMo-V2.6-Pro, DeepSeek-V4.1-Flash) stay in "not shown".
4. Data-pass suggestion (expert review): store each config's actual `context_length` (1,048,576 where the
   config says so) with a note "listed as 1M"; this page then prints the configured value.

**Graph changes:** none. (The node has no dependents; `serving-calculator` could list it as a soft prereq,
but that is the Serving owner's call.)

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (3 Must). Status is now "approved (expert review)". The §6 reproducer was
re-run and matches; the zoom-bar width (49 ÷ 160 × 340 px = 104 px) and the GPU shares (5.0%, 15.0%) were
computed with `node -e`.

Rulings applied (lesson 20): "1M" is settled: 2²⁰ = 1,048,576 is a labelled slider stop on `kv-cache` and
`long-context-attention`, and a card page prints the model's configured context from the data file (here
1,000,000, as stored); fields beyond the card (frames 4, 7, 9) are kept and labeled "from the paper /
config"; reported-only cards stay out; Nemotron's NVFP4 question is resolved (both models).

Must (3/3): decimal units ("kB" in frames 6 and 10, §6, §8 and the sketch); GPU share named as `sharePct`
from `math/memory.js` in frame 6 and the live outputs; frame 2's 3.1% segment gets a ×10 zoom bar (104 px),
and the toy's active-share bars get the same treatment (README lesson 19).
Should (7/7 applicable, no rebuttals): the 1M ruling recorded and the alternative deleted; "from the paper /
config" labels with the card's own fields first; "(at 1,000,000 tokens)" in misconception 2; frame 6 caption
"Its paper gives only a ratio"; Mistral's chip says "both reported"; `FIELD_GUIDE.lesson` may be a list
(context → `long-context-attention`, `rope`); reported-only cards ruled out.
Nice (1/2): try-this 1 says why routed share is below active share; the hover note on the sketch is left out.
- Data pass 2026-10-07: the Nemotron 3 NVFP4 question is resolved (Super and Ultra both NVFP4; no conflict chip); DeepSeek-V4-Pro \`expert_hidden\` and \`context_stages\` are in data; no numbers changed.
- Data pass 2026-10-07 (addendum): Mistral Large 4 total and active parameters are \`reported\` (sources differ: 52B vs 49B active); the page keeps both labeled.
