# MQA, GQA, MLA (`kv-compression`)

Track: architecture · Section: architecture · Prereqs: kv-cache
Next: `long-context-attention`, `paged-attention` (the two slugs whose `prereqs` list `kv-compression` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1 ("Attention" row), §2 (formula block, MHA/MQA/GQA, MLA), §3 (decoupled RoPE), §4, §7 · 04 §3.1 · 05 §1.1, §2. Nothing beyond the briefs.

The page that answers `kv-cache`'s try-this 3 ("store fewer key/value sets"). It reuses `attention`'s heads A
and B and their numbers for the one frame that computes a pattern, and `math/memory.js` (`kvBytesPerToken`,
`kvBytesPerTokenMla`, `kvCacheBytes`) for every byte count. `paged-attention` reads "what one slot weighs"
from here.

## 1. Learning objective
After this page you can draw how MHA, MQA and GQA wire query heads to stored key/value heads (frames 1–3),
show that heads sharing keys and values still attend differently (frame 4), explain what MLA caches
instead and what it pays for it (frames 5–7), and compute any model's bytes per token from its KV heads or
latent size (frames 8–9, toy).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "Heads that share keys and values attend to the same tokens." → **Reality:** each
  query head keeps its own W_Q, so its scores against the shared keys differ. In the toy, head B reading
  head A's keys puts 0.731 on "The" from "cat", where head A puts 0.679 on "cat". (first principles; 01 §2
  GQA paragraph: "shares K,V within groups of query heads") · corrected by frame 4, try-this 2
- **Misconception:** "Fewer KV heads means fewer heads." → **Reality:** the number of query heads, and
  so of attention patterns, is unchanged; only the stored key/value sets shrink. The cache shrinks in
  proportion: GPT-3's shape with 8 KV heads instead of 96 stores 393,216 B per token instead of 4,718,592,
  12× less. (04 §3.1 "the heads term is KV heads") · corrected by frames 2–3, try-this 1
- **Misconception:** "MLA compresses the cache by storing fewer tokens." → **Reality:** MLA stores one
  entry for *every* token, just a much shorter one: 576 numbers per layer in DeepSeek-V3 instead of 32,768
  for full multi-head attention at its head count (56.9× less). Merging tokens is a different technique
  (`long-context-attention`). (01 §2 MLA paragraph) · corrected by frames 5 and 8, try-this 3
- **Misconception:** "MLA is where attention ended up." → **Reality:** it is the incumbent, not the
  endpoint. Its decode compute is high (GLM-5 reworked it), and DeepSeek-V4 replaced it with one wide KV
  head plus compression and sparsity. (01 §2: "'MLA is final' is not true; it is the incumbent, not the
  endpoint") · corrected by frame 7's cost line, frame 9's page text and try-this 3

## 3. Hook and intuition (final wording)
**Hook:** In GPT-3, every one of the 96 heads in every layer stored its own keys and values. Do the heads
really need separate copies, and how did DeepSeek-V3 store 57 times less per layer than it would with a key and value per head, without
taking those separate keys and values away?

In `kv-cache` the size of the cache came down to one product: 2 × layers × KV heads × head size × bytes.
The layers and head size are the model's shape. The KV heads term is a choice. In the original design
(multi-head attention, MHA) every query head has its own key head and value head, so the cache stores a
full set per head. But a query head only needs *some* keys to score against; nothing says each query head
needs its own.

So let several query heads share one stored set. If all of them share one, that is multi-query attention
(MQA): the smallest cache, with some loss in quality. If they share in groups, it is grouped-query
attention (GQA), and 8 KV heads became the common middle setting: Llama-3.1-70B runs 64 query heads on 8 KV heads. The heads keep their own queries, so
they still ask different questions of the same keys and still produce different patterns.

Multi-head latent attention (MLA) goes another way. It stores one short "latent" vector per token and
learns how to rebuild every head's own key and value from it, so each head keeps its own keys and values
while the cache stays close to a two-head GQA. The rebuild costs compute instead of memory, and most of it can be folded into the query and output matrices. Position is the exception: RoPE rotates each key by its token's position, which cannot be folded in that way, so MLA stores one small extra key per token that carries position ([[rope]]). Every 2026 design on this page trades a little quality or a little compute for a much
smaller cache.

## 4. Visual metaphor
**Toy layer:** 8 query heads, each 4 numbers wide (d_head = 4, as in `attention`), one layer drawn. A
visible line says why the count changed: "This layer has 8 query heads so the sharing is visible;
`attention`'s toy had 2." MLA toy: latent 8 numbers plus a 2-number position key. The pattern frame
(frame 4) uses `attention`'s two heads and their exact Q, K, V (`TOY.heads.A`, `.B`), so its numbers
match that page. A visible line under the stage (README lesson 10): "Head counts and the latent size here
are hand-picked toy sizes; frame 4 reuses the attention page's numbers; every byte count is exact."

**Terms introduced, one per frame** (README lesson 3): 1 KV head (a stored key/value set) and MHA as its
name · 2 MQA · 3 group / GQA · 4 none (the pattern from `attention`) · 5 latent · 6 position key (named;
RoPE itself is `rope`) · 7 absorption ("folding the rebuild into the query") · 8 none (comparison) · 9
none (real models). **Terms assumed from prereqs:** query / key / value, head, d_head, attention pattern
(`attention`); KV cache, bytes per token, the formula (`kv-cache`). **Named and deferred:** RoPE and why it
cannot be applied to the latent (`rope`); merging tokens, windows and sparse reads (`long-context-attention`);
FP8 caches (`quantization`).

**Indexing:** heads are lettered (A, B) where they match `attention` and numbered 1–8 otherwise, 1-based.
No memory addresses on this page.

**Layout** (stage ≈ 580 × 366):
- Top row: 8 small `block`s "Q1 … Q8" (60 × 28 px each, 8 × 66 = 528 px).
- Bottom row: the stored KV heads as `kvStack`s (one per KV head, 4 tiles = the four tokens), centered
  under the query heads they serve; plain `flow` lines (carry `kv`) from each KV head up to the query blocks that read it.
  The number of `kvStack`s is the quantity (8, 2, 1); the lines are equal-weight wiring, never weighted.
- Right gutter / under the stage: a plain text readout "stored per token per layer: N numbers".
- Frame 4: two `heatmap`s at `NUMBER_CELL` (4 × 43 = 172 px each, side by side = 360 px) labeled "head A"
  and "head B, reading head A's keys".
- Frames 5–7: one `kvStack` whose tiles are a `vector` of 8 cells (the latent) plus 2 cells (position
  key); `block`s "rebuild K" and "rebuild V" between the latent and each query head; in frame 7 the
  rebuild blocks slide up into the query blocks.
- Frames 8–9: plain text ladders (page-text table under the stage for frame 9).

Glyphs used (from spec §5.1): block (query heads, rebuild matrices), kvStack (stored KV heads and
latents), flow (wiring, carry `kv`), heatmap (frame 4), vector (latent cells), token (the four chips in
frame 4).
New glyphs proposed: none. Small marks (README lesson 15): the "stored per token per layer" readout, the
scheme labels (MHA, MQA, GQA, MLA) and the cost lines are plain text labels.

Color: heatmap cells use the value scale (weights, maxAbs 1); latent cells use the value scale with
hand-picked stand-in values on the quarter grid (`c = [0.5, −0.25, 1, 0, −0.5, 0.25, 0, 0.75]`,
position key `[0.5, −0.5]`, hover-only detail, never read out). `kvStack` tiles use `--sem-memory`. The
followed query head (Q1, and head A in frame 4) carries the selection outline in every frame; row "sat"
keeps its outline in frame 4's heatmaps, as on `attention`.

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | 8 query blocks Q1–Q8 (Q1 outlined); under them 8 `kvStack`s, one wired to each query block. Readout "stored per token per layer: 64 numbers". Label "MHA". | The wires draw top to bottom; the readout counts to 64. | In GPT-3-style attention, each query head has its own stored key head and value head, a KV head. Every token stores all of them, in every layer. | 8 KV heads × 4 numbers × 2 (K and V) = 64 numbers per token per layer |
| 2 | Same query row; the 8 stacks merge into 1; all 8 wires now land on it. Label "MQA". A plain cost line: "some quality loss". | Seven stacks slide into the first and fade; the wires re-route; readout counts down 64 → 8. | Multi-query attention lets all the query heads share one key head and one value head. The cache is 8 times smaller, at some cost in quality. | 1 × 4 × 2 = 8 numbers · 64 ÷ 8 = 8× smaller |
| 3 | The single stack splits into 2; Q1–Q4 wire to the first, Q5–Q8 to the second; a bracket "group of 4" under each. Label "GQA". | The stack splits; wires for Q5–Q8 swing to the new stack; readout 8 → 16. | Grouped-query attention is the middle way: query heads share within groups. Two groups here; 8 KV heads became a common choice in real models. | 2 × 4 × 2 = 16 numbers · 64 ÷ 16 = 4× smaller · "Llama-3.1-70B and gpt-oss-120b: 8 KV heads" (page text, from data) |
| 4 | Branch label "same keys, different questions". Four token chips; two `heatmap`s: head A with its own keys, and head B reading head A's keys and values. Row "sat" outlined in both. | Head B's heatmap fills row by row; rows "cat" and "down" settle visibly differently from head A's. | Two heads that share keys and values still have their own queries. Their patterns differ: from "cat", head B looks mostly at "The" while head A looks at "cat". | head A rows: [1] · [0.321, 0.679] · [0.095, 0.703, 0.202] · [0.114, 0.656, 0.129, 0.101] · head B on A's keys: [1] · [0.731, 0.269] · [0.119, 0.685, 0.196] · [0.196, 0.153, 0.366, 0.285] |
| 5 | Back to 8 query blocks. Under them one `kvStack` per token whose tile is an 8-cell `vector` "latent"; between the stack and each query block a dim `block` "rebuild K, V". Label "MLA". | The 8 latent cells fill; dots run from the latent through each head's rebuild block to its query. | Multi-head latent attention stores one short latent vector per token. Learned matrices rebuild each head's own key and value from it. | latent = 8 numbers per token per layer · rebuilds 8 heads × (4 + 4) = 64 numbers on the fly |
| 6 | Same; a 2-cell `vector` "position key" appears beside the latent in the stack, labeled "carries position (`rope`)". | The two cells slide in next to the latent; the readout ticks 8 → 10. | A small extra key carries each token's position, because the latent itself cannot be rotated by position. It is stored beside the latent. | 8 + 2 = 10 numbers per token per layer · DeepSeek-V3: 512 + 64 = 576 |
| 7 | The "rebuild K" blocks slide up and merge into the query blocks, now labeled "query with K-rebuild folded in"; the "rebuild V" blocks merge into a `block` "W_O". A plain cost line: "more compute per decode step". | Rebuild blocks travel up into Q1–Q8 and down into W_O; the latent stack stays untouched. | At decode time the rebuild folds into the query and output matrices, so attention reads the latent directly. Memory is saved; compute per step grows. | 0 rebuilt K or V rows stored · q · (c W_UK) = (q W_UKᵀ) · c (worked check in §7) |
| 8 | A plain text ladder, per token per layer, toy beside DeepSeek-V3 shape: MHA · GQA-8 · GQA-2 · MQA · MLA. The toy column shows a dash in the GQA-8 row ("—, needs more than 8 query heads"). | Rows type in top to bottom; the MLA row lands between GQA-2 and MQA in the toy, and next to GQA-2 at DeepSeek-V3's shape. | MLA stores close to what two shared KV heads would, yet every head keeps its own keys and values. That is the trade it was designed for. | toy: 64 · — · 16 · 8 · 10 · V3 shape (128 heads × 128): 32,768 · 2,048 · 512 · 256 · 576 · MHA ÷ MLA = 56.9× · visible line: "at 128 heads of 128 numbers; V3's real query/key head is 192 wide, which would make the baseline 71×" (K 192 + V 128 per head: 40,960 ÷ 576 = 71.1) |
| 9 | **Key frame.** Real models per token (all layers, 2 bytes per number), as page text under the stage: GPT-3 (2020, MHA) · Llama-3.1-70B (2024, GQA-8) · MiniMax-M3 (2026, GQA-4) · DeepSeek-V3 (2024, MLA). On stage, four `kvStack`s whose tile count encodes nothing; each labeled with its bytes. Visible lines: "GPT-3 at 131,072 tokens is a what-if at its shape; its own context was 2,048" and "2026 moved on: DeepSeek-V4 keeps one wide KV head and compresses it (`long-context-attention`)". | The four stacks appear with their byte labels; then each shows its size at 131,072 tokens. | Real models stack these choices across every layer. From GPT-3 to DeepSeek-V3, sharing and latents cut the cache per token by 67 times. | 4,718,592 B (4.72 MB) · 327,680 B (328 kB) · 122,880 B (123 kB) · 70,272 B (70.3 kB) · at 131,072 tokens: 618 GB · 42.9 GB · 16.1 GB · 9.21 GB · 4,718,592 ÷ 70,272 = 67.1× |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. The query row keeps its position in frames 1–3 and 5–7; Q1 keeps its selection outline throughout.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 27 · 26 · 24 · 29 · 21 ·
23 · 25 · 27 · 23.

Absolutes checked (README lesson 7): frame 1's "every token stores all of them, in every layer" is
restricted to MHA, the scheme on screen; frame 3 says "a common choice", not "the standard" (01 §2:
"8 KV heads became the default compromise", with MiniMax-M3 at 4); frame 6's "cannot be rotated" is
01 §3: "RoPE would not commute with the absorbed up-projection".

Branches (README lesson 13): frame 4 leaves the 8-head layer for `attention`'s two heads and is labeled on
screen; frame 5 returns with the 8 query blocks redrawn.

## 6. Toy
Title on page: "Share, group or compress." One state object, one `render()`. A visible line above the
controls repeats the stand-in note.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `scheme` | Keys and values | segmented | MHA · GQA · MQA · MLA | MHA | — |
| `kvHeads` | KV heads (GQA) | Slider (snapped to divisors of the query heads) | toy: [1, 2, 4, 8]; real: the preset's divisors | 2 | — |
| `model` | Shape | preset chips | toy · GPT-3 · Llama-3.1-70B · MiniMax-M3 · DeepSeek-V3 | toy | from `data/models.json` (§8); a chip sets layers, query heads, head size and its real scheme |
| `latent` | Latent size (MLA) | Slider (snapped) | toy [4, 8, 16]; real [256, 512, 1,024] | 8 / 512 | DeepSeek-V3 → 512 |
| `context` | Tokens in the cache | Slider (snapped) | [2,048 · 32,768 · 131,072 · 1,048,576] | 131,072 | — |
| `pattern` | Same keys, different questions: head B reads | toggle | its own keys · head A's keys | head A's keys | — |

The position key stays at its preset value (toy 2, DeepSeek-V3 64) and is printed, not a slider.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Wiring diagram (which query head reads which KV head) | `kvGroups({ queryHeads, kvHeads })` | the frame 1–3 drawing |
| Stored per token per layer | `kvBytesPerToken({ layers: 1, kvHeads, headDim, bytesPerElem: 1 })` (numbers, not bytes); MLA: `kvBytesPerTokenMla({ layers: 1, dLatent, dRope, bytesPerElem: 1 })` | count |
| Bytes per token, all layers | `kvBytesPerToken({ layers, kvHeads, headDim, bytesPerElem: 2 })` or `kvBytesPerTokenMla(…)` | exact bytes + `formatBytes` (decimal) |
| Times smaller than MHA (the one definition, README lesson 16) | `kvBytesPerToken({ layers, kvHeads: queryHeads, headDim, bytesPerElem })` ÷ the scheme's bytes per token | "56.9×", 3 significant figures, trailing zeros dropped (12×, 56.9×); "—" when query heads are not in the data (Llama-3.1-70B until the data pass) |
| Cache at this context | `kvCacheBytes({ bytesPerToken, tokens: context })`; past the chip's `context_length` the label "a what-if at this shape; its own context was N" shows (README lesson 22) | decimal GB |
| Frame 4 heatmaps | `attentionHead(TOY.heads.B.Q, K, V)` from `math/attention.js` with K, V = head A's or head B's own | `heatmap` at `NUMBER_CELL`, 3 d.p. |

**Check my work**: numbers per token per layer = 2 × KV heads × head size (MLA: latent + position key); bytes per token = that × layers × 2 bytes; compared with MHA at this shape: N× smaller (`tests/kv-compression-expected.js`).

**Try this** (each leads to a named insight)
1. Shape **GPT-3**, scheme MHA: 4,718,592 B per token. Switch to **GQA** with 8 KV heads: 393,216 B (12×
   smaller); **MQA**: 49,152 B (96×). → **Insight: the cache shrinks exactly in proportion to KV heads;
   the 96 query heads, and the 96 patterns, stay** (`attention`'s head B on head A's keys is the toy proof:
   try-this 2).
2. Toggle **pattern** between "its own keys" and "head A's keys": head B's row for "cat" goes from [0.798,
   0.202] to [0.731, 0.269], and its row for "down" from most weight on "sat" (0.578) to 0.366 on "sat",
   0.285 on itself; head A's own row stays [0.321, 0.679] and [0.114, 0.656, 0.129, 0.101]. → **Insight:
   sharing keys changes what a head can find, not whether heads differ.** Patterns come from the queries,
   and every head keeps its own.
3. Shape **DeepSeek-V3**, scheme MLA: 576 numbers per token per layer, 70,272 B per token, 56.9× smaller
   than MHA at its 128 heads. Switch to GQA with 2 KV heads: 512 numbers; MQA: 256. → **Insight: MLA
   costs about as much memory as two shared KV heads but keeps a key and value per head; it pays in
   compute at decode instead** (GLM-5 enlarged its head dim and cut its head count to tame that, §8).

**Functions** (pure, no DOM; tests first). Byte counts reuse `math/memory.js` from `kv-cache`; the frame 4
pattern reuses `attentionHead` from `math/attention.js`. One addition to `math/memory.js`:

```js
// Which KV head each query head reads (0-based array index; drawn 1-based as Q1…Q8).
// Throws RangeError('kvGroups: kvHeads must divide queryHeads') otherwise.
kvGroups({ queryHeads, kvHeads }) → number[]
//   ({ queryHeads: 8, kvHeads: 8 }) → [0, 1, 2, 3, 4, 5, 6, 7]       (MHA)
//   ({ queryHeads: 8, kvHeads: 2 }) → [0, 0, 0, 0, 1, 1, 1, 1]       (GQA, frame 3)
//   ({ queryHeads: 8, kvHeads: 1 }) → [0, 0, 0, 0, 0, 0, 0, 0]       (MQA)
//   ({ queryHeads: 8, kvHeads: 3 }) → throws RangeError
```

Worked examples from `math/memory.js` (as specified in `kv-cache`):
```js
kvBytesPerToken({ layers: 1, kvHeads: 8, headDim: 4, bytesPerElem: 1 })   → 64      // toy MHA, numbers per layer
kvBytesPerToken({ layers: 1, kvHeads: 2, headDim: 4, bytesPerElem: 1 })   → 16      // toy GQA-2
kvBytesPerToken({ layers: 1, kvHeads: 1, headDim: 4, bytesPerElem: 1 })   → 8       // toy MQA
kvBytesPerTokenMla({ layers: 1, dLatent: 8, dRope: 2, bytesPerElem: 1 })  → 10      // toy MLA
kvBytesPerToken({ layers: 96, kvHeads: 8, headDim: 128, bytesPerElem: 2 }) → 393_216 // GPT-3 shape, GQA-8
kvBytesPerToken({ layers: 96, kvHeads: 1, headDim: 128, bytesPerElem: 2 }) → 49_152  // GPT-3 shape, MQA
kvBytesPerToken({ layers: 60, kvHeads: 4, headDim: 128, bytesPerElem: 2 }) → 122_880 // MiniMax-M3
kvBytesPerToken({ layers: 61, kvHeads: 128, headDim: 128, bytesPerElem: 2 }) → 3_997_696 // DeepSeek-V3 shape as MHA
kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 }) → 70_272 // DeepSeek-V3; 3_997_696 ÷ 70_272 = 56.9
kvCacheBytes({ bytesPerToken: 122_880, tokens: 131_072 })                  → 16_106_127_360  (16.1 GB)
```
Frame 4 (from `math/attention.js`, `attentionHead(TOY.heads.B.Q, TOY.heads.A.K, TOY.heads.A.V).weights`):
`[[1,0,0,0],[0.731,0.269,0,0],[0.119,0.685,0.196,0],[0.196,0.153,0.366,0.285]]`; its scaled scores
`[[0.625,−0.125,−0.125,0.5],[1,0,−0.25,0.625],[−0.375,1.375,0.125,−0.25],[0.25,0,0.875,0.625]]`
(masked above the diagonal).

Tests to write first: `kvGroups` returns `queryHeads` entries, each KV head serves exactly
`queryHeads / kvHeads` query heads, and throws on non-divisors; the examples above to the byte; "times
smaller than MHA" equals `queryHeads / kvHeads` for every GQA setting (README lesson 16: the prose claim
"shrinks exactly in proportion" is asserted); MLA toy (10) lies between MQA (8) and GQA-2 (16) and
DeepSeek-V3 (576) lies between GQA-2 (512) and GQA-4 (1,024), so frame 8's "close to two shared KV heads"
is checked at both shapes (lesson 17).

**Reproducer** (run from the repo root on 2026-10-07 against the shipped `math/core.js`, with the
`memory.js` formulas inlined; output matched every number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({matmul,transpose,softmax,causalMask,randomMatrix,formatBytes})=>{
const r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):Number(x.toFixed(d));
const head=(Q,K,V)=>{const S=matmul(Q,transpose(K)).map(row=>row.map(v=>v/2));const m=causalMask(Q.length);
 const W=S.map((row,i)=>softmax(row.map((v,j)=>m[i][j]?v:-Infinity)));return{S,W}};
const A={Q:[[1,0,-0.5,0],[0.5,0.5,0,-1],[0,2,0.5,0],[0.5,2,0,0]],K:[[1,-0.5,0,0.5],[0,1.5,0,-0.5],[-0.5,0,1,0.5],[0.5,-0.5,0.5,1]],V:[[1,0,-1,0],[0,2,0,1],[-1,0,1,0.5],[0.5,-1,0,1]]};
const B={Q:[[1,0,0,0.5],[2,0,0.5,0],[0,2,0,0.5],[0.5,0,2,0]],K:[[1.5,0,-0.5,0],[0,1.5,0,-0.5],[-0.5,0,1.5,0],[0,-0.5,0,1.5]],V:[[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]};
console.log("A",JSON.stringify(r(head(A.Q,A.K,A.V).W)),"Bshared",JSON.stringify(r(head(B.Q,A.K,A.V).W)),"Bown",JSON.stringify(r(head(B.Q,B.K,B.V).W)));
const kv=({layers,kvHeads,headDim,bytesPerElem})=>2*layers*kvHeads*headDim*bytesPerElem;
const mla=({layers,dLatent,dRope,bytesPerElem})=>layers*(dLatent+dRope)*bytesPerElem;
console.log("toy",[8,4,2,1].map(h=>kv({layers:1,kvHeads:h,headDim:4,bytesPerElem:1})),mla({layers:1,dLatent:8,dRope:2,bytesPerElem:1}));
console.log("v3 layer",[128,8,4,2,1].map(h=>kv({layers:1,kvHeads:h,headDim:128,bytesPerElem:1})),576,(32768/576).toFixed(1));
const g3=kv({layers:96,kvHeads:96,headDim:128,bytesPerElem:2}),ll=kv({layers:80,kvHeads:8,headDim:128,bytesPerElem:2}),
 m3=kv({layers:60,kvHeads:4,headDim:128,bytesPerElem:2}),v3=mla({layers:61,dLatent:512,dRope:64,bytesPerElem:2}),
 v3mha=kv({layers:61,kvHeads:128,headDim:128,bytesPerElem:2});
console.log("real",g3,ll,m3,v3,v3mha,(v3mha/v3).toFixed(1),(g3/v3).toFixed(1),"gpt3 shape",kv({layers:96,kvHeads:8,headDim:128,bytesPerElem:2}),kv({layers:96,kvHeads:1,headDim:128,bytesPerElem:2}));
console.log("at 131072",[g3,ll,m3,v3].map(x=>formatBytes(x*131072)).join(" "),"m3",m3*131072);
const c=randomMatrix(1,8,21,1),W=randomMatrix(8,4,22,0.5),q=randomMatrix(1,4,23,1);
console.log("v3 192-wide baseline",(192+128)*128/576,(40960/576).toFixed(1),"by hand",5*(1*3+2*4),15*1+20*2);
console.log("absorb",matmul(q,transpose(matmul(c,W)))[0][0].toFixed(6),matmul(matmul(q,transpose(W)),transpose(c))[0][0].toFixed(6));
})'
```
Output on 2026-10-07: `A [[1,0,0,0],[0.321,0.679,0,0],[0.095,0.703,0.202,0],[0.114,0.656,0.129,0.101]]` ·
`Bshared [[1,0,0,0],[0.731,0.269,0,0],[0.119,0.685,0.196,0],[0.196,0.153,0.366,0.285]]` ·
`Bown [[1,0,0,0],[0.798,0.202,0,0],[0.168,0.664,0.168,0],[0.129,0.146,0.578,0.146]]` ·
`toy [64,32,16,8] 10` · `v3 layer [32768,2048,1024,512,256] 576 56.9` ·
`real 4718592 327680 122880 70272 3997696 56.9 67.1 gpt3 shape 393216 49152` ·
`at 131072 618 GB 42.9 GB 16.1 GB 9.21 GB m3 16106127360` · `v3 192-wide baseline 71.11111111111111 71.1 by hand 55 55` · `absorb 0.412632 0.412632`.

## 7. Show me the math
```tex
\text{MHA: } K_h = X W_K^{h},\; V_h = X W_V^{h}\ \ (h = 1..n_H)
\qquad
\text{GQA: } K_{g(h)},\; V_{g(h)},\ \ g(h) = \Big\lfloor \tfrac{h-1}{n_H / n_{kv}} \Big\rfloor + 1
\qquad
\text{MQA: } n_{kv} = 1
```
```tex
\text{bytes per token} = 2 \cdot L \cdot \htmlClass{hl-kvh}{n_{kv}} \cdot d_{\text{head}} \cdot b
\qquad\text{(kv-cache)}
```
```tex
\text{MLA: } \htmlClass{hl-lat}{c_t} = x_t W_{DKV} \in \mathbb{R}^{d_c},\quad
K_h = \htmlClass{hl-lat}{c}\, W_{UK}^{h},\quad V_h = \htmlClass{hl-lat}{c}\, W_{UV}^{h},\quad
\htmlClass{hl-rope}{k^{R}_t} = \operatorname{RoPE}(x_t W_{KR}) \in \mathbb{R}^{d_{\text{rope}}}
```
```tex
\text{absorbed score: } q_h \cdot \big(c\, W_{UK}^{h}\big) = \big(q_h\, W_{UK}^{h\top}\big) \cdot c
\qquad \text{(checked numerically: } 0.412632 = 0.412632)
```
```tex
\text{by hand: } c = [1, 2],\ W_{UK}^{h} = [3, 4]^{\top},\ q = 5:\quad
q\,(c\,W) = 5 \cdot 11 = 55,\qquad (q\,W^{\top}) \cdot c = [15, 20] \cdot [1, 2] = 55
```
```tex
\text{MLA bytes per token} = L\,(\htmlClass{hl-lat}{d_c} + \htmlClass{hl-rope}{d_{\text{rope}}})\,b,
\qquad \text{DeepSeek-V3: } 61 \cdot (512 + 64) \cdot 2 = 70{,}272\ \text{B};\quad
\frac{2 \cdot 128 \cdot 128}{576} = 56.9
```
Shapes (toy in parentheses): X [n × d_model]; W_K^h, W_V^h [d_model × d_head] (· × 4); per token per
layer the cache holds 2 · n_kv · d_head numbers (64 / 16 / 8) or d_c + d_rope (8 + 2 = 10). W_DKV
[d_model × d_c], W_UK^h and W_UV^h [d_c × d_head]. Color links: `hl-kvh` → the stored `kvStack`s in frames
1–3; `hl-lat` → the latent cells in frames 5–7; `hl-rope` → the position-key cells in frame 6. KaTeX with
`trust: true, strict: false`. The panel adds one sentence on frame 6: the position rotation depends on the
token's position, so it cannot be folded into a fixed matrix the way W_UK is; the position key is kept
outside the latent for that reason (01 §3, decoupled RoPE).

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Almost no 2026 model uses plain MHA. Most use GQA with 4 or 8 KV heads or
MLA; DeepSeek-V4 went back to one very wide shared KV head and compresses it further
(`long-context-attention`)."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GPT-3 (2020): MHA, 96 KV heads × 128, 4,718,592 B per token | `models.gpt-3.n_heads`, `.n_kv_heads`, `.head_dim`, `.kv_bytes_per_token` (proposed in `kv-cache` §12 / `decoder-anatomy` §12.3) | 01 §4 table |
| Llama-3.1-70B (2024): GQA with 8 KV heads × 128, 327,680 B per token | `models.llama-3.1-70b.n_kv_heads`, `.head_dim`, `.kv_bytes_per_token` (entry proposed by `paged-attention` §13) | 01 §2 GQA paragraph, §4 table |
| gpt-oss-120b (2025): 64 query heads share 8 KV heads, each 64 wide | `models.gpt-oss-120b.n_heads` = 64, `.n_kv_heads` = 8, `.head_dim` = 64 (proposed, `attention` ruling 2) | 01 §7 [C] |
| MiniMax-M3: 64 query heads, 4 KV heads × 128, 122,880 B per token (derived) | `models.minimax-m3.n_heads` = 64, `.n_kv_heads` = 4, `.head_dim` = 128 (proposed, `attention` ruling 2), `.kv_bytes_per_token` = 122,880 (proposed, derived) | 01 §2 MSA bullet [C], §4 table (derived) |
| Qwen3.8 (2026): its attention layers use 64 query heads and 4 KV heads | `models.qwen3.8.attention` (existing, confirmed) | 01 §7 [C] |
| DeepSeek-V3 (Dec 2024): MLA, latent 512 + position key 64 per layer: 70,272 B per token, 56.9× less than MHA at its 128 heads | `models.deepseek-v3.mla_kv_rank` = 512, `.mla_rope_dim` = 64, `.n_heads` = 128, `.kv_bytes_per_token` = 70,272 (entry proposed, `paged-attention` §13, `decoder-anatomy` §12.3) | 01 §2 MLA paragraph |
| MLA is also used by Kimi K3 (in its 24 full-attention layers) and GLM-5.3 (latent 512, query latent 2,048) | `models.kimi-k3.attention` (existing, confirmed); `models.glm-5.3.attention` (existing, confirmed) + `.mla_kv_rank` = 512, `.mla_q_rank` = 2048 (proposed; source its config.json) | 01 §2 MLA paragraph [C] |
| The cost: MLA's decode compute is high; GLM-5 enlarged the head dim from 192 to 256 and cut the head count ("MLA-256"), and needed a modified optimizer (Muon Split) to match GQA-8 quality | `models.glm-5.attention_note` (proposed *new* entry `glm-5`; source arXiv 2602.15763, confirmed) | 01 §2 MLA paragraph [C] |
| DeepSeek-V4-Pro (2026): one KV head, 512 wide, shared by all query heads, then compressed (`long-context-attention`) | `models.deepseek-v4-pro.attention` (existing, confirmed) | 01 §2 [C] |

Not shown: GLM-5.3's bytes per token (its position-key size is assumed, 01 §4 [U]); Kimi K3's latent
sizes (not read, 01 §4 [U]); Mistral Large 4's attention (undisclosed, 01 §7).

## 9. Takeaways
1. The cache counts KV heads, not query heads. MQA shares one key/value set among all query heads, GQA
   shares within groups (8 KV heads is common), and the cache shrinks in exact proportion (frames 1–3,
   try-this 1).
2. Sharing keys does not make heads alike: each head keeps its own queries and so its own pattern
   (frame 4, try-this 2).
3. MLA stores one short latent per token (plus a small position key) and rebuilds each head's keys and
   values from it: about the memory of two shared KV heads, paid for with extra compute at decode
   (frames 5–8, try-this 3). It is the 2024–26 incumbent, not the endpoint.

## 10. Next and go deeper
Next: `long-context-attention` (windows, sparse reads, compressed and linear layers: the other ways to
shrink or skip the cache) and `paged-attention` (how a serving engine lays these bytes out in memory);
both list `kv-compression` in `shared/concepts.json`. In-page link: `rope` (why the position key is
separate).

Go deeper (brief 05 §1.1, §1.2; 01 §2): Sebastian Raschka, LLM Architecture Gallery
(https://sebastianraschka.com/llm-architecture-gallery/), with GQA and MLA explainers per model and a
memory calculator · Ainslie et al., *GQA* (https://arxiv.org/abs/2305.13245) · DeepSeek-AI, *DeepSeek-V2*
(https://arxiv.org/pdf/2405.04434), the paper that introduced MLA.

## 11. Key-frame sketch
Frame 3 (GQA with two groups), desktop width; the readout and frame 9's numbers from the §6 reproducer
(`toy [64,32,16,8] 10`, `real …`).
```text
┌──────────────────────────────────────────────────────────┐
│ GQA                         stored per token per layer   │
│ ►[Q1][Q2][Q3][Q4]  [Q5][Q6][Q7][Q8]        16 numbers    │
│    \  |   |  /      \  |   |  /           (MHA: 64)      │
│     \ |   | /        \ |   | /                           │
│      [K▪▪▪▪]           [K▪▪▪▪]                           │
│      [V▪▪▪▪]           [V▪▪▪▪]                           │
│     group of 4        group of 4                         │
├──────────────────────────────────────────────────────────┤
│ Grouped-query attention is the middle way: query heads   │
│ share within groups. Two groups here; 8 KV heads became  │
│ a common choice in real models.                          │
│ [◄] [Pause] [►]  ━━━━●━━━━━━━  3 / 9   speed [1×]        │
└──────────────────────────────────────────────────────────┘
 Per token (frame 9): GPT-3 4.72 MB · Llama-3.1-70B
 328 kB · MiniMax-M3 123 kB · DeepSeek-V3 70.3 kB
```
"►" is the selection outline on Q1, the followed query head. Each stack is one KV head with the four
tokens' tiles. At 400 px the stage scrolls inside its container.

## 12. Open questions for the reviewer
**Data-pass keys:**
1. Reuse `attention` ruling 2's `n_heads`, `n_kv_heads`, `head_dim` for gpt-3, gpt-oss-120b, minimax-m3,
   deepseek-v3, llama-3.1-70b. *New:* `minimax-m3.kv_bytes_per_token` = 122,880 (derived, 01 §4);
   `glm-5.3.mla_kv_rank` = 512, `.mla_q_rank` = 2048 (01 §2, its config.json, confirmed).
2. *New entry* `glm-5` (Feb 2026, arXiv 2602.15763): `attention_note` = "MLA-256: head dim 192 → 256,
   fewer heads; Muon Split needed to match GQA-8", confirmed. Alternatively attach the note to `glm-5.3`
   (whose card says "no architectural changes"); the reviewer picks.
3. `llama-3.1-70b.n_heads` (query heads) is not in any brief; without it the "times smaller than MHA"
   output shows "—" for Llama. Add it from the Llama 3.1 config with a re-verification line, or leave
   the dash.

**Graph changes:** none. `rope` is an in-page link, not a prerequisite (frame 6 names the position key and
defers the why).

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (1 Must). Status is now "approved (expert review)". Every number touched was
regenerated with the §6 reproducer (the 71.1× baseline and the by-hand absorption line were added to it).

Rulings applied (lesson 20): 8 query heads in the toy, accepted; toy MLA 8 + 2, accepted (ordering tested at
both shapes); MHA baseline follows the brief (2 · 128 · 128) with the 192-wide qualification printed on frame
8; one absorption frame, no branch label. Data pass: add `llama-3.1-70b.n_heads` (64) from its config with a
re-verification line, so the Llama chip prints its ratio.

Must (1/1): decimal units in frame 9, §6 and §11.
Should (7/7, no rebuttals): hook names its baseline ("than it would with a key and value per head");
misconception 4 now cites frame 9's page text ("2026 moved on: DeepSeek-V4 …") and try-this 3; a by-hand
absorption line in §7; frame 8 prints the 192-wide qualification (71×); try-this 1 points to try-this 2;
"Almost no 2026 model uses plain MHA" kept; lesson 22 what-if line on frame 9 and in the toy.
Nice (2/2): the toggle is labeled "Same keys, different questions"; the toy ladder carries a GQA-8 dash.
