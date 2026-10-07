# Reaching 1M tokens (`long-context-attention`)

Track: architecture · Section: architecture · Prereqs: kv-compression, rope
Next: `model-card` (the one slug whose `prereqs` list `long-context-attention` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1 ("Attention sink" row), §2 (formula block; sliding window; sparse attention; hybrid linear attention; FlashAttention), §3 ("What is mainstream for 1M"), §4 table, §7 · 04 §3.1, §7.5, §8.2, §10 item 3 · 05 §1.1, §2. Nothing beyond the briefs.

This page owns the attention *sink* mechanism (with numbers) and the misconception the spec names: "sparse
attention shrinks the KV cache". `decoder-recap` names sinks and hybrid layers in one line each and links
here. It reuses `attention`'s heatmap convention (rows = queries, columns = keys, hatched = not read),
`kv-cache`'s `kvStack` and bytes-per-token vocabulary, `kv-compression`'s KV heads and latents, and adds two
functions to `math/memory.js` (`stackKvBytes`, `linearStateBytes`).

## 1. Learning objective
After this page you can name the two costs of full attention at length, reads per token and entries stored
(frame 1); say what a sliding window, an attention sink, a sparse top-k read, a compressed entry and a
linear-attention state each change about those two costs (frames 2–8, toy); and explain the two routes 2026
models take to a million tokens (frames 9–10).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "Sparse attention shrinks the KV cache." → **Reality:** sparse attention (DSA,
  MiniMax's MSA) chooses which stored keys to *read*; it must still store every token so its indexer can
  look at them. Reads per token drop (2,048 of 1,048,576 at 1M for DSA's top-k); storage does not. Only
  merging tokens (compressed attention), windows and linear-attention layers shrink the cache. (01 §2:
  "DSA does not shrink the KV cache"; MSA: "Like DSA, KV is still stored") · corrected by frame 6,
  try-this 1
- **Misconception:** "A sliding-window layer cannot use anything older than its window." → **Reality:**
  that one layer cannot, but window layers are interleaved with full layers (5 : 1 in MiMo-V2-Flash, 1 : 1 in
  gpt-oss), and the full layers carry old information forward into the stream. (01 §2 sliding-window
  bullets) · corrected by frame 3
- **Misconception:** "Softmax always has to give its weight to some real token." → **Reality:** a learned
  sink logit gives the row a "nothing here" option; in the toy it takes 0.600 of the weight when no key in
  the window is relevant. (01 §1 "Attention sink" row; §2 sliding window) · corrected by frame 4,
  try-this 2
- **Misconception:** "Linear attention is just a faster version of the same thing." → **Reality:** it
  keeps a fixed-size state instead of every token, which is why it is cheap, and also why it recalls one
  exact old token less reliably; hybrids keep 1 full-attention layer in 4 for that. (01 §2 hybrid bullet:
  "the 1-in-4 full-attention layers do exact retrieval") · corrected by frames 8–9, try-this 3
- **Misconception:** "There is one standard way to reach 1M tokens in 2026." → **Reality:** the field is
  split. DeepSeek, GLM and MiniMax kept softmax attention and made it sparse or compressed; Qwen and
  Moonshot mixed in linear-attention layers. (01 §2: "the field is split") · corrected by frame 10

## 3. Hook and intuition (final wording)
**Hook:** At a million tokens, full attention would make each new token read a million keys in every layer
and keep a million entries per conversation. How do 2026 models get there, and what do they give up?

Full attention has two costs that grow with the conversation. Every new token's query reads every stored
key, so the *reading* per token grows with the length. And the cache keeps one entry for every token, so
the *memory* grows too. `kv-compression` made each entry smaller. This page changes how many entries are
read, how many are kept, or both.

The simplest move is a window: a layer only looks at its last few thousand tokens, or its last 128, and
forgets the rest, so both costs stop growing. To keep long-range ability, models interleave window layers
with full ones. A window needs a "nothing here" option, because softmax must put all its weight somewhere;
a learned sink logit gives it one. Sparse attention keeps everything but reads only the most promising
entries, picked by a cheap scorer: less reading, the same memory. Compressed attention merges several
tokens' keys and values into one entry, which cuts both.

The other route replaces most attention layers with linear attention: a fixed-size memory matrix that
every token writes into and every query reads from, so it never grows. It is cheap and blurry, so
models keep one full-attention layer in four for exact recall. Each trick costs something: windows forget,
sparse reads can miss, compression blurs neighbors, and retrofitting windows onto a model trained with full
attention went badly in GLM-5's tests.

## 4. Visual metaphor
**Toy:** 16 positions (the first four are "The cat sat down"; the rest are filler words, printed as numbers
1–16 on the axes), one head. The followed query is **token 16**, the newest one: in decode it is the token
that reads the cache (as "on₅" was on `kv-cache`). Window w = 4; sparse top-k k = 4; compression merges
m = 4 tokens into one entry and reads the top 1 merged entry plus the window. Indexer scores for the sparse
frame are `randomMatrix(16, 16, 7)` (seeded, from `math/core.js`); the sink frame's window scores
[−1, −0.5, −1, −0.75] and the linear-attention frame's K, V and q are hand-picked or reused from
`attention` (head A, tokens 1–3, query "sat"). A visible line under the stage (README lesson 10): "Sizes
are toy choices (16 tokens, window 4, top 4); the indexer scores are seeded random numbers and the window
scores are hand-picked. Real models use windows of 128 to 1,024 and top-k of 1,024 to 2,048."

**Terms introduced, one per frame** (README lesson 3): 1 none (two costs, named from `kv-cache`) · 2
sliding window · 3 local and global layers · 4 attention sink · 5 indexer · 6 none (storage vs reading) ·
7 compressed entry · 8 linear-attention state · 9 hybrid · 10 none (the split). **Terms assumed from
prereqs:** score, softmax, causal mask, heatmap (`attention`); KV cache, bytes per token, decode
(`kv-cache`); KV heads, MLA (`kv-compression`); RoPE base (`rope`). **Named and deferred:** staged length
training (`midtraining`); how a serving engine pages the remaining cache (`paged-attention`); FlashAttention
(timeless prose in §8, from `attention`).

**Indexing:** token positions 1–16 on screen, 1-based. No addresses.

**Layout** (stage ≈ 580 × 366):
- Left: the pattern grid, a `heatmap` 16 × 16 at 18 px cells (288 × 288 px). Cells hold no numbers (README
  constraint: small grids are hover-only, and nothing load-bearing is in them): a cell is either *read*
  (filled with the read tint) or *not read* (hatched, as masked cells on `attention`). Row 16 carries the
  selection outline in every frame. Compressed entries draw as one cell spanning 4 columns.
- Right top: a `kvStack` "stored in this layer" (1 tile per stored entry; 16 at most at 14 px = 256 px).
- Right bottom: two plain text counters for row 16: "entries read" and "entries stored", plus "cells read,
  all rows".
- Frame 4: a row `vector` at `NUMBER_CELL` of the four window weights, then five with the sink (5 × 43 =
  215 px).
- Frame 8: the state `matrix` S [4 × 4] at `NUMBER_CELL` (172 × 172 px) replaces the grid; beside it the
  output `vector`.
- Frames 3 and 9: `blockStack` (`shown = 2`, halves labeled by layer kind), with page text under the stage.

Glyphs used (from spec §5.1 and the accepted `decoder-anatomy` proposals): heatmap, kvStack, vector, matrix,
token (row 16's chip), block, flow (carry `kv`), blockStack.
New glyphs proposed: none. Small marks (README lesson 15): counters, the "1 entry = 4 tokens" label, the
layer-kind labels and the frame 10 table are plain text labels. The indexer's 16 scores in frame 5 are a
row `vector` at 18 px (hover-only); the four picked positions are printed as text.

Color: "read" cells use the Architecture accent's light tint (a state, not an amount); unread cells are
hatched. Weights in frames 4 and 8 use the value scale (maxAbs 1 for weights, 6 for S and the output). Cache
tiles use `--sem-memory`. The followed query (row 16, or "sat" in frame 8) carries the selection outline.

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Full causal grid 16 × 16 (lower triangle read, upper hatched); row 16 outlined; `kvStack` with 16 tiles. Counters. Page-text line: "at 1M tokens: 1,048,576 reads per token, per query head, per layer". The two counters carry the visible legend "reads" (compute) and "stored" (memory). | The grid fills row by row; the stack grows to 16. | Full attention has two costs that grow with length. Each new token reads every stored key, and the cache keeps every token. | row 16 reads 16 · stored 16 · all rows 136 cells (16 × 17 ÷ 2) · at 1M: 1,048,576 reads |
| 2 | The grid becomes a band of width 4; the `kvStack` keeps only the 4 newest tiles, older ones slide off and fade. | Cells outside the band hatch over, diagonal by diagonal; old tiles leave the stack. | A sliding-window layer reads only its last 4 tokens, so the cache stops growing at 4 entries. Anything older is invisible to this layer. | row 16 reads 13–16 (4) · stored 4 · all rows 58 cells · real windows: 128 (gpt-oss, MiMo-V2-Flash); Gemma 3's 1,024 shows only if the data pass adds its source |
| 3 | `blockStack` of 6 layers: 5 labeled "window", 1 labeled "full"; a `flow` dot from token 1 rides the residual lane up through the full layer. Page text: "MiMo-V2-Flash: 5 window : 1 full, window 128 · gpt-oss-120b: 1 : 1, window 128" and a plain cost chip "GLM-5: adding windows to an already trained model hurt long-context tasks badly". | The stack builds; the dot from token 1 is picked up by the full layer and carried up to the top. | Models interleave window layers with full ones. The full layers still read everything and carry old information forward through the stream. | 1 of 6 layers grows a cache (MiMo-V2-Flash style) · gpt-oss-120b: 18 window + 18 full layers · the window layers themselves never see old tokens, so enough full layers stay for recall |
| 4 | Back to row 16's window: a row `vector` of four weights [0.203, 0.334, 0.203, 0.260] labeled "scores all low"; then a fifth cell "sink" slides in at the left and the weights redraw. | The four cells shrink as the sink cell grows to 0.600; "Σ = 1.000" stays printed. | Softmax must spend all its weight somewhere, even when nothing in the window matters. A learned sink logit gives it a place to put that weight. | window scores [−1, −0.5, −1, −0.75] → [0.203, 0.334, 0.203, 0.260] · with sink logit 1.0 → [0.081, 0.134, 0.081, 0.104] + sink 0.600 |
| 5 | Full grid restored. Under row 16 a thin 16-cell `vector` "indexer scores" (18 px, hover-only); the four highest light up and print "1, 5, 6, 13". Row 16 of the grid: only those 4 cells read. | The indexer row fills; four picks rise; the grid's row 16 keeps 4 cells, the others hatch. | Sparse attention runs a cheap indexer over every stored key, then lets the real attention read only the top few. Here token 16 reads 4 of 16. | indexer picks for row 16: 1, 5, 6, 13 · entries read 4 · all rows 58 cells · DSA: top 2,048 of every past token · the indexer still scores every stored key, cheaply |
| 6 | A plain text prompt first, for one second: "Did the cache shrink?" Then the same grid; the `kvStack` beside it still has 16 tiles; the unread 12 are drawn dim, not removed. Counter "stored: 16". Plain label "the indexer needs them all". | The 12 unread tiles dim; none leave. | Reading fewer keys does not mean storing fewer. Every token stays in the cache, because the indexer must score all of them each step. | read 4 · stored 16 · at 1M: read 2,048, stored 1,048,576 |
| 7 | Columns 1–16 merge in groups of 4 into 4 wide cells; the `kvStack` becomes 4 wide tiles labeled "1 entry = 4 tokens" plus 4 narrow window tiles. Row 16 reads the top merged entry (tokens 1–4) and its window 13–16. | Tiles fuse in fours; row 16's read set forms: one wide cell plus the band. | Compressed attention merges every 4 tokens' keys and values into one entry, then reads the best few plus a short window. This cuts both reading and memory. | stored 4 merged + 4 window = 8 (was 16) · row 16 reads 1 merged + 4 = 5 entries · DeepSeek-V4-Pro at 1M: CSA stores 262,144 merged entries (every 4) and reads 1,024 + 128 window; HCA merges every 128 and reads all 8,192 |
| 8 | The grid gives way to a `matrix` S [4 × 4] "state" and the three tokens The, cat, sat feeding it; then the query "sat" (outlined) reads it. A counter: "cache after 16 tokens: 128 numbers · state: 16, always". Visible line under the state: "No softmax here: the output is a raw-score blend (−1, 3 and 0.5 times each value), so compare the *shape* with softmax's row, not the size." | Each token adds its value-times-key grid into S (three fills); then q_sat multiplies S and the output `vector` fills. | Linear attention keeps a fixed 4 by 4 state instead of a cache. Each token adds its value-times-key grid; a query reads the state with one multiplication. | S after 3 tokens = [[1.5, −0.5, −1, 0], [0, 3, 0, −1], [−1.5, 0.5, 1, 0], [−0.25, 1.5, 0.5, −0.25]] · output S·q_sat = [−1.5, 6, 1.5, 3.25] (softmax attention gave [−0.106, 1.407, 0.106, 0.804]) |
| 9 | `blockStack`: 3 "linear" layers then 1 "full", repeated; only the full layer has a `kvStack`. Page text: "Qwen3.8: 23 × (3 linear + 1 full) = 92 layers · Kimi K3: 69 linear + 24 full (MLA) = 93". | The stack builds in groups of four; a cache appears beside each fourth layer only. | Hybrid models make three of every four layers linear and keep one full layer for exact recall. Only the full layers grow a cache. | Qwen3.5-397B: 15 of 60 layers full, 30,720 B per token (32.2 GB at 1M) + a fixed state per linear layer |
| 10 | **Key frame.** Two lanes labeled "softmax, made sparse or compressed" (DeepSeek-V4, GLM-5.3, MiniMax-M3) and "linear hybrids" (Qwen3.8, Kimi K3); the frame 10 table as page text under the stage. | Each lane's models type in; then both lanes point to one label "1M tokens". | 2026 models split into two routes to a million tokens: sparse or compressed softmax attention, or linear-attention hybrids. Both also train in stages up to that length. | see the table under the stage (reads and stored entries per layer at 1,048,576 tokens) |

Frame 10 page text (under the stage; reads and stored entries per attention layer per token at 1,048,576
tokens, from `readsAndStores`):
```text
                          reads per token     stored entries
full attention            1,048,576           1,048,576
window 128                128                 128
DSA top-2,048             2,048 (+ indexer)   1,048,576
MSA 16 blocks of 128      about 2,048 (the    1,048,576
                          newest block may
                          be extra)
CSA (merge 4, top 1,024)  1,024 + 128         262,144 + 128
HCA (merge 128)           8,192 + 128         8,192 + 128
linear layer              fixed state         fixed state
```

Determinism: every frame is a pure function of (step, progress); the sparse picks are fixed by the seed.
Reduced motion shows each frame's end state. The grid keeps its position in frames 1–7; row 16 keeps its
outline; the stack keeps its position.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 22 · 24 · 21 · 26 · 27 ·
24 · 27 · 27 · 24 · 27.

Absolutes checked (README lesson 7): frame 6's "every token stays in the cache" is DSA and MSA (01 §2:
"you still store everything so the indexer can look at it"; "Like DSA, KV is still stored"). Frame 4's
"must spend all its weight somewhere" is softmax summing to 1 (first principles). Frame 9's "three of every
four" is the Qwen pattern (01 §2: "3 Gated-DeltaNet : 1 gated full attention"); Kimi K3 is 69 : 24, about
3 : 1, printed beside it.

Branches (README lesson 13): frame 3 and frame 9 switch to the layer stack and are labeled by layer kind;
frame 8 swaps the grid for the state and says so in its caption.

## 6. Toy
Title on page: "What does each trick read, and what does it keep?" One state object, one `render()`. The
stand-in line from §4 is printed above the controls.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `pattern` | Layer type | segmented | full · window · window + sink · sparse top-k · compressed · linear | full | — |
| `window` | Window | Slider (snapped) | [2, 4, 8] tokens | 4 | — |
| `topK` | Top-k (sparse, compressed) | Slider (snapped) | [1, 2, 4, 8] | 4 | — |
| `merge` | Tokens per entry (compressed) | Slider (snapped) | [2, 4, 8] | 4 | — |
| `query` | Follow token | Slider | 1–16 | 16 | — |
| `real` | At real scale | preset chips | full · gpt-oss (window 128) · GLM-5.3 (DSA 2,048) · MiniMax-M3 (MSA) · DeepSeek-V4-Pro (CSA / HCA) · Qwen3.5 (hybrid) | full | values from `data/models.json` (§8) |
| `context` | Context | Slider (snapped) | [131,072 · 1,048,576] | 1,048,576 | — |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Pattern grid | `attentionPattern({ n: 16, kind, window, topK, merge, scores: randomMatrix(16, 16, 7) }).mask` | `heatmap`, read vs hatched |
| Entries read by the followed token | `.readsPerRow[query − 1]` | count; "(1 entry = 4 tokens)" when compressed |
| Cells read, all rows | `.cellsRead` | count |
| Entries stored in this layer | `.stored` | count, and the `kvStack` |
| Real scale: reads and stored per token per layer | `readsAndStores({ kind, n: context, window, topK, merge })` | counts, `formatCount` |
| Real scale: cache per conversation, whole model | `stackKvBytes({ groups, tokens: context })` for gpt-oss and Qwen3.5; V4-Pro shows both layer mixes as a range | decimal GB; "estimate" label on V4-Pro |
| Sink split (window + sink) | `core.softmax([...windowScores, sinkLogit])` with a `sinkLogit` slider shown only in this mode, [−2, 0, 1, 2], default 1 | 5 cells at `NUMBER_CELL` |
| Linear mode: state and output | `linearState({ keys, values, gate })`, `linearRead(S, q)` with a `gate` slider shown only in this mode, [1, 0.5] | `matrix` 4 × 4 and a 4-cell `vector` |

**Try this** (each leads to a named insight)
1. Pattern **sparse top-k**, k = 4: token 16 reads 4 entries, stores 16. Switch to **compressed** (merge 4,
   top 1): reads 5, stores 8. At real scale, tap **GLM-5.3** then **DeepSeek-V4-Pro**: DSA reads 2,048 and
   stores 1,048,576; CSA reads 1,152 and stores 262,272. → **Insight: picking what to read saves compute;
   only merging, windows or a fixed state save memory.**
2. Pattern **window + sink**, sink logit 1: the sink takes 0.600 and the four tokens share 0.400. Set the
   sink logit to **−2**: [0.189, 0.311, 0.189, 0.242] + sink 0.069, almost the no-sink row. → **Insight:
   the sink is a learned "nothing here" option; when a window holds nothing useful, the layer can attend to
   nothing instead of to noise.**
3. Pattern **linear**, gate 1: output for "sat" is [−1.5, 6, 1.5, 3.25], weighted by the raw scores −1, 3,
   0.5 with no softmax. Set the gate to **0.5**: the state halves before each new token, output [−0.75, 3,
   0.75, 1.75]. The state stays 16 numbers however long the text. → **Insight: a linear layer trades exact
   lookup for a fixed-size memory that blends and fades; that is why hybrids keep some full layers.**

**`math/longctx.js`** (pure, no DOM, inputs never mutated; tests first; imports `softmax`, `randomMatrix`
from `./core.js`):
```js
// Which (query, key) cells a layer reads in an n-token toy, plus per-row reads and what it stores.
// kind: 'full' | 'window' | 'sparse' | 'compressed'. sparse: top-k keys ≤ the query by `scores` (ties → lower index).
// compressed: the window, plus the top-k merged entries among the merged groups that end before the window
// (a group's score = the sum of its tokens' scores). readsPerRow counts entries (a merged entry counts 1).
attentionPattern({ n, kind, window, topK, merge, scores }) → { mask, readsPerRow, cellsRead, stored }
//   ({ n: 16, kind: 'full' })                                   → readsPerRow[15] 16 · cellsRead 136 · stored 16
//   ({ n: 16, kind: 'window', window: 4 })                      → readsPerRow[15] 4 · cellsRead 58 · stored 4
//   ({ n: 16, kind: 'sparse', topK: 4, scores: S7 })            → row 16 reads tokens [1, 5, 6, 13] · cellsRead 58 · stored 16
//   ({ n: 16, kind: 'compressed', window: 4, topK: 1, merge: 4, scores: S7 })
//                                                               → row 16 reads tokens 1–4 (one entry) + 13–16 · readsPerRow[15] 5 · stored 8
//   (S7 = randomMatrix(16, 16, 7); its row 16, 2 d.p.: [0.85, 0.35, 0.59, −0.33, 0.82, 0.94, 0.69, −0.99,
//    −0.67, 0.51, −0.79, −0.83, 0.82, −0.46, 0.18, −0.48])

// Closed forms at real scale, per token per layer. 'msa': topBlocks · blockSize read (the always-kept recent block
// is counted inside topBlocks; Open question 3). 'linear': reads and stores a fixed state (returns { fixed: true }).
readsAndStores({ kind, n, window, topK, merge, topBlocks, blockSize }) → { reads, stored, indexed }
//   ({ kind: 'full', n: 1_048_576 })                                  → { reads: 1_048_576, stored: 1_048_576 }
//   ({ kind: 'window', n: 1_048_576, window: 128 })                   → { reads: 128, stored: 128 }
//   ({ kind: 'sparse', n: 1_048_576, topK: 2048 })                    → { reads: 2048, stored: 1_048_576, indexed: 1_048_576 }
//   ({ kind: 'msa', n: 1_048_576, topBlocks: 16, blockSize: 128 })    → { reads: 2048, stored: 1_048_576, indexed: 8192 }
//   ({ kind: 'compressed', n: 1_048_576, merge: 4, topK: 1024, window: 128 })   → { reads: 1152, stored: 262_272 }
//   ({ kind: 'compressed', n: 1_048_576, merge: 128, topK: Infinity, window: 128 }) → { reads: 8320, stored: 8320 }   (HCA: dense over merged entries)

// Linear attention with a scalar gate: S ← gate · S + v kᵀ for each token; read o = S q.
linearState({ keys, values, gate = 1 }) → number[][]      // [d_v × d_k]
linearRead(S, q) → number[]
//   linearState({ keys: A.K.slice(0, 3), values: A.V.slice(0, 3) })
//     → [[1.5, −0.5, −1, 0], [0, 3, 0, −1], [−1.5, 0.5, 1, 0], [−0.25, 1.5, 0.5, −0.25]]      (exact)
//   linearRead(that, [0, 2, 0.5, 0]) → [−1.5, 6, 1.5, 3.25]                                   (exact)
//   linearState({ …, gate: 0.5 }) → [[0.75, −0.125, −1, −0.375], [0, 1.5, 0, −0.5], [−0.75, 0.125, 1, 0.375], [−0.25, 0.75, 0.5, 0]]
//   linearRead(that, q_sat) → [−0.75, 3, 0.75, 1.75]                                          (exact)
```

**Additions to `math/memory.js`** (owned by `kv-cache`; new names, specified here):
```js
// Whole-model cache for a stack of layer groups. kind 'full': grows by bytesPerTokenPerLayer per token per layer;
// 'window': fixed at window × bytesPerTokenPerLayer per layer; 'compressed': tokens / merge entries of
// bytesPerEntry per layer (exact division, as brief 04 §8.2 does); 'linear': fixed stateBytes per layer.
stackKvBytes({ groups, tokens }) → { perToken, fixed, total }
//   gpt-oss-120b: groups [{ layers: 18, kind: 'full',   bytesPerTokenPerLayer: 2048 },
//                         { layers: 18, kind: 'window', window: 128, bytesPerTokenPerLayer: 2048 }], tokens 131_072
//     → { perToken: 36_864, fixed: 4_718_592, total: 4_836_556_800 }       (4.84 GB; 2048 = 2 · 8 KV · 64 · 2 B)
//   Qwen3.5-397B: [{ layers: 15, kind: 'full', bytesPerTokenPerLayer: 2048 }, { layers: 45, kind: 'linear', stateBytes: null }]
//     → { perToken: 30_720, fixed: null (state size not in the brief), total at 1_048_576: 32_212_254_720 + state }
//   DeepSeek-V4-Pro estimate, tokens 1_000_000, bytesPerEntry 512 · b:
//     [{ 30, 'compressed', merge 4 }, { 30, 'compressed', merge 128 }]  b = 1 → 3_960_000_000 · b = 2 → 7_920_000_000
//     [{ 45, 'compressed', merge 4 }, { 15, 'compressed', merge 128 }]  b = 1 → 5_820_000_000 · b = 2 → 11_640_000_000
//     (per token 3,960–11,640 B: the data file's reported 4,000–12,000 B range; window branch and indexer keys left out, as in 04 §8.2)

// Fixed state of linear-attention layers: layers · heads · dKey · dValue · b.
linearStateBytes({ layers, heads, dKey, dValue, bytesPerElem }) → number
//   ({ layers: 1, heads: 1, dKey: 4, dValue: 4, bytesPerElem: 2 }) → 32      (the toy's 16 numbers)
```

Tests to write first: every example above; `attentionPattern(kind: 'sparse').stored === n` and
`kind: 'window'` stores `min(n, window)` (the misconception-1 claim as a test, README lesson 16);
`readsAndStores` sparse `stored === n` for every n; `cellsRead` of full is n(n+1)/2; each mask row only
reads keys ≤ its query (causal); `linearState` with gate 1 equals Σ vᵢkᵢᵀ; `stackKvBytes` totals are linear
in tokens for 'full' and constant for 'window' and 'linear'; inputs not mutated.

**Reproducer** (run from the repo root on 2026-10-07 with these functions inlined; output matched every
number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({softmax,randomMatrix,formatBytes})=>{
const r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):Number(x.toFixed(d)),n=16,S=randomMatrix(16,16,7,1);
const mask=(kind,o={})=>Array.from({length:n},(_,i)=>{
 if(kind==="full")return Array.from({length:n},(_,j)=>j<=i);
 if(kind==="window")return Array.from({length:n},(_,j)=>j<=i&&j>i-o.w);
 if(kind==="sparse"){const s=S[i],c=Array.from({length:i+1},(_,j)=>j).sort((a,b)=>s[b]-s[a]||a-b).slice(0,o.k);return Array.from({length:n},(_,j)=>c.includes(j));}
 const rr=Array(n).fill(false);for(let j=Math.max(0,i-o.w+1);j<=i;j++)rr[j]=true;const nb=Math.floor(Math.max(0,i-o.w+1)/o.m);
 Array.from({length:nb},(_,b)=>({b,s:S[i].slice(b*o.m,b*o.m+o.m).reduce((a,x)=>a+x,0)})).sort((a,b)=>b.s-a.s||a.b-b.b).slice(0,o.k)
  .forEach(({b})=>{for(let j=b*o.m;j<b*o.m+o.m;j++)rr[j]=true;});return rr;});
const cnt=M=>M.reduce((s,row)=>s+row.filter(Boolean).length,0),pick=row=>row.map((v,j)=>v?j+1:0).filter(Boolean);
const P={full:mask("full"),win:mask("window",{w:4}),sp:mask("sparse",{k:4}),cmp:mask("compressed",{m:4,w:4,k:1})};
for(const [k,M] of Object.entries(P))console.log(k,"cells",cnt(M),"row16",pick(M[15]).join(","));
console.log("indexer row16",JSON.stringify(r(S[15],2)));
console.log("sink none",r(softmax([-1,-0.5,-1,-0.75])),"sink 1",r(softmax([-1,-0.5,-1,-0.75,1])),"sink -2",r(softmax([-1,-0.5,-1,-0.75,-2])));
const K=[[1,-0.5,0,0.5],[0,1.5,0,-0.5],[-0.5,0,1,0.5]],V=[[1,0,-1,0],[0,2,0,1],[-1,0,1,0.5]],q=[0,2,0.5,0];
const st=g=>K.reduce((A,k,t)=>A.map((row,i)=>row.map((x,j)=>g*x+V[t][i]*k[j])),Array.from({length:4},()=>Array(4).fill(0)));
const rd=(A,x)=>A.map(row=>row.reduce((s,a,j)=>s+a*x[j],0));
console.log("S",JSON.stringify(st(1)),"o",rd(st(1),q),"gated",JSON.stringify(st(0.5)),"o",rd(st(0.5),q));
const N=1048576,b=(kv,d)=>2*kv*d*2;
console.log("real",N,128,2048,16*128,N/4+128,1024+128,N/128+128);
console.log("gptoss",18*b(8,64),18*128*b(8,64),18*b(8,64)*131072+18*128*b(8,64),"qwen3.5",15*b(2,256),15*b(2,256)*N);
const v4=(mix,bb)=>1e6*mix.reduce((s,[L,m])=>s+L/m,0)*512*bb;
console.log("v4",v4([[30,4],[30,128]],1),v4([[30,4],[30,128]],2),v4([[45,4],[15,128]],1),v4([[45,4],[15,128]],2));
})'
```
Output on 2026-10-07: `full cells 136 row16 1,…,16` · `win cells 58 row16 13,14,15,16` ·
`sp cells 58 row16 1,5,6,13` · `cmp cells 94 row16 1,2,3,4,13,14,15,16` ·
`indexer row16 [0.85,0.35,0.59,-0.33,0.82,0.94,0.69,-0.99,-0.67,0.51,-0.79,-0.83,0.82,-0.46,0.18,-0.48]` ·
`sink none [0.203,0.334,0.203,0.26] sink 1 [0.081,0.134,0.081,0.104,0.6] sink -2 [0.189,0.311,0.189,0.242,0.069]` ·
`S [[1.5,-0.5,-1,0],[0,3,0,-1],[-1.5,0.5,1,0],[-0.25,1.5,0.5,-0.25]] o [-1.5,6,1.5,3.25]
gated [[0.75,-0.125,-1,-0.375],[0,1.5,0,-0.5],[-0.75,0.125,1,0.375],[-0.25,0.75,0.5,0]] o [-0.75,3,0.75,1.75]` ·
`real 1048576 128 2048 2048 262272 1152 8320` · `gptoss 36864 4718592 4836556800 qwen3.5 30720 32212254720` ·
`v4 3960000000 7920000000 5820000000 11640000000`. (The compressed toy's "cells" counts token cells
covered, 94; its row 16 reads 5 entries.)

## 7. Show me the math
```tex
\text{window } w:\quad \htmlClass{hl-read}{\text{reads}_t} = \min(t, w),\qquad \htmlClass{hl-store}{\text{stored}} = \min(t, w)
\qquad
\text{sparse top-}k:\quad \text{reads}_t = \min(t, k),\quad \text{stored} = t
```
```tex
\text{compressed (merge } m\text{)}:\quad \text{stored} = \tfrac{t}{m} + w,\qquad
\text{reads}_t = \min\!\big(\tfrac{t}{m}, k\big) + w
```
```tex
\text{sink:}\quad a_j = \frac{e^{s_j}}{e^{\htmlClass{hl-sink}{s_{\varnothing}}} + \sum_{i} e^{s_i}},
\qquad \text{worked: } \frac{e^{1}}{e^{1} + e^{-1} + e^{-0.5} + e^{-1} + e^{-0.75}} = 0.600
```
```tex
\text{linear (gated): } \htmlClass{hl-state}{S_t} = \alpha\, S_{t-1} + v_t k_t^{\top},\qquad o_t = S_t\, q_t
\qquad
\text{delta rule (Gated DeltaNet): } S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^{\top}\big) + \beta_t\, v_t k_t^{\top}
```
```tex
\text{cache per token (stack)} = \sum_{\text{full layers}} b_{\ell}
\;+\; \frac{1}{\text{tokens}}\Big(\sum_{\text{window}} w\, b_{\ell} + \sum_{\text{linear}} \text{state}_{\ell}\Big),
\qquad \text{gpt-oss: } 18 \cdot 2{,}048 = 36{,}864\ \text{B} + 4.7\ \text{MB fixed}
```
Shapes: S [d_v × d_k] (4 × 4); q, k, v [d_head] (4). The toy's linear layer has no normalization and no
softmax, so its output is the raw-score-weighted sum Σ (q·kᵢ) vᵢ = −1·v_The + 3·v_cat + 0.5·v_sat; real GDN
and KDA layers add the delta rule above (erase what a key already holds before writing), with KDA's gate per
channel instead of one α (01 §2). Color links: `hl-read` → the read cells; `hl-store` → the `kvStack`;
`hl-sink` → the sink cell (frame 4); `hl-state` → S (frame 8). KaTeX with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Every 2026 model that reaches about 1M tokens combines staged length
training (`midtraining`) with attention that is cheap at length; the softmax route also raises its RoPE base
(`rope`), while Kimi K3's full layers use no position encoding at all. Which cheap attention is where the
labs split. All of them still run on FlashAttention-style kernels, which compute
exact attention tile by tile on-chip; they make any pattern fast but do not change what is read or stored
(01 §2)."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| gpt-oss-120b (2025): alternating full and window-128 layers with a learned sink per head; 36,864 B per token plus about 4.7 MB fixed | `models.gpt-oss-120b.attention` (existing, confirmed); `.window` = 128, `.layer_pattern` = "1 full : 1 window" (proposed); `.kv_bytes_per_token` = 36,864, `.kv_fixed_bytes` (proposed in `kv-cache` §12) | 01 §1, §2 [C]; §4 (derived) |
| Gemma 3: 5 window layers to 1 full, window 1,024 (shown only if the data pass adds a primary source; otherwise the row is dropped and MiMo-V2-Flash carries the 5 : 1 example) | `models.gemma-3.layer_pattern` = "5 window : 1 full", `.window` = 1024 (proposed *new* entry; background in 01 §2, source the Gemma 3 report, needs a URL) | 01 §2 (untagged background) |
| MiMo-V2-Flash: 5 window : 1 full, window 128, with a learned sink bias | `models.mimo-v2-flash.layer_pattern`, `.window` = 128 (proposed *new* entry; arXiv 2601.02780) | 01 §2 [C] |
| A learned sink logit per head: gpt-oss, MiMo-V2, DeepSeek-V4 | `models.*.attention_sink` = true (proposed, `attention` ruling 2) | 01 §1 "Attention sink" row [C] |
| DeepSeek V3.2's DSA (Dec 2025) and GLM-5.3: an indexer scores every past token; attention reads the top 2,048; the cache still holds every token | `models.glm-5.3.attention` (existing, confirmed: "MLA + DSA (index top-k 2048)"); `models.glm-5.3.sparse_top_k` = 2048 (proposed) | 01 §2 DSA bullet [C] |
| MiniMax-M3 (2026): GQA with 4 KV heads plus block-sparse attention, blocks of 128, top 16 per group, from layer 3; claims 28.4× less attention compute at 1M on a 109B test model; the cache is still stored | `models.minimax-m3.attention` (existing, confirmed); `.sparse_block` = 128, `.sparse_top_blocks` = 16 (proposed); `.msa_compute_claim` = 28.4 (proposed; note: "109B test model, 1M tokens, MiniMax's claim") | 01 §2 MSA bullet [C] |
| DeepSeek-V4-Pro (2026): every 4 tokens merged then top 1,024 (CSA), or every 128 merged (HCA), alternating, each with a 128-token window; at 1M tokens it uses 27% of V3.2's per-token compute and 10% of its KV cache | `models.deepseek-v4-pro.attention` (existing, confirmed); `.v32_flops_ratio_1m` = 0.27, `.v32_kv_ratio_1m` = 0.10 (proposed) | 01 §2 CSA bullet [C]; 04 §7.5 [C] |
| DeepSeek-V4-Pro's cache is about 4–12 KB per token: a formula estimate; sources disagree on the CSA : HCA layer mix (1 : 1 from its config, 3 : 1 in blog summaries), so both are shown | `models.deepseek-v4-pro.kv_bytes_per_token` = [4000, 12000] (existing, **reported**) | 04 §8.2, §10 item 3 (spec §7 conflict, kept open) |
| Qwen3.5 (Feb 2026) and Qwen3.8 (Aug 2026): 3 Gated DeltaNet layers to 1 full; Qwen3.5-397B keeps 15 full layers of 60 with 2 KV heads × 256: 30,720 B per token | `models.qwen3.8.attention`, `.layers` (existing, confirmed); `models.qwen3.5-397b.layers` = 60, `.full_attention_layers` = 15, `.n_kv_heads` = 2, `.head_dim` = 256, `.kv_bytes_per_token` = 30,720 (proposed *new* entry, also asked for by `rope`) | 01 §2 hybrid bullet [C]; §4 table (derived) |
| Kimi K3 (2026): 69 Kimi Delta Attention layers + 24 MLA layers; Kimi Linear (Oct 2025, a 48B / 3B test model) reported up to 75% less KV cache and about 6× decode throughput at 1M (as reported; the unit is not stated in the brief) | `models.kimi-k3.attention`, `.layers` (existing, confirmed); `models.kimi-linear.kv_reduction_pct` = 75, `.decode_speedup_1m` = 6 (proposed *new* entry; arXiv 2510.26692; note names the test model) | 01 §2 hybrid bullet [C] |
| The split: DeepSeek, GLM and MiniMax chose sparse or compressed softmax attention; Qwen and Moonshot chose linear hybrids | derived from the `attention` strings above (no new key) | 01 §2 "Adoption verdict" |
| The cost of retrofitting: adding windows to an already trained full-attention model hurt long-context tasks badly, even with half the layers kept full (GLM-5 report) | `models.glm-5.swa_retrofit_note` (proposed; same *new* `glm-5` entry as `kv-compression` §12; arXiv 2602.15763, confirmed) | 01 §2 sliding-window caveat [C] |

Not shown: MiniMax's earlier move away from linear attention (01 §2 [U]); Nemotron 3's Mamba hybrid
(reported only, 01 §2 [R]; a candidate row once the data pass has a primary source).

## 9. Takeaways
1. Full attention costs grow two ways with length: reads per token and entries stored. Windows cap both;
   sparse attention cuts only the reads; merging tokens cuts both (frames 1–7, try-this 1).
2. A learned sink logit lets a row put its weight on "nothing", which keeps window layers stable; local and
   global layers are interleaved so the full layers still reach back (frames 3–4, try-this 2).
3. Linear attention keeps a fixed state that blends and fades; hybrids keep one full layer in four for exact
   recall. The 2026 field is split between sparse or compressed softmax attention and linear hybrids, both
   on top of staged length training (and, on the softmax route, a large RoPE base) (frames 8–10, try-this 3).

## 10. Next and go deeper
Next: `model-card` (lists `long-context-attention` as a prerequisite in `shared/concepts.json`). In-page
links: `midtraining` (staged length training), `paged-attention` (paging what remains), `decoder-recap`
(where this page's sinks and hybrid layers appear in the block).

Go deeper (brief 05 §1.1; 01 §2): Sebastian Raschka, LLM Architecture Gallery
(https://sebastianraschka.com/llm-architecture-gallery/), with sliding-window, sparse and hybrid layers per
model · Xiao et al., *Efficient Streaming Language Models with Attention Sinks*
(https://arxiv.org/abs/2309.17453) · Yang et al., *Gated Delta Networks* (https://arxiv.org/abs/2412.06464).

## 11. Key-frame sketch
Frame 6 (sparse reads, full storage), desktop width. `■` read cell, `░` not read (hatched), `·` above the
diagonal (masked). Picks and counts from the §6 reproducer (`sp … row16 1,5,6,13`).
```text
┌──────────────────────────────────────────────────────────┐
│ keys → 1 . . . 5 . . . . . . . 13. . 16    stored in     │
│  1   ■ · · · · · · · · · · · · · · ·       this layer    │
│  …   (rows 2–15: top 4 each)               ▮▮▮▮▮▮▮▮      │
│ ►16  ■ ░ ░ ░ ■ ■ ░ ░ ░ ░ ░ ░ ■ ░ ░ ░       ▮▮▮▮▮▮▮▮      │
│                                            (16 tiles,    │
│  indexer picks for token 16: 1, 5, 6, 13    12 dim)      │
│                                                          │
│  entries read: 4        entries stored: 16               │
│  at 1M tokens: read 2,048 · stored 1,048,576             │
├──────────────────────────────────────────────────────────┤
│ Reading fewer keys does not mean storing fewer. Every    │
│ token stays in the cache, because the indexer must score │
│ all of them each step.                                   │
│ [◄] [Pause] [►]  ━━━━━━●━━━━━  6 / 10  speed [1×]        │
└──────────────────────────────────────────────────────────┘
```
"►" is the selection outline on row 16, the followed (newest) token. The grid is 288 px at 18 px cells and
prints no numbers; every number is in the counters. At 400 px the stack moves below the grid.

## 12. Open questions for the reviewer
**Data-pass keys:**
1. *New:* `window`, `layer_pattern` (gpt-oss-120b, mimo-v2-flash), `sparse_top_k` (glm-5.3 = 2048),
   `sparse_block`, `sparse_top_blocks`, `msa_compute_claim` (minimax-m3), `v32_flops_ratio_1m`,
   `v32_kv_ratio_1m` (deepseek-v4-pro), `full_attention_layers` (qwen3.5-397b). *New entries:*
   `qwen3.5-397b` (shared with `rope`), `kimi-linear` (test-model claims, note required), `mimo-v2-flash`,
   `gemma-3` (01 gives Gemma 3 as untagged background with no URL; drop the row if no primary source is
   added), `glm-5` (shared with `kv-compression`).
2. Reuse `attention` ruling 2's `attention_sink` boolean for gpt-oss-120b, mimo-v2-flash, deepseek-v4-pro.

**Graph changes:** none. `midtraining` and `paged-attention` are in-page links.

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (2 Must). Status is now "approved (expert review)". No worked number changed;
the §6 reproducer was re-run and matches.

Rulings applied (lesson 20): sink mechanism owned here, `decoder-recap` names and links; the un-normalized
linear toy accepted with the visible scale note; MSA read count printed as "about 2,048 (the newest block may
be extra)", with the MSA paper (arXiv 2606.13392) as a data-pass check; the V4-Pro estimate reproduces brief
04's arithmetic as written (30 + 30 layers, N = 1,000,000) with the reported range as the fact.

Must (2/2): frame 10 caption, §8 framing and takeaway 3 no longer say every route relies on a large RoPE base
(Kimi K3's full layers use NoPE; README lesson 29); frame 8 carries the visible "compare the shape, not the
size" line (lessons 12, 21).
Should (7/7, no rebuttals): frame 3 names the ratios and the GLM-5 retrofit cost chip; frame 7 names CSA and
HCA; the MSA uncertainty is printed in the frame 10 table; Gemma 3 is conditional on a source, with
MiMo-V2-Flash (5 : 1, window 128, [C]) as the fallback example in misconception 2 and frames 2–3; frame 1
says "per query head"; Kimi Linear's 6× carries "(as reported; the unit is not stated in the brief)"; frame 1's
counters carry the "reads" / "stored" legend.
Nice (2/2): frame 5 prints the indexer's cost; a "Did the cache shrink?" prompt opens frame 6.
