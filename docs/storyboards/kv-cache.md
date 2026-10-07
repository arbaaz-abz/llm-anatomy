# The KV cache (`kv-cache`)

Track: architecture · Section: architecture · Prereqs: attention
Next: `kv-compression`, `serving-overview` (the two slugs whose `prereqs` list `kv-cache` in `shared/concepts.json`)
Status: draft
Sources: 01 §1, §2 (formula block), §4 · 04 §1.1, §1.4, §3.1, §7.5 · 05 §1.1, §2, §5 (item 15). Nothing beyond the briefs.

This page owns `math/memory.js`. `paged-attention` imports `kvBytesPerToken` and `kvBytesPerTokenMla` from it,
`decoder-anatomy` quotes DeepSeek-V3's 70,272 B, and `kv-compression`, `long-context-attention` and
`model-card` read their byte counts from it. It reuses `attention`'s four tokens, its head A K and V rows and
its glyph layout (rows = positions, K and V as `matrix` glyphs), and picks up exactly where
`decoder-anatomy` frame 9 stops: "on" has just been chosen.

## 1. Learning objective
After this page you can explain why a model stores every earlier token's keys and values instead of
recomputing them (frames 1–3), say what prefill and decode are (frame 4), compute the cache's size per
token with 2 × layers × KV heads × head size × bytes and multiply it by context and conversations
(frames 6–8, toy B), and say why that number, and not the arithmetic, became the limit on long context
(frames 8–9, try-this 2–3).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "To write each new word, the model reruns the whole conversation." → **Reality:**
  with a causal mask, earlier positions' keys and values never change, so each decode step runs only the
  new token through the blocks and *reads* the stored K and V of the rest. Over a 1,000-token prompt and a
  1,000-token reply that is 1,999 positions computed instead of 1,499,500. (01 §4 mechanism paragraph) ·
  corrected by frames 2–3 and 5, try-this 1
- **Misconception:** "With a KV cache, token 10,000 costs the same as token 10." → **Reality:** the
  multiplications by the weights are the same, but the new token's query still reads every stored key and
  value: 10,000 per head per layer at token 10,000 against 10 at token 10. In Llama-3.1-70B that read is
  3.28 GB of cache per step at 10,000 tokens against 3.28 MB at 10. The cache removes recomputation, not
  the reading. (01 §4: "one token of compute plus a read of the whole cache"; 04 §1.1 attention row) ·
  corrected by frame 3's "read" counter, frame 5 and try-this 1's second output
- **Misconception:** "GPU memory is mostly the model's weights." → **Reality:** each conversation has
  its own cache, and at long context it is the larger item: one 128K-token Llama-3.1-70B conversation
  holds 42.9 GB, 54% of an 80 GB H100, so two do not fit on one GPU. The cache caps how many
  conversations a GPU can serve at once. (04 §1.4 KV term, §7.5 item 1) · corrected by frame 8 and
  try-this 2
- **Misconception:** "The formula counts attention heads." → **Reality:** it counts *KV* heads, the
  number of distinct key/value sets a layer stores. In GPT-3 every head had its own (96); Llama-3.1-70B
  stores 8 per layer and shares them between query heads, which is most of why its per-token cache is
  14.4× smaller. (04 §3.1 "Careful: the heads term is KV heads"; 01 §2 GQA paragraph) · corrected by
  try-this 3; how the sharing works is deferred to `kv-compression`

## 3. Hook and intuition (final wording)
**Hook:** Every new token has to look at every earlier token's keys and values. Why does a model store
them instead of recomputing them, and why did that storage, not the arithmetic, become the limit on long
context?

When a model writes a reply, it produces one token per pass. To pick the token after "on", the new token
"on" has to go through every block, and in each attention layer its query is compared with the keys of all
five positions and blends their values. The keys and values of "The", "cat", "sat" and "down" were already
computed on the previous pass, and they come out the same every time: the causal mask means a position
only ever looks backwards, so nothing that comes later can change them.

So the model keeps them. Each attention layer stores a K row and a V row per position, and every decode
step adds one more of each and reads all of them. That stored table is the **KV cache**. It turns "rerun
the whole conversation for every word" into "run one token, then read the past", which is why chat
replies stream at all.

The price is memory, and it grows with everything: one K and one V row per position, per key/value head,
per layer, per conversation. In GPT-3 (2020) that came to 4.5 MiB for every single token. Each decode
step also has to *read* the whole cache back from GPU memory, so a long conversation is slow as well as
big. That is why most attention redesigns since 2023 attack this one number: by 2026 the largest open
models store a few KB per token, roughly 400 to 1,200 times less than GPT-3 (`kv-compression`,
`long-context-attention`).

## 4. Visual metaphor
Toy dimensions (shared with `attention` and `decoder-anatomy`): the prompt "The cat sat down" (4 tokens),
then "on₅" chosen by `decoder-anatomy` frame 9; head A of `attention` (d_head = 4); 2 heads per layer,
each with its own K and V; N = 2 blocks.

**Numbers.** K and V rows 1–4 are `attention`'s `TOY.heads.A.K` and `.V`, unchanged. The new row for
"on" is a hand-picked stand-in on the same quarter grid: `k_on = [0.5, 0, 0.5, 0]`,
`v_on = [0, 0.5, 0.5, 0]`. A visible line under the stage says so (README lesson 10): "K and V rows 1–4
are the `attention` page's numbers; the row for 'on' is hand-picked. Every count and byte size is
exact." The counts and byte sizes come from `math/memory.js` (§6).

**Units** (visible line under toy B, and in the frame 6 caption's "Numbers shown"): bytes per token are
printed exactly and in powers of 1,024 (KiB, MiB), as `decoder-anatomy` and brief 01 do; totals are in
powers of 1,000 (MB, GB), as GPU memory is sold. So 4,718,592 B = 4.5 MiB per token, and 9.66 GB per
2,048-token conversation. (Open question 1: `paged-attention` uses decimal for both.)

**Terms introduced, one per frame** (README lesson 3): 1 decode step · 2 recompute · 3 cache read (the
KV cache itself was named in `decoder-anatomy` frame 9; here it is opened) · 4 prefill · 5 none (a count
over a whole reply) · 6 bytes per token · 7 context length · 8 conversations served at once ("batch",
used by `batching`) · 9 none (real models, the 2026 drop). **Terms assumed from prereqs:** token, query /
key / value, causal mask, head, d_head, block / layer (`attention`, `decoder-anatomy`). **Named and
deferred:** why decode is memory-bound (`prefill-decode`), sharing K/V between heads and latent KV
(`kv-compression`), windows and compressed caches (`long-context-attention`), blocks and paging
(`paged-attention`), reuse across requests (`prefix-caching`).

**Indexing** (README lesson 11): token positions are 1-based on screen (The₁ … on₅). This page shows no
memory addresses.

**Layout** (stage ≈ 580 × 366; at 400 px it scrolls inside its container):
- Top: the token chips The₁ cat₂ sat₃ down₄, and in frames 1–5 "on₅".
- Left: `matrix` K and `matrix` V [5 × 4] at 20 px cells, rows labeled by token (values on hover only;
  nothing in them is load-bearing except row 5, which is lifted out).
- Center: row 5 lifted out as two row `vector`s, `k_on` and `v_on`, at `NUMBER_CELL` (4 × 43 = 172 px
  each), selection outline on both (the followed item, same mark in frames 1–5).
- Right: two plain text counters, "positions computed this step" and "keys read this step (per head,
  per layer)". In frame 5, a two-row plain text table.
- Frames 6–8: the toy model as `blockStack` (`shown = 2`, halves "attention" / "MLP") with one `kvStack`
  beside each block's attention half (5 tiles, the newest `highlight`ed). Frames 8–9: `gpu` glyphs whose
  memory fill is the cache's share of 80 GB, with the share printed under each.

Glyphs used (from spec §5.1 and the accepted `decoder-anatomy` proposals): token, matrix, vector, block,
flow (carry `kv` for cache reads and writes, `activation` for the forward pass), kvStack, gpu,
blockStack.
New glyphs proposed: none. Small marks (README lesson 15): the two counters, the frame 5 table, the
"= same as stored" check marks in frame 2, the units line and every share under a `gpu` are plain text
labels.

Color: K and V cells use the diverging value scale (maxAbs 2, as `attention`'s Q/K/V use 3; here only row
5 prints numbers, so maxAbs 2 keeps 0.5 visible). Cache tiles use `--sem-memory` through `kvStack`;
recomputed work in frame 2 uses the `activation` flow dot, cache reads the `kv` dot. The followed row
("on", row 5) carries the selection outline in K, V and the lifted vectors in every frame it appears.
Hovering a term in the math panel outlines the matching glyph with the Architecture accent.

## 5. Animation script
Hero: token 5, "on". Counters are per head and per layer unless the caption says otherwise.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | The four prompt chips; K and V [4 × 4] (head A) filled; a fifth chip "on₅" slides in from the right, outlined. Empty fifth rows in K and V. Counters blank. | "on₅" slides in; the empty fifth rows draw as dashed outlines. | The model just chose "on". One decode step runs it through every block, and in each attention layer it must look at all five positions. | 4 → 5 positions · K, V of rows 1–4 = `attention` head A · 2 blocks × 2 heads |
| 2 | **Without a cache.** All five chips send an `activation` flow into the W block; rows 1–4 of K and V refill from scratch; beside each refilled row a plain "= same as before" mark. Counter "positions computed: 5". | Rows 1–4 blank and refill top to bottom; the check marks appear one by one; row 5 fills last. | Without a cache, every position is computed again for each new token. Rows 1 to 4 come out exactly as before, because a position never looks ahead. | positions computed this step: 5 · rows 1–4 identical to the stored ones · `k_on = [0.5, 0, 0.5, 0]` · `v_on = [0, 0.5, 0.5, 0]` |
| 3 | **With a cache.** Rows 1–4 of K and V stay put, now drawn as stored (`kvStack` tint); only "on₅" sends a flow; `k_on`, `v_on` lift out at `NUMBER_CELL` and append as row 5; then a `kv` flow fans from row 5's query position to all five K rows. Counters "positions computed: 1" and "keys read: 5". | Row 5 is computed and slides into place; then five `kv` dots run from the stored rows to the new token. | With a cache, only the new token is computed; the past's stored keys and values are read back. Work shrinks, but the read still covers every position. | positions computed: 1 · keys read: 5 (and 5 values) · `k_on = [0.5, 0, 0.5, 0]` · `v_on = [0, 0.5, 0.5, 0]` |
| 4 | Rewind label "before 'on'": the four prompt chips enter together; K and V rows 1–4 fill in one pass; a plain label "prefill: 4 positions in one pass"; then "on₅" arrives and a label "decode: 1 position per step". | Rows 1–4 fill simultaneously; then row 5 fills alone. | The prompt goes in all at once and fills the cache in one pass: that is prefill. After it, each decode step adds one row. | prefill: 4 positions, 10 key reads (1 + 2 + 3 + 4) · decode step for "on": 1 position, 5 key reads |
| 5 | Plain text table under the matrices, two columns "no cache" and "cache", for a reply of 4 tokens ("on", "the", "mat", "."); K and V grow to 8 rows (20 px) as the four steps play. | The four steps tick by; both columns count up; the cache column's "positions" grows by 1 per step, the no-cache one by 5, 6, 7. | Over a four-token reply, the cache computes 7 positions instead of 22. The stored rows are still read every step, so reads grow with the length. | 4-token prompt, 4-token reply · positions computed 22 vs 7 · key reads 74 vs 28 · (a 1,000-token prompt and reply: 1,499,500 vs 1,999 positions) |
| 6 | The toy model as `blockStack` (2 blocks); beside each attention half a `kvStack` of 5 tiles; one tile lifts and expands to 2 heads × (a K row of 4 + a V row of 4). A mono line "K and V × 2 heads × 4 numbers × 2 blocks". | One tile per block highlights; the expanded tile shows 16 small cells; the two blocks' tiles stack into a "32 numbers" readout, then "× 2 bytes = 64 bytes". | Each position stores one key row and one value row per head in every attention layer. In this toy that is 32 numbers, 64 bytes per token. | 2 × 2 heads × 4 × 2 blocks = 32 numbers · at 2 bytes per number: 64 B per token · 5 tokens: 320 B |
| 7 | The same picture relabeled for GPT-3 (2020): "96 blocks · 96 heads · 128 numbers per head", the `blockStack` count printed as "× 96"; a plain readout. A `kvStack` stretches to "2,048 positions" (tiles collapse to "⋯ 2,048"). | The labels morph from toy to GPT-3 values; the byte readout counts up; the stack extends with a "× 2,048" label. | GPT-3 stored 4.5 MiB for every token, because all 96 heads in all 96 layers kept their own keys and values. Its whole 2,048-token context needed 9.66 GB. | 2 × 96 × 96 × 128 × 2 B = 4,718,592 B = 4.5 MiB per token · × 2,048 tokens = 9,663,676,416 B = 9.66 GB (GPT-3 paper, 2020, at 2 bytes per number) |
| 8 | Two `gpu` glyphs labeled "H100, 80 GB". One conversation of 131,072 tokens for Llama-3.1-70B (2024) fills the first to 54%; a second conversation's 42.9 GB arrives and spills past the first GPU's top into a plain label "does not fit". | The first fill rises to 54%; the second conversation's bar rises, overflows, and the overflow label appears. | Every conversation has its own cache. One 128K-token Llama-3.1-70B conversation needs 42.9 GB, so a single 80 GB GPU cannot hold two. | 327,680 B per token (80 layers × 8 KV heads × 128 × 2 × 2 B) · × 131,072 tokens = 42,949,672,960 B = 42.9 GB = 53.7% of 80 GB · two: 85.9 GB |
| 9 | **Key frame.** A plain text column of per-token sizes, each with its year: GPT-3 (2020) 4.5 MiB · Llama-3.1-70B (2024) 320 KiB · DeepSeek-V3 (2024) 68.6 KiB · DeepSeek-V4-Pro (2026) about 4–12 KB, "reported estimate". Beside each 2024–2026 row a `gpu` glyph (80 GB) filled by one 1M-token (2²⁰) conversation. | The rows type in top to bottom; each `gpu` fills; the V4-Pro row shows a range (two fills, light and full). | By 2026 the biggest open models keep a few KB per token, hundreds of times less than GPT-3. That is what makes a million-token context affordable. | per token: 4,718,592 B · 327,680 B · 70,272 B · 4,000–12,000 B (reported) · at 1,048,576 tokens: 344 GB (4.3 H100s) · 73.7 GB (92%) · 4.2–12.6 GB (5–16%) · GPT-3 ÷ V4-Pro: 393–1,180× |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. K and V keep their position from frame 1 to frame 5; the lifted `k_on` / `v_on` strip sits in the
same place in frames 2 and 3; "on₅" keeps its selection outline in frames 1–5.

Caption word counts (README lesson 2, ≤ 30 words and ≤ 2 sentences, no operators; split on spaces):
25 · 27 · 27 · 25 · 26 · 27 · 28 · 22 · 26.

Absolutes checked (README lesson 7): "a position never looks ahead" is the causal mask (01 §4: "K and V of
earlier tokens never change (causal mask)"). "Every conversation has its own cache" is true for the
cache's growing part; sharing a common prefix across conversations is `prefix-caching`'s topic, named in
§10. Frame 6's "every attention layer" is qualified on screen in frame 9 by a visible note: "Some 2026
layers keep a fixed-size state or a window instead of one row per token: `long-context-attention`."

Nothing load-bearing is behind hover (README lesson 5): the counters, the frame 5 table, the "same as
before" marks and every byte readout and GPU share are printed. Hover on a 20 px K or V cell only shows a
value that `attention` already printed.

## 6. Toy
Title on page: "Count the work, then weigh the memory." Two panels share one state object and one
`render()`; each has a "Check my work" box (`aria-live="polite"`).

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `cache` | KV cache | toggle | on / off | on | — |
| `prompt` | Prompt length | Slider (snapped) | [4, 16, 128, 1,000, 8,192, 100,000] tokens | 4 | — |
| `reply` | Reply length | Slider (snapped) | [1, 2, 4, 16, 128, 1,000] tokens | 4 | — |
| `model` | Model | preset chips | toy · GPT-3 · Llama-3.1-70B · DeepSeek-V3 · DeepSeek-V4-Pro (reported) | toy | real chips load layers, KV heads, head size from `data/models.json` (§8). DeepSeek-V3 switches the formula to the latent one and shows "how: `kv-compression`". DeepSeek-V4-Pro shows the reported 4,000–12,000 B range and disables the shape sliders, with the note "formula-derived estimate; the layer mix is uncertain". |
| `context` | Tokens in the cache | Slider (snapped) | [2,048 · 8,192 · 32,768 · 131,072 · 262,144 · 1,048,576] | 2,048 | the model's own context length is marked on the track |
| `sequences` | Conversations at once | Slider (snapped) | [1, 2, 4, 8, 16, 32, 64] | 1 | — |

Panel B also shows the selected model's shape as read-only readouts (layers, KV heads, head size, bytes
per number). They become editable sliders only on the `toy` chip (layers 1–96, KV heads 1–96, head size
[4, 64, 128], bytes per number [1, 2]), so a learner can build GPT-3 from the toy by hand.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Panel A: positions computed, whole reply | `decodeWork({ prompt, generated: reply, cache }).positions` | count, `formatCount` past 10,000 |
| Panel A: keys read per head per layer, whole reply | `.keyReads` | count |
| Panel A: the same two numbers with the cache flipped, and their ratio | `decodeWork({ …, cache: !cache })` | "× fewer" with 3 significant figures |
| Panel B: bytes per token | `kvBytesPerToken({ layers, kvHeads, headDim, bytesPerElem })`; MLA presets `kvBytesPerTokenMla(…)` | exact bytes and `formatBytes(…, { binary: true })` |
| Panel B: cache per conversation | `kvCacheBytes({ bytesPerToken, tokens: context })` | `formatBytes` (decimal) |
| Panel B: cache for all conversations | `kvCacheBytes({ bytesPerToken, tokens: context, sequences })` | decimal GB |
| Panel B: share of one GPU | `sharePct(total, hbm_gb × 1e9)` for H100 (80 GB) and B300 (288 GB); fill of two `gpu` glyphs | % (one decimal); "does not fit" above 100% |
| Panel B: cache read per decode step at this context | `kvCacheBytes({ bytesPerToken, tokens: context })` (one step reads the whole cache of one conversation) | decimal MB / GB |

"Check my work" for the default state (this exact text appears on the page):
```text
no cache: 4 + 5 + 6 + 7                  = 22 positions
cache:    4 (prefill) + 1 + 1 + 1        =  7 positions
bytes per token = 2 (K and V) × 2 blocks × 2 KV heads × 4 numbers × 2 bytes
                = 64 B
× 2,048 tokens                           = 131,072 B = 131 KB
```

**Try this** (each leads to a named insight)
1. Panel A, cache **off**, prompt 4, reply 4: 22 positions. Cache **on**: 7. Now prompt 1,000 and reply
   1,000: 1,499,500 against 1,999, 750× fewer. Read the second output: keys read are 1,166,666,500
   against 1,999,000, still almost two million. → **Insight: the cache removes recomputation, not
   reading.** Work per step stays flat; the read per step grows with every token in the conversation.
2. Panel B, **Llama-3.1-70B**, tokens 131,072, conversations 1: 42.9 GB, 53.7% of an H100. Set
   conversations to **2**: 85.9 GB, "does not fit". Now **GPT-3** at its own 2,048 tokens with 8
   conversations: 77.3 GB, 96.6%. → **Insight: the cache, not the arithmetic, sets how many conversations
   one GPU can serve,** and it grows with context × conversations.
3. Panel B, tokens 131,072: tap **GPT-3 → Llama-3.1-70B → DeepSeek-V3**: 4.5 MiB → 320 KiB → 68.6 KiB
   per token (618 GB → 42.9 GB → 9.21 GB per conversation). Then tap the toy chip and set KV heads from 96
   to 8 with GPT-3's other numbers: 4.5 MiB → 384 KiB. → **Insight: the formula multiplies KV heads,
   not query heads, so storing fewer key/value sets is the biggest lever.** How models do that without
   losing quality is `kv-compression`.

**`math/memory.js`** (owned by this page; pure, no DOM, inputs never mutated; every function throws
`RangeError` with the argument's name if a size is not a positive finite number or a count is not a
positive integer; tests first). `paged-attention` imports the first two by these exact names.

```js
// Bytes stored per token by every attention layer that keeps K and V per KV head.
// MHA, GQA and MQA are the same formula with kvHeads = heads, groups, or 1.
kvBytesPerToken({ layers, kvHeads, headDim, bytesPerElem }) → number      // 2 · L · n_kv · d_head · b
//   ({ layers: 2,  kvHeads: 2,  headDim: 4,   bytesPerElem: 2 }) → 64              (toy; exact)
//   ({ layers: 96, kvHeads: 96, headDim: 128, bytesPerElem: 2 }) → 4_718_592       (GPT-3, 4.5 MiB)
//   ({ layers: 80, kvHeads: 8,  headDim: 128, bytesPerElem: 2 }) → 327_680         (Llama-3.1-70B, 320 KiB)
//   ({ layers: 96, kvHeads: 8,  headDim: 128, bytesPerElem: 2 }) → 393_216         (GPT-3 with 8 KV heads; try-this 3)
//   ({ layers: 0, … })                                           → throws RangeError('kvBytesPerToken: layers must be a positive integer')

// Bytes per token for multi-head latent attention: one latent vector plus one small RoPE key per layer.
// No factor of 2: K and V are both rebuilt from the same stored latent (kv-compression).
kvBytesPerTokenMla({ layers, dLatent, dRope, bytesPerElem }) → number     // L · (d_c + d_rope) · b
//   ({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 }) → 70_272            (DeepSeek-V3, 68.6 KiB; matches decoder-anatomy §8 and paged-attention §6)

// Total cache for `sequences` conversations of `tokens` each.
kvCacheBytes({ bytesPerToken, tokens, sequences = 1 }) → number
//   ({ bytesPerToken: 64,        tokens: 2_048 })                → 131_072
//   ({ bytesPerToken: 4_718_592, tokens: 2_048 })                → 9_663_676_416     (9.66 GB)
//   ({ bytesPerToken: 4_718_592, tokens: 2_048, sequences: 8 })  → 77_309_411_328    (77.3 GB, 96.6% of 80 GB)
//   ({ bytesPerToken: 327_680,   tokens: 131_072 })              → 42_949_672_960    (42.9 GB; paged-attention's reserveMaxBytes agrees)
//   ({ bytesPerToken: 327_680,   tokens: 1_048_576 })            → 343_597_383_680   (344 GB)
//   ({ bytesPerToken: 70_272,    tokens: 131_072 })              → 9_210_691_584     (9.21 GB)
//   ({ bytesPerToken: 70_272,    tokens: 1_048_576 })            → 73_685_532_672    (73.7 GB)
//   ({ bytesPerToken: 327_680,   tokens: 10_000 })               → 3_276_800_000     (one decode step's read at token 10,000; misconception 2)

// Work to generate `generated` tokens after a `prompt`-token prompt. The first pass is the prefill.
// positions: positions run through the blocks; keyReads: query–key scores computed per head per layer.
// cache = false recomputes every position on every pass (a pass over t positions costs t positions and t(t+1)/2 reads);
// cache = true runs the prompt once, then one position per pass (a pass at length t costs 1 position and t reads).
decodeWork({ prompt, generated, cache }) → { positions, keyReads }
//   ({ prompt: 4, generated: 1, cache: true })      → { positions: 4, keyReads: 10 }           (prefill only)
//   ({ prompt: 4, generated: 2, cache: true })      → { positions: 5, keyReads: 15 }           (frame 3: + 1 position, + 5 reads)
//   ({ prompt: 4, generated: 4, cache: false })     → { positions: 22, keyReads: 74 }
//   ({ prompt: 4, generated: 4, cache: true })      → { positions: 7, keyReads: 28 }
//   ({ prompt: 1000, generated: 1000, cache: false }) → { positions: 1_499_500, keyReads: 1_166_666_500 }
//   ({ prompt: 1000, generated: 1000, cache: true })  → { positions: 1_999, keyReads: 1_999_000 }
```

Share of a GPU uses `sharePct(part, whole)`, the one definition `paged-attention` already specified
(Open question 2 proposes moving it into this module so both pages import it from here). The functions
`long-context-attention` adds to this module (`stackKvBytes`, `linearStateBytes`) are specified on that
page, under different names.

Tests to write first: the worked examples above to the byte; `kvBytesPerToken` with `kvHeads = heads`
equals 2 · layers · heads · headDim · b for GPT-3 (MHA) and with `kvHeads = 1` equals the MQA value;
`kvBytesPerTokenMla` has no factor of 2 (DeepSeek-V3 = 61 · 576 · 2); every function throws on 0,
negative, NaN and non-integer counts; `decodeWork` with `generated = 1` is identical for cache on and
off (the prefill); `decodeWork(cache: true).positions === prompt + generated − 1`;
`kvCacheBytes` is linear in `tokens` and `sequences`; inputs are not mutated (frozen argument objects).

**Reproducer** (README lesson 6; run from the repo root on 2026-10-07 with the formulas above inlined,
since `math/memory.js` does not exist yet; output matched every number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({formatBytes})=>{
const kv=({layers,kvHeads,headDim,bytesPerElem})=>2*layers*kvHeads*headDim*bytesPerElem;
const mla=({layers,dLatent,dRope,bytesPerElem})=>layers*(dLatent+dRope)*bytesPerElem;
const cache=({bytesPerToken,tokens,sequences=1})=>bytesPerToken*tokens*sequences;
const work=({prompt,generated,cache})=>{let pos=0,reads=0;for(let i=0;i<generated;i++){const t=prompt+i;
 if(!cache||i===0){pos+=t;reads+=t*(t+1)/2;}else{pos+=1;reads+=t;}}return{pos,reads};};
const toy=kv({layers:2,kvHeads:2,headDim:4,bytesPerElem:2}),g3=kv({layers:96,kvHeads:96,headDim:128,bytesPerElem:2}),
 ll=kv({layers:80,kvHeads:8,headDim:128,bytesPerElem:2}),v3=mla({layers:61,dLatent:512,dRope:64,bytesPerElem:2});
console.log("per token",toy,g3,formatBytes(g3,{binary:true}),ll,formatBytes(ll,{binary:true}),v3,formatBytes(v3,{binary:true}),
 kv({layers:96,kvHeads:8,headDim:128,bytesPerElem:2}));
console.log("gpt3 2048",cache({bytesPerToken:g3,tokens:2048}),"x8",cache({bytesPerToken:g3,tokens:2048,sequences:8})/80e9);
for(const t of [131072,1048576])for(const [n,b] of [["gpt3",g3],["llama",ll],["v3",v3]])
 console.log(n,t,cache({bytesPerToken:b,tokens:t}),formatBytes(cache({bytesPerToken:b,tokens:t})),(cache({bytesPerToken:b,tokens:t})/80e9).toFixed(3));
console.log("v4 range 1M",4000*1048576,12000*1048576,(4000*1048576/80e9).toFixed(3),(12000*1048576/80e9).toFixed(3),"ratios",g3/12000,g3/4000,g3/v3,g3/ll);
console.log("read/step llama",cache({bytesPerToken:ll,tokens:10}),cache({bytesPerToken:ll,tokens:10000}),"toy 2048",cache({bytesPerToken:toy,tokens:2048}));
for(const [p,g] of [[4,1],[4,2],[4,4],[1000,1000]])console.log(p,g,JSON.stringify(work({prompt:p,generated:g,cache:false})),JSON.stringify(work({prompt:p,generated:g,cache:true})));
})'
```
Output on 2026-10-07: `per token 64 4718592 4.5 MiB 327680 320 KiB 70272 68.6 KiB 393216` ·
`gpt3 2048 9663676416 x8 0.9663676416` · `gpt3 131072 618475290624 618 GB 7.731` ·
`llama 131072 42949672960 42.9 GB 0.537` · `v3 131072 9210691584 9.21 GB 0.115` ·
`gpt3 1048576 4947802324992 4.95 TB 61.848` · `llama 1048576 343597383680 344 GB 4.295` ·
`v3 1048576 73685532672 73.7 GB 0.921` · `v4 range 1M 4194304000 12582912000 0.052 0.157 ratios 393.216
1179.648 67.14754098360656 14.4` · `read/step llama 3276800 3276800000 toy 2048 131072` ·
`4 1 {"pos":4,"reads":10} {"pos":4,"reads":10}` · `4 2 {"pos":9,"reads":25} {"pos":5,"reads":15}` ·
`4 4 {"pos":22,"reads":74} {"pos":7,"reads":28}` ·
`1000 1000 {"pos":1499500,"reads":1166666500} {"pos":1999,"reads":1999000}`.
Two Llama-3.1-70B conversations: 2 × 42,949,672,960 = 85,899,345,920 B = 85.9 GB.

## 7. Show me the math
```tex
\text{decode step } t:\quad
\htmlClass{hl-k}{K_{1:t}} = \big[\,\htmlClass{hl-cache}{K_{1:t-1}} \;;\; x_t W_K\,\big],\qquad
\htmlClass{hl-v}{V_{1:t}} = \big[\,\htmlClass{hl-cache}{V_{1:t-1}} \;;\; x_t W_V\,\big],\qquad
o_t = \operatorname{softmax}\!\Big(\tfrac{q_t \htmlClass{hl-k}{K_{1:t}}^{\top}}{\sqrt{d_{\text{head}}}}\Big)\htmlClass{hl-v}{V_{1:t}}
```
```tex
\text{positions computed: } \underbrace{\textstyle\sum_{t=P}^{P+G-1} t}_{\text{no cache}}
\quad\text{vs}\quad \underbrace{P + (G-1)}_{\text{cache}},
\qquad \text{worked: } 4+5+6+7 = 22 \ \text{vs}\ 4 + 3 = 7
```
```tex
\htmlClass{hl-bytes}{\text{bytes per token}} = 2 \cdot L \cdot n_{kv} \cdot d_{\text{head}} \cdot b,
\qquad \text{worked (GPT-3): } 2 \cdot 96 \cdot 96 \cdot 128 \cdot 2 = 4{,}718{,}592\ \text{B}
```
```tex
\text{cache} = \htmlClass{hl-bytes}{\text{bytes per token}} \times \text{tokens} \times \text{conversations},
\qquad \text{worked: } 327{,}680 \times 131{,}072 = 42.9\ \text{GB}
```
```tex
\text{latent attention (kv-compression): } \text{bytes per token} = L\,(d_c + d_{\text{rope}})\,b,
\qquad 61 \cdot 576 \cdot 2 = 70{,}272\ \text{B}
```
Shapes: K, V per head per layer [t × d_head] (5 × 4 here); per position the cache holds
2 · n_kv · d_head numbers per layer (2 · 2 · 4 = 16 here). P = prompt length, G = reply length, L =
attention layers that keep a growing cache, n_kv = KV heads, b = bytes per number. Color links: `hl-cache`
→ the stored rows (`kvStack` tint) in frames 3–6; `hl-k`, `hl-v` → the K and V matrices; `hl-bytes` → the
byte readouts in frames 6–9. KaTeX with `trust: true, strict: false` for `\htmlClass` (brief 05 §2,
pattern 4). The panel notes that reading the whole cache each step is why decode is limited by memory
bandwidth, with the full argument in `prefill-decode` (01 §4, 04 §1.1).

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Every model in this table keeps a KV cache; what changed between 2020 and
2026 is how many bytes each token costs. The 2026 models get there with fewer key/value sets
(`kv-compression`), a compressed latent (`kv-compression`), and windows, compression or fixed-size states
in some layers (`long-context-attention`)."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GPT-3 (2020): 96 layers × 96 heads × 128, each head with its own K and V: 4,718,592 B (4.5 MiB) per token at 2 bytes; 9.66 GB for its 2,048-token context | `models.gpt-3.kv_bytes_per_token` = 4,718,592, `.layers` = 96, `.n_heads` = 96, `.n_kv_heads` = 96, `.head_dim` = 128, `.context_length` = 2,048 (all proposed; entry and source arXiv 2005.14165 per `attention` ruling 3 and `decoder-anatomy` §12.3) | 01 §4 worked-numbers table |
| Llama-3.1-70B (2024): 80 layers, 8 KV heads × 128: 327,680 B (320 KiB) per token; 42.9 GB at its 131,072-token context, 54% of an 80 GB H100 | `models.llama-3.1-70b.kv_bytes_per_token`, `.layers`, `.n_kv_heads`, `.head_dim`, `.context_length` (entry proposed by `paged-attention` §13; its note names the Llama-3 → 3.1 correction); `hardware.h100.hbm_gb` (existing, reported) | 01 §4 table; 04 §3.1, §1.4 |
| DeepSeek-V3 (Dec 2024): one 512-number latent plus a 64-number position key per layer, 61 layers: 70,272 B (68.6 KiB) per token, derived from its config; 73.7 GB at 1M tokens | `models.deepseek-v3.kv_bytes_per_token` = 70,272 (derived), `.mla_kv_rank`, `.mla_rope_dim`, `.layers` (entry and keys proposed by `paged-attention` §13 / `decoder-anatomy` §12.3) | 01 §2 MLA paragraph, §4 table |
| DeepSeek-V4-Pro (2026): about 4–12 KB per token, a formula-derived estimate (the layer mix of its compressed attention is uncertain); about 4–12 GB for a 1M-token conversation | `models.deepseek-v4-pro.kv_bytes_per_token` = [4000, 12000] (existing, **reported**, note kept) | 01 §4 table; 04 §3.1, §8.2 (spec §7 conflict: shown as a range, both mixes in `long-context-attention`) |
| gpt-oss-120b (2025): only its 18 full-attention layers grow a cache, 36,864 B per token; its 18 sliding-window layers hold a fixed ~4.7 MB per conversation | `models.gpt-oss-120b.kv_bytes_per_token` = 36,864 (proposed, derived), `.kv_fixed_bytes` = 4,718,592 (proposed, derived: 18 × 128 × 2,048 B) | 01 §4 table (derived) |
| Reusing a cache across requests that share a prompt prefix is standard in serving (`prefix-caching`) | — (timeless prose, no number) | 01 §4 mechanism paragraph |

Rendered with `renderFact` (source link, "reported" chip where the data file says so). Not shown: Kimi K3's
per-token cache (its latent dims were not read, 01 §4 [U]), GLM-5.3's (RoPE dim assumed, [U]), DeepSeek-V4.1-Flash's
890 B (a single secondary blog).

## 9. Takeaways
1. Earlier positions' keys and values never change, so a model stores them: each decode step computes one
   new position and reads the stored rest (frames 2–3). The prompt fills the cache in one prefill pass.
2. The cache removes recomputation, not reading: every step still reads every stored key and value, so a
   longer conversation means a bigger read per token (frame 5, try-this 1).
3. Bytes per token = 2 × layers × KV heads × head size × bytes, times tokens, times conversations. At long
   context the cache, not the weights, decides how many conversations fit on a GPU; it fell from 4.5 MiB
   per token in GPT-3 to a few KB in 2026 models (frames 6–9).

## 10. Next and go deeper
Next: `kv-compression` (how MQA, GQA and MLA store fewer bytes per token) and `serving-overview` (one
request's journey through a production stack); both list `kv-cache` as their prerequisite in
`shared/concepts.json`. In-page links that are not graph edges: `prefill-decode` (why reading the cache
makes decode memory-bound), `long-context-attention`, `paged-attention`, `prefix-caching`.

Go deeper (brief 05 §1.1, §1.2; 04 §9.2): Brendan Lynskey, *LLM Inference Explained*
(https://llm-inference-explained.vercel.app), the generation loop and KV cache in long form · Sebastian
Raschka, LLM Architecture Gallery (https://sebastianraschka.com/llm-architecture-gallery/), with a memory
calculator across 100+ models · Hugging Face, *Continuous batching from first principles*
(https://huggingface.co/blog/continuous_batching), which derives serving from attention plus the cache.

## 11. Key-frame sketch
Frame 3 (with a cache), desktop width; the frame 9 table is the toy's closing readout. Numbers from the
§6 reproducer (`4 2` cache line: + 1 position, 5 reads).
```text
┌──────────────────────────────────────────────────────────┐
│ [The]₁ [cat]₂ [sat]₃ [down]₄ ►[on]₅                       │
│                                                          │
│  K [5×4]        V [5×4]       new row (computed)         │
│  The ▪▪▪▪ ◄┐    The ▪▪▪▪      k_on │ 0.5 │  0  │ 0.5 │ 0 │ │
│  cat ▪▪▪▪ ◄┤    cat ▪▪▪▪      v_on │  0  │ 0.5 │ 0.5 │ 0 │ │
│  sat ▪▪▪▪ ◄┤    sat ▪▪▪▪                                 │
│  down▪▪▪▪ ◄┤    down▪▪▪▪      positions computed: 1      │
│ ►on  ▪▪▪▪ ◄┘   ►on  ▪▪▪▪      keys read: 5               │
│   (rows 1–4 stored = read)                               │
├──────────────────────────────────────────────────────────┤
│ With a cache, only the new token is computed; the past's │
│ stored keys and values are read back. Work shrinks, but  │
│ the read still covers every position.                    │
│ [◄] [Pause] [►]  ━━━━●━━━━━━━  3 / 9   speed [1×]        │
└──────────────────────────────────────────────────────────┘
 K and V rows 1–4 are the attention page's numbers; the row
 for "on" is hand-picked. Every count and byte size is exact.
```
"►" is the selection outline on the followed token "on" (chip, K row, V row). At 400 px the stage scrolls
inside its container; the caption, controls and the stand-in line stack below at full width.

## 12. Open questions for the reviewer
**Data-pass keys** (reuse the pilots' proposals; new keys only where marked *new*):
1. `models.gpt-3.*`: `kv_bytes_per_token` = 4,718,592, `n_kv_heads` = 96 (*new* on this entry; the
   others are `decoder-anatomy` §12.3 / `attention` ruling 3), source arXiv 2005.14165, confirmed.
2. `models.llama-3.1-70b.*` and `models.deepseek-v3.*`: as accepted in `paged-attention` §13; this page
   needs `layers`, `n_kv_heads`, `head_dim`, `context_length` (Llama) and `layers`, `mla_kv_rank`,
   `mla_rope_dim`, `kv_bytes_per_token` (V3). Brief 01 §4 calls it "Llama-3-70B"; the entry is 3.1
   (lesson 8).
3. *new*: `models.gpt-oss-120b.kv_bytes_per_token` = 36,864 and `.kv_fixed_bytes` = 4,718,592, both
   derived (01 §4 table: 18 full layers × 8 KV × 64; 18 window layers × 128 tokens), source the existing
   config.json URL, confidence confirmed (config) with note "derived".
4. `hardware.b300.hbm_gb` = 288 exists (reported); the toy's second GPU uses it. Confirm the reviewer is
   happy to show a reported capacity next to H100's (also reported in the data file).

**Graph changes:** none. The hand-offs to `prefill-decode`, `paged-attention` and `prefix-caching` are
in-page links (transitive through `serving-overview`).

**Judgment calls:**
1. **Units.** `decoder-anatomy` prints per-token sizes in binary units (4.5 MiB, 69 KiB), as brief 01
   does; `paged-attention` ruled "decimal units throughout" (5.2 MB per block, 42.9 GB). This page owns
   the formula and proposes: per-token sizes in binary (KiB/MiB) with the exact byte count beside them,
   totals in decimal (GB, as GPUs are sold), stated in one visible line. Accept, or force one system
   course-wide? (If decimal everywhere: 4.72 MB, 328 kB, 70.3 kB per token.)
2. **`sharePct` home.** `paged-attention` §6 defines `sharePct(part, whole)` in `math/paging.js`; this
   page needs the same metric earlier in the course. Proposal: define it once in `math/memory.js` and
   have `paging.js` import it (lesson 16: one definition). Needs the Serving owner's agreement.
3. **The spec hook.** Spec §4's example hook ("token 10,000 cost the same compute as token 10, but more
   memory") is only half true: the weight multiplications are flat, but the attention read grows with
   the context. This page uses a different hook and turns the claim into misconception 2. Confirm.
4. **Prefill vs decode overlap.** Spec §3.1 puts "prefill vs decode" on this page; `prefill-decode`
   (Serving) owns the compute-bound vs memory-bound story. This page names both phases (frame 4) and
   counts their work, and defers "bound by what" to `prefill-decode`. Confirm the Serving agent's page
   does not re-teach the cache mechanism.
5. **DeepSeek-V4-Pro chip.** It shows the reported 4,000–12,000 B range and disables the shape sliders,
   because no single config gives one number (spec §7 conflict, kept open). The derivation of both ends
   lives in `long-context-attention`. Keep the chip, or leave V4-Pro out of this toy until the conflict
   is resolved?
