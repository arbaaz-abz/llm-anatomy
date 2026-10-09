# PagedAttention (`paged-attention`)

Track: serving · Section: serving · Prereqs: batching, kv-compression
Status: approved (expert review)
Sources: 04 §1.2, §1.3, §2.1, §2.3, §3.1–3.5, §6.3, §7.5, §9.1, §10; 01 §2, §4, §7; 05 §1–2. Beyond the briefs: the PagedAttention paper (arXiv 2309.06180; §2, §4, §7.1, §7.2 read as PDF text via `pdftotext`), the vLLM launch post (vllm.ai/blog/2023-06-20-vllm), the vLLM docs (`DEFAULT_BLOCK_SIZE`; design/prefix_caching: "we only cache full blocks") and the LMSYS GB300 long-context post (lmsys.org/blog/2026-02-19-gb300-longctx), all re-verified by the storyboard author (Fable) on 2026-10-07. Expert review (Opus 5.5, 2026-10-07) applied; see §13.

## 1. Learning objective
After this page you can explain why reserving KV memory per request wasted most of a GPU (frames 2–3), how fixed-size blocks plus a per-request block table fix it and what they still waste (frames 4–7, try-this 1–2), and how a reference-counted block lets two sequences share one prompt (frames 8–9), the mechanism you will meet again in `prefix-caching`.

## 2. Misconceptions to correct
- **Misconception:** "A request's KV cache has to be one contiguous tensor, so the server must know its final length up front." → **Reality:** the attention kernel only needs to *find* each token's K and V. A lookup table (logical block → physical block) is enough, so blocks can be handed out one at a time, from anywhere in the pool. Corrected by frames 5–6. (04 §3.2; paper §4.)
- **Misconception:** "PagedAttention makes attention faster." → **Reality:** per-token compute is unchanged. In the 2023 paper, the paged kernel was 20–26% slower than FasterTransformer's attention kernel (§7.1); the win was never speed per token. The 2–4× is *throughput at equal latency*: less waste → more requests fit in memory → bigger decode batches (`batching`). Corrected by frame 3 and try-this 1. (04 §1.3, §3.2.)
- **Misconception:** "Paging removes the waste." → **Reality:** each running request still wastes up to `block_size − 1` slots in its last block. Bigger blocks waste more; the paper's measured compromise is 16 tokens: smaller blocks cannot keep the GPU busy, larger ones fragment and share less. Corrected by try-this 2. (paper §7.2; 04 §3.2.)
- **Misconception:** "Reusing a prompt's KV across sequences needs a separate cache." → **Reality:** it is the same pool and the same blocks. A physical block carries a reference count; sharing is one more pointer to it. Keeping full blocks after their request finishes, keyed by content, is prefix caching (deferred to `prefix-caching`). Corrected by frames 8–9 and try-this 3. (04 §3.2–3.3.)

## 3. Hook and intuition (final wording)
**Hook:** Why did early LLM servers run out of KV memory while 60–80% of it held nothing?

In its attention layers, a request keeps a K and a V tile for each token it has seen, and it gets one more pair per decode step. (Hybrid models' linear layers keep a fixed-size state instead; see `long-context-attention`.) Nobody knows in advance how many: a reply can be 3 tokens or 3,000. Early servers solved this the blunt way: the moment a request arrived they reserved one contiguous strip big enough for the longest reply the model could give.

Most of that strip stays empty for most of the request's life, and it cannot be lent to anyone else. Strips of different sizes also leave gaps between them that are too small for the next request. The toy below shows only the first kind of waste, so its worst case (52.1%) is lower than the measured 60–80%. The vLLM team measured that only 20–38% of KV memory in such systems held real tokens. The number of requests you can run at once is set by how many fit in memory, so that waste caps the batch, and in decode a bigger batch means more tokens per second, up to a point (`batching`). These are the same four requests as `batching`: its 3-seat lane is this page's before-lane and its 4-seat lane the after-lane, so here the seats become memory.

PagedAttention borrows the operating system's answer to the same problem. RAM is handed out in fixed-size pages, and each program has a page table that maps its own numbering to wherever its pages actually landed. Here the "page" is a block of KV for a fixed number of tokens (16 in vLLM, 4 in this page's toy), handed out only when a request fills its current block, and each request's block table tells the attention kernel where every block is.

Two things fall out. Waste shrinks to at most one partly filled block per request, and finished blocks go straight back to the pool for anyone. And a block becomes something several sequences can point at: two continuations of one prompt share its blocks, with a reference count and copy-on-write. That sharing is the bridge to the next lesson. The price is that paging over-commits: reserving the maximum guaranteed every admitted request could finish, blocks on demand do not, so if the pool runs dry mid-reply the scheduler preempts a request, freeing its blocks and recomputing them later (vLLM V1 recomputes; `batching` names the knob, 04 §2.3).

## 4. Visual metaphor
The page is two lanes of the same 48-slot KV pool, one above the other, driven by the same requests: **"Before: one strip per request"** and **"After: blocks of 4"**. Requests are labeled A–D; each slot shows its owner's letter, solid fill when it holds a token, hatched when it is reserved but empty, faint fill (no outline) when free. An accent frame marks only the request being followed (its bar and its blocks): A in frame 5, C in frame 7, D in frames 8–9. Under the after-lane sits the followed request's block table (logical → physical), and under each lane a usage bar: useful · reserved-empty · free, with "wasted" as the headline number. The `request` bars at the top follow the after-lane's timeline; the before-lane carries its own plain text label for D ("D waiting until step 3").

Layout for the 580 × 366 stage: each lane is 2 rows of 6 blocks × 4 slots (24 cells of 20 px wide); the block table and usage bar sit to the right of the after-lane at desktop width and below it at 400 px. No cell prints a number: slots print one letter, block numbers and "ref" badges sit outside the cells (README lesson 18). Each `memBar` is 500 px wide, so one slot is 10.4 px; a nonzero segment narrower than 18 px (the 1-slot waste at step 5) is drawn at 18 px under a bracket, with its true value printed, as README lesson 19 asks.

Terms introduced on this page (one per frame, defined on screen when first shown): slot (frame 1), reserved strip (2), waiting / queue (3, from `batching`), block (4), block table (5), free block (7), reference count (8), copy-on-write (9). Terms assumed from prereqs: K and V tiles, KV cache and bytes per token (`kv-cache`, `kv-compression`); decode step, batch, prefill (`prefill-decode`, `batching`).

Glyphs used (from spec §5.1):
- `request` (horizontal bar, prefill segment vs decode ticks) for the A–D timeline at the top; `token` chips for the request letters and for the token being written in frames 6 and 9.
- `kvStack` (K/V tiles, one per token) inside each block to show that a slot *is* a K and a V tile; `highlight` marks the token written this step.
- `block` for each physical block's frame (state `active` while being written, `idle` when holding tokens, `dim` when free).
- `flow` with `carry: 'kv'` for the block-table pointers (logical entry → physical block) and for the copy in frame 9.

Plain text labels (not glyphs): the legend "1 slot = one token's K and V, for every layer" (frames 1 and 4); "sizes are hand-picked so you can count; real prompts are thousands of tokens" (frame 1); "only reserved-but-empty waste is shown; real systems also lost gaps between strips" (frame 2); "waiting" badge and "D waiting until step 3" (frame 3); "tokens count from 1; blocks and slots from 0, like memory addresses" (frame 5); "What if?" branch label (frames 8–9); the 2026-engine note under frame 9; "ref 2" badges are the `blockTable` `ref` column plus an optional per-block badge in `blockPool` (below).

Glyphs added to the shared library for this page (P4-R11, P4-R12; none of the above could draw them):
- `blockPool(parent, { x, y, blocks, blockSize, slots, cell, refs })`: a grid of `blocks` fixed-size blocks, each slot drawn as a mini K/V tile with the owner letter and one of three states `filled | reserved | free` (reserved is hatched with the same hatch as `heatmap`'s masked cells; free is a faint fill, never an outline, since outlines mean selection). `refs` is an optional per-block reference-count badge (printed small in the block when > 1). `prefix-caching` uses a fourth slot state, `cached` (holds data, no running owner), which the same glyph carries. Hatch here means "reserved, doesn't count", the course-wide meaning (README lesson 24); `batching` reuses it for idle seats. Why: `kvStack` grows a single row for one request; the pool is many requests interleaved in fixed cells, and the hatched "reserved but empty" state is the whole point of the lesson.
- `blockTable(parent, { x, y, rows: [{ logical, physical, ref }] })`: a two-column mono table (logical i → physical j) with an optional reference-count column. Why: `matrix` encodes numbers as colors; this is a lookup table whose entries must be read as text.
- `memBar(parent, { x, y, w, useful, reserved, free, minSegment = 0 })`: a stacked usage bar with three segments, hatched middle, and the percentages printed; the default `minSegment = 0` is today's drawing, and this page passes `minSegment: 18`: a nonzero segment narrower than 18 px is widened to it under a bracket mark, the others rescaled, the printed value always true. Why: `gpu`'s memory fill bar has one segment and no hatch.
- Categorical request colors `--req-1 … --req-4` (four hues, colorblind-safe, defined in both themes). Why: spec §5.2 only defines a diverging value scale and chrome accents; request identity is categorical. The letter is printed in every slot, so color is never the only encoding.

Semantic colors: `--sem-ok` for useful, `--sem-bad` for reserved-empty, `--line` (faint) for free, `--sem-memory` for pointers; the serving accent frame for the followed request only.

## 5. Animation script
Fixed scenario (also the toy's default). Pool = 48 KV slots (12 blocks × 4). The toy model's context limit is 16 tokens, so "before" reserves 16 per request. Step 0 = prefill of the arrivals; each later step is one decode iteration (one token per running request); a request with `output` tokens finishes at the end of its `output`-th decode step and its memory is free from the next step. Ordering rule (needed to reproduce the block tables): within a step, running requests extend first, in admission order; then arrivals are admitted in arrival order; the lowest free physical block (or, in the before-lane, the lowest free strip) is always handed out first. Each physical block carries a reference count; a write into a block with ref > 1 first copies it to the lowest free block and decrements the original, a write into a block with ref 1 is in place. The resulting tables for steps 0–4 are in §6. The request sizes and the 16-token limit are hand-picked stand-ins so every number can be counted; a visible label on frame 1 says so. Every step is one forward pass of decode on one GPU (serving has no backward pass); steps are counted, not timed, and `batching` times the same steps with the same `TOY_REQUESTS` (its 3-seat continuous lane is the before-lane here, D 3 → 6; its 4-seat lane is the after-lane, D 1 → 4).

| Request | Arrives | Prompt | Output | Final length |
|---|---|---|---|---|
| A | step 0 | 8 | 4 | 12 |
| B | step 0 | 5 | 2 | 7 |
| C | step 0 | 10 | 6 | 16 |
| D | step 1 | 6 | 3 | 9 |

Fragmentation shown: the before-lane reserves the same 16 slots for everyone, so it shows only *internal* fragmentation (reserved, never used); §3 ¶2 names the *external* kind (gaps between strips of different sizes, 04 §3.2) and says why the toy's 52.1% sits below the paper's 60–80%.

On-screen token positions count from 1; block numbers and slots count from 0 (course rule; the page says so in frame 5).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Top: three `request` bars A, B, C (prefill 8, 5, 10; no decode ticks yet). Below: empty 48-slot pool drawn as 12 blocks of 4 in faint fill. Text labels: "1 slot = one token's K and V, for every layer"; "sizes are hand-picked so you can count; real prompts are thousands of tokens". | The three bars slide in from the left. | Three requests arrive. In attention layers, each token a request has seen keeps its K and V in GPU memory until that request finishes. | prompts 8 + 5 + 10 = 23 tokens · pool 48 slots |
| 2 | "Before" lane only: A reserves slots 0–15, B 16–31, C 32–47. Prompt slots fill with letters; the rest turn hatched. `memBar` appears. Text label under the bar: "only reserved-but-empty waste is shown; real systems also lost gaps between strips". | Strips extend to 16 each; then prompt slots fill left→right. | Before paging, each request reserved a contiguous strip for the longest answer it might give. Three requests fill the whole pool before a single token is generated. | useful 23 (47.9%) · reserved-empty 25 (52.1%) · free 0 |
| 3 | Steps 1→2 play in the before-lane: one slot per request fills per step. A text label "D waiting until step 3" with a "waiting" badge sits under the lane (the top bars stay on the after-lane timeline). At the end of step 2, B's strip clears. | Tokens fill; the waiting label pulses once per step. | D arrives and must wait: the pool is 100% reserved and 39.6% empty. In decode a bigger batch means more tokens per second, so waiting costs throughput. | step 2: useful 29 · reserved-empty 19 (39.6%) · D waiting · D starts at step 3 (in B's old strip), finishes at step 6 |
| 4 | "After" lane only, step 0: the same pool as 12 blocks of 4; legend label repeated. A gets blocks 0,1; B gets 2,3; C gets 4,5,6. Empty slots in B's and C's last block are hatched. | Blocks light up one at a time as the prompts fill them. | PagedAttention cuts KV memory into blocks of 4 tokens (vLLM uses 16) and hands a block out only when a request fills its current one. | 7 blocks = 28 slots · useful 23 · reserved-empty 5 (10.4%) · free 20 (41.7%) |
| 5 | Step 1 end state of the after-lane. A is framed (followed). Its `blockTable` [0→0, 1→1, 2→7] beside the pool; `flow` arrows from each row to the physical block. Token chip "A, token 9" travels to its slot. Text label: "tokens count from 1; blocks and slots from 0, like memory addresses". | Arrows draw from table rows to blocks; the token chip travels to block 7, slot 0. | Each request keeps a block table saying where each of its blocks sits in the pool. A's third block landed at 7; the kernel follows the table to find it. | A: 9 tokens → 3 blocks · table 0→0, 1→1, 2→7 · token 9 → block 7, slot 0 (address rule: block = table[⌊(t−1)/4⌋], slot = (t−1) mod 4; §7) |
| 6 | Step 1 in the after-lane for everyone: A opens block 7, B's new token fills a hatched slot in block 3, C's fills block 6. D arrives and gets blocks 8, 9 at once. | One token chip per request lands in a slot; D's two blocks light up. | A request gets a new block only when its current one fills, from anywhere in the pool. That leaves room for D, so D starts now. | 10 of 12 blocks · useful 32 · reserved-empty 8 (16.7%) · free 8 · D starts at step 1 |
| 7 | Steps 2→3: B finishes at the end of step 2 (7 tokens) and blocks 2, 3 fade to free. At step 3 C (now framed) needs a fourth block and takes block 2. C's `blockTable` shows [4, 5, 6, 2]. | Blocks 2, 3 fade to faint; block 2 relights with C's letter; C's table gains a row. | A finished request returns all its blocks at once, and any request can use them right away. Any free block fits anyone, so no gap is ever too small. | step 3: 9 of 12 blocks · useful 32 · reserved-empty 4 (8.3%) · free 12 · D finishes at step 4 (vs 6 before) |
| 8 | Branch: a "What if?" label appears over the pool and the main timeline (bars, step counter) dims. Step 1 replays with D (framed) asking for two answers (parallel sampling, n = 2): D₁ and D₂ each show a `blockTable` [0→8, 1→9]; blocks 8 and 9 carry "ref 2" badges. | The second table's rows draw arrows to the *same* two blocks; ref badges tick 1→2. | What if D asks for two answers? Both point at the same prompt blocks, and each block counts its pointers: ref 2, one stored copy of K and V. | 2 physical blocks instead of 4 · block 8: 4/4, ref 2 · block 9: 2/4, ref 2 |
| 9 | Same "What if?" branch, step 2: D₁ writes its first generated token into logical block 1. Block 9 (ref 2) is copied to free block 10 (`flow`, carry kv), ref(9) drops to 1, the token lands in block 10. D₂ then writes into block 9 with no copy. Visible text note under the lane: "This is the 2023 paper's mechanism. Today's vLLM caches and shares only full blocks, keyed by content (vLLM docs, read 2026-10-07); Prefix caching covers the 2026 engines." | Copy arrow 9→10; badge 2→1; token chips land in 10 and then 9. | A shared block still being filled is copied the first time a sequence writes into it: copy-on-write. Full shared blocks are never copied; prefix caching builds on that. | 3 physical blocks instead of 4 · block 8 stays shared (ref 2) · 11 of 12 blocks in use |

Determinism: every frame is a pure function of (step, progress). The allocator (`simulatePaged`, `simulateContiguous`, `forkBlocks`) recomputes the whole state from step 0 for the requested step, so scrubbing backward is exact. Reduced motion shows each frame's end state.

## 6. Toy
Both lanes are always visible, side by side on desktop and stacked at 400px. Changing any control re-renders both from the `math/` functions. A visible line above the controls repeats: "The four requests and the 16-token limit are hand-picked so you can count; real prompts are thousands of tokens."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `blockSize` | Block size | Slider (snapped) | 2, 4, 8, 16 tokens | 4 | — |
| `step` | Time | Slider | 0 … 6 steps (0 = prefill) | 3 | — |
| `sharedPrefix` | All prompts start with the same 4-token system prompt | Toggle | off / on | off | — |
| `model` | Scale it up | Preset chips | toy (slots only) · GPT-3 · Llama-3.1-70B · DeepSeek-V3 | toy | from `data/models.json` (`kv_bytes_per_token`, `context_length`) |

Pool (48 slots), the four requests, and the before-lane reservation (16) are fixed constants shown beside the controls; the pool is 24 / 12 / 6 / 3 blocks for block size 2 / 4 / 8 / 16.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Before lane: useful / reserved-empty / free, and **wasted %** | `simulateContiguous({requests, poolSlots: 48, maxLen: 16, step})`; wasted % = `waste / poolSlots` | slots; shares through `sharePct` (1 decimal) |
| After lane: useful / reserved-empty / free, **wasted %**, blocks in use | `simulatePaged({requests, poolSlots: 48, blockSize, step, sharedPrefix: on ? 4 : 0})` | slots; `sharePct`; `blocksUsed / poolBlocks` |
| Per request: tokens, blocks, wasted slots, status (waiting / running / done) | `live[]` from the two simulations | mono table |
| Block table of the followed request (tap a request bar; default C) | `live[i].table` | `logical → physical` rows |
| D starts at step / finishes at step, both lanes | `live[3].admitted`, `admitted + output` | step numbers |
| Time-averaged wasted % over steps 0–6, both lanes | mean of `waste / poolSlots` | % |
| Blocks saved by the shared prefix | `blocksSaved` (= `sharedPrefixBlocksSaved`) | blocks |
| Scale-up: bytes per token (exact bytes ≈ kB), bytes per block, one request's max reservation, share of an H100 | `kvBytesPerToken` / `kvBytesPerTokenMla` (`math/memory.js`), `kvBytesPerBlock(bytesPerToken, 16)`, `reserveMaxBytes(bytesPerToken, context_length)` (= `kvCacheBytes`, `kv-cache`'s one definition), `sharePct(reserve, hbmFor(h100).bytes)` (`sharePct` in `math/memory.js`; `hbmFor` in `math/serving.js`, usable when present, else nominal, with its basis word) | "327,680 B ≈ 328 kB"; bytes through `formatBytes` (3 s.f., decimal SI), shares through `sharePct`; the GPU figure is labeled "80 GB nominal" until the data pass adds an H100 usable figure |

**Check my work:** none (simulation).

**Try this** (each leads to a named insight)
1. Keep block size 4 and scrub Time from 0 to 6. Before: 52.1% wasted at step 0, D waits until step 3 and finishes at step 6. After: waste peaks at 16.7% and D starts at step 1, finishes at step 4. → **Insight: reserving for the worst case, not the tokens themselves, is what wastes memory and shrinks the batch.**
2. Set block size to 16: the after-lane becomes identical to the before-lane (52.1% at step 0, D waits, 34.2% averaged). Set it to 2: averaged waste 2.7%, but a 12-token request now needs 6 table entries and 6 scattered reads instead of 3. A visible line under the readout says: "the toy counts slots, not kernel time, so smaller always looks better here; the 2023 paper measured that blocks under 16 could not keep the GPU busy." → **Insight: block size is a dial between internal waste and table/kernel overhead; "one strip per request" is just block size = max length. vLLM's 16 is a measured middle.**
3. Turn on the shared system prompt at step 1, block size 4: the four requests share one physical block, 7 blocks in use instead of 10. Switch to block size 8: 0 blocks saved, because the 4 shared tokens never fill a whole block. → **Insight: sharing works on whole blocks, so block size also sets the granularity of the prefix cache (`prefix-caching`).**

The "Scale it up" chips are a readout, not a prompt: picking Llama-3.1-70B shows "327,680 B ≈ 328 kB per token; one 16-token block is 5.24 MB; the old scheme's reservation for a single request at its 128K context is 42.9 GB, 53.7% of an H100 (80 GB nominal)". A visible line says what is being compared: the grain of allocation (one block) against the worst-case reservation (one request's whole context), both in bytes for one request (README lesson 21). The 42.9 GB reproduces `kv-cache`'s frame 8 number (README lesson 27).

**`math/paging.js`** (pure, no DOM). `kvBytesPerToken`, `kvBytesPerTokenMla`, `kvCacheBytes` and `sharePct(part, whole)` live in `math/memory.js` (owner `kv-cache`) and are imported from there; `sharePct` was moved out of this module by the course-wide settlement. `kvBytesPerBlock` and `reserveMaxBytes` stay here (`paging.js` has one owner: this page); `reserveMaxBytes(bytesPerToken, maxLen)` delegates to `kvCacheBytes({ bytesPerToken, tokens: maxLen })` so the course has one definition of "cache for N tokens" (README lesson 16; a test asserts equality). All functions return new objects; inputs are never mutated. `TOY_REQUESTS` (the table above) is imported from `math/serving.js` (P4-R3), where `batching` and `serving-overview` import it too; it is not defined here.

```js
blocksNeeded(tokens, blockSize) → number            // ⌈tokens / blockSize⌉
internalWaste(tokens, blockSize) → number           // blocksNeeded·blockSize − tokens, in [0, blockSize−1]

simulateContiguous({ requests, poolSlots, maxLen, step })
  → { live: [{ id, admitted, running, waiting, tokens, reserved, strip }], useful, waste, free, poolSlots }
  // admits in arrival order while reserved + maxLen ≤ poolSlots, lowest free strip first;
  // frees a strip after the finish step

simulatePaged({ requests, poolSlots, blockSize, step, sharedPrefix = 0 })
  → { live: [{ id, admitted, running, waiting, tokens, blocks, waste, table }], useful, logicalTokens, waste, free,
      blocksUsed, poolBlocks, blocksSaved }
  // admits while free blocks ≥ blocksNeeded(prompt); ordering rule and ref counts as in §5;
  // sharedPrefix > 0: the first ⌊sharedPrefix / blockSize⌋ blocks of every running request are one physical block;
  // useful = physical slots holding a token (shared slots count once), logicalTokens = tokens summed over requests,
  // so useful + waste + free === poolSlots always;
  // over-commit: if a running request needs a block and none is free, throws Error('pool exhausted at step s')
  // (real engines preempt instead, 04 §2.3; the fixed scenario never hits this for any control setting)

sharedPrefixBlocksSaved({ running, prefixLen, blockSize }) → number   // max(running − 1, 0) · ⌊prefixLen / blockSize⌋
forkBlocks({ prompt, generated, samples, blockSize })
  → { physical, withoutSharing, saved, copies }     // parallel sampling with copy-on-write on the partial block
kvBytesPerBlock(bytesPerToken, blockSize) → number
reserveMaxBytes(bytesPerToken, maxLen) → number     // === kvCacheBytes({ bytesPerToken, tokens: maxLen }) from math/memory.js
// sharePct(part, whole) is imported from math/memory.js (one decimal), not defined here
```

Worked examples (every number below, and every number in §5 and §11, was produced on 2026-10-07 by a scratch implementation of these signatures and must be reproduced by `math/paging.js`; `R` = `TOY_REQUESTS`):
```
blocksNeeded(10, 4) → 3        internalWaste(10, 4) → 2
blocksNeeded(5, 2)  → 3        internalWaste(5, 2)  → 1
blocksNeeded(8, 16) → 1        internalWaste(8, 16) → 8

simulateContiguous({ requests: R, poolSlots: 48, maxLen: 16, step: 0 })
  → useful 23, waste 25, free 0            (wasted 52.1%); strips A 0, B 1, C 2
simulateContiguous({ …, step: 2 })
  → useful 29, waste 19, free 0, live[3] = { id: 'D', waiting: true, admitted: null }
simulateContiguous({ …, step: 3 })
  → useful 30, waste 18, free 0, live[3].admitted = 3, live[3].strip = 1   (B's old strip)
simulateContiguous({ …, step: 6 })
  → useful 25, waste 7, free 16            (A, B done; C and D finish this step)

simulatePaged({ requests: R, poolSlots: 48, blockSize: 4, step: 0 })
  → useful 23, waste 5, free 20, blocksUsed 7, poolBlocks 12
    live: A 8t/2b/w0 [0,1] · B 5t/2b/w3 [2,3] · C 10t/3b/w2 [4,5,6]
simulatePaged({ …, blockSize: 4, step: 1 })
  → useful 32, waste 8, free 8, blocksUsed 10
    tables: A [0,1,7] · B [2,3] · C [4,5,6] · D (admitted 1) [8,9] · free {10, 11}
simulatePaged({ …, blockSize: 4, step: 3 })
  → useful 32, waste 4, free 12, blocksUsed 9
    tables: A [0,1,7] · C [4,5,6,2] (B's freed block reused; block 2 holds 1 token + 3 reserved) · D [8,9] · free {3, 10, 11}
simulatePaged({ …, blockSize: 4, step: 4 })
  → tables: A [0,1,7] · C [4,5,6,2] · D [8,9,3] · free {10, 11}
simulatePaged({ …, blockSize: 16, step: 1 })
  → useful 26, waste 22, free 0, blocksUsed 3 of 3; D waiting      (identical to contiguous)
simulatePaged({ …, blockSize: 4, step: 1, sharedPrefix: 4 })
  → useful 20, logicalTokens 32, waste 8, free 20, blocksUsed 7 (10 logical − 3 saved), blocksSaved 3
simulatePaged({ …, blockSize: 8, step: 1, sharedPrefix: 4 })
  → useful 32, logicalTokens 32, waste 16, free 0, blocksUsed 6 of 6, blocksSaved 0

Fork of D (frames 8–9), block size 4: step 1 D₁ [8,9], D₂ [8,9], ref(8) = ref(9) = 2, free {10, 11};
step 2 D₁ writes → copies 9 → 10, ref(9) = 1, D₁ [8,10]; D₂ writes into 9 in place, D₂ [8,9]; 11 of 12 blocks, free {11}.

Time-averaged wasted % over steps 0–6: contiguous 34.2%; paged 2.7% / 8.0% / 17.6% / 34.2% for block size 2 / 4 / 8 / 16.
Peak paged usage at block size 4 with the prefix off is 10 of 12 blocks (steps 1, 2, 4); 12 of 12 only in the frame-8 fork at step 4. No control setting runs the pool dry.

forkBlocks({ prompt: 6, generated: 0, samples: 2, blockSize: 4 }) → { physical: 2, withoutSharing: 4, saved: 2, copies: 0 }
forkBlocks({ prompt: 6, generated: 1, samples: 2, blockSize: 4 }) → { physical: 3, withoutSharing: 4, saved: 1, copies: 1 }
forkBlocks({ prompt: 6, generated: 3, samples: 2, blockSize: 4 }) → { physical: 5, withoutSharing: 6, saved: 1, copies: 1 }

per token (from math/memory.js): 327,680 B ≈ 328 kB (Llama-3.1-70B) · 70,272 B ≈ 70.3 kB (DeepSeek-V3) · 4,718,592 B ≈ 4.72 MB (GPT-3)
kvBytesPerBlock(327_680, 16)   → 5_242_880    (5.24 MB, Llama-3.1-70B)
kvBytesPerBlock(70_272, 16)    → 1_124_352    (1.12 MB, DeepSeek-V3)
kvBytesPerBlock(4_718_592, 16) → 75_497_472   (75.5 MB, GPT-3)
reserveMaxBytes(327_680, 131_072)  → 42_949_672_960  (42.9 GB; sharePct(…, 80e9) = 53.7% of an H100, 80 GB nominal)
reserveMaxBytes(70_272, 131_072)   → 9_210_691_584   (9.21 GB)
reserveMaxBytes(4_718_592, 2_048)  → 9_663_676_416   (9.66 GB)
```

Reproducer calls (run once `math/paging.js` exists; the same calls produced §5, §6 and §11 today against the scratch implementation; `TOY_REQUESTS` comes from `math/serving.js` as `t.TOY_REQUESTS`):
```
node -e "Promise.all([import('./math/paging.js'), import('./math/serving.js')]).then(([m, t]) => { for (let s = 0; s <= 6; s++) console.log(s, m.simulateContiguous({ requests: t.TOY_REQUESTS, poolSlots: 48, maxLen: 16, step: s })) })"
node -e "Promise.all([import('./math/paging.js'), import('./math/serving.js')]).then(([m, t]) => { for (const b of [2, 4, 8, 16]) for (let s = 0; s <= 6; s++) console.log(b, s, m.simulatePaged({ requests: t.TOY_REQUESTS, poolSlots: 48, blockSize: b, step: s })) })"
node -e "Promise.all([import('./math/paging.js'), import('./math/serving.js')]).then(([m, t]) => { for (const b of [2, 4, 8]) console.log(b, m.simulatePaged({ requests: t.TOY_REQUESTS, poolSlots: 48, blockSize: b, step: 1, sharedPrefix: 4 })) })"
node -e "import('./math/paging.js').then(m => [0, 1, 3].forEach(g => console.log(g, m.forkBlocks({ prompt: 6, generated: g, samples: 2, blockSize: 4 }))))"
node -e "Promise.all([import('./math/paging.js'), import('./math/memory.js')]).then(([p, mem]) => [[327680, 131072], [70272, 131072], [4718592, 2048]].forEach(([b, c]) => console.log(b.toLocaleString('en-US') + ' B', p.kvBytesPerBlock(b, 16), p.reserveMaxBytes(b, c), mem.kvCacheBytes({ bytesPerToken: b, tokens: c }), mem.sharePct(p.reserveMaxBytes(b, c), 80e9))))"
```
(Catch-up pass 2026-10-07, run against the scratch implementation: `327,680 B ≈ 328 kB`, `70,272 B ≈ 70.3 kB`, `4,718,592 B ≈ 4.72 MB`; blocks 5.24 / 1.12 / 75.5 MB; `sharePct(42_949_672_960, 80e9) = 53.7`; one memBar slot at 500 px = 10.4 px, two slots 20.8 px.)
Sketch rows (§11) are printed from the step-3 outputs: before-lane strips in strip order (A, D, C), each `id.repeat(tokens) + '/'.repeat(16 − tokens)`; after-lane cell for physical block p owned by logical index i of request q is `q.id.repeat(min(4, q.tokens − 4·i)).padEnd(4, '/')`, free blocks `....`.

Tests to write first: the table above; `useful + waste + free === poolSlots` for every step, block size and prefix setting; `simulatePaged` never allocates more than `poolBlocks` and never throws on `TOY_REQUESTS` for any control setting; it throws `pool exhausted` on a crafted overflow (e.g. `poolSlots: 8`, `blockSize: 4`, step 1); waste per running request ≤ `blockSize − 1`; `blockSize === maxLen` reproduces `simulateContiguous` exactly for every step; inputs are not mutated (`Object.freeze` the request list).

## 7. Show me the math
```tex
\text{blocks}(n) = \left\lceil \frac{n}{B} \right\rceil,\qquad
\htmlClass{hl-waste}{w_{\text{paged}}(n)} = B\left\lceil \frac{n}{B} \right\rceil - n \;\le\; B-1
```
```tex
\htmlClass{hl-waste}{w_{\text{contig}}(n)} = L_{\max} - n
\qquad
\text{wasted \%} = \frac{\sum_{\text{running}} w}{\text{pool slots}}
```
```tex
\text{token } t \ (t = 1, 2, \dots):\quad
\text{block} = \htmlClass{hl-table}{\text{table}}\!\left[\left\lfloor \tfrac{t-1}{B} \right\rfloor\right],\qquad
\text{slot} = (t-1) \bmod B
```
```tex
\text{bytes per block} = B \cdot \underbrace{2\, L\, n_{kv}\, d_{\text{head}}\, b}_{\text{bytes per token (GQA)}}
\qquad
\text{bytes per token (MLA)} = L\,(d_c + d_{\text{rope}})\, b
```
```tex
\text{fork of } s \text{ samples: } \text{physical} = \left\lfloor \tfrac{P}{B} \right\rfloor + s\left(\left\lceil \tfrac{P+g}{B} \right\rceil - \left\lfloor \tfrac{P}{B} \right\rfloor\right)\ \ (g \ge 1)
```
Shapes: one slot = one K tile and one V tile of `[n_kv × d_head]` per layer, so a block of `B` tokens is `[B × 2 L n_kv d_head]` elements · Indexing: tokens `t` count from 1 on screen; blocks and slots from 0 · Color links (`\htmlClass` names `waste`, `table`): `w` terms in the hatched reserved-empty color (`--sem-bad`); `table` in the pointer color (`--sem-memory`). No color promise is made for `B`, `n` or `P + g` (P4-R16).

## 8. In today's models (Oct 2026)
`data/serving.json` (same `Fact` shape, added to the validator's file list) and the `llama-3.1-70b` / `deepseek-v3` model entries are accepted for the data-extension pass (§13); keys marked *proposed* do not exist yet.

**The 2023 result** (Kwon et al., SOSP'23; the brief marks these *(prior)*; re-verified by the storyboard author on 2026-10-07 from the arXiv 2309.06180 PDF text and the vLLM post of 2023-06-20)

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| In 2023, only 20.4–38.2% of KV cache memory held actual tokens in existing systems (paper Fig. 2); the launch post puts the waste at 60–80%, and under 4% after paging | `serving.json/pagedattention.kv_useful_before_pct = [20.4, 38.2]`, `.waste_before_pct = [60, 80]`, `.waste_after_pct_max = 4` *(proposed; range notes name Fig. 2 and the post)* | 04 §3.2 *(prior)*; paper §2 and post re-verified |
| 2–4× throughput (the paper's request-rate metric, served requests per second at the same normalized latency) vs FasterTransformer and Orca (2023) | `serving.json/pagedattention.throughput_gain = [2, 4]` *(proposed; note names the metric)* | 04 §3.2 *(prior)*; abstract re-verified |
| The 2023 kernel was 20–26% slower than FasterTransformer's attention kernel: the win was memory, not speed per token | `serving.json/pagedattention.kernel_overhead_2023_pct = [20, 26]` *(proposed; note: 2023 measurement, paper §7.1)* | not in the brief; paper §7.1 read 2026-10-07 (§13) |
| vLLM's default block size is 16 tokens; too small starves GPU parallelism, too large fragments and shares less | `serving.json/vllm.default_block_size = 16` *(proposed)* | 04 §3.2; paper §7.2 and vLLM docs `DEFAULT_BLOCK_SIZE` re-verified |
| Paged KV is the default memory model in vLLM, SGLang and TensorRT-LLM ("paged KV cache") | `serving.json/pagedattention.engines` *(proposed; confidence `reported`)* | 04 §3.2 |

**What one block weighs** (decimal units; bytes per token from the `kv-compression` formulas)

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Scale-up presets: Llama-3.1-70B 327,680 B ≈ 328 kB per token (GQA, 2 × 80 L × 8 KV × 128 × 2 B, K and V): 5.24 MB per 16-token block, 42.9 GB reserved for one request at its 128K context = 53.7% of an H100 (80 GB nominal) · DeepSeek-V3 70,272 B ≈ 70.3 kB per token (MLA, 61 × 576 × 2 B): 1.12 MB per block, 9.21 GB at 128K · GPT-3 4,718,592 B ≈ 4.72 MB per token (MHA): 75.5 MB per block, 9.66 GB at its 2,048 context | `models.json/llama-3.1-70b.kv_bytes_per_token`, `.context_length` *(entry proposed; note: the briefs say "Llama-3-70B", but 128K context is the 3.1 checkpoint; same KV shape)*; `models.json/deepseek-v3.kv_bytes_per_token`, `.context_length` *(entry proposed)*; `models.json/gpt-3.kv_bytes_per_token`, `.context_length`; `hardware.json/h100.hbm_gb` (80, nominal; no usable figure is in data; read through `hbmFor`) | 01 §4; 04 §3.1; 04 §1.4 |
| Hybrid models page only their attention layers: Kimi K3 has 24 gated-MLA layers among 69 KDA layers, Qwen3.8 runs 3 Gated-DeltaNet layers per attention layer; the linear layers keep a fixed-size state instead of per-token KV | `models.json/kimi-k3.attention`, `models.json/qwen3.8.attention` | 01 §2 (hybrid linear attention), §4, §7 [C] |

**Where blocks go next**

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| At DeepSeek's production average of 4,989 KV tokens per request (V3/R1, Feb 2025), a 16-token block wastes at most 15 slots = 0.3%; the toy's 8–17% comes from 5–16-token sequences | `serving.json/deepseek-v3-production.avg_kv_length_tokens = 4989` *(proposed)* | 04 §6.3 CONFIRMED |
| Bridge: 56.3% of DeepSeek's input tokens hit the KV cache (prefix reuse), same production day | `serving.json/deepseek-v3-production.kv_hit_rate_pct = 56.3` *(proposed)* | 04 §2.4, §6.3 CONFIRMED |
| vLLM can spill blocks to CPU memory and storage (tiered KV offloading, 2026-09): the numbers are in `prefix-caching` | `serving.json/vllm-tiered-kv.tiers = "HBM → host DRAM → storage"` *(proposed)* | 04 §3.4 CONFIRMED |
| FP8 KV halves bytes per block, but a naive version broke 128K recall (vLLM, 2026-04): details in `quantization` | `serving.json/vllm-fp8-kv.bytes_factor = 0.5`, `.naive_niah_128k_pct = 13` *(proposed)* | 04 §3.5 CONFIRMED |
| Concurrency is free HBM ÷ KV per sequence: DeepSeek-R1 (NVFP4) at 128K input / 8K output has a theoretical cap of 40 requests per GPU on GB300 NVL72 (288 GB) vs 24 on GB200 NVL72 (186 GB); LMSYS's practical target is 36 and 20, about 85% of the cap (LMSYS, 2026-02-19) | `hardware.json/gb300-nvl72.concurrent_128k_per_gpu = 40`, `gb200-nvl72.concurrent_128k_per_gpu = 24` (theoretical); `.concurrent_128k_per_gpu_target` = 36 and 20 (note names the model and 128K/8K workload) | 04 §1.2, §7.5 CONFIRMED; post re-verified 2026-10-07 |

Not shown (brief 04 §10): engine market share, Dynamo's "7×", any UNVERIFIED offload post.

## 9. Takeaways
1. KV memory per request grows one token at a time, and nobody knows the final length; reserving the maximum up front left 60–80% of KV memory empty in 2023 systems and capped the batch (frames 2–3).
2. Fixed-size blocks plus a per-request block table (an OS page table for KV) allocate on demand: waste is at most one partial block per request, freed blocks are reused at once, and the 2023 paper measured 2–4× more throughput at the same latency. Block size is the dial (try-this 2), and the price is over-commit: a dry pool means preemption (`batching`).
3. A block is a unit of sharing: reference counts and copy-on-write let several sequences point at one prompt's blocks (frames 8–9); keeping full blocks after the request ends is prefix caching, which you'll meet in `prefix-caching`.

## 10. Next and go deeper
Next: `prefix-caching` (keep full blocks keyed by content; radix trees; tiered offload; why cached tokens cost ~10× less) · Related: `kv-compression` (what one slot weighs), `batching` (why more requests in memory means more throughput, and preemption), `quantization` (FP8 KV), `disaggregation` (moving blocks between GPUs).

Go deeper (brief 05 §1.2, 04 §9.2): vLLM, "Easy, Fast, and Cheap LLM Serving with PagedAttention" (https://vllm.ai/blog/2023-06-20-vllm) · Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP'23 (https://arxiv.org/abs/2309.06180) · Lynskey, *LLM Inference Explained* (https://llm-inference-explained.vercel.app) for the long form.

## 11. Key-frame sketch
Frame 7 end state (step 3), desktop width; rows generated from the step-3 outputs by the rule in §6. `#` filled slot, `/` reserved-empty, `.` free.
```text
PagedAttention                step 3 / 6   [<] [Play] [>]
A ########|###  11   B done (7)
C ##########|###  13  D ######|##  8   (prompt | generated)
--- Before: one strip of 16 per request -------------------
|AAAAAAAAAAA/////|DDDDDD//////////|CCCCCCCCCCCCC///|
useful 30 (62.5%) | reserved-empty 18 (37.5%) | free 0
D waited until step 3 (B's old strip), finishes at step 6
--- After: blocks of 4, on demand -------------------------
 0 AAAA  1 AAAA  2 C///  3 ....  4 CCCC  5 CCCC
 6 CCCC  7 AAA/  8 DDDD  9 DDDD 10 .... 11 ....
useful 32 (66.7%) | reserved-empty 4 (8.3%) | free 12 (25%)
D started at step 1, finishes at step 4
block table C:  0->4  1->5  2->6  3->2  (reused B's block)
```

## 12. Open questions for the reviewer
- **Real block sizes vary** (reviewer nice-to-have): DeepSeek's 64-token MLA blocks would sharpen try-this 2 but no source is in data, so the page states no FlashMLA block size; `prefix-caching` already states SGLang's token-by-token matching. Left out here.
- **Hybrid-model row keys.** The row uses the existing string facts `kimi-k3.attention` / `qwen3.8.attention`. If the data pass stores layer counts (e.g. `attention_layers`), switch the row to those.
- **H100 usable memory.** The 54% share is computed on 80 GB nominal and labeled so; data holds no H100 usable-memory figure, so the toy keeps nominal.

## 13. Reviewer rulings (main session)
Hand-checked: contiguous step 0 (23/25/0); paged B = 4 at steps 0, 1 and 3 (blocks, useful, waste, free); Llama 327,680 B/token, 5.2 MB block, 42.9 GB = 53.7% of an H100; DeepSeek-V3 70,272 B/token; GPT-3 4,718,592 B/token.
- Data: accept `data/serving.json` (same Fact shape, added to the validator's file list) and the `deepseek-v3` model entry, in the data-extension pass after Task 12. Accept `concurrent_128k_per_gpu` on the two NVL72 hardware entries. **Expert review 2026-10-07: the model entry is `llama-3.1-70b`, not `llama-3-70b` (Llama 3 70B has an 8,192 context; 128K is Llama 3.1, same KV shape). The data pass must create `llama-3.1-70b` and name the correction in its `note`; the NVL72 concurrency facts carry the model (DeepSeek-R1 NVFP4) and the 128K/8K workload in their `note`.**
- "Paged KV is the default in vLLM, SGLang, TensorRT-LLM" is stored as `reported`.
- Scope: keep one line each for tiering and FP8 KV here, linking out. The full treatment goes to `prefix-caching` (tiering/eviction) and `quantization` (FP8 KV). **Expert review 2026-10-07: enforced; §8 now carries one line each and is grouped into three short tables.**
- Fragmentation honesty: agreed. Internal fragmentation only, with the prose naming external fragmentation. **Expert review 2026-10-07: the prose did not name it; §3 ¶2 now does, and explains why the toy tops out at 52.1% below the 60–80% headline.**
- Glyphs: `blockPool`, `blockTable`, `memBar` and categorical `--req-1…4` are accepted and handed to the Task 10 visual-language brief. **Expert review 2026-10-07: `blockPool` gains an optional `refs` badge; free slots are a faint fill, not an outline (outline = selection, gallery rule).**
- Captions: two sentences are fine (storyboard-wide rule). Dwell time is handled in Plan 2 by scaling the stepper dwell with caption length. **Expert review 2026-10-07: frames 5 and 6 exceeded the rule; rewritten to ≤ 2 sentences and ≤ 30 words, address rule moved to Numbers shown and §7; frame 7 caption corrected (B frees both blocks at once); frames 8–9 labeled as a "What if?" branch.**
- `kvBytesPerToken({ layers, kvHeads, headDim, bytesPerElem })` and `kvBytesPerTokenMla({ layers, dLatent, dRope, bytesPerElem })` in `math/memory.js` are passed to the Task 12 Architecture agent as the shared interface.
- The 20–26% kernel overhead stays, sourced to arXiv 2309.06180 §7.1, `confirmed` (primary source read). **Expert review 2026-10-07: kept, but dated "in the 2023 paper" wherever it appears (misconception 2, §8 key renamed `kernel_overhead_2023_pct`).**
- **Settled (`batching` §13, 2026-10-07):** neither `math/paging.js` nor `math/batching.js` simulates preemption; `simulatePaged`'s `pool exhausted` throw contract stands, and `batching` names vLLM V1's recompute behaviour. **Settled (`prefix-caching` §13, 2026-10-07):** no primary source states how vLLM V1 handles the partial prompt block for n > 1, so frame 9's note stays neutral and the "recomputes the partial one" alternative is dropped; Gordić's "Inside vLLM" is `prefix-caching`'s first go-deeper link.
- **Catch-up pass 2026-10-07 (README lessons 16–28 and settlements):** `sharePct` imported from `math/memory.js` (§6 signatures and reproducer updated); `reserveMaxBytes` declared equal to `kvCacheBytes` so "cache for N tokens" has one definition (lesson 16); per-token sizes print exact bytes ≈ decimal kB/MB and all units are decimal SI (settlement); the H100 share is labeled "80 GB nominal" with the usable-figure fallback rule from `prefill-decode` (settlement, lesson 23); the 2–4× row names the paper's metric (lesson 23); try-this 2 says on screen that the toy counts slots, not kernel time, so the real-world "16 wins" is not reversed by the toy (lesson 17); no cell prints a number, and `memBar` widens sub-18 px segments under a bracket (lessons 18–19); the scale-up readout says what it compares and that 42.9 GB reproduces `kv-cache`'s number (lessons 21, 27); §5 states that every step is one forward pass on one GPU, counted not timed (lessons 22, 26); hatch's course-wide meaning and `prefix-caching`'s `cached` state recorded in §4 (lesson 24); settled §12 items moved here and their alternatives deleted (lesson 20); `batching`'s replay of the A–D lanes (3 seats = before, 4 seats = after) cross-referenced in §3 and §5. Numbers touched were regenerated (recorded under the reproducer calls in §6).
- **Expert review 2026-10-07, further rulings followed:** the §11 sketch was wrong in two rows (D sits in B's old strip; C's block 2 holds 1 token) and is now generated from the reproducer; "batch size is throughput" softened to "up to a point"; over-commit/preemption named in §3 and in the `simulatePaged` contract; a slot is defined on screen; the before-lane has its own D label; indexing stated on screen (tokens from 1, blocks and slots from 0, per the README rule; the reviewer's "both from 0" wording was not used because the README fixes token positions at 1-based); decimal units throughout; Gordić's "Inside vLLM" moved to `prefix-caching`; "indirection table" → "lookup table"; hybrid-model line added; plain text labels listed in §4.
- Data pass 2026-10-07: GB200 memory is 186 GB (was 192); 40 and 24 concurrent requests are theoretical caps, with practical targets of 36 and 20 (about 85% of the cap); no H100 usable-memory or FlashMLA block-size claim (nominal 80 GB kept).
