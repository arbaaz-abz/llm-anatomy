# Prefix caching (`prefix-caching`)

Track: serving · Section: serving · Prereqs: paged-attention
Status: approved (expert review)
Sources: 04 §2.4, §3.3, §3.4, §3.6, §6.3, §7.4, §7.5, §9.1 (prefix-cache and router toys), §9.2; 05 §1.2. Beyond the briefs, read by the storyboard author (Opus 5.5) on 2026-10-07: the vLLM design doc "Prefix Caching" (docs.vllm.ai/en/latest/design/prefix_caching.html: "We only cache full blocks"; block hash = parent hash + block tokens + extra hashes such as LoRA IDs and multimodal input hashes; freed blocks join the tail of the free queue in reverse order; a hit "touches" a block, raising its reference count and removing it from the free queue; eviction pops the queue's head, the least recently used block) and Gordić, "Inside vLLM" (vllm.ai/blog/2025-09-05-anatomy-of-vllm: default block size 16; incomplete blocks cannot be cached). Neither source addresses how a partial prompt block is handled for parallel samples (n > 1); see §12. Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Running example: `paged-attention`'s blocks of 4 (vLLM uses 16) and its letters A–D with the same hues. The prompts below are hand-picked stand-ins; A's answer is the course-wide `The cat sat down`. These four requests are this page's own cast (only the letters and `--req` hues are the course's, not the A–D table of `batching` and `paged-attention`). Each request runs to completion before the next arrives, so this page can follow one cache.

| Request | Prompt (tokens) | Answer | Prompt length |
|---|---|---|---|
| A | `You are a cat` `. Reply in rhyme` `Where did you sit` | `The cat sat down` | 12 |
| B | `You are a cat` `. Reply in rhyme` `Do you like fish` `?` | `Yes , fish` | 13 |
| C (A's second turn) | `You are a cat` `. Reply in rhyme` `Where did you sit` `The cat sat down` `Why down there ?` | `It was warm .` | 20 |
| D | `You are a dog` `. Reply in prose` `Where did you sit` | `On the mat .` | 12 |

## 1. Learning objective
After this page you can explain how a server reuses the KV of a prompt it has already seen: full blocks keyed by their whole prefix (frames 1–2), matched from the first token (frames 3–6), evicted least-recently-used first (frame 7), counted as a hit rate (frame 8), steered by a cache-aware router (frame 9) and spilled to slower tiers (frame 10). You can also say why cached input tokens are priced at a tenth or less of normal ones, and what the provider still pays for (frame 11, try-this 3).

## 2. Misconceptions to correct
- **Misconception:** "If the same text appears anywhere in my prompt, its KV can be reused." → **Reality:** a token's keys and values depend on every token before it, so reuse works only for a shared *prefix*, matched from the first token. D repeats A's whole question `Where did you sit`, after a different system prompt, and reuses nothing. Corrected by frames 2 and 6. (04 §3.3; first principles from `attention`)
- **Misconception:** "Every token of a matching prefix is reused." → **Reality:** engines that cache by block reuse only full blocks; B's 13th prompt token sits in a partly filled block and is recomputed. Block size sets the grain: at 16, B's 8-token system prompt no longer fills a block and its hit drops to 0. Corrected by frame 4 and try-this 1. (vLLM design doc, read 2026-10-07; 04 §3.3)
- **Misconception:** "Cached tokens are cheap because the provider stores them for free." → **Reality:** a hit skips prefill math (707 ms of GPU time for a 10,000-token prefix on `prefill-decode`'s H200 example), but the KV has to be held somewhere until it is reused, 3.28 GB for that same prefix. That is why writing to the cache costs *more* than plain input (1.25× for a 5-minute entry, 2× for an hour, at Anthropic) and why entries expire. Corrected by frame 11 and try-this 3. (04 §3.6)
- **Misconception:** "Prefix caching is a separate cache next to the KV pool." → **Reality:** it is the same pool and the same blocks as `paged-attention`; a finished request's blocks simply keep their labels until they are evicted. Corrected by frames 1 and 7. (04 §3.3; vLLM design doc)

## 3. Hook and intuition (final wording)
**Hook:** Why do API providers charge about a tenth as much (or less) for input tokens they have seen before, and why does putting them in the cache cost extra?

In `paged-attention` a finished request handed its blocks back to the pool. Prefix caching keeps them instead. Each full block stays in GPU memory with a label: its own tokens plus the label of the block before it. That chain matters, because a token's keys and values depend on every earlier token. The same four words after a different start are a different block. When a new request arrives, the server walks its prompt block by block from the first token, and every block it finds already labeled is a hit: its keys and values are already there, so prefill skips those tokens entirely.

System prompts, few-shot examples, documents, and above all multi-turn chats and agent loops resend the same beginning again and again, so hits are common. DeepSeek reported 56.3% of its input tokens hitting the cache in production. That makes caching a router problem too: a hit is only possible on the replica that holds the blocks, so 2026 routers send each request where its prefix already lives.

Memory still runs out. Unused labeled blocks wait in a free queue, oldest first, and are evicted only when someone needs space, so prefixes that keep getting reused (a shared system prompt) stay, and one-off tails go. Newer engines move evicted blocks to CPU memory or storage rather than dropping them, because reloading a long prefix is cheaper than recomputing it.

That is the answer to the hook. A cache hit saves the provider the prefill math for those tokens, which for a long prompt is most of the request's input cost, so it can sell them for a tenth or less. But someone has to keep the KV around until the next request comes, in fast memory that could be serving other users. Writing to the cache is priced above plain input, and entries expire after minutes or an hour, depending on what you pay. The exact multipliers are business decisions that vendors do not explain.

## 4. Visual metaphor
Top half: a **prefix tree** of blocks (new glyph `prefixTree`, below). Each node is one full block of 4 tokens, drawn as a `block` labeled with its 4 words in mono text; edges run parent → child. The root is "start of prompt". A request's path through the tree is the followed item: its nodes wear the accent frame and its letter chip sits at the leaf. Hit nodes fill with `--sem-ok`, newly computed nodes with the request's `--req` hue.

Bottom half: `paged-attention`'s `blockPool` with 8 physical blocks of 4 slots (blocks numbered from 0), each slot showing the owner letter of the request that computed it. Labeled-but-unused blocks use the library's fourth slot state, `cached` (`--ink-muted` fill, letter kept; P4-R11). Under the pool, a plain text row "free queue (evict from the left): 5 4 7 6 3 2 1 0". A `memBar` is not used here: the quantity is hits, not waste.

Counters at the right (printed numbers): "prompt tokens", "from cache", "computed", "hit rate".

Layout for 580 × 366: tree in the top 190 px. The longest path has 6 blocks (A's turn plus C's), so nodes are 64 × 26 and print their first word + "…"; the followed request's nodes print all four words, and the page passes `prefixTree`'s `wrapAfter: 6`: at 64 px plus 14 px edges, the longest path (6 blocks plus the root, 7 nodes) takes 532 px in one row, and a deeper path wraps to a second row after its 6th block; pool 2 rows × 4 blocks at the bottom (cell 14, as in `paged-attention`). At 400 px the counters move under the pool.

Terms introduced (one per frame): cached block (1), block key (2), cache hit (3), full blocks only (4), multi-turn reuse (5), prefix match from the start (6), LRU eviction (7), hit rate (8), KV-aware routing (9), offload tier (10), cache read / write price (11). Terms assumed from `paged-attention`: block, slot, block size, pool, reference count, free block; from `prefill-decode`: prefill, step time.

Glyphs used: `block`, `blockPool` (with its `cached` state), `prefixTree`, `token` (request letter chips, `owner`), `bars` (frame 11's price chart), `flow` (carry `kv` for reuse arrows and for offload in frame 10), `gpu` and `block` "CPU memory" / "storage" (frame 10).

Plain text labels: "hand-picked prompts; blocks of 4 (vLLM uses 16)" (frame 1); "tokens count from 1; blocks from 0" (frame 1); "free queue (evict from the left)"; "What if?" is not used (all frames are on the main thread); the vLLM numbers' conditions in frame 10; "prices read 2026-10-07; they change" (frame 11).

Glyphs added to the shared library (P4-R11, P4-R14):
- **`prefixTree(parent, { x, y, nodes: [{ id, parent, label, state: 'hit' | 'new' | 'cached' | 'evicted', owner, followed }] })`**: a left-to-right tree of `block` nodes with plain edges (not `flow`: nothing moves along them). Why: no library glyph draws a tree, and the radix/prefix tree is the brief's suggested centerpiece (04 §3.3). Used by this page's animation and toy only (frame 9 draws two small copies); a one-page glyph, built in the shared library because `shared/**` is frozen for page builders; wraps after 4 children.
- **`blockPool` `cached` slot state**: holds data, no running owner, evictable. Why: the three existing states (filled, reserved, free) cannot show "kept for reuse". `--ink-muted` fill, letter printed, no outline (outlines mean selection); free slots stay a faint fill.

## 5. Animation script
Numbers: `simulatePrefixCache({ requests: PREFIX_REQUESTS, blockSize: 4, poolBlocks: 8 })` (§6). On-screen token positions count from 1; physical blocks from 0 (frame 1 says so).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | A (framed) has just finished. Pool: blocks 0–3 hold A's 16 tokens. Tree: root → `You are a cat` → `. Reply in rhyme` → `Where did you sit` → `The cat sat down`. Labels: "hand-picked prompts; blocks of 4 (vLLM uses 16)"; "tokens count from 1; blocks from 0"; "1 slot = one token's K and V, for every layer (from PagedAttention)". | A's four blocks turn from `filled` to `cached` instead of fading to free; the tree's four nodes draw one by one. | Request A is done, but its four full blocks of keys and values stay in GPU memory instead of being freed. Each one is labeled by its tokens. | A: 12 prompt + 4 answer = 16 tokens = 4 blocks (0–3) |
| 2 | The `Where did you sit` node is enlarged; its label expands to show its key: `You are a cat · . Reply in rhyme · Where did you sit`. | The parent labels slide into the node's key from the left. | A block's label is its own tokens plus everything before it. Keys and values depend on earlier tokens, so the same words after another start are a different block. | key of block 2 = 3 blocks of tokens (12 tokens) |
| 3 | B (framed) arrives. Its prompt walks the tree: `You are a cat` ✓, `. Reply in rhyme` ✓, then `Do you like fish` is new. Hit nodes fill `--sem-ok`; counters update. | A `flow` (carry `kv`) runs from blocks 0, 1 into B's block table; the new block lights in B's hue. | B starts with the same system prompt. Its first two blocks match, so prefill skips 8 tokens and computes only the 5 that differ. | B: 13 prompt tokens · from cache 8 · computed 5 |
| 4 | B's 13th prompt token `?` sits alone in a partly filled block (3 slots hatched as in `paged-attention`). After B's answer `Yes , fish` fills it, the block gains a tree node `? Yes , fish`. | The partial block fills with three answer tokens, then gets its label. | B's last prompt block holds only one token, so it is not cached yet. Engines share only full blocks; this one becomes cacheable once B's answer fills it. | partial block: 1 of 4 → full after 3 answer tokens · B uses blocks 0, 1, 4, 5 |
| 5 | C (framed, labeled "A's second turn") arrives: 20 prompt tokens. Its path matches 4 nodes, including A's answer `The cat sat down`; one new node `Why down there ?`. | The path lights node by node; four turn `--sem-ok`. | C is A's next turn, so its prompt resends the whole conversation so far. Sixteen of its twenty prompt tokens hit, including A's own answer. | C: 20 prompt · from cache 16 · computed 4 |
| 6 | D (framed) arrives. A second root branch: `You are a dog` → `. Reply in prose` → `Where did you sit`. A solid 1 px `--ink-muted` connector joins D's `Where did you sit` to A's identical node, labeled "same words, different key". | D's first block is checked against `You are a cat` and fails; the whole branch draws in D's hue. | D repeats A's question word for word, but its first block differs. A match has to start at the first token, so D reuses nothing. | D: 12 prompt · from cache 0 · computed 12 (token-level matching would reuse 3: `You are a`) |
| 7 | D (framed) still needs four blocks of its own, and the pool is full (8 blocks, all `cached`). The free-queue row reads `5 4 7 6 3 2 1 0`. Blocks 5, 4, 7, 6 are popped from the left; their tree nodes (`? Yes , fish`, `Do you like fish`, `It was warm .`, `Why down there ?`) turn `evicted` (hatched: no longer counts). | Four blocks leave the queue's left end one at a time; their tree nodes fade; D's tokens fill them. | When D needs four blocks, the least recently used ones are evicted first: B's and C's own. The shared system prompt was touched most recently, so it survives. | evicted: blocks 5, 4, 7, 6 · kept: 0–3 (system prompt, A's turn) · queue after D: 3 2 1 0 6 7 4 5 |
| 8 | Counters for all four requests in a small table; the hit-rate counter in large type. Below it, plain text: "DeepSeek production, Feb 2025: 56.3%". | Each row's numbers count up; the total ticks to 42.1%. | Across the four requests, 24 of 57 prompt tokens came from the cache, a 42.1% hit rate. DeepSeek measured 56.3% across its production traffic in 2025. | 0 + 8 + 16 + 0 = 24 of 12 + 13 + 20 + 12 = 57 → 42.1% |
| 9 | Two replica `block`s side by side, each with a mini tree and a plain label: "replica 1 ran A", "replica 2 ran B". A line under them: "routers also balance load, so they trade some hits for even queues". C arrives at a `block` "router". Two arrows: "round-robin → replica 2", "KV-aware → replica 1", each with its hit count. | Both arrows draw; the KV-aware one turns `active`. | With several replicas, the router decides whether a hit is even possible. Sent where A ran, C reuses 16 tokens; sent anywhere else, only 8. | replica 1: 16 hit tokens · replica 2: 8 |
| 10 | A three-tier strip: `gpu` "HBM" → `block` "CPU memory" → `block` "storage". Evicted blocks slide down a tier instead of vanishing (`flow`, carry `kv`). Text under it: the vLLM result with its conditions. | Blocks move HBM → CPU memory; one later moves back up on a hit. | Newer engines move evicted blocks to CPU memory or storage instead of dropping them. Reloading a long prefix is cheaper than computing it again. | vLLM tiered offload (2026-09): Qwen-35B on 2 × H100, up to 64 conversations fit in HBM; 64–128 needed CPU offload; beyond 128, storage offload more than doubled throughput |
| 11 | A `bars` chart (P4-R17) of the price per 1M input tokens at Anthropic Sonnet 5.5: "plain $2.00", "cache write (5 min) $2.50", "cache read $0.20"; and DeepSeek V4-Pro off-peak: "miss $0.66", "hit $0.022". Label "prices read 2026-10-07; they change". | The bars draw; the read bar is a tenth of the plain one. A readout under the strip: "what a hit saves: 707 ms of GPU math for a 10,000-token prefix; what it costs: 3.28 GB held until reuse (Prefill vs decode's H200 example)". | Providers pass the saving on: at Anthropic a cached input token costs a tenth or less of a normal one. Cache writes cost a quarter more, to hold it. | Sonnet 5.5: $2.00 / $2.50 / $0.20 per M (Opus 5.5 reads at 0.05×) · DeepSeek V4-Pro off-peak: $0.66 miss / $0.022 hit (3.3%) · 707 ms skipped, 3.28 GB held |

Determinism: every frame is a pure function of (step, progress); the cache is recomputed from the first request. Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter, all ≤ 30 words and ≤ 2 sentences.

## 6. Toy
"Grow the prefix tree." Requests arrive in order; the tree, the pool and the counters update live. A visible line above the controls: "Four hand-picked prompts; each finishes before the next arrives. Real system prompts run to thousands of tokens."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `requests` | Who arrives (in order) | Toggle chips | A, B, C, D, then "B again" | A, B, C, D on; "B again" off | — |
| `blockSize` | Block size | Slider (snapped) | 1 (token by token, like SGLang's radix tree), 2, 4, 8, 16 | 4 | — |
| `pool` | Pool size | Preset chips | 6 · 8 · 12 blocks | 8 | — |
| `provider` | Price | Preset chips | Anthropic Sonnet 5.5 · Anthropic Opus 5.5 · DeepSeek V4-Pro (off-peak) | Sonnet 5.5 | `serving.json` pricing entries (§8) |
| `writePremium` | Charge the cache write | Toggle | on / off (DeepSeek has none) | on | — |
| `hitRate` | Hit rate for pricing | Slider + chips | 0–100%, step 1; chips "this toy" (live value), "DeepSeek 2025" (56.3%) | "this toy" | — |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Tree, pool, free queue | `simulatePrefixCache({ requests, blockSize, poolBlocks })` → `log[]`, `cachedBlocks` | drawn |
| Per request: prompt tokens, from cache, computed, blocks used, evicted | `log[i]` | `formatInt`; evicted block labels |
| Hit rate | `.hitTokens / .promptTokens` | `sharePct` (1 decimal) |
| Scale-up line: "a 10,000-token shared prefix on `prefill-decode`'s H200 example" — prefill time skipped, KV bytes held | `stepTime({ …running example…, tokens: 10000, seqs: 0, context: 0 }).timeS`; `kvCacheBytes({ bytesPerToken: 327680, tokens: 10000 })` (`kv-cache`) | `formatDuration`; `formatBytes` |
| Blended price per M input tokens, and vs no caching | `blendedInputPrice({ hitRate, basePrice, readMult, writeMult })` | $ per M, 2 decimals (4 for DeepSeek) |

**Check my work** (default state: Sonnet 5.5, write premium on, hit rate "this toy"):
```
h = 24 hit tokens ÷ 57 prompt tokens = 42.1%
price per M = (1 − h) · base · write + h · base · read
            = 0.579 · $2.00 · 1.25 + 0.421 · $2.00 · 0.1 = $1.45 + $0.084 = $1.53
```

**Try this** (each leads to a named insight)
1. Slide block size 4 → 16 → 1 (the order matters: the effect is not monotone). From 4 to 16 the hit rate falls from 42.1% to 28.1%. B's 8-token system prompt no longer fills a block (0 hits), while C still reuses 16 tokens because A's whole first turn filled exactly one block. Now slide to 1: 47.4%, because D reuses `You are a`. → **Insight: block size sets the grain of reuse.** Smaller blocks catch more, at the cost of bigger tables and smaller memory reads (`paged-attention` try-this 2); vLLM uses 16, SGLang matches token by token.
2. Keep pool 8 and turn on "B again" after D. D evicted B's and C's private blocks, but B still reuses its 8-token system prompt; this time A's turn (`Where did you sit`, `The cat sat down`) is evicted to make room. Switch the pool to 6: C already loses B's blocks, and D evicts A's turn too. → **Insight: least-recently-used eviction keeps whatever keeps getting reused,** which is usually the shared beginning.
3. Pricing, Sonnet 5.5, write premium on. Hit rate 0%: $2.50 per M, more than not caching at all ($2.00). This toy's 42.1%: $1.53. DeepSeek's 56.3%: $1.21. Switch to DeepSeek V4-Pro: $0.66 → $0.30 at 56.3%. → **Insight: caching pays only when hits come back.** A write costs extra because the provider must hold your KV for minutes; a hit is cheap because it skips prefill math.

**`math/prefix.js`** (pure, no DOM):
```js
PREFIX_REQUESTS                                   // the table above, frozen: [{ id, prompt: [tokens], output: [tokens] }]
blockKeys(tokens, blockSize) → [key]              // one key per FULL block; key = parent key + this block's tokens
matchPrefix(cachedKeys, tokens, blockSize) → { hitBlocks, hitTokens }   // walk from the first block, stop at the first miss
simulatePrefixCache({ requests, blockSize, poolBlocks = Infinity })
  → { log: [{ id, promptTokens, hitTokens, computed, blocks, evicted, cached, freeQueue }], promptTokens, hitTokens, hitRatePct, cachedBlocks }
  // requests run one after another; hits are touched (removed from the free queue); new blocks pop the queue head,
  // evicting its label if it had one; a block is labeled once full (prompt or answer tokens); a finished request's
  // blocks join the queue tail in reverse order (vLLM design doc); throws Error('pool exhausted at <id>') if a request
  // needs more blocks than the pool holds
routeHits({ replicas: [Set<key>], request, blockSize }) → [hitTokens per replica]
blendedInputPrice({ hitRate, basePrice, readMult, writeMult = 1 }) → number   // (1 − h)·base·write + h·base·read
```
Worked examples (scratch implementation, 2026-10-07):
```
simulatePrefixCache({ requests: PREFIX_REQUESTS, blockSize: 4, poolBlocks: 8 })
  A: 12 prompt, 0 hit, blocks [0,1,2,3], queue after 4 5 6 7 3 2 1 0
  B: 13 prompt, 8 hit, 5 computed, blocks [0,1,4,5], queue 6 7 3 2 5 4 1 0
  C: 20 prompt, 16 hit, 4 computed, blocks [0,1,2,3,6,7], queue 5 4 7 6 3 2 1 0
  D: 12 prompt, 0 hit, blocks [5,4,7,6], evicted '? Yes , fish', 'Do you like fish', 'It was warm .', 'Why down there ?'; queue 3 2 1 0 6 7 4 5
  total 24 / 57 = 42.1%
+ "B again": 8 hit, blocks [0,1,3,2], evicted 'The cat sat down', 'Where did you sit'
poolBlocks 6: C evicts B's two blocks; D evicts 'It was warm .', 'Why down there ?', 'The cat sat down', 'Where did you sit'
hit rate by block size (unlimited pool): 1 → 27/57 47.4% · 2 → 26/57 45.6% · 4 → 24/57 42.1% · 8 → 24/57 42.1% · 16 → 16/57 28.1%
  (per request at 16: A 0, B 0, C 16, D 0; at 1: D 3)
routeHits({ replicas: [A's keys, B's keys], request: C, blockSize: 4 }) → [16, 8]
blendedInputPrice: Sonnet 5.5 (2, read 0.1, write 1.25): h 0 → 2.50, 0.421 → 1.53, 0.563 → 1.21 · no write premium, 0.563 → 0.99
                   Opus 5.5 (4, read 0.05, write 1.25): 0.563 → 2.30
                   DeepSeek V4-Pro off-peak (0.66, read 0.022/0.66, write 1): 0 → 0.66, 0.563 → 0.3008
scale-up: stepTime(RUNNING_EXAMPLE, 10,000 prefill tokens) → 707.4 ms (prints 707 ms) skipped; kvCacheBytes({ bytesPerToken: 327680, tokens: 10000 }) → 3,276,800,000 B (3.28 GB) held
```
Reproducer (run once `math/prefix.js` exists):
```
node -e "import('./math/prefix.js').then(m => { for (const bs of [1, 2, 4, 8, 16]) { const s = m.simulatePrefixCache({ requests: m.PREFIX_REQUESTS, blockSize: bs }); console.log(bs, s.hitTokens, s.promptTokens, s.hitRatePct); } console.log(m.simulatePrefixCache({ requests: m.PREFIX_REQUESTS, blockSize: 4, poolBlocks: 8 }).log); console.log(m.simulatePrefixCache({ requests: [...m.PREFIX_REQUESTS, m.PREFIX_REQUESTS[1]], blockSize: 4, poolBlocks: 8 }).log.at(-1)); for (const h of [0, 0.421, 0.563]) console.log(h, m.blendedInputPrice({ hitRate: h, basePrice: 2, readMult: 0.1, writeMult: 1.25 })) })"
```
Tests to write first: the table above; `hitTokens` is a multiple of `blockSize`; hits never exceed the prompt length; `blockKeys` of two prompts that differ only in the first token share no key; the pool never holds more than `poolBlocks` blocks; a request needing more blocks than the pool throws `pool exhausted`; `blendedInputPrice` at h = 1 equals `basePrice · readMult`; inputs not mutated.

## 7. Show me the math
```tex
\text{key}(b_i) = \operatorname{hash}\big(\text{key}(b_{i-1}),\ \text{tokens}(b_i)\big),\qquad \text{key}(b_{-1}) = \text{root}
```
```tex
\htmlClass{hl-hit}{\text{hit tokens}} = B \cdot \max\{\,j : \text{key}(b_0), \dots, \text{key}(b_{j-1}) \in \text{cache}\,\}
\qquad
\text{hit rate} = \frac{\sum \htmlClass{hl-hit}{\text{hit tokens}}}{\sum \text{prompt tokens}}
```
```tex
\text{price per M input} = (1-h)\cdot p_{\text{in}}\cdot m_{\text{write}} + h \cdot p_{\text{in}}\cdot \htmlClass{hl-hit}{m_{\text{read}}}
```
Shapes: one block = `B` tokens of `[n_kv × d_head]` K and V per layer, as in `paged-attention`. Block indices count from 0. Color links: `hl-hit` = the `--sem-ok` hit nodes and the cache-read price bar. What the hit saves (first principles, `prefill-decode`): `2 · N_active · hit tokens` FLOPs of prefill; what it costs: `hit tokens × KV bytes per token` held until reuse.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| vLLM automatic prefix caching: each full block hashed by (parent hash, token ids, extra keys); only full blocks are cached; unused cached blocks evicted least recently used first | `serving.json/vllm-prefix-cache.granularity = "full blocks"`, `.eviction = "LRU free queue"` *(proposed, confirmed: design doc read 2026-10-07)* | 04 §3.3; design doc (header) |
| vLLM's default block size is 16 tokens | `serving.json/vllm.default_block_size = 16` *(proposed by `paged-attention`; same key)* | 04 §3.2; Gordić post |
| SGLang's RadixAttention keeps cached KV in a radix tree keyed by token sequences, with LRU leaf eviction | `serving.json/sglang.radix_attention` *(proposed; `reported`: brief marks the paper "prior")* | 04 §3.3 |
| DeepSeek production (V3/R1, Feb 2025): 56.3% of 608B input tokens per day hit the KV cache | `serving.json/deepseek-v3-production.kv_hit_rate_pct = 56.3` *(proposed by `paged-attention`)*, `.input_tokens_per_day = 608e9` *(proposed)* | 04 §2.4, §6.3 CONFIRMED |
| KV-aware routing: NVIDIA Dynamo 1.0's router sends requests "to GPUs that already have the most relevant short-term memory"; llm-d routes prefix-aware; engines publish KV events so routers know what each replica holds | `serving.json/dynamo.kv_aware_router` *(confirmed)*, `serving.json/llm-d.prefix_aware_routing` *(reported)*, `serving.json/vllm-tiered-kv.kv_events` *(confirmed)* *(all proposed)* | 04 §2.4 |
| Mooncake (Kimi): a global KV pool with a cache-aware scheduler; 59–498% more effective request capacity under SLOs (reported) | `serving.json/mooncake.capacity_gain_pct = [59, 498]` *(proposed, `reported`)* | 04 §2.4 |
| vLLM tiered KV offloading (2026-09-10): HBM → host DRAM → filesystem / S3 / peer RDMA; Qwen-35B on 2 × H100: up to 64 conversations in HBM, 64–128 need CPU offload, beyond 128 storage offload more than doubled throughput | `serving.json/vllm-tiered-kv.tiers` *(proposed by `paged-attention`)*, `.hbm_conversations = 64`, `.cpu_offload_range = [64, 128]`, `.storage_gain = "more than 2x beyond 128"` *(proposed)* | 04 §3.4 CONFIRMED |
| Anthropic (read 2026-10-07): cache read 0.1× input for most models, cache write 1.25× (5 min) or 2× (1 h); Sonnet 5.5 $2/M input; Opus 5.5 $4/M with reads at 0.05× ($0.20) | `serving.json/pricing-anthropic.cache_read_mult = 0.1`, `.cache_write_5m_mult = 1.25`, `.cache_write_1h_mult = 2`; `pricing-anthropic-sonnet-5.5.input_usd_per_m = 2`; `pricing-anthropic-opus-5.5.input_usd_per_m = 4`, `.cache_read_mult = 0.05` *(proposed)* | 04 §3.6 CONFIRMED |
| DeepSeek V4-Pro off-peak (read 2026-10-07): $0.66 per M input on a miss, $0.022 on a hit (3.3%), no write fee; peak rates are 2× | `serving.json/pricing-deepseek-v4-pro.input_miss_usd_per_m = 0.66`, `.input_hit_usd_per_m = 0.022`, `.peak_mult = 2` *(proposed)* | 04 §3.6 CONFIRMED |
| Why a write costs more and a read less is not published; this page's reasoning (prefill skipped, KV held) is labeled as explanation, not a vendor statement | plain sentence | 04 §3.6 (ANALYSIS) |

## 9. Takeaways
1. Prefix caching keeps finished requests' full blocks, labeled by their whole prefix, and a new request reuses every block it matches from its first token; the same words after a different start do not match (frames 1–6).
2. Hits depend on the grain (block size), on eviction (least recently used first, so shared beginnings survive) and on routing (the hit is only on the replica that holds the blocks) (frames 7–9: eviction, tally, routing; try-this 1–2).
3. A hit saves prefill math but the KV must be held somewhere, which is why cache reads are priced at a tenth or less of plain input and cache writes above it (frame 11, try-this 3).

## 10. Next and go deeper
Next: `serving-calculator` · Related: `paged-attention` (the blocks being reused), `batching` (the scheduler that admits hits), `disaggregation` (moving KV between GPUs).

Go deeper (brief 04 §9.2, 05 §1.2): Aleksa Gordić, "Inside vLLM: anatomy of a high-throughput LLM inference system" (https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), handed over from `paged-attention` · vLLM, "Tiered KV offloading" (https://vllm.ai/blog/2026-09-10-tiered-kv-offloading) · Zheng et al., SGLang / RadixAttention (https://arxiv.org/abs/2312.07104).

## 11. Key-frame sketch
Frame 7 end state (D just admitted), desktop width; rows from the `poolBlocks: 8` log in §6. `#` filled slot, `c` cached (labeled, unused), `x` evicted node.
```text
Prefix caching                  step 7 / 11   [<] [Play] [>]
start -+- S1 - S2 -+- Aq - Aa - xCq - xCa
       |           '- xBq - xBx
       '- T1 - T2 - Dq - Da                <- D (new path)
S1 You are a cat   S2 . Reply in rhyme   Aq Where did you sit
Aa The cat sat down   T1 You are a dog   T2 . Reply in prose
x = evicted: Bq Bx (B's), Cq Ca (C's)
pool  0 cccc  1 cccc  2 cccc  3 cccc
      4 DDDD  5 DDDD  6 DDDD  7 DDDD
free queue (evict from the left): 3 2 1 0
prompt 57 | from cache 24 | computed 33 | hit 42.1%
```
On the stage, nodes are 64 px wide with their first word + "…" (all four words on the followed path); the sketch abbreviates them the same way.

## 12. Open questions for the reviewer
**Handoff from `paged-attention` §12**
- **Partial block for parallel samples (n > 1) in vLLM V1.** Checked 2026-10-07: the vLLM prefix-caching design doc says only full blocks are cached, and Gordić's "Inside vLLM" says incomplete blocks cannot be cached; neither describes what V1 does with the partial prompt block when n > 1 (copy or recompute). Per the dispatch rule, nothing is stated; `paged-attention` frame 9's note stays neutral. A future pass could read vLLM V1's scheduler source for parallel sampling.
- **Gordić's "Inside vLLM"** is now this page's first go-deeper link, as `paged-attention` handed over.

**Glyph proposals**
- `prefixTree` (new; evicted nodes hatched, meaning "no longer counts") and `blockPool`'s `cached` slot state (extension), §4; both accepted by the expert review.

**Data-pass keys**
- All `serving.json` keys in §8; `deepseek-v3-production.*`, `vllm.default_block_size` and `vllm-tiered-kv.tiers` reuse `paged-attention`'s proposed ids.

**Judgment calls**
- **Sequential requests.** The toy runs requests one after another so one cache can be followed; real servers interleave them (`batching`), which changes eviction order but not the matching rule.
- **Token-level matching as block size 1.** Presenting SGLang's radix tree as "block size 1" is a simplification (SGLang also supports larger page sizes); the slider label says "like SGLang's radix tree", not "SGLang".

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled:** eviction (frame 7) comes before the tally (frame 8), so the main timeline never jumps back; frame pointers in §1, §2, §4, §9 and the §11 sketch (now frame 7) follow.
- **Settled:** the partial-block question for n > 1 stays neutral (no primary source addresses it); `paged-attention` frame 9's note stands. Gordić's post is this page's first go-deeper link.
- **Settled:** cache size comes from `kvCacheBytes` (`kv-cache`).
- Fixed: the blended price at the toy's 42.1% hit rate is $1.53 (was $1.55).
- Applied: entry lifetimes "minutes or an hour, depending on what you pay"; frame 11 says "a tenth or less" (Opus 5.5 reads at 0.05×) and shows the 707 ms / 3.28 GB readout on screen; the "business choice" sentence appears once (§3); frame 1's label names its source; try-this 1 names the 4 → 16 → 1 order; frame 9 labels which replica ran which request and names routing's load-balance trade-off.
- Kept as is: the slider label "like SGLang's radix tree" (SGLang's default page size is not in our sources).

