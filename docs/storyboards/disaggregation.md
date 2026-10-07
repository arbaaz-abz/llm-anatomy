# Disaggregated serving (`disaggregation`)

Track: serving · Section: serving · Prereqs: batching, parallelism
Status: approved (expert review)
Sources: 04 §1.1, §1.2, §2.2, §6.1, §6.2, §6.3, §6.4, §8.1, §9.1 (disaggregation and expert-parallelism toys), §9.2; 03 §2.1 (link bandwidths); 05 §1.2.

Running examples: `batching`'s "What if D's prompt were 4,096 tokens?" branch (requests A, C, D on `prefill-decode`'s Llama-3.1-70B / FP8 / H200 example) for the first half; DeepSeek-V4-Pro (384 routed experts, 6 per token, ≈ 865 GB as shipped) on GB300 for the expert-parallel half, the same model `serving-calculator` sizes. Expert review (Fable 5.1, 2026-10-07) applied; see §13.

## 1. Learning objective
After this page you can explain why running prefill and decode on the same GPUs makes the two latency targets fight (frames 1–2), what splitting them into two pools buys and what it costs, including the KV transfer (frames 3–5, try-this 1–2), why a mixture-of-experts model starves its experts at small scale and how wide expert parallelism feeds them (frames 6–8, try-this 3), and why rack-scale NVLink systems suit it (frame 9).

## 2. Misconceptions to correct
- **Misconception:** "Chunked prefill already solves prefill/decode interference." → **Reality:** it bounds the stall, but decode steps still carry prefill work: in `batching`'s branch, every step A and C had left grew from 14.6 to 36.2 ms (two steps for A, four for C). On a separate decode GPU they stay at 14.6 ms. Corrected by frames 1 and 3. (04 §2.2, §6.1)
- **Misconception:** "Moving the KV cache between GPUs costs more than it saves." → **Reality:** transfer time and prefill time both grow with the prompt, so for any prompt past the 217-token crossover their ratio is fixed by the model and the link: for the running example 9.3% over a 400 Gb/s network port and 0.5% over NVLink. It hurts when the link is slow, and, the brief adds, when prompts are short and models small; the short-prompt case comes from a fixed start-up cost per transfer that the toy does not model. Corrected by frame 4 and try-this 1–2. (04 §6.1; `kvTransferTime`)
- **Misconception:** "A 384-expert model runs like a small model, since each token uses only 6 experts." → **Reality:** each expert sees only the tokens routed to it. With 64 users on one GPU, an expert gets 1 token per step: it is read from memory to do almost no math. Pooling 16 GPUs' users through wide expert parallelism gives it 16. Corrected by frames 6–7 and try-this 3. (04 §1.1, §6.2)
- **Misconception:** "Disaggregation is free throughput." → **Reality:** it costs a second pool to size and scale, KV traffic, and all-to-all traffic plus hot-expert imbalance for wide EP; the published gains are about meeting both latency targets (goodput), not raw tokens. Corrected by frames 2 and 10. (04 §6.1, §6.2)

## 3. Hook and intuition (final wording)
**Hook:** Why would you run the first second of a request on different GPUs from the rest of it?

In `batching` one long prompt stalled everyone's stream, and chunked prefill only spread the stall out. The deeper problem is that prefill and decode want different things. Prefill is compute-bound and likes big chunks of tokens; decode is memory-bound and needs short, steady steps, because users feel every gap between tokens. Users judge a server by two numbers, time to first token and time per output token, and on shared GPUs improving one tends to hurt the other. The measure that matters is goodput: how many requests per second meet both targets.

Disaggregation gives each phase its own GPUs. A prefill pool reads prompts in big compute-bound batches. When a prompt is done, its KV cache is shipped to a decode pool, which runs nothing but short decode steps. Each pool gets its own size, its own parallelism, even its own hardware. The price is the shipping and the bookkeeping: the KV cache has to cross a link, and there are now two pools to keep in balance as traffic shifts. Over a fast link the transfer is a few percent of the prefill it follows.

Mixture-of-experts models add a second reason to spread out. Each token uses only a few of hundreds of experts, so on one GPU every expert sees a token or two per step and is read from memory for almost no math. Wide expert parallelism spreads the experts over many GPUs and sends each token to its experts' GPU (an all-to-all exchange at every MoE layer), so tokens from all of those GPUs' users pool at each expert. That all-to-all is why the 72-GPU NVLink rack matters: it keeps every exchange on the fastest link.

## 4. Visual metaphor
Pools of GPUs drawn as `rack`s of `gpu`s, labeled "prefill pool" and "decode pool", with `request` bars (owner hues, letters printed) running on each pool's time axis. KV transfers are `flow` arrows with `carry: 'kv'` between pools, their thickness fixed (outlines and thickness never encode amounts); the link's speed is printed on the arrow. For the MoE half, a row of GPUs each holding a few expert `block`s; token chips (`token`, small, owner hues) fly along `flow`s (carry `activation`) to the GPU of their expert; a printed counter per expert ("tokens this step").

Followed item: request D (accent frame) in frames 1–5; expert 0's tokens in frames 6–8.

Terms introduced (one per frame): colocated (1), goodput (2), disaggregation / prefill pool / decode pool (3), KV transfer (4), pool ratio (5), tokens per expert (6), wide EP (7), redundant experts (8), NVLink domain (9), the bill (10). Terms assumed: from `batching`: chunked prefill, stall, step; from `prefill-decode`: TTFT, TPOT, memory-bound, compute-bound, intensity, ridge; from `parallelism`: expert parallelism, all-to-all, tensor parallelism, data parallelism; from `moe`: router, experts, top-k.

Glyphs used: `rack`, `gpu`, `request` (with the `owner` and `idle` options proposed in `serving-overview` and `batching`), `flow` (carry `kv`, `activation`), `block`, `token`, `shareBar` (memory per GPU).

Plain text labels: "What if?" is not used; "from `batching`'s branch" (frame 1); "all link speeds per GPU, each way: network 400 Gb/s = 50 GB/s; NVLink5 = 900 GB/s" (frame 4; the convention `cluster-topology` uses); "measured, dated" over published numbers (frames 5, 8, 9).

New glyphs proposed: none (the `request` options are proposed elsewhere).

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | One `gpu` with `batching`'s branch timeline in ms: A and C decoding, D's 4,096-token prompt arriving at step 3; two versions stacked, "no chunks" and "chunks of 512". Label "from `batching`'s branch". | The 290 ms step draws, then breaks into nine short ones; A's and C's ticks spread out in both. | On shared GPUs, prefill and decode take turns. D's long prompt stalled A and C for 290 ms, and even with chunks their steps stretched to 36 ms. | no chunks: one 289.9 ms step · chunks of 512: D's prompt takes eight steps of 36.2 ms; A sits through two of them, C through four |
| 2 | Two `clipLine`s, one per target: "time to first token" and "time per token", each with its target band; dots for each request turning ✓/✗ (`verdict`). | Moving the chunk size trades one gauge against the other. | Users judge two numbers: time to first token and time per token. Goodput counts only the requests that meet both targets, and on shared GPUs they fight. | goodput = requests per second meeting both (DistServe's definition) |
| 3 | Two `gpu`s: "prefill" (left) and "decode" (right). D's prefill runs on the left; A and C keep decoding on the right, their ticks evenly spaced. Visible line: "same time axis; D's bar is twenty of A's steps long". | D's prefill bar fills on the left GPU while ticks keep landing on the right at the old pace. | Disaggregation gives prefill and decode their own GPUs. D's long prompt runs on the prefill GPU, while A and C keep their 14.6 ms steps. | decode steps 14.59 ms throughout · D's prefill 289.8 ms on its own GPU |
| 4 | A `flow` (carry `kv`) from the prefill GPU to the decode GPU carrying D's KV tiles, written by the prefill forward pass; the link label switches between "network 400 Gb/s" and "NVLink". Label "all link speeds per GPU, each way: network 400 Gb/s = 50 GB/s; NVLink5 = 900 GB/s". | The KV dot travels; the transfer bar draws for each link. | Then D's keys and values must move to the decode GPU: 1.3 GB, about 27 ms over a 400 Gb/s network port and 1.5 ms over NVLink. | 4,096 × 327,680 B = 1.342 GB · 26.84 ms at 50 GB/s (9.3% of the prefill) · 1.49 ms at 900 GB/s (0.51%) |
| 5 | A `rack` of 16 GPUs (`cols: 8`, 16 px cells: 218 × 62 px) split into four 2-GPU prefill groups and one 8-GPU decode group, KV arrows from each prefill group to the decode group. Label "measured, dated". | The groups outline in turn; arrows draw. | Each pool gets its own size and parallelism. vLLM's best DeepSeek-R1 setup on GB200 was four 2-GPU prefill groups feeding one 8-GPU decode group. | 4 × 2 + 1 × 8 = 16 GPUs · 26.2K prefill / 10.1K decode tok/s per GPU at 2K in / 2K out (vLLM, 2026-02-03) |
| 6 | One `gpu` with 384 expert cells in a grid (small, no numbers) and 64 users' tokens; each token's 6 chosen experts flash. A counter on expert 0 (followed): "tokens this step". | Tokens route into the grid; expert 0's counter ticks to 1. | With 384 experts and 6 per token, 64 users give each expert about one token per step. Each expert is read from memory to do almost no math. | 64 × 6 ÷ 384 = 1 token per expert · expert intensity 3.6 FLOP/byte (FP4 weights) |
| 7 | Four `gpu`s in a row (24 experts each) and a plain mark "+ 12 others (EP 16)" (README lesson 18: sixteen 96 px GPUs do not fit); every GPU's 64 users' tokens fly (all-to-all) to the GPU of their experts. Expert 0's counter climbs. | Token chips stream between GPUs; counters rise. | Wide expert parallelism spreads the experts over 16 GPUs and sends each token to its expert's GPU. Pooling all their users gives each expert 16 tokens per step. | 64 × 16 × 6 ÷ 384 = 16 tokens per expert · 865 GB would not fit one 288 GB GPU; spread over 16 it is 54 GB of weights per GPU, 234 GB free · expert intensity 56.5 (ridge for FP4 on GB300: 1,875; compute-bound only from 699 tokens per expert) |
| 8 | DeepSeek's production layout as two `rack` groups: prefill "EP32 (4 nodes)" and decode "EP144 (18 nodes)", one decode GPU zoomed: "2 routed + 1 shared expert". Label "measured, dated". | The zoom opens on one decode GPU. | DeepSeek's 2025 production did exactly this: prefill spread over 32 GPUs, decode over 144, each decode GPU holding just two routed experts and one shared. | 256 routed experts + 32 redundant copies = 288 ÷ 144 = 2 per GPU · 8 active per token · Feb 2025 |
| 9 | A `rack` of 72 GPUs (`cols: 9`, 16 px cells: 244 × 218 px, as `cluster-topology` draws it) inside one NVLink frame vs two 8-GPU nodes joined by a thin network link (thickness fixed; speeds printed). | All-to-all dots run inside the rack; the cross-node version queues at the link. | Wide EP exchanges tokens at every MoE layer, so it wants every GPU on the fastest link. A 72-GPU NVLink rack keeps the whole exchange inside one domain. | NVL72: 72 GPUs, ~130 TB/s aggregate NVLink (NVIDIA's figure; it equals 72 GPUs × 1.8 TB/s, so it counts both directions) · GB300's 288 GB allowed EP 16 and 2.83× more tokens per GPU than GB200 at 27 tok/s/user (InferenceX, 2026-05) |
| 10 | A plain list beside both pictures, items appearing one by one: "two pools to size and rescale as traffic shifts", "KV transfer per request", "all-to-all every MoE layer", "hot experts → extra copies (EPLB)". | Items fade in. | The bill: two pools to balance, KV crossing a link for every request, and constant expert traffic. With short prompts, small models or slow links it may not pay. | DistServe (2024, printed on screen): 7.4× more requests or 12.6× tighter targets (reported) · Splitwise (2023, printed on screen): 1.4× throughput at 20% lower cost (reported) |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter, all ≤ 30 words and ≤ 2 sentences.

## 6. Toy
Two panels. "Ship the KV" (running example) and "Feed the experts" (DeepSeek-V4-Pro on GB300). A visible line: "Floors from bytes, bandwidth and FLOPs; real transfers add start-up latency, and real all-to-all adds communication time not modeled here."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `prompt` | Prompt length | Slider (snapped) | 128, 512, 4,096, 8,192, 32,768, 131,072 tokens | 4,096 | — |
| `link` | Link between pools (per GPU, each way) | Preset chips | NVLink5 900 GB/s · network 800 Gb/s (100 GB/s) · network 400 Gb/s (50 GB/s) | 400 Gb/s | `hardware.json` link entries (§8, proposed numeric keys) |
| `kvFormat` | KV cache | Toggle | BF16 (327,680 B/token) / FP8 (163,840) | BF16 | — |
| `ep` | GPUs sharing the experts (EP size) | Slider (snapped) | 1, 8, 16, 32, 72 | 16 | — |
| `usersPerGpu` | Users per GPU | Preset chips | 16 · 64 · 256 | 64 | — |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| KV to ship | `kvCacheBytes({ bytesPerToken, tokens: prompt })` (`kv-cache`) | GB, 3 decimals |
| Transfer time; prefill time (H200, FP8; one forward pass); transfer as % of prefill (basis: one request's prompt) | `kvTransferTime(prompt, kvBytesPerToken, linkBytesPerS)`; `stepTime({ …running example…, tokens: prompt, seqs: 0, context: 0 })`; ratio | ms; ms; % |
| Same prompt with MLA-sized KV (DeepSeek-V3, 70,272 B/token): KV and transfer only | `kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 })` → `kvTransferTime` | GB; ms (no ratio: a different model) |
| Experts per GPU, weights per GPU, free memory per GPU | `384 / ep`; `865e9 / ep`; `freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: ep })` | count; GB; GB ("does not fit" when negative); a weights segment under 18 px (12 GB of 288 at EP 72) is printed as text beside the memory bar (README lesson 19) |
| Tokens per expert per step; expert multiply intensity vs ridge; tokens per expert needed to be compute-bound | `tokensPerExpert({ usersPerGpu, epSize: ep, expertsActive: 6, expertsTotal: 384 })`; `arithmeticIntensity(matmulCost({ m: tokens, k: 7168, n: 3072, bytesPerElem: bytesPerElement('nvfp4') }))` vs `ridgePoint({ peakTflops: 15000, bandwidthTBps: 8 })`; `tokensToComputeBound({ peakTflops: 15000, bandwidthTBps: 8, bytesPerElem: 0.5625, k: 7168, n: 3072 })` (all from `math/roofline.js`) | 2 decimals; FLOP/byte; tokens |

**Try this** (each leads to a named insight)
1. Network 400 Gb/s, BF16 KV. Slide the prompt 512 → 4,096 → 131,072: transfer 3.36 → 26.84 → 859 ms, prefill 36.2 → 289.8 → 9,272 ms, ratio 9.26% every time. Now slide down to 128: transfer 0.84 ms, prefill 15.04 ms, ratio 5.58%; a note beside the stop reads "below 217 tokens prefill is one weight read, so the ratio is smaller here; real transfers add a fixed start-up cost the toy does not model, which is why short prompts gain least". → **Insight: past the crossover, the transfer-to-prefill ratio is set by the model and the link, not the prompt.** Long prompts do not make disaggregation worse in this model; slow links do.
2. Keep 4,096 tokens and switch the link: 400 Gb/s 9.26% · 800 Gb/s 4.63% · NVLink 0.51%; then FP8 KV halves each (4.63%, 2.32%, 0.26%). The MLA readout: 0.288 GB, 5.76 ms at 400 Gb/s. → **Insight: a fast link or a smaller KV makes the split nearly free;** this is one more reason MLA and FP8 KV matter (`kv-compression`, `quantization`).
3. Feed the experts, 64 users per GPU: EP 1 → "does not fit" (865 GB vs 288 GB) with 1 token per expert; EP 8 → 8 tokens per expert, 108 GB of weights per GPU; EP 16 → 16, 54 GB; EP 72 → 72, 12 GB, 276 GB free. → **Insight: wide EP both shrinks each GPU's share of the weights and pools users' tokens at each expert,** so every expert does more math per byte read. Even at EP 72 with 256 users per GPU (288 tokens per expert) the intensity is 903, still under the FP4 ridge of 1,875, which an expert reaches only at 699 tokens.

**`math/serving.js`** additions used here (signatures in `prefill-decode` §6):
```js
kvTransferTime(promptTokens, kvBytesPerToken, linkBytesPerSecond) → seconds   // promptTokens · kvBytesPerToken / link
tokensPerExpert({ usersPerGpu, epSize, expertsActive, expertsTotal }) → number // usersPerGpu · epSize · expertsActive / expertsTotal
```
Worked examples (scratch implementation, 2026-10-07):
```
kvTransferTime(4096, 327680, 50e9)   → 0.02684 s (1.342 GB)     prefill (H200, FP8, 4,096 tokens) 289.8 ms → 9.26%
kvTransferTime(4096, 327680, 100e9)  → 13.42 ms (4.63%)          kvTransferTime(4096, 327680, 0.9e12) → 1.49 ms (0.51%)
kvTransferTime(4096, 163840, 50e9)   → 13.42 ms (4.63%)          (FP8 KV)
kvTransferTime(512, 327680, 50e9) → 3.36 ms vs prefill 36.2 ms ; 131072 → 858.99 ms vs 9,272.4 ms   (9.26% both)
below the crossover (prefill is one weight read): 100 → 0.66 vs 14.94 ms (4.39%) · 128 → 0.84 vs 15.04 ms (5.58%) · 200 → 1.31 vs 15.30 ms (8.57%) · 217 → 1.42 vs 15.36 ms (9.26%)
kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 }) → 70,272 ; 4,096 tokens → 0.288 GB, 5.76 ms at 50 GB/s
tokensPerExpert({ usersPerGpu: 64, epSize: 1,  expertsActive: 6, expertsTotal: 384 }) → 1      (expert intensity 3.6)
                                         epSize: 8 → 8 (28.3) · 16 → 16 (56.5) · 32 → 32 (112.1) · 72 → 72 (247.7)
tokensPerExpert({ usersPerGpu: 256, epSize: 72, … }) → 288 (intensity 903.1) ; usersPerGpu 16, epSize 16 → 4
  intensity = arithmeticIntensity(matmulCost({ m: tokens, k: 7168, n: 3072, bytesPerElem: 0.5625 })) (V4-Pro: hidden 7,168, expert width 3,072)
tokensToComputeBound({ peakTflops: 15000, bandwidthTBps: 8, bytesPerElem: 0.5625, k: 7168, n: 3072 }) → 698.7
freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus }) → EP 1 −577 GB · 8 179.9 · 16 233.9 · 32 261.0 · 72 276.0 (weights 865 / 108.1 / 54.1 / 27.0 / 12.0 GB)
ridgePoint({ peakTflops: 15000, bandwidthTBps: 8 }) → 1,875 (GB300, FP4 dense, reported)
DeepSeek decode: (256 + 32) / 144 = 2 routed experts per GPU ; tokensPerExpert({ usersPerGpu: 16, epSize: 144, expertsActive: 8, expertsTotal: 256 }) → 72
```
Reproducer (run once `math/serving.js` exists):
```
node -e "import('./math/serving.js').then(m => { for (const bw of [50e9, 100e9, 0.9e12]) for (const t of [100, 128, 200, 217, 512, 4096, 131072]) console.log(bw, t, m.kvTransferTime(t, 327680, bw), m.stepTime({ activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, dModel: 8192, actBytesPerElem: 1, kvBytesPerToken: 327680, peakTflops: 1979, bandwidthTBps: 4.8, tokens: t, seqs: 0, context: 0 }).timeS); for (const ep of [1, 8, 16, 32, 72]) console.log(ep, m.tokensPerExpert({ usersPerGpu: 64, epSize: ep, expertsActive: 6, expertsTotal: 384 }), m.freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: ep })) })"
```
Tests to write first: the table above; `kvTransferTime` linear in prompt and in KV bytes; the transfer/prefill ratio is independent of the prompt above the crossover and smaller below it; `tokensPerExpert` equals `usersPerGpu · k / E` at EP 1; inputs not mutated.

## 7. Show me the math
```tex
\htmlClass{hl-kv}{t_{\text{transfer}}} = \frac{\text{prompt}\cdot \text{KV bytes/token}}{\text{link bandwidth}},
\qquad
t_{\text{prefill}} \approx \frac{2\,N_{\text{active}}\cdot\text{prompt}}{\text{peak FLOP/s}}
\;\Rightarrow\;
\frac{\htmlClass{hl-kv}{t_{\text{transfer}}}}{t_{\text{prefill}}} = \frac{\text{KV bytes/token}\cdot \text{peak}}{2\,N_{\text{active}}\cdot\text{link}}
```
```tex
\text{tokens per expert per step} = \frac{\text{users per GPU}\cdot \text{EP}\cdot k}{E},
\qquad
\text{weights per GPU} \approx \frac{W}{\text{EP}}\ \ (\text{replicated non-expert weights ignored})
```
```tex
\text{goodput} = \max\{\text{request rate} : \text{TTFT} \le \text{target}_1 \text{ and } \text{TPOT} \le \text{target}_2\}
```
Shapes: one expert's weights are a few `[7168 × 3072]` matrices in V4-Pro; with t tokens each multiply is `[t × 7168] · [7168 × 3072]`, intensity from `matmulCost` (≈ `2t / bytes per weight` while t is small). Color links: `hl-kv` = the KV `flow` arrows (`--sem-memory`).

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| P/D disaggregation: prefill and decode on different pools, KV moved between them; gains are no prefill-induced stalls and per-pool parallelism; costs are KV transfer and two pools to size | `serving.json/disaggregation.note` *(proposed)* | 04 §6.1 |
| DistServe (2024): goodput framing; 7.4× more requests or 12.6× tighter SLOs · Splitwise (2023): 1.4× throughput at 20% lower cost, or 2.35× at the same cost | `serving.json/distserve.goodput_gain = 7.4`, `.slo_gain = 12.6`; `serving.json/splitwise.throughput_gain = 1.4`, `.cost_cut_pct = 20`, `.throughput_gain_same_cost = 2.35` (all `confirmed` from the papers' abstracts) | 04 §1.2, §6.1 |
| NIXL moves KV over NVLink and InfiniBand / RoCE RDMA, with TCP fallback; used by Dynamo, llm-d, vLLM, Ray Serve LLM | `serving.json/nixl.transports` *(proposed, confirmed)* | 04 §6.1 CONFIRMED |
| vLLM on GB200 (DeepSeek-R1, NVFP4, Feb 2026): best layout 4 prefill instances × 2 GPUs + 1 decode instance × 8 GPUs; 26.2K prefill / 10.1K decode tok/s per GPU at 2K/2K | `serving.json/vllm-gb200-dsr1.prefill_instances = 4`, `.prefill_gpus_each = 2`, `.decode_gpus = 8`, `.prefill_tok_s_gpu = 26200`, `.decode_tok_s_gpu = 10100` *(proposed)* | 04 §6.1, §6.3 CONFIRMED |
| Disaggregation helps most for MoE with EP: one compute-bound prefill in the EP group delays the whole group's forward pass (vLLM) | `serving.json/vllm-large-scale.disagg_moe_note` *(proposed)* | 04 §6.1 CONFIRMED |
| DeepSeek V3/R1 production (Feb 2025): prefill EP32 over 4 nodes (9 routed + 1 shared expert per GPU); decode EP144 over 18 nodes (2 routed + 1 shared per GPU); 8 of 256 experts active | `serving.json/deepseek-v3-production.prefill_ep = 32`, `.decode_ep = 144`, `.decode_experts_per_gpu = "2 routed + 1 shared"` *(proposed)* | 04 §6.3 CONFIRMED |
| Wide-EP = data-parallel attention + expert-parallel MoE ("a single set of experts shared across ranks"); helpers DeepEP, DeepGEMM, EPLB (redundant experts), two-batch overlap | `serving.json/vllm-large-scale.wide_ep_note` *(proposed)* | 04 §6.2 CONFIRMED |
| SGLang open reproduction (May 2025): 96 H100s, PD + EP with 288 experts (256 + 32 redundant): up to 5× output throughput over TP16; EPLB 1.49× prefill / 2.54× decode | `serving.json/sglang-large-ep.output_gain_vs_tp16 = 5`, `.eplb_prefill_gain = 1.49`, `.eplb_decode_gain = 2.54` *(proposed)* | 04 §6.3 CONFIRMED |
| DeepSeek-V4-Pro: 384 routed (+1 shared) experts, 6 per token; ≈ 865 GB as shipped | `models.json/deepseek-v4-pro.experts_total`, `.experts_active`; `.checkpoint_gb = 865` *(proposed, `reported`)* | 04 §8 |
| GB300 NVL72: 72 GPUs in one NVLink domain, ~130 TB/s aggregate; InferenceX attributes GB300's 2.83× per-GPU throughput over GB200 at 27 tok/s/user to its 288 GB allowing EP 16 and larger prefill batches (V4-Pro, 2026-05) | `hardware.json/gb300-nvl72.rack_nvlink_tbps`; `hardware.json/b300.hbm_gb`; `serving.json/inferencex-v4-pro-gb300.per_gpu_gain_vs_gb200 = 2.83` *(proposed)* | 04 §6.4 CONFIRMED |
| Link speeds per GPU, each way (the `cluster-topology` convention): NVLink5 900 GB/s (NVIDIA quotes 1.8 TB/s counting both directions, B200/B300); network ports 400 Gb/s = 50 GB/s in the H100 era and 800 Gb/s = 100 GB/s (ConnectX-8) on Blackwell, so NVLink is 9× the network on Blackwell | `hardware.json/b200.nvlink_gb_s_each_way = 900`, `hardware.json/network-400g.gb_s_each_way = 50`, `hardware.json/network-800g.gb_s_each_way = 100` *(proposed numeric keys; settled naming `<chip>.nvlink_gb_s_each_way`)* | 03 §2.1 |

## 9. Takeaways
1. On shared GPUs prefill and decode fight over every step, so improving time to first token tends to hurt time per token; disaggregation gives each phase its own pool and judges the result by goodput (frames 1–3).
2. The KV cache must cross a link, but past the 217-token crossover its transfer time scales with the prompt just like prefill, so the cost is a fixed fraction set by the model and the link: 0.5% over NVLink, 9% over a 400 Gb/s port for the running example (frames 4–5, try-this 1–2).
3. Wide expert parallelism spreads a MoE's experts over many GPUs so each expert sees enough tokens per step and each GPU holds less; it needs fast all-to-all, which is why NVL72 racks host it, and it adds pools, traffic and imbalance to manage (frames 6–10, try-this 3).

## 10. Next and go deeper
Next: `serving-calculator` · Related: `parallelism` (EP and all-to-all), `cluster-topology` (NVLink domains), `batching`, `moe`.

Go deeper (brief 04 §9.2, 05 §1.2): DeepSeek, "Inference System Overview" (https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) · LMSYS, "Large-scale expert parallelism" (https://lmsys.org/blog/2025-05-05-large-scale-ep/) · Hao AI Lab, "Throughput is not all you need" (DistServe blog; the brief lists it as prior, URL to check: https://hao-ai-lab.github.io/blogs/distserve).

## 11. Key-frame sketch
Frame 4 end state, desktop width; numbers from the `kvTransferTime` rows in §6.
```text
Disaggregated serving           step 4 / 10   [<] [Play] [>]
 prefill GPU (H200)                decode GPU (H200)
 D |===== prefill 289.8 ms =====|  A |#|#|#|#  14.6 ms steps
                                   C |#|#|#|#|#|#
          KV of D: 1.342 GB  ------------------>  D |#|#|#
 network 400 Gb/s (50 GB/s): 26.84 ms (9.26% of prefill)
 NVLink5 (900 GB/s each way): 1.49 ms (0.51% of prefill)
 ratio is the same for any prompt past 217 tokens
```

## 12. Open questions for the reviewer
**Data-pass keys**
- Numeric link keys (settled name `<chip>.nvlink_gb_s_each_way`; `hardware.json/b200.nvlink_gb_s_each_way = 900`), plus each-way network-port keys (`network-400g.gb_s_each_way = 50`, `network-800g.gb_s_each_way = 100`); all `serving.json` keys in §8.

**Judgment calls**
- **Prefill on one H200.** The transfer ratio uses the running example's single-GPU prefill time. A real prefill instance spans several GPUs; if each GPU ships its own shard over its own NIC, the ratio is unchanged, which is why the page states the ratio rather than a cluster-level time.
- **No start-up latency.** `kvTransferTime` has no fixed per-transfer cost, so the toy cannot show why very short prompts gain little (brief 04 §6.1). The toy's visible line and frame 10's caption say so rather than inventing a constant.
- **Expert intensity** uses `gpu-primer`'s `matmulCost` on one expert's 7,168 × 3,072 multiply at NVFP4's 4.5 bits per element (activations counted at the same width, as vLLM's NVFP4 dispatch quantizes them), and ignores attention and shared experts.
- **Goodput is defined, not simulated.** A request-rate simulation with SLO percentiles would need arrival randomness; frame 2 states the definition and the DistServe numbers are shown as the paper's claims (confirmed from its abstract).

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled:** links are per GPU, each way (NVLink5 900 GB/s; network 50 / 100 GB/s), keyed `<chip>.nvlink_gb_s_each_way`; NVIDIA's 130 TB/s rack figure is printed with its both-directions arithmetic.
- **Settled:** the fixed transfer/prefill ratio holds only past the 217-token crossover; the slider has a 128-token stop showing the smaller ratio below it, and the short-prompt penalty is attributed to the unmodeled start-up cost.
- **Settled:** prefill on one H200 is accepted for the ratio (the page states a ratio, not a cluster time).
- **Settled:** cache size comes from `kvCacheBytes` (`kv-cache`).
- Fixed: A and C sit through two and four 36.2 ms steps (not eight); frame 1's numbers say D's prompt takes eight.
- Applied: frame 7 draws four GPUs plus "+ 12 others (EP 16)" (lesson 18); `rack` cell sizes for frames 5 and 9; the 865 GB fit note moves to frame 7; frame 3's same-axis line; frame 2 uses two `clipLine`s; DistServe and Splitwise years on screen; the EP 72 weights sliver printed as text; takeaway 2 scoped to past the crossover.

- Data pass 2026-10-07: DistServe and Splitwise facts are \`confirmed\` (primary abstracts), not \`reported\`; no numbers changed.
