# Serving a 1T model (`serving-calculator`)

Track: serving · Section: serving · Prereqs: disaggregation, quantization, prefix-caching, speculative-decoding
Status: approved (expert review)
Sources: 04 §1.2, §1.3, §1.4, §2.4, §3.6, §5, §6.3, §6.4, §7.4, §7.5, §8 (the whole worked example), §9.1 (cost-calculator toy), §10 items 3–4; 03 §1.4. Every number on this page comes from a function defined in an earlier lesson (§6 lists which) or from a dated `data/*.json` fact; estimates are labeled "floor" or "estimate" on screen (spec §7). Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Worked example (spec §3.3): **DeepSeek-V4-Pro** (1.6T total / 49B active, 61 layers, d_model 7,168, 384 routed experts with 6 per token, FP4 experts + FP8 everything else, ≈ 865 GB as shipped) **on GB300 NVL72** (288 GB and 8 TB/s per GPU, ~5 PF dense FP8, ~15 PF dense FP4), **16 GPUs per replica sharing the experts (EP 16)**, **8,192 tokens in and 1,024 out** (the InferenceX test's conditions). Second preset: `prefill-decode`'s Llama-3.1-70B on one H200.

V4-Pro's KV per token is a **range, 4,000–12,000 bytes**: formula-derived from the config, with a layer mix that sources disagree on (spec §7 known conflict; `models.json/deepseek-v4-pro.kv_bytes_per_token`, `reported`). It cannot come from `kvBytesPerToken` / `kvBytesPerTokenMla`, which do not model compressed attention. Every KV-dependent number is shown for both ends, low first, and a visible line says why.

## 1. Learning objective
After this page you can size a deployment of a ~1T-parameter MoE yourself: how many GPUs its weights need (frames 2–3), how much KV one user costs at 8K and at 1M tokens and so how many users fit (frames 4–5), the fastest a user can be served (frame 6), how many users a latency target allows in the ideal (frame 7), how far measured systems sit from that ideal and why (frame 8), and what a million tokens costs, including why output is priced above input (frames 9–10, try-this 1–4).

## 2. Misconceptions to correct
- **Misconception:** "A model's size tells you how many GPUs it needs." → **Reality:** weights set a minimum (865 GB needs at least 4 GB300s), but the KV cache decides how many users fit once it is loaded: at 8K tokens, 6,345 users' KV fit beside the weights on each of 16 GPUs; at 1M tokens, 58 (or 19 at the high KV estimate). Corrected by frames 2–5 and try-this 1. (04 §8.1–8.3)
- **Misconception:** "The roofline tells you the throughput you will get." → **Reality:** it gives a floor on time and a ceiling on throughput. At 27 tok/s per user the ideal is 51,020 tok/s per GPU; InferenceX measured 6,182 (its counting convention is unstated: if it includes input tokens, output-only is about 690), and its 27 tok/s per user implies 37 ms per token, 5.48× the time to read the weights alone (6.76 ms). The gap is all-to-all traffic, attention, expert imbalance, prefill GPUs and scheduling. Corrected by frame 8 and try-this 2. (04 §8.4)
- **Misconception:** "Output tokens cost more because generating is harder math." → **Reality:** each output token needs no more math than an input token. Input tokens are processed many per weight read, at high intensity, and often come from the cache; output tokens come one per step per user, and each user's KV stays in memory for the whole answer; and output speed is what users feel, so providers keep latency headroom (smaller batches) for it. DeepSeek's production nodes processed about 5× more input than output tokens per second, close to the 3–5× price ratios. Corrected by frame 10 and try-this 3. (04 §7.4, ANALYSIS labeled)
- **Misconception:** "Speculative decoding always cuts cost." → **Reality:** in this model MTP speeds each user up 1.73× at 128K context (memory-bound), but at the 8K target's 1,889 ideal users per GPU, where decode is compute-bound, MTP slows it down: 0.9 of the plain speed. Corrected by try-this 4. (04 §4.1; `batchSpeedup`)

## 3. Hook and intuition (final wording)
**Hook:** How many GPUs does it take to serve a 1.6-trillion-parameter model, and what does a million output tokens cost?

Start with the weights. DeepSeek-V4-Pro ships its experts in FP4 and everything else in FP8, about 865 GB. A GB300 holds 288 GB, so the weights alone need at least four GPUs. In practice one copy of the model, a replica, spreads its experts over 16 GPUs (`disaggregation`'s wide expert parallelism), leaving each GPU 54 GB of weights and 234 GB for users.

Then the users. Each one's KV cache lives on the GPU for the whole answer. V4-Pro's compressed attention keeps that small, about 4 to 12 KB per token, so at 8K tokens a user costs 37–111 MB and thousands fit. At 1M tokens a user costs 4–12 GB and only 19–58 fit per GPU. Once memory is full of KV, every decode step has to read most of the GPU's 288 GB, which in this floor caps each user near 28 tokens per second (sparse attention reads less, so real long-context models can do better).

Speed and cost come last. The floor from `prefill-decode` says one user could get 148 tokens per second; a 27 tokens-per-second target leaves room, in the ideal, for about 1,900 users per GPU. Measured systems get far less, because communication, attention and scheduling eat most of each step. InferenceX measured 6,182 tokens per second per GPU at 27 per user, which at $2.65 per GPU-hour is $0.12 per million tokens, input and output counted together. That is far below DeepSeek's list price ($0.66 input, $1.98 output off-peak), and the gap between input and output prices follows from everything in this track: input tokens are read many at a time and often come from cache, while output tokens come one per step and hold memory while they do.

## 4. Visual metaphor
The followed GPU drawn large as a `gpu` glyph, beside a `rack` of 16 small cells (`cols: 8`, 16 px cells: 218 × 62 px) labeled "one replica, EP 16" (README lesson 18: sixteen 96 px `gpu` glyphs do not fit). Each GPU's memory is a `shareBar` with three named parts: `weights`, `KV`, `free`. The followed item is **one GPU** (`G.selectionMark`, the same GPU in every frame), whose bar is drawn large beside the rack. Users are `token`-style chips with request letters A–D (hues 1–4) standing for "many more"; the count is a printed number, never an implied crowd (README lesson 18). Speed uses `prefill-decode`'s `stepBar` (`weights read`, `KV read`, `activations` in the reading row; `arithmetic` in its own row). A `curvePlot` (proposed in `prefill-decode`) plots per-user speed against users per GPU, with the floor curve solid and the two InferenceX measurements as separate markers labeled "measured". A `bars` chart of $ per M tokens (P4-R17) ends the page.

Every readout carries its conditions in a visible footer line: "DeepSeek-V4-Pro · GB300 NVL72 · EP 16 · 8K in / 1K out (context 1,000,000 when the 1M chip is on) · KV 4 KB/token (low) – 12 KB (high) · floor = bytes and FLOPs only, output tokens".

Terms introduced (one per frame): replica (2), weights per GPU (3), KV per user (4), memory-bound ceiling (5), latency floor (6), users at a target (7), measured gap (8), cost per million tokens (9), output-vs-input price (10). Terms assumed from prereqs: everything in `prefill-decode` (step time, TPOT, tokens/s per user and per GPU), `quantization` (FP4/FP8 bytes), `disaggregation` (EP, all-to-all, pools), `prefix-caching` (hit rate), `speculative-decoding` (MTP, `batchSpeedup`), `paged-attention` (KV memory per user).

Glyphs used: `rack` (with `G.selectionMark` on the followed GPU), `gpu`, `shareBar` (the memory bar), `stepBar` (frames 5–6, P4-R8), `token` (with `owner`), `curvePlot` (with `refY`), `bars` (frames 9–10, P4-R17), `block` (model card in frame 1).

Plain text labels: the conditions footer (all frames); "estimate" / "floor" / "measured, dated" tags on every number; "sparse attention reads less than the whole cache; this floor assumes it all" (frame 5); "InferenceX may count input tokens in tok/s/GPU; the source does not say" (frames 8–9); "list prices read 2026-10-07" (frame 9).

New glyphs proposed: none.

## 5. Animation script
Numbers: §6 functions on the worked example; KV at the low end (4,000 B/token) with the high end (12,000) printed beside it.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | A `block` model card: "DeepSeek-V4-Pro · 1.6T total · 49B active · 384 experts, 6 per token · FP4 experts + FP8 rest · ≈ 865 GB (reported)". Footer with conditions. | Card fields type in. | The worked example: a 1.6-trillion-parameter MoE that uses 49 billion parameters per token. Shipped in 4- and 8-bit formats, it weighs about 865 GB. | 1.6T / 49B · 865 GB (vs 3.2 TB in BF16, 1.6 TB in FP8) |
| 2 | One `gpu` "GB300, 288 GB" with the 865 GB weight bar overflowing; three more GPUs appear until it fits. Plain label: "replica = one copy of the model spread over several GPUs". | GPUs slide in one at a time; the bar splits across them. | A GB300 holds 288 GB; three hold 864, a gigabyte short of the weights and with no room for anything else. The weights alone need four. | ⌈865 ÷ 288⌉ = 4 (3 × 288 = 864 GB; the checkpoint size is reported, ≈ 865) · GB200 NVL72 (186 GB per GPU, the rack total over 72): 5 |
| 3 | The replica `rack` of 16 GPUs; the followed GPU's bar large: `weights` 54.1 GB, `free` 234 GB. | The four GPUs become sixteen; each bar's weight part shrinks. | In practice the replica spreads the experts over 16 GPUs, as in `disaggregation`. Each GPU now holds 54 GB of weights and has 234 GB left for users. | 865 GB ÷ 16 = 54.1 GB · 288 GB − 54.1 GB = 234 GB free (upper bound: activations and runtime buffers not subtracted) |
| 4 | User chips A–D enter the followed GPU; its `KV` part grows by 36.9 MB per user, with a running printed total "users × 36.9 MB = … GB" beside the bar until the segment reaches 18 px; a counter "users that fit". Visible line: "KV per token is a range: 4 KB (low) – 12 KB (high), formula-derived, layer mix uncertain". | The counter runs up to 6,345 while the KV part fills the free space. | Each user's KV cache stays on the GPU for the whole answer. At 8K tokens a user costs 37 to 111 MB, so thousands fit beside the weights. | 4,000 × 9,216 = 36.9 MB → 6,345 users · 12,000 × 9,216 = 111 MB → 2,115 |
| 5 | Context jumps to 1M tokens: each chip's KV part becomes 4 GB; the counter drops to 58. The `stepBar` for 58 users: `KV read` 29 ms + `weights read` 6.76 ms. Label: "sparse attention reads less than the whole cache; this floor assumes it all, so 28 tok/s is a lower bound on per-user speed at 1M". | The KV part refills with far fewer, far larger blocks; the step bar stretches. | At a million tokens a user costs 4–12 GB, so only 19–58 fit per GPU. Each step then re-reads a full GPU, capping users near 28 tokens per second. | 4 GB → 58 users, step 35.9 ms, 27.9 tok/s each, 1,618 tok/s per GPU · 12 GB → 19 users, 35.3 ms, 28.3 |
| 6 | Back to 8K. One user on the followed GPU: the `stepBar` is just `weights read` 6.76 ms. | The bar draws; a readout "148 tok/s per user, at best". | The fastest one user can go: each step must read the GPU's 54 GB of weights, which takes 6.76 ms. That is at most 148 tokens per second. | 54.1 GB ÷ 8 TB/s = 6.76 ms → 148 tok/s (floor) |
| 7 | `curvePlot`: x = users per GPU, y = tokens/s per user (floor), with a horizontal reference line at 27 (`curvePlot` `refY`). The curve crosses 27 at 1,889 users; a readout shows tokens/s per GPU at that point. | The marker slides along the curve from 1 user to the crossing. | A target of 27 tokens per second gives each step 37 ms. In the ideal, 1,889 users fit in that step before the math, not memory, runs out. | 1,889 users · step 37 ms (compute-bound, FP8 math) · 51,020 tok/s per GPU (floor) |
| 8 | Two markers labeled "measured, dated" join the plot: 6,182 tok/s per GPU at 27 tok/s/user and 11,056 at 13.1. A plain list beside them: "all-to-all · attention · expert imbalance · prefill GPUs · scheduling". Label: "InferenceX may count input tokens in tok/s/GPU; the source does not say. If it does, output-only throughput is about 690 (6,182 divided by 9 at 8K in / 1K out) and the gap to the floor is 74.3×, not 8.25×". Visible line: "the floor counts output tokens only; compare the two as a range, not a ratio". | The measured markers drop in far below the floor curve; the list fades in item by item. | Measured, it is far less: InferenceX got 6,182 tokens per second per GPU. Its 27 tokens per second per user implies 37 ms per token, 5.48 times the weight-read floor. | floor 51,020 output tok/s/GPU vs measured 6,182: between 8.25× (if output-only) and 74.3× (if input + output) · implied: 1 ÷ 27 = 37.0 ms per token vs 6.76 ms = 5.48× · the listed causes are the brief's estimate |
| 9 | A `bars` chart (P4-R17) of $ per M tokens, with the readout "$2.65 per GPU-hour ÷ 6,182 tok/s → $0.12 per M tokens"; beside it GB200: $0.28; and DeepSeek's off-peak list price for V4-Pro: $0.66 input / $1.98 output. Label: "list prices read 2026-10-07". | The division animates as a plain readout; bars draw. | A GPU-hour's price over the tokens it makes is the cost: about 12 cents per million input and output tokens on GB300. DeepSeek lists 66 cents in, $1.98 out. | `costPerMillion(2.65, 6182)` = $0.119 per M tokens, input and output together (InferenceX's convention) · `costPerMillion(2.21, 2189)` = $0.280 · list $0.66 / $1.98 (off-peak) |
| 10 | A `bars` chart (P4-R17) of two bars: "input tokens per second per node 73.7K" vs "output 14.8K" (DeepSeek production, Feb 2025), ratio 4.98×; under them price ratios: Anthropic 5×, DeepSeek V4-Pro 3×. A small floor readout: input ceiling 51,020 tok/s per GPU vs decode at 128K 11,433 (4.46×). Visible line: "different hardware and units (per node vs per GPU); only the ratios are being compared, each as tokens per second on one basis (forward passes, input tokens vs output tokens)". | Bars draw; ratios type in. | Output costs more because a GPU makes far fewer output tokens per second than it reads input. DeepSeek's nodes read five times more than they wrote. | 73.7K ÷ 14.8K = 4.98 · prices: Anthropic output 5× input, DeepSeek V4-Pro 3× · floor at 128K + 8K: 51,020 ÷ 11,433 = 4.46 |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter, all ≤ 30 words and ≤ 2 sentences.

## 6. Toy
"Size a deployment." Every output group shows its formula with the numbers substituted (brief 05 §2 item 15, "Check my work", pinned below), and the conditions footer updates with the controls. A visible line above the controls: "Floors from bytes and FLOPs only; real systems are slower (frame 8). V4-Pro's KV per token is an estimate range; both ends are shown."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `model` | Model | Preset chips | DeepSeek-V4-Pro · Llama-3.1-70B | V4-Pro | `models.json/deepseek-v4-pro.*`, `llama-3.1-70b.*` |
| `weights` | Weight format | Preset chips | as shipped (V4: FP4 experts + FP8, 865 GB; chip note "FP8 math rate (conservative; experts run FP4)") · BF16 · FP8 · NVFP4 | as shipped (V4) / FP8 (Llama) | bits from `bitsPerElement` (`math/roofline.js`) |
| `hw` | GPU | Preset chips | H200 (141 GB nominal) · B200 (180 GB usable of 192 nominal) · GB200 NVL72 (186 GB: the rack total over 72) · GB300 NVL72 (288 GB nominal) | GB300 NVL72 | `hardware.json` `h200`, `b200` (usable), `gb200-nvl72`; `gb300-nvl72` per-GPU values (288 GB nominal; no usable figure is in data); memory read through `hbmFor(entry)` → `{ bytes, basis }`, the basis word printed with the number |
| `gpus` | GPUs per replica | Slider (snapped) | 1, 2, 4, 8, 16, 32, 72 | 16 (V4) / 1 (Llama) | — |
| `context` | Tokens per user (in + out) | Preset chips | 8K + 1K (9,216) · 128K + 8K (139,264) · 1M = 1,000,000 (V4-Pro's configured context; V4 only) | 8K + 1K | — |
| `kvEnd` | V4 KV estimate | Toggle | low 4,000 B/token / high 12,000 B/token | low | `models.json/deepseek-v4-pro.kv_bytes_per_token` |
| `kvFormat` | KV cache (Llama) | Toggle | BF16 / FP8 | BF16 | — |
| `users` | Users per GPU | Slider (snapped) | 1, 4, 16, 64, 256, 1,024, …, "most at target", "max that fits" | "most at target" | — |
| `target` | Target speed per user | Preset chips | 13.1 · 27 tok/s | 27 | InferenceX operating points: `serving.json/inferencex-v4-pro-gb300.max_throughput_tok_s_user = 13.1` and `.interactivity_tok_s_user = 27`. There is no V4-Pro point at 90, so there is no 90 chip (P4-R19) |
| `price` | $ per GPU-hour | Preset chips | 2.65 (GB300) · 2.21 (GB200) · 2.00 (DeepSeek's H800 accounting) | 2.65 | `serving.json/inferencex-v4-pro-gb300.gpu_hour_usd`, `…-gb200.gpu_hour_usd`, `deepseek-v3-production.gpu_hour_usd` |
| `hitRate` | Prompt cache hit rate | Preset chips | 0% · 56.3% (DeepSeek 2025) | 0% | `serving.json/deepseek-v3-production.kv_hit_rate_pct` |
| `mtp` | MTP speculation (k = 1, α = 0.85) | Toggle | off / on | off | `serving.json/deepseek-v3-mtp.acceptance_pct` (low end) |

"As shipped" uses the reported 865 GB and FP8 math (attention and shared layers are FP8; using the FP4 rate is the NVFP4 chip). NVFP4 uses the GPU's FP4 rate and is disabled on the H200.

**Live outputs** (grouped as the page shows them; each with its source function)
| Output | Formula / `math/` function (lesson) | Units / format |
|---|---|---|
| **Fit:** weights total, GPUs needed for weights alone, weights per GPU, free per GPU | `weightBytes({ params, bitsPerParam: bitsPerElement(f) })` with `f` ∈ `bf16`, `fp8_e4m3`, `nvfp4` (exact `FORMATS` keys), or 865e9 shipped; `minGpusForWeights(w, hbm)`; `w / gpus`; `freeHbmPerGpu({ hbmBytes, weightBytes, gpus })` (`prefill-decode`, `quantization`) | `formatBytes`; `formatInt`; "does not fit" when negative |
| KV per user; users that fit per GPU | `kvCacheBytes({ bytesPerToken, tokens: context })` (`kv-cache`; summed users' KV with `sequences: users`); `maxUsersPerGpu(free, kvPerUser)` (`prefill-decode`; Llama's bytes per token from `kvBytesPerToken`, `kv-cache`) | `formatBytes`; `formatInt`, both KV ends for V4 |
| **Speed floor** at the chosen users: step time, bound, tok/s per user, tok/s per GPU | `stepTime({ activeParamsPerGpu: active, weightBytesPerGpu: w / gpus, dModel, actBytesPerElem, tokens: users, seqs: users, context, kvBytesPerToken, peakTflops, bandwidthTBps })` (`prefill-decode`) | `formatDuration`; "memory-" / "compute-bound"; `formatCount` (3 s.f.); `formatInt` |
| Most users at the target, and what limits them | `usersAtTarget({ targetTokPerUser, …same model…, maxUsers })` → `users`, `limit` ("compute", "bandwidth", "memory capacity"; a tie goes to compute, then bandwidth; "memory capacity" only when `maxUsers` is below both caps) | `formatInt`; text |
| With MTP on: tok/s per user × speedup | `batchSpeedup({ alpha: 0.85, k: 1, c: 0.05, batch: users, model })` (`speculative-decoding`) | `formatRatio` |
| **Cost** (floor), labeled "per M output tokens (this page's definition)" | `costPerMillion(price, tokPerSecPerGpu)` | $, 3 decimals |
| Cost (measured anchor, fixed row, on its own line), labeled "per M tokens, input and output together (InferenceX's convention)": InferenceX GB300 / GB200 | `costPerMillion(2.65, 6182)`, `costPerMillion(2.21, 2189)` with the data values | $, 3 decimals, "measured 2026-05-22" |
| **Input side** (floor): prefill tokens/s per GPU, with cache hits; input $ per M; output ÷ input cost | `prefillTokPerSecCeiling({ activeParamsPerGpu, peakTflops })` ÷ (1 − hit rate); `costPerMillion`; ratio | `formatInt`; $; `formatRatio` |

All derived metrics use the single definitions in `prefill-decode` §7 (README lesson 16): users per GPU = sequences in one GPU's decode batch; tokens/s per GPU = users ÷ step time, output tokens only; KV per user = KV bytes per token × context.

**Check my work** (pinned; default state: V4-Pro as shipped, GB300 NVL72, 16 GPUs, 8K + 1K, KV low, 1,889 users, target 27, $2.65, no cache hits, MTP off; one line per readout group):
```
Fit:    865 GB ÷ 288 GB → 4 GPUs for weights alone · 865 GB ÷ 16 = 54.1 GB per GPU · 288 GB − 54.1 GB = 234 GB free
KV:     4,000 B × 9,216 tokens = 36.9 MB per user · 234 GB ÷ 36.9 MB = 6,345 users fit
Speed:  t_math = 2 × 49e9 × 1,889 ÷ 5,000 TFLOP/s = 37 ms · t_read = (54.1 GB + 25.8 GB + 69.6 GB) ÷ 8 TB/s = 18.7 ms · step = max = 37 ms
Target: 1 ÷ 27 tok/s = 37 ms · compute allows 1,889 users, memory 4,793 · users = min = 1,889 (compute)
Cost:   $2.65 ÷ (51,020 tok/s × 3,600 s) × 1,000,000 = $0.0144 per M output tokens
Input:  5,000 TFLOP/s ÷ (2 × 49e9) = 51,020 tok/s · output ÷ input cost = 51,020 ÷ 51,020 = 1
```
With another state each line substitutes that state's numbers; the default text is what the page test pins. MTP on adds `batchSpeedup` = tokens per round × plain ÷ round, with its numbers.

**Try this** (each leads to a named insight)
1. Defaults, then switch context 8K → 128K → 1M (KV low): users that fit 6,345 → 419 → 58; at the 27 tok/s target the limit changes from "compute" (1,889 users) to "memory capacity" (419, then 58); floor tokens/s per GPU 51,020 → 11,433 → 1,618. Flip the KV estimate to high: 2,115 → 139 → 19. Then try GB200 (186 GB) at 1M with 16 GPUs: 32 users instead of 58. → **Insight: at long context, memory for KV, not compute, sets how many users a GPU serves,** and when memory is full every step re-reads it, so per-user speed tops out near HBM size over bandwidth (288 GB at 8 TB/s, about 36 ms per step).
2. Defaults: floor 51,020 output tokens/s per GPU at 27 tok/s per user; the measured row says 6,182 tokens/s per GPU, possibly counting input too, so the gap is a range: 8.25× to 74.3×. Now set users to 1: 148 tok/s per user and per GPU. → **Insight: the roofline is a ceiling, not a forecast.** It tells you which wall is closest (here: math at 8K, memory at 1M), while the measured 27 tok/s per user implies 37 ms per token, 5.48× the weight-read floor; plan with measurements and use the floor to reason about changes.
3. Input side at defaults: prefill ceiling 51,020 tok/s per GPU, the same as decode's ideal at 8K, so the floor's output/input cost ratio is 1.0, and a visible line says why: "V4-Pro's few-KB KV lets the ideal decode batch reach the ridge, so at 8K the floor makes output as cheap as input; real decode never gets there (frame 8)". Switch to 128K: 4.46×; at 1M: 31.5×. Turn the cache hit rate to 56.3%: input work per token falls to 43.7%, so input cost per token falls 2.29×. → **Insight: in the ideal, output costs more only when KV memory caps the decode batch;** in practice decode never reaches the ideal, measured nodes make ~5× more input than output tokens (frame 10), and cache hits help only the input side.
4. Turn MTP on. At 256 users and 8K: 1.48×. At "most at target" (1,889 users, compute-bound): MTP slows each user down to 0.9 of the plain speed. At 128K with 419 users: 1.73×. → **Insight: MTP pays where decode is memory-bound,** as LMSYS measured at 128K on GB300 and `speculative-decoding`'s batch curve shows. The toy prints LMSYS's record beside its own number: measured +87% (1.87×, DeepSeek-R1, a different model) vs the floor's 1.73×, which uses α = 0.85, the low end of DeepSeek's reported 85–90% acceptance.

**Functions used** (no new module; everything is imported):
- `math/serving.js` (`prefill-decode` §6, `disaggregation` §6): `RUNNING_EXAMPLE` (the Llama preset), `hbmFor`, `weightBytes`, `stepTime`, `freeHbmPerGpu`, `maxUsersPerGpu`, `prefillTokPerSecCeiling`, plus these three defined for this page:
```js
minGpusForWeights(weightBytes, hbmBytes) → number                    // ⌈weights / hbm⌉
usersAtTarget({ targetTokPerUser, activeParamsPerGpu, weightBytesPerGpu, dModel, actBytesPerElem, context, kvBytesPerToken, peakTflops, bandwidthTBps, maxUsers })
  → { users, byMem, byCompute, maxUsers, limit: 'compute' | 'bandwidth' | 'memory capacity' }
  // T = 1 / target; byMem = ⌊(T · bandwidth − weights) / (context · kvBytesPerToken + 2 · active · actBytes / dModel)⌋;
  // byCompute = ⌊T · peak / (2 · active)⌋; users = min(byMem, byCompute, maxUsers)
costPerMillion(dollarsPerGpuHour, tokPerSecPerGpu) → number          // $ / (tok/s · 3600) · 1e6 (InferenceX convention)
```
- `math/roofline.js` (`gpu-primer`): `bitsPerElement`, `bytesPerElement`.
- `math/specdec.js` (`speculative-decoding`): `batchSpeedup`.
- `math/memory.js` (`kv-cache`): `kvBytesPerToken` for Llama-3.1-70B (327,680; 163,840 in FP8).

Worked examples (scratch implementation, 2026-10-07; `V` = `{ activeParamsPerGpu: 49e9, weightBytesPerGpu: 865e9 / 16, dModel: 7168, actBytesPerElem: 1, peakTflops: 5000, bandwidthTBps: 8 }`):
```
minGpusForWeights(865e9, 288e9) → 4 · (865e9, 186e9) → 5 · BF16 3.2e12 → 12 · FP8 1.6e12 → 6 · NVFP4 0.9e12 → 4
freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: 16 }) → 233.94 GB (EP 8: 179.88 · EP 72: 275.99; GB200 EP 8: 77.88, EP 16: 131.94)
kvCacheBytes({ bytesPerToken: 4000, tokens: 9216 }) → 36.9 MB → maxUsersPerGpu → 6,345 · (12000) 111 MB → 2,115
kvCacheBytes({ bytesPerToken: 4000, tokens: 139264 }) → 557 MB → 419 · (12000) 1.67 GB → 139
kvCacheBytes({ bytesPerToken: 4000, tokens: 1e6 }) → 4.0 GB → 58 · (12000) 12 GB → 19 ; GB200 EP 16: 32 (low)
stepTime({ ...V, kvBytesPerToken: 4000, tokens: 1, seqs: 1, context: 9216 }) → 6.76 ms, 147.8 tok/s per user
  users 64 → 7.16 ms, 139.6 / 8,936 · 256 → 8.37 ms, 119.4 / 30,567 · 1,024 → 20.07 ms, compute-bound, 49.8 / 51,020
  at max users: 8K 124.36 ms (compute) · 128K 36.65 ms, 27.3 / 11,433 · 1M 35.86 ms, 27.9 / 1,618 (high KV: 35.29 ms, 28.3 / 538)
usersAtTarget({ targetTokPerUser: 27, ...V, context: 9216, kvBytesPerToken: 4000, maxUsers: 6345 }) → 1,889 (byCompute 1,889, byMem 4,793), step 37.02 ms, 51,020 tok/s/GPU
  at 128K: 419 (memory capacity) · at 1M: 58 (memory capacity) · FP4 math (peak 15,000): 4,793 (bandwidth) · target 13.1: 3,894 (compute)
costPerMillion(2.65, 6182) → 0.1191 (input + output, InferenceX) · (2.21, 2189) → 0.2804 · (2.65, 11056) → 0.0666 · floor, output only: (2.65, 51020) → 0.0144 · (2.65, 147.84) → 4.979
output-only reading of InferenceX if it counts input too: 6,182 ÷ 9 = 686.9 → floor gap 51,020 ÷ 686.9 = 74.3× (vs 8.25× if it is output-only)
prefillTokPerSecCeiling({ activeParamsPerGpu: 49e9, peakTflops: 5000 }) → 51,020 ; ÷ 11,433 → 4.46 ; ÷ 1,618 → 31.5 ; with 56.3% hits, input cost per token × 0.437
batchSpeedup({ alpha: 0.85, k: 1, c: 0.05, model: { ...V, kvBytesPerToken: 4000, context } }): 8K 256 users → 1.48 · 8K 1,889 → 0.90 · 128K 419 → 1.73 · 1M 58 → 1.76
weights-only floor 54.06 GB ÷ 8 TB/s = 6.76 ms vs implied time per token 1 ÷ 27 = 37.04 ms → 5.48×
Llama-3.1-70B on one H200, FP8, 9,216 tokens: 23 users fit (memory capacity), 29.14 ms, 34.3 tok/s each, 789 tok/s/GPU, $0.932 per M at $2.65 ; 131,072 tokens: 1 user, 42 tok/s/GPU
```
Reproducer (run once the modules exist):
```
node -e "Promise.all([import('./math/serving.js'), import('./math/specdec.js'), import('./math/memory.js')]).then(([s, d, mem]) => { const V = { activeParamsPerGpu: 49e9, weightBytesPerGpu: 865e9 / 16, dModel: 7168, actBytesPerElem: 1, peakTflops: 5000, bandwidthTBps: 8 }; const free = s.freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: 16 }); for (const ctx of [9216, 139264, 1e6]) for (const kv of [4000, 12000]) { const max = s.maxUsersPerGpu(free, mem.kvCacheBytes({ bytesPerToken: kv, tokens: ctx })); const u = s.usersAtTarget({ targetTokPerUser: 27, ...V, kvBytesPerToken: kv, context: ctx, maxUsers: max }); const st = s.stepTime({ ...V, kvBytesPerToken: kv, tokens: u.users, seqs: u.users, context: ctx }); console.log(ctx, kv, max, u, st.timeS * 1e3, u.users / st.timeS, s.costPerMillion(2.65, u.users / st.timeS)); } console.log(s.minGpusForWeights(865e9, 288e9), s.costPerMillion(2.65, 6182), s.prefillTokPerSecCeiling({ activeParamsPerGpu: 49e9, peakTflops: 5000 }), d.batchSpeedup({ alpha: 0.85, k: 1, c: 0.05, batch: 419, model: { ...V, kvBytesPerToken: 4000, context: 139264 } })) })"
```
Tests to write first: the table above; `usersAtTarget` never exceeds `maxUsers`; at the returned `users`, `stepTime(...).timeS ≤ 1 / target` and at `users + 1` it is above (unless capped by memory); `costPerMillion(2.65, 6182)` rounds to the InferenceX $0.12; inputs not mutated.

## 7. Show me the math
```tex
\text{GPUs}_{\min} = \left\lceil \frac{W}{\text{HBM}} \right\rceil,
\qquad
\text{free} = \text{HBM} - \frac{W}{G},
\qquad
\text{users}_{\max} = \left\lfloor \frac{\text{free}}{\htmlClass{hl-kv}{\text{KV/token}}\cdot \text{context}} \right\rfloor
```
```tex
t_{\text{step}}(u) = \max\!\left(\frac{2 N_{\text{active}}\, u}{\text{peak}},\ \frac{W/G + \tfrac{2 N b}{d}\,u + u\cdot\text{context}\cdot\htmlClass{hl-kv}{\text{KV/token}}}{\text{BW}}\right),
\qquad
u_{\text{target}} = \max\{u : t_{\text{step}}(u) \le 1/\text{target}\}
```
```tex
\$\text{ per M tokens} = \frac{\$\text{ per GPU-hour}}{3600\cdot \text{tokens/s per GPU}}\cdot 10^6,
\qquad
\frac{\text{output cost}}{\text{input cost}} = \frac{\text{prefill tokens/s per GPU}}{\text{decode tokens/s per GPU}}
```
Shapes: none (scalars per GPU). `G` = GPUs per replica, `u` = users per GPU, `b` = bytes per element, `d` = d_model. With experts spread evenly and attention data-parallel, each GPU holds its own users' KV and `W / G` of the weights (replicated non-expert weights ignored, brief 04 §8.1). Color links: `hl-kv` = the KV part of every memory bar, drawn twice (low and high) for V4-Pro.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| DeepSeek-V4-Pro: 1.6T total, 49B active, 61 layers, 384 routed experts (6 per token), 1M context; d_model 7,168; expert width 3,072 | `models.json/deepseek-v4-pro.total_params`, `.active_params`, `.layers`, `.experts_total`, `.experts_active`, `.context_length`; `.d_model = 7168`, `.expert_hidden = 3072` | 04 §8 CONFIRMED (config.json) |
| Experts FP4, everything else FP8 (QAT); checkpoint ≈ 865 GB | `models.json/deepseek-v4-pro.weight_formats` *(proposed by `quantization`)*, `.checkpoint_gb = 865` *(proposed, `reported`)* | 04 §5, §8.1 |
| KV per token 4,000–12,000 bytes: formula-derived, layer mix uncertain (config suggests alternating 128/4 compression; summaries say 3:1) | `models.json/deepseek-v4-pro.kv_bytes_per_token = [4000, 12000]` (`reported`, existing) | 04 §8.2, §10 item 3 |
| GB300 NVL72 per GPU: 288 GB HBM3e, 8 TB/s, ~5 PF dense FP8, ~15 PF dense FP4 (reported); GB200 NVL72 per GPU: 186 GB (13.4 TB ÷ 72), 8 TB/s | `hardware.json/gb300-nvl72.hbm_gb`, `.hbm_tbps`, `.fp8_e4m3_dense_tflops`, `.nvfp4_dense_tflops`; `gb200-nvl72.hbm_gb`, `.hbm_tbps` (GB300 memory is nominal; no usable figure in data) | 03 §1.4; 04 §1.4 |
| InferenceX (V4-Pro, ISL 8K / OSL 1K, FP4, disaggregated Dynamo + vLLM, measured 2026-05-22): GB300 6,182 tok/s/GPU at 27 tok/s/user, $0.12/M at $2.65/GPU-hr; GB200 2,189 at 27, $0.28/M at $2.21; peak 11,056 (GB300) / 8,933 (GB200) at ~13–15 tok/s/user. Whether tok/s/GPU includes input tokens is not stated | `serving.json/inferencex-v4-pro-gb300.throughput_tok_s_gpu`, `.interactivity_tok_s_user`, `.cost_per_m = 0.12`, `.gpu_hour_usd = 2.65`, `.max_throughput_tok_s_gpu`; same on `inferencex-v4-pro-gb200` (`.cost_per_m = 0.28`, `.gpu_hour_usd = 2.21`) *(proposed in `prefill-decode`, extended here)* | 04 §1.2, §1.3, §8.4, §10 item 4 CONFIRMED |
| DeepSeek production (V3/R1, Feb 2025): ~73.7K input vs ~14.8K output tok/s per H800 node; $87,072/day at $2/GPU-hr; 56.3% cache hits; average KV length 4,989 tokens | `serving.json/deepseek-v3-production.prefill_tok_s_node = 73700`, `.decode_tok_s_node = 14800`, `.cost_per_day_usd = 87072`, `.gpu_hour_usd = 2` *(proposed)*; `.kv_hit_rate_pct`, `.avg_kv_length_tokens` *(proposed by `paged-attention`)* | 04 §6.3, §7.4 CONFIRMED |
| List prices (read 2026-10-07): DeepSeek V4-Pro off-peak $0.66 input (miss) / $1.98 output; Anthropic output = 5× input across tiers (Sonnet 5.5 $2 / $10); batch API 50% off | `serving.json/pricing-deepseek-v4-pro.input_miss_usd_per_m`, `.output_usd_per_m = 1.98`; `pricing-anthropic.output_input_ratio = 5`, `pricing-anthropic-sonnet-5.5.output_usd_per_m = 10`, `pricing-anthropic.batch_discount_pct = 50` *(proposed)* | 04 §3.6, §7.4 CONFIRMED |
| Why output costs more: prefill at high intensity and often cached, decode one token per step holding KV, latency headroom provisioned for output (brief's analysis, labeled as such) | plain sentence, "analysis" tag | 04 §7.4 (ANALYSIS) |
| MTP raised per-user throughput 87% for DeepSeek-R1 on GB300 NVL72 at 128K in / 8K out (LMSYS, 2026-02-19); up to 40 concurrent 128K requests per GPU on GB300 vs 24 on GB200 (theoretical caps; LMSYS's practical target is 36 and 20, about 85% of the cap) | `serving.json/lmsys-gb300-longctx.mtp_per_user_gain_pct` *(proposed in `speculative-decoding`)*; `hardware.json/gb300-nvl72.concurrent_128k_per_gpu = 40`, `gb200-nvl72.concurrent_128k_per_gpu = 24`; practical targets: `.concurrent_128k_per_gpu_target` = 36 (GB300) and 20 (GB200) | 04 §4.3, §7.5 CONFIRMED |
| Claude 4.6+ charges 1M-token context at a flat per-token price | `serving.json/pricing-anthropic.flat_1m_context = true` *(proposed)* | 04 §7.4 CONFIRMED |

Not shown: the "~229 streams per GPU" implied by dividing InferenceX's two numbers (UNVERIFIED, depends on the unknown token-counting convention), DeepSeek's "545% cost profit margin" (theoretical), engine rankings.

## 9. Takeaways
1. Weights set the minimum GPUs (865 GB → at least 4 GB300s; 16 in a practical replica), but the KV cache sets how many users fit: thousands per GPU at 8K, a few dozen at 1M, where a full GPU caps each user near 28 tokens per second (frames 2–5, try-this 1).
2. The roofline floor says what is possible (148 tok/s for one user; ~1,900 users at 27 tok/s in the ideal); measured systems sit far below it, so cost is measured throughput priced per GPU-hour: about $0.12 per million input and output tokens for V4-Pro on GB300 (frames 6–9, try-this 2).
3. Output costs more than input because GPUs make far fewer output tokens per second than they read input, and long contexts widen the gap; cache hits cheapen only the input side, and MTP helps only where decode is memory-bound (frame 10, try-this 3–4).

## 10. Next and go deeper
Next: none (the serving track's capstone) · Related: `model-card` (reading the specs this page sizes), `prefill-decode`, `disaggregation`, `quantization`, `prefix-caching`, `speculative-decoding`.

Go deeper (brief 04 §9.2, 05 §1.2): InferenceX (https://inferencex.semianalysis.com/about), live tokens-per-GPU vs per-user curves and $/M · DeepSeek, "Inference System Overview" (https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) · LMSYS, "GB300 long context" (https://lmsys.org/blog/2026-02-19-gb300-longctx).

## 11. Key-frame sketch
Frame 8 end state, desktop width; numbers from §6 (KV low end).
```text
Serving a 1T model              step 8 / 10   [<] [Play] [>]
replica: 16 x GB300 (EP 16)    followed GPU: 288 GB
|w 54|################ KV for 1,889 users: 69.6 GB |.. free ..|
tok/s per user (floor)
 148 |*
 120 |   *
  50 |          *
  27 |-------------------*----- target: 1,889 users, 51,020 tok/s/GPU
     |   measured: o 6,182 tok/s/GPU at 27 (InferenceX; may include input)
     +-----------------------------------------------
       1    256     1,024   1,889   users per GPU
implied 37.0 ms per token vs weight-read floor 6.8 ms (5.48x)
V4-Pro | GB300 | EP 16 | 8K+1K | KV 4 KB/token (low)
floor counts output tokens only; compare as a range
```

## 12. Open questions for the reviewer
**Corrections to the brief**
- **Brief 04 §8.1 says ⌈865 ÷ 288⌉ = 3 GB300s; it is 4** (3 × 288 = 864 GB < 865 GB). This page shows 4 and the arithmetic. The same section's "GB300, EP=16 leaves ~230 GB" matches 233.94 GB.

**Data-pass keys**
- `hardware.json/gb300-nvl72.hbm_gb = 288`, `.hbm_tbps = 8`, `.fp8_e4m3_dense_tflops = 5000`, `.nvfp4_dense_tflops = 15000` (now in data on `gb300-nvl72`, all `reported`; the toy reads them directly, so the `b300.*` fallback is gone). No usable-memory figure exists for GB300, so 288 GB is labeled nominal.
- `models.json/deepseek-v4-pro.d_model = 7168`, `.expert_hidden = 3072`, `.checkpoint_gb = 865` (`reported`), `.weight_formats`.
- `serving.json` keys in §8 (pricing entries, InferenceX cost and price keys, DeepSeek production throughput keys).

**Judgment calls**
- **FP8 math for "as shipped".** V4-Pro's attention and shared layers are FP8 and its experts FP4; the toy uses the FP8 rate (5 PF) for the floor's math, the conservative choice, and the NVFP4 chip shows the FP4 rate (15 PF: the 27 tok/s target then allows 4,793 users, bandwidth-limited).
- **The floor makes output as cheap as input at 8K.** With V4-Pro's tiny KV, the ideal decode batch at 27 tok/s is compute-bound, so the floor's output/input ratio is 1.0 at 8K. The page does not hide this: frame 10 uses measured numbers (DeepSeek's 5×) and try-this 3 shows the floor's ratio growing with context (4.46× at 128K).
- **Whole-cache reads at 1M.** V4-Pro's sparse attention (CSA top-k) reads only part of the cache per step, so the 1M-context floor (28 tok/s per user) is pessimistic; the capacity numbers (58 / 19 users) are not affected. A visible label says so.
- **No activation reserve.** Free memory is HBM minus weights, an upper bound (the brief's ~200 GB leaves room for activations); the toy labels it.

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled (like with like):** the floor's $/M and tok/s count output tokens only, and are labeled so; InferenceX's $0.12 is "per M tokens, input and output together". The floor and the measurement are compared in tok/s per GPU, as a range (8.25× if InferenceX is output-only, 74.3× if it counts input too), never as a single ratio.
- **Settled:** "37 ms" is implied by 27 tok/s per user, not measured, and is tagged so everywhere (misconception 2, frame 8, try-this 2, sketch).
- **Settled:** frame 2 no longer hinges on the 1 GB margin of a reported checkpoint size; four GPUs, with the arithmetic and the "reported" tag.
- **Settled:** the 1M chip reads "1M = 1,000,000 (V4-Pro's configured context)".
- **Settled:** the input-side readout stays at every context, with the one-clause reason it reads 1.0 at 8K; "as shipped" uses the FP8 math rate, labeled on the chip; free memory has no activation reserve, labeled.
- **Settled:** cache size comes from `kvCacheBytes` (`kv-cache`); chip memory is labeled (B200 180 usable of 192 nominal; GB200 186 = rack total over 72; GB300 288 nominal).
- Applied: the followed GPU plus a 16-cell `rack` (lesson 18); frame 4's running KV total; frame 10's per-node vs per-GPU line (lesson 21); frame 5's lower-bound note; "HBM size over bandwidth"; misconception 3 adds latency headroom.
- Not done (Nice): a Kimi K2.5 point on the frame 8 plot. Its published numbers are $/M at a per-user speed, not tok/s per GPU, so it would add a third counting convention to one plot.

- Data pass 2026-10-07: FLOPS keys renamed and read from `gb300-nvl72`; `expert_hidden`; the 40 and 24 concurrency are theoretical (practical targets 36 and 20); GB300 memory labeled nominal.
