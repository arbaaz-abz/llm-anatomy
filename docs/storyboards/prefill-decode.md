# Prefill vs decode (`prefill-decode`)

Track: serving · Section: serving · Prereqs: serving-overview, gpu-primer
Status: draft
Sources: 04 §0, §1.1, §1.2, §1.3, §1.4, §3.1, §9.1 (two-phase and step-time toys); 03 §1.2 (roofline, ridge point), §1.4 (hardware table); 05 §1.1. Roofline functions come from `gpu-primer`'s module (assumed names; §12).

Running example for the serving track (used again by `batching`, `speculative-decoding`, `quantization`, `disaggregation` and `serving-calculator`): **Llama-3.1-70B with FP8 weights (70 GB) and a BF16 KV cache (327,680 bytes per token), on one H200 (141 GB, 4.8 TB/s, 1,979 TFLOP/s dense FP8).** It is the brief's own back-of-envelope (04 §1.4) and the model `paged-attention` scales up to. "70B" is rounded (§12).

## 1. Learning objective
After this page you can explain why one decode step costs about as much time as reading 206 prompt tokens on an H200 (frames 2–6): decode does ~2 operations per byte of weights it reads and waits on memory, while prefill reuses each byte for every prompt token and waits on arithmetic. You can estimate a decode step from bytes ÷ bandwidth (frame 2, try-this 2), and explain why batching users raises tokens per GPU while slowing each user, until KV memory runs out (frames 7–9, try-this 2–3).

## 2. Misconceptions to correct
- **Misconception:** "Generating a token is expensive because of all the math." → **Reality:** for one decode token the arithmetic takes 0.07 ms and reading the 70 GB of weights takes 14.6 ms on an H200. Decode waits on memory, not math. Corrected by frame 2. (04 §1.1; 03 §1.2)
- **Misconception:** "A longer prompt makes every token of the answer slower in proportion." → **Reality:** prompt tokens are processed in parallel at high arithmetic intensity; 206 of them cost the same 14.6 ms as one decode step. What a long prompt does add is KV to re-read at every decode step (frame 8, try-this 3). Corrected by frames 3–6. (04 §1.1, §1.4)
- **Misconception:** "Serving more users at once always slows everyone down in proportion." → **Reality:** in the memory-bound regime, eight users share one read of the weights: the step grows from 14.7 to 15.7 ms while the GPU makes 7.5× more tokens. Corrected by frame 7 and try-this 2. (04 §1.3)
- **Misconception:** "If you batch enough users, decode becomes compute-bound." → **Reality:** at 2,048 tokens of context the H200's memory is full at 105 users, with an arithmetic intensity of 105, far below the ridge (412). For a dense model at real context lengths, KV memory usually caps the batch first. Corrected by frame 9 and try-this 3. (04 §1.4, §7.5; numbers from `stepTime`)

## 3. Hook and intuition (final wording)
**Hook:** Why can a GPU read 206 tokens of your prompt in the time it takes to write one token of the answer?

Every token that passes through a 70B model does about two arithmetic operations per weight: 140 billion operations. To do them, the GPU must bring every weight from its high-bandwidth memory (HBM) into its compute units. In FP8 that is 70 GB, and the H200 reads 4.8 TB per second, so one pass over the weights takes 14.6 ms no matter how few tokens use it. The arithmetic for one token takes 0.07 ms. Decode makes one token per request per step, so it spends almost the whole step waiting for memory.

Prefill is the opposite case. All prompt tokens go through together, so each weight, once read, is used for every one of them. With 1,000 prompt tokens the arithmetic takes 70.7 ms while the reading still takes 14.6 ms: now the GPU waits on math. The roofline from `gpu-primer` says where the switch happens: at the ridge point, 412 operations per byte for an H200 in FP8. A 206-token prefill sits exactly there, so it costs the same 14.6 ms as one decode step. That is the answer to the hook: reading a prompt reuses each weight hundreds of times, writing an answer uses it once.

This is why servers batch decode. Eight users decoding together share one read of the weights, so the step barely grows and the GPU makes nearly eight times more tokens. The catch is the KV cache: each user's cached keys and values must also be read every step, and they take memory. At 2,048 tokens of context the H200 is full at 105 users, long before the batch would reach the ridge. More users per GPU means cheaper tokens but a slower stream for each user. Real servers pick a point on that curve.

## 4. Visual metaphor
Two lanes on one `gpu` glyph. The **step-time bar** is the hero: for each step, a horizontal `shareBar` (decoder-anatomy proposal) with named parts `weights read`, `KV read`, `arithmetic`, drawn to the same ms scale in every frame, with the longer of "reading" (weights + KV) and "arithmetic" winning (the shorter is drawn faint and labeled "overlapped": the roofline's max). Above it, `token` chips show how many tokens ride on the step: one chip for decode; for prefill, eight chips and a plain text mark "+ 992 others" (README lesson 18: 1,000 chips do not fit). A `curvePlot` (proposed below) shows the roofline in frames 5–6 and the per-user vs per-GPU curve in frames 8–10.

Colors: `--sem-memory` for the reading parts, `--sem-compute` for arithmetic (spec §5.2 semantic colors). The followed request A (hue 1, letter printed) has the accent frame on its chip in every frame; in batched frames it is one of the chips, still framed.

Terms introduced (one per frame): operations per token (frame 1), memory time / arithmetic time (2), weight reuse (3), arithmetic intensity (4, defined in `gpu-primer`, recalled on screen), memory-bound / compute-bound at the ridge (5), crossover (6), tokens per second per GPU (7), KV read (8), memory cap (9), operating point (10). Terms assumed: from `serving-overview`: prefill, decode step, TTFT, TPOT, batch; from `gpu-primer`: HBM, bandwidth, FLOP/s, roofline, ridge point, FP8; from `kv-cache`: KV cache, bytes per token.

Glyphs used: `gpu`, `token`, `request` (with the `owner` hue proposed in `serving-overview`), `shareBar` (decoder-anatomy proposal), `flow` (carry `activation` from HBM to the compute die).

Plain text labels: "70B is rounded; the toy uses 70,000,000,000 parameters" (frame 1); "+ 992 others" (frame 3); "overlapped" on the losing part of the step bar (frames 2–9); "each user: 2,048 tokens of context" (frames 7–9); "floor: real engines reach roughly 60–80% of this bandwidth (brief 04 §1.4)" (frame 2 and the toy); "measured, not this toy's model" over the InferenceX points (frame 10).

New glyphs proposed:
- **`curvePlot(parent, { x, y, w, h, xAxis: { label, log, ticks }, yAxis: { label, log, ticks }, series: [{ points, label, style }], markers: [{ x, y, label, followed }] })`**: a small line chart with optional log axes, labeled ticks and point markers (a marker with `followed: true` gets the accent frame). Why: the roofline and the latency/throughput curve are functions of a slider, and no library glyph plots a curve. If `gpu-primer` proposes a roofline glyph, the roofline here uses it and `curvePlot` covers the other curves. Reused by `speculative-decoding` (expected tokens vs k, speedup vs batch) and `serving-calculator` (users per GPU vs per-user speed).

## 5. Animation script
Numbers: `stepTime` on the running example (§6). Frames 2–6 use a short context (KV read under 0.1 ms, not drawn); frames 7–9 give every user 2,048 tokens of context, and the label says so.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | `gpu` "H200" with its memory bar: 70 GB `weights` filled, 71 GB free. Text: "Llama-3.1-70B, FP8: 1 byte per weight" and "70B is rounded; the toy uses 70,000,000,000 parameters". One chip `down` (A, framed). | The weights fill the memory bar left to right. | The model's 70 billion weights sit in GPU memory as 70 GB. Every token that passes through does about two operations per weight. | 70 GB of 141 GB · 140 billion operations per token |
| 2 | **Decode.** Chip `down` enters. The step bar draws: `weights read` 14.58 ms (solid), `arithmetic` 0.07 ms (faint, "overlapped"). Label: "floor: real engines reach roughly 60–80% of this bandwidth". | A `flow` from the HBM stacks to the die carries the weights; the reading bar grows to 14.58 ms while the arithmetic bar stops at 0.07 ms. | One decode token: the GPU must read all 70 GB to do its math. Reading takes 14.6 ms, the math 0.07 ms, so the step waits on memory. | read 70 GB ÷ 4.8 TB/s = 14.58 ms · math 140 GFLOP ÷ 1,979 TFLOP/s = 0.07 ms · step 14.58 ms |
| 3 | **Prefill, 1,000 tokens.** Eight chips (`The` `cat` `sat` …, first framed) and the mark "+ 992 others" enter together. Step bar: `arithmetic` 70.74 ms (solid), `weights read` 14.58 ms (faint, "overlapped"). | The same `flow` runs once; all chips light together; the arithmetic bar grows past the reading bar. | Prefill, 1,000 tokens: the same 70 GB is read once and reused for every token. Now the math takes 70.7 ms while the reading still takes 14.6. | math 140 TFLOP → 70.74 ms · read 14.58 ms · step 70.74 ms |
| 4 | Two readouts side by side under the two bars: "decode: 2 operations per byte read" and "prefill (1,000): 2,000 operations per byte read". | The two numbers count up from 0. | That ratio, operations per byte read, is the arithmetic intensity from `gpu-primer`. One decode token scores 2; a 1,000-token prefill scores 2,000. | 140 GFLOP ÷ 70 GB = 2 · 140 TFLOP ÷ 70 GB = 2,000 |
| 5 | `curvePlot` roofline for H200 FP8 (log x: operations per byte; y: attainable TFLOP/s), ridge marked at 412. Two markers: "decode (2)" on the sloped side, "prefill 1,000 (2,000)" on the flat roof. | The roofline draws; the two markers drop onto it; the ridge tick pulses. | On the H200's roofline, the ridge sits at 412 operations per byte. Decode lands far left, waiting on memory; prefill lands on the flat part, waiting on math. | ridge 1,979 ÷ 4.8 = 412.3 · decode 2 (memory-bound) · prefill 2,000 (compute-bound) |
| 6 | Prefill chips shrink back to 8 + "+ 198 others". Both bars now end at the same ms: `arithmetic` 14.57 ms and `weights read` 14.58 ms, next to the decode bar from frame 2. | The arithmetic bar shrinks from 70.7 to 14.6 ms and lines up with the decode bar. | At 206 prompt tokens, math and reading take the same 14.6 ms, as long as one decode step. A prompt reuses each weight hundreds of times; an answer uses it once. | crossover 412 ÷ 2 = 206 tokens · 14.57 vs 14.58 ms |
| 7 | **Back to decode, now eight users** (A–D and four more; chips A–H, A framed). Label "each user: 2,048 tokens of context". Step bars for 1 user and 8 users stacked: `weights read` 14.58 ms, `KV read` 0.14 vs 1.12 ms. Readout: tokens per second per GPU. | Seven more chips join A's; the KV part of the bar grows a little; the per-GPU counter jumps. | Batch eight users and they share one read of the weights. The step grows from 14.7 to 15.7 ms while the GPU makes 7.5 times more tokens per second. | 1 user: 14.72 ms, 68 tok/s per GPU · 8 users: 15.70 ms, 509 tok/s per GPU |
| 8 | `curvePlot`: x = tokens/s per user, y = tokens/s per GPU, points for 1, 8, 64, 105 users; the step bar for 64 users shows `KV read` 8.94 ms beside `weights read` 14.58 ms. | Markers appear from 1 user to 64; the `KV read` part grows each time. | Each user's cached keys and values are read every step too. At 64 users that adds 9 ms, so each user's stream slows while the GPU's total keeps rising. | 64 users: step 23.53 ms, 42.5 tok/s per user, 2,720 tok/s per GPU |
| 9 | Memory bar full: 70 GB weights + 70.5 GB KV of 141 GB. The 105-user marker lands at the end of the curve. Roofline inset: intensity 105, still left of the ridge (412). | The KV part of the memory bar fills to the top; the curve stops. | At 105 users the GPU's memory is full, and the batch is still far left of the ridge. At this context, KV memory, not arithmetic, caps the batch. | 105 users · step 29.26 ms · 34.2 tok/s per user · 3,588 tok/s per GPU · intensity 104.7 vs ridge 412 |
| 10 | A second `curvePlot` labeled "measured, not this toy's model": two InferenceX points for DeepSeek-V4-Pro on GB300 NVL72, with their conditions printed under the plot. | The two points appear; a dashed connector joins them. | Real servers choose a point on this curve. Measured on DeepSeek-V4-Pro and GB300, asking for twice the speed per user roughly halved the tokens per GPU. | 13.1 tok/s/user → 11,056 tok/s/GPU · 27 tok/s/user → 6,182 tok/s/GPU · ISL 8K / OSL 1K, FP4, 2026-05-22 |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter on this file, all captions ≤ 30 words and ≤ 2 sentences, no operators.

## 6. Toy
"Step-time calculator." Choose a phase and a load, watch which side of the roofline you land on. A visible line above the controls: "These are floors from bytes and FLOPs alone. Real engines reach roughly 60–80% of the bandwidth floor (brief 04 §1.4); attention math, communication and kernel overheads are not modeled."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `phase` | Phase | Toggle | prefill / decode | decode | — |
| `hw` | GPU | Preset chips | H100 · H200 · B200 | H200 | `hardware.json` `h100`, `h200`, `b200`: `hbm_gb`, `hbm_tbps`, `bf16_dense_tflops`, `fp8_dense_tflops` |
| `weights` | Weight format | Preset chips | BF16 (2 B) · FP8 (1 B) | FP8 | — |
| `promptTokens` | Prompt tokens (prefill) | Slider (snapped, log) | 1, 16, 64, 206, 512, 1,000, 2,048, 8,192 | 1,000 | — |
| `users` | Users in the batch (decode) | Slider (snapped) | 1, 2, 4, 8, 16, 32, 64, 128, …, "max that fits" | 8 | — |
| `context` | Context per user (decode) | Slider (snapped) | 512, 2,048, 8,192, 32,768, 131,072 tokens | 2,048 | — |

Model is fixed (Llama-3.1-70B: 70e9 parameters, 327,680 KV bytes per token from `kvBytesPerToken({ layers: 80, kvHeads: 8, headDim: 128, bytesPerElem: 2 })`). If `users` exceeds the memory limit, the slider clamps to "max that fits" and a visible note says so. If the weights alone do not fit (BF16 on H100: 140 GB > 80 GB), every output is replaced by "does not fit on one GPU: see `serving-calculator`".

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Operations, bytes read | `stepTime(...).flops`, `.bytes` | GFLOP / TFLOP, GB (decimal) |
| Arithmetic intensity, ridge point | `arithmeticIntensity(flops, bytes)`, `ridgePoint(peakFlops, bandwidth)` (gpu-primer's module) | FLOP/byte, 1 decimal |
| Math time, reading time, step time, bound | `stepTime(...)` → `computeS`, `memoryS`, `timeS`, `bound` | ms, 2 decimals; "memory-bound" / "compute-bound" |
| Prefill: tokens per second, TTFT for this prompt | `promptTokens / timeS`; TTFT = `timeS` (no queue) | tok/s; ms |
| Decode: tokens/s per user, tokens/s per GPU | `1 / timeS`; `users / timeS` | 1 decimal; integer |
| Max users that fit | `maxUsersPerGpu(freeHbmPerGpu({ hbmBytes, weightBytes, gpus: 1 }), kvBytesPerSequence(327680, context))` | integer |
| Memory bar: weights · KV · free | `weightBytes`, `users · kvBytesPerSequence`, rest | GB |
| Curve: per user vs per GPU, users 1 → max | `stepTime` swept over the snapped user values | `curvePlot`, current point framed |

**Try this** (each leads to a named insight)
1. Phase prefill, H200, FP8. Slide the prompt from 1 to 8,192 and watch the bound: memory-bound up to 206, compute-bound from there (14.58 ms at 206; 70.74 ms at 1,000; 579.5 ms at 8,192). Switch to B200: the flip moves to 281. Switch weights to BF16 on the H200: the flip stays at 206. → **Insight: the crossover in tokens is set by the GPU's ratio of math to bandwidth.** FP8 halves the bytes and doubles the math rate, so on H100 and H200 it moves the ridge but not the token crossover.
2. Phase decode, context 2,048. Slide users 1 → 8 → 64 → max: per user 67.9 → 63.7 → 42.5 → 34.2 tok/s; per GPU 68 → 509 → 2,720 → 3,588 tok/s. → **Insight: batching trades each user's speed for the GPU's total,** cheaply at first (one shared read of the weights), then more steeply as KV reads grow.
3. Keep users at "max that fits" and slide context 2,048 → 8,192 → 32,768 → 131,072: max users 105 → 26 → 6 → 1; tokens per GPU 3,588 → 893 → 214 → 42. → **Insight: at long context the KV cache, not the weights, decides how many users a GPU can hold,** and with them its throughput. That is the problem `paged-attention`, `kv-compression` and FP8 KV (`quantization`) attack.
4. Switch weights to BF16 on the H200 at context 2,048: 1 GB is left and only 1 user fits (34.1 tok/s per GPU). → **Insight: the weight format decides whether there is room for users at all.** You'll see what FP8 and FP4 cost in quality in `quantization`.

**Roofline interface needed from `gpu-primer`** (assumed module `math/roofline.js`; reconcile names at review, §12):
```js
arithmeticIntensity(flops, bytes) → number                       // FLOP per byte
ridgePoint(peakFlops, bandwidth) → number                         // FLOP/s ÷ bytes/s
rooflineTime({ flops, bytes, peakFlops, bandwidth })
  → { computeS, memoryS, timeS, bound: 'compute' | 'memory' }     // timeS = max(computeS, memoryS)
```

**`math/serving.js`** (pure, no DOM; owned by the serving track; every serving lesson imports from here). All sizes in bytes, rates in SI units per second.
```js
weightBytes({ params, bitsPerParam }) → number                    // params · bits / 8
stepTime({ activeParamsPerGpu, weightBytesPerGpu, tokens, seqs, context, kvBytesPerToken, peakFlops, bandwidth })
  → { flops, bytes, computeS, memoryS, timeS, bound }
  // one engine iteration on one GPU: flops = 2 · activeParamsPerGpu · tokens;
  // bytes = weightBytesPerGpu + seqs · context · kvBytesPerToken; then rooflineTime(...)
  // prefill: tokens = prompt, seqs = 0 · decode: tokens = seqs = users · spec-decode verify: tokens = users · (k + 1), seqs = users
kvBytesPerSequence(kvBytesPerToken, contextTokens) → number
freeHbmPerGpu({ hbmBytes, weightBytes, gpus }) → number          // hbm − weights / gpus (may be negative: does not fit)
maxUsersPerGpu(freeBytes, kvBytesPerSequence) → number            // floor, ≥ 0
prefillTokPerSecCeiling({ activeParams, peakFlops }) → number     // peak / (2 · active)
requestTimeline(...)                                               // serving-overview §6
usersAtTarget(...), minGpusForWeights(...), costPerMillion(...)   // serving-calculator §6
kvTransferTime(...), tokensPerExpert(...)                          // disaggregation §6
```
`kvBytesPerToken` / `kvBytesPerTokenMla` are imported from `math/memory.js` (`kv-cache`), never redefined.

Worked examples (scratch implementation, 2026-10-07; `L` = `{ activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, kvBytesPerToken: 327680, peakFlops: 1979e12, bandwidth: 4.8e12 }`):
```
ridgePoint(1979e12, 4.8e12) → 412.3   (H200 FP8)   ridgePoint(989e12, 3.35e12) → 295.2 (H100 BF16)   ridgePoint(4500e12, 8e12) → 562.5 (B200 FP8)
stepTime({ ...L, tokens: 1,    seqs: 1, context: 0 })    → compute 0.071 ms, memory 14.583 ms, step 14.58 ms, memory-bound, intensity 2
stepTime({ ...L, tokens: 206,  seqs: 0, context: 0 })    → compute 14.573, memory 14.583 → 14.58 ms, intensity 412
stepTime({ ...L, tokens: 1000, seqs: 0, context: 0 })    → compute 70.743, memory 14.583 → 70.74 ms, compute-bound, intensity 2,000
stepTime({ ...L, tokens: 8192, seqs: 0, context: 0 })    → 579.53 ms
decode, context 2,048 (users: step ms · tok/s per user · tok/s per GPU · intensity):
  1: 14.72 · 67.9 · 68 · 2.0     8: 15.70 · 63.7 · 509 · 14.9     64: 23.53 · 42.5 · 2,720 · 79.3     105: 29.26 · 34.2 · 3,588 · 104.7
maxUsersPerGpu(71e9, kvBytesPerSequence(327680, c)) → 105 (2,048) · 26 (8,192) · 6 (32,768) · 1 (131,072)
at max users: 8,192 → 29.12 ms, 893 tok/s/GPU · 32,768 → 28.01 ms, 214 · 131,072 → 23.53 ms, 42
BF16 on H200 (weightBytesPerGpu 140e9): free 1.0 GB, max users 1, step 29.31 ms
B200 FP8 (hbm 192e9, bw 8e12, peak 4500e12): crossover 281 tokens; at 2,048 context max 181 users, 7,563 tok/s/GPU
prefillTokPerSecCeiling({ activeParams: 70e9, peakFlops: 1979e12 }) → 14,135.7
```
Reproducer (run once `math/serving.js` and gpu-primer's roofline module exist):
```
node -e "import('./math/serving.js').then(m => { const L = { activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, kvBytesPerToken: 327680, peakFlops: 1979e12, bandwidth: 4.8e12 }; for (const t of [1, 206, 1000, 8192]) console.log('prefill', t, m.stepTime({ ...L, tokens: t, seqs: t === 1 ? 1 : 0, context: 0 })); for (const c of [2048, 8192, 32768, 131072]) { const max = m.maxUsersPerGpu(71e9, m.kvBytesPerSequence(327680, c)); for (const u of [1, 8, 64, max]) if (u <= max) { const s = m.stepTime({ ...L, tokens: u, seqs: u, context: c }); console.log(c, u, (s.timeS * 1e3).toFixed(2), (1 / s.timeS).toFixed(1), Math.round(u / s.timeS)); } } })"
```
Tests to write first: the table above; `timeS === max(computeS, memoryS)`; decode at one user and zero context equals `weightBytes / bandwidth`; `maxUsersPerGpu` never negative; `freeHbmPerGpu` negative for BF16 on H100; inputs not mutated.

## 7. Show me the math
```tex
\htmlClass{hl-compute}{t_{\text{math}}} = \frac{2\,N_{\text{active}}\cdot \text{tokens}}{\text{peak FLOP/s}}
\qquad
\htmlClass{hl-memory}{t_{\text{read}}} = \frac{W_{\text{bytes}} + \text{users}\cdot \text{context}\cdot \text{KV bytes/token}}{\text{HBM bandwidth}}
```
```tex
t_{\text{step}} = \max\!\left(\htmlClass{hl-compute}{t_{\text{math}}},\ \htmlClass{hl-memory}{t_{\text{read}}}\right)
\qquad
I = \frac{2 N \cdot \text{tokens}}{W_{\text{bytes}}} \;(\text{short context})
\qquad
\text{crossover tokens} = \frac{\text{ridge}\cdot b_{\text{weight}}}{2}
```
```tex
\text{TPOT} = t_{\text{step}}, \qquad
\text{tokens/s per user} = \frac{1}{t_{\text{step}}}, \qquad
\text{tokens/s per GPU} = \frac{\text{users}}{t_{\text{step}}}
```
Shapes: a dense layer's matmul is `[tokens × d_in] · [d_in × d_out]`; with `tokens = 1` it is a matrix-vector product, which is why decode reuses nothing. Ignored on this page: attention FLOPs (about 1% of a 1,000-token prefill for this model), KV writes, activation traffic, communication. `b_weight` = bytes per weight (1 for FP8, 2 for BF16). Color links: `hl-compute` = the arithmetic part of the step bar (`--sem-compute`); `hl-memory` = the reading parts (`--sem-memory`).

Metric definitions shared by every serving lesson (README lesson 16): **step time** = one engine iteration on one GPU = `stepTime(...).timeS`, a floor; **TPOT** = decode step time; **tokens/s per user** = 1 / TPOT; **users per GPU** = sequences in the decode batch held on one GPU; **tokens/s per GPU** = users per GPU / TPOT, output tokens only; **KV bytes per sequence** = KV bytes per token × context tokens.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| H200: 141 GB HBM3e, 4.8 TB/s, 1,979 TFLOP/s dense FP8 (FP8 = 2 × BF16, reported) | `hardware.json/h200.hbm_gb`, `.hbm_tbps`, `.fp8_dense_tflops` | 03 §1.4; 04 §1.4 |
| H100 ridge ≈ 295 FLOP/byte in BF16; B200 8 TB/s, ~4.5 PF FP8 | `hardware.json/h100.bf16_dense_tflops`, `.hbm_tbps`; `b200.hbm_tbps`, `.fp8_dense_tflops`, `.hbm_gb` | 03 §1.2, §1.4 |
| Llama-3.1-70B: 80 layers, 8 KV heads × 128, 131,072-token context; 327,680 KV bytes per token in BF16 | `models.json/llama-3.1-70b.layers`, `.kv_heads`, `.head_dim`, `.context_length`, `.kv_bytes_per_token`, `.total_params = 70e9` *(entry proposed by `paged-attention`; `total_params` added here, rounded, see §12)* | 01 §4; 04 §1.4, §3.1 |
| Real engines reach maybe 60–80% of the bandwidth floor (ESTIMATE in the brief, shown as such) | `serving.json/roofline-practice.bandwidth_efficiency_pct = [60, 80]` *(proposed, `reported`, note "brief estimate")* | 04 §1.4 |
| DeepSeek-V4-Pro on GB300 NVL72: 11,056 tok/s/GPU at 13.1 tok/s/user (max throughput) and 6,182 tok/s/GPU at 27 tok/s/user; on GB200, 8,933 at 15.3 and 2,189 at 27 (ISL 8K / OSL 1K, FP4, InferenceX, 2026-05-22). Whether tok/s/GPU counts input tokens too is not stated by the source; the page says so | `serving.json/inferencex-v4-pro-gb300.max_throughput_tok_s_gpu = 11056`, `.max_throughput_tok_s_user = 13.1`, `.throughput_tok_s_gpu = 6182`, `.interactivity_tok_s_user = 27`; same keys on `inferencex-v4-pro-gb200` (8933, 15.3, 2189, 27) *(proposed)* | 04 §1.3, §10 item 4 CONFIRMED |
| Kimi K2.5 on B200 (NVFP4): $0.140/M at 32 tok/s/user vs $0.347/M at 90 tok/s/user | `serving.json/inferencex-kimi-k2.5-b200.cost_per_m_at_32 = 0.140`, `.cost_per_m_at_90 = 0.347` *(proposed)* | 04 §1.3 CONFIRMED |

## 9. Takeaways
1. Decode reads every weight to make one token per request, so it waits on memory (14.6 ms for 70 GB on an H200); prefill reuses each weight for every prompt token, so it waits on math (frames 2–5).
2. The crossover is the GPU's ridge point: 206 prompt tokens cost one decode step on an H200, which is why reading a prompt is cheap per token and writing an answer is not (frame 6, try-this 1).
3. Batching decode shares the weight read, raising tokens per GPU while slowing each user; KV memory usually caps the batch before the ridge, and long contexts shrink it fast (frames 7–9, try-this 2–3).

## 10. Next and go deeper
Next: `batching`, `speculative-decoding` · Related: `gpu-primer` (the roofline), `quantization` (fewer bytes per weight), `serving-calculator`.

Go deeper (brief 05 §1.2, 04 §9.2): Google, *How to Scale Your Model*, inference chapter (https://jax-ml.github.io/scaling-book/) · InferenceX dashboard (https://inferencex.semianalysis.com/about), live per-user vs per-GPU curves · Lynskey, *LLM Inference Explained* (https://llm-inference-explained.vercel.app).

## 11. Key-frame sketch
Frame 9 end state, desktop width; numbers from the decode rows of the §6 reproducer.
```text
Prefill vs decode               step 9 / 10   [<] [Play] [>]
H200 memory |######## weights 70 GB ########|#### KV 70.5 GB ###|
            141 GB full at 105 users (2,048 tokens each)
step 29.26 ms  |== weights read 14.58 ==|== KV read 14.68 ==|
               |ar 7.43 (overlapped)|
per GPU tok/s
 3588 |                                   * 105 users
 2720 |                        * 64
  509 |          * 8
   68 |* 1
      +--------------------------------------------------
        34.2   42.5        63.7  67.9   tok/s per user
intensity 104.7  <  ridge 412.3   -> still memory-bound
```

## 12. Open questions for the reviewer
**Graph and cross-track interfaces**
- **Roofline names (`gpu-primer`).** This page assumes `math/roofline.js` with `arithmeticIntensity(flops, bytes)`, `ridgePoint(peakFlops, bandwidth)` and `rooflineTime({ flops, bytes, peakFlops, bandwidth }) → { computeS, memoryS, timeS, bound }`, all in SI units. If `gpu-primer` names them differently, `stepTime` imports the new names; no frame changes. The coordinator's ruling: gpu-primer owns these; serving does not redefine them.
- **Roofline glyph.** If `gpu-primer` proposes a roofline chart glyph, frame 5 uses it; `curvePlot` is still needed for frames 8–10 and other serving pages.

**Data-pass keys**
- `models.json/llama-3.1-70b.total_params = 70e9`, note "rounded; Meta's card lists 70.6B (not in the briefs; the data pass may set the exact value, which changes every ms on this page by under 1%)". Toy numbers on this page use 70e9 to match the brief's 70 GB back-of-envelope.
- `serving.json/roofline-practice.bandwidth_efficiency_pct`, `inferencex-v4-pro-gb300.*`, `inferencex-v4-pro-gb200.*`, `inferencex-kimi-k2.5-b200.*` as in §8.
- `hardware.json/h200.fp8_dense_tflops` is `reported` (2 × BF16); the page's 412 ridge inherits that confidence and says "dense FP8, reported".

**Judgment calls**
- The decode model ignores attention FLOPs and treats every user's whole cache as read each step. For a dense GQA model that is close; for sparse-attention models it overstates KV reads (`long-context-attention`). The visible floor line covers it.
- Frame 10 shows a different model on different hardware from the toy's running example, under an on-screen "measured, not this toy's model" label, because no measured curve exists for the running example in the briefs.
