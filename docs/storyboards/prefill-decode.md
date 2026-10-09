# Prefill vs decode (`prefill-decode`)

Track: serving · Section: serving · Prereqs: serving-overview, gpu-primer
Status: approved (expert review)
Sources: 04 §0, §1.1, §1.2, §1.3, §1.4, §3.1, §9.1 (two-phase and step-time toys); 03 §1.2 (roofline, ridge point), §1.4 (hardware table); 05 §1.1. Roofline math is imported from `math/roofline.js`, owned by `gpu-primer` (`matmulCost`, `arithmeticIntensity`, `ridgePoint`, `rooflineTime`, `tokensToComputeBound`), and this page uses its framing: compute-bound is counted in **tokens per weight read**. Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Running example for the serving track (used again by `batching`, `speculative-decoding`, `quantization`, `disaggregation` and `serving-calculator`): **Llama-3.1-70B with FP8 weights (70 GB) and a BF16 KV cache (327,680 bytes per token), on one H200 (141 GB, 4.8 TB/s, 1,979 TFLOP/s dense FP8).** It is the brief's own back-of-envelope (04 §1.4) and the model `paged-attention` scales up to. "70B" is rounded (§12). Its d_model is 8,192, so its big multiplies are exactly the 8,192 × 8,192 multiplies `gpu-primer` draws; the step-time model treats the whole model as 70e9 ÷ 8,192² ≈ 1,043 such multiplies.

## 1. Learning objective
After this page you can explain why reading 217 prompt tokens costs an H200 about as much time as writing one token of the answer (frames 2–6): decode does ~2 operations per byte it reads and waits on memory, while prefill reuses each weight for every prompt token and waits on arithmetic. You can estimate a decode step from bytes ÷ bandwidth (frame 2, try-this 2), and explain why batching users raises tokens per GPU while slowing each user, until KV memory runs out (frames 7–9, try-this 2–3).

## 2. Misconceptions to correct
- **Misconception:** "Generating a token is expensive because of all the math." → **Reality:** for one decode token the arithmetic takes 70.7 µs and reading the 70 GB of weights takes 14.6 ms on an H200. Decode waits on memory, not math. Corrected by frame 2. (04 §1.1; 03 §1.2)
- **Misconception:** "A longer prompt makes every token of the answer slower in proportion." → **Reality:** prompt tokens are processed together, each weight read once for all of them; 217 of them take 15.4 ms, barely more than one 14.6 ms decode step. What a long prompt does add is KV to re-read at every decode step (frame 8, try-this 3). Corrected by frames 3–6. (04 §1.1, §1.4)
- **Misconception:** "Serving more users at once always slows everyone down in proportion." → **Reality:** in the memory-bound regime, eight users share one read of the weights: the step grows from 14.7 to 15.7 ms while the GPU makes 7.49× more tokens. Corrected by frame 7 and try-this 2. (04 §1.3)
- **Misconception:** "If you batch enough users, decode becomes compute-bound." → **Reality:** at 2,048 tokens of context the H200's memory is full at 105 users, with an arithmetic intensity of 103, far below the ridge (412). For a dense model at real context lengths, KV memory usually caps the batch first. Models with a few KB of KV per token (DeepSeek-V4-Pro) are the exception: their decode batch can reach the ridge (`serving-calculator`). Corrected by frame 9 and try-this 3. (04 §1.4, §7.5; numbers from `stepTime`)

## 3. Hook and intuition (final wording)
**Hook:** Why can a GPU read 217 tokens of your prompt in about the time it takes to write one token of the answer?

Every token that passes through a 70B model does about two arithmetic operations per weight (a multiply and an add, counted as two): 140 billion operations. To do them, the GPU must bring every weight from its high-bandwidth memory (HBM) into its compute units. In FP8 that is 70 GB, and the H200 reads 4.8 TB per second, so one pass over the weights takes 14.6 ms no matter how few tokens use it. The arithmetic for one token takes 70.7 µs. Decode makes one token per request per step, so it spends almost the whole step waiting for memory.

Prefill is the opposite case. All prompt tokens go through together, so each weight, once read, is used for every one of them. With 1,000 prompt tokens the arithmetic takes 70.7 ms, while the reading (weights plus the tokens' own activations) takes 18.1 ms: now the GPU waits on math. `gpu-primer` counts the switch in tokens per weight read: an H200 needs 217. At 217 prompt tokens math and reading both take 15.4 ms, about as long as one decode step. That is the answer to the hook: reading a prompt reuses each weight hundreds of times, writing an answer uses it once.

This is why servers batch decode. Eight users decoding together share one read of the weights, so the step barely grows and the GPU makes nearly eight times more tokens. The catch is the KV cache: each user's cached keys and values must also be read every step, and they take memory. At 2,048 tokens of context the H200 is full at 105 users, long before the batch would reach the ridge. More users per GPU means cheaper tokens but a slower stream for each user. Real servers pick a point on that curve.

## 4. Visual metaphor
Two lanes on one `gpu` glyph. The **step-time bar** is the hero: for each step, a `stepBar` (P4-R8: two rows on one fixed seconds scale held across frames) whose `reading` row holds the parts `weights read`, `KV read`, `activations` and whose `arithmetic` row is one segment, with the longer row winning (the shorter is drawn faint and labeled "overlapped": the roofline's max). Segment rule (README lesson 19, `stepBar`'s minimum): a part the learner must read is at least 18 px wide; a narrower one (frame 2's 70.7 µs math, frame 3's activations, frame 7's 140 µs KV read for one user) widens to 18 px under a bracket mark with its true value printed; `activations` (3.56 µs to 374 µs in decode) is folded into "others" with its number printed in frames 7–9. Above it, `token` chips show how many tokens ride on the step: one chip for decode; for prefill, eight chips and a plain text mark "+ 992 others" (README lesson 18: 1,000 chips do not fit). A `curvePlot` (proposed below) shows the roofline in frames 5–6 and the per-user vs per-GPU curve in frames 8–10.

Colors: `--sem-memory` for the reading parts, `--sem-compute` for arithmetic (spec §5.2 semantic colors). The followed request A (hue 1, letter printed) has the accent frame on its chip in every frame; in batched frames it is one of the chips, still framed.

Terms introduced (one per frame): operations per token (frame 1), memory time / arithmetic time (2), weight reuse (3), arithmetic intensity (4, defined in `gpu-primer`, recalled on screen), memory-bound / compute-bound at the ridge (5), tokens per weight read (6), tokens per second per GPU (7), KV read (8), memory cap (9), operating point (10). Terms assumed: from `serving-overview`: prefill, decode step, TTFT, TPOT, batch; from `gpu-primer`: HBM, bandwidth, FLOP/s, roofline, ridge point, tokens to compute-bound, FP8; from `kv-cache`: KV cache, bytes per token.

Glyphs used: `gpu` (`memFill` plus the label "weights read"; the glyph has no HBM-highlight option, P4-R17), `token`, `request` (with the `owner` option), `stepBar` (the step bar; P4-R8), `shareBar` (the memory bar: weights · KV · free), `curvePlot`, `roofline`. No `flow`: nothing on this page is an activation, gradient, KV or token in transit.

Plain text labels: "70B is rounded; the toy uses 70,000,000,000 parameters" (frame 1); "+ 992 others" / "+ 209 others" (frames 3, 6); "overlapped" on the losing part of the step bar (frames 2–9); "each user: 2,048 tokens of context" (frames 7–9); "floor: real engines reach less than this bandwidth, so real steps take longer" (frame 2 and the toy); "measured, not this toy's model" over the InferenceX points and "a different model and GPU; compare the shape (asking each user to go faster costs tokens per GPU), not the numbers" (frame 10); "+ 4 others" beside chips A–D (frame 7).

New glyphs proposed:
- **`curvePlot(parent, { x, y, w, h, xAxis: { label, log, ticks }, yAxis: { label, log, ticks }, series: [{ points, label, style }], markers: [{ x, y, label, followed }] })`**: a small line chart with optional log axes, labeled ticks and point markers (a marker with `followed: true` gets the accent frame). It also takes `bands: [{ from, to, label }]` (a shaded x-range with a printed label, from the recipe track's proposal; it encodes a range, not a quantity). Why: the latency/throughput curve and the speedup curves are functions of a slider, and no library glyph plots a curve. Frame 5 calls `gpu-primer`'s `roofline`, which is a thin wrapper over `curvePlot` (settled). Reused by `speculative-decoding` (expected tokens vs k, speedup vs batch) and `serving-calculator` (per-user speed vs users per GPU).

## 5. Animation script
Numbers: `stepTime` on the running example (§6). Frames 2–6 use no context (KV read zero); frames 7–9 give every user 2,048 tokens of context, and the label says so.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | `gpu` "H200" with its memory bar: 70 GB `weights` filled, 71 GB free. Text: "Llama-3.1-70B, FP8: 1 byte per weight" and "70B is rounded; the toy uses 70,000,000,000 parameters". One chip `down` (A, framed). | The weights fill the memory bar left to right. | The model's 70 billion weights sit in GPU memory as 70 GB. Every token that passes through does about two operations per weight. | 70 GB of 141 GB · 140 billion operations per token |
| 2 | **Decode.** Chip `down` enters. The step bar draws: `weights read` 14.6 ms (solid), `arithmetic` 70.7 µs (faint, "overlapped"). Label: "floor: real engines reach less than this bandwidth, so real steps take longer". The 70.7 µs math part widens to 18 px under a bracket, its true value printed. | The `gpu`'s memory fill pulses once with the label "weights read" (weights are not one of `flow`'s carried kinds); the reading bar grows to 14.6 ms while the arithmetic bar stops at 70.7 µs. | One decode token: the GPU must read all 70 GB to do its math. Reading takes 14.6 ms, the math 70.7 µs, so the step waits on memory. | read 70 GB ÷ 4.8 TB/s = 14.6 ms · math 140 GFLOP ÷ 1,979 TFLOP/s = 70.7 µs · step 14.6 ms |
| 3 | **Prefill, 1,000 tokens.** Eight chips (`The` `cat` `sat` …, first framed) and the mark "+ 992 others" enter together. Step bar: `arithmetic` 70.7 ms (solid); `weights read` 14.6 ms + `activations` 3.56 ms (faint, "overlapped"). | The memory fill pulses once with the label "weights read"; all chips light together; the arithmetic bar grows past the reading bar. | Prefill, 1,000 tokens: the same 70 GB is read once and reused for every token. Now the math takes 70.7 ms and the reading only 18.1. | math 140 TFLOP → 70.7 ms · read 70 GB weights + 17.1 GB activations → 18.1 ms · step 70.7 ms |
| 4 | Two readouts side by side under the two bars: "decode: 2 operations per byte read" and "prefill (1,000): about 1,600 operations per byte read". | The two numbers count up from 0. | That ratio, operations per byte read, is the arithmetic intensity from `gpu-primer`. One decode token scores 2; a 1,000-token prefill scores about 1,600. | 140 GFLOP ÷ 70 GB = 2.0 · 140 TFLOP ÷ 87.1 GB = 1,607.5 |
| 5 | `roofline` (gpu-primer's glyph) for H200 FP8 (log x: operations per byte; y: attainable TFLOP/s), ridge marked at 412. Two markers: "decode (2)" on the sloped side, "prefill 1,000 (1,608)" on the flat roof. | The roofline draws; the two markers drop onto it; the ridge tick pulses. | On the H200's roofline the ridge sits at 412 operations per byte. Decode lands far left, waiting on memory; prefill lands on the flat part, waiting on math. | ridge 1,979 ÷ 4.8 = 412.3 · decode 2.0 (memory-bound) · prefill 1,607.5 (compute-bound) |
| 6 | Prefill chips shrink to 8 + "+ 209 others". Both bars end together: `arithmetic` 15.4 ms and reading 15.4 ms, next to the decode bar from frame 2 (14.6 ms). | The arithmetic bar shrinks from 70.7 to 15.4 ms and lines up with the reading bar, a hair longer than the decode bar. | At 217 prompt tokens, math and reading both take 15.4 ms, barely longer than one decode step. A prompt reuses each weight hundreds of times; an answer uses it once. | `tokensToComputeBound` = 217.1 tokens per weight read (H200, FP8 or BF16) · on-screen: "H100: 318 (A GPU for LLM people); the H200's faster 4.8 TB/s memory lowers the crossover" · 15.4 vs 15.4 ms · decode 14.6 ms |
| 7 | **Back to decode, now eight users**: chips A–D in their hues (A framed) and a plain mark "+ 4 others". Label "each user: 2,048 tokens of context". Step bars for 1 user and 8 users stacked: `weights read` 14.6 ms, `KV read` 140 µs vs 1.12 ms (the 140 µs part widens to 18 px under a bracket), `others` (activations) printed. Readout: tokens per second per GPU. | Three more chips and the "+ 4 others" mark join A's; the KV part of the bar grows a little; the per-GPU counter jumps. | Batch eight users and they share one read of the weights. The step grows from 14.7 to 15.7 ms while the GPU makes 7.49 times more tokens per second. | 1 user: 14.7 ms, 68 tok/s per GPU · 8 users: 15.7 ms, 509 tok/s per GPU |
| 8 | `curvePlot`: x = tokens/s per user, y = tokens/s per GPU, points for 1, 8, 64, 105 users; the step bar for 64 users shows `KV read` 8.95 ms beside `weights read` 14.6 ms. | Markers appear from 1 user to 64; the `KV read` part grows each time. | Each user's cached keys and values are read every step too. At 64 users that adds 9 ms, so each user's stream slows while the GPU's total keeps rising. | 64 users: step 23.8 ms, 42.1 tok/s per user, 2,694 tok/s per GPU |
| 9 | Memory bar full: 70 GB weights + 70.5 GB KV of 141 GB. The 105-user marker lands at the end of the curve. Roofline inset: intensity 103, still left of the ridge (412). | The KV part of the memory bar fills to the top; the curve stops. | At 105 users the GPU's memory is full, and the batch is still far left of the ridge. At this context, KV memory, not arithmetic, caps the batch. | 105 users · step 29.6 ms · 33.7 tok/s per user · 3,543 tok/s per GPU · intensity 103.3 vs ridge 412.3 |
| 10 | A second `curvePlot` labeled "measured, not this toy's model": two InferenceX points for DeepSeek-V4-Pro on GB300 NVL72, with their conditions printed under the plot and the visible line "a different model and GPU; compare the shape (asking each user to go faster costs tokens per GPU), not the numbers". | The two points appear; a solid 1 px `--ink-muted` connector joins them. | Real servers choose a point on this curve. Measured on DeepSeek-V4-Pro and GB300, asking for twice the speed per user roughly halved the tokens per GPU. | 13.1 tok/s/user → 11,056 tok/s/GPU · 27 tok/s/user → 6,182 tok/s/GPU · ISL 8K / OSL 1K, FP4, 2026-05-22 |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter on this file, all captions ≤ 30 words and ≤ 2 sentences, no operators.

## 6. Toy
"Step-time calculator." Choose a phase and a load, watch which side of the roofline you land on. A visible line above the controls: "These are floors from bytes and FLOPs alone. Real engines fall short of the bandwidth floor (no efficiency figure is sourced, so none is printed); attention math, communication and kernel overheads are not modeled."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `phase` | Phase | Toggle | prefill / decode | decode | — |
| `hw` | GPU | Preset chips | H100 (80 GB nominal) · H200 (141 GB nominal) · B200 (180 GB usable of 192 nominal) | H200 | `hardware.json` `h100`, `h200`, `b200`: usable memory (`b200.hbm_usable_gb`; H100 and H200 have none in data, so the toy uses nominal `hbm_gb`), `hbm_tbps`, `bf16_dense_tflops`, `fp8_e4m3_dense_tflops` |
| `weights` | Weight format | Preset chips | BF16 · FP8 | FP8 | `bytesPerElement('bf16' | 'fp8_e4m3')` from `math/roofline.js` (exact `FORMATS` keys; a bare `fp8` throws) |
| `promptTokens` | Prompt tokens (prefill) | Slider (snapped, log) | 1, 16, 64, 217, 512, 1,000, 2,048, 8,192 | 1,000 | — |
| `users` | Users in the batch (decode) | Slider (snapped) | 1, 2, 4, 8, 16, 32, 64, 128, …, "max that fits" | 8 | — |
| `context` | Context per user (decode) | Slider (snapped) | 512, 2,048, 8,192, 32,768, 131,072 tokens | 2,048 | — |

Model is fixed (Llama-3.1-70B: 70e9 parameters, d_model 8,192, 327,680 KV bytes per token from `kvBytesPerToken({ layers: 80, kvHeads: 8, headDim: 128, bytesPerElem: 2 })`). If `users` exceeds the memory limit, the slider clamps to "max that fits" and a visible note says so. If the weights alone do not fit (BF16 on H100: 140 GB > 80 GB), every output is replaced by "does not fit on one GPU: see `serving-calculator`".

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Operations, bytes read (weights · activations · KV) | `stepTime(...).flops`, `.bytes`, `.actBytes`, `.kvBytes` | GFLOP / TFLOP; bytes through `formatBytes` (3 s.f.) |
| Arithmetic intensity, ridge point, tokens per weight read to be compute-bound | `arithmeticIntensity(cost)`, `ridgePoint({ peakTflops, bandwidthTBps })`, `tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k: 8192, n: 8192 })` | FLOP/byte, 1 decimal; tokens |
| Math time, reading time, step time, bound | `stepTime(...)` → `computeS`, `memoryS`, `timeS`, `bound` | `formatDuration` (3 s.f.); "memory-bound" / "compute-bound" |
| Prefill: tokens per second, TTFT for this prompt | `promptTokens / timeS`; TTFT = `timeS` (no queue) | tok/s through `formatInt`; time through `formatDuration` |
| Decode: tokens/s per user, tokens/s per GPU | `1 / timeS`; `users / timeS` | `formatCount` (3 s.f.); `formatInt` |
| Max users that fit | `maxUsersPerGpu(freeHbmPerGpu({ hbmBytes: usable, weightBytes, gpus: 1 }), kvCacheBytes({ bytesPerToken: 327680, tokens: context }))` | `formatInt` |
| Memory bar: weights · KV · free | `weightBytes`, `kvCacheBytes({ bytesPerToken: 327680, tokens: context, sequences: users })`, rest | `formatBytes` |
| Curve: per user vs per GPU, users 1 → max | `stepTime` swept over the snapped user values | `curvePlot`, current point framed |

**Check my work** (default state: decode, H200, FP8, 8 users, 2,048 tokens of context):
```
t_math = 2 × 70e9 × 8 ÷ 1,979 TFLOP/s = 566 µs
t_read = (70 GB weights + 137 MB activations + 5.37 GB KV) ÷ 4.8 TB/s = 75.5 GB ÷ 4.8 TB/s = 15.7 ms
t_step = max(566 µs, 15.7 ms) = 15.7 ms, memory-bound
```

**Try this** (each leads to a named insight)
1. Phase prefill, H200, FP8. Slide the prompt from 1 to 8,192 and watch the bound: memory-bound up to 217 tokens, compute-bound from there (15.4 ms at 217; 70.7 ms at 1,000; 580 ms at 8,192). Switch weights to BF16: the flip stays at 217. Switch to B200: it moves to 302, in BF16 as in FP8. → **Insight: the crossover, counted in tokens per weight read, is set by the GPU's ratio of math to bandwidth.** FP8 halves the bytes and doubles the math rate, so on an H200 it moves the ridge but not the token crossover (the same point `gpu-primer` makes for the H100's 318).
2. Phase decode, context 2,048. Slide users 1 → 8 → 64 → max: per user 67.9 → 63.6 → 42.1 → 33.7 tok/s; per GPU 68 → 509 → 2,694 → 3,543 tok/s. → **Insight: batching trades each user's speed for the GPU's total,** cheaply at first (one shared read of the weights), then more steeply as KV reads grow.
3. Keep users at "max that fits" and slide context 2,048 → 8,192 → 32,768 → 131,072: max users 105 → 26 → 6 → 1; tokens per GPU 3,543 → 890 → 214 → 42. → **Insight: at long context the KV cache, not the weights, decides how many users a GPU can hold,** and with them its throughput. That is the problem `paged-attention`, `kv-compression` and FP8 KV (`quantization`) attack.
4. Switch weights to BF16 on the H200 at context 2,048: 1 GB is left and only 1 user fits (34 tok/s per GPU). → **Insight: the weight format decides whether there is room for users at all.** You'll see what FP8 and FP4 cost in quality in `quantization`.

**Imported from `math/roofline.js`** (owned by `gpu-primer`; signatures as in its §6, not redefined here):
```js
bytesPerElement(format) → number
matmulCost({ m, k, n, bytesPerElem }) → { flops, bytes }          // 2mkn; bytesPerElem · (mk + kn + mn)
arithmeticIntensity({ flops, bytes }) → number
ridgePoint({ peakTflops, bandwidthTBps }) → number
rooflineTime({ flops, bytes, peakTflops, bandwidthTBps }) → { computeS, memoryS, timeS, bound }
tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k, n }) → number
```

**`math/serving.js`** (pure, no DOM; owned by the serving track; every serving lesson imports from here). Sizes in bytes; hardware in `gpu-primer`'s units (TFLOP/s, TB/s). Also exported: `FORWARD_FLOPS_PER_PARAM_TOKEN` (2), `RUNNING_EXAMPLE` (frozen; the `L` below, pinned to `llama-3.1-70b` and `h200` in `data/*.json`), `TOY_REQUESTS` (the A–D table), `hbmFor(entry) → { bytes, basis: 'usable' | 'nominal' }`. `math/core.js` adds `formatInt`.
```js
weightBytes({ params, bitsPerParam }) → number                    // params · bits / 8
stepTime({ activeParamsPerGpu, weightBytesPerGpu, dModel, actBytesPerElem, tokens, seqs, context, kvBytesPerToken, peakTflops, bandwidthTBps })
  → { flops, bytes, actBytes, kvBytes, computeS, memoryS, timeS, bound }
  // one engine iteration on one GPU. The model is activeParams ÷ dModel² square [dModel × dModel] multiplies, so summing
  // matmulCost({ m: tokens, k: dModel, n: dModel }) over them gives: flops = 2 · active · tokens;
  // actBytes = 2 · active · actBytesPerElem · tokens ÷ dModel (each multiply reads X and writes Y);
  // kvBytes = seqs · context · kvBytesPerToken; bytes = weightBytesPerGpu + actBytes + kvBytes; then rooflineTime(...)
  // prefill: tokens = prompt, seqs = 0 · decode: tokens = seqs = users · spec-decode verify: tokens = users · (k + 1), seqs = users
// cache per user: kvCacheBytes({ bytesPerToken, tokens, sequences = 1 }) from math/memory.js (kv-cache), not redefined here
freeHbmPerGpu({ hbmBytes, weightBytes, gpus }) → number          // hbm − weights / gpus (negative: does not fit)
maxUsersPerGpu(freeBytes, cacheBytesPerUser) → number             // floor, ≥ 0
prefillTokPerSecCeiling({ activeParamsPerGpu, peakTflops }) → number    // peak / (2 · active)
requestTimeline(...)                                               // serving-overview §6
usersAtTarget(...), minGpusForWeights(...), costPerMillion(...)   // serving-calculator §6
kvTransferTime(...), tokensPerExpert(...)                          // disaggregation §6
```
With `tokens` equal to `tokensToComputeBound(..., k: dModel, n: dModel)`, `stepTime`'s compute and memory times are equal: the whole-model crossover is `gpu-primer`'s one-multiply crossover, by construction (README lesson 16: one definition of "tokens to compute-bound"). `kvBytesPerToken` / `kvBytesPerTokenMla` are imported from `math/memory.js` (`kv-cache`), never redefined.

Worked examples (scratch implementation of both modules' signatures, 2026-10-07; `L` = `{ activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, dModel: 8192, actBytesPerElem: 1, kvBytesPerToken: 327680, peakTflops: 1979, bandwidthTBps: 4.8 }`):
```
ridgePoint({ peakTflops: 1979, bandwidthTBps: 4.8 }) → 412.3
tokensToComputeBound({ peakTflops: 1979, bandwidthTBps: 4.8, bytesPerElem: 1, k: 8192, n: 8192 }) → 217.1   (BF16 989 / 2 B → 217.0; B200 FP8 4500 / 8 → 302.0; H100 BF16 → 318.2)
stepTime({ ...L, tokens: 1,    seqs: 1, context: 0 }) → compute 0.071 ms, memory 14.587 ms → 14.59 ms, memory-bound, intensity 2.0
stepTime({ ...L, tokens: 217,  seqs: 0, context: 0 }) → compute 15.351, memory 15.356 → memory-bound by a hair; 218 → compute-bound
stepTime({ ...L, tokens: 1000, seqs: 0, context: 0 }) → compute 70.743, memory 18.144 (bytes 87.09 GB) → 70.74 ms, intensity 1,607.5
stepTime({ ...L, tokens: 8192, seqs: 0, context: 0 }) → 579.53 ms
decode, context 2,048 (users: step ms · tok/s per user · tok/s per GPU · intensity · KV read ms):
  1: 14.73 · 67.9 · 68 · 2.0 · 0.14     8: 15.73 · 63.6 · 509 · 14.8 · 1.12     64: 23.76 · 42.1 · 2,694 · 78.6 · 8.95     105: 29.64 · 33.7 · 3,543 · 103.3 · 14.68
maxUsersPerGpu(71e9, kvCacheBytes({ bytesPerToken: 327680, tokens: c })) → 105 (2,048) · 26 (8,192) · 6 (32,768) · 1 (131,072)
at max users: 8,192 → 29.22 ms, 890 tok/s/GPU · 32,768 → 28.03 ms, 214 · 131,072 → 23.53 ms, 42
BF16 on H200 (weightBytesPerGpu 140e9, actBytesPerElem 2, peak 989): free 1.0 GB, max users 1, step 29.31 ms
B200 FP8 (180e9 usable, peak 4500, 8 TB/s): 1 user 8.84 ms; 163 users max at 2,048, 22.77 ms, 7,158 tok/s/GPU (with 192 nominal it would be 181 users: the chip says usable)
prefillTokPerSecCeiling({ activeParamsPerGpu: 70e9, peakTflops: 1979 }) → 14,135.7
```
Reproducer (run once `math/serving.js` and `math/roofline.js` exist):
```
node -e "Promise.all([import('./math/serving.js'), import('./math/roofline.js'), import('./math/memory.js')]).then(([m, r, mem]) => { const L = { activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, dModel: 8192, actBytesPerElem: 1, kvBytesPerToken: 327680, peakTflops: 1979, bandwidthTBps: 4.8 }; console.log(r.ridgePoint({ peakTflops: 1979, bandwidthTBps: 4.8 }), r.tokensToComputeBound({ peakTflops: 1979, bandwidthTBps: 4.8, bytesPerElem: 1, k: 8192, n: 8192 })); for (const t of [1, 217, 1000, 8192]) console.log('prefill', t, m.stepTime({ ...L, tokens: t, seqs: t === 1 ? 1 : 0, context: 0 })); for (const c of [2048, 8192, 32768, 131072]) { const max = m.maxUsersPerGpu(71e9, mem.kvCacheBytes({ bytesPerToken: 327680, tokens: c })); for (const u of [1, 8, 64, max]) if (u <= max) { const s = m.stepTime({ ...L, tokens: u, seqs: u, context: c }); console.log(c, u, (s.timeS * 1e3).toFixed(2), (1 / s.timeS).toFixed(1), Math.round(u / s.timeS)); } } })"
```
Tests to write first: the table above; `timeS === max(computeS, memoryS)`; at `tokens = tokensToComputeBound(...)` compute and memory times agree to 1e−9; decode at one user and zero context equals `(weightBytes + actBytes) / bandwidth`; `maxUsersPerGpu` never negative; `freeHbmPerGpu` negative for BF16 on H100; inputs not mutated.

## 7. Show me the math
```tex
\htmlClass{hl-compute}{t_{\text{math}}} = \frac{2\,N_{\text{active}}\cdot \text{tokens}}{\text{peak FLOP/s}}
\qquad
\htmlClass{hl-memory}{t_{\text{read}}} = \frac{W_{\text{bytes}} + \tfrac{2 N b}{d_{\text{model}}}\cdot\text{tokens} + \text{users}\cdot \text{context}\cdot \text{KV bytes/token}}{\text{HBM bandwidth}}
```
```tex
t_{\text{step}} = \max\!\left(\htmlClass{hl-compute}{t_{\text{math}}},\ \htmlClass{hl-memory}{t_{\text{read}}}\right)
\qquad
m^{*} = \frac{I^{*}\, b\, d^{2}}{2d^{2} - 2 I^{*} b\, d}\quad(\text{tokens to compute-bound, } I^{*} = \text{ridge})
```
```tex
\text{TPOT} = t_{\text{step}}, \qquad
\text{tokens/s per user} = \frac{1}{t_{\text{step}}}, \qquad
\text{tokens/s per GPU} = \frac{\text{users}}{t_{\text{step}}}
```
Shapes: each multiply is `[tokens × 8192] · [8192 × 8192]`; with `tokens = 1` it is a matrix-vector product, which is why decode reuses nothing. `b` = bytes per element (1 for FP8, 2 for BF16). Ignored on this page: attention FLOPs (about 1% of a 1,000-token prefill for this model), KV writes, communication. Color links: `hl-compute` = the arithmetic part of the step bar (`--sem-compute`); `hl-memory` = the reading parts (`--sem-memory`).

Basis (README lessons 25–26): every step on the serving pages is a **forward pass** on **one GPU** (inference has no backward pass), and cross-mechanism ratios (prefill vs decode, transfer vs prefill, input vs output cost) are compared per token on that basis. The factor 2 in `t_math` is `FORWARD_FLOPS_PER_PARAM_TOKEN` in `math/serving.js`, the forward-pass share of training's 6 FLOPs per parameter per token: a test pins it to `FLOPS_PER_PARAM_TOKEN / 3` (`math/scale.js`). Metric definitions shared by every serving lesson (README lesson 16): **step time** = one engine iteration on one GPU = `stepTime(...).timeS`, a floor; **TPOT** = decode step time; **tokens/s per user** = 1 / TPOT; **users per GPU** = sequences in the decode batch held on one GPU; **tokens/s per GPU** = users per GPU / TPOT, output tokens only; **cache per user** = `kvCacheBytes({ bytesPerToken, tokens: context })` (`kv-cache`); **GPU memory** = usable capacity, labeled as such; **tokens to compute-bound** = `tokensToComputeBound` (gpu-primer).

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| H200: 141 GB HBM3e, 4.8 TB/s, 1,979 TFLOP/s dense FP8 (FP8 = 2 × BF16, reported) | `hardware.json/h200.hbm_gb`, `.hbm_tbps`, `.fp8_e4m3_dense_tflops` | 03 §1.4; 04 §1.4 |
| H100 crosses into compute-bound at 318 tokens per weight read; B200 at 302 (BF16 or FP8) | `hardware.json/h100.*`, `b200.*` via `tokensToComputeBound` (gpu-primer's numbers) | 03 §1.2, §1.4 |
| Llama-3.1-70B: 80 layers, 8 KV heads × 128, d_model 8,192, 131,072-token context; 327,680 KV bytes per token in BF16 | `models.json/llama-3.1-70b.layers`, `.n_kv_heads`, `.head_dim`, `.context_length`, `.kv_bytes_per_token`, `.total_params = 70e9`, `.d_model = 8192` *(entry proposed by `paged-attention`; `total_params` and `d_model` added here, see §12)* | 01 §4; 04 §1.4, §3.1 |
| DeepSeek-V4-Pro on GB300 NVL72: 11,056 tok/s/GPU at 13.1 tok/s/user (max throughput) and 6,182 tok/s/GPU at 27 tok/s/user; on GB200, 8,933 at 15.3 and 2,189 at 27 (ISL 8K / OSL 1K, FP4, InferenceX, 2026-05-22). Whether tok/s/GPU counts input tokens too is not stated by the source; the page says so | `serving.json/inferencex-v4-pro-gb300.max_throughput_tok_s_gpu = 11056`, `.max_throughput_tok_s_user = 13.1`, `.throughput_tok_s_gpu = 6182`, `.interactivity_tok_s_user = 27`; same keys on `inferencex-v4-pro-gb200` (8933, 15.3, 2189, 27) *(proposed)* | 04 §1.3, §10 item 4 CONFIRMED |
| Kimi K2.5 on B200 (NVFP4): $0.140/M at 32 tok/s/user vs $0.347/M at 90 tok/s/user | `serving.json/inferencex-kimi-k2.5-b200.cost_per_m_at_32 = 0.140`, `.cost_per_m_at_90 = 0.347` *(proposed)* | 04 §1.3 CONFIRMED |

## 9. Takeaways
1. Decode reads every weight to make one token per request, so it waits on memory (14.6 ms for 70 GB on an H200); prefill reuses each weight for every prompt token, so it waits on math (frames 2–5).
2. The crossover is the GPU's ridge, counted as tokens per weight read: 217 prompt tokens on an H200 take about as long as one decode step, which is why reading a prompt is cheap per token and writing an answer is not (frame 6, try-this 1).
3. Batching decode shares the weight read, raising tokens per GPU while slowing each user; KV memory usually caps the batch before the ridge, and long contexts shrink it fast (frames 7–9, try-this 2–3).

## 10. Next and go deeper
Next: `batching`, `speculative-decoding`, `quantization` · Related: `gpu-primer` (the roofline), `serving-calculator`.

Go deeper (brief 05 §1.2, 04 §9.2): Google, *How to Scale Your Model*, inference chapter (https://jax-ml.github.io/scaling-book/) · InferenceX dashboard (https://inferencex.semianalysis.com/about), live per-user vs per-GPU curves · Lynskey, *LLM Inference Explained* (https://llm-inference-explained.vercel.app).

## 11. Key-frame sketch
Frame 9 end state, desktop width; numbers from the decode rows of the §6 worked examples.
```text
Prefill vs decode               step 9 / 10   [<] [Play] [>]
H200 memory |######## weights 70 GB ########|#### KV 70.5 GB ###|
            141 GB full at 105 users (2,048 tokens each)
step 29.6 ms  |== weights 14.6 ms ==|== KV 14.7 ms ==|act 374 µs|
              |math 7.43 ms (overlapped)|
per GPU tok/s
 3543 |                                   * 105 users
 2694 |                        * 64
  509 |          * 8
   68 |* 1
      +--------------------------------------------------
        33.7   42.1        63.6  67.9   tok/s per user
intensity 103.3  <  ridge 412.3   -> still memory-bound
```

## 12. Open questions for the reviewer
**Graph and cross-track interfaces**
- **Roofline module (settled).** This page imports `math/roofline.js` from `gpu-primer` (signatures above) and uses its "tokens per weight read" framing; `stepTime` models the whole model as `active ÷ d_model²` square multiplies so its crossover equals `tokensToComputeBound(k = n = d_model)`. Frame 5 calls `roofline`, a thin wrapper over `curvePlot`.

**Data-pass keys**
- `models.json/llama-3.1-70b.total_params = 70e9`, note "rounded; Meta's card lists 70.6B (not in the briefs; the data pass may set the exact value, which shifts every ms on this page by under 1%)"; `.d_model = 8192` (from the Llama 3 paper; not in the briefs, needs a source in the data pass).
- `inferencex-v4-pro-gb300.*`, `inferencex-v4-pro-gb200.*`, `inferencex-kimi-k2.5-b200.*` as in §8.
- `hardware.json/b200.hbm_usable_gb = 180` (reported; nominal 192 stays in `hbm_gb`); the B200 chip reads usable memory. Gaps, not printed: a bandwidth-efficiency figure (unsourced) and H100/H200 usable memory (nominal is used, labeled).
- `hardware.json/h200.fp8_e4m3_dense_tflops` is `reported` (2 × BF16); the page's 412 ridge inherits that confidence and says "dense FP8, reported".

**Judgment calls**
- The decode model ignores attention FLOPs and treats every user's whole cache as read each step. For a dense GQA model that is close; for sparse-attention models it overstates KV reads (`long-context-attention`). The visible floor line covers it.
- Treating every weight matrix as 8,192 × 8,192 is a simplification (Llama's K/V projections are 8,192 × 1,024 and its MLP 8,192 × 28,672); it changes the activation bytes by a few percent and keeps one crossover definition across the two tracks.
- Frame 10 shows a different model on different hardware from the toy's running example, under an on-screen "measured, not this toy's model" label, because no measured curve exists for the running example in the briefs.

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled:** cache per user comes from `kvCacheBytes` in `math/memory.js` (`kv-cache`); `kvBytesPerSequence` is gone from `math/serving.js`.
- **Settled:** capacity math uses usable chip memory, labeled; the B200 chip is 180 GB usable of 192 nominal (163 users at 2,048 tokens, 7,158 tok/s per GPU).
- **Settled:** frame 5 draws `roofline`, a wrapper over `curvePlot`; `curvePlot` accepts `bands`.
- **Settled:** `quantization` lists `prefill-decode` as a prereq, so this page's `Next:` adds `quantization`.
- Applied: frame 6 bridges 217 to `gpu-primer`'s 318; frame 7 draws A–D plus "+ 4 others"; step-bar slivers move to a zoomed bar and activations fold into "others" (lesson 19); frame 10's like-with-like line (lesson 21); misconception 4's V4-Pro exception; "multiply and add counted as two"; no internal citation on screen; try-this 1 names B200's BF16 crossover.
- Data pass 2026-10-07: H200 FP8 cites `fp8_e4m3_dense_tflops`; the unsourced 60–80% bandwidth-efficiency claim is dropped (text now says real engines fall short of the floor); H100 and H200 memory is labeled nominal.
