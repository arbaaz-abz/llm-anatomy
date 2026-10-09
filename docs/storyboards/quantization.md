# Quantization (`quantization`)

Track: serving · Section: serving · Prereqs: gpu-primer, prefill-decode
Status: approved (expert review)
Sources: 04 §1.4, §3.5, §5, §8.1, §9.1 (quantization toy); 03 §1.3 (number formats); 05 §1.1 (Grootendorst's visual guide as link-out). Beyond the briefs, read by the storyboard author (Opus 5.5) on 2026-10-07: Rouhani et al., "Microscaling Data Formats for Deep Learning" (arXiv 2310.10537, Algorithm 1: shared exponent = ⌊log₂ max|V|⌋ − emax_elem, elements past the range clamped to the max) and NVIDIA, "Introducing NVFP4" (developer.nvidia.com blog: E2M1 values 0, 0.5, 1, 1.5, 2, 3, 4, 6 with a sign; 16-value blocks with an E4M3 scale plus a per-tensor FP32 scale; 4.5 bits per value; FP8 → NVFP4 on DeepSeek-R1-0528 cost "1% or less" accuracy on most benchmarks). Expert review (Fable 5.1, 2026-10-07) applied; see §13. The blog does not give the per-block scale formula; the toy's NVFP4 rule (the largest value divided by 6, rounded to the nearest FP8 value) is a labeled stand-in.

Number formats themselves (bits, exponent, mantissa, what tensor cores run) are taught in `gpu-primer`, and training in low precision in `training-memory`; this page is about storing and running an already-trained model in fewer bits.

Running example: a row of eight hand-picked weights with one outlier, `w = [0.12, −0.31, 0.05, 0.47, −0.08, 0.22, −0.64, 2.10]`, then `prefill-decode`'s Llama-3.1-70B on an H200 (and a B200 for native FP4).

## 1. Learning objective
After this page you can quantize a block of weights by hand (scale, round, multiply back; frames 2–3), explain why one outlier hurts and why small blocks with their own scales help (frames 4–5, try-this 1), tell MXFP4 from NVFP4 (frame 7), compute a model's bytes from its bits per value (frame 8), and say which kind of quantization speeds up decode and which speeds up prefill (frame 9, try-this 3). You can also name the risk: errors that look harmless on short tests can break long-context recall (frame 10).

## 2. Misconceptions to correct
- **Misconception:** "4-bit means every weight is one of 16 evenly spaced values across the whole model." → **Reality:** each small block gets its own scale, so the grid stretches to fit the block. With one scale for all eight toy weights, three small ones round to zero; with a scale per block of four, only one does and the average error halves (0.064 → 0.032). Corrected by frames 4–5 and try-this 1. (04 §5; 03 §1.3)
- **Misconception:** "FP4 is just INT4 with a different name." → **Reality:** FP4 (E2M1) can only hold 0, 0.5, 1, 1.5, 2, 3, 4 and 6 (with a sign): fine steps near zero, coarse ones far out, matching how weights are spread. Corrected by frame 6. (NVIDIA NVFP4 blog, read 2026-10-07)
- **Misconception:** "Quantizing the weights makes everything faster." → **Reality:** weight-only 4-bit cuts the bytes decode must read (14.7 → 8.35 ms per step on the H200 example) but the math still runs in 16-bit, so on the H200, which has no FP4 tensor cores, a compute-bound 4,096-token prefill takes 580 ms, the same as BF16. FP8 with FP8 math halves it (290 ms). Corrected by frame 9 and try-this 3. (04 §5)
- **Misconception:** "If a quantized model scores fine on benchmarks, it is fine everywhere." → **Reality:** errors add up across many attended tokens. vLLM's naive FP8 KV cache dropped 128K needle-in-a-haystack accuracy from 91% to 13% before a fix; the final configuration recovers 97–98% of baseline AUC at 128K. Corrected by frame 10. (04 §3.5)

## 3. Hook and intuition (final wording)
**Hook:** How can you throw away three-quarters of every weight's bits and still get nearly the same answers, and why bother?

Why bother is the easy half. In `prefill-decode` a decode step was mostly the time to read the weights. Store each weight in 4 bits instead of 16 and there are about a quarter as many bytes to read, so steps get faster, and the freed memory holds more users' KV caches. A 70B model in BF16 fills an H200 and leaves room for one user at 2,048 tokens; in a 4-bit format it leaves room for 151.

The how is a scale and a rounding. Take a small block of weights, divide them by a scale so the largest lands at the top of a tiny grid, round each to the nearest grid point, and store the grid points plus the scale. To use a weight, multiply back. Each weight comes back slightly off, but the errors are small and point in random directions, so their effect on a layer's output mostly cancels. Two things break this. One large weight in a block stretches the scale and crushes its neighbors to zero, which is why modern formats give every 16 or 32 weights their own scale. And errors that are tiny per token can pile up where a model attends over hundreds of thousands of tokens.

The 2026 formats differ in their grids and scales. FP8 keeps 8-bit floating-point numbers and roughly halves bytes and math time compared with BF16. MXFP4 and NVFP4 store 4-bit floats; MXFP4 shares a power-of-two scale across 32 weights, NVFP4 an 8-bit float scale across 16, which costs a little more memory (4.5 bits per weight instead of 4.25) and rounds more finely. The safest 4-bit models are trained to expect their format: gpt-oss and DeepSeek-V4 shipped their weights in 4-bit formats, and Kimi K2.x is reported to ship INT4.

## 4. Visual metaphor
The hero is the eight-weight row as a `vector` (orientation `row`) of `NUMBER_CELL` cells (8 × 43 = 344 px, README lesson 18) with printed values on the value scale. Under it, the quantized codes (a second row of cells, integers or E2M1 values printed), then the restored row and an error row. A block boundary is a plain vertical rule between cells 4 and 5 with a scale chip above each block ("scale 0.067"). A `numberLine` (below) shows the grid the block's values are rounded to: 15 integers for INT4, the 15 signed E2M1 values for FP4, with each weight's dot snapping to its nearest grid point.

For the model-scale frames, `gpu` with a `shareBar` memory bar (weights / KV / free) and `prefill-decode`'s `stepBar` (frame 9).

The followed weight is `0.47` (accent frame) in every frame: it rounds well with its own block's scale and gets clipped by MXFP4's power-of-two scale (frame 7).

Terms introduced (one per frame): bits per weight (1), scale (2), restored value / rounding error (3), outlier (4), block scale (5), E2M1 (6), MXFP4 vs NVFP4 (7), bits per value with scales (8), weight-only vs W8A8 (9), KV quantization (10), quantization-aware training (11). Terms assumed from `gpu-primer`: BF16, FP8 (E4M3), FP4, tensor cores, bytes vs FLOPs; from `prefill-decode`: decode step time, prefill time, users per GPU.

Glyphs used: `vector`, `cell`, `gpu`, `numberLine`, `shareBar` (the memory bar), `stepBar` (frame 9, P4-R8), `bars` (frame 10's accuracy bars, P4-R17), `block` (frame 11's model cards), `token` (frame 10 "needle" chip).

Plain text labels: "hand-picked weights; real blocks hold 16 or 32" (frame 1); block rule and "scale" chips; "this toy's NVFP4 scale: largest value over six, rounded to FP8 (a stand-in; the exact rule is not in our sources)" (frame 6); "decode and prefill rows use different time scales; read the numbers, compare within a row; each bar is one forward pass on one GPU" (frame 9); "KV 671 MB (1 user)" printed beside the BF16 memory bar (frame 8); "H200 has no FP4 tensor cores: 4-bit weights are expanded before the math" (frame 8); the vLLM test conditions (frame 10).

Glyph added to the shared library (P4-R14; built in `shared/glyphs/plots.js` because `shared/**` is frozen for page builders):
- **`numberLine(parent, { x, y, w, lo, hi, grid: [numbers], points: [{ value, snapped, followed }] })`**: a horizontal axis with tick marks at the representable values and dots for the block's values, each with a short drop line to its snapped grid point. Why: `clipLine` has one band and one marker, not a grid; the uneven E2M1 spacing is the point of frame 6. Reused by `gpu-primer` if it draws number formats (its call) and by `quantization`'s toy.

## 5. Animation script
Numbers: `quantizeBlocks(w, { format, blockSize })` and `weightBytes` / `stepTime` (§6).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | The weight row w in 8 cells, `0.47` framed. Label "hand-picked weights; real blocks hold 16 or 32". A byte counter. | Cells fill left to right; the counter counts 2 bytes per cell. | Eight weights in BF16, two bytes each. Decode reads every weight at every step, so fewer bytes per weight means faster steps. | 8 × 16 bits = 16 bytes |
| 2 | A scale chip "scale 0.30" above the row; the `numberLine` shows the 15 integers −7 to 7; each weight's dot (value ÷ scale) slides to its nearest integer. The codes row fills. | Dots move from their exact positions to the nearest tick. | To store a weight in 4 bits, divide by a scale and round to a whole number from −7 to 7. Here the scale is the largest weight over seven. | scale = 2.10 ÷ 7 = 0.30 · codes [0, −1, 0, 2, 0, 1, −2, 7] |
| 3 | The restored row (code × scale) and an error row under it; the three zeroed cells marked by a plain label "became 0". | Restored values type in; error cells fill on the value scale. | Multiply back by the scale and each weight returns with a small rounding error. Three small weights came back as exactly zero. | restored [0, −0.30, 0, 0.60, 0, 0.30, −0.60, 2.10] · mean error 0.064 · 3 zeroed |
| 4 | The 2.10 cell pulses; the number line rescales to show how far apart its ticks are at scale 0.30 (spacing 0.30) compared with the small weights' sizes. | Ticks spread; the small weights' dots crowd between 0 and 1 tick. | The culprit is one large weight, 2.10. It sets the scale for all eight, so the small ones get a grid far too coarse for them. | tick spacing 0.30 vs weights 0.05–0.47 |
| 5 | A block rule splits the row into [0.12, −0.31, 0.05, 0.47] and [−0.08, 0.22, −0.64, 2.10], each with its own scale chip. The first block's codes and restored values update. | The rule drops in; the first block's scale chip changes 0.30 → 0.067 and its dots re-snap. | Give each block of four its own scale. The first block now rounds on a fine grid, and the average error halves. | scales 0.067 and 0.30 · codes [2, −5, 1, 7 · 0, 1, −2, 7] · mean error 0.064 → 0.032 · zeroed 3 → 1 |
| 6 | The number line switches to E2M1: ticks at 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6. Label: "this toy's NVFP4 scale: largest value over six, rounded to FP8 (a stand-in)". Codes become E2M1 values. | Ticks rearrange from even to uneven spacing; dots re-snap. | FP4 stores a tiny floating-point number instead: 0, 0.5, 1, 1.5, 2, 3, 4 or 6, with a sign. Its steps are finest near zero, where most weights sit. | NVFP4-style, blocks of 4: scales 0.0781 and 0.3438 · codes [1.5, −4, 0.5, 6 · 0, 0.5, −2, 6] · mean error 0.029 |
| 7 | Two rows compared, both blocks of 4: "MXFP4 (power-of-two scale)" and "NVFP4 (8-bit float scale)". In the MX row, `0.47` (framed) is clipped: its dot overshoots 6 and clamps. | The MX scale snaps from 0.0781 to the power of two 0.0625; the followed dot overshoots the end of the line and is pulled back to 6. | MXFP4's scale must be a power of two, so a block's largest weight can land past 6 and be clipped. NVFP4's 8-bit float scale fits closer: error 0.029, not 0.062. | MX scale 2⁻⁴ = 0.0625: 0.47 → 7.52 → clamped to 6 → 0.375 · NV scale 0.0781: 0.47 → 6.02 → 6 → 0.469 · mean error MX 0.062, NV 0.029 |
| 8 | Zoom out to `gpu` "H200" with the memory `shareBar` (weights / KV at 2,048 tokens / free) for four formats, one under another; label "H200 has no FP4 tensor cores: 4-bit weights are expanded before the math". The BF16 bar's one-user KV (671 MB of 141 GB, under 18 px) is printed beside the bar as "KV 671 MB (1 user)" rather than drawn to scale. | Bars swap as the format chip steps BF16 → FP8 → NVFP4. | Counting the scales, NVFP4 costs 4.5 bits per weight. The 70B model shrinks from 140 GB to 39 GB, and the H200 fits 151 users instead of one. | bits: BF16 16, FP8 8, MXFP4 4.25, NVFP4 4.5 · weights 140 / 70 / 39.4 GB · users at 2,048 tokens: 1 / 105 / 151 |
| 9 | Two `stepBar`s per format (`prefill-decode`'s glyph): decode (1 user) and prefill (4,096 tokens) on the H200: BF16, FP8 (FP8 math), 4-bit weights with 16-bit math. The decode row and the prefill row each have their own ms scale, with the visible line "decode and prefill rows use different time scales; read the numbers, compare within a row"; every bar prints its ms. | Decode bars shrink with the bytes; the 4-bit prefill bar stays as long as BF16's. | Smaller weights speed up memory-bound decode. Compute-bound prefill gets faster only if the math itself runs in low precision, as FP8 does here and FP4 does on Blackwell. | decode 29.3 / 14.7 / 8.35 ms · prefill 580 / 290 / 580 ms · B200 native FP4: prefill 63.7 ms |
| 10 | A `token` "needle" chip deep in a long row of faint context; under it a `bars` chart (P4-R17) with two bars "BF16 KV 91%" and "naive FP8 KV 13%" and the test's conditions. The memory bar shows KV halved. | The KV part of the memory bar halves; then the accuracy bar drops from 91 to 13 and the label "fixed: 97–98% of baseline AUC" appears. | The KV cache can be stored in FP8 too, which doubles the users that fit. Done naively, it broke long-context recall in vLLM's tests until a fix restored it. | users at 2,048 tokens: 105 → 211 · 128K needle test: 91% → 13% (naive), then fixed: 97–98% of baseline AUC at 128K on Llama-3.3-70B (vLLM, 2026-04-22) |
| 11 | Three model cards as `block`s: "gpt-oss-120b (116.83B): MXFP4 MoE weights, fits one 80 GB GPU", "DeepSeek-V4-Pro: FP4 experts, FP8 the rest, ~865 GB", "Kimi K2.x: INT4 weights (quantization-aware; reported)". | The cards fade in one by one. | The safest 4-bit models are trained to expect it. gpt-oss and DeepSeek-V4 shipped their weights in 4-bit formats, and Kimi K2.x is reported to ship INT4. | gpt-oss-120b 116.83B params on one 80 GB GPU · V4-Pro ≈ 865 GB (reported) |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter, all ≤ 30 words and ≤ 2 sentences.

## 6. Toy
Two panels. "Round a block" works on the eight weights; "Shrink a model" works on the running example. A visible line: "The eight weights are hand-picked, with one outlier on purpose. Rounding is round-to-nearest; GPTQ and AWQ choose roundings more cleverly." A second visible line in the "Round a block" panel: "On a hand-sized block INT4 can match FP4's error; FP4's advantage is that Blackwell tensor cores run it natively (frame 9)."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `format` | Format | Preset chips | INT4 · MXFP4 · NVFP4 | INT4 | — |
| `blockSize` | Weights per scale | Preset chips | 8 · 4 · 2 | 8 | — |
| `outlier` | Last weight | Toggle | 2.10 (outlier) / 0.30 | 2.10 | — |
| `modelFormat` | Model weights | Preset chips | BF16 · FP8 · 4-bit weights, 16-bit math · NVFP4 · MXFP4 | FP8 | — |
| `hw` | GPU | Preset chips | H200 (141 GB) · B200 (180 GB usable of 192 nominal) | H200 | `hardware.json` `h200`, `b200` (usable memory, labeled) |
| `kv` | KV cache | Toggle | BF16 / FP8 | BF16 | — |

NVFP4 and MXFP4 model chips are disabled on the H200 with a visible note ("no FP4 tensor cores; use 4-bit weights, 16-bit math"). "4-bit weights, 16-bit math" is counted at 4.5 bits (4-bit codes + one 8-bit scale per 16), the same bytes as NVFP4, so the two chips differ only in where the math runs.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Codes, scales, restored row, error row, number line | `quantizeBlocks(w, { format, blockSize })` → `blocks[{ scale, codes, restored }]`, `restored`, `err` | `NUMBER_CELL` cells, 2–3 decimals |
| Mean error, weights rounded to zero, clipped weights | `.meanAbsErr`, `.zeroed`, `.clipped` | 3 decimals; counts |
| Bits per weight including scales, for the real formats | `bitsPerElement('bf16' | 'fp8_e4m3' | 'mxfp4' | 'nvfp4')` (exact `FORMATS` keys; a bare `fp8` throws) from `math/roofline.js` (gpu-primer) | 2 decimals |
| Model bytes, free memory, users at 2,048 tokens | `weightBytes({ params: 70e9, bitsPerParam: bitsPerElement(format) })`, `freeHbmPerGpu`, `maxUsersPerGpu(free, kvCacheBytes({ bytesPerToken: kvBytes, tokens: 2048 }))` (`kv-cache`) with kvBytes 327,680 (BF16) or 163,840 (FP8) | `formatBytes`; `formatInt` |
| Decode step (1 user, 2,048 context) and prefill (4,096 tokens) | `stepTime({ …, dModel: 8192, actBytesPerElem, peakTflops })` with `peakTflops` from the math precision's column of `hardware.json` (`bf16_dense_tflops`, `fp8_e4m3_dense_tflops`, `nvfp4_dense_tflops` or, on mi355x, `mxfp4_dense_tflops`) and `actBytesPerElem` = `bytesPerElement` of that precision (keys `bf16`, `fp8_e4m3`, `nvfp4`, `mxfp4`) | `formatDuration` (3 s.f.) |

**Check my work** (default state: INT4, block 8, last weight 2.10; the followed weight 0.47):
```
scale = max|w| / 7 = 2.10 / 7 = 0.30
code = round(w / scale) = round(0.47 / 0.30) = 2
restored = code × scale = 2 × 0.30 = 0.60
```

**Try this** (each leads to a named insight)
1. INT4, block 8: mean error 0.064, 3 weights zeroed. Switch the outlier off (last weight 0.30): error 0.024. Outlier back on, block 4: 0.032, 1 zeroed. → **Insight: one outlier ruins a shared scale, and smaller blocks contain the damage.** That is why 2026 formats scale every 16 or 32 weights (DeepSeek's FP8 weights use 128 × 128 tiles).
2. Blocks of 4: INT4 0.032 · MXFP4 0.062 (0.47 clipped to 0.375) · NVFP4 0.029. Blocks of 8: INT4 0.064 · MXFP4 0.073 · NVFP4 0.049. Blocks of 2: INT4 0.011 · MXFP4 0.054 (two clipped) · NVFP4 0.018. → **Insight: the scale's precision matters as much as the grid.** A power-of-two scale is cheap to store but can waste range or clip; an 8-bit float scale fits each block more closely, for 0.25 more bits per weight.
3. Shrink a model, H200: BF16 → FP8 → 4-bit weights. Decode 29.3 → 14.7 → 8.35 ms; prefill 580 → 290 → 580 ms; users 1 → 105 → 151. Switch to B200 (180 GB usable) and NVFP4: prefill 63.7 ms, decode 5.01 ms, 209 users. → **Insight: bytes speed up decode; only low-precision math speeds up prefill.** Weight-only 4-bit is a decode trick; native FP8 or FP4 helps both.
4. On the H200 with FP8 weights, flip the KV cache to FP8: users 105 → 211. → **Insight: at long context the KV cache is the bigger target,** and its quantization needs its own care (frame 10).

**`math/quant.js`** (pure, no DOM; `FORMATS`, `bitsPerElement` and `bytesPerElement` are imported from `math/roofline.js` (gpu-primer) and not redefined; `weightBytes`, `stepTime`, `hbmFor` (usable memory when present, else nominal, with its basis word) and friends from `math/serving.js`):
```js
E2M1_GRID = [0, 0.5, 1, 1.5, 2, 3, 4, 6]               // magnitudes; sign stored separately
roundToGrid(x, grid) → number                          // nearest magnitude, sign kept
roundE4M3(x) → number                                  // 3 mantissa bits; 0 → 0 (an all-zero block gets scale 0 and codes 0);
                                                       // |x| below E4M3's normal range (2⁻⁶) throws RangeError (toy scales never get there)
quantizeBlocks(values, { format: 'int4' | 'mxfp4' | 'nvfp4', blockSize })
  → { blocks: [{ scale, codes, restored }], restored, err, meanAbsErr, zeroed, clipped }
  // int4:  scale = max|v| / 7, code = clamp(round(v / scale), −7, 7)
  // mxfp4: scale = 2^(⌊log2 max|v|⌋ − 2)  (MX paper Algorithm 1, emax(E2M1) = 2), code = roundToGrid(clamp(v / scale, −6, 6))
  // nvfp4: scale = roundE4M3(max|v| / 6) (stand-in rule; per-tensor scale = 1), code = roundToGrid(clamp(v / scale, −6, 6))
  // restored = code · scale (4 decimals); clipped = values that land past the grid's last rounding boundary,
  //   |v / scale| > 7 for E2M1 (midway from 6 to a missing 8) or > 7.5 for int4, so the clamp changed the result
// bits per weight with scales: bitsPerElement(format) from math/roofline.js ('nvfp4' → 4.5, 'mxfp4' → 4.25)
```
Worked examples (scratch implementation, 2026-10-07; `w` = the row above):
```
quantizeBlocks(w, { format: 'int4', blockSize: 8 })  → scale 0.30, codes [0,−1,0,2,0,1,−2,7], restored [0,−0.3,0,0.6,0,0.3,−0.6,2.1], meanAbsErr 0.0638, zeroed 3
quantizeBlocks(w, { format: 'int4', blockSize: 4 })  → scales 0.0671, 0.30; codes [2,−5,1,7 | 0,1,−2,7]; restored [0.1343,−0.3357,0.0671,0.47 | 0,0.3,−0.6,2.1]; 0.0321, zeroed 1
quantizeBlocks(w, { format: 'mxfp4', blockSize: 8 }) → scale 0.5, codes [0,−0.5,0,1,0,0.5,−1.5,4], restored [0,−0.25,0,0.5,0,0.25,−0.75,2]; 0.0725, zeroed 3
quantizeBlocks(w, { format: 'mxfp4', blockSize: 4 }) → scales 0.0625, 0.5; codes [2,−4,1,6 | 0,0.5,−1.5,4]; restored [0.125,−0.25,0.0625,0.375 | 0,0.25,−0.75,2]; 0.0616, zeroed 1, clipped 1 (0.47)
quantizeBlocks(w, { format: 'nvfp4', blockSize: 8 }) → scale 0.3438, codes [0.5,−1,0,1.5,0,0.5,−2,6]; 0.0493, zeroed 2
quantizeBlocks(w, { format: 'nvfp4', blockSize: 4 }) → scales 0.0781, 0.3438; codes [1.5,−4,0.5,6 | 0,0.5,−2,6]; restored [0.1172,−0.3125,0.0391,0.4688 | 0,0.1719,−0.6875,2.0625]; 0.0288, zeroed 1, clipped 0
blockSize 2: int4 0.0105 (zeroed 0) · mxfp4 0.0536 (clipped 2) · nvfp4 0.0175
outlier off (last weight 0.30), block 8: int4 0.0241 · mxfp4 0.0394 · nvfp4 0.0171
bitsPerElement('nvfp4') → 4.5 · bitsPerElement('mxfp4') → 4.25 (gpu-primer's worked examples)
weightBytes({ params: 70e9, bitsPerParam }): 16 → 140 GB · 8 → 70 GB · 4.5 → 39.4 GB · 4.25 → 37.2 GB (formatBytes)
H200 (2,048 context): users 1 / 105 / 151 (BF16 / FP8 / 4.5-bit); FP8 KV: 2 / 211 / 302; decode 29.3 / 14.7 / 8.35 ms; prefill 4,096: 580 / 290 / 580 ms
B200 (180e9 usable): BF16 59 users, decode 17.6 ms, prefill 255 ms · FP8 163 users, decode 8.84 ms, prefill 127 ms · NVFP4 209 users, decode 5.01 ms, prefill 63.7 ms · MXFP4 212 users, decode 4.73 ms, prefill 63.7 ms (FP8 KV: 119 / 327 / 419 / 425)
weightBytes({ params: 1.6e12, bitsPerParam }): 16 → 3.2 TB · 8 → 1.6 TB · 4.5 → 900 GB (V4-Pro ships ≈ 865 GB, reported)
```
Reproducer (run once `math/quant.js` exists):
```
node -e "import('./math/quant.js').then(m => { const w = [0.12, -0.31, 0.05, 0.47, -0.08, 0.22, -0.64, 2.10]; for (const f of ['int4', 'mxfp4', 'nvfp4']) for (const b of [8, 4]) { const r = m.quantizeBlocks(w, { format: f, blockSize: b }); console.log(f, b, r.blocks.map(x => [x.scale, x.codes]), r.restored, r.meanAbsErr, r.zeroed, r.clipped); } })"
# data pass 2026-10-07: Kimi K2.5 cost gain, B200 NVFP4 vs H200 INT4, low end 2.71 (was 2.75); data holds [2.71, 2.95]
node -e "console.log(require('./data/serving.json').entries.find(e => e.id === 'inferencex-kimi-k2.5-b200').facts.cost_gain_vs_h200.value)"
```
Tests to write first: the table above; every NVFP4/MXFP4 code is in ±E2M1_GRID; INT4 codes are integers in [−7, 7]; INT4 and NVFP4 never report `clipped` (their scale comes from the block's largest value), MXFP4 can; `meanAbsErr` with blockSize 1 is 0 for INT4 (each value is its own max); inputs not mutated.

## 7. Show me the math
```tex
\htmlClass{hl-scale}{s} = \frac{\max_i |w_i|}{q_{\max}},\qquad
\htmlClass{hl-code}{q_i} = \operatorname{round}_{\text{grid}}\!\left(\operatorname{clamp}\!\left(\frac{w_i}{\htmlClass{hl-scale}{s}}, -q_{\max}, q_{\max}\right)\right),\qquad
\hat w_i = \htmlClass{hl-code}{q_i}\,\htmlClass{hl-scale}{s}
```
```tex
\text{INT4: } q_{\max}=7,\ \text{grid}=\mathbb{Z}
\qquad
\text{E2M1: grid} = \{0, 0.5, 1, 1.5, 2, 3, 4, 6\},\ q_{\max}=6
\qquad
\text{MX: } \htmlClass{hl-scale}{s} = 2^{\lfloor \log_2 \max|w| \rfloor - 2}
```
```tex
\text{bits per weight} = b_{\text{elem}} + \frac{b_{\text{scale}}}{\text{block}}
\quad(\text{NVFP4: } 4 + 8/16 = 4.5;\ \text{MXFP4: } 4 + 8/32 = 4.25),
\qquad
\text{bytes} = \text{params}\cdot\frac{\text{bits}}{8}
```
Shapes: a weight matrix `[d_in × d_out]` is cut into blocks of 16 or 32 consecutive values along one dimension, one scale each (DeepSeek-V4's FP8 weights use 128 × 128 tiles instead). Color links: `hl-scale` = the scale chips above each block; `hl-code` = the codes row; `ŵ` = the restored row.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Weight-only INT4 (GPTQ, AWQ): about 4× fewer bytes read in decode, math still 16-bit, so less useful at large compute-bound batches | `serving.json/quant-weight-only.note` *(proposed; GPTQ/AWQ marked "prior" in the brief)* | 04 §5 |
| FP8 W8A8 (E4M3, block scales) is native in DeepSeek V3/V4 checkpoints: 128 × 128 blocks with UE8M0 scales; roughly 2× BF16 in bytes and math | `models.json/deepseek-v4-pro.weight_formats = "FP4 experts + FP8 rest (FP8 block 128×128, UE8M0 scales)"` *(proposed)* | 04 §5 CONFIRMED (config.json) |
| MXFP4: 4-bit E2M1, 32-value blocks, power-of-two (E8M0) scale; NVFP4: 16-value blocks, E4M3 scale + FP32 per-tensor scale, ~4.5 bits per value, native on Blackwell | `hardware.json/formats.mxfp4_block_size` = 32, `.mxfp4_scale_bits` = 8 (E8M0); `.nvfp4_block_size` = 16, `.nvfp4_scale_format` = "E4M3" (the per-tensor FP32 scale is in its note), `.nvfp4_bits_per_value` = 4.5 | 04 §5; 03 §1.3 CONFIRMED |
| gpt-oss-120b shipped post-trained with MXFP4 MoE weights: 116.83B total / 5.13B active (`formatCount(total, { digits: 5 })` and `formatCount(active)`, as `decoder-anatomy` prints it), fits one 80 GB GPU | `models.json/gpt-oss-120b.total_params`, `.active_params`; `.weight_format = "MXFP4 (MoE weights)"`, `.fits_gpu_gb = 80` *(proposed)* | 04 §5 CONFIRMED |
| DeepSeek-V4-Pro: experts FP4, other weights FP8, trained with QAT; checkpoint ≈ 865 GB (size reported) | `models.json/deepseek-v4-pro.checkpoint_gb = 865` *(proposed, `reported`)* | 04 §5, §8.1 |
| Kimi K2 Thinking / K2.5 / K2.6 ship native INT4 MoE weights (QAT) | `models.json/kimi-k2.5.weight_format = "INT4 (QAT)"` *(proposed entry, `reported`)* | 04 §5 REPORTED |
| Precision plus hardware: Kimi K2.5 on B200 NVFP4 vs H200 INT4 gave 2.71–2.95× lower $ per M tokens (InferenceX) | `serving.json/inferencex-kimi-k2.5-b200.cost_gain_vs_h200 = [2.71, 2.95]` (the article's range; the brief said 2.75) | 04 §5 CONFIRMED |
| FP8 → NVFP4 on DeepSeek-R1-0528: 1% or less accuracy loss on most benchmarks (NVIDIA's own measurement) | `serving.json/nvfp4.r1_accuracy_loss_pct_max = 1` (`reported`: vendor claim; source NVIDIA blog read 2026-10-07)* | beyond the brief (header) |
| vLLM FP8 KV cache (2026-04-22): halves KV bytes; a naive kernel dropped 128K needle-in-a-haystack from 91% to 13%; the final configuration recovers 97–98% of the baseline AUC at 128K for Llama-3.3-70B (the post's words; it does not tie the figure to one fix); +14.9% throughput on Llama; break-even ~7K tokens, avoid for short contexts | `serving.json/vllm-fp8-kv.bytes_factor = 0.5`, `.naive_niah_128k_pct = 13` *(proposed by `paged-attention`)*; `.baseline_niah_128k_pct = 91`, `.auc_recovery_128k_pct = [97, 98]`, `.throughput_gain_llama_pct = 14.9`, `.breakeven_tokens = 7000` | 04 §3.5 CONFIRMED |

Not shown: MXFP4 vs NVFP4 model-quality head-to-heads (UNVERIFIED in the brief; the toy compares rounding error on eight numbers only, and says so).

## 9. Takeaways
1. Quantizing is scale, round, multiply back; giving every small block its own scale keeps one outlier from crushing its neighbors (frames 2–5, try-this 1).
2. FP4 formats round to an uneven grid; NVFP4's 8-bit scale per 16 weights fits more closely than MXFP4's power-of-two scale per 32, at 4.5 vs 4.25 bits (frames 6–8, try-this 2).
3. Fewer weight bytes speed up decode and free memory for users; only low-precision math speeds up prefill; and long-context accuracy is where careless quantization fails (frames 8–10, try-this 3–4).

## 10. Next and go deeper
Next: `serving-calculator` · Related: `gpu-primer` (the formats and tensor cores), `training-memory` (training in low precision), `prefill-decode`, `paged-attention` (FP8 KV per block).

Go deeper (brief 05 §1.2): Maarten Grootendorst, "A Visual Guide to Quantization" (https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) · NVIDIA, "Introducing NVFP4 for efficient and accurate low-precision inference" (https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) · vLLM, "FP8 KV cache" (https://vllm.ai/blog/2026-04-22-fp8-kvcache).

## 11. Key-frame sketch
Frame 7 end state, desktop width; numbers from the blocks-of-4 rows in §6. `[ ]` = followed weight.
```text
Quantization                    step 7 / 11   [<] [Play] [>]
w        0.12 -0.31  0.05 [0.47]| -0.08  0.22 -0.64  2.10
MXFP4  scale 0.0625             | scale 0.5
codes     2    -4     1   [6*] |   0    0.5  -1.5    4
restored 0.13 -0.25  0.06 [0.38]|  0    0.25 -0.75  2.00
NVFP4  scale 0.0781             | scale 0.3438
codes    1.5  -4    0.5  [6]   |   0    0.5   -2     6
restored 0.12 -0.31  0.04 [0.47]|  0    0.17 -0.69  2.06
* clipped: 0.47 / 0.0625 = 7.52 > 6
mean error  MXFP4 0.062   NVFP4 0.029
```

## 12. Open questions for the reviewer
**Graph**
- **Prerequisites (settled):** `quantization.prereqs = ["gpu-primer", "prefill-decode"]`, which reaches `kv-cache` through `serving-overview`; the header matches. Frame 10 still links to `kv-cache`.

**Sources beyond the briefs**
- The MXFP4 scale rule (MX paper Algorithm 1) and the E2M1 value set (NVIDIA blog) were read by the storyboard author on 2026-10-07. The NVFP4 per-block scale (largest value → 6, rounded to E4M3) is a stand-in, labeled on screen; if the data pass finds NVIDIA's exact rule, `quantizeBlocks` should use it (frame 6–7 numbers may shift in the third decimal).

**Data-pass keys**
- All proposed keys in §8, including a new `models.json/kimi-k2.5` entry (or attach the INT4 fact to an existing Kimi entry if the data pass prefers).

**Judgment calls**
- **"4-bit weights, 16-bit math" at 4.5 bits.** The toy counts weight-only 4-bit at the same 4.5 bits as NVFP4 (one 8-bit scale per 16) so the comparison isolates where the math runs; real GPTQ/AWQ group sizes and scale formats vary and are not in the briefs.
- **INT4 symmetric −7..7.** Many INT4 schemes use −8..7 or an offset; symmetric keeps the toy hand-checkable and is stated in the toy's line.
- The page teaches inference quantization only; training formats and FP8/FP4 tensor cores are linked to `gpu-primer` and `training-memory` per the dispatch.

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled:** prereqs are `gpu-primer` and `prefill-decode` (step time, users per GPU and the running example come from `prefill-decode`).
- **Settled:** capacity math uses usable chip memory, labeled: B200 180 GB usable of 192 nominal (users 59 / 163 / 209 / 212 for BF16 / FP8 / NVFP4 / MXFP4 at 2,048 tokens).
- **Settled:** cache size comes from `kvCacheBytes` (`kv-cache`); formats from `math/roofline.js`.
- **Settled:** the NVFP4 scale rule is a labeled stand-in; clipping means "past the grid's last rounding boundary".
- Applied: Kimi K2.x's INT4 marked reported (§3, frame 11); frame 9's two rows on their own scales with the visible line (lessons 19, 21); frame 8 prints the BF16 KV sliver; the "INT4 can match FP4 on a small block" line (lesson 17); the label without the ÷ operator; misconception 3 names the H200; gpt-oss at 116.83B; `roundE4M3`'s below-range behavior; DeepSeek's 128 × 128 tiles in try-this 1.

- Data pass 2026-10-07: format facts cite `hardware.json/formats` flat keys (not the removed serving-file mxfp4 and nvfp4 entries); Kimi K2.5 B200-vs-H200 cost gain is 2.71–2.95× (was 2.75); FLOPS columns use the renamed keys.
