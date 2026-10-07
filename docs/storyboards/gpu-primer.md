# A GPU for LLM people (`gpu-primer`)

Track: training · Section: gpus · Prereqs: none (matches `shared/concepts.json`; the GPUs & scale section's front door)
Next: `training-memory`, `prefill-decode`, `quantization` (the three slugs whose `prereqs` list `gpu-primer`)
Status: draft
Sources: 03 §1.1, §1.2, §1.3, §1.4, §7.1 (toys 2, 10), §7.2 · 02 §1.7 · 05 §1.1, §1.2, §2

This page owns the roofline. `prefill-decode` reuses its plot, its vocabulary (arithmetic intensity,
ridge point, memory-bound, compute-bound) and its module `math/roofline.js`; `quantization` reuses its
number-format table (`FORMATS`, `bitsPerElement`). Neither page re-teaches them.

Running example: the last multiply of `attention`, the four tokens "The cat sat down" (rows, d_model 8)
times W_O [8 × 8]; row "sat" is the followed row, as on every Architecture page. Frame 8 then keeps the
four tokens and swaps in a real-sized 8,192 × 8,192 weight matrix.

## 1. Learning objective
After this page you can say what limits a GPU on a given operation: count its FLOPs and bytes
(frame 4), divide to get its arithmetic intensity (frame 5), compare that with the chip's ridge point
(frame 6) and call it memory-bound or compute-bound (frames 7–8, try-this 1); explain why the number of
tokens sharing each weight read is the lever (frame 8); and explain what BF16, FP8 and FP4 change: bytes
per number, tensor-core rate, and so the tokens needed to keep the tensor cores busy (frames 9–11,
try-this 2–3).

## 2. Misconceptions to correct
- **Misconception:** "A GPU rated at 989 TFLOPS does 989 trillion operations a second on any matmul." →
  **Reality:** that is a ceiling. A multiply that moves many bytes per FLOP waits on HBM: four tokens
  through an 8,192 × 8,192 matrix run at 1.35% of an H100's peak. Corrected by frame 7 and try-this 1.
  (03 §1.2)
- **Misconception:** "Arithmetic intensity is a property of the chip." → **Reality:** intensity belongs
  to the operation (FLOPs per byte it moves) and changes with its shape, above all with how many tokens
  share each weight read. The chip contributes the ridge point. Corrected by frames 5–6 and 8. (03 §1.2)
- **Misconception:** "Newer GPUs make memory-bound operations rarer." → **Reality:** the BF16 ridge point
  has barely moved since 2022 (H100 295, B200 281, B300 313 FLOPs per byte), because bandwidth grew about
  as fast as BF16 compute. Only the FP4 tensor cores outgrew their byte savings, and they need *more*
  tokens per weight read (B300: 338 tokens in BF16, 605 in NVFP4). Corrected by frame 11 and try-this 3.
  (03 §1.2, §1.4; derived)
- **Misconception:** "FP8 just halves memory." → **Reality:** it also doubles the tensor-core rate
  (H100, as listed), so a matmul gets about 2× faster whether it is memory-bound or compute-bound, and
  the tokens needed to cross the ridge stay the same (318 on an H100). Corrected by frame 9 and try-this
  2. (03 §1.3, §1.4)
- **Misconception:** "FP4 numbers take 4 bits." → **Reality:** 4 bits hold only 16 values, so blocks of
  numbers share a scale: NVFP4 is 4.5 bits per number (one 8-bit scale per 16), MXFP4 4.25 (one per 32),
  which is where gpt-oss's "about 4.25 bits per parameter" comes from. Corrected by frame 10. (03 §1.3;
  02 §1.7)

## 3. Hook and intuition (final wording)
**Hook:** Why does an H100, rated at 989 trillion operations a second, run a four-token matrix multiply
at about 1% of that speed?

A GPU is two machines glued together. One is compute: 132 small processors on an H100, each with tensor
cores that do nothing but multiply matrices, fast. The other is memory: stacks of HBM beside the chip
that hold the weights and everything else the GPU works on. Every operation pays twice, once to move its
numbers between HBM and the processors and once to do the math, and on a modern GPU the math is the
cheap part. An H100 can do 295 operations in the time it takes HBM to deliver one byte.

So the question for any operation is how much math it does per byte it moves: its arithmetic intensity.
A matrix multiply reads each weight once and uses it once for every token in the batch. With four
tokens, each weight byte buys about four operations, far below 295, and the tensor cores sit idle 98.6%
of the time, waiting on memory. That is the hook's answer: the chip isn't slow, the multiply is starved.
With 4,096 tokens the same weights buy 2,048 operations per byte and the tensor cores become the limit.
Training pushes millions of tokens through each weight per step, so it lives on the compute side;
generating text one token at a time for a few users lives on the memory side (`prefill-decode`).

Smaller number formats move both sides. FP8 halves the bytes and, on an H100, doubles the tensor-core
rate, so most multiplies simply run twice as fast. FP4 is the twist: its tensor cores grew faster than
its byte savings, so a B300 needs roughly twice as many tokens per weight read before FP4 math, rather
than memory, sets the pace. The price of low precision is range and resolution: FP4 has 16 values, so
blocks of numbers share a scale, and accuracy is a recipe question (`quantization`, `scale-reliability`).

## 4. Visual metaphor
One `gpu` glyph (die with SM grid in `--sem-compute`, HBM stacks in `--sem-memory`, as the library draws
it) stays at the left of the stage for frames 1–4; numbers travel between its HBM and SM grid as `flow`
dots. From frame 6 the right side of the stage is a log-log **roofline**: x = arithmetic intensity
(FLOPs per byte), y = attainable TFLOPS; the sloped roof is `--sem-memory`, the flat roof
`--sem-compute`, and the ridge point is printed at the bend. The operation being followed is one dot
with the selection outline in every frame (it is the "sat"-row matmul in frames 4–7 and the same four
tokens at real size from frame 7). Time comparisons are two lanes on one time axis (compute, memory).

Glyphs used (from spec §5.1 and the built library): `gpu` (frames 1–4; `memFill` 0, its memory bar is
not used on this page: what HBM *holds* is `training-memory`'s subject), `block` (frame 3: one SM zoomed,
"tensor cores" and "SRAM" as two labeled blocks inside it), `matrix` (X [4 × 8], W_O [8 × 8], Y [4 × 8],
20 px cells without numbers: shapes only, the counted numbers print beside them at `NUMBER_CELL` size),
`token` (the four row chips, "sat" with the selection outline), `flow` (carry `activation` for X and Y,
HBM ↔ SM; a plain labeled arrow for W_O, see open question 5), `cell` (`NUMBER_CELL` readouts for bytes,
FLOPs and intensity).

New glyphs proposed:
- `roofline(parent, { x, y, w, h, peakTflops, bandwidthTBps, xDomain: [0.1, 1e4], yDomain, ridgeRange,
  points: [{ intensity, label, followed }] })`: log-log axes with decade ticks, the two roof segments
  (`--sem-memory` sloped, `--sem-compute` flat), the ridge intensity printed at the bend (or a shaded
  band with both ends printed when `ridgeRange` is given, for Rubin), and each point as a dot with its
  label and intensity printed beside it. Why: no library glyph plots one quantity against another on
  log axes. Reused by `prefill-decode` (prefill vs decode dots on the same plot), `quantization` (weight
  format moves the decode dot) and `serving-calculator`.
- `laneTimeline(parent, { x, y, unit, lanes: [{ label, segments: [{ start, len, kind, text }] }] })`:
  horizontal lanes on one shared time axis; `kind` ∈ `compute | memory | comm | forward | backward |
  idle | save | lost | restart` maps to a fill (`--sem-compute`, `--sem-memory`, …), `idle` is hatched with the
  `heatmap` mask hatch; `text` prints inside a segment only when it is ≥ 36 px wide, otherwise beside
  it. Why: `request` draws one request's prefill/decode ticks, not several resources sharing a clock.
  Reused by `parallelism` (pipeline schedule), `scale-reliability` (overlap lanes; checkpoint and
  failure timeline) and possibly `batching`.
- `bitLayout(parent, { x, y, fields: [{ role: 'sign' | 'exponent' | 'mantissa' | 'scale', bits }],
  bitW = 14, label, sharedBy })`: a row of bit cells grouped and labeled by role; `sharedBy` draws a
  bracket "shared by 16 numbers" under a scale field. Why: formats are bit fields, not values; `vector`
  would color them by value. Reused by `quantization` and `scale-reliability` (FP8 tiles).

Plain labeled marks (no glyph): "dense numbers; vendor sparse figures are 2× and not used" (frame 2);
"toy size: inputs and outputs are half the bytes; at real size the weights are 99.9%" (frame 7);
"bandwidth: sources conflict, 19.2 or 22 TB/s" under any Rubin point (frame 11, toy); the "H100" chip
name over the plot.

Terms introduced (one per frame, defined on screen where first shown): HBM (frame 1), tensor core
(2), SRAM (3), FLOP (4), arithmetic intensity (5), ridge point (6), memory-bound (7), compute-bound (8),
FP8 (9), block scale (10). Frame 11 introduces no term. Terms assumed from prereqs: none (front door);
matrix multiply, token and the W_O matrix are linked to `attention` as optional background.

Indexing: no positions or addresses on this page.

## 5. Animation script
All numbers from `math/roofline.js` (reproducer in §6). H100 values are `hardware.json/h100.*`:
989 TFLOPS BF16 dense, 1,979 FP8 dense (reported, 2 × BF16), 3.35 TB/s, 80 GB.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | One large `gpu` glyph, labeled "H100". Its die (SM grid) is labeled "compute"; its HBM stacks are labeled "HBM: high-bandwidth memory". | The die and the HBM stacks fade in one after the other. | A GPU is compute next to stacks of memory called HBM. On an H100, HBM holds 80 GB: the weights and everything else the GPU works on. | 80 GB HBM3 |
| 2 | The die's SM grid lights; a label "132 SMs, each with tensor cores" and a readout "989 TFLOPS". A `flow` arrow from HBM to the die with the readout "3.35 TB/s". Plain mark: "dense numbers; vendor sparse figures are 2× and not used". | The arrow draws; both readouts type in. | Tensor cores inside the compute grid do matrix multiplies at 989 trillion operations a second. HBM feeds them only 3.35 trillion bytes a second. | 132 SMs · 989 TFLOPS (BF16, dense) · 3.35 TB/s |
| 3 | Zoom on one SM: two labeled `block`s, "tensor cores" and "SRAM: hundreds of KB, on chip"; HBM (80 GB, off chip) stays at the edge. | The zoom; the SRAM block lights. | Each SM also has a tiny on-chip memory, SRAM, much faster than HBM. Fast kernels such as FlashAttention keep their working numbers there between steps. | SRAM: hundreds of KB per SM (rough) · HBM: 80 GB |
| 4 | Back to the full GPU. Left: token chips The, cat, sat (outlined), down beside `matrix` X [4 × 8]; `matrix` W_O [8 × 8]; empty Y [4 × 8]. Two `NUMBER_CELL` readouts: "bytes" and "FLOPs". | X and W_O travel from HBM into the die (`flow` dots), Y travels back; the bytes cell counts 0 → 256 as each matrix moves; then Y's cells fill row by row and the FLOPs cell counts 0 → 512 ("sat"'s row adds 128). | Take attention's last multiply: four tokens times W_O. It moves 256 bytes and does 512 floating-point operations, or FLOPs: 16 for each output number. | bytes (BF16, 2 per number): X 32 + W_O 64 + Y 32 = 128 numbers → 256 · FLOPs: 4 · 8 outputs × (8 multiplies + 8 adds) = 512 · row "sat": 128 |
| 5 | The two readouts and a third `NUMBER_CELL` "FLOPs per byte". | The third cell types 2. | Arithmetic intensity is FLOPs per byte moved. This multiply does 2 FLOPs for every byte it reads or writes. | 512 ÷ 256 = 2.00 |
| 6 | `roofline` appears at the right, H100 BF16: sloped roof up to the bend, flat roof after it; the bend is labeled "ridge point 295". The followed dot (outlined) sits at x = 2 on the sloped roof. | The sloped roof draws from the left, the flat roof from the right; they meet; the dot drops onto the roof at x = 2. | Plot speed against intensity and the roof bends once. The bend, the ridge point, is where moving a byte takes as long as 295 FLOPs on an H100. | ridge = 989 ÷ 3.35 = 295.2 FLOPs/byte · dot: intensity 2 → attainable 6.7 TFLOPS (0.68% of peak) |
| 7 | The matrices grow to real size: X [4 × 8,192], W [8,192 × 8,192]; the same four chips. Plain mark: "toy size: inputs and outputs are half the bytes; at real size the weights are 99.9%". Below the plot, a `laneTimeline` with two lanes on one axis: "memory 40.1 µs" (`--sem-memory`) and "compute 0.54 µs" (`--sem-compute`). The dot moves to x = 4. | The memory lane extends to 40.1 µs; the compute lane stops at 0.54 µs; its idle remainder hatches. | At real size the weights are nearly all the bytes, so four tokens get only 4 FLOPs per byte. This multiply is memory-bound: HBM sets the pace. | FLOPs 536,870,912 · bytes 134,348,800 · intensity 4.00 · memory 40.10 µs vs compute 0.54 µs (74×) · attainable 13.4 TFLOPS = 1.35% of peak, tensor cores idle 98.6% of the time |
| 8 | The token count ticks 4 → 64 → 256 → 4,096 (a plain counter; the chip row is replaced by "4,096 tokens"). The dot slides right along the roof, crosses the bend, and rides the flat roof. Lanes update. | Counter and dot move together; at the crossing the lanes swap which is longer. | Each weight read from HBM is used once per token. At 4,096 tokens the multiply does 2,048 FLOPs per byte and is compute-bound: the tensor cores set its pace. | intensity 4 → 63 → 241 → 2,048 · crossing at 318 tokens · 4,096 tokens: compute 555.9 µs vs memory 80.1 µs |
| 9 | Precision switch on the plot: label "FP8: 1 byte per number, tensor cores 2× faster". The roof's flat part doubles to 1,979 TFLOPS; the bend moves to 591. Two dots (outlined: 4 tokens; plain: 4,096) move right by 2×. A `bitLayout` row: BF16 1/8/7 over FP8 1/4/3. | The flat roof rises, the bend slides right, both dots slide right; the 4-token lanes halve. | FP8 stores each number in 1 byte and, on an H100, runs the tensor cores twice as fast. Both sides double, so most multiplies just get twice as fast. | ridge 590.7 · 4 tokens: intensity 7.99, memory 20.1 µs · crossing still at 318 tokens · H100 FP8 1,979 TFLOPS (reported) |
| 10 | `bitLayout`: an NVFP4 block collapsed to fit the stage (lesson 18): three 4-bit numbers (1/2/1) drawn, a plain "… 13 others", and one 8-bit E4M3 `scale` field with a bracket "shared by 16 numbers" (3 × 4 + 8 = 20 bit cells × 14 px = 280 px); below it MXFP4 the same way: three numbers, "… 29 others", one 8-bit scale "shared by 32". | The scale field slides under the block; the counter "bits per number" goes 4 → 4.5 (NVFP4) and 4 → 4.25 (MXFP4). | A 4-bit number has only 16 possible values, so each block of numbers shares one scale. NVFP4 shares one per 16 numbers: 4.5 bits each, not 4. | NVFP4 4 + 8 ÷ 16 = 4.5 bits · MXFP4 4 + 8 ÷ 32 = 4.25 bits (gpt-oss: "about 4.25 bits per parameter") |
| 11 | A plain ladder of ridge points beside the plot, one row per chip and format: H100 BF16 295 · B200 BF16 281 · B300 BF16 313 · B300 NVFP4 1,875 · Rubin NVFP4 1,591–1,823 (with the "sources conflict" mark). A second column: tokens to cross, 318 · 302 · 338 · 605 · 502–586. | Rows type in top to bottom; the B300 NVFP4 row's dot appears on a B300 roofline. | From H100 to B300 the BF16 ridge barely moved. FP4 tensor cores grew six-fold while bytes shrank about four-fold, so a B300 needs about 600 tokens per weight read. | ridges and crossings as listed · B300 FP4 15,000 vs BF16 2,500 TFLOPS (6×); NVFP4 bytes 0.5625 vs 2 per number (3.6×) |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 4–5 draw `matmulCost({ m: 4, k: 8, n: 8, bytesPerElem: 2 })`; frames 7–8 use `k = n =
8192`; frame 11 is the `ridgePoint` / `tokensToComputeBound` table in §6.

Caption check (2026-10-07, the `rlvr-grpo` counter adapted to this file): frames 1–11 are 27/2, 24/2,
25/2, 24/2, 19/2, 28/2, 27/2, 29/2, 29/2, 27/2, 29/2 (words/sentences); no formulas or operators.

## 6. Toy
"Where does this multiply sit on the roof?" One matmul, X [tokens × 8,192] · W [8,192 × 8,192], on a
chip and in a format you pick. A visible line above the controls: "The 8,192 × 8,192 weight matrix is a
stand-in for one real layer's weight; the chips' numbers are vendor dense peaks, not measured speeds."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `chip` | Chip | Preset chips | H100 · H200 · B200 · B300 · MI355X · TPU v7 · Rubin | H100 | from `data/hardware.json` (§8) |
| `fmt` | Number format | Segmented | BF16 · FP8 · FP4 (NVFP4 on NVIDIA, MXFP4 on AMD) | BF16 | FP4 disabled with the visible note "no FP4 figure in data" on H100, H200, TPU v7; Rubin is FP4 only ("no settled BF16/FP8 figure") |
| `tokens` | Tokens sharing each weight read | Slider (powers of 2) | 1 … 8,192 | 4 | chips `4`, `256`, `4,096` |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| FLOPs and bytes of the multiply | `matmulCost({ m: tokens, k: 8192, n: 8192, bytesPerElem: bytesPerElement(fmt) })` | count (`formatCount`), bytes (`formatBytes`, decimal) |
| Arithmetic intensity | `arithmeticIntensity(cost)` | FLOPs/byte, 1 decimal |
| Ridge point | `ridgePoint({ peakTflops, bandwidthTBps })`; Rubin: called once per end of `hbm_tbps` | FLOPs/byte; Rubin as "1,591–1,823" |
| **Tokens needed to be compute-bound** | `tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k: 8192, n: 8192 })` | tokens, integer (rounded up) |
| Verdict | `rooflineTime(...).bound` | "memory-bound" (`--sem-memory` text) / "compute-bound" (`--sem-compute` text) |
| Attainable speed and % of peak | `attainableTflops({ intensity, peakTflops, bandwidthTBps })`, `/ peakTflops` | TFLOPS, % (2 decimals) |
| Time lanes: memory vs compute | `rooflineTime({ flops, bytes, peakTflops, bandwidthTBps })` → `memoryS`, `computeS` | µs, 2 decimals, in a `laneTimeline` |
| `roofline` plot | the chip's roof; dots: this multiply (outlined) and a residual add of 4 × 8,192 numbers | dot labels print intensity |

The residual-add dot uses `elementwiseCost({ elements: 32768, inputs: 2, flopsPerElem: 1,
bytesPerElem })`: 0.17 FLOPs/byte in BF16, always memory-bound (03 §1.2: elementwise ops have intensity
about 1 or less, hence kernel fusion). A visible line under the plot says so.

**Try this** (each leads to a named insight)
1. H100, BF16, 4 tokens: 1.35% of peak, memory-bound, 40.10 µs of memory time for 0.54 µs of math.
   Predict how many tokens it takes to become compute-bound, then slide: 64 (21.3%), 256 (81.6%, still
   memory-bound), 512 (compute-bound). The readout says 318. → **Insight: intensity is roughly the
   number of tokens sharing each weight read, so batch size is the lever.** Training pushes thousands of
   tokens through every weight and is compute-bound; decoding for a few users is not (`prefill-decode`).
2. H100 at 256 tokens: switch BF16 → FP8. Intensity 241 → 482, ridge 295 → 591, still memory-bound;
   memory time 42.57 → 21.28 µs; "tokens needed" stays 318. → **Insight: on an H100, FP8 doubles both
   sides of the roof, so a multiply runs about twice as fast on either side of the ridge, but the batch
   needed to cross does not change.**
3. B300 at 512 tokens: BF16 is compute-bound (intensity 455 vs ridge 313). Switch to FP4: intensity
   1,618 vs ridge 1,875, now memory-bound; tokens needed 338 → 605. Then step H100 → B200 → B300 in
   BF16: the ridge reads 295, 281, 313. → **Insight: the BF16 ridge has been flat since 2022; FP4 is
   where compute outran memory, so FP4 needs bigger batches to pay off.** (03 §1.2's "lower precision
   pushes the ridge right" is true in FLOPs per byte; in tokens it holds only for FP4.)

**`math/roofline.js`** (new module, owned by this page; `prefill-decode` and `quantization` import it;
pure, no DOM, inputs never mutated; tests first). Units: TFLOPS and TB/s in, so `peakTflops /
bandwidthTBps` is FLOPs per byte directly.

```js
FORMATS  // frozen: { fp32: { bits: 32, layout: '1/8/23' }, bf16: { bits: 16, layout: '1/8/7' },
         //   fp16: { bits: 16, layout: '1/5/10' }, fp8: { bits: 8, layout: '1/4/3' },     // E4M3
         //   mxfp4: { bits: 4, layout: '1/2/1', blockSize: 32, scaleBits: 8 },
         //   nvfp4: { bits: 4, layout: '1/2/1', blockSize: 16, scaleBits: 8 } }       // E4M3 scale; per-tensor FP32 scale ignored
bitsPerElement(format: string) → number      // bits + scaleBits / blockSize; throws RangeError on an unknown name
//   'bf16' → 16 · 'fp8' → 8 · 'nvfp4' → 4.5 · 'mxfp4' → 4.25
bytesPerElement(format: string) → number     // bitsPerElement / 8
//   'bf16' → 2 · 'nvfp4' → 0.5625 · 'mxfp4' → 0.53125
matmulCost({ m, k, n, bytesPerElem }) → { flops, bytes }   // flops 2mkn; bytes = bytesPerElem·(mk + kn + mn): read X and W once, write Y once
//   { m: 4, k: 8, n: 8, bytesPerElem: 2 }       → { flops: 512, bytes: 256 }
//   { m: 4, k: 8192, n: 8192, bytesPerElem: 2 } → { flops: 536870912, bytes: 134348800 }
//   { m: 4096, k: 8192, n: 8192, bytesPerElem: 2 } → { flops: 549755813888, bytes: 268435456 }
elementwiseCost({ elements, inputs = 2, flopsPerElem = 1, bytesPerElem }) → { flops, bytes }   // reads `inputs` tensors, writes one
//   { elements: 32768, bytesPerElem: 2 } → { flops: 32768, bytes: 196608 }   (intensity 0.1667)
arithmeticIntensity({ flops, bytes }) → number
//   { flops: 512, bytes: 256 } → 2 · the 4-token real-size multiply → 3.9961 · 4,096 tokens → 2048
ridgePoint({ peakTflops, bandwidthTBps }) → number   // FLOPs per byte
//   (989, 3.35) → 295.22 · (1979, 3.35) → 590.75 · (2250, 8) → 281.25 · (2500, 8) → 312.5 · (15000, 8) → 1875
//   (989, 4.8) → 206.04 · (2307, 7.38) → 312.6 · (35000, 22) → 1590.9 · (35000, 19.2) → 1822.9
attainableTflops({ intensity, peakTflops, bandwidthTBps }) → number   // min(peak, intensity · bandwidth)
//   (2, 989, 3.35) → 6.7 · (3.9961, 989, 3.35) → 13.39 · (2048, 989, 3.35) → 989
rooflineTime({ flops, bytes, peakTflops, bandwidthTBps }) → { computeS, memoryS, timeS, bound }   // timeS = max; bound 'compute' | 'memory'
//   4-token real-size multiply on H100 BF16 → { computeS: 5.428e−7, memoryS: 4.0104e−5, bound: 'memory' }
//   4,096 tokens → { computeS: 5.5587e−4, memoryS: 8.013e−5, bound: 'compute' }
tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k, n }) → number   // smallest real m with intensity ≥ ridge; Infinity if unreachable
//   m = R·b·k·n / (2kn − R·b·(k + n)),  R = ridge, b = bytesPerElem
//   H100 BF16 (989, 3.35, 2, 8192, 8192) → 318.2 · H100 FP8 (1979, 3.35, 1) → 318.3 · B300 BF16 → 338.3
//   B300 NVFP4 (15000, 8, 0.5625) → 605.3 · Rubin NVFP4 (35000, 22 | 19.2, 0.5625) → 502.3 | 586.1
```

Chip presets live in the concept module and are read from `data/hardware.json` at load (peak per
format, `hbm_tbps`, FP4 flavor `nvfp4` for NVIDIA and `mxfp4` for MI355X). FP8 is counted at 8 bits
(per-tensor scale; the per-tile scales of `scale-reliability` are not counted, a visible note says so).

Reproducer (run 2026-10-07 against the scratch implementation of these signatures; re-run once
`math/roofline.js` exists):
```sh
node -e '
import("./math/roofline.js").then((m) => {
  const f = (x, d = 2) => Number(x.toFixed(d)), H = { peakTflops: 989, bandwidthTBps: 3.35 };
  const toy = m.matmulCost({ m: 4, k: 8, n: 8, bytesPerElem: 2 }); console.log("toy", toy, m.arithmeticIntensity(toy));
  for (const B of [1, 4, 64, 256, 318, 512, 1024, 4096, 8192]) { const c = m.matmulCost({ m: B, k: 8192, n: 8192, bytesPerElem: 2 });
    const I = m.arithmeticIntensity(c), t = m.rooflineTime({ ...c, ...H }), a = m.attainableTflops({ intensity: I, ...H });
    console.log(B, c.flops, c.bytes, f(I, 3), f(a, 1), f(100 * a / 989), f(t.computeS * 1e6, 3), f(t.memoryS * 1e6, 3), t.bound); }
  const chips = { h100: { bf16: 989, fp8: 1979, bw: 3.35 }, h200: { bf16: 989, fp8: 1979, bw: 4.8 }, b200: { bf16: 2250, fp8: 4500, fp4: 9000, bw: 8 },
    b300: { bf16: 2500, fp8: 5000, fp4: 15000, bw: 8 }, mi355x: { bf16: 2500, fp8: 5000, fp4: 10000, bw: 8, fp4f: "mxfp4" },
    tpuv7: { bf16: 2307, fp8: 4614, bw: 7.38 }, rubinLo: { fp4: 35000, bw: 19.2 }, rubinHi: { fp4: 35000, bw: 22 } };
  for (const [id, c] of Object.entries(chips)) for (const p of ["bf16", "fp8", "fp4"]) if (c[p]) {
    const b = m.bytesPerElement(p === "fp4" ? (c.fp4f || "nvfp4") : p), x = { peakTflops: c[p], bandwidthTBps: c.bw };
    const I = (B) => m.arithmeticIntensity(m.matmulCost({ m: B, k: 8192, n: 8192, bytesPerElem: b }));
    console.log(id, p, f(m.ridgePoint(x), 1), f(m.tokensToComputeBound({ ...x, bytesPerElem: b, k: 8192, n: 8192 }), 1), f(I(4)), f(I(256), 1), f(I(512), 1)); }
});'
```
Output (2026-10-07): toy `{ flops: 512, bytes: 256 }`, intensity 2. Real size, H100 BF16 (tokens:
intensity / attainable TFLOPS / % peak / compute µs / memory µs): 1: 1.000 / 3.3 / 0.34 / 0.136 / 40.075 ·
4: 3.996 / 13.4 / 1.35 / 0.543 / 40.104 · 64: 63.015 / 211.1 / 21.34 / 8.685 / 40.691 · 256: 240.941 /
807.2 / 81.61 / 34.742 / 42.569 · 512: 455.111 / 989 / 100 / 69.484 / 45.073 (compute) · 4,096: 2,048 /
989 / 100 / 555.87 / 80.13 · 8,192: 2,730.667. Chips (ridge / tokens to cross / intensity at 4, 256,
512 tokens): H100 BF16 295.2 / 318.2 / 4.00, 240.9, 455.1 · H100 FP8 590.7 / 318.3 / 7.99, 481.9, 910.2 ·
H200 BF16 206.0 / 217.0 · B200 BF16 281.3 / 302.0 · B200 NVFP4 1,125 / 342.9 · B300 BF16 312.5 / 338.3 ·
B300 FP8 625 / 338.3 · B300 NVFP4 1,875 / 605.3 / 14.21, —, 1,618.2 · MI355X BF16 312.5 / 338.3 ·
MI355X MXFP4 1,250 / 361.3 · TPU v7 BF16 312.6 / 338.4 · TPU v7 FP8 625.2 / 338.4 · Rubin NVFP4 1,822.9
(19.2 TB/s) / 586.1 and 1,590.9 (22 TB/s) / 502.3. FP8 at 256 tokens: memory 21.28 µs (BF16 42.57).

Tests to write first: the worked examples above; `ridgePoint` and `tokensToComputeBound` agree (the
intensity at the returned m equals the ridge to 1e−9); `rooflineTime.bound` flips exactly at that m;
`attainableTflops ≤ peakTflops` always; `bitsPerElement` throws on unknown names; `FORMATS` is frozen.

## 7. Show me the math
```tex
\htmlClass{hl-int}{I} = \frac{\text{FLOPs}}{\text{bytes}}
\qquad
\text{matmul } X_{[m \times k]} W_{[k \times n]}:\quad
I = \frac{2mkn}{b\,(mk + kn + mn)} \;\approx\; \frac{2m}{b} \ \ (m \ll k, n)
```
```tex
\htmlClass{hl-ridge}{I^{*}} = \frac{\htmlClass{hl-peak}{P_{\text{peak}}}}{\htmlClass{hl-bw}{\text{BW}}}
\qquad
\text{attainable} = \min\!\big(\htmlClass{hl-peak}{P_{\text{peak}}},\; \htmlClass{hl-int}{I}\cdot \htmlClass{hl-bw}{\text{BW}}\big)
\qquad
t = \max\!\Big(\tfrac{\text{FLOPs}}{P_{\text{peak}}},\; \tfrac{\text{bytes}}{\text{BW}}\Big)
```
```tex
m^{*} = \frac{I^{*} b\, k n}{2kn - I^{*} b\,(k + n)}
\qquad
\text{bits per number (block-scaled)} = \text{bits} + \frac{\text{scale bits}}{\text{block size}}
```
Shapes: X [m × k] (tokens × d), W [k × n], Y [m × n]; `b` = bytes per number. Worked: H100 BF16,
k = n = 8,192 → I* = 295.2, m* = 318. Color links: `hl-int` = the followed dot and the intensity cell;
`hl-peak` = the flat roof (`--sem-compute`); `hl-bw` = the sloped roof and the HBM `flow` arrow
(`--sem-memory`); `hl-ridge` = the bend label. Note shown in the panel: real kernels add launch and
latency overheads, so at a few hundred bytes (frame 4) the roofline is a bound, not a prediction.

## 8. In today's models (Oct 2026)
Rendered as the "2026 hardware table" (dense numbers only, decimal GB):

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| H100 SXM (2022): 80 GB HBM3, 3.35 TB/s, 989 TF BF16, 1,979 TF FP8 (reported, 2 × BF16), NVLink 900 GB/s; ridge 295 | `hardware.json/h100.hbm_gb`, `.hbm_tbps`, `.bf16_dense_tflops`, `.fp8_dense_tflops`, `.scale_up_domain` | 03 §1.4, §1.2 |
| H100 has 132 SMs | `hardware.json/h100.sm_count` = 132 *(proposed)* | 03 §1.1 (CONFIRMED via the DeepSeek-V3 paper's H800 figure) |
| H200: 141 GB, 4.8 TB/s, same compute as H100; ridge 206: the extra bandwidth makes it *easier* to be compute-bound | `hardware.json/h200.*` | 03 §1.4 |
| B200 (HGX): 192 GB nominal (about 180 usable), 8 TB/s, ~2.25 PF BF16, ~4.5 FP8, ~9 FP4 (reported) | `hardware.json/b200.*` | 03 §1.4 |
| B300: 288 GB, 8 TB/s, ~2.5 PF BF16, ~5 FP8, 15 PF FP4 (reported) | `hardware.json/b300.*` | 03 §1.4 |
| MI355X: 288 GB, 8 TB/s, 2.5 / 5 / 10 PF (reported) | `hardware.json/mi355x.*` | 03 §1.4 |
| TPU v7 Ironwood: 192 GiB (206 GB), 7.38 TB/s, 2,307 TF BF16, 4,614 FP8 | `hardware.json/tpu-v7.*` (unit GiB; the page converts and says so) | 03 §1.4 |
| Rubin (shipping since mid-2026): 288 GB HBM4, 35 PF NVFP4 for training (not confirmed as strictly dense), HBM bandwidth **19.2 or 22 TB/s: NVIDIA's own pages disagree**; shown as a range everywhere | `hardware.json/rubin.hbm_gb`, `.fp4_dense_tflops`, `.hbm_tbps` = [19.2, 22] | 03 §1.4 (flag), spec §7 |
| Number formats: FP32 1/8/23, BF16 1/8/7, FP16 1/5/10, FP8 E4M3 1/4/3 or E5M2 1/5/2, MXFP4 (blocks of 32, 8-bit power-of-two scale), NVFP4 (blocks of 16, E4M3 scale, plus a per-tensor FP32 scale) | `hardware.json/formats.*` *(proposed entry `formats`, keys `bf16.layout`, `fp8.layout`, `mxfp4.block_size` = 32, `nvfp4.block_size` = 16, `nvfp4.scale_format` = "E4M3")* | 03 §1.3 |
| gpt-oss-120b stores MoE weights in MXFP4, about 4.25 bits per parameter, so the 120B fits one 80 GB GPU (post-training quantization, not native FP4 training) | `models.json/gpt-oss-120b.weight_format` *(proposed)* | 02 §1.7 item 3 |
| NVIDIA trained a 12B model on 10T tokens in NVFP4 and matched its FP8 loss (2025) | `models.json/nvidia-nvfp4-12b.pretrain_format` *(proposed; confirmed in 03, reported in 02)* | 03 §1.3 |
| FlashAttention tiles attention so the score matrix stays in SRAM instead of HBM (2022) | none (mechanism; paper link arXiv 2205.14135) | 03 §1.1 |

Not shown: Rubin BF16/FP8 (brief flags a likely sparse/dense mismatch), MI455X, Trainium, TPU 8 (not in
`data/`, or UNVERIFIED final specs), FlashAttention-4's 71% figure (REPORTED via abstract only).

## 9. Takeaways
1. Every operation pays for bytes moved and FLOPs done; its arithmetic intensity (FLOPs per byte)
   against the chip's ridge point (peak ÷ bandwidth, 295 on an H100) says which one sets its speed.
2. A matrix multiply's intensity is roughly the number of tokens sharing each weight read: a few tokens
   are memory-bound (1.35% of peak at 4), a few hundred cross the ridge, and training batches sit far
   on the compute side.
3. Lower precision shrinks bytes and speeds up tensor cores: FP8 makes an H100 multiply about 2× faster
   without changing the batch needed; FP4 needs a shared block scale (4.5 bits in NVFP4) and, on a B300,
   about twice the tokens to stay compute-bound. The BF16 ridge has been flat since 2022.

## 10. Next and go deeper
Next: `training-memory` (what fills those 80 GB during training) · `prefill-decode` (the same roofline
with prefill and decode as the two dots) · `quantization` (what fewer bits cost in quality).
Go deeper (brief 05 §1.2, 03 §7.2): Google, *How to Scale Your Model*, the roofline chapter
(https://jax-ml.github.io/scaling-book/roofline/) · Horace He, *Making Deep Learning Go Brrrr*
(https://horace.io/brrr_intro.html) · Modal, *GPU Glossary* (https://modal.com/gpu-glossary).

## 11. Key-frame sketch
Frame 7 end state (H100, BF16, real size), desktop width. Numbers from the §6 reproducer.
```text
GPU primer                       7 / 11   [<] [Play] [>]
 [The][cat][(sat)][down]  X [4 x 8,192] . W [8,192 x 8,192]
                    TFLOPS
 +------+----+       1000 |            ___________ 989
 |######| HB |            |          / ridge 295
 |######| M  |        100 |        /
 |######|    |         10 |   (o) intensity 4.0
 +------+----+          1 |__/_____________________
  compute  HBM            1     10    100   1000  FLOPs/B
 memory  [==================================] 40.10 us
 compute [=]//////////////////////////////// 0.54 us
 FLOPs 536,870,912 | bytes 134,348,800 | 1.35% of peak
 At real size the weights are nearly all the bytes ...
```

## 12. Open questions for the reviewer
**Data-pass keys**
- `hardware.json/h100.sm_count = 132` (count, 03 §1.1; the brief's source is the DeepSeek-V3 paper's
  "132 SMs available in the H800"; the data pass should confirm against an NVIDIA H100 page).
- `hardware.json/formats` entry (new): `fp32.layout "1/8/23"`, `bf16.layout "1/8/7"`, `fp16.layout
  "1/5/10"`, `fp8_e4m3.layout "1/4/3"`, `fp8_e5m2.layout "1/5/2"`, `mxfp4.block_size 32`,
  `mxfp4.scale_bits 8`, `nvfp4.block_size 16`, `nvfp4.scale_format "E4M3"` (03 §1.3; NVIDIA NVFP4 blog
  CONFIRMED; OCP MX for MXFP4).
- `models.json/gpt-oss-120b.weight_format = "MXFP4 (MoE weights, post-training), ~4.25 bits/param"`
  (02 §1.7, CONFIRMED model card).
- `models.json/nvidia-nvfp4-12b` (new entry): `params 12e9`, `pretrain_tokens 10e12`,
  `pretrain_format "NVFP4"`, note "matched FP8 loss" (03 §1.3 CONFIRMED NVIDIA blog; 02 marks the
  related paper REPORTED; cite the blog).
- `hardware.json/memory-hierarchy.sram_per_sm = "hundreds of KB"` (reported, rough; 03 §1.1). If the
  data pass prefers not to store a rough range, frame 3 drops the number and says "tiny".
**Graph changes:** none.
**Judgment calls**
1. **Lesson 17 correction to the brief.** 03 §1.2 says lower precision and newer chips push the ridge
   right "so memory-bound gets easier to hit". The toy's own numbers show this holds in FLOPs per byte
   but, in tokens per weight read, only for FP4 (H100 BF16 and FP8 both cross at 318 tokens; B300 BF16
   338 vs NVFP4 605), and the BF16 ridge is flat across H100/B200/B300. The page teaches the tokens
   framing. Confirm.
2. **Rubin as a range.** Rubin appears only in FP4, with ridge 1,591–1,823 and crossing 502–586 tokens,
   plus a visible "sources conflict" mark. Alternatively leave Rubin out of the toy entirely until spec
   §7's conflict is resolved.
3. **Block-scale overhead.** NVFP4 counted at 4.5 bits, MXFP4 at 4.25, FP8 at 8 (per-tensor scale only);
   NVFP4's per-tensor FP32 scale is ignored. `quantization` should import `FORMATS` / `bitsPerElement`
   rather than define its own.
4. **Frame 1–2 terms.** "SM" appears as a label (the count "132 SMs") in frame 2, alongside the frame's
   term, tensor core. If the reviewer reads "SM" as a second new term, frame 2's label becomes "132
   compute units" and SM moves to frame 3's caption.
5. **Weights as a `flow` carry.** W_O travels HBM → SM in frame 4; `flow` carries are activation,
   gradient, kv and token. The page draws W_O's arrow as a plain labeled arrow (no dot). `parallelism`
   and `training-memory` also move weights (ZeRO-3 all-gathers); a shared `carry: 'weight'` may be
   worth adding (proposed in `parallelism` §12).
6. **Stage budget.** Frames 4–5 use 20 px shape-only matrices (no numbers inside; counts print in
   `NUMBER_CELL` readouts). Frame 11's ladder is a five-row plain text table beside the plot.
