// quantization storyboard §5 captions, §6 "Check my work" and the try-this list, verbatim. Imported by the page test and the e2e spec.
// Corrections the page makes to the storyboard (R3, the page is right): caption 8 prints 39.4 GB (formatBytes, 3 significant figures);
// the NVFP4 block-of-2 error prints 0.017 (0.01746 unrounded, the storyboard's 0.018 came from a rounded intermediate).

export const CAPTIONS = [
  'Eight weights in BF16, two bytes each. Decode reads every weight at every step, so fewer bytes per weight means faster steps.',
  'To store a weight in 4 bits, divide by a scale and round to a whole number from −7 to 7. Here the scale is the largest weight over seven.',
  'Multiply back by the scale and each weight returns with a small rounding error. Three small weights came back as exactly zero.',
  'The culprit is one large weight, 2.10. It sets the scale for all eight, so the small ones get a grid far too coarse for them.',
  'Give each block of four its own scale. The first block now rounds on a fine grid, and the average error halves.',
  'FP4 stores a tiny floating-point number instead: 0, 0.5, 1, 1.5, 2, 3, 4 or 6, with a sign. Its steps are finest near zero, where most weights sit.',
  'MXFP4\'s scale must be a power of two, so a block\'s largest weight can land past 6 and be clipped. NVFP4\'s 8-bit float scale fits closer: error 0.029, not 0.062.',
  'Counting the scales, NVFP4 costs 4.5 bits per weight. The 70B model shrinks from 140 GB to 39.4 GB, and the H200 fits 151 users instead of one.',
  'Smaller weights speed up memory-bound decode. Compute-bound prefill gets faster only if the math itself runs in low precision, as FP8 does here and FP4 does on Blackwell.',
  'The KV cache can be stored in FP8 too, which doubles the users that fit. Done naively, it broke long-context recall in vLLM\'s tests until a fix restored it.',
  'The safest 4-bit models are trained to expect it. gpt-oss and DeepSeek-V4 shipped their weights in 4-bit formats, and Kimi K2.x is reported to ship INT4.',
];

export const CHECK_WORK = [
  'scale = max|w| / 7 = 2.10 / 7 = 0.30',
  'code = round(w / scale) = round(0.47 / 0.30) = 2',
  'restored = code × scale = 2 × 0.30 = 0.60',
].join('\n');

// Storyboard §6 "Try this", as the page prints it ("prompt → Insight: … rest"; [[slug]] prints as the lesson's title).
export const TRY_THIS = [
  'With Format INT4 and Weights per scale 8: mean error 0.064, 3 weights zeroed. Set Last weight to 0.30: error 0.024. Set it back to 2.10 (outlier) and Weights per scale to 4: 0.032, 1 zeroed. → Insight: one outlier ruins a shared scale, and smaller blocks contain the damage. That is why 2026 formats scale every 16 or 32 weights (DeepSeek\'s FP8 weights use 128 × 128 tiles).',
  'Weights per scale 4: INT4 0.032 · MXFP4 0.062 (0.47 clipped to 0.375) · NVFP4 0.029. Weights per scale 8: INT4 0.064 · MXFP4 0.073 · NVFP4 0.049. Weights per scale 2: INT4 0.011 · MXFP4 0.054 (two clipped) · NVFP4 0.017. → Insight: the scale\'s precision matters as much as the grid. A power-of-two scale is cheap to store but can waste range or clip; an 8-bit float scale fits each block more closely, for 0.25 more bits per weight.',
  'With Model weights BF16 → FP8 → 4-bit weights, 16-bit math on the H200: decode 29.3 ms → 14.7 ms → 8.35 ms; prefill 580 ms → 290 ms → 580 ms; users 1 → 105 → 151. Switch GPU to B200 (180 GB usable) and Model weights to NVFP4: prefill 63.7 ms, decode 5.01 ms, 209 users. → Insight: bytes speed up decode; only low-precision math speeds up prefill. Weight-only 4-bit is a decode trick; native FP8 or FP4 helps both.',
  'On the H200 with Model weights FP8, set KV cache to FP8: users 105 → 211. → Insight: at long context the KV cache is the bigger target, and its quantization needs its own care (step 10).',
];

export const NO_FP4_NOTE = 'NVFP4, MXFP4: no FP4 tensor cores; use 4-bit weights, 16-bit math';
