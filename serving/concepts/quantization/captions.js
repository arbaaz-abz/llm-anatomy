// quantization storyboard §5 captions, verbatim (one per frame); content.js puts them on the steps, frames.js labels each stage.
// Caption 8 prints the weights at formatBytes' three significant figures (39.4 GB, README lesson 35); the storyboard's "39 GB" is corrected (R3).
export const CAPTIONS = Object.freeze([
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
]);
