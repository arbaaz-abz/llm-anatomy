// quantization's hand-authored stage constants (storyboard §4–§5). The stage renders without ctx.data, so the numbers it
// draws are restated here; tests/quantization-page.test.js asserts each equals data/*.json (the P3-R13 pattern).
// RUNNING_EXAMPLE is imported, never restated: it already carries the H200's FP8 peak and bandwidth.
import { deepFreeze } from '@math/core.js';

export const WEIGHTS = deepFreeze([0.12, -0.31, 0.05, 0.47, -0.08, 0.22, -0.64, 2.1]); // the hand-picked row, one outlier
export const OUTLIER = 2.1; // the last weight, switched on
export const CALM = 0.3; // the last weight, switched off
export const FOLLOWED = 3; // the followed weight, 0.47 (0-based index)

export const CONTEXT = 2048; // tokens in one user's cache
export const PREFILL_TOKENS = 4096;
export const BLOCK_SIZES = Object.freeze([8, 4, 2]);

// The two chips the page uses, as chipPreset (hardware.js) builds them from data/hardware.json.
// Peaks are dense TFLOPS; b200 memory is the usable 180 GB (basis printed with the number).
export const STAGE_CHIPS = deepFreeze({
  h200: { label: 'H200', hbmGb: 141, basis: 'nominal', nominalGb: 141, bandwidthTBps: 4.8, peak: { bf16: 989, fp8: 1979, fp4: null } },
  b200: { label: 'B200', hbmGb: 180, basis: 'usable', nominalGb: 192, bandwidthTBps: 8, peak: { bf16: 2250, fp8: 4500, fp4: 9000 } },
});

// Frame 10: vLLM's FP8 KV-cache post (128K needle-in-a-haystack, Llama-3.3-70B-Instruct).
export const VLLM = deepFreeze({ baselinePct: 91, naivePct: 13, recoveryPct: [97, 98] });

// Frame 11: the model cards.
export const MODEL_CARDS = deepFreeze({
  gptOss: { totalParams: 116830000000, activeParams: 5130000000, fitsGpuGb: 80 },
  v4Pro: { checkpointGb: 865 },
});

export const SCALE_RULE = Object.freeze({ int4: 7, e2m1: 6, mxOffset: 2 }); // q_max for INT4 and E2M1, E2M1's exponent (the MX rule)
