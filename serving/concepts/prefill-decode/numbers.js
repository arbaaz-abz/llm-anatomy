// prefill-decode's hand-authored stage constants (storyboard §4–§5). The stage renders without ctx.data, so the few data
// figures it draws beyond RUNNING_EXAMPLE are restated here; tests/prefill-decode-page.test.js asserts each equals
// data/*.json (the P3-R13 pattern: a page constant that restates data carries an equality test).
import { deepFreeze } from '@math/core.js';

export const PREFILL_TOKENS = 1000; // frames 3–5: the 1,000-token prompt
export const CROSS_TOKENS = 217; // frame 6: the whole-token crossover on an H200 (tokensToComputeBound rounds to it)
export const CONTEXT = 2048; // frames 7–9: each user's context
export const USERS_SHOWN = Object.freeze([1, 8, 64]); // frames 7–8; frame 9 adds the max that fits (model.js)
export const CHIPS_SHOWN = 8; // prefill chips drawn before "+ N others" (README lesson 18)
export const PROMPT_WORDS = Object.freeze(['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'a']); // request A's prompt chips
export const ANSWER_WORD = 'down'; // request A's next token (frames 1–2)
export const BATCH_OWNERS = Object.freeze(['A', 'B', 'C', 'D']); // frames 7–9: four lettered users, the rest "+ N others"

// The H200 the stage draws (hardware.json h200: hbm_gb, nominal, and bf16_dense_tflops for frame 6's BF16 crossover;
// its FP8 peak and bandwidth live in RUNNING_EXAMPLE).
export const H200 = deepFreeze({ label: 'H200', hbmBytes: 141e9, basis: 'nominal', peakBf16Tflops: 989 });
// The H100 that frame 6 compares against (hardware.json h100: bf16_dense_tflops, hbm_tbps): gpu-primer's 318.
export const H100 = deepFreeze({ label: 'H100', peakTflops: 989, bandwidthTBps: 3.35, bytesPerElem: 2 });

// Frame 10: two InferenceX points for DeepSeek-V4-Pro on GB300 NVL72 (serving.json inferencex-v4-pro-gb300), with
// the conditions their notes state (ISL 8192 / OSL 1024, FP4, measured 2026-05-22).
export const MEASURED = deepFreeze({
  model: 'DeepSeek-V4-Pro', hardware: 'GB300 NVL72',
  points: [
    { perUser: 13.1, perGpu: 11056, label: 'max throughput' }, // max_throughput_tok_s_user, max_throughput_tok_s_gpu
    { perUser: 27, perGpu: 6182, label: 'interactive' }, // interactivity_tok_s_user, throughput_tok_s_gpu
  ],
  conditions: 'ISL 8K / OSL 1K · FP4 · InferenceX, measured 2026-05-22',
});
