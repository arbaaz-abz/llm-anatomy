// gpu-primer's hand-authored stage constants (storyboard §4–§5). The stage renders without ctx.data, so the chip numbers
// it draws are restated here; tests/gpu-primer-page.test.js asserts each equals data/hardware.json through chipPreset
// (the P3-R13 pattern: a page constant that restates data carries an equality test).
import { deepFreeze } from '@math/core.js';

export const TOKENS = Object.freeze(['The', 'cat', 'sat', 'down']); // the running example, rows of X
export const SAT = 2; // the followed row
export const D_TOY = 8; // toy d_model: X [4 × 8] · W_O [8 × 8] → Y [4 × 8]
export const D_REAL = 8192; // frames 7–11 and the toy: W [8,192 × 8,192]
export const FRAME8_TOKENS = Object.freeze([4, 64, 256, 4096]); // the counter's stops
export const FRAME11_TOKENS = 512; // try-this 3's B300 batch

// Dense peaks (TFLOPS) and HBM bandwidth (TB/s) the stage draws; formats as the toy names them.
export const STAGE_CHIPS = deepFreeze({
  h100: { label: 'H100', peak: { bf16: 989, fp8: 1979 }, bandwidths: [3.35], hbmGb: 80 },
  h200: { label: 'H200', peak: { bf16: 989 }, bandwidths: [4.8] },
  b200: { label: 'B200', peak: { bf16: 2250, fp4: 9000 }, bandwidths: [8] },
  b300: { label: 'B300', peak: { bf16: 2500, fp4: 15000 }, bandwidths: [8] },
  rubin: { label: 'Rubin', peak: { fp4: 35000 }, bandwidths: [22, 19.2] },
});
export const H100_SMS = 132; // hardware.json h100.sm_count
export const SRAM_PER_SM = 'hundreds of KB'; // hardware.json memory-hierarchy.sram_per_sm (rough, reported)
export const SPARSE_FACTOR = 2; // vendor sparse figures are 2× the dense ones (frame 2's plain mark)

// Frame 10's block-scaled formats as drawn (FORMATS keys), with how many of the block's numbers are drawn.
export const BLOCK_SHOWN = 3;
