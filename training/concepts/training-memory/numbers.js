// training-memory's hand-authored stage constants (storyboard §5): GPT-3's shape, the 64-GPU group, the chips of frame 11.
// Each restates a data/*.json value; the page test pins them to the data (P3-R13). Everything else on the stage is computed
// from math/training-memory.js.
import { deepFreeze } from '@math/core.js';

export const GPT3 = deepFreeze({ params: 175e9, layers: 96, hidden: 12288, heads: 96, seq: 2048, microBatch: 1 });
export const ZERO_DP = 64; // data-parallel GPUs in frames 8–10
export const H100_HBM = 80e9; // nominal
export const SHOWN_GPUS = 4; // the GPU row collapses to this many glyphs plus "+ N more" (README lesson 18)
export const ONE_T = 1e12;
export const V4_PRO = deepFreeze({ total: 1.6e12, active: 49e9 });
export const MUON_BYTES = 12; // DeepSeek-V4-Pro's own recipe (TRAINING_RECIPES.muon total)

// Frame 11's chips: capacity label per the conventions (nominal, except B200's usable 180 of 192 nominal).
export const FRAME11_CHIPS = deepFreeze([
  { name: 'H100', capacity: '80 GB nominal', hbm: 80e9 },
  { name: 'H200', capacity: '141 GB nominal', hbm: 141e9 },
  { name: 'B200', capacity: '180 GB usable', hbm: 180e9 },
  { name: 'B300', capacity: '288 GB nominal', hbm: 288e9 },
]);
