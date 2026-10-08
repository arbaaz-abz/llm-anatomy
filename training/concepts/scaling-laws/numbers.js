// scaling-laws hand-authored stand-in constants (storyboard §5, §8). The model rows restate data/models.json; the page test
// keeps them equal to the data (P3-R13 spirit). Frozen with deepFreeze.
import { deepFreeze } from '@math/core.js';
import { computeOptimal } from '@math/scaling.js';

export const STAGE_BUDGET = 1e24; // frames 2-7: FLOPs
export const BUDGETS = Object.freeze([1e22, 1e23, 1e24, 1e25, 1e26]); // frame 5 and the toy's chips
export const CHINCHILLA_RULE = 20; // "about 20 tokens per parameter", the 2022 rule of thumb
export const SERVE_WHAT_IF = 1e14; // frames 6-7: a what-if serving volume (100T tokens), not any model's traffic
const OPTIMUM_N = computeOptimal(STAGE_BUDGET).N; // the middle size is the optimum itself, so frames 2-7 print one size (95.9B)
export const FRAME_2_SIZES = Object.freeze([1e9, OPTIMUM_N, 1e12]); // 1B, the optimum, 1T: the three blocks (and frame 3's points)
export const CURVE_TABLE_SIZES = Object.freeze([1e9, 1e10, 3e10, OPTIMUM_N, 3e11, 1e12]); // frame 3's loss table

// Frame 1 and frame 8: (id in data/models.json, label, active parameters, total parameters, pretraining tokens).
export const MODELS = deepFreeze({
  deepseekV4Pro: { id: 'deepseek-v4-pro', label: 'DeepSeek-V4-Pro', active: 49e9, total: 1.6e12, tokens: 33e12 },
  llama31: { id: 'llama-3.1-405b', label: 'Llama 3.1 405B (2024)', active: 405e9, total: 405e9, tokens: 15.6e12 },
  nemotron: { id: 'nemotron-3-super', label: 'Nemotron 3 Super', active: 12e9, total: 120e9, tokens: 25e12 },
  deepseekV4Flash: { id: 'deepseek-v4-flash', label: 'DeepSeek-V4-Flash', active: 13e9, total: 284e9, tokens: 32e12 },
});
// Frame 8's row order (Chinchilla's rule sits first, with no per-total column).
export const TABLE_ORDER = Object.freeze(['llama31', 'deepseekV4Pro', 'nemotron', 'deepseekV4Flash']);

// Frames 9-10: one weight matrix's two stand-in singular values, and DeepSeek-V4's Newton-Schulz schedule
// (8 steps, then 2 steps). The data string models.deepseek-v4-pro.muon_ns_schedule says the same.
export const SINGULAR_VALUES = Object.freeze([3, 0.3]);
const MUON_BULK = Object.freeze([3.4445, -4.775, 2.0315]);
const MUON_FINISH = Object.freeze([2, -1.5, 0.5]);
export const MUON_SCHEDULE = deepFreeze([...Array(8).fill(MUON_BULK), ...Array(2).fill(MUON_FINISH)]);
export const MUON_COEFFICIENTS = deepFreeze({ bulk: MUON_BULK, finish: MUON_FINISH, finishSteps: 2 });
