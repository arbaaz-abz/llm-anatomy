// kv-compression's hand-picked stand-ins (storyboard §4): the toy layer's sizes and the latent values. Frozen; none is a model fact.
import { deepFreeze } from '@math/core.js';

// One layer drawn: 8 query heads of 4 numbers (attention's d_head), a latent of 8 numbers and a 2-number position key.
export const TOY_SHAPE = deepFreeze({ layers: 1, queryHeads: 8, headDim: 4, dLatent: 8, dRope: 2 });
export const BYTES_PER_NUMBER = 2; // BF16, as every byte count on the page
export const FOUR_TOKENS = deepFreeze(['The', 'cat', 'sat', 'down']);

// One latent (8 numbers) and one position key (2 numbers) per token, on the quarter grid; "sat" carries the storyboard's c and [0.5, −0.5].
export const LATENTS = deepFreeze([
  [0.25, 0.5, -0.5, 0.75, 0, -0.25, 1, 0.25],
  [-0.5, 0.25, 0.75, -0.25, 0.5, 0, -0.75, 0.5],
  [0.5, -0.25, 1, 0, -0.5, 0.25, 0, 0.75],
  [0, 0.75, -0.25, 0.5, 0.25, -0.5, 0.5, -1],
]);
export const POSITION_KEYS = deepFreeze([[0.25, 0.75], [-0.5, 0.25], [0.5, -0.5], [0.75, 0]]);
export const VALUE_SCALE = 1; // the latent cells' color scale (their hover-only values lie in −1 … 1)

// Slider stops: a choice of the toy, not model facts.
export const TOY_LATENTS = deepFreeze([4, 8, 16]);
export const REAL_LATENTS = deepFreeze([256, 512, 1024]);
export const CONTEXTS = deepFreeze([2048, 32768, 131072, 1048576]);
export const DEFAULT_CONTEXT = 131072;
