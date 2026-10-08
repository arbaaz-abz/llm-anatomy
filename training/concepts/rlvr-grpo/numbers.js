// rlvr-grpo's hand-authored stand-ins (storyboard §6) and the constants the stage and toy share. Pure, frozen.
// The answer pools, the slot order and the target are GROUP_TOY in math/grpo.js (ruling P3-R5); the ratios stay here.
import { deepFreeze } from '@math/core.js';

export const GROUP_SIZE = 8;
export const DEFAULT_K = 2; // right answers out of 8 in the animation and the toy's opening state
export const PI_OLD = 0.2; // every token was sampled at this probability (a stand-in)
export const EPS_LOW = 0.2;
export const EPS_HIGH_PPO = 0.2; // the "PPO" chip; the clip-higher chip reads data/models.json glm-5.rl_clip_eps_high
export const ADV_MAX_ABS = 2.65; // value scale for advantages: the largest reachable at G = 8 (k = 1: 2.6458)
export const ZOOM_RATIO = 1.25; // frames 8–9: row 1's `56`
export const ZOOM_EPS_HIGH = 0.28;

// r = π_θ / π_old per token, keyed by the answer's text (the pools are consumed by position, so the key is stable).
export const RATIOS = deepFreeze({
  '7 × 8 = 56': [1.02, 0.98, 1.05, 1.0, 1.25],
  '7 + 8 = 56': [1.0, 1.0, 1.0, 1.0, 1.35],
  '7 × 8 = 48 , so 48': [1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 1.0, 1.1],
});

export const ratioAt = (tokens, index) => RATIOS[tokens.join(' ')]?.[index] ?? 1;

// The followed item (README: mark it the same way in every frame): row 1's final `56`, the inspector's opening selection.
export const FOLLOWED = Object.freeze({ row: 0, token: 4 });
