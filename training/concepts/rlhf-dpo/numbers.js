// rlhf-dpo stand-in constants (storyboard §4: hand-picked scores and probabilities) and the numbers derived from them.
// Everything derived goes through math/preference.js, math/lm.js or math/grpo.js, never typed.
import { deepFreeze } from '@math/core.js';
import { klDivergence } from '@math/lm.js';
import { clippedSurrogate } from '@math/grpo.js';
import { bradleyTerry, dpoLoss, klPenalizedReward } from '@math/preference.js';

export const PROMPT = 'Why is the sky blue? (asked by a 7-year-old)';
export const ANSWER_A = Object.freeze(['Air', 'scatters', 'blue', 'light', 'the', 'most']);
export const ANSWER_B = Object.freeze(['Rayleigh', 'scattering', ',', 'λ⁻⁴']);
export const ANSWER_FRESH = Object.freeze(['Air', 'scatters', 'blue', 'light', 'more', 'than', 'red']);
export const CANDIDATES = Object.freeze(['Air', 'The', 'Rayleigh', 'Great']);
export const HACKED_ANSWER = 'Great question ! Air scatters blue …';

export const REWARD_A = 1.2;
export const REWARD_B = -0.3;
export const REWARD_FRESH = 1.0;
export const CRITIC_VALUE = 0.8;
export const REWARD_HONEST = 1.0;
export const REWARD_HACKED = 1.3;
export const BETA_RLHF = 0.5;
export const EPSILON = 0.2;
export const CLIP_LINE = Object.freeze({ lo: 0.6, hi: 1.6, band: Object.freeze([0.8, 1.2]), label: 'ratio r = now / when sampled' });
export const RATIO_START = 1.0;
export const RATIO_MOVED = 1.25;
export const RATIO_SAFE = 1.1;

export const ROWS = deepFreeze({ reference: [0.40, 0.30, 0.20, 0.10], policy: [0.70, 0.15, 0.10, 0.05], hacked: [0.01, 0.01, 0.01, 0.97] });
export const DPO_DEFAULT = Object.freeze({ dChosen: 0.5, dRejected: -0.2, beta: 0.1 });

// ---- derived (math/) ----
export const BT = Object.freeze({ right: bradleyTerry(REWARD_A, REWARD_B), flipped: bradleyTerry(REWARD_B, REWARD_A) });
export const KL = Object.freeze({ honest: klDivergence(ROWS.policy, ROWS.reference), hacked: klDivergence(ROWS.hacked, ROWS.reference) });
export const TOTALS = Object.freeze({
  honest: klPenalizedReward(REWARD_HONEST, KL.honest, BETA_RLHF),
  hacked: klPenalizedReward(REWARD_HACKED, KL.hacked, BETA_RLHF),
});
export const ADVANTAGE = Number((REWARD_FRESH - CRITIC_VALUE).toFixed(10)); // 0.2, without the float tail of 1.0 − 0.8
export const CLIP = Object.freeze({ moved: clippedSurrogate(RATIO_MOVED, ADVANTAGE), safe: clippedSurrogate(RATIO_SAFE, ADVANTAGE) });
export const DPO = Object.freeze(dpoLoss(DPO_DEFAULT));

export const PIPELINE_STAGES = Object.freeze(['pretrain', 'mid-train', 'SFT', 'specialist RL', 'merge', 'polish']);
