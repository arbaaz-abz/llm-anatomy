// midtraining's hand-authored stand-ins and the fallback copy of the published run numbers (frozen). facts.js reads the
// data first; this copy is what the pure LESSON (no data) draws, and tests/midtraining-page.test.js pins it to the data.
import { deepFreeze } from '@math/core.js';
import { attentionCostRatio } from '@math/schedule.js';

export const WARMUP_DRAW = 0.05; // stand-in: the warmup is drawn 5% of the run wide so it is visible (Nemotron's real one is 0.8%)
export const PLAIN_COSINE_AT = Object.freeze([0.25, 0.5, 0.6, 0.75, 1]); // frame 2's printed positions
export const BRANCH = Object.freeze({ from: 0.6, to: 0.8 }); // frame 3: a branch leaves the plateau at 60% and decays by 80%
export const DECAY_START = 0.8; // frame 3: the WSD curve's decay starts at 80% of the run
export const CURVE_SAMPLES = 100;

// The context stages as nominal labels with their token counts (null = not published).
export const FALLBACK = deepFreeze({
  glm5: { name: 'GLM-5', stages: [['4K', 27e12], ['32K', 1e12], ['128K', 0.5e12], ['200K', 0.05e12]], reportedTotal: 28.5e12 },
  nemotron: { peak: 4.5e-4, floor: 4.5e-6, warmupTokens: 200e9, decayTokens: 5e12, total: 25e12, shape: 'minus-sqrt' },
  minimax: { constantTokens: 19.9e12, decayTokens: 9.3e12 },
  runs: {
    'glm-5': { name: 'GLM-5', labels: ['4K', '32K', '128K', '200K'] },
    'minimax-m2': { name: 'MiniMax-M2', labels: ['8K', '32K', '192K'] },
    'deepseek-v4-pro': { name: 'DeepSeek-V4', labels: ['4K', '16K', '64K', '1M'] },
    'kimi-k3': { name: 'Kimi K3', labels: ['8K', '64K', '256K', '1M'] },
  },
});

// Frame 8: the slowest RoPE pair is a stand-in, chosen so that 200K is three quarters of a turn.
export const DIAL_STAND_IN = deepFreeze({ turnsAt200K: 0.75, turnsAt4K: 0.75 / attentionCostRatio(200, 4) }); // 0.015
