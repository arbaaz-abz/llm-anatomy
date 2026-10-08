// decoder-recap's stage numbers (storyboard §4–§5 "Numbers shown"). Pure, no DOM. The two worked rows are earlier pages'
// hand-picked stand-ins (decoder-anatomy's row "sat", attention's head-A query); everything else comes from math/.
import { deepFreeze, formatCount, formatBytes } from '@math/core.js';
import { rms, meanAndSd, rmsNorm, layerNorm } from '@math/norms.js';
import { TOY, scores, scaleScores, attentionWeights } from '@math/attention.js';
import { kvBytesPerToken } from '@math/memory.js';
import { configFor, breakdownFor, cacheBytes, INITIAL_STATE, MODERN_STATE, EXPERT, int } from './format.js';

// ---- frame 2: decoder-anatomy's row "sat" (tests/decoder-recap-page.test.js pins it to decoder-anatomy's X[2]) ----
export const X_SAT = deepFreeze([0, 1, 0.5, 0, -0.5, 1, 0, 0.5]);
export const ROW_NORMS = deepFreeze({
  layer: layerNorm(X_SAT),
  rms: rmsNorm(X_SAT),
  mean: meanAndSd(X_SAT).mean,
  spread: meanAndSd(X_SAT).sd,
  rootMeanSquare: rms(X_SAT),
});

// ---- frame 7: attention's head-A query for "sat" against the keys The, cat, sat; scores are divided by sqrt(d_head) = 2 ----
const HEAD_A = TOY.heads.A;
export const Q_SAT = deepFreeze(HEAD_A.Q[2]);
export const KEYS = deepFreeze(HEAD_A.K.slice(0, 3));
export const KEY_NAMES = Object.freeze(TOY.tokens.slice(0, 3));
export const SCALE = Math.sqrt(TOY.dHead);
export const Q_MULTIPLES = Object.freeze([1, 10, 100]); // the multiples frame 7 shows: q, q grew 10×, q grew again

const timesQ = (k) => Q_SAT.map((v) => v * k);
const rowScores = (q, keys) => scaleScores(scores([q], keys), SCALE)[0];
export const QK = deepFreeze({
  plain: (k = 1) => rowScores(timesQ(k), KEYS),
  plainWeights: (k = 1) => attentionWeights([rowScores(timesQ(k), KEYS)])[0],
  normed: (k = 1) => rowScores(rmsNorm(timesQ(k)), KEYS.map((key) => rmsNorm(key))),
  normedWeights: (k = 1) => attentionWeights([rowScores(rmsNorm(timesQ(k)), KEYS.map((key) => rmsNorm(key)))])[0],
  qRms: rms(Q_SAT),
  keyRms: KEYS.map((key) => rms(key)),
});

// ---- GPT-3 and the fully modernized block (frames 1, 3, 4, 5, 6, 10) ----
const GPT3 = breakdownFor(INITIAL_STATE);
const MODERN = breakdownFor(MODERN_STATE);
const DENSE_MODERN = breakdownFor({ ...MODERN_STATE, experts: 'dense', mlp: 'swiglu' });
const GPT3_CONFIG = configFor(INITIAL_STATE);
const SHARED_KV = 8;

export const GPT3_FACTS = deepFreeze({
  layers: GPT3_CONFIG.layers,
  heads: GPT3_CONFIG.attention.nHeads,
  dModel: GPT3_CONFIG.dModel,
  total: GPT3.total,
  totalText: formatCount(GPT3.total, { digits: 4 }), // "174.6B" (frame 1)
  totalShort: formatCount(GPT3.total), // "175B" (frame 10)
  cacheBytes: cacheBytes(INITIAL_STATE),
  cacheText: formatBytes(cacheBytes(INITIAL_STATE)),
  positionTable: GPT3.parts.positional,
  positions: GPT3_CONFIG.maxPositions,
  geluHidden: GPT3_CONFIG.mlp.hidden,
});
export const MODERN_FACTS = deepFreeze({
  stored: formatCount(MODERN.total),
  active: formatCount(MODERN.active),
  dense: formatCount(DENSE_MODERN.total),
  cacheText: formatBytes(cacheBytes(MODERN_STATE)),
  cacheBytes: cacheBytes(MODERN_STATE),
  sharedKv: SHARED_KV,
  group: GPT3_CONFIG.attention.nHeads / SHARED_KV, // 12 query heads share each set
  experts: EXPERT.routed,
  topK: EXPERT.topK,
  expertHidden: EXPERT.hidden,
  swigluHidden: configFor(MODERN_STATE).mlp.hidden,
  router: MODERN.parts.router,
});
export const KV_FORMULA = (kvHeads) => `2 × ${GPT3_CONFIG.layers} × ${kvHeads} × ${GPT3_CONFIG.attention.dHead} × 2 B = ${int(kvBytesPerToken({ layers: GPT3_CONFIG.layers, kvHeads, headDim: GPT3_CONFIG.attention.dHead, bytesPerElem: 2 }))} B`;

// ---- the 64 expert tiles of frame 5: eight lit (top-8), chosen once so the picture never changes ----
export const LIT_EXPERTS = Object.freeze([3, 10, 21, 26, 38, 45, 52, 60]);
