// GRPO with verifiable rewards (rlvr-grpo §11; reused by rlhf-dpo, agentic-rl and distillation).
// Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('<fn>: <arg> must be …').
// FROZEN after Plan 3 S3: every export keeps this exact signature.
import { deepFreeze } from './core.js';

const split = (answer) => answer.split(' ');

// The running example's answer pools (rlvr-grpo §6, tokens whitespace-split), its slot order and its
// checker target (ruling P3-R5). buildGroup(2, GROUP_TOY) is the default group: 8 rows, 39 tokens,
// which agentic-rl reuses row for row and whose row 2 (`7 × 8 = 54`) distillation reuses.
export const GROUP_TOY = deepFreeze({
  correct: ['7 × 8 = 56', '7 + 8 = 56', '56', '8 × 7 = 56', '49 + 7 = 56', '70 − 14 = 56', 'so 56', '7 eights are 56'].map(split),
  wrong: ['7 × 8 = 54', '7 + 8 = 15', '7 × 8 = 48 , so 48', '63', '7 × 8 = 58', '8 × 8 = 64', '7 × 8 = 42', '7 × 8 = 65'].map(split),
  slotOrder: [1, 5, 3, 7, 2, 6, 4, 8],
  target: '56',
});

function requireNumbers(fn, name, values) {
  if (!Array.isArray(values) || values.length === 0) throw new RangeError(`${fn}: ${name} must be a non-empty array`);
  values.forEach((v, i) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new RangeError(`${fn}: ${name}[${i}] must be a finite number, got ${v}`);
  });
}

const isTokens = (tokens) => Array.isArray(tokens) && tokens.every((t) => typeof t === 'string');

// 0/1 outcome reward: the last token is the final answer.
//   (['7','×','8','=','56'], '56') → 1 · (['7','+','8','=','56'], '56') → 1 (wrong working, right final token)
//   (['7','×','8','=','48',',','so','48'], '56') → 0
export function verifyFinalAnswer(tokens, target) {
  if (!isTokens(tokens) || tokens.length === 0) throw new RangeError('verifyFinalAnswer: tokens must be a non-empty array of strings');
  if (typeof target !== 'string') throw new RangeError('verifyFinalAnswer: target must be a string');
  return tokens[tokens.length - 1] === target ? 1 : 0;
}

const allEqual = (values) => values.every((v) => v === values[0]);

// Group mean and population std (divides by G). An all-equal group has std exactly 0 (no float residue).
//   [1,0,0,0,1,0,0,0] → { mean: 0.25, std: 0.4330 } · [1,1,0] → { 0.6667, 0.4714 } · [1,1,1,1] → { 1, 0 }
export function groupStats(rewards) {
  requireNumbers('groupStats', 'rewards', rewards);
  if (allEqual(rewards)) return { mean: rewards[0], std: 0 };
  const mean = rewards.reduce((s, r) => s + r, 0) / rewards.length;
  const variance = rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / rewards.length;
  return { mean, std: Math.sqrt(variance) };
}

// Whether the group has any spread, i.e. std > 0.   [1,0,0,0,1,0,0,0] → true · [1 × 8] → false · [0,0] → false
export function hasSignal(rewards) {
  requireNumbers('hasSignal', 'rewards', rewards);
  return !allEqual(rewards);
}

// Group-relative advantages (R − mean) / std, or R − mean with normalizeStd false (Dr.GRPO, DeepSeek-V3.2).
// Every entry is 0 when std is 0 (no signal).
//   [1,0,0,0,1,0,0,0] → [1.7321, −0.5774 …] (sums to 0) · [1,0,0,1] → [1, −1, −1, 1]
//   [1,1,0], { normalizeStd: false } → [0.3333, 0.3333, −0.6667] · [0,0,0] → [0, 0, 0]
export function groupAdvantages(rewards, { normalizeStd = true } = {}) {
  if (typeof normalizeStd !== 'boolean') throw new RangeError('groupAdvantages: normalizeStd must be true or false');
  const { mean, std } = groupStats(rewards);
  if (std === 0) return rewards.map(() => 0);
  return rewards.map((r) => (normalizeStd ? (r - mean) / std : r - mean));
}

// Σ|A_i|: how much total push a group delivers.   groupAdvantages([1,0,0,0,1,0,0,0]) → 6.9282 · [1,−1,1,−1] → 4
export function totalPush(advantages) {
  requireNumbers('totalPush', 'advantages', advantages);
  return advantages.reduce((s, a) => s + Math.abs(a), 0);
}

// Per-token loss weights, one row per answer. 'sample': 1/(G·|o_i|) (original GRPO); 'token': 1/Σ|o_i| (DAPO).
//   [5, 8, 1] → [[0.0667 ×5], [0.0417 ×8], [0.3333]] · [5, 8, 1], { aggregation: 'token' } → 1/14 everywhere
export function tokenWeights(lengths, { aggregation = 'sample' } = {}) {
  if (!Array.isArray(lengths) || lengths.length === 0) throw new RangeError('tokenWeights: lengths must be a non-empty array');
  lengths.forEach((n, i) => {
    if (!Number.isInteger(n) || n < 1) throw new RangeError(`tokenWeights: lengths[${i}] must be a positive integer, got ${n}`);
  });
  if (aggregation !== 'sample' && aggregation !== 'token') throw new RangeError(`tokenWeights: aggregation must be 'sample' or 'token', got ${aggregation}`);
  const totalTokens = lengths.reduce((s, n) => s + n, 0);
  return lengths.map((n) => Array(n).fill(aggregation === 'sample' ? 1 / (lengths.length * n) : 1 / totalTokens));
}

function checkClipArgs(ratio, advantage, epsLow, epsHigh) {
  if (typeof ratio !== 'number' || !Number.isFinite(ratio) || ratio <= 0) throw new RangeError(`clippedSurrogate: ratio must be a finite number > 0, got ${ratio}`);
  if (typeof advantage !== 'number' || !Number.isFinite(advantage)) throw new RangeError(`clippedSurrogate: advantage must be a finite number, got ${advantage}`);
  if (typeof epsLow !== 'number' || !(epsLow >= 0 && epsLow < 1)) throw new RangeError(`clippedSurrogate: epsLow must be a number in [0, 1), got ${epsLow}`);
  if (typeof epsHigh !== 'number' || !Number.isFinite(epsHigh) || epsHigh < 0) throw new RangeError(`clippedSurrogate: epsHigh must be a finite number ≥ 0, got ${epsHigh}`);
}

// PPO/GRPO clipped surrogate for one token: min(r·A, clip(r, 1−ε_low, 1+ε_high)·A).
// `clipped` = true means the gradient is zero: A > 0 and r > 1+ε_high, or A < 0 and r < 1−ε_low.
// Moving the other way is never clipped.
//   (1.25, 1.7321) → { 2.0785, clipped } · (1.25, 1.7321, { epsHigh: 0.28 }) → { 2.1651, not clipped }
//   (0.75, −0.5774) → { −0.4619, clipped } · (1.50, −0.5774) → { −0.8661, not clipped } · (1.25, 0.2) → { 0.24, clipped }
export function clippedSurrogate(ratio, advantage, { epsLow = 0.2, epsHigh = 0.2 } = {}) {
  checkClipArgs(ratio, advantage, epsLow, epsHigh);
  const bounded = Math.min(Math.max(ratio, 1 - epsLow), 1 + epsHigh);
  return {
    objective: Math.min(ratio * advantage, bounded * advantage),
    clipped: (advantage > 0 && ratio > 1 + epsHigh) || (advantage < 0 && ratio < 1 - epsLow),
  };
}

function checkPool(name, pool, needed) {
  if (!Array.isArray(pool)) throw new RangeError(`buildGroup: ${name} must be an array of answers`);
  pool.forEach((tokens, i) => {
    if (!isTokens(tokens)) throw new RangeError(`buildGroup: ${name}[${i}] must be an array of strings`);
  });
  if (pool.length < needed) throw new RangeError(`buildGroup: ${name} must hold at least ${needed} answers, got ${pool.length}`);
}

function checkSlotOrder(slotOrder) {
  if (!Array.isArray(slotOrder) || slotOrder.length === 0) throw new RangeError('buildGroup: slotOrder must be a non-empty array');
  const G = slotOrder.length;
  const sorted = [...slotOrder].sort((a, b) => a - b);
  if (!sorted.every((slot, i) => slot === i + 1)) throw new RangeError(`buildGroup: slotOrder must be a permutation of 1–${G}`);
}

// Deterministic group assembly. The first k entries of slotOrder (1-based) name the slots that hold a
// correct answer; walking slots 1..G in order, a correct slot takes the next unused correct answer and any
// other slot the next unused wrong one. Rows are fresh arrays.
//   k = 2 → [C1, W1, W2, W3, C2, W4, W5, W6] · k = 3 → [C1, W1, C2, W2, C3, W3, W4, W5] · k = 8 → [C1 … C8]
export function buildGroup(k, { correct, wrong, slotOrder } = {}) {
  checkPool('correct', correct, 0);
  checkPool('wrong', wrong, 0);
  checkSlotOrder(slotOrder);
  const G = slotOrder.length;
  if (!Number.isInteger(k) || k < 0 || k > G) throw new RangeError(`buildGroup: k must be an integer 0–${G}, got ${k}`);
  checkPool('correct', correct, k);
  checkPool('wrong', wrong, G - k);
  const correctSlots = new Set(slotOrder.slice(0, k));
  const slots = Array.from({ length: G }, (_, i) => correctSlots.has(i + 1));
  const rankWithin = (i) => slots.slice(0, i).filter((isCorrect) => isCorrect === slots[i]).length;
  return slots.map((isCorrect, i) => [...(isCorrect ? correct : wrong)[rankWithin(i)]]);
}
