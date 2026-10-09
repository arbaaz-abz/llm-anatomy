// Speculative decoding math (speculative-decoding §6; batchSpeedup is also imported by serving-calculator, P4-R3).
// Pure: no DOM, inputs never mutated. FROZEN after Plan 4's shared prep (S6): change only through a shared patch.

import { stepTime } from './serving.js';

const SUM_TOLERANCE = 1e-9;

const isNumber = (x) => typeof x === 'number' && Number.isFinite(x);

function requireProbability(fn, name, value) {
  if (!isNumber(value) || value < 0 || value > 1) throw new RangeError(`${fn}: ${name} must be a number in [0, 1], got ${value}`);
}

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`${fn}: ${name} must be an integer ≥ 0, got ${value}`);
}

function requireNonNegative(fn, name, value) {
  if (!isNumber(value) || value < 0) throw new RangeError(`${fn}: ${name} must be a finite number ≥ 0, got ${value}`);
}

function requireDistribution(fn, name, dist) {
  if (!Array.isArray(dist) || dist.length === 0 || dist.some((x) => !isNumber(x) || x < 0)) {
    throw new RangeError(`${fn}: ${name} must be a non-empty array of numbers ≥ 0`);
  }
  const sum = dist.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > SUM_TOLERANCE) throw new RangeError(`${fn}: ${name} must sum to 1, got ${sum}`);
}

function requirePair(fn, p, q) {
  requireDistribution(fn, 'p', p);
  requireDistribution(fn, 'q', q);
  if (p.length !== q.length) throw new RangeError(`${fn}: p and q must have the same length, got ${p.length} and ${q.length}`);
}

// Tokens produced per round when each of k guesses is kept with probability alpha and a guess counts only if every
// earlier one was kept: (1 − α^(k+1)) / (1 − α), and k + 1 when α = 1. Always at least 1 (the target's own token).
export function expectedTokens(alpha, k) {
  requireProbability('expectedTokens', 'alpha', alpha);
  requireCount('expectedTokens', 'k', k);
  return alpha === 1 ? k + 1 : (1 - alpha ** (k + 1)) / (1 - alpha);
}

// Speed-up at batch 1 when the drafter costs c of a target step per guess: E / (1 + k·c).
export function simpleSpeedup(alpha, k, c) {
  requireNonNegative('simpleSpeedup', 'c', c);
  return expectedTokens(alpha, k) / (1 + k * c);
}

// The chance a token drafted from q survives the check against p: Σ min(p_i, q_i).
export function acceptanceRate(p, q) {
  requirePair('acceptanceRate', p, q);
  return p.reduce((sum, pi, i) => sum + Math.min(pi, q[i]), 0);
}

// One position of the check. accept = min(1, p/q) for the guessed token; rejectMass = 1 − acceptanceRate is the chance
// of any rejection; residual = normalize(max(0, p − q)) is where a rejected guess is redrawn from (all zeros when p = q).
export function verifyToken(p, q, guessIndex) {
  requirePair('verifyToken', p, q);
  if (!Number.isInteger(guessIndex) || guessIndex < 0 || guessIndex >= p.length) {
    throw new RangeError(`verifyToken: guessIndex must be an integer in [0, ${p.length - 1}], got ${guessIndex}`);
  }
  if (q[guessIndex] === 0) throw new RangeError(`verifyToken: the drafter never guesses index ${guessIndex} (q is 0 there)`);
  const leftover = p.map((pi, i) => Math.max(0, pi - q[i]));
  const leftoverSum = leftover.reduce((a, b) => a + b, 0);
  return {
    accept: Math.min(1, p[guessIndex] / q[guessIndex]),
    residual: leftoverSum === 0 ? leftover : leftover.map((x) => x / leftoverSum),
    rejectMass: 1 - acceptanceRate(p, q),
  };
}

// The distribution the whole check produces: q·accept + rejectMass·residual. It equals p whatever q is.
export function outputDistribution(p, q) {
  requirePair('outputDistribution', p, q);
  const { residual, rejectMass } = verifyToken(p, q, q.findIndex((x) => x > 0));
  return p.map((pi, i) => Math.min(pi, q[i]) + rejectMass * residual[i]);
}

// Speed-up at a real batch: the verify pass reads the weights once for batch·(k+1) tokens, so it fades as the batch fills
// the arithmetic. plain = one ordinary decode step; verify = the (k+1)-tokens-per-user pass; round = k drafts + one verify.
// model = { activeParamsPerGpu, weightBytesPerGpu, dModel, actBytesPerElem, kvBytesPerToken, peakTflops, bandwidthTBps, context }.
export function batchSpeedup({ alpha, k, c, batch, model }) {
  requireNonNegative('batchSpeedup', 'c', c);
  if (!Number.isInteger(batch) || batch < 1) throw new RangeError(`batchSpeedup: batch must be a positive integer, got ${batch}`);
  if (!model || typeof model !== 'object') throw new RangeError('batchSpeedup: model must be a step-time model object');
  const tokensPerRound = expectedTokens(alpha, k);
  const plainS = stepTime({ ...model, tokens: batch, seqs: batch }).timeS;
  const verifyS = stepTime({ ...model, tokens: batch * (k + 1), seqs: batch }).timeS;
  const roundS = k * c * plainS + verifyS;
  return { plainMs: plainS * 1e3, verifyMs: verifyS * 1e3, roundMs: roundS * 1e3, tokensPerRound, speedup: (tokensPerRound * plainS) / roundS };
}
