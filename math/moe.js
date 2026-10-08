// Mixture-of-Experts routing and balancing (moe storyboard §6). Pure: no DOM, inputs never mutated.
// Parameter counts are not here: they come from math/params.js (paramBreakdown, mlpParams).
import { deepFreeze, softmax, randomMatrix } from './core.js';

// Hand-picked router scores for The, cat, sat, down over E1–E8 (quarter grid; "sat" picks E3 and E6 as in decoder-anatomy).
export const ROUTER_TOY = deepFreeze([
  [0.5, 1.5, 0, -0.5, 1.0, 0, -1, 0.25],
  [-0.5, 0, 0.5, 1.75, 0, 1.25, 0.25, -1],
  [0, -0.5, 2.0, 0.25, -1, 1.5, 0.5, 0],
  [1.25, 0, 0.5, -0.25, 0.75, 0, 1.0, -0.5],
]);

const isInt = (n) => Number.isInteger(n);

function checkScores(fn, scores) {
  if (!Array.isArray(scores) || scores.length === 0) throw new RangeError(`${fn}: scores must be a non-empty array`);
  if (!scores.every(Number.isFinite)) throw new RangeError(`${fn}: scores must be finite numbers`);
}

function checkK(fn, k, max) {
  if (!isInt(k) || k < 1 || k > max) throw new RangeError(`${fn}: k must be an integer from 1 to ${max}, got ${k}`);
}

function checkBias(fn, bias, n) {
  if (!Array.isArray(bias) || bias.length !== n) throw new RangeError(`${fn}: bias must have ${n} entries, got ${bias?.length}`);
}

// Indices (0-based) of the k highest scores after adding the bias (the bias only affects the choice). Ties go to the lower index.
export function routeTopK(scores, k, bias = scores.map(() => 0)) {
  checkScores('routeTopK', scores);
  checkK('routeTopK', k, scores.length);
  checkBias('routeTopK', bias, scores.length);
  return scores
    .map((score, index) => ({ index, value: score + bias[index] }))
    .sort((a, b) => b.value - a.value || a.index - b.index)
    .slice(0, k)
    .map((entry) => entry.index);
}

// Softmax over the chosen experts' RAW scores (never the biased ones).
export function gateWeights(scores, picks) {
  checkScores('gateWeights', scores);
  if (!Array.isArray(picks) || picks.length === 0) throw new RangeError('gateWeights: picks must be a non-empty array');
  const bad = picks.find((p) => !isInt(p) || p < 0 || p >= scores.length);
  if (bad !== undefined) throw new RangeError(`gateWeights: pick ${bad} is outside 0–${scores.length - 1}`);
  if (new Set(picks).size !== picks.length) throw new RangeError('gateWeights: picks must be distinct');
  return softmax(picks.map((p) => scores[p]));
}

// Tokens per expert for a batch of score rows.
export function expertLoads(rows, k, bias) {
  if (!Array.isArray(rows) || rows.length === 0) throw new RangeError('expertLoads: rows must be a non-empty array');
  const loads = new Array(rows[0].length).fill(0);
  rows.forEach((row) => routeTopK(row, k, bias).forEach((expert) => { loads[expert] += 1; }));
  return loads;
}

function checkBatch(fn, { tokens, k }) {
  if (!isInt(tokens) || tokens < 1) throw new RangeError(`${fn}: tokens must be a positive integer, got ${tokens}`);
  if (!isInt(k) || k < 1) throw new RangeError(`${fn}: k must be a positive integer, got ${k}`);
}

// Busiest load ÷ fair share (tokens · k ÷ experts). The one definition of imbalance (README lesson 16).
export function imbalance(loads, { tokens, k }) {
  if (!Array.isArray(loads) || loads.length === 0) throw new RangeError('imbalance: loads must be a non-empty array');
  checkBatch('imbalance', { tokens, k });
  return Math.max(...loads) / ((tokens * k) / loads.length);
}

// Aux-loss-free update: bias_e += gamma · sign(fairShare − load_e). Returns a new array.
export function balanceStep(bias, loads, { gamma, tokens, k }) {
  if (!Array.isArray(bias) || !Array.isArray(loads) || bias.length !== loads.length) throw new RangeError('balanceStep: bias and loads must have the same length');
  if (!(Number.isFinite(gamma) && gamma >= 0)) throw new RangeError(`balanceStep: gamma must be a number ≥ 0, got ${gamma}`);
  checkBatch('balanceStep', { tokens, k });
  const fairShare = (tokens * k) / loads.length;
  return bias.map((b, e) => b + gamma * Math.sign(fairShare - loads[e]));
}

function checkRun({ tokens, experts, k, steps, popularity }) {
  if (!isInt(experts) || experts < 1) throw new RangeError(`simulateBalancing: experts must be a positive integer, got ${experts}`);
  checkK('simulateBalancing', k, experts);
  if (!isInt(steps) || steps < 1) throw new RangeError(`simulateBalancing: steps must be a positive integer, got ${steps}`);
  if (popularity.length !== experts) throw new RangeError(`simulateBalancing: popularity must have ${experts} entries, got ${popularity.length}`);
  checkBatch('simulateBalancing', { tokens, k });
}

// Each step routes a fresh seeded batch (randomMatrix(tokens, experts, seed + step) plus the popularity offset)
// with the current bias, records what happened, then updates the bias. `bias` is the bias that step chose with.
export function simulateBalancing({ tokens, experts, k, gamma, steps, seed, popularity = null }) {
  const offset = popularity ?? new Array(experts).fill(0);
  checkRun({ tokens, experts, k, steps, popularity: offset });
  const history = [];
  let bias = new Array(experts).fill(0);
  for (let step = 0; step < steps; step += 1) {
    const rows = randomMatrix(tokens, experts, seed + step, 1).map((row) => row.map((score, e) => score + offset[e]));
    const loads = expertLoads(rows, k, bias);
    history.push({ step, loads, imbalance: imbalance(loads, { tokens, k }), bias });
    bias = balanceStep(bias, loads, { gamma, tokens, k });
  }
  return history;
}

// Number of ways to choose k of n experts (BigInt, exact).
export function expertCombinations(n, k) {
  if (!isInt(n) || n < 1) throw new RangeError(`expertCombinations: n must be a positive integer, got ${n}`);
  if (!isInt(k) || k < 0 || k > n) throw new RangeError(`expertCombinations: k must be an integer from 0 to ${n}, got ${k}`);
  let ways = 1n;
  for (let i = 0n; i < BigInt(k); i += 1n) ways = (ways * (BigInt(n) - i)) / (i + 1n);
  return ways;
}
