// Scaling laws (scaling-laws §11): the Chinchilla loss fit as Epoch AI re-estimated it (2024), the compute-optimal
// split, over-training, and Muon's Newton-Schulz polynomial on singular values. Pure: no DOM, inputs never mutated.
// 6ND is NOT defined here: trainingFlops / FLOPS_PER_PARAM_TOKEN come from ./scale.js (README lesson 16).
import { deepFreeze } from './core.js';
import { trainingFlops, FLOPS_PER_PARAM_TOKEN } from './scale.js';

// L(N, D) = E + A / N^alpha + B / D^beta. Restates data/papers.json "chinchilla-refit-2024"; tests/scaling.test.js keeps them equal (P3-R13).
export const CHINCHILLA_FIT = deepFreeze({ E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658 });

const SERVE_FLOPS_PER_PARAM_TOKEN = 2; // forward pass only
const DEFAULT_LOG_N_MIN = 8;
const DEFAULT_LOG_N_MAX = 13;
const DEFAULT_STEPS = 2000;

const isPositive = (x) => typeof x === 'number' && Number.isFinite(x) && x > 0;

function requirePositive(fn, name, value) {
  if (!isPositive(value)) throw new RangeError(`${fn}: ${name} must be a positive finite number, got ${value}`);
}

function requireNonNegative(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new RangeError(`${fn}: ${name} must be a finite number >= 0, got ${value}`);
}

function requireFit(fn, fit) {
  const ok = fit && ['E', 'A', 'B', 'alpha', 'beta'].every((key) => isPositive(fit[key]));
  if (!ok) throw new RangeError(`${fn}: fit must be { E, A, B, alpha, beta }, each a positive finite number`);
}

// D / N. Callers pass active parameters for a MoE (the page's one definition).
export function tokensPerParam(D, N) {
  requirePositive('tokensPerParam', 'D', D);
  requirePositive('tokensPerParam', 'N', N);
  return D / N;
}

export function fittedLoss(N, D, fit = CHINCHILLA_FIT) {
  requirePositive('fittedLoss', 'N', N);
  requirePositive('fittedLoss', 'D', D);
  requireFit('fittedLoss', fit);
  return fit.E + fit.A / N ** fit.alpha + fit.B / D ** fit.beta;
}

// The loss of an N-parameter model given budget C: it reads D = C / 6N tokens.
export function isoFlopLoss(C, N, fit = CHINCHILLA_FIT) {
  requirePositive('isoFlopLoss', 'C', C);
  requirePositive('isoFlopLoss', 'N', N);
  requireFit('isoFlopLoss', fit);
  const D = C / FLOPS_PER_PARAM_TOKEN / N;
  return { D, tokensPerParam: tokensPerParam(D, N), loss: fittedLoss(N, D, fit) };
}

// Closed form: N* = G (C/6)^(beta / (alpha + beta)), G = (alpha A / beta B)^(1 / (alpha + beta)), D* = C / 6N*.
export function computeOptimal(C, fit = CHINCHILLA_FIT) {
  requirePositive('computeOptimal', 'C', C);
  requireFit('computeOptimal', fit);
  const { A, B, alpha, beta } = fit;
  const G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta));
  const N = G * (C / FLOPS_PER_PARAM_TOKEN) ** (beta / (alpha + beta));
  const D = C / FLOPS_PER_PARAM_TOKEN / N;
  return { N, D, tokensPerParam: tokensPerParam(D, N), loss: fittedLoss(N, D, fit) };
}

// D(N) = (B / (L* - E - A N^-alpha))^(1/beta); Infinity when N alone cannot reach L*.
export function tokensForLoss(N, targetLoss, fit = CHINCHILLA_FIT) {
  requirePositive('tokensForLoss', 'N', N);
  requirePositive('tokensForLoss', 'targetLoss', targetLoss);
  requireFit('tokensForLoss', fit);
  const room = targetLoss - fit.E - fit.A / N ** fit.alpha;
  return room > 0 ? (fit.B / room) ** (1 / fit.beta) : Infinity;
}

// Training (6ND, from scale.js) plus serving at 2 FLOPs per parameter per token.
export function lifetimeFlops(N, D, inferenceTokens) {
  requireNonNegative('lifetimeFlops', 'inferenceTokens', inferenceTokens);
  return trainingFlops({ params: N, tokens: D }) + SERVE_FLOPS_PER_PARAM_TOKEN * N * inferenceTokens;
}

// The cheapest lifetime model that still reaches targetLoss: a grid search over log10 N.
export function inferenceAwareOptimum({ targetLoss, inferenceTokens, fit = CHINCHILLA_FIT, logNMin = DEFAULT_LOG_N_MIN, logNMax = DEFAULT_LOG_N_MAX, steps = DEFAULT_STEPS }) {
  requirePositive('inferenceAwareOptimum', 'targetLoss', targetLoss);
  requireNonNegative('inferenceAwareOptimum', 'inferenceTokens', inferenceTokens);
  requireFit('inferenceAwareOptimum', fit);
  if (!(Number.isFinite(logNMin) && Number.isFinite(logNMax) && logNMax > logNMin)) throw new RangeError(`inferenceAwareOptimum: logNMax must be greater than logNMin, got ${logNMin} and ${logNMax}`);
  if (!Number.isInteger(steps) || steps < 1) throw new RangeError(`inferenceAwareOptimum: steps must be an integer >= 1, got ${steps}`);
  let best = null;
  for (let i = 0; i <= steps; i += 1) {
    const N = 10 ** (logNMin + ((logNMax - logNMin) * i) / steps);
    const D = tokensForLoss(N, targetLoss, fit);
    if (!Number.isFinite(D)) continue;
    const total = lifetimeFlops(N, D, inferenceTokens);
    if (best === null || total < best.total) best = { N, D, total };
  }
  if (best === null) throw new RangeError(`inferenceAwareOptimum: targetLoss ${targetLoss} is not reachable by any model size in 10^${logNMin}-10^${logNMax}`);
  const train = trainingFlops({ params: best.N, tokens: best.D });
  return { N: best.N, D: best.D, tokensPerParam: tokensPerParam(best.D, best.N), train, serve: best.total - train, total: train + (best.total - train) };
}

// Muon applies a X + b X^3 + c X^5 to each singular value of the Frobenius-normalized update. Returns the whole
// trace: index 0 is the normalized input, index k the values after k steps of `schedule` ([a, b, c] per step).
export function newtonSchulzSingular(singularValues, schedule) {
  const valid = Array.isArray(singularValues) && singularValues.length > 0 && singularValues.every(isPositive);
  if (!valid) throw new RangeError('newtonSchulzSingular: singularValues must be a non-empty array of positive finite numbers');
  if (!Array.isArray(schedule) || !schedule.every((s) => Array.isArray(s) && s.length === 3 && s.every(Number.isFinite))) {
    throw new RangeError('newtonSchulzSingular: schedule must be a list of [a, b, c] triples of finite numbers');
  }
  const norm = Math.sqrt(singularValues.reduce((sum, s) => sum + s * s, 0));
  const start = singularValues.map((s) => s / norm);
  return schedule.reduce((trace, [a, b, c]) => {
    const last = trace[trace.length - 1];
    return [...trace, last.map((x) => a * x + b * x ** 3 + c * x ** 5)];
  }, [start]);
}
