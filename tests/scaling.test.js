import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  CHINCHILLA_FIT, tokensPerParam, fittedLoss, isoFlopLoss, computeOptimal, tokensForLoss, lifetimeFlops,
  inferenceAwareOptimum, newtonSchulzSingular,
} from '../math/scaling.js';
import { trainingFlops } from '../math/scale.js';

const sig = (x, s) => Number(x.toPrecision(s));
const round = (x, d) => Number(x.toFixed(d));
const close = (a, b, rel = 5e-4) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} ≉ ${b}`);

const DEEPSEEK_SCHEDULE = Object.freeze([...Array(8).fill([3.4445, -4.775, 2.0315]), ...Array(2).fill([2, -1.5, 0.5])]);

// ---- P3-R13: the page constant restates data/papers.json; they must agree ----

test('CHINCHILLA_FIT equals papers.chinchilla-refit-2024 in the data (P3-R13)', async () => {
  const papers = JSON.parse(await readFile(new URL('../data/papers.json', import.meta.url), 'utf8'));
  const facts = papers.entries.find((e) => e.id === 'chinchilla-refit-2024').facts;
  for (const key of ['E', 'A', 'B', 'alpha', 'beta']) assert.equal(CHINCHILLA_FIT[key], facts[key].value, key);
  assert.deepEqual(Object.keys(CHINCHILLA_FIT).sort(), ['A', 'B', 'E', 'alpha', 'beta']);
});

test('CHINCHILLA_FIT is frozen', () => {
  assert.ok(Object.isFrozen(CHINCHILLA_FIT));
});

// ---- tokensPerParam ----

test('tokensPerParam: D ÷ N, the page\'s one definition (frame 8 and the toy)', () => {
  assert.equal(round(tokensPerParam(33e12, 49e9), 2), 673.47);
  assert.equal(round(tokensPerParam(32e12, 13e9), 2), 2461.54);
  assert.equal(round(tokensPerParam(25e12, 12e9), 2), 2083.33);
  assert.equal(round(tokensPerParam(15.6e12, 405e9), 2), 38.52);
  assert.equal(round(tokensPerParam(33e12, 1.6e12), 2), 20.63);
  assert.equal(round(tokensPerParam(25e12, 120e9), 2), 208.33);
  assert.equal(round(tokensPerParam(32e12, 284e9), 2), 112.68);
  assert.equal(Math.round(tokensPerParam(5.9e12, 32e9)), 184);
  assert.equal(Math.round(tokensPerParam(5.9e12, 7e9)), 843);
});

// ---- fittedLoss and isoFlopLoss ----

test('fittedLoss at the 10^24 optimum is 1.9597', () => {
  assert.equal(round(fittedLoss(9.586e10, 1.739e12), 4), 1.9597);
});

test('fittedLoss terms: E + A / N^alpha + B / D^beta', () => {
  const { E, A, B, alpha, beta } = CHINCHILLA_FIT;
  assert.equal(fittedLoss(1e10, 1e12), E + A / 1e10 ** alpha + B / 1e12 ** beta);
});

test('isoFlopLoss: D = C / 6N and the loss along the 10^24 valley (frames 2-3)', () => {
  const at = (n) => isoFlopLoss(1e24, n);
  const cases = [[1e9, 2.1874], [1e10, 2.0079], [3e10, 1.972], [9.586e10, 1.9597], [3e11, 1.9718], [1e12, 2.0133]];
  for (const [n, loss] of cases) assert.equal(round(at(n).loss, 4), loss, `N = ${n}`);
  close(at(1e12).D, 1.667e11, 1e-3);
  close(at(1e12).tokensPerParam, 0.1667, 1e-3);
  close(at(1e9).D, 1.667e14, 1e-3);
  close(at(1e9).tokensPerParam, 166666.67, 1e-3);
});

test('isoFlopLoss uses trainingFlops: 6 · N · D gives the budget back', () => {
  const { D } = isoFlopLoss(1e24, 5e10);
  close(trainingFlops({ params: 5e10, tokens: D }), 1e24, 1e-12);
});

// ---- computeOptimal ----

test('computeOptimal: the frame 4-5 and try-this 3 table', () => {
  const rows = [[1e22, 9.045e9, 1.843e11, 20.37, 2.1411], [1e24, 9.586e10, 1.739e12, 18.14, 1.9597], [1e26, 1.016e12, 1.641e13, 16.15, 1.8799]];
  for (const [c, n, d, ratio, loss] of rows) {
    const o = computeOptimal(c);
    close(o.N, n, 5e-4); close(o.D, d, 5e-4);
    assert.equal(round(o.tokensPerParam, 2), ratio, `C = ${c}`);
    assert.equal(round(o.loss, 4), loss, `C = ${c}`);
  }
});

test('computeOptimal spends the whole budget and beats nearby model sizes', () => {
  for (const c of [1e22, 1e24, 1e26]) {
    const o = computeOptimal(c);
    close(trainingFlops({ params: o.N, tokens: o.D }), c, 1e-12);
    for (const k of [0.5, 0.8, 0.95, 1.05, 1.25, 2]) {
      assert.ok(o.loss <= isoFlopLoss(c, o.N * k).loss, `C = ${c}, ${k}x`);
    }
    assert.equal(o.loss, fittedLoss(o.N, o.D));
  }
});

// ---- tokensForLoss ----

test('tokensForLoss inverts fittedLoss', () => {
  close(tokensForLoss(5e10, fittedLoss(5e10, 2e12)), 2e12, 1e-9);
  close(tokensForLoss(1e12, fittedLoss(1e12, 3e11)), 3e11, 1e-9);
});

test('tokensForLoss is Infinity when N alone cannot reach the loss', () => {
  assert.equal(tokensForLoss(1e9, 2.0), Infinity);
  assert.equal(tokensForLoss(1e10, CHINCHILLA_FIT.E), Infinity);
});

// ---- lifetimeFlops ----

test('lifetimeFlops: 6ND + 2N · inference tokens (frame 6)', () => {
  close(lifetimeFlops(9.586e10, 1.739e12, 1e14), 2.017e25, 1e-3);
  close(lifetimeFlops(9.586e10, 1.739e12, 0), trainingFlops({ params: 9.586e10, tokens: 1.739e12 }), 1e-12);
});

// ---- inferenceAwareOptimum ----

test('inferenceAwareOptimum: serving 100T tokens at the 10^24 loss (frame 7)', () => {
  const r = inferenceAwareOptimum({ targetLoss: computeOptimal(1e24).loss, inferenceTokens: 1e14 });
  close(r.N, 2.917e10, 1e-3); close(r.D, 1.444e13, 1e-3);
  assert.equal(round(r.tokensPerParam, 2), 494.93);
  close(r.train, 2.528e24, 1e-3); close(r.serve, 5.835e24, 1e-3); close(r.total, 8.362e24, 1e-3);
  assert.equal(r.total, r.train + r.serve);
});

test('inferenceAwareOptimum: try-this 2, the cheapest model at each serving volume', () => {
  const target = computeOptimal(1e24).loss;
  const base = lifetimeFlops(computeOptimal(1e24).N, computeOptimal(1e24).D, 0);
  const rows = [[0, 9.605e10, 18.07, 0], [1e12, 7.898e10, 27.11, 0.016], [1e13, 4.842e10, 88.83, 0.24], [1e14, 2.917e10, 494.93, 0.585], [1e15, 2.101e10, 3005.49, 0.741]];
  for (const [serve, n, ratio, saving] of rows) {
    const r = inferenceAwareOptimum({ targetLoss: target, inferenceTokens: serve });
    close(r.N, n, 1e-3);
    assert.equal(round(r.tokensPerParam, 2), ratio, `serve ${serve}`);
    const opt = computeOptimal(1e24);
    assert.equal(Math.abs(round(1 - r.total / lifetimeFlops(opt.N, opt.D, serve), 3)), saving, `serve ${serve}`);
  }
  assert.ok(base > 0);
});

test('inferenceAwareOptimum with 0 serving tokens lands within one grid step of computeOptimal', () => {
  const o = computeOptimal(1e24);
  const r = inferenceAwareOptimum({ targetLoss: o.loss, inferenceTokens: 0 });
  const step = 10 ** ((13 - 8) / 2000);
  assert.ok(r.N / o.N < step && o.N / r.N < step, `${r.N} vs ${o.N}`);
});

test('inferenceAwareOptimum: tokens per parameter never shrink as serving grows', () => {
  const target = computeOptimal(1e24).loss;
  const ratios = [0, 1e12, 1e13, 1e14, 1e15].map((s) => inferenceAwareOptimum({ targetLoss: target, inferenceTokens: s }).tokensPerParam);
  ratios.slice(1).forEach((r, i) => assert.ok(r >= ratios[i], `${r} < ${ratios[i]}`));
});

test('inferenceAwareOptimum throws when no model size in range reaches the loss', () => {
  assert.throws(() => inferenceAwareOptimum({ targetLoss: 1.8, inferenceTokens: 0 }), RangeError);
});

// ---- newtonSchulzSingular ----

test('newtonSchulzSingular: the frame 9-10 trace for [3, 0.3] under DeepSeek-V4\'s 8 + 2 schedule', () => {
  const trace = newtonSchulzSingular([3, 0.3], DEEPSEEK_SCHEDULE);
  assert.equal(trace.length, 11);
  const expected = [[0.995, 0.0995], [0.7047, 0.3381], [1.1093, 0.9889], [0.7153, 0.7097], [1.0967, 1.1034], [0.7020, 0.7087], [1.1125, 1.1047], [0.7192, 0.7100], [1.0918, 1.1031], [1.0071, 1.0094], [1.0, 1.0]];
  expected.forEach((row, i) => row.forEach((v, j) => assert.equal(round(trace[i][j], 4), v, `step ${i}, value ${j}`)));
  assert.ok(Math.abs(trace[10][0] - 1) < 1e-4 && Math.abs(trace[10][1] - 1) < 1e-4);
  assert.equal(sig(trace[10][0], 7), 1.000027);
  assert.equal(sig(trace[10][1], 7), 1.000047);
});

test('newtonSchulzSingular: (2, -1.5, 0.5) has 1 as a fixed point, and step 0 is Frobenius-normalized', () => {
  const fixed = newtonSchulzSingular([5], [[2, -1.5, 0.5], [2, -1.5, 0.5]]);
  assert.deepEqual(fixed, [[1], [1], [1]]);
  const trace = newtonSchulzSingular([3, 4], []);
  assert.deepEqual(trace, [[0.6, 0.8]]);
});

// ---- errors and immutability ----

test('bad arguments throw RangeError(<fn>: <arg> must be …)', () => {
  assert.throws(() => tokensPerParam(0, 1), /^RangeError: tokensPerParam: D must be/);
  assert.throws(() => tokensPerParam(1, -1), /^RangeError: tokensPerParam: N must be/);
  assert.throws(() => fittedLoss(Number.NaN, 1), /^RangeError: fittedLoss: N must be/);
  assert.throws(() => isoFlopLoss(0, 1e9), /^RangeError: isoFlopLoss: C must be/);
  assert.throws(() => computeOptimal(-1), /^RangeError: computeOptimal: C must be/);
  assert.throws(() => tokensForLoss(1e9, Number.NaN), /^RangeError: tokensForLoss: targetLoss must be/);
  assert.throws(() => lifetimeFlops(1e9, 1e9, -1), /^RangeError: lifetimeFlops: inferenceTokens must be/);
  assert.throws(() => inferenceAwareOptimum({ targetLoss: 2, inferenceTokens: -1 }), /^RangeError: inferenceAwareOptimum: inferenceTokens must be/);
  assert.throws(() => inferenceAwareOptimum({ targetLoss: 2, inferenceTokens: 0, logNMin: 9, logNMax: 8 }), /^RangeError: inferenceAwareOptimum: logNMax must be/);
  assert.throws(() => inferenceAwareOptimum({ targetLoss: 2, inferenceTokens: 0, steps: 0 }), /^RangeError: inferenceAwareOptimum: steps must be/);
  assert.throws(() => newtonSchulzSingular([], []), /^RangeError: newtonSchulzSingular: singularValues must be/);
  assert.throws(() => newtonSchulzSingular([0, 0], []), /^RangeError: newtonSchulzSingular: singularValues must be/);
  assert.throws(() => newtonSchulzSingular([1], [[1, 2]]), /^RangeError: newtonSchulzSingular: schedule/);
  assert.throws(() => fittedLoss(1e9, 1e9, { E: 1 }), /^RangeError: fittedLoss: fit must be/);
});

test('inputs are not mutated', () => {
  const values = Object.freeze([3, 0.3]);
  const schedule = Object.freeze(DEEPSEEK_SCHEDULE.map((row) => Object.freeze([...row])));
  const fit = Object.freeze({ ...CHINCHILLA_FIT });
  newtonSchulzSingular(values, schedule);
  computeOptimal(1e24, fit);
  isoFlopLoss(1e24, 1e11, fit);
  inferenceAwareOptimum({ targetLoss: 1.96, inferenceTokens: 1e14, fit });
  assert.deepEqual([...values], [3, 0.3]);
  assert.equal(JSON.stringify(CHINCHILLA_FIT), JSON.stringify(fit));
});
