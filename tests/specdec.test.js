import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32 } from '../math/core.js';
import { VOCAB } from '../math/sampling.js';
import { RUNNING_EXAMPLE } from '../math/serving.js';
import { expectedTokens, simpleSpeedup, acceptanceRate, verifyToken, outputDistribution, batchSpeedup } from '../math/specdec.js';

const printed = (x, d) => Number(x.toFixed(d));
const close = (a, b, eps = 1e-12) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const M = { ...RUNNING_EXAMPLE, context: 1024 }; // the running example at context 1,024
const P = [0.6, 0.25, 0.1, 0.05];
const Q = [0.7, 0.2, 0.05, 0.05];

test('expectedTokens: the storyboard table, k = 1 / 2 / 3 / 5 at four acceptance rates', () => {
  const rows = {
    0.5: [1.5, 1.75, 1.88, 1.97],
    0.7: [1.7, 2.19, 2.53, 2.94],
    0.85: [1.85, 2.57, 3.19, 4.15],
    0.9: [1.9, 2.71, 3.44, 4.69],
  };
  for (const [alpha, expected] of Object.entries(rows)) {
    assert.deepEqual([1, 2, 3, 5].map((k) => printed(expectedTokens(Number(alpha), k), 2)), expected, `alpha ${alpha}`);
  }
  assert.equal(printed(expectedTokens(0.7, 8), 2), 3.2);
});

test('expectedTokens: k = 0 is 1; alpha 1 gives k + 1; alpha 0 gives 1; grows with k and alpha, never above k + 1', () => {
  for (const alpha of [0, 0.3, 0.7, 1]) assert.equal(expectedTokens(alpha, 0), 1);
  for (const k of [0, 1, 4, 9]) assert.equal(expectedTokens(1, k), k + 1);
  assert.equal(expectedTokens(0, 5), 1);
  for (const alpha of [0.2, 0.5, 0.85, 0.99]) {
    for (let k = 0; k < 10; k += 1) {
      assert.ok(expectedTokens(alpha, k + 1) > expectedTokens(alpha, k));
      assert.ok(expectedTokens(alpha, k) <= k + 1);
    }
  }
  for (let k = 1; k < 8; k += 1) assert.ok(expectedTokens(0.8, k) > expectedTokens(0.6, k));
});

test('expectedTokens: errors name the argument', () => {
  assert.throws(() => expectedTokens(-0.1, 3), /^RangeError: expectedTokens: alpha must be a number in \[0, 1\]/);
  assert.throws(() => expectedTokens(1.1, 3), RangeError);
  assert.throws(() => expectedTokens(0.5, -1), /k must be an integer ≥ 0/);
  assert.throws(() => expectedTokens(0.5, 1.5), RangeError);
  assert.throws(() => expectedTokens(NaN, 1), RangeError);
});

test('simpleSpeedup: alpha 0.7, c 0.05 for k = 1 / 2 / 3 / 5 / 8, and c 0.1', () => {
  assert.deepEqual([1, 2, 3, 5, 8].map((k) => printed(simpleSpeedup(0.7, k, 0.05), 2)), [1.62, 1.99, 2.2, 2.35, 2.28]);
  assert.equal(printed(simpleSpeedup(0.7, 3, 0.1), 2), 1.95);
  assert.equal(printed(simpleSpeedup(0.7, 5, 0.1), 2), 1.96);
  assert.equal(simpleSpeedup(0.7, 0, 0.05), 1);
  assert.throws(() => simpleSpeedup(0.7, 3, -0.1), /c must be a finite number ≥ 0/);
});

test('the zoomed position: acceptance 0.900, keep chance 0.857, leftover [0, 0.5, 0.5, 0], reject mass 0.10 (three decimals)', () => {
  close(acceptanceRate(P, Q), 0.9);
  assert.equal(printed(acceptanceRate(P, Q), 3), 0.9);
  const v = verifyToken(P, Q, 0);
  assert.equal(printed(v.accept, 3), 0.857);
  v.residual.forEach((x, i) => close(x, [0, 0.5, 0.5, 0][i]));
  close(v.rejectMass, 0.1);
  assert.equal(verifyToken(P, Q, 1).accept, 1);
  assert.equal(verifyToken(P, Q, 2).accept, 1);
  assert.deepEqual(outputDistribution(P, Q).map((x) => printed(x, 3)), [0.6, 0.25, 0.1, 0.05]);
  outputDistribution(P, Q).forEach((x, i) => close(x, P[i]));
});

test('the zoomed position uses the sampling words down on up big (VOCAB indices 3, 4, 12, 13)', () => {
  assert.deepEqual([3, 4, 12, 13].map((i) => VOCAB[i]), ['down', 'on', 'up', 'big']);
});

test('verifyToken: accept is 1 whenever p ≥ q; identical distributions give an all-zero leftover and no reject mass', () => {
  const rand = mulberry32(7);
  for (let trial = 0; trial < 50; trial += 1) {
    const p = randomDistribution(rand, 5);
    const q = randomDistribution(rand, 5);
    p.forEach((pi, i) => {
      if (pi >= q[i]) assert.equal(verifyToken(p, q, i).accept, 1);
    });
  }
  const same = verifyToken(P, P, 0);
  assert.deepEqual(same.residual, [0, 0, 0, 0]);
  assert.equal(same.rejectMass, 0);
  assert.equal(acceptanceRate(P, P), 1);
});

function randomDistribution(rand, n) {
  const raw = Array.from({ length: n }, () => 0.05 + rand());
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map((x) => x / sum);
}

test('outputDistribution equals p for random p, q (seeded): rejection plus the leftover repays exactly what it removed', () => {
  const rand = mulberry32(2026);
  for (let trial = 0; trial < 200; trial += 1) {
    const n = 2 + Math.floor(rand() * 8);
    const p = randomDistribution(rand, n);
    const q = randomDistribution(rand, n);
    outputDistribution(p, q).forEach((x, i) => close(x, p[i], 1e-12));
  }
});

test('outputDistribution handles a drafter that gives some tokens zero probability', () => {
  const p = [0.5, 0.3, 0.2];
  const q = [0.0, 0.6, 0.4];
  outputDistribution(p, q).forEach((x, i) => close(x, p[i]));
  assert.throws(() => verifyToken(p, q, 0), /the drafter never guesses index 0/);
});

test('distribution inputs are validated: length, sum, sign, index', () => {
  assert.throws(() => acceptanceRate([0.5, 0.5], [1]), /p and q must have the same length/);
  assert.throws(() => acceptanceRate([0.5, 0.4], [0.5, 0.5]), /p must sum to 1/);
  assert.throws(() => acceptanceRate([1.5, -0.5], [0.5, 0.5]), /p must be a non-empty array of numbers ≥ 0/);
  assert.throws(() => acceptanceRate([], []), RangeError);
  assert.throws(() => acceptanceRate('ab', [1]), RangeError);
  assert.throws(() => verifyToken(P, Q, 4), /guessIndex must be an integer in \[0, 3\]/);
  assert.throws(() => verifyToken(P, Q, -1), RangeError);
  assert.throws(() => outputDistribution(P, [1]), RangeError);
});

test('batchSpeedup: alpha 0.7, k 3, c 0.05 — plain ms / verify ms / speedup by batch (context 1,024)', () => {
  const rows = [
    [1, 14.66, 14.67, 2.2], [16, 15.76, 15.93, 2.18], [64, 19.29, 19.97, 2.14], [96, 21.64, 27.17, 1.8],
    [128, 23.99, 36.22, 1.53], [160, 26.34, 45.28, 1.36], [211, 30.08, 59.71, 1.19],
  ];
  for (const [batch, plainMs, verifyMs, speedup] of rows) {
    const r = batchSpeedup({ alpha: 0.7, k: 3, c: 0.05, batch, model: M });
    assert.deepEqual([printed(r.plainMs, 2), printed(r.verifyMs, 2), printed(r.speedup, 2)], [plainMs, verifyMs, speedup], `batch ${batch}`);
    assert.equal(r.tokensPerRound, expectedTokens(0.7, 3));
    close(r.roundMs, 3 * 0.05 * r.plainMs + r.verifyMs, 1e-9);
  }
});

test('batchSpeedup: other k — speedup fades faster with a longer guess; k 5 at 211 users is a slow-down (0.91)', () => {
  const at = (k, batch) => printed(batchSpeedup({ alpha: 0.7, k, c: 0.05, batch, model: M }).speedup, 2);
  assert.deepEqual([64, 128, 211].map((b) => at(5, b)), [1.77, 1.17, 0.91]);
  assert.deepEqual([64, 96, 211].map((b) => at(8, b)), [1.27, 0.99, 0.66]);
});

test('batchSpeedup: MTP (k 1, c 0.05) — alpha 0.85 is 1.76 at batch 1, 1.73 at 128, 1.72 at 211; alpha 0.90 is 1.81 at batch 1 and 1.78 at 128 (the storyboard prints "1.81 / 1.78" without naming the batches)', () => {
  const at = (alpha, batch) => printed(batchSpeedup({ alpha, k: 1, c: 0.05, batch, model: M }).speedup, 2);
  assert.equal(printed(expectedTokens(0.85, 1), 2), 1.85);
  assert.equal(printed(expectedTokens(0.9, 1), 2), 1.9);
  assert.deepEqual([1, 128, 211].map((b) => at(0.85, b)), [1.76, 1.73, 1.72]);
  assert.deepEqual([1, 128].map((b) => at(0.9, b)), [1.81, 1.78]);
});

test('batchSpeedup at batch 1 matches simpleSpeedup within 1% while verify is memory-bound', () => {
  for (const [alpha, k, c] of [[0.7, 3, 0.05], [0.5, 2, 0.1], [0.85, 1, 0.05], [0.9, 5, 0.05]]) {
    const r = batchSpeedup({ alpha, k, c, batch: 1, model: M });
    assert.ok(Math.abs(r.speedup / simpleSpeedup(alpha, k, c) - 1) < 0.01, `${alpha}/${k}/${c}`);
  }
});

test('batchSpeedup: k = 0 is plain decoding (speedup 1); errors name the argument; inputs are not mutated', () => {
  assert.equal(batchSpeedup({ alpha: 0.7, k: 0, c: 0.05, batch: 8, model: M }).speedup, 1);
  const frozen = Object.freeze({ ...M });
  const r = batchSpeedup({ alpha: 0.7, k: 3, c: 0.05, batch: 4, model: frozen });
  assert.deepEqual(r, batchSpeedup({ alpha: 0.7, k: 3, c: 0.05, batch: 4, model: frozen }));
  assert.throws(() => batchSpeedup({ alpha: 0.7, k: 3, c: 0.05, batch: 0, model: M }), /batch must be a positive integer/);
  assert.throws(() => batchSpeedup({ alpha: 0.7, k: 3, c: -1, batch: 1, model: M }), /c must be a finite number ≥ 0/);
  assert.throws(() => batchSpeedup({ alpha: 0.7, k: 3, c: 0.05, batch: 1, model: null }), /model must be a step-time model object/);
  assert.throws(() => batchSpeedup({ alpha: 2, k: 3, c: 0.05, batch: 1, model: M }), /alpha/);
});

test('serving-calculator\'s MTP readouts: V4-Pro on GB300, alpha 0.85, k 1, c 0.05 (1.48 / 0.90 / 1.73 / 1.76)', () => {
  const V = { activeParamsPerGpu: 49e9, weightBytesPerGpu: 865e9 / 16, dModel: 7168, actBytesPerElem: 1, peakTflops: 5000, bandwidthTBps: 8, kvBytesPerToken: 4000 };
  const at = (batch, context) => printed(batchSpeedup({ alpha: 0.85, k: 1, c: 0.05, batch, model: { ...V, context } }).speedup, 2);
  assert.equal(at(256, 9216), 1.48);
  assert.equal(at(1889, 9216), 0.9);
  assert.equal(at(419, 139264), 1.73);
  assert.equal(at(58, 1e6), 1.76);
});
