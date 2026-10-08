import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rmsNorm, layerNorm, rms, meanAndSd } from '../math/norms.js';
import { softmax } from '../math/core.js';
import { TOY, scores, scaleScores } from '../math/attention.js';

// decoder-recap §6: x_sat is decoder-anatomy's row "sat"; q_sat is attention's head-A query for "sat".
const X_SAT = Object.freeze([0, 1, 0.5, 0, -0.5, 1, 0, 0.5]);
const Q_SAT = Object.freeze([0, 2, 0.5, 0]);
const near = (actual, expected, tol = 5e-4) => {
  assert.equal(actual.length, expected.length);
  expected.forEach((e, i) => assert.ok(Math.abs(actual[i] - e) < tol, `[${i}] ${actual[i]} ≉ ${e}`));
};
const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;

test('rmsNorm divides by the root mean square and keeps zeros at zero (frame 2)', () => {
  near(rmsNorm(X_SAT), [0, 1.706, 0.853, 0, -0.853, 1.706, 0, 0.853]);
  near([rms(X_SAT)], [0.586]);
  assert.equal(rmsNorm(X_SAT)[0], 0);
  assert.equal(rmsNorm(X_SAT)[3], 0);
});

test('rmsNorm on frame 7\'s query: rms 1.031', () => {
  near(rmsNorm(Q_SAT), [0, 1.940, 0.485, 0]);
  near([rms(Q_SAT)], [1.031]);
});

test('rmsNorm output has root mean square 1 (γ = 1), whatever the scale', () => {
  [X_SAT, Q_SAT, [3, -4], [0.001, 0.002, -0.003]].forEach((row) => {
    near([rms(rmsNorm(row))], [1], 1e-12);
    near([rms(rmsNorm(row.map((v) => v * 10)))], [1], 1e-12);
  });
  near(rmsNorm(Q_SAT.map((v) => v * 10)), rmsNorm(Q_SAT), 1e-12);
});

test('layerNorm gives the storyboard row: mean 0.3125, sd 0.496', () => {
  near(layerNorm(X_SAT), [-0.630, 1.386, 0.378, -0.630, -1.638, 1.386, -0.630, 0.378]);
  const { mean: mu, sd } = meanAndSd(X_SAT);
  near([mu, sd], [0.3125, 0.496]);
});

test('layerNorm output has mean 0 and variance 1', () => {
  [X_SAT, Q_SAT, [5, -1, 2]].forEach((row) => {
    const out = layerNorm(row);
    near([mean(out)], [0], 1e-12);
    near([mean(out.map((v) => v * v))], [1], 1e-12);
  });
});

test('γ scales and β shifts, as a number or per element', () => {
  assert.deepEqual(rmsNorm([3, 4], { gamma: 2 }).map((v) => Number(v.toFixed(6))), [Number((2 * 3 / Math.sqrt(12.5)).toFixed(6)), Number((2 * 4 / Math.sqrt(12.5)).toFixed(6))]);
  near(rmsNorm([3, 4], { gamma: [1, 0] }), [3 / Math.sqrt(12.5), 0]);
  near(layerNorm([1, 3], { gamma: 2, beta: 5 }), [3, 7]);
  near(layerNorm([1, 3], { gamma: [1, 2], beta: [0, 1] }), [-1, 3]);
});

test('QK-norm: the scores stay in range and the weights stay put when q grows 10× (frame 7)', () => {
  const K = TOY.heads.A.K.slice(0, 3); // The, cat, sat
  const weightsFor = (q, k) => softmax(scaleScores(scores([q], k), 2)[0]);
  near(weightsFor(Q_SAT, K), [0.095, 0.703, 0.202]);
  near(scaleScores(scores([Q_SAT], K), 2)[0], [-0.5, 1.5, 0.25]);
  near(scaleScores(scores([Q_SAT.map((v) => v * 10)], K), 2)[0], [-5, 15, 2.5]);
  near(weightsFor(Q_SAT.map((v) => v * 10), K), [0, 1, 0]);
  const kn = K.map((k) => rmsNorm(k));
  near(K.map((k) => rms(k)), [0.612, 0.791, 0.612]);
  near(scaleScores(scores([rmsNorm(Q_SAT)], kn), 2)[0], [-0.792, 1.841, 0.396]);
  near(weightsFor(rmsNorm(Q_SAT), kn), [0.055, 0.765, 0.180]);
  near(weightsFor(rmsNorm(Q_SAT.map((v) => v * 10)), kn), weightsFor(rmsNorm(Q_SAT), kn), 1e-12);
  const bound = scores([rmsNorm(Q_SAT)], kn)[0].map(Math.abs); // |s| ≤ d_head with γ = 1
  assert.ok(bound.every((s) => s <= 4 + 1e-12));
});

test('bad input throws a RangeError that names the function', () => {
  assert.throws(() => rmsNorm([]), /rmsNorm: row must be a non-empty array/);
  assert.throws(() => rmsNorm('abc'), /rmsNorm: row must be a non-empty array/);
  assert.throws(() => rmsNorm([1, Number.NaN]), /rmsNorm: row\[1\] must be a finite number/);
  assert.throws(() => rmsNorm([0, 0, 0]), /rmsNorm: row is all zeros/);
  assert.throws(() => rmsNorm([1, 2], { gamma: [1] }), /rmsNorm: gamma has 1 entries/);
  assert.throws(() => rmsNorm([1, 2], { gamma: Number.NaN }), /rmsNorm: gamma must be a finite number/);
  assert.throws(() => layerNorm([2, 2, 2]), /layerNorm: row has no spread/);
  assert.throws(() => layerNorm([]), /layerNorm: row must be a non-empty array/);
  assert.throws(() => layerNorm([1, 2], { beta: [1, 2, 3] }), /layerNorm: beta has 3 entries/);
  assert.throws(() => rms([]), /rms: row must be a non-empty array/);
});

test('inputs are not mutated', () => {
  const row = Object.freeze([...X_SAT]);
  const gamma = Object.freeze([1, 1, 1, 1, 1, 1, 1, 1]);
  assert.doesNotThrow(() => { rmsNorm(row, { gamma }); layerNorm(row, { gamma, beta: gamma }); meanAndSd(row); });
  assert.deepEqual([...row], [...X_SAT]);
});
