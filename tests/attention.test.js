import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TOY, scores, scaleScores, applyMask, attentionWeights, weightedSum, attentionHead, multiHead, projectQKV } from '../math/attention.js';
import { causalMask, randomMatrix } from '../math/core.js';

const near = (actual, expected, tol = 5e-4) => {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); expected.forEach((e, i) => near(actual[i], e, tol)); return; }
  if (!Number.isFinite(expected)) { assert.equal(actual, expected); return; }
  assert.ok(Math.abs(actual - expected) < tol, `${actual} ≉ ${expected}`);
};
const { A, B } = TOY.heads;

test('scores = Q · Kᵀ', () => {
  assert.deepEqual(scores([[1, 2]], [[3, 4], [0, 1]]), [[11, 2]]);
  assert.deepEqual(scores(A.Q, A.K)[2], [-1, 3, 0.5, -0.75]);
  assert.deepEqual(scores(B.Q, B.K)[3], [-0.25, 0, 2.75, 0]);
  assert.deepEqual(scores(A.Q, A.K), [[1, 0, -1, 0.25], [-0.25, 1.25, -0.75, -1], [-1, 3, 0.5, -0.75], [-0.5, 3, -0.25, -0.75]]);
});

test('scaleScores divides every cell; a divisor ≤ 0 throws', () => {
  assert.deepEqual(scaleScores([[-1, 3, 0.5, -0.75]], 2), [[-0.5, 1.5, 0.25, -0.375]]);
  assert.deepEqual(scaleScores([[1, 2]], 0.5), [[2, 4]]);
  assert.throws(() => scaleScores([[1]], 0), /divisor must be > 0/);
  assert.throws(() => scaleScores([[1]], Number.NaN), /divisor must be > 0/);
});

test('applyMask: true = visible, hidden → −∞; shape mismatch throws', () => {
  assert.deepEqual(applyMask([[1, 2, 3]], [[true, true, false]]), [[1, 2, -Infinity]]);
  assert.deepEqual(applyMask([[1, 2]], [[true, true]]), [[1, 2]]);
  assert.throws(() => applyMask([[1, 2]], [[true]]), RangeError);
});

test('attentionWeights: row-wise softmax; a fully masked row throws', () => {
  near(attentionWeights([[-0.5, 1.5, 0.25, -Infinity]])[0], [0.095, 0.703, 0.202, 0]);
  assert.deepEqual(attentionWeights([[0.5, -Infinity, -Infinity, -Infinity]]), [[1, 0, 0, 0]]);
  assert.throws(() => attentionWeights([[-Infinity, -Infinity]]), RangeError);
});

test('weightedSum = A · V', () => {
  assert.deepEqual(weightedSum([[0.5, 0.5]], [[1, 0], [0, 2]]), [[0.5, 1]]);
  assert.deepEqual(weightedSum([[1, 0, 0, 0]], A.V), [[1, 0, -1, 0]]);
  const w = attentionWeights(applyMask(scaleScores(scores(A.Q, A.K), 2), causalMask(4)));
  near(weightedSum(w, A.V)[2], [-0.106, 1.407, 0.106, 0.804]);
});

test('attentionHead, head A: the weights grid, the hero output, the toy try-this rows', () => {
  const head = attentionHead(A.Q, A.K, A.V);
  near(head.weights, [[1, 0, 0, 0], [0.321, 0.679, 0, 0], [0.095, 0.703, 0.202, 0], [0.114, 0.656, 0.129, 0.101]]);
  near(head.output[2], [-0.106, 1.407, 0.106, 0.804]);
  near(head.output, [[1, 0, -1, 0], [0.321, 1.358, -0.321, 0.679], [-0.106, 1.407, 0.106, 0.804], [0.035, 1.212, 0.015, 0.821]]);
  near(head.masked[2], [-0.5, 1.5, 0.25, -Infinity], 1e-12);
  near(head.exps[2], [0.607, 4.482, 1.284, 0]);
  near(head.expSums[2], 6.372);
  near(head.exps[0], [Math.exp(0.5), 0, 0, 0]);
  near(attentionHead(A.Q, A.K, A.V, { causal: false }).weights[2], [0.086, 0.635, 0.182, 0.097]);
  near(attentionHead(A.Q, A.K, A.V, { divisor: 0.5 }).weights[2], [0.000, 0.993, 0.007, 0]);
  near(attentionHead(A.Q, A.K, A.V, { divisor: 8 }).weights[2], [0.259, 0.428, 0.313, 0]);
  head.weights.forEach((row) => assert.ok(Math.abs(row.reduce((s, x) => s + x, 0) - 1) < 1e-9));
});

test('the causal mask zeroes the future exactly, and the visible weights still sum to 1', () => {
  const { weights } = attentionHead(A.Q, A.K, A.V);
  weights.forEach((row, i) => row.forEach((w, j) => { if (j > i) assert.equal(w, 0); }));
});

test('attentionHead, head B: the previous-token pattern', () => {
  const head = attentionHead(B.Q, B.K, B.V);
  near(head.weights, [[1, 0, 0, 0], [0.798, 0.202, 0, 0], [0.168, 0.664, 0.168, 0], [0.129, 0.146, 0.578, 0.146]]);
  near(head.output, [[0, 1, 0, 0], [0.202, 0.798, 0, 0], [0.664, 0.168, 0, 0.168], [0.146, 0.129, 0.146, 0.578]]);
});

test('multiHead joins head outputs row by row', () => {
  near(multiHead([A, B]).concat[2], [-0.106, 1.407, 0.106, 0.804, 0.664, 0.168, 0, 0.168]);
  assert.deepEqual(multiHead([A, B]).concat[0], [1, 0, -1, 0, 0, 1, 0, 0]);
  assert.deepEqual(multiHead([A]).concat.map((r) => r.length), [4, 4, 4, 4]);
  assert.throws(() => multiHead([]), /at least one head/);
});

test('projectQKV: X · W for each of Q, K, V; pins the seeded preset', () => {
  assert.deepEqual(projectQKV([[1, 0], [0, 1], [1, 1]], { Wq: [[1, 2], [3, 4]], Wk: [[0, 1], [1, 0]], Wv: [[2, 0], [0, 2]] }),
    { Q: [[1, 2], [3, 4], [4, 6]], K: [[0, 1], [1, 0], [1, 1]], V: [[2, 0], [0, 2], [2, 2]] });
  const X = randomMatrix(4, 8, 1);
  const Wq = randomMatrix(8, 4, 2, 0.5);
  near(projectQKV(X, { Wq, Wk: Wq, Wv: Wq }).Q[0][0], -0.287);
  assert.throws(() => projectQKV([[1, 2]], { Wq: [[1]], Wk: [[1]], Wv: [[1]] }), RangeError);
});

test('TOY is frozen all the way down, and no function mutates its inputs', () => {
  assert.ok(Object.isFrozen(TOY.heads.A.Q[2]));
  const S = [[1, 2], [3, 4]];
  scaleScores(S, 2); applyMask(S, [[true, false], [true, true]]);
  assert.deepEqual(S, [[1, 2], [3, 4]]);
});
