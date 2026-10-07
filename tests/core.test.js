import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mulberry32, randomMatrix, dot, transpose, matmul, softmax, causalMask,
  formatBytes, formatCount, deepFreeze,
} from '../math/core.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

test('mulberry32 is deterministic and in [0,1)', () => {
  const a = mulberry32(42), b = mulberry32(42);
  const xs = Array.from({ length: 5 }, a), ys = Array.from({ length: 5 }, b);
  assert.deepEqual(xs, ys);
  xs.forEach((x) => assert.ok(x >= 0 && x < 1));
});

test('randomMatrix has the requested shape and range', () => {
  const m = randomMatrix(3, 4, 7, 0.5);
  assert.equal(m.length, 3);
  m.forEach((row) => { assert.equal(row.length, 4); row.forEach((v) => assert.ok(v >= -0.5 && v < 0.5)); });
  assert.deepEqual(m, randomMatrix(3, 4, 7, 0.5));
});

test('dot and length mismatch', () => {
  assert.equal(dot([1, 2, 3], [4, 5, 6]), 32);
  assert.throws(() => dot([1], [1, 2]), RangeError);
});

test('transpose and matmul', () => {
  assert.deepEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
  assert.deepEqual(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]]);
  assert.throws(() => matmul([[1, 2]], [[1, 2]]), /matmul/);
});

test('softmax sums to 1, is stable for large logits, respects temperature', () => {
  const p = softmax([1, 2, 3]);
  close(p.reduce((s, x) => s + x, 0), 1);
  const big = softmax([1000, 1001]);
  close(big[1] / big[0], Math.E, 1e-6);
  const base = softmax([1, 2])[1];
  const sharp = softmax([1, 2], 0.1), flat = softmax([1, 2], 10);
  assert.ok(sharp[1] > base && flat[1] < base && flat[1] < 0.53);
});

test('softmax with masked entries and invalid input', () => {
  const p = softmax([0, -Infinity, 0]);
  assert.deepEqual(p.map((x) => Number(x.toFixed(6))), [0.5, 0, 0.5]);
  assert.throws(() => softmax([-Infinity, -Infinity]), /masked/);
  assert.throws(() => softmax([1, 2], 0), /temperature/);
});

test('softmax handles real vocabulary sizes without overflowing the stack', () => {
  const logits = Array.from({ length: 152_064 }, (_, i) => Math.sin(i));
  close(softmax(logits).reduce((s, x) => s + x, 0), 1);
});

test('softmax rejects empty, NaN and +Infinity logits', () => {
  assert.throws(() => softmax([]), (e) => e instanceof RangeError && /empty/.test(e.message));
  assert.throws(() => softmax([1, NaN]), (e) => e instanceof RangeError && /NaN/.test(e.message));
  assert.throws(() => softmax([1, Infinity]), (e) => e instanceof RangeError && /Infinity/.test(e.message));
});

test('causalMask lets query i see keys j ≤ i', () => {
  assert.deepEqual(causalMask(3), [
    [true, false, false],
    [true, true, false],
    [true, true, true],
  ]);
});

test('formatBytes uses decimal by default and binary on request', () => {
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(1536), '1.54 KB');
  assert.equal(formatBytes(80e9), '80 GB');
  assert.equal(formatBytes(999_600), '1 MB');
  assert.equal(formatBytes(4.5 * 1024 ** 2, { binary: true }), '4.5 MiB');
});

test('formatCount', () => {
  assert.equal(formatCount(950), '950');
  assert.equal(formatCount(49e9), '49B');
  assert.equal(formatCount(1.05e12), '1.05T');
  assert.equal(formatCount(675e6), '675M');
});

test('formatCount rolls over to the next unit when rounding reaches 1000', () => {
  assert.equal(formatCount(999_600), '1M');
  assert.equal(formatCount(999.7e9), '1T');
  assert.equal(formatCount(999.6), '1K');
});

test('deepFreeze returns a frozen deep copy: arrays stay arrays, nested values frozen, the input untouched', () => {
  const input = { a: [1, [2, 3]], b: { c: 4 }, d: null, e: 'x' };
  const out = deepFreeze(input);
  assert.deepEqual(out, input);
  assert.ok(Array.isArray(out.a) && Array.isArray(out.a[1]));
  assert.ok([out, out.a, out.a[1], out.b].every(Object.isFrozen));
  assert.ok(!Object.isFrozen(input) && !Object.isFrozen(input.a));
  assert.equal(deepFreeze(7), 7);
});
