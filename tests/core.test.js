import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mulberry32, randomMatrix, dot, transpose, matmul, softmax, causalMask,
  formatBytes, formatCount, deepFreeze, formatUsd,
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
  assert.equal(formatBytes(1536), '1.54 kB');
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

test('formatCount takes significant digits; 3 stays the default', () => {
  assert.equal(formatCount(116_829_149_760, { digits: 5 }), '116.83B');
  assert.equal(formatCount(116.8e9, { digits: 5 }), '116.8B');
  assert.equal(formatCount(174_604_259_328, { digits: 5 }), '174.6B');
  assert.equal(formatCount(671_026_404_352, { digits: 5 }), '671.03B');
  assert.equal(formatCount(1.6e12), '1.6T');
  assert.equal(formatCount(116_829_149_760), '117B');
  assert.throws(() => formatCount(1, { digits: 0 }), /digits must be an integer 1–21/);
});

test('formatBytes prints decimal kB (lowercase k) with three significant figures (README byte rule)', () => {
  assert.equal(formatBytes(327_680), '328 kB');
  assert.equal(formatBytes(70_272), '70.3 kB');
  assert.equal(formatBytes(4_718_592), '4.72 MB');
  assert.equal(formatBytes(42_949_672_960), '42.9 GB');
  assert.equal(formatBytes(999_600), '1 MB');
});

// ---- Shared prep S3 (Plan 3, ruling P3-R15): formatRatio and formatDuration ----
import { formatRatio, formatDuration, formatInt } from '../math/core.js';

test('formatRatio: 3 significant figures, trailing zeros dropped (X-3)', () => {
  assert.deepEqual([12, 2, 73.94, 3.556, 56.94, 1180.4, 4.333].map(formatRatio), ['12×', '2×', '73.9×', '3.56×', '56.9×', '1,180×', '4.33×']);
  assert.throws(() => formatRatio(0.27), RangeError);
});

test('formatRatio: the X-3 and storyboard cases, 1× allowed, thousands grouped', () => {
  assert.deepEqual([73.9, 1, 2.897, 2.72, 9, 6, 3.14159, 999.6, 12_345, 1_234_567].map(formatRatio),
    ['73.9×', '1×', '2.9×', '2.72×', '9×', '6×', '3.14×', '1,000×', '12,300×', '1,230,000×']);
  assert.equal(formatRatio(40.104 / 0.5428), '73.9×');
  assert.equal(formatRatio(2 / 0.5625), '3.56×');
  assert.equal(formatRatio(520 / 120), '4.33×');
});

test('formatRatio: below 1, non-finite or non-number throws "formatRatio: x must be …"', () => {
  for (const bad of [0.999, 0, -2, Number.NaN, Infinity, '2', undefined, null]) {
    assert.throws(() => formatRatio(bad), /^RangeError: formatRatio: x must be a finite number ≥ 1/);
  }
});

test('formatDuration: 3 significant figures, unit by magnitude', () => {
  assert.deepEqual([2.5e-6, 0.0146, 0.12345, 1.25, 75, 5400, 54 * 86400].map(formatDuration),
    ['2.5 µs', '14.6 ms', '123 ms', '1.25 s', '1.25 min', '1.5 h', '54 days']);
  assert.throws(() => formatDuration(-1), RangeError);
});

test('formatDuration: below 1 in its unit it still prints 3 significant figures', () => {
  assert.equal(formatDuration(5.428e-7), '0.543 µs');
  assert.equal(formatDuration(1.36e-7), '0.136 µs');
  assert.deepEqual([1180, 40.1, 2.5, 13.6, 74.4].map((x) => formatDuration(x * 1e-6)), ['1.18 ms', '40.1 µs', '2.5 µs', '13.6 µs', '74.4 µs']);
});

test('formatDuration: unit edges (µs < 1 ms ≤ ms < 1 s ≤ s < 60 s ≤ min < 60 min ≤ h < 48 h ≤ days)', () => {
  assert.deepEqual([1e-3, 0.999e-3, 1, 0.9994, 60, 59.9, 3600, 3500, 47 * 3600, 48 * 3600, 1e9].map(formatDuration),
    ['1 ms', '999 µs', '1 s', '999 ms', '1 min', '59.9 s', '1 h', '58.3 min', '47 h', '2 days', '11,600 days']);
});

test('formatDuration: rounds first, so a value that rounds up to the next unit prints in it', () => {
  assert.equal(formatDuration(0.0009996), '1 ms');
  assert.equal(formatDuration(0.99996), '1 s');
  assert.equal(formatDuration(59.97), '1 min');
  assert.equal(formatDuration(3599.9), '1 h');
  assert.equal(formatDuration(47.99 * 3600), '2 days');
});

test('formatDuration: zero is "0 s"; negative, non-finite or non-number throws', () => {
  assert.equal(formatDuration(0), '0 s');
  for (const bad of [-1e-9, -1, Number.NaN, Infinity, '1', undefined, null]) {
    assert.throws(() => formatDuration(bad), /^RangeError: formatDuration: seconds must be a finite number ≥ 0/);
  }
});

test('formatInt: exact integer with separators and a real minus (P4-R6)', () => {
  assert.deepEqual([51020.4, 6345, 0, 105, 1e6].map(formatInt), ['51,020', '6,345', '0', '105', '1,000,000']);
  assert.equal(formatInt(-3), '−3');
  assert.throws(() => formatInt(Infinity), RangeError);
  assert.throws(() => formatInt(NaN), RangeError);
});

test('formatInt: rounds half away from zero in magnitude, never prints "-0", rejects non-numbers', () => {
  assert.equal(formatInt(2.5), '3');
  assert.equal(formatInt(-2.5), '−3');
  assert.equal(formatInt(-0.4), '0');
  assert.equal(formatInt(-1234567.2), '−1,234,567');
  for (const bad of ['3', undefined, null, -Infinity]) assert.throws(() => formatInt(bad), /^RangeError: formatInt: n must be a finite number/);
});

test('formatUsd: three significant figures, at least two decimals, separators, real minus (S7 shared-6)', () => {
  assert.equal(formatUsd(0.11907), '$0.119');
  assert.equal(formatUsd(0.2804), '$0.28');
  assert.equal(formatUsd(2.5), '$2.50');
  assert.equal(formatUsd(0.014421), '$0.0144');
  assert.equal(formatUsd(0.0222), '$0.0222');
  assert.equal(formatUsd(87072), '$87,072');
  assert.equal(formatUsd(0), '$0.00');
  assert.equal(formatUsd(-1.5), '−$1.50');
  assert.throws(() => formatUsd(NaN), RangeError);
  assert.throws(() => formatUsd(Infinity), RangeError);
});
