import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FORMATS, bitsPerElement } from '../math/roofline.js';
import { weightBytes, freeHbmPerGpu, maxUsersPerGpu, stepTime, RUNNING_EXAMPLE } from '../math/serving.js';
import { E2M1_GRID, roundToGrid, roundE4M3, quantizeBlocks } from '../math/quant.js';

const W = Object.freeze([0.12, -0.31, 0.05, 0.47, -0.08, 0.22, -0.64, 2.1]);
const q = (format, blockSize, values = W) => quantizeBlocks(values, { format, blockSize });
const near = (a, b, eps = 5e-5) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const nearList = (a, b, eps = 5e-5) => { assert.equal(a.length, b.length); a.forEach((x, i) => near(x, b[i], eps)); };

test('E2M1_GRID is the eight magnitudes and is frozen', () => {
  assert.deepEqual([...E2M1_GRID], [0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  assert.ok(Object.isFrozen(E2M1_GRID));
});

test('roundToGrid: nearest magnitude with the sign kept, ties toward zero', () => {
  assert.equal(roundToGrid(2.4, E2M1_GRID), 2);
  assert.equal(roundToGrid(2.6, E2M1_GRID), 3);
  assert.equal(roundToGrid(-2.6, E2M1_GRID), -3);
  assert.equal(roundToGrid(0.25, E2M1_GRID), 0);
  assert.equal(roundToGrid(2.5, E2M1_GRID), 2);
  assert.equal(roundToGrid(-2.5, E2M1_GRID), -2);
  assert.equal(roundToGrid(5, E2M1_GRID), 4);
  assert.equal(roundToGrid(7.52, E2M1_GRID), 6);
  assert.ok(Object.is(roundToGrid(-0.1, E2M1_GRID), 0), 'no negative zero');
  assert.throws(() => roundToGrid(NaN, E2M1_GRID), RangeError);
  assert.throws(() => roundToGrid(1, []), RangeError);
});

test('roundE4M3: three mantissa bits, zero stays zero, below the normal range throws', () => {
  near(roundE4M3(0.47 / 6), 0.078125, 1e-12);
  near(roundE4M3(2.1 / 6), 0.34375, 1e-12);
  assert.equal(roundE4M3(0), 0);
  assert.equal(roundE4M3(1), 1);
  assert.equal(roundE4M3(448), 448);
  assert.equal(roundE4M3(-0.34), -0.34375);
  assert.throws(() => roundE4M3(2 ** -7), RangeError);
  assert.throws(() => roundE4M3(1000), RangeError);
  assert.throws(() => roundE4M3(Infinity), RangeError);
});

test('INT4 worked examples: blocks of 8 and 4', () => {
  const r8 = q('int4', 8);
  nearList([r8.blocks[0].scale], [0.3]);
  assert.deepEqual(r8.blocks[0].codes, [0, -1, 0, 2, 0, 1, -2, 7]);
  nearList(r8.restored, [0, -0.3, 0, 0.6, 0, 0.3, -0.6, 2.1]);
  near(r8.meanAbsErr, 0.0638, 5e-5);
  assert.equal(r8.zeroed, 3);
  const r4 = q('int4', 4);
  nearList(r4.blocks.map((b) => b.scale), [0.0671, 0.3], 5e-5);
  assert.deepEqual(r4.blocks.map((b) => b.codes), [[2, -5, 1, 7], [0, 1, -2, 7]]);
  nearList(r4.restored, [0.1343, -0.3357, 0.0671, 0.47, 0, 0.3, -0.6, 2.1]);
  near(r4.meanAbsErr, 0.0321, 5e-5);
  assert.equal(r4.zeroed, 1);
});

test('MXFP4 worked examples: power-of-two scales, clipping', () => {
  const r8 = q('mxfp4', 8);
  assert.equal(r8.blocks[0].scale, 0.5);
  assert.deepEqual(r8.blocks[0].codes, [0, -0.5, 0, 1, 0, 0.5, -1.5, 4]);
  nearList(r8.restored, [0, -0.25, 0, 0.5, 0, 0.25, -0.75, 2]);
  near(r8.meanAbsErr, 0.0725, 5e-5);
  assert.equal(r8.zeroed, 3);
  assert.equal(r8.clipped, 0);
  const r4 = q('mxfp4', 4);
  assert.deepEqual(r4.blocks.map((b) => b.scale), [0.0625, 0.5]);
  assert.deepEqual(r4.blocks.map((b) => b.codes), [[2, -4, 1, 6], [0, 0.5, -1.5, 4]]);
  nearList(r4.restored, [0.125, -0.25, 0.0625, 0.375, 0, 0.25, -0.75, 2]);
  near(r4.meanAbsErr, 0.0616, 5e-5);
  assert.equal(r4.zeroed, 1);
  assert.equal(r4.clipped, 1);
});

test('NVFP4 worked examples: an 8-bit float scale, nothing clipped', () => {
  const r8 = q('nvfp4', 8);
  near(r8.blocks[0].scale, 0.34375, 1e-12);
  assert.deepEqual(r8.blocks[0].codes, [0.5, -1, 0, 1.5, 0, 0.5, -2, 6]);
  near(r8.meanAbsErr, 0.0493, 5e-5);
  assert.equal(r8.zeroed, 2);
  const r4 = q('nvfp4', 4);
  nearList(r4.blocks.map((b) => b.scale), [0.078125, 0.34375], 1e-12);
  assert.deepEqual(r4.blocks.map((b) => b.codes), [[1.5, -4, 0.5, 6], [0, 0.5, -2, 6]]);
  nearList(r4.restored, [0.1172, -0.3125, 0.0391, 0.4688, 0, 0.1719, -0.6875, 2.0625]);
  near(r4.meanAbsErr, 0.0288, 5e-5);
  assert.equal(r4.zeroed, 1);
  assert.equal(r4.clipped, 0);
});

test('blocks of 2 and the outlier switched off', () => {
  const r = (f) => q(f, 2);
  near(r('int4').meanAbsErr, 0.0105, 5e-5);
  assert.equal(r('int4').zeroed, 0);
  near(r('mxfp4').meanAbsErr, 0.0536, 5e-5);
  assert.equal(r('mxfp4').clipped, 2);
  near(r('nvfp4').meanAbsErr, 0.0175, 5e-5);
  const calm = [...W.slice(0, 7), 0.3];
  near(q('int4', 8, calm).meanAbsErr, 0.0241, 5e-5);
  near(q('mxfp4', 8, calm).meanAbsErr, 0.0394, 5e-5);
  near(q('nvfp4', 8, calm).meanAbsErr, 0.0171, 5e-5);
});

test('the err row is restored minus original, and its mean absolute value is meanAbsErr', () => {
  const r = q('int4', 4);
  nearList(r.err, r.restored.map((x, i) => x - W[i]), 1e-12);
  near(r.meanAbsErr, r.err.reduce((s, e) => s + Math.abs(e), 0) / W.length, 1e-12);
});

test('every FP4 code is in ±E2M1_GRID; INT4 codes are integers in [−7, 7]', () => {
  for (const format of ['mxfp4', 'nvfp4']) {
    for (const blockSize of format === 'nvfp4' ? [8, 4, 2] : [8, 4, 2, 1]) { // nvfp4's scale needs blocks of 2+ (roundE4M3's range)
      for (const b of q(format, blockSize).blocks) b.codes.forEach((c) => assert.ok(E2M1_GRID.includes(Math.abs(c)), `${format} ${blockSize}: ${c}`));
    }
  }
  for (const blockSize of [8, 4, 2, 1]) {
    for (const b of q('int4', blockSize).blocks) b.codes.forEach((c) => assert.ok(Number.isInteger(c) && Math.abs(c) <= 7));
  }
});

test('INT4 and NVFP4 never report clipped; MXFP4 can', () => {
  for (const blockSize of [8, 4, 2, 1]) assert.equal(q('int4', blockSize).clipped, 0);
  for (const blockSize of [8, 4, 2]) assert.equal(q('nvfp4', blockSize).clipped, 0);
  assert.throws(() => q('nvfp4', 1), RangeError); // 0.05 / 6 is below E4M3's normal range
  assert.ok([8, 4, 2, 1].some((b) => q('mxfp4', b).clipped > 0));
});

test('blockSize 1 gives INT4 zero error (each value is its own max)', () => {
  near(q('int4', 1).meanAbsErr, 0, 1e-4);
});

test('a ragged last block is allowed; an all-zero block gets scale 0 and codes 0', () => {
  const r = quantizeBlocks([1, 2, 3], { format: 'int4', blockSize: 2 });
  assert.equal(r.blocks.length, 2);
  assert.equal(r.blocks[1].codes.length, 1);
  for (const format of ['int4', 'mxfp4', 'nvfp4']) {
    const z = quantizeBlocks([0, 0, 0, 0], { format, blockSize: 4 });
    assert.equal(z.blocks[0].scale, 0);
    assert.deepEqual(z.blocks[0].codes, [0, 0, 0, 0]);
    assert.equal(z.meanAbsErr, 0);
    assert.equal(z.zeroed, 0); // nothing non-zero was lost
  }
});

test('bad arguments throw RangeError with the function name', () => {
  assert.throws(() => quantizeBlocks(W, { format: 'fp8', blockSize: 4 }), /quantizeBlocks: format must be/);
  assert.throws(() => quantizeBlocks(W, { format: 'int4', blockSize: 0 }), /quantizeBlocks: blockSize must be/);
  assert.throws(() => quantizeBlocks(W, { format: 'int4', blockSize: 1.5 }), RangeError);
  assert.throws(() => quantizeBlocks([], { format: 'int4', blockSize: 4 }), /quantizeBlocks: values must be/);
  assert.throws(() => quantizeBlocks([1, NaN], { format: 'int4', blockSize: 4 }), RangeError);
  assert.throws(() => quantizeBlocks('abc', { format: 'int4', blockSize: 4 }), RangeError);
});

test('inputs are not mutated and the same input twice gives deepEqual output', () => {
  const frozen = Object.freeze([...W]);
  const a = quantizeBlocks(frozen, { format: 'nvfp4', blockSize: 4 });
  const b = quantizeBlocks(frozen, { format: 'nvfp4', blockSize: 4 });
  assert.deepEqual(a, b);
  assert.deepEqual([...frozen], [...W]);
});

test('bits per weight with scales come from math/roofline.js, not from here', () => {
  assert.equal(bitsPerElement('nvfp4'), 4.5);
  assert.equal(bitsPerElement('mxfp4'), 4.25);
  assert.equal(FORMATS.nvfp4.blockSize, 16);
});

test('model-scale rows on the H200: weight bytes, users at 2,048 tokens, decode and prefill times', () => {
  const H200 = 141e9;
  const kv = (bytes) => bytes * 2048;
  const users = (bits, kvBytes) => maxUsersPerGpu(freeHbmPerGpu({ hbmBytes: H200, weightBytes: weightBytes({ params: 70e9, bitsPerParam: bits }), gpus: 1 }), kv(kvBytes));
  assert.deepEqual([16, 8, 4.5].map((b) => weightBytes({ params: 70e9, bitsPerParam: b })), [140e9, 70e9, 39.375e9]);
  assert.deepEqual([16, 8, 4.5].map((b) => users(b, 327680)), [1, 105, 151]);
  assert.deepEqual([16, 8, 4.5].map((b) => users(b, 163840)), [2, 211, 302]);
  const ms = (x) => Number((x * 1e3).toPrecision(3));
  const step = (bits, actBytesPerElem, peakTflops, tokens, seqs, context) => stepTime({ ...RUNNING_EXAMPLE, weightBytesPerGpu: weightBytes({ params: 70e9, bitsPerParam: bits }), actBytesPerElem, peakTflops, tokens, seqs, context }).timeS;
  assert.deepEqual([[16, 2, 989], [8, 1, 1979], [4.5, 2, 989]].map(([b, a, p]) => ms(step(b, a, p, 1, 1, 2048))), [29.3, 14.7, 8.35]);
  assert.deepEqual([[16, 2, 989], [8, 1, 1979], [4.5, 2, 989]].map(([b, a, p]) => ms(step(b, a, p, 4096, 0, 0))), [580, 290, 580]);
});

test('model-scale rows on the B200 (180 GB usable)', () => {
  const B200 = 180e9;
  const bitsOf = { bf16: 16, fp8_e4m3: 8, nvfp4: 4.5, mxfp4: 4.25 };
  const row = (fmt, kvBytes) => maxUsersPerGpu(freeHbmPerGpu({ hbmBytes: B200, weightBytes: weightBytes({ params: 70e9, bitsPerParam: bitsOf[fmt] }), gpus: 1 }), kvBytes * 2048);
  assert.deepEqual(['bf16', 'fp8_e4m3', 'nvfp4', 'mxfp4'].map((f) => row(f, 327680)), [59, 163, 209, 212]);
  assert.deepEqual(['bf16', 'fp8_e4m3', 'nvfp4', 'mxfp4'].map((f) => row(f, 163840)), [119, 327, 419, 425]);
  const run = { ...RUNNING_EXAMPLE, bandwidthTBps: 8 };
  const ms = (x) => Number((x * 1e3).toPrecision(3));
  const dec = (bits, act, peak) => stepTime({ ...run, weightBytesPerGpu: weightBytes({ params: 70e9, bitsPerParam: bits }), actBytesPerElem: act, peakTflops: peak, tokens: 1, seqs: 1, context: 2048 }).timeS;
  const pre = (bits, act, peak) => stepTime({ ...run, weightBytesPerGpu: weightBytes({ params: 70e9, bitsPerParam: bits }), actBytesPerElem: act, peakTflops: peak, tokens: 4096, seqs: 0, context: 0 }).timeS;
  assert.deepEqual([ms(dec(16, 2, 2250)), ms(dec(8, 1, 4500)), ms(dec(4.5, 1, 9000)), ms(dec(4.25, 1, 9000))], [17.6, 8.84, 5.01, 4.73]);
  assert.deepEqual([ms(pre(16, 2, 2250)), ms(pre(8, 1, 4500)), ms(pre(4.5, 1, 9000)), ms(pre(4.25, 1, 9000))], [255, 127, 63.7, 63.7]);
});

test('V4-Pro size at 16 / 8 / 4.5 bits', () => {
  assert.deepEqual([16, 8, 4.5].map((b) => weightBytes({ params: 1.6e12, bitsPerParam: b })), [3.2e12, 1.6e12, 0.9e12]);
});
