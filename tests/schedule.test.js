import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deepFreeze } from '../math/core.js';
import { lrAt, attentionCostRatio, lrCurve } from '../math/schedule.js';

const at4 = (x) => Number(x.toFixed(4));
const close = (a, b, rel = 5e-4) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} vs ${b}`);
const NEMOTRON = { kind: 'wsd', peak: 4.5e-4, floor: 4.5e-6, warmup: 0.2, total: 25, decayStart: 20, decayShape: 'minus-sqrt' };
const COSINE = { kind: 'cosine', total: 1 };
const WSD = { kind: 'wsd', total: 1, decayStart: 0.8 };

test('lrAt: cosine (midtraining §6 and §11 reproducer)', () => {
  assert.deepEqual([0.25, 0.5, 0.6, 0.75, 0.9, 1].map((t) => at4(lrAt(t, COSINE))), [0.8536, 0.5, 0.3455, 0.1464, 0.0245, 0]);
  assert.equal(lrAt(0, COSINE), 1);
});

test('lrAt: warmup-stable-decay, linear decay (Check my work and try-this 1)', () => {
  assert.deepEqual([0.25, 0.5, 0.6, 0.75, 0.8].map((t) => lrAt(t, WSD)), [1, 1, 1, 1, 1]);
  close(lrAt(0.9, WSD), 0.5);
  assert.equal(lrAt(1, WSD), 0);
  close(lrAt(0.85, { ...WSD, decayStart: 0.7 }), 0.5);
});

test('lrAt: Nemotron 3 Super preset in tokens (T), minus-sqrt decay to the floor', () => {
  close(lrAt(0.1, NEMOTRON), 2.25e-4);
  close(lrAt(0.2, NEMOTRON), 4.5e-4);
  close(lrAt(20, NEMOTRON), 4.5e-4);
  close(lrAt(21.25, NEMOTRON), 2.272e-4);
  close(lrAt(22.5, NEMOTRON), 4.5e-6 + (4.5e-4 - 4.5e-6) * (1 - Math.SQRT1_2));
  close(lrAt(25, NEMOTRON), 4.5e-6);
});

test('lrAt: decay shapes (linear, minus-sqrt, cosine) all end at the floor and start at the peak', () => {
  ['linear', 'minus-sqrt', 'cosine'].forEach((decayShape) => {
    const o = { ...WSD, floor: 0.1, decayShape };
    assert.equal(lrAt(0.8, o), 1);
    assert.equal(lrAt(1, o), 0.1);
  });
  close(lrAt(0.9, { ...WSD, decayShape: 'cosine' }), 0.5);
  close(lrAt(0.9, { ...WSD, decayShape: 'minus-sqrt' }), 1 - Math.SQRT1_2);
});

test('lrAt: warmup ramps linearly from 0 and is continuous at its end, for both kinds', () => {
  const w = { ...WSD, warmup: 0.1 };
  assert.equal(lrAt(0, w), 0);
  close(lrAt(0.05, w), 0.5);
  assert.equal(lrAt(0.1, w), 1);
  assert.equal(lrAt(0.1 - 1e-12, w) < 1, true);
  close(lrAt(0.1 - 1e-9, w), lrAt(0.1, w), 1e-6);
  const c = { ...COSINE, warmup: 0.1 };
  close(lrAt(0.1 - 1e-9, c), lrAt(0.1, c), 1e-6);
  close(lrAt(0.55, c), 0.5);
});

test('lrAt: continuous at decayStart, constant on the plateau, equal to the floor at total, cosine never rises after warmup', () => {
  const o = { ...WSD, floor: 0.02, warmup: 0.05 };
  close(lrAt(0.8 - 1e-9, o), lrAt(0.8, o), 1e-6);
  assert.deepEqual([0.05, 0.3, 0.79].map((t) => lrAt(t, o)), [1, 1, 1]);
  const c = { kind: 'cosine', total: 10, warmup: 1, floor: 0.1 };
  assert.equal(lrAt(10, c), 0.1);
  let prev = Infinity;
  for (let t = 1; t <= 10; t += 0.25) { const v = lrAt(t, c); assert.ok(v <= prev + 1e-12); prev = v; }
});

test('lrAt: errors say which argument is wrong', () => {
  assert.throws(() => lrAt(-0.1, COSINE), /RangeError|lrAt: t must be/);
  assert.throws(() => lrAt(1.1, COSINE), { name: 'RangeError', message: /lrAt: t must be/ });
  assert.throws(() => lrAt(0.5, { kind: 'step', total: 1 }), { name: 'RangeError', message: /lrAt: kind must be/ });
  assert.throws(() => lrAt(0.5, { kind: 'cosine' }), { name: 'RangeError', message: /lrAt: total must be/ });
  assert.throws(() => lrAt(0.5, { kind: 'wsd', total: 1 }), { name: 'RangeError', message: /lrAt: decayStart must be/ });
  assert.throws(() => lrAt(0.5, { ...WSD, decayStart: 1 }), { name: 'RangeError', message: /lrAt: decayStart must be/ });
  assert.throws(() => lrAt(0.5, { ...WSD, warmup: 0.9 }), { name: 'RangeError', message: /lrAt: decayStart must be/ });
  assert.throws(() => lrAt(0.5, { ...WSD, decayShape: 'step' }), { name: 'RangeError', message: /lrAt: decayShape must be/ });
  assert.throws(() => lrAt(0.5, { ...COSINE, warmup: 1 }), { name: 'RangeError', message: /lrAt: warmup must be/ });
  assert.throws(() => lrAt(0.5, { ...COSINE, floor: 2 }), { name: 'RangeError', message: /lrAt: floor must be/ });
  assert.throws(() => lrAt(0.5, { ...COSINE, peak: 0 }), { name: 'RangeError', message: /lrAt: peak must be/ });
  assert.throws(() => lrAt(Number.NaN, COSINE), { name: 'RangeError', message: /lrAt: t must be/ });
  assert.throws(() => lrAt(0.5), { name: 'RangeError', message: /lrAt: options must be/ });
});

test('lrAt does not mutate its options', () => {
  const o = deepFreeze({ ...WSD, warmup: 0.05 });
  assert.doesNotThrow(() => lrAt(0.9, o));
});

test('lrCurve: samples lrAt at evenly spaced positions from 0 to total', () => {
  const pts = lrCurve({ ...WSD }, 10);
  assert.equal(pts.length, 11);
  assert.deepEqual(pts[0], [0, 1]);
  assert.deepEqual(pts[10], [1, 0]);
  close(pts[9][1], 0.5);
  assert.throws(() => lrCurve(WSD, 0), { name: 'RangeError', message: /lrCurve: samples must be/ });
  assert.throws(() => lrCurve(WSD, 2.5), { name: 'RangeError', message: /lrCurve: samples must be/ });
  assert.deepEqual(lrCurve({ ...WSD, total: 1 }, 4, { from: 0.5, to: 1 }).map((p) => p[0]), [0.5, 0.625, 0.75, 0.875, 1]);
  assert.throws(() => lrCurve(WSD, 4, { from: 0.9, to: 0.5 }), { name: 'RangeError', message: /lrCurve: from and to must be/ });
});

test('attentionCostRatio: GLM-5 stages vs the first (midtraining frame 7)', () => {
  assert.deepEqual([32, 128, 200].map((k) => attentionCostRatio(k, 4)), [8, 32, 50]);
  assert.equal(attentionCostRatio(4, 4), 1);
  assert.equal(attentionCostRatio(1000, 8), 125);
  assert.throws(() => attentionCostRatio(0, 4), { name: 'RangeError', message: /attentionCostRatio: ctx must be/ });
  assert.throws(() => attentionCostRatio(4, -1), { name: 'RangeError', message: /attentionCostRatio: baseCtx must be/ });
  assert.throws(() => attentionCostRatio(4, Number.POSITIVE_INFINITY), { name: 'RangeError', message: /attentionCostRatio: baseCtx must be/ });
});
