import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deepFreeze } from '../math/core.js';
import { budgetShares } from '../math/pipeline.js';

const at4 = (x) => Number(x.toFixed(4));
const pct = (q, digits) => (q.share === null ? 'n/p' : (100 * q.share).toFixed(digits));
const parts = (pairs) => pairs.map(([name, value]) => ({ name, value }));
const shares = (out) => out.parts.map((q) => (q.share === null ? null : at4(q.share)));

test('budgetShares: GLM-5 with post-training unpublished (training-pipeline §11 signature)', () => {
  const out = budgetShares(parts([['pretrain', 27e12], ['mid-train', 1.55e12], ['post-training', null]]));
  assert.equal(out.knownTotal, 2.855e13);
  assert.equal(out.unknownCount, 1);
  assert.deepEqual(shares(out), [0.9457, 0.0543, null]);
  assert.deepEqual(out.parts.map((q) => pct(q, 1)), ['94.6', '5.4', 'n/p']);
  assert.deepEqual(out.parts.map((q) => q.unknown), [false, false, true]);
  assert.deepEqual(out.parts.map((q) => q.name), ['pretrain', 'mid-train', 'post-training']);
  assert.deepEqual(out.parts.map((q) => q.value), [27e12, 1.55e12, null]);
});

test('budgetShares: GLM-5 context stages (midtraining §11, training-pipeline §11)', () => {
  const out = budgetShares(parts([['4K', 27e12], ['32K', 1e12], ['128K', 0.5e12], ['200K', 0.05e12]]));
  assert.equal(at4(out.knownTotal / 1e13), 2.855);
  assert.equal(out.unknownCount, 0);
  assert.deepEqual(shares(out), [0.9457, 0.0350, 0.0175, 0.0018]);
  assert.deepEqual(out.parts.map((q) => pct(q, 2)), ['94.57', '3.50', '1.75', '0.18']);
  assert.equal((100 * 1.55e12 / out.knownTotal).toFixed(2), '5.43');
  assert.deepEqual(out.parts.map((q) => (560 * q.share).toFixed(1)), ['529.6', '19.6', '9.8', '1.0']);
});

test('budgetShares: midtraining zoomed bar and MiniMax-M2 schedule (midtraining reproducer)', () => {
  const zoom = budgetShares(parts([['32K', 1e12], ['128K', 0.5e12], ['200K', 0.05e12]]));
  assert.equal(zoom.knownTotal.toExponential(3), '1.550e+12');
  assert.deepEqual(zoom.parts.map((q) => pct(q, 2)), ['64.52', '32.26', '3.23']);
  assert.deepEqual(zoom.parts.map((q) => pct(q, 1)), ['64.5', '32.3', '3.2']);
  assert.deepEqual(zoom.parts.map((q) => (560 * q.share).toFixed(1)), ['361.3', '180.6', '18.1']);
  const minimax = budgetShares(parts([['constant', 19.9e12], ['decay', 9.3e12]]));
  assert.deepEqual(minimax.parts.map((q) => pct(q, 1)), ['68.2', '31.8']);
});

test('budgetShares: the other training-pipeline presets (Olmo 3, DeepSeek-V4, Nemotron 3 Super, Kimi K3)', () => {
  const olmo = budgetShares(parts([['pretrain', 5.9e12], ['mid-train', 0.1e12], ['post-training', null]]));
  assert.equal(olmo.knownTotal.toExponential(1), '6.0e+12');
  assert.deepEqual(olmo.parts.map((q) => pct(q, 1)), ['98.3', '1.7', 'n/p']);
  const v4 = budgetShares(parts([['pretrain incl. mid-train', 33e12], ['post-training', null]]));
  assert.equal(v4.knownTotal, 33e12);
  assert.deepEqual(v4.parts.map((q) => pct(q, 1)), ['100.0', 'n/p']);
  const nemotron = budgetShares(parts([['pretrain', 25e12], ['post-training', null]]));
  assert.equal(nemotron.knownTotal, 25e12);
  assert.equal(nemotron.parts[0].share, 1);
  const kimi = budgetShares(parts([['pretrain', null], ['mid-train', null], ['post-training', null]]));
  assert.deepEqual({ knownTotal: kimi.knownTotal, unknownCount: kimi.unknownCount }, { knownTotal: 0, unknownCount: 3 });
  assert.deepEqual(shares(kimi), [null, null, null]);
  assert.ok(kimi.parts.every((q) => q.unknown));
});

test('budgetShares: nothing published leaves every share null, even a known zero', () => {
  const out = budgetShares(parts([['a', 0], ['b', null]]));
  assert.equal(out.knownTotal, 0);
  assert.equal(out.unknownCount, 1);
  assert.deepEqual(shares(out), [null, null]);
  assert.deepEqual(out.parts.map((q) => q.unknown), [false, true]);
});

test('budgetShares: known shares sum to 1 whenever knownTotal > 0, and a zero part has share 0', () => {
  for (const values of [[27e12, 1.55e12, null], [1, 2, 3, 4], [5, 0, null, 7], [3e9]]) {
    const out = budgetShares(values.map((value, i) => ({ name: `p${i}`, value })));
    const sum = out.parts.filter((q) => !q.unknown).reduce((s, q) => s + q.share, 0);
    assert.ok(Math.abs(sum - 1) < 1e-12, `${values} sums to ${sum}`);
  }
  assert.equal(budgetShares(parts([['a', 5], ['b', 0]])).parts[1].share, 0);
});

test('budgetShares: unknown parts never change the known shares', () => {
  const known = budgetShares(parts([['a', 27e12], ['b', 1e12], ['c', 0.5e12]]));
  const mixed = budgetShares(parts([['x', null], ['a', 27e12], ['y', null], ['b', 1e12], ['c', 0.5e12], ['z', null]]));
  assert.equal(mixed.knownTotal, known.knownTotal);
  assert.equal(mixed.unknownCount, 3);
  assert.deepEqual(mixed.parts.filter((q) => !q.unknown).map((q) => q.share), known.parts.map((q) => q.share));
});

test('budgetShares: does not mutate its input and returns new part objects', () => {
  const input = deepFreeze(parts([['pretrain', 27e12], ['mid-train', 1.55e12], ['post-training', null]]));
  const before = JSON.stringify(input);
  const out = budgetShares(input);
  assert.equal(JSON.stringify(input), before);
  assert.notEqual(out.parts, input);
  out.parts.forEach((q, i) => assert.notEqual(q, input[i]));
});

test('budgetShares: throws RangeError on an empty list, a negative value or a malformed part', () => {
  assert.throws(() => budgetShares([]), { name: 'RangeError', message: 'budgetShares: parts must be a non-empty array' });
  assert.throws(() => budgetShares(null), /budgetShares: parts must be a non-empty array/);
  assert.throws(() => budgetShares(parts([['a', 1], ['b', -1]])), { name: 'RangeError', message: 'budgetShares: parts[1].value must be null or a finite number ≥ 0, got -1' });
  for (const bad of [Number.NaN, Infinity, '3', undefined]) {
    assert.throws(() => budgetShares([{ name: 'a', value: bad }]), /budgetShares: parts\[0\]\.value must be null or a finite number ≥ 0/);
  }
  assert.throws(() => budgetShares([{ value: 1 }]), { name: 'RangeError', message: 'budgetShares: parts[0].name must be a string' });
  assert.throws(() => budgetShares([null]), /budgetShares: parts\[0\] must be an object/);
});
