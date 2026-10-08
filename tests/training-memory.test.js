import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  TRAINING_RECIPES, trainingBytesPerParam, zeroPerGpuBytes, activationBytesPerLayer, activationBytes, gpusToHoldStates,
} from '../math/training-memory.js';

const G = 1e9;
const gb = (bytes, d = 2) => Number((bytes / G).toFixed(d));
const close = (a, b, rel = 1e-12) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} ≉ ${b}`);
const A = TRAINING_RECIPES.adam;
const GPT3 = { seq: 2048, microBatch: 1, hidden: 12288, heads: 96 };

// ---- TRAINING_RECIPES and trainingBytesPerParam (training-memory §6) ----

test('TRAINING_RECIPES: bytes per parameter by part', () => {
  assert.deepEqual(Object.keys(TRAINING_RECIPES), ['adam', 'adamBf16Moments', 'muon']);
  assert.deepEqual(TRAINING_RECIPES.adam, { weight: 2, grad: 2, master: 4, optimizer: 8 });
  assert.deepEqual(TRAINING_RECIPES.adamBf16Moments, { weight: 2, grad: 2, master: 4, optimizer: 4 });
  assert.deepEqual(TRAINING_RECIPES.muon, { weight: 2, grad: 2, master: 4, optimizer: 4 });
});

test('TRAINING_RECIPES is deeply frozen', () => {
  assert.ok(Object.isFrozen(TRAINING_RECIPES));
  for (const r of Object.values(TRAINING_RECIPES)) assert.ok(Object.isFrozen(r));
  assert.throws(() => { TRAINING_RECIPES.adam.optimizer = 4; }, TypeError);
});

test('trainingBytesPerParam: adam 16 · adamBf16Moments 12 · muon 12 · Kimi K2 resident + FP32 grad 6', () => {
  assert.deepEqual(trainingBytesPerParam(A), { weight: 2, grad: 2, master: 4, optimizer: 8, total: 16 });
  assert.equal(trainingBytesPerParam(TRAINING_RECIPES.adamBf16Moments).total, 12);
  assert.equal(trainingBytesPerParam(TRAINING_RECIPES.muon).total, 12);
  assert.deepEqual(trainingBytesPerParam({ weight: 2, grad: 4, master: 0, optimizer: 0 }), { weight: 2, grad: 4, master: 0, optimizer: 0, total: 6 });
});

// ---- zeroPerGpuBytes (training-memory §6; parallelism §6 with modelShards = tp · pp) ----

test('zeroPerGpuBytes: ZeRO paper 7.5B, Adam, dp 64 (weights / grads / optimizer / total GB)', () => {
  const rows = [[0, 15, 15, 90, 120], [1, 15, 15, 1.41, 31.41], [2, 15, 0.23, 1.41, 16.64], [3, 0.23, 0.23, 1.41, 1.88]];
  for (const [stage, w, g, o, t] of rows) {
    const z = zeroPerGpuBytes({ params: 7.5e9, recipe: A, stage, dp: 64 });
    assert.deepEqual([gb(z.weights), gb(z.grads), gb(z.optimizer), gb(z.total)], [w, g, o, t], `stage ${stage}`);
  }
  assert.deepEqual([0, 1, 2, 3].map((stage) => zeroPerGpuBytes({ params: 7.5e9, recipe: A, stage, dp: 64 }).total),
    [120e9, 31.40625e9, 16.640625e9, 1.875e9]);
});

test('zeroPerGpuBytes: GPT-3 175B, Adam, dp 64', () => {
  const rows = [[0, 350, 350, 2100, 2800], [1, 350, 350, 32.81, 732.81], [2, 350, 5.47, 32.81, 388.28], [3, 5.47, 5.47, 32.81, 43.75]];
  for (const [stage, w, g, o, t] of rows) {
    const z = zeroPerGpuBytes({ params: 175e9, recipe: A, stage, dp: 64 });
    assert.deepEqual([gb(z.weights), gb(z.grads), gb(z.optimizer), gb(z.total)], [w, g, o, t], `stage ${stage}`);
  }
});

test('zeroPerGpuBytes: parallelism 405B (Llama 3.1 "8K GPUs": tp 8, pp 16, dp 64) → ZeRO-2 7.02 GB', () => {
  const at = (tp, pp, dp, stage) => gb(zeroPerGpuBytes({ params: 405e9, recipe: TRAINING_RECIPES.adam, stage, dp, modelShards: tp * pp }).total);
  assert.deepEqual([0, 1, 2, 3].map((s) => at(8, 16, 64, s)), [50.63, 13.25, 7.02, 0.79]);
  assert.equal(at(1, 1, 8192, 3), 0.79);
  assert.equal(at(1, 1, 1, 0), 6480);
  close(zeroPerGpuBytes({ params: 405e9, recipe: A, stage: 2, dp: 64, modelShards: 128 }).total, (405e9 * (2 + 2 / 64 + 12 / 64)) / 128);
});

test('zeroPerGpuBytes: a custom recipe over a model-parallel group (Kimi K2: 6 B over 256 GPUs)', () => {
  const k2 = { weight: 2, grad: 4, master: 0, optimizer: 0 };
  assert.equal(zeroPerGpuBytes({ params: 1e12, recipe: k2, stage: 0, dp: 1, modelShards: 256 }).total, 23.4375e9);
  assert.equal(gb(zeroPerGpuBytes({ params: 1.04e12, recipe: k2, stage: 0, dp: 1, modelShards: 256 }).total), 24.38);
});

test('zeroPerGpuBytes: at dp = 1 every stage equals stage 0', () => {
  for (const recipe of Object.values(TRAINING_RECIPES)) {
    const base = zeroPerGpuBytes({ params: 7.5e9, recipe, stage: 0, dp: 1 });
    for (const stage of [1, 2, 3]) assert.deepEqual(zeroPerGpuBytes({ params: 7.5e9, recipe, stage, dp: 1 }), base);
  }
});

test('zeroPerGpuBytes: total is the sum of the parts, and non-increasing in stage', () => {
  for (const recipe of Object.values(TRAINING_RECIPES)) {
    for (const dp of [1, 2, 8, 64, 1024]) {
      for (const modelShards of [1, 8, 128]) {
        const totals = [0, 1, 2, 3].map((stage) => {
          const z = zeroPerGpuBytes({ params: 175e9, recipe, stage, dp, modelShards });
          close(z.total, z.weights + z.grads + z.optimizer);
          return z.total;
        });
        totals.slice(1).forEach((t, i) => assert.ok(t <= totals[i], `dp ${dp} stage ${i + 1}`));
      }
    }
  }
});

test('zeroPerGpuBytes: optimizer is master + moments; modelShards divides everything first', () => {
  const z = zeroPerGpuBytes({ params: 1e9, recipe: A, stage: 0, dp: 4, modelShards: 2 });
  assert.deepEqual(z, { weights: 1e9, grads: 1e9, optimizer: 6e9, total: 8e9 });
});

// ---- activations (training-memory §6, 03 §3.2) ----

test('activationBytesPerLayer: GPT-3 (2048, 1, 12288, 96): none · selective · full', () => {
  assert.equal(activationBytesPerLayer({ ...GPT3 }), 2_868_903_936);
  assert.equal(activationBytesPerLayer({ ...GPT3, recompute: 'none' }), 2_868_903_936);
  assert.equal(activationBytesPerLayer({ ...GPT3, recompute: 'selective' }), 855_638_016);
  assert.equal(activationBytesPerLayer({ ...GPT3, recompute: 'full' }), 50_331_648);
});

test('activationBytesPerLayer: tp divides none and selective; full keeps 2·s·b·h (as specified)', () => {
  assert.equal(activationBytesPerLayer({ ...GPT3, tp: 8 }), 2_868_903_936 / 8);
  assert.equal(activationBytesPerLayer({ ...GPT3, tp: 8, recompute: 'selective' }), 855_638_016 / 8);
  assert.equal(activationBytesPerLayer({ ...GPT3, tp: 8, recompute: 'full' }), 50_331_648);
});

test('activationBytes: GPT-3, 96 layers → none 275.41 GB · selective 82.14 · full 4.83', () => {
  const at = (recompute) => gb(activationBytes({ layers: 96, ...GPT3, recompute }));
  assert.deepEqual(['none', 'selective', 'full'].map(at), [275.41, 82.14, 4.83]);
  assert.equal(activationBytes({ layers: 96, ...GPT3 }), 96 * 2_868_903_936);
});

test('activationBytesPerLayer: full ≤ selective ≤ none', () => {
  for (const seq of [512, 2048, 8192]) {
    for (const tp of [1, 2, 8]) {
      const [n, s, f] = ['none', 'selective', 'full'].map((recompute) => activationBytesPerLayer({ ...GPT3, seq, tp, recompute }));
      assert.ok(f <= s && s <= n, `seq ${seq} tp ${tp}`);
    }
  }
});

test('training-memory reproducer: GPT-3 per-GPU memory = state + activations (ZeRO-3 and ZeRO-0, H100 try-this 2)', () => {
  const state = (stage) => zeroPerGpuBytes({ params: 175e9, recipe: A, stage, dp: 64 }).total;
  const act = (recompute) => activationBytes({ layers: 96, ...GPT3, recompute });
  assert.equal(gb(state(3) + act('full')), 48.58);
  assert.equal(gb(state(3) + act('selective')), 125.89);
  assert.equal(gb(state(3) + act('none')), 319.16);
  assert.equal(gb(state(0) + act('full')), 2804.83);
});

// ---- gpusToHoldStates ----

test('gpusToHoldStates: ceil(params · bytesPerParam / hbmBytes) for H100 80 / H200 141 / B200 180 usable / B300 288', () => {
  const row = (params, bytesPerParam) => [80e9, 141e9, 180e9, 288e9].map((hbmBytes) => gpusToHoldStates({ params, bytesPerParam, hbmBytes }));
  assert.deepEqual(row(1e12, 16), [200, 114, 89, 56]);
  assert.deepEqual(row(1.6e12, 16), [320, 182, 143, 89]);
  assert.deepEqual(row(1.6e12, 12), [240, 137, 107, 67]);
  assert.deepEqual(row(175e9, 16), [35, 20, 16, 10]);
  assert.equal(Number(((100 * 80) / 114).toFixed(1)), 70.2);
});

test('gpusToHoldStates: an exact fit needs no extra GPU', () => {
  assert.equal(gpusToHoldStates({ params: 5e9, bytesPerParam: 16, hbmBytes: 80e9 }), 1);
  assert.equal(gpusToHoldStates({ params: 5e9 + 1, bytesPerParam: 16, hbmBytes: 80e9 }), 2);
});

// ---- validation and purity ----

test('zeroPerGpuBytes throws on a stage outside 0–3 and on bad sizes', () => {
  for (const stage of [-1, 4, 1.5, '2', undefined]) {
    assert.throws(() => zeroPerGpuBytes({ params: 1e9, recipe: A, stage, dp: 8 }), /^RangeError: zeroPerGpuBytes: stage must be 0, 1, 2 or 3/);
  }
  for (const bad of [0, -1, Number.NaN, Infinity, '1']) {
    assert.throws(() => zeroPerGpuBytes({ params: bad, recipe: A, stage: 0, dp: 8 }), /^RangeError: zeroPerGpuBytes: params must be a positive finite number/);
  }
  for (const bad of [0, 1.5, -2, '8']) {
    assert.throws(() => zeroPerGpuBytes({ params: 1e9, recipe: A, stage: 0, dp: bad }), /^RangeError: zeroPerGpuBytes: dp must be a positive integer/);
    assert.throws(() => zeroPerGpuBytes({ params: 1e9, recipe: A, stage: 0, dp: 8, modelShards: bad }), /^RangeError: zeroPerGpuBytes: modelShards must be a positive integer/);
  }
  assert.throws(() => zeroPerGpuBytes({ params: 1e9, recipe: 'adam', stage: 0, dp: 8 }), /^RangeError: zeroPerGpuBytes: recipe must be an object/);
  assert.throws(() => zeroPerGpuBytes({ params: 1e9, recipe: { ...A, grad: -1 }, stage: 0, dp: 8 }), /^RangeError: zeroPerGpuBytes: recipe\.grad must be a finite number ≥ 0/);
});

test('trainingBytesPerParam, activation and GPU-count functions throw "<fn>: <arg> must be …"', () => {
  assert.throws(() => trainingBytesPerParam({ ...A, master: Number.NaN }), /^RangeError: trainingBytesPerParam: master must be a finite number ≥ 0/);
  assert.throws(() => trainingBytesPerParam(null), /^RangeError: trainingBytesPerParam: recipe must be an object/);
  assert.throws(() => activationBytesPerLayer({ ...GPT3, recompute: 'some' }), /^RangeError: activationBytesPerLayer: recompute must be 'none', 'selective' or 'full'/);
  for (const bad of [0, 1.5, -1, '8', undefined]) {
    assert.throws(() => activationBytesPerLayer({ ...GPT3, seq: bad }), /^RangeError: activationBytesPerLayer: seq must be a positive integer/);
    assert.throws(() => activationBytesPerLayer({ ...GPT3, heads: bad }), /^RangeError: activationBytesPerLayer: heads must be a positive integer/);
    assert.throws(() => activationBytes({ ...GPT3, layers: bad }), /^RangeError: activationBytes: layers must be a positive integer/);
  }
  assert.throws(() => activationBytesPerLayer({ ...GPT3, tp: 0 }), /^RangeError: activationBytesPerLayer: tp must be a positive integer/);
  assert.throws(() => gpusToHoldStates({ params: 1e9, bytesPerParam: 16, hbmBytes: 0 }), /^RangeError: gpusToHoldStates: hbmBytes must be a positive finite number/);
  assert.throws(() => gpusToHoldStates({ params: 1e9, bytesPerParam: -1, hbmBytes: 80e9 }), /^RangeError: gpusToHoldStates: bytesPerParam must be a positive finite number/);
});

test('inputs are never mutated; results are fresh objects', () => {
  const recipe = Object.freeze({ weight: 2, grad: 2, master: 4, optimizer: 8 });
  const args = Object.freeze({ params: 7.5e9, recipe, stage: 3, dp: 64 });
  zeroPerGpuBytes(args);
  const parts = trainingBytesPerParam(recipe);
  assert.notEqual(parts, recipe);
  activationBytes(Object.freeze({ layers: 96, ...GPT3 }));
  assert.deepEqual(recipe, { weight: 2, grad: 2, master: 4, optimizer: 8 });
});

test('math/training-memory.js imports nothing from math/memory.js (kvBytesPerToken stays there)', async () => {
  const source = await readFile(new URL('../math/training-memory.js', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /from\s+['"]\.\/memory\.js['"]/);
  const memory = await import('../math/memory.js');
  assert.equal(typeof memory.kvBytesPerToken, 'function');
  assert.equal(typeof memory.kvBytesPerTokenMla, 'function');
});
