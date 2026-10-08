import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  FLOPS_PER_PARAM_TOKEN, trainingFlops, gpuHoursAt, mfuFrom, achievedTflopsPerGpu, wallClockDays, clusterMtbfHours,
  checkpointLossParts, checkpointLoss, bestInterval, trainingStepMs, runCost, runPlan,
} from '../math/scale.js';
import * as scale from '../math/scale.js';

const round = (x, d) => Number(x.toFixed(d));
const sig = (x, s) => Number(x.toPrecision(s));
const close = (a, b, rel = 1e-12) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} ≉ ${b}`);

const LLAMA = trainingFlops({ params: 405e9, tokens: 15.6e12 });
const DSV3 = trainingFlops({ params: 37e9, tokens: 14.8e12 });
const PER_GPU_MTBF = ((54 * 24) / 419) * 16384; // Llama 3.1: 419 interruptions in 54 days on 16,384 GPUs
const BASE = Object.freeze({
  params: 405e9, tokens: 15.6e12, gpus: 16384, peakTflops: 989, mfu: 0.40,
  perGpuMtbfHours: PER_GPU_MTBF, saveH: 30 / 3600, restartH: 3 / 60, dollarsPerGpuHour: 2,
});

// ---- 6ND (scale-reliability §6; imported by scaling-laws and cluster-topology) ----

test('FLOPS_PER_PARAM_TOKEN is 6 (2 forward + 4 backward)', () => {
  assert.equal(FLOPS_PER_PARAM_TOKEN, 6);
});

test('trainingFlops: 6ND for Llama 3.1 405B, DeepSeek-V3 and scaling-laws frame 1', () => {
  assert.equal(sig(LLAMA, 5), 3.7908e25);
  assert.equal(sig(DSV3, 5), 3.2856e24);
  assert.equal(trainingFlops({ params: 405e9, tokens: 15.6e12 }).toExponential(3), '3.791e+25');
  assert.equal(trainingFlops({ params: 49e9, tokens: 33e12 }).toExponential(3), '9.702e+24');
  assert.equal(trainingFlops({ params: 2, tokens: 3 }), 2 * 3 * FLOPS_PER_PARAM_TOKEN);
});

// ---- GPU-hours and MFU ----

test('gpuHoursAt: flops / (peak · 1e12 · mfu) / 3600', () => {
  assert.equal(round(gpuHoursAt({ flops: LLAMA, peakTflops: 989, mfu: 1 }) / 1e6, 3), 10.647);
  assert.equal(round(gpuHoursAt({ flops: LLAMA, peakTflops: 989, mfu: 0.40 }) / 1e6, 2), 26.62);
  assert.equal(round(gpuHoursAt({ flops: LLAMA, peakTflops: 989, mfu: 0.38 }) / 1e6, 2), 28.02);
});

test('mfuFrom: run-average MFU = flops / (gpuHours · 3600 · peak)', () => {
  assert.equal(round(mfuFrom({ flops: LLAMA, gpuHours: 30.84e6, peakTflops: 989 }), 4), 0.3452);
  assert.equal(round(mfuFrom({ flops: DSV3, gpuHours: 2.664e6, peakTflops: 989 }), 4), 0.3464);
  assert.equal(round(mfuFrom({ flops: DSV3, gpuHours: 2.664e6, peakTflops: 1979 }), 4), 0.1731);
});

test('mfuFrom inverts gpuHoursAt', () => {
  for (const mfu of [0.15, 0.3452, 0.4, 0.6, 1]) {
    for (const peakTflops of [989, 1979, 2250, 5000]) {
      close(mfuFrom({ flops: LLAMA, gpuHours: gpuHoursAt({ flops: LLAMA, peakTflops, mfu }), peakTflops }), mfu);
    }
  }
});

test('achievedTflopsPerGpu: Llama 341.4, DeepSeek-V3 342.6 (frame 7: 343 → 34.6% / 17.3%)', () => {
  assert.equal(round(achievedTflopsPerGpu({ flops: LLAMA, gpuHours: 30.84e6 }), 1), 341.4);
  const ds = achievedTflopsPerGpu({ flops: DSV3, gpuHours: 2.664e6 });
  assert.equal(round(ds, 1), 342.6);
  assert.equal(round(ds / 989, 3), 0.346);
  assert.equal(round(ds / 1979, 3), 0.173);
});

test('wallClockDays: gpuHours / gpus / 24 (frames 2, 3, 11)', () => {
  assert.equal(round(wallClockDays({ gpuHours: 10.647e6, gpus: 16384 }), 2), 27.08);
  assert.equal(round(wallClockDays({ gpuHours: 30.84e6, gpus: 16384 }), 2), 78.43);
  assert.equal(round(wallClockDays({ gpuHours: 2.664e6, gpus: 2048 }), 2), 54.2);
  assert.equal(round(wallClockDays({ gpuHours: gpuHoursAt({ flops: LLAMA, peakTflops: 989, mfu: 0.4 }), gpus: 16384 }), 1), 67.7);
});

// ---- failures and checkpoints ----

test('clusterMtbfHours: per-GPU MTBF / GPUs (frame 8: one every 3.09 h, 50,677 h per GPU)', () => {
  assert.equal(round(clusterMtbfHours({ perGpuMtbfHours: 50677, gpus: 16384 }), 3), 3.093);
  assert.equal(round(clusterMtbfHours({ perGpuMtbfHours: 50677, gpus: 100000 }), 3), 0.507);
  assert.equal(Math.round(PER_GPU_MTBF), 50677);
  assert.equal(round(clusterMtbfHours({ perGpuMtbfHours: PER_GPU_MTBF, gpus: 16384 }), 2), 3.09);
});

test('checkpointLoss: save/T + (T/2 + restart)/MTBF', () => {
  const at = (intervalH) => checkpointLoss({ intervalH, saveH: 30 / 3600, restartH: 3 / 60, mtbfH: 3.093 });
  assert.equal(round(at(13.6 / 60), 4), 0.0896);
  assert.equal(round(at(5 / 60), 3), 0.13);
  assert.equal(round(at(1), 3), 0.186);
});

test('checkpointLossParts: frame 9 splits 9.0% into save 3.7% + lost work 3.7% + restarts 1.6%', () => {
  const mtbfH = clusterMtbfHours({ perGpuMtbfHours: PER_GPU_MTBF, gpus: 16384 });
  const intervalH = bestInterval({ saveH: 30 / 3600, mtbfH });
  const parts = checkpointLossParts({ intervalH, saveH: 30 / 3600, restartH: 3 / 60, mtbfH });
  assert.deepEqual([parts.save, parts.lostWork, parts.restart, parts.total].map((x) => round(100 * x, 1)), [3.7, 3.7, 1.6, 9]);
  assert.equal(parts.total, parts.save + parts.lostWork + parts.restart);
  assert.equal(checkpointLoss({ intervalH, saveH: 30 / 3600, restartH: 3 / 60, mtbfH }), parts.total);
});

test('bestInterval: √(2 · save · MTBF)', () => {
  const best = (saveS, mtbfH) => bestInterval({ saveH: saveS / 3600, mtbfH });
  // Storyboard prints 0.2271; √(2 · 30/3600 · 3.093) = 0.227046 (0.227049 at the unrounded MTBF), so 0.2270 at 4 decimals.
  assert.equal(round(best(30, 3.093), 4), 0.227);
  assert.equal(round(best(30, 3.093), 3), 0.227);
  assert.equal(round(best(30, 3.093) * 60, 1), 13.6);
  assert.equal(round(best(30, 0.507), 4), 0.0919);
  assert.equal(round(best(30, 0.507) * 60, 1), 5.5);
  assert.equal(round(best(15, 3.093), 4), 0.1605);
  assert.equal(round(best(15, 3.093) * 60, 1), 9.6);
});

test('checkpointLoss is minimized at bestInterval (scan ±1%)', () => {
  for (const [saveH, mtbfH, restartH] of [[30 / 3600, 3.093, 0.05], [15 / 3600, 0.507, 0.05], [600 / 3600, 20, 0.5]]) {
    const best = bestInterval({ saveH, mtbfH });
    const at = (intervalH) => checkpointLoss({ intervalH, saveH, restartH, mtbfH });
    for (const f of [0.99, 0.995, 0.999, 1.001, 1.005, 1.01]) assert.ok(at(best * f) > at(best), `×${f}`);
  }
});

test('trainingStepMs (P3-R4, not stepTime): overlap ? max : sum (frames 4–5)', () => {
  assert.equal(trainingStepMs({ computeMs: 100, commMs: 40, overlap: false }), 140);
  assert.equal(trainingStepMs({ computeMs: 100, commMs: 40, overlap: true }), 100);
  assert.equal(trainingStepMs({ computeMs: 30, commMs: 40, overlap: true }), 40);
  assert.equal(Math.round((100 * 100) / trainingStepMs({ computeMs: 100, commMs: 40, overlap: false })), 71);
  assert.equal('stepTime' in scale, false);
});

test('runCost: GPU-hours × price (frame 11: DeepSeek-V3 $5.576M)', () => {
  assert.equal(runCost({ gpuHours: 2.788e6, dollarsPerGpuHour: 2 }), 5.576e6);
});

// ---- runPlan ----

test('runPlan reproducer (MTBF h / interval min / loss % / M GPU-h / days / $M)', () => {
  const cases = [
    ['base', {}, [3.093, 13.6, 9.0, 29.24, 74.4, 58.5]],
    ['mfu38', { mfu: 0.38 }, [3.093, 13.6, 9.0, 30.78, 78.3]],
    ['32k', { gpus: 32768 }, [1.547, 9.6, 13.6, 30.81, 39.2]],
    ['100k', { gpus: 100000 }, [0.507, 5.5, 28.0, 36.97, 15.4, 73.9]],
    ['T5', { intervalH: 5 / 60 }, [3.093, 5, 13.0, 30.58]],
    ['T60', { intervalH: 1 }, [3.093, 60, 18.6, 32.71]],
    ['save15', { saveH: 15 / 3600 }, [3.093, 9.6, 6.8, 28.56]],
    ['save15 100k', { saveH: 15 / 3600, gpus: 100000 }, [0.507, 3.9, 22.7, 34.43]],
  ];
  for (const [name, override, want] of cases) {
    const r = runPlan({ ...BASE, ...override });
    const got = [round(r.mtbfH, 3), round(r.intervalH * 60, 1), round(100 * r.loss, 1), round(r.gpuHours / 1e6, 2), round(r.days, 1), round(r.cost / 1e6, 1)];
    assert.deepEqual(got.slice(0, want.length), want, name);
  }
});

test('runPlan: the result fields, and frame 10 (useful 26.62M; lost 2.62M vs 10.35M GPU-h)', () => {
  const r = runPlan(BASE);
  assert.deepEqual(Object.keys(r), ['flops', 'usefulGpuHours', 'mtbfH', 'intervalH', 'loss', 'valid', 'gpuHours', 'days', 'cost']);
  assert.equal(r.flops, LLAMA);
  assert.equal(round(r.usefulGpuHours / 1e6, 2), 26.62);
  assert.equal(round((r.gpuHours - r.usefulGpuHours) / 1e6, 2), 2.62);
  const big = runPlan({ ...BASE, gpus: 100000 });
  assert.equal(round((big.gpuHours - big.usefulGpuHours) / 1e6, 2), 10.35);
  assert.equal(r.valid, true);
});

test('runPlan: the DeepSeek-V3 preset returns 2.664e6 GPU-hours (MFU while training 0.35637)', () => {
  const ds = runPlan({ ...BASE, params: 37e9, tokens: 14.8e12, gpus: 2048, mfu: 0.35637 });
  assert.equal(round(ds.loss, 4), 0.028);
  assert.equal(round(ds.usefulGpuHours / 1e6, 3), 2.589);
  assert.equal(sig(ds.gpuHours, 4), 2.664e6);
  assert.equal(sig(ds.gpuHours, 3), 2.66e6);
  assert.equal(round(ds.days, 1), 54.2);
  assert.equal(ds.valid, true);
  // The preset's 0.35637 is back-solved: run-average 0.3464 (unrounded) ÷ (1 − loss 0.0280 (unrounded)).
  assert.equal(round(mfuFrom({ flops: DSV3, gpuHours: 2.664e6, peakTflops: 989 }) / (1 - ds.loss), 5), 0.35637);
});

test('runPlan: run-average MFU = while-training × (1 − loss) (Llama 0.364)', () => {
  for (const override of [{}, { gpus: 100000 }, { mfu: 0.38, intervalH: 1 }]) {
    const p = { ...BASE, ...override };
    const r = runPlan(p);
    close(mfuFrom({ flops: r.flops, gpuHours: r.gpuHours, peakTflops: p.peakTflops }), p.mfu * (1 - r.loss));
  }
  const r = runPlan(BASE);
  assert.equal(round(mfuFrom({ flops: r.flops, gpuHours: r.gpuHours, peakTflops: 989 }), 3), 0.364);
});

test('runPlan.valid is false exactly when intervalH + restartH > mtbfH / 2', () => {
  assert.equal(runPlan({ ...BASE, gpus: 100000 }).valid, true);
  assert.equal(runPlan({ ...BASE, gpus: 200000, restartH: 0.5 }).valid, false);
  for (const gpus of [256, 16384, 50000, 100000, 150000, 200000]) {
    for (const restartH of [1 / 60, 0.05, 0.25, 0.5]) {
      const r = runPlan({ ...BASE, gpus, restartH });
      assert.equal(r.valid, r.intervalH + restartH <= r.mtbfH / 2, `${gpus} GPUs, restart ${restartH} h`);
    }
  }
});

test('runPlan: with no failures the loss is 0 and gpuHours === usefulGpuHours', () => {
  const r = runPlan({ ...BASE, perGpuMtbfHours: Infinity });
  assert.equal(r.loss, 0);
  assert.equal(r.gpuHours, r.usefulGpuHours);
  assert.equal(r.intervalH, Infinity);
  assert.equal(r.valid, true);
  assert.equal(checkpointLoss({ intervalH: 1, saveH: 30 / 3600, restartH: 0.05, mtbfH: Infinity }), 30 / 3600);
});

test('runPlan: a loss of 100% or more makes GPU-hours, days and cost Infinity', () => {
  const r = runPlan({ ...BASE, gpus: 200000, intervalH: 4, restartH: 0.5 });
  assert.ok(r.loss >= 1);
  assert.equal(r.gpuHours, Infinity);
  assert.equal(r.days, Infinity);
  assert.equal(r.cost, Infinity);
  assert.equal(r.valid, false);
});

// ---- validation and purity ----

test('every bad argument throws "<fn>: <arg> must be …"', () => {
  for (const bad of [0, -1, Number.NaN, Infinity, '1', undefined]) {
    assert.throws(() => trainingFlops({ params: bad, tokens: 1 }), /^RangeError: trainingFlops: params must be a positive finite number/);
    assert.throws(() => gpuHoursAt({ flops: 1, peakTflops: bad, mfu: 0.4 }), /^RangeError: gpuHoursAt: peakTflops must be a positive finite number/);
    assert.throws(() => mfuFrom({ flops: 1, gpuHours: bad, peakTflops: 989 }), /^RangeError: mfuFrom: gpuHours must be a positive finite number/);
    assert.throws(() => achievedTflopsPerGpu({ flops: bad, gpuHours: 1 }), /^RangeError: achievedTflopsPerGpu: flops must be a positive finite number/);
    assert.throws(() => bestInterval({ saveH: bad, mtbfH: 3 }), /^RangeError: bestInterval: saveH must be a positive finite number/);
  }
  for (const bad of [0, -0.1, 1.01, Number.NaN, '0.4']) {
    assert.throws(() => gpuHoursAt({ flops: 1, peakTflops: 989, mfu: bad }), /^RangeError: gpuHoursAt: mfu must be a number in \(0, 1\]/);
    assert.throws(() => runPlan({ ...BASE, mfu: bad }), /^RangeError: gpuHoursAt: mfu must be a number in \(0, 1\]/);
  }
  for (const bad of [0, 1.5, -1, '8', undefined]) {
    assert.throws(() => wallClockDays({ gpuHours: 1, gpus: bad }), /^RangeError: wallClockDays: gpus must be a positive integer/);
    assert.throws(() => clusterMtbfHours({ perGpuMtbfHours: 1, gpus: bad }), /^RangeError: clusterMtbfHours: gpus must be a positive integer/);
  }
  for (const bad of [0, -1, Number.NaN, '1']) {
    assert.throws(() => clusterMtbfHours({ perGpuMtbfHours: bad, gpus: 8 }), /^RangeError: clusterMtbfHours: perGpuMtbfHours must be a positive number/);
    assert.throws(() => checkpointLoss({ intervalH: bad, saveH: 0, restartH: 0, mtbfH: 3 }), /^RangeError: checkpointLossParts: intervalH must be a positive number/);
  }
  for (const bad of [-1, Number.NaN, Infinity, '1']) {
    assert.throws(() => checkpointLoss({ intervalH: 1, saveH: bad, restartH: 0, mtbfH: 3 }), /^RangeError: checkpointLossParts: saveH must be a finite number ≥ 0/);
    assert.throws(() => trainingStepMs({ computeMs: bad, commMs: 1, overlap: true }), /^RangeError: trainingStepMs: computeMs must be a finite number ≥ 0/);
    assert.throws(() => runCost({ gpuHours: 1, dollarsPerGpuHour: bad }), /^RangeError: runCost: dollarsPerGpuHour must be a finite number ≥ 0/);
  }
  assert.throws(() => trainingStepMs({ computeMs: 1, commMs: 1, overlap: 'yes' }), /^RangeError: trainingStepMs: overlap must be true or false/);
  assert.throws(() => runPlan({ ...BASE, intervalH: 0 }), /^RangeError: checkpointLossParts: intervalH must be a positive number/);
});

test('inputs are never mutated', () => {
  const args = Object.freeze({ ...BASE, intervalH: 0.25 });
  const a = runPlan(args), b = runPlan(args);
  assert.deepEqual(a, b);
  assert.notEqual(a, b);
  checkpointLossParts(Object.freeze({ intervalH: 1, saveH: 0.01, restartH: 0.05, mtbfH: 3 }));
  assert.deepEqual(BASE.mfu, 0.4);
});
