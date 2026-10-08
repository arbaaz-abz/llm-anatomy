import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ringAllReduceBytes, reduceScatterBytes, allGatherBytes, allToAllBytes, gpuCount, pipelineBubble, pipelineSchedule,
} from '../math/parallel.js';
import { ROUTER_TOY, routeTopK } from '../math/moe.js';

const round = (x, d) => Number(x.toFixed(d));
const row = (s) => s.trim().split(/\s+/);
const SCHEDULES = ['gpipe', '1f1b'];

// ---- collectives (parallelism §6; cluster-topology imports ringAllReduceBytes) ----

test('ringAllReduceBytes: 2(N − 1)/N · S', () => {
  assert.equal(ringAllReduceBytes(3152, 2), 3152);
  assert.equal(ringAllReduceBytes(64, 2), 64);
  assert.equal(ringAllReduceBytes(350e9, 64), 689.0625e9);
  assert.equal(round(ringAllReduceBytes(350e9, 64) / 1e9, 2), 689.06);
  assert.equal(ringAllReduceBytes(3152, 64), 6205.5);
});

test('reduceScatterBytes and allGatherBytes: (N − 1)/N · S; a ring all-reduce is one of each', () => {
  assert.equal(allGatherBytes(350e9, 64), 344.53125e9);
  assert.equal(round(allGatherBytes(350e9, 64) / 1e9, 2), 344.53);
  assert.equal(reduceScatterBytes(350e9, 64), 344.53125e9);
  for (const [s, n] of [[3152, 2], [64, 4], [350e9, 64], [1, 3]]) {
    assert.equal(ringAllReduceBytes(s, n), reduceScatterBytes(s, n) + allGatherBytes(s, n));
  }
});

test('allToAllBytes: (N − 1)/N · S, each GPU keeps 1/N (uniform traffic)', () => {
  assert.equal(allToAllBytes(64, 4), 48);
});

test('collectives send nothing at N = 1, and nothing for an empty tensor', () => {
  for (const fn of [ringAllReduceBytes, reduceScatterBytes, allGatherBytes, allToAllBytes]) {
    assert.equal(fn(350e9, 1), 0);
    assert.equal(fn(0, 8), 0);
  }
});

test('cluster-topology frame 3: 4 ring all-reduces of s·b·h BF16 over tp 8 = 352.32 MB per layer per step', () => {
  const activation = 2048 * 1 * 12288 * 2;
  assert.equal(4 * ringAllReduceBytes(activation, 8), 352_321_536);
  assert.equal(round((4 * ringAllReduceBytes(activation, 8)) / 1e6, 2), 352.32);
});

test('parallelism frame 10: ROUTER_TOY top-2 routes on 4 GPUs × 2 experts → 4 of 8 copies cross, loads 2/2/3/1, 64 B', () => {
  const routes = ROUTER_TOY.map((scores) => routeTopK(scores, 2));
  assert.deepEqual(routes, [[1, 4], [3, 5], [2, 5], [0, 6]]);
  const gpuOf = (expert) => Math.floor(expert / 2);
  const crossing = routes.flatMap((experts, token) => experts.filter((e) => gpuOf(e) !== token));
  assert.equal(crossing.length, 4);
  const loads = [0, 1, 2, 3].map((g) => routes.flat().filter((e) => gpuOf(e) === g).length);
  assert.deepEqual(loads, [2, 2, 3, 1]);
  assert.equal(crossing.length * 8 * 2, 64);
});

// ---- gpuCount ----

test('gpuCount: tp · cp · pp · dp (Llama 3.1 presets), each defaulting to 1', () => {
  assert.equal(gpuCount({ tp: 8, cp: 1, pp: 16, dp: 64 }), 8192);
  assert.equal(gpuCount({ tp: 8, pp: 16, dp: 128 }), 16384);
  assert.equal(gpuCount({ tp: 8, cp: 16, pp: 16, dp: 8 }), 16384);
  assert.equal(gpuCount({}), 1);
  assert.equal(gpuCount({ dp: 4 }), 4);
});

// ---- pipelineBubble ----

test('pipelineBubble: (p − 1)/(m + p − 1), interleaving divides the bubble by v', () => {
  const cases = [[3, 4, 0.3333], [4, 4, 0.4286], [4, 8, 0.2727], [4, 16, 0.1579], [4, 32, 0.0857], [8, 32, 0.1795]];
  for (const [stages, microBatches, want] of cases) assert.equal(round(pipelineBubble({ stages, microBatches }), 4), want);
  assert.equal(round(pipelineBubble({ stages: 4, microBatches: 8, virtualStages: 2 }), 4), 0.1579);
  assert.equal(pipelineBubble({ stages: 1, microBatches: 8 }), 0);
});

test('parallelism try-this 1: GPipe, 4 stages, 4 → 8 → 16 → 32 micro-batches; 8 stages × 32', () => {
  const pct = (stages, microBatches) => round(100 * pipelineBubble({ stages, microBatches }), 1);
  assert.deepEqual([4, 8, 16, 32].map((m) => pct(4, m)), [42.9, 27.3, 15.8, 8.6]);
  assert.equal(pct(8, 32), 17.9);
  assert.equal(round(100 * pipelineBubble({ stages: 4, microBatches: 32 }), 2), 8.57);
  assert.equal(round(100 * pipelineBubble({ stages: 8, microBatches: 32 }), 2), 17.95);
});

// ---- pipelineSchedule ----

test('pipelineSchedule: GPipe (3, 4) grid, idle 1/3, peak 4 4 4', () => {
  const r = pipelineSchedule({ schedule: 'gpipe', stages: 3, microBatches: 4 });
  assert.equal(r.columns, 12);
  assert.deepEqual(r.grid, [
    row('F1 F2 F3 F4 .  .  .  .  B1 B2 B3 B4'),
    row('.  F1 F2 F3 F4 .  .  B1 B2 B3 B4 .'),
    row('.  .  F1 F2 F3 F4 B1 B2 B3 B4 .  .'),
  ]);
  assert.equal(round(r.idleFraction, 4), 0.3333);
  assert.deepEqual(r.peakInFlight, [4, 4, 4]);
});

test('pipelineSchedule: 1F1B (3, 4) grid, idle 1/3, peak 3 2 1', () => {
  const r = pipelineSchedule({ schedule: '1f1b', stages: 3, microBatches: 4 });
  assert.equal(r.columns, 12);
  assert.deepEqual(r.grid, [
    row('F1 F2 F3 .  .  B1 F4 B2 .  B3 .  B4'),
    row('.  F1 F2 .  B1 F3 B2 F4 B3 .  B4 .'),
    row('.  .  F1 B1 F2 B2 F3 B3 F4 B4 .  .'),
  ]);
  assert.equal(round(r.idleFraction, 4), 0.3333);
  assert.deepEqual(r.peakInFlight, [3, 2, 1]);
});

test('parallelism reproducer: columns, idle % and peaks for both schedules', () => {
  const rows = [[3, 4, 12, 33.33], [4, 4, 14, 42.86], [3, 8, 20, 20], [4, 8, 22, 27.27], [4, 16, 38, 15.79], [8, 16, 46, 30.43]];
  for (const schedule of SCHEDULES) {
    for (const [stages, microBatches, columns, idle] of rows) {
      const r = pipelineSchedule({ schedule, stages, microBatches });
      assert.equal(r.columns, columns, `${schedule} ${stages}/${microBatches}`);
      assert.equal(round(100 * r.idleFraction, 2), idle, `${schedule} ${stages}/${microBatches}`);
    }
  }
  assert.deepEqual(pipelineSchedule({ schedule: 'gpipe', stages: 3, microBatches: 8 }).peakInFlight, [8, 8, 8]);
  assert.deepEqual(pipelineSchedule({ schedule: '1f1b', stages: 3, microBatches: 8 }).peakInFlight, [3, 2, 1]);
  assert.deepEqual(pipelineSchedule({ schedule: 'gpipe', stages: 4, microBatches: 16 }).peakInFlight, [16, 16, 16, 16]);
  assert.deepEqual(pipelineSchedule({ schedule: '1f1b', stages: 4, microBatches: 16 }).peakInFlight, [4, 3, 2, 1]);
});

test('pipelineSchedule: idle fraction equals pipelineBubble and columns = 2(m + p − 1), p ∈ {2, 3, 4, 8}, m ∈ 1..32', () => {
  for (const schedule of SCHEDULES) {
    for (const stages of [2, 3, 4, 8]) {
      for (let microBatches = 1; microBatches <= 32; microBatches += 1) {
        const r = pipelineSchedule({ schedule, stages, microBatches });
        assert.equal(r.columns, 2 * (microBatches + stages - 1));
        assert.ok(Math.abs(r.idleFraction - pipelineBubble({ stages, microBatches })) < 1e-12, `${schedule} ${stages}/${microBatches}`);
      }
    }
  }
});

test('pipelineSchedule: every micro-batch runs exactly once as F and once as B on every stage, in dependency order', () => {
  for (const schedule of SCHEDULES) {
    for (const [stages, microBatches] of [[2, 1], [3, 4], [4, 3], [4, 16], [8, 5]]) {
      const { grid } = pipelineSchedule({ schedule, stages, microBatches });
      const at = (s, op) => grid[s].indexOf(op);
      for (let s = 0; s < stages; s += 1) {
        const ops = grid[s].filter((c) => c !== '.').sort();
        const want = Array.from({ length: microBatches }, (_, j) => [`B${j + 1}`, `F${j + 1}`]).flat().sort();
        assert.deepEqual(ops, want);
        for (let j = 1; j <= microBatches; j += 1) {
          if (s > 0) assert.ok(at(s, `F${j}`) > at(s - 1, `F${j}`));
          assert.ok(at(s, `B${j}`) > (s === stages - 1 ? at(s, `F${j}`) : at(s + 1, `B${j}`)));
        }
      }
    }
  }
});

test('pipelineSchedule: 1F1B peakInFlight[0] = min(p, m); GPipe = m', () => {
  for (const stages of [2, 3, 4, 8]) {
    for (const microBatches of [1, 2, 3, 4, 7, 8, 16, 32]) {
      assert.equal(pipelineSchedule({ schedule: '1f1b', stages, microBatches }).peakInFlight[0], Math.min(stages, microBatches));
      assert.equal(pipelineSchedule({ schedule: 'gpipe', stages, microBatches }).peakInFlight[0], microBatches);
    }
  }
});

test('pipelineSchedule: one stage never idles', () => {
  const r = pipelineSchedule({ schedule: '1f1b', stages: 1, microBatches: 3 });
  assert.deepEqual(r.grid, [row('F1 B1 F2 B2 F3 B3')]);
  assert.equal(r.idleFraction, 0);
});

// ---- validation and purity ----

test('every bad argument throws "<fn>: <arg> must be …"', () => {
  for (const bad of [-1, Number.NaN, Infinity, '8', undefined]) {
    assert.throws(() => ringAllReduceBytes(bad, 8), /^RangeError: ringAllReduceBytes: sizeBytes must be a finite number ≥ 0/);
    assert.throws(() => allToAllBytes(bad, 8), /^RangeError: allToAllBytes: sizeBytes must be a finite number ≥ 0/);
  }
  for (const bad of [0, 1.5, -2, '8', undefined]) {
    assert.throws(() => ringAllReduceBytes(64, bad), /^RangeError: ringAllReduceBytes: ranks must be a positive integer/);
    assert.throws(() => reduceScatterBytes(64, bad), /^RangeError: reduceScatterBytes: ranks must be a positive integer/);
    assert.throws(() => allGatherBytes(64, bad), /^RangeError: allGatherBytes: ranks must be a positive integer/);
    assert.throws(() => pipelineBubble({ stages: bad, microBatches: 4 }), /^RangeError: pipelineBubble: stages must be a positive integer/);
    assert.throws(() => pipelineSchedule({ schedule: 'gpipe', stages: 3, microBatches: bad }), /^RangeError: pipelineSchedule: microBatches must be a positive integer/);
  }
  for (const bad of [0, 1.5, -2, '8', null]) {
    assert.throws(() => gpuCount({ tp: bad }), /^RangeError: gpuCount: tp must be a positive integer/);
    assert.throws(() => pipelineBubble({ stages: 4, microBatches: 4, virtualStages: bad }), /^RangeError: pipelineBubble: virtualStages must be a positive integer/);
  }
  for (const bad of ['GPipe', 'zero-bubble', undefined]) {
    assert.throws(() => pipelineSchedule({ schedule: bad, stages: 3, microBatches: 4 }), /^RangeError: pipelineSchedule: schedule must be 'gpipe' or '1f1b'/);
  }
});

test('inputs are never mutated; each call returns fresh rows', () => {
  const args = Object.freeze({ schedule: '1f1b', stages: 3, microBatches: 4 });
  const a = pipelineSchedule(args), b = pipelineSchedule(args);
  assert.notEqual(a.grid, b.grid);
  assert.notEqual(a.grid[0], b.grid[0]);
  gpuCount(Object.freeze({ tp: 8, pp: 16 }));
  pipelineBubble(Object.freeze({ stages: 4, microBatches: 8 }));
});
