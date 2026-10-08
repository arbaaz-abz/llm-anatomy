import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deepFreeze } from '../math/core.js';
import { buildGroup, GROUP_TOY, groupAdvantages, verifyFinalAnswer } from '../math/grpo.js';
import { rolloutSchedule, mismatchRatio, isCorrection } from '../math/agentic.js';

const at4 = (x) => Number(x.toFixed(4));
const near = (a, b, tol = 5e-4) => assert.ok(Math.abs(a - b) <= tol, `${a} ≈ ${b}`);
const DURATIONS = [3, 2, 4, 16, 5, 3, 9, 6]; // agentic-rl §5: episode minutes for rows 1–8
const rowsOf = (flags) => flags.map((c, i) => (c ? i + 1 : 0)).filter(Boolean);

// ---- rolloutSchedule (§11 reproducer) ----
test('rolloutSchedule sync waits for the slowest episode: 16 min, busy 48, idle 80, 37.5% utilization', () => {
  const s = rolloutSchedule(DURATIONS);
  assert.deepEqual([s.iterationTime, s.busy, s.idle], [16, 48, 80]);
  assert.equal(at4(s.utilization), 0.375);
  assert.deepEqual(s.carried, Array(8).fill(false));
});

test('rolloutSchedule partial at λ = 0.75 stops at the 6th-shortest episode and carries rows 4 and 7', () => {
  const s = rolloutSchedule(DURATIONS, { mode: 'partial', lambda: 0.75 });
  assert.deepEqual([s.iterationTime, s.busy, s.idle], [6, 35, 13]);
  assert.equal(at4(s.utilization), 0.7292);
  assert.deepEqual(rowsOf(s.carried), [4, 7]);
});

test('rolloutSchedule partial at λ = 0.5 carries rows 4, 5, 7 and 8', () => {
  const s = rolloutSchedule(DURATIONS, { mode: 'partial', lambda: 0.5 });
  assert.deepEqual([s.iterationTime, s.busy, s.idle], [4, 28, 4]);
  assert.equal(at4(s.utilization), 0.875);
  assert.deepEqual(rowsOf(s.carried), [4, 5, 7, 8]);
});

test('rolloutSchedule with λ = 1 equals sync; equal durations give utilization 1', () => {
  assert.deepEqual(rolloutSchedule(DURATIONS, { mode: 'partial', lambda: 1 }), rolloutSchedule(DURATIONS));
  assert.equal(rolloutSchedule([5, 5, 5]).utilization, 1);
});

test('rolloutSchedule: busy + idle = G · iterationTime, and carried means longer than the iteration', () => {
  for (const [mode, lambda] of [['sync', 1], ['partial', 0.75], ['partial', 0.5], ['partial', 0.25]]) {
    const s = rolloutSchedule(DURATIONS, { mode, lambda });
    near(s.busy + s.idle, DURATIONS.length * s.iterationTime, 1e-12);
    DURATIONS.forEach((d, i) => assert.equal(s.carried[i], d > s.iterationTime, `${mode} ${lambda} row ${i + 1}`));
  }
});

test('rolloutSchedule is not fooled by λ·G float residue (0.7 · 10 = 7.000000000000001)', () => {
  const s = rolloutSchedule([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], { mode: 'partial', lambda: 0.7 });
  assert.equal(s.iterationTime, 7);
});

test('rolloutSchedule never mutates its input and rejects bad arguments', () => {
  const durations = deepFreeze([3, 2, 4]);
  assert.doesNotThrow(() => rolloutSchedule(durations, { mode: 'partial', lambda: 0.5 }));
  assert.throws(() => rolloutSchedule([]), RangeError);
  assert.throws(() => rolloutSchedule([3, 0]), RangeError);
  assert.throws(() => rolloutSchedule([3, -1]), RangeError);
  assert.throws(() => rolloutSchedule([3, NaN]), RangeError);
  assert.throws(() => rolloutSchedule('3,2'), RangeError);
  assert.throws(() => rolloutSchedule([3, 2], { mode: 'partial', lambda: 0 }), RangeError);
  assert.throws(() => rolloutSchedule([3, 2], { mode: 'partial', lambda: 1.01 }), RangeError);
  assert.throws(() => rolloutSchedule([3, 2], { mode: 'eager' }), RangeError);
});

// ---- mismatchRatio ----
test('mismatchRatio: FP16 shrinks log ρ by 8×; BF16 and ρ = 1 are unchanged', () => {
  assert.equal(mismatchRatio(3.2), 3.2);
  near(mismatchRatio(3.2, { precision: 'fp16' }), 1.1565);
  near(mismatchRatio(0.4, { precision: 'fp16' }), 0.8918);
  near(mismatchRatio(2.4, { precision: 'fp16' }), 1.1156);
  near(mismatchRatio(1.1, { precision: 'fp16' }), 1.012, 5e-4);
  near(mismatchRatio(0.9, { precision: 'fp16' }), 0.9869);
  assert.equal(mismatchRatio(1, { precision: 'fp16' }), 1);
  assert.equal(mismatchRatio(1, { precision: 'bf16' }), 1);
});

test('mismatchRatio keeps the side of 1 and rejects bad arguments', () => {
  for (const rho of [0.1, 0.5, 0.99, 1.01, 2, 50]) {
    const shrunk = mismatchRatio(rho, { precision: 'fp16' });
    assert.equal(Math.sign(shrunk - 1), Math.sign(rho - 1));
    assert.ok(Math.abs(Math.log(shrunk)) < Math.abs(Math.log(rho)));
  }
  assert.throws(() => mismatchRatio(0), RangeError);
  assert.throws(() => mismatchRatio(-1), RangeError);
  assert.throws(() => mismatchRatio(Infinity), RangeError);
  assert.throws(() => mismatchRatio(2, { precision: 'fp8' }), RangeError);
});

// ---- isCorrection ----
test('isCorrection: the four treatments of ρ = 3.2', () => {
  assert.deepEqual(isCorrection(3.2, { mode: 'none' }), { weight: 1, masked: false });
  assert.deepEqual(isCorrection(3.2), { weight: 1, masked: false }, 'none is the default');
  assert.deepEqual(isCorrection(3.2, { mode: 'full' }), { weight: 3.2, masked: false });
  assert.deepEqual(isCorrection(3.2, { mode: 'tis' }), { weight: 2, masked: false });
  assert.deepEqual(isCorrection(3.2, { mode: 'icepop' }), { weight: 0, masked: true });
});

test('isCorrection: IcePop masks exactly ρ < ½ or ρ > 2 and keeps the band edges', () => {
  assert.deepEqual(isCorrection(0.4, { mode: 'icepop' }), { weight: 0, masked: true });
  assert.deepEqual(isCorrection(2.4, { mode: 'icepop' }), { weight: 0, masked: true });
  assert.deepEqual(isCorrection(1.1, { mode: 'icepop' }), { weight: 1, masked: false });
  assert.deepEqual(isCorrection(0.5, { mode: 'icepop' }), { weight: 1, masked: false });
  assert.deepEqual(isCorrection(2, { mode: 'icepop' }), { weight: 1, masked: false });
  assert.equal(isCorrection(2.0000001, { mode: 'icepop' }).masked, true);
  assert.equal(isCorrection(0.4999999, { mode: 'icepop' }).masked, true);
});

test('isCorrection: truncated IS caps only from above; the cap and band are options', () => {
  assert.equal(isCorrection(0.4, { mode: 'tis' }).weight, 0.4);
  assert.equal(isCorrection(1.5, { mode: 'tis' }).weight, 1.5);
  assert.equal(isCorrection(5, { mode: 'tis', cap: 4 }).weight, 4);
  assert.deepEqual(isCorrection(3, { mode: 'icepop', band: [0.25, 4] }), { weight: 1, masked: false });
});

test('isCorrection rejects bad arguments', () => {
  assert.throws(() => isCorrection(0), RangeError);
  assert.throws(() => isCorrection(-2, { mode: 'full' }), RangeError);
  assert.throws(() => isCorrection(NaN), RangeError);
  assert.throws(() => isCorrection(2, { mode: 'clip' }), RangeError);
  assert.throws(() => isCorrection(2, { mode: 'tis', cap: 0 }), RangeError);
  assert.throws(() => isCorrection(2, { mode: 'icepop', band: [2, 0.5] }), RangeError);
  assert.throws(() => isCorrection(2, { mode: 'icepop', band: [0.5] }), RangeError);
});

// ---- the §11 reproducer: pushes on the default group (math/grpo.js, unchanged) ----
test('row 4\'s 48 push under each correction: −0.5774, −1.8475, −1.1547, 0', () => {
  const rows = buildGroup(2, GROUP_TOY);
  const A = groupAdvantages(rows.map((t) => verifyFinalAnswer(t, GROUP_TOY.target)));
  near(A[3], -0.5774);
  const push = (mode) => A[3] * isCorrection(3.2, { mode }).weight;
  near(push('none'), -0.5774);
  near(push('full'), -1.8475);
  near(push('tis'), -1.1547);
  assert.equal(Math.abs(push('icepop')), 0, 'a masked token pushes nothing (the page prints a negative zero as 0)');
});
