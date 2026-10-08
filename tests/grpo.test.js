import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deepFreeze } from '../math/core.js';
import {
  verifyFinalAnswer, groupStats, groupAdvantages, hasSignal, totalPush, tokenWeights, clippedSurrogate, buildGroup, GROUP_TOY,
} from '../math/grpo.js';

const at4 = (x) => Number(x.toFixed(4));
const at2 = (x) => Number(x.toFixed(2));
const near = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) <= tol, `${a} ≈ ${b}`);
const words = (s) => s.split(' ');

// rlvr-grpo §6 "Seeded data", verbatim (tokens are whitespace-split).
const STORYBOARD_CORRECT = ['7 × 8 = 56', '7 + 8 = 56', '56', '8 × 7 = 56', '49 + 7 = 56', '70 − 14 = 56', 'so 56', '7 eights are 56'];
const STORYBOARD_WRONG = ['7 × 8 = 54', '7 + 8 = 15', '7 × 8 = 48 , so 48', '63', '7 × 8 = 58', '8 × 8 = 64', '7 × 8 = 42', '7 × 8 = 65'];
// rlvr-grpo §5 table: the default group, k = 2.
const DEFAULT_ROWS = ['7 × 8 = 56', '7 × 8 = 54', '7 + 8 = 15', '7 × 8 = 48 , so 48', '7 + 8 = 56', '63', '7 × 8 = 58', '8 × 8 = 64'];
const R2 = [1, 0, 0, 0, 1, 0, 0, 0];
const LENGTHS = [5, 5, 5, 8, 5, 1, 5, 5];

const group = (k) => buildGroup(k, GROUP_TOY);
const rewardsOf = (rows) => rows.map((tokens) => verifyFinalAnswer(tokens, GROUP_TOY.target));

test('GROUP_TOY: the pools, slot order and target are rlvr-grpo §6 row for row (ruling P3-R5)', () => {
  assert.deepEqual(GROUP_TOY.correct, STORYBOARD_CORRECT.map(words));
  assert.deepEqual(GROUP_TOY.wrong, STORYBOARD_WRONG.map(words));
  assert.deepEqual(GROUP_TOY.slotOrder, [1, 5, 3, 7, 2, 6, 4, 8]);
  assert.equal(GROUP_TOY.target, '56');
  assert.equal(GROUP_TOY.correct.length + GROUP_TOY.wrong.length, 16);
  assert.ok(GROUP_TOY.correct.every((t) => t.at(-1) === '56'));
  assert.ok(GROUP_TOY.wrong.every((t) => t.at(-1) !== '56'));
});

test('GROUP_TOY: deeply frozen (the ROUTER_TOY / VOCAB precedent)', () => {
  assert.ok(Object.isFrozen(GROUP_TOY));
  for (const key of ['correct', 'wrong', 'slotOrder']) assert.ok(Object.isFrozen(GROUP_TOY[key]), key);
  assert.ok([...GROUP_TOY.correct, ...GROUP_TOY.wrong].every(Object.isFrozen));
});

test('GROUP_TOY: the default group is rlvr-grpo §5, 39 tokens, reused row for row by agentic-rl', () => {
  const rows = group(2);
  assert.deepEqual(rows, DEFAULT_ROWS.map(words));
  assert.deepEqual(rows.map((t) => t.length), LENGTHS);
  assert.equal(rows.flat().length, 39);
  assert.deepEqual(rewardsOf(rows), R2);
  const A = groupAdvantages(rewardsOf(rows));
  assert.equal(rows.filter((_, i) => A[i] > 0).flat().length, 10); // 10 of 39 chips positive (rlvr-grpo frame 5)
  assert.equal(rows.filter((_, i) => A[i] < 0).flat().length, 29);
  // agentic-rl's five mismatch tokens (frames 7–8, §6): rows 1 and 5 `56`, row 4's second `48`, row 6 `63`, row 7 `58`.
  assert.deepEqual([rows[0][4], rows[4][4], rows[3][7], rows[5][0], rows[6][4]], ['56', '56', '48', '63', '58']);
  assert.equal(39 - 5, 34); // "most tokens agree … (34 of 39 here)"
});

test('GROUP_TOY: distillation frame 6 reuses row 2, `7 × 8 = 54`, at A = −0.58 on all 5 tokens', () => {
  const rows = group(2);
  assert.deepEqual(rows[1], ['7', '×', '8', '=', '54']);
  const A = groupAdvantages(rewardsOf(rows));
  assert.equal(at4(A[1]), -0.5774);
  assert.equal(at2(A[1]), -0.58);
});

test('verifyFinalAnswer: rlvr-grpo §11 signature', () => {
  assert.equal(verifyFinalAnswer(['7', '×', '8', '=', '56'], '56'), 1);
  assert.equal(verifyFinalAnswer(['7', '+', '8', '=', '56'], '56'), 1); // wrong working, right final token
  assert.equal(verifyFinalAnswer(['7', '×', '8', '=', '48', ',', 'so', '48'], '56'), 0);
  assert.equal(verifyFinalAnswer(['56', 'so', '48'], '56'), 0); // only the last token counts
});

test('verifyFinalAnswer: throws on bad tokens or target', () => {
  assert.throws(() => verifyFinalAnswer([], '56'), { name: 'RangeError', message: 'verifyFinalAnswer: tokens must be a non-empty array of strings' });
  assert.throws(() => verifyFinalAnswer(['7', 56], '56'), /verifyFinalAnswer: tokens must be a non-empty array of strings/);
  assert.throws(() => verifyFinalAnswer('56', '56'), /verifyFinalAnswer: tokens must be a non-empty array of strings/);
  assert.throws(() => verifyFinalAnswer(['56'], 56), { name: 'RangeError', message: 'verifyFinalAnswer: target must be a string' });
});

test('groupStats: rlvr-grpo §11 signature (population std, divides by G)', () => {
  const a = groupStats(R2);
  assert.deepEqual([a.mean, at4(a.std)], [0.25, 0.4330]);
  assert.equal(at2(a.std), 0.43); // frame 4: std = √(0.25 · 0.75)
  near(a.std, Math.sqrt(0.25 * 0.75));
  const b = groupStats([1, 1, 0]);
  assert.deepEqual([at4(b.mean), at4(b.std)], [0.6667, 0.4714]);
  assert.deepEqual(groupStats([1, 1, 1, 1]), { mean: 1, std: 0 });
});

test('groupStats: an all-equal group has std exactly 0 and its own value as mean', () => {
  assert.deepEqual(groupStats([0.1, 0.1, 0.1]), { mean: 0.1, std: 0 });
  assert.deepEqual(groupStats([0]), { mean: 0, std: 0 });
});

test('groupStats: throws on an empty group or a non-finite reward', () => {
  assert.throws(() => groupStats([]), { name: 'RangeError', message: 'groupStats: rewards must be a non-empty array' });
  assert.throws(() => groupStats([1, Number.NaN]), { name: 'RangeError', message: 'groupStats: rewards[1] must be a finite number, got NaN' });
  assert.throws(() => groupStats([1, '0']), /groupStats: rewards\[1\] must be a finite number/);
});

test('groupAdvantages: rlvr-grpo §11 signature', () => {
  assert.deepEqual(groupAdvantages(R2).map(at4), [1.7321, -0.5774, -0.5774, -0.5774, 1.7321, -0.5774, -0.5774, -0.5774]);
  assert.deepEqual(groupAdvantages([1, 0, 0, 1]), [1, -1, -1, 1]);
  assert.deepEqual(groupAdvantages([1, 1, 0], { normalizeStd: false }).map(at4), [0.3333, 0.3333, -0.6667]);
  assert.deepEqual(groupAdvantages([0, 0, 0]), [0, 0, 0]);
  assert.deepEqual(groupAdvantages(R2).map(at2), [1.73, -0.58, -0.58, -0.58, 1.73, -0.58, -0.58, -0.58]); // §5 table
});

test('groupAdvantages: rlvr-grpo reproducer over k = 0, 1, 2, 4, 8 (std on, and off as Dr.GRPO)', () => {
  const row = (k) => {
    const R = rewardsOf(group(k));
    return { R: R.join(''), A: groupAdvantages(R).map(at4), D: groupAdvantages(R, { normalizeStd: false }).map(at4), signal: hasSignal(R) };
  };
  assert.deepEqual(row(0), { R: '00000000', A: Array(8).fill(0), D: Array(8).fill(0), signal: false });
  const k1 = row(1);
  assert.deepEqual([k1.R, new Set(k1.A), new Set(k1.D)], ['10000000', new Set([2.6458, -0.3780]), new Set([0.875, -0.125])]);
  const k2 = row(2);
  assert.deepEqual([k2.R, new Set(k2.A), new Set(k2.D)], ['10001000', new Set([1.7321, -0.5774]), new Set([0.75, -0.25])]);
  const k4 = row(4);
  assert.deepEqual([k4.R, new Set(k4.A), new Set(k4.D)], ['10101010', new Set([1, -1]), new Set([0.5, -0.5])]);
  assert.deepEqual(row(8), { R: '11111111', A: Array(8).fill(0), D: Array(8).fill(0), signal: false });
});

test('groupAdvantages: rlvr-grpo try-this 1 (a right answer\'s push shrinks as k grows)', () => {
  const right = (k, normalizeStd = true) => at2(groupAdvantages(rewardsOf(group(k)), { normalizeStd })[0]);
  assert.deepEqual([2, 4, 8].map((k) => right(k)), [1.73, 1, 0]);
  assert.deepEqual([2, 4, 8].map((k) => right(k, false)), [0.75, 0.5, 0]);
  assert.equal(right(1), 2.65);
  assert.equal(at2(groupAdvantages(rewardsOf(group(1)))[1]), -0.38);
  assert.equal(groupAdvantages(rewardsOf(group(1)), { normalizeStd: false })[0], 0.875);
});

test('groupAdvantages: sum to zero for every k in 0..8, and all-equal groups give exact zeros', () => {
  for (let k = 0; k <= 8; k += 1) {
    const R = rewardsOf(group(k));
    for (const normalizeStd of [true, false]) near(groupAdvantages(R, { normalizeStd }).reduce((s, a) => s + a, 0), 0);
  }
  for (const R of [[0.1, 0.1, 0.1], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0]]) {
    for (const normalizeStd of [true, false]) assert.ok(groupAdvantages(R, { normalizeStd }).every((a) => Object.is(a, 0)));
  }
});

test('groupAdvantages: the shared prefix `7 × 8 =` nets to zero across rows 1, 2, 4, 7 (rlvr-grpo §3)', () => {
  const A = groupAdvantages(R2);
  near(A[0] + A[1] + A[3] + A[6], 0, 1e-15);
});

test('groupAdvantages: dividing by G − 1 instead makes every advantage 1.069× smaller, +1.62 (rlvr-grpo §7 note a)', () => {
  const factor = Math.sqrt(8 / 7);
  assert.equal(at4(factor), 1.0690);
  assert.equal(at4(groupAdvantages(R2)[0] / factor), 1.6202);
});

test('groupAdvantages: throws on a bad option or reward', () => {
  assert.throws(() => groupAdvantages(R2, { normalizeStd: 'yes' }), { name: 'RangeError', message: 'groupAdvantages: normalizeStd must be true or false' });
  assert.throws(() => groupAdvantages([]), /groupStats: rewards must be a non-empty array/);
});

test('hasSignal: rlvr-grpo §11 signature', () => {
  assert.equal(hasSignal(R2), true);
  assert.equal(hasSignal([1, 1, 1, 1, 1, 1, 1, 1]), false);
  assert.equal(hasSignal([0, 0]), false);
  assert.equal(hasSignal([0.1, 0.1, 0.1]), false);
  assert.throws(() => hasSignal([]), /hasSignal: rewards must be a non-empty array/);
});

test('totalPush: rlvr-grpo §11 signature', () => {
  assert.equal(at4(totalPush(groupAdvantages(R2))), 6.9282);
  assert.equal(totalPush([1, -1, 1, -1]), 4);
  assert.equal(totalPush([0, 0, 0]), 0);
});

test('totalPush: rlvr-grpo try-this 1, std amplification (the ratio equals 1/std)', () => {
  const rows = [[1, 0, 0, 0, 0, 0, 0, 0], R2, [1, 0, 1, 0, 1, 0, 1, 0]].map((R) => {
    const a = totalPush(groupAdvantages(R));
    const d = totalPush(groupAdvantages(R, { normalizeStd: false }));
    near(a / d, 1 / groupStats(R).std, 1e-12);
    return [a, d, a / d].map((x) => x.toFixed(4));
  });
  assert.deepEqual(rows, [['5.2915', '1.7500', '3.0237'], ['6.9282', '3.0000', '2.3094'], ['8.0000', '4.0000', '2.0000']]);
  assert.deepEqual([2, 4, 8].map((k) => at2(totalPush(groupAdvantages(rewardsOf(group(k)))))), [6.93, 8, 0]);
});

test('totalPush: Σ|A| = 2G·√(p(1−p)) for 0/1 rewards (rlvr-grpo §7 shapes)', () => {
  for (let k = 0; k <= 8; k += 1) {
    const p = k / 8;
    near(totalPush(groupAdvantages(rewardsOf(group(k)))), 2 * 8 * Math.sqrt(p * (1 - p)), 1e-12);
  }
});

test('totalPush: throws on a bad advantage list', () => {
  assert.throws(() => totalPush([]), { name: 'RangeError', message: 'totalPush: advantages must be a non-empty array' });
  assert.throws(() => totalPush([1, Infinity]), { name: 'RangeError', message: 'totalPush: advantages[1] must be a finite number, got Infinity' });
});

test('tokenWeights: rlvr-grpo §11 signature', () => {
  assert.deepEqual(tokenWeights([5, 8, 1]).map((row) => row.map(at4)), [Array(5).fill(0.0667), Array(8).fill(0.0417), [0.3333]]);
  assert.deepEqual(tokenWeights([5, 8, 1], { aggregation: 'token' }).map((row) => row.map(at4)), [Array(5).fill(0.0714), Array(8).fill(0.0714), [0.0714]]);
  assert.equal(tokenWeights([5, 8, 1], { aggregation: 'token' })[0][0], 1 / 14);
  const sample = tokenWeights(LENGTHS);
  assert.deepEqual([sample[0][0], at4(sample[3][0]), sample[5][0]], [0.025, 0.0156, 0.125]);
  assert.equal(at4(tokenWeights(LENGTHS, { aggregation: 'token' })[3][0]), 0.0256);
  assert.equal(tokenWeights(LENGTHS, { aggregation: 'token' })[3][0], 1 / 39);
});

test('tokenWeights: rlvr-grpo try-this 2, rows 4 and 6 under both aggregations', () => {
  const A = groupAdvantages(R2);
  const ws = tokenWeights(LENGTHS);
  const wt = tokenWeights(LENGTHS, { aggregation: 'token' });
  assert.deepEqual([A[3] * ws[3][0], A[3] * wt[3][0], A[3] * ws[3][0] * 8, A[3] * wt[3][0] * 8].map(at4), [-0.0090, -0.0148, -0.0722, -0.1184]);
  assert.deepEqual([A[5] * ws[5][0], A[5] * wt[5][0]].map(at4), [-0.0722, -0.0148]);
  near(A[3] * ws[3].reduce((s, w) => s + w, 0), A[5] * ws[5].reduce((s, w) => s + w, 0)); // same push per answer
});

test('tokenWeights: rows sum to 1/G under sample, the table sums to 1 under both', () => {
  for (const lengths of [LENGTHS, [5, 8, 1], [1], [3, 3]]) {
    const G = lengths.length;
    tokenWeights(lengths).forEach((row) => near(row.reduce((s, w) => s + w, 0), 1 / G));
    for (const aggregation of ['sample', 'token']) near(tokenWeights(lengths, { aggregation }).flat().reduce((s, w) => s + w, 0), 1);
  }
});

test('tokenWeights: throws on bad lengths or aggregation', () => {
  assert.throws(() => tokenWeights([]), { name: 'RangeError', message: 'tokenWeights: lengths must be a non-empty array' });
  for (const bad of [0, -1, 1.5, Number.NaN, '5']) {
    assert.throws(() => tokenWeights([5, bad]), { name: 'RangeError', message: `tokenWeights: lengths[1] must be a positive integer, got ${bad}` });
  }
  assert.throws(() => tokenWeights([5], { aggregation: 'sequence' }), { name: 'RangeError', message: 'tokenWeights: aggregation must be \'sample\' or \'token\', got sequence' });
});

test('clippedSurrogate: rlvr-grpo §11 signature and frames 8–9', () => {
  const objective = (r) => ({ objective: at4(r.objective), clipped: r.clipped });
  assert.deepEqual(objective(clippedSurrogate(1.25, 1.7321)), { objective: 2.0785, clipped: true });
  assert.deepEqual(objective(clippedSurrogate(1.25, 1.7321, { epsHigh: 0.28 })), { objective: 2.1651, clipped: false });
  assert.deepEqual(objective(clippedSurrogate(0.75, -0.5774)), { objective: -0.4619, clipped: true });
  assert.deepEqual(objective(clippedSurrogate(1.50, -0.5774)), { objective: -0.8661, clipped: false });
  const A = groupAdvantages(R2);
  assert.equal(at2(clippedSurrogate(1.25, A[0]).objective), 2.08);
  assert.equal(at2(clippedSurrogate(1.25, A[0], { epsHigh: 0.28 }).objective), 2.17);
  assert.equal(clippedSurrogate(1.25, A[0]).objective.toFixed(3), '2.078'); // try-this 3
  assert.equal(clippedSurrogate(1.25, A[0], { epsHigh: 0.28 }).objective.toFixed(3), '2.165');
});

test('clippedSurrogate: rlvr-grpo reproducer at ε_high 0.20 and 0.28', () => {
  const A = groupAdvantages(R2);
  const clip = (eh) => [
    clippedSurrogate(1.25, A[0], { epsHigh: eh }).clipped, clippedSurrogate(1.35, A[4], { epsHigh: eh }).clipped,
    clippedSurrogate(0.75, A[3], { epsHigh: eh }).clipped, clippedSurrogate(1.5, A[3], { epsHigh: eh }).clipped,
  ];
  assert.deepEqual(clip(0.2), [true, true, true, false]);
  assert.deepEqual(clip(0.28), [false, true, true, false]);
});

test('clippedSurrogate: rlhf-dpo frame 5 (A = +0.2, ε = 0.2)', () => {
  const at125 = clippedSurrogate(1.25, 0.2);
  assert.deepEqual([at4(at125.objective), at125.clipped], [0.24, true]);
  const at110 = clippedSurrogate(1.1, 0.2);
  assert.deepEqual([at4(at110.objective), at110.clipped], [0.22, false]);
  assert.equal(at125.objective, Math.min(1.25 * 0.2, 1.2 * 0.2));
});

test('clippedSurrogate: rlvr-grpo try-this 3, the hatched chips of the default group', () => {
  // rlvr-grpo's page constants (§6): per-token ratios, 1.00 everywhere except rows 1, 5 and 4.
  const ratios = LENGTHS.map((n) => Array(n).fill(1));
  ratios[0] = [1.02, 0.98, 1.05, 1.00, 1.25];
  ratios[4] = [1.00, 1.00, 1.00, 1.00, 1.35];
  ratios[3] = [1.00, 1.00, 1.00, 1.00, 0.75, 1.00, 1.00, 1.10];
  const A = groupAdvantages(R2);
  const hatched = (epsHigh) => ratios.flatMap((row, i) => row.flatMap((r, t) => (clippedSurrogate(r, A[i], { epsLow: 0.2, epsHigh }).clipped ? [`${i + 1}:${t + 1}`] : [])));
  assert.deepEqual(hatched(0.2), ['1:5', '4:5', '5:5']);
  assert.deepEqual(hatched(0.28), ['4:5', '5:5']);
});

test('clippedSurrogate: clipped exactly when A > 0 and r > 1+ε_high or A < 0 and r < 1−ε_low, and then the gradient is zero', () => {
  const step = 1e-6;
  for (const A of [-1.7, -0.58, 0, 0.2, 1.73]) {
    for (let r = 0.5; r <= 1.6; r += 0.05) {
      for (const [epsLow, epsHigh] of [[0.2, 0.2], [0.2, 0.28], [0.1, 0.3]]) {
        const out = clippedSurrogate(r, A, { epsLow, epsHigh });
        assert.equal(out.clipped, (A > 0 && r > 1 + epsHigh) || (A < 0 && r < 1 - epsLow), `r ${r} A ${A}`);
        assert.equal(out.objective, Math.min(r * A, Math.min(Math.max(r, 1 - epsLow), 1 + epsHigh) * A));
        const slope = (clippedSurrogate(r + step, A, { epsLow, epsHigh }).objective - out.objective) / step;
        const isFlat = Math.abs(slope) < 1e-9;
        const nearEdge = Math.abs(r - (1 + epsHigh)) < 2 * step || Math.abs(r - (1 - epsLow)) < 2 * step;
        if (A !== 0 && !nearEdge) assert.equal(isFlat, out.clipped, `slope at r ${r} A ${A}`);
      }
    }
  }
});

test('clippedSurrogate: throws on a bad ratio, advantage or clip range', () => {
  for (const bad of [0, -1, Number.NaN, Infinity, '1']) {
    assert.throws(() => clippedSurrogate(bad, 1), { name: 'RangeError', message: `clippedSurrogate: ratio must be a finite number > 0, got ${bad}` });
  }
  assert.throws(() => clippedSurrogate(1, Number.NaN), { name: 'RangeError', message: 'clippedSurrogate: advantage must be a finite number, got NaN' });
  for (const bad of [-0.1, 1, Number.NaN, '0.2']) {
    assert.throws(() => clippedSurrogate(1, 1, { epsLow: bad }), { name: 'RangeError', message: `clippedSurrogate: epsLow must be a number in [0, 1), got ${bad}` });
  }
  for (const bad of [-0.1, Infinity, Number.NaN]) {
    assert.throws(() => clippedSurrogate(1, 1, { epsHigh: bad }), { name: 'RangeError', message: `clippedSurrogate: epsHigh must be a finite number ≥ 0, got ${bad}` });
  }
});

test('buildGroup: rlvr-grpo §11 signature (k = 2, 3, 8)', () => {
  const C = GROUP_TOY.correct;
  const W = GROUP_TOY.wrong;
  assert.deepEqual(group(2), [C[0], W[0], W[1], W[2], C[1], W[3], W[4], W[5]]);
  assert.deepEqual(group(3), [C[0], W[0], C[1], W[1], C[2], W[2], W[3], W[4]]);
  assert.deepEqual(group(8), C);
  assert.deepEqual(group(0), W);
  assert.deepEqual(group(3)[4], ['56']); // k = 3 puts C3 (`56`) in slot 5
});

test('buildGroup: slot k holds a correct answer exactly for the first k entries of the slot order', () => {
  for (let k = 0; k <= 8; k += 1) {
    const correctSlots = new Set(GROUP_TOY.slotOrder.slice(0, k));
    assert.deepEqual(rewardsOf(group(k)), [1, 2, 3, 4, 5, 6, 7, 8].map((slot) => (correctSlots.has(slot) ? 1 : 0)));
  }
});

test('buildGroup: pure, returns fresh rows, never mutates the pools', () => {
  const pools = deepFreeze({ correct: [['a'], ['b']], wrong: [['x'], ['y']], slotOrder: [2, 1] });
  const first = buildGroup(1, pools);
  assert.deepEqual(first, [['x'], ['a']]);
  assert.deepEqual(buildGroup(1, pools), first);
  first[0].push('changed');
  assert.deepEqual(buildGroup(1, pools), [['x'], ['a']]);
  assert.notEqual(group(2)[0], GROUP_TOY.correct[0]);
});

test('buildGroup: throws on k ∉ {0..slotOrder.length}, a non-integer k, a bad slot order or short pools', () => {
  for (const bad of [-1, 9, 1.5, Number.NaN, '2']) {
    assert.throws(() => group(bad), { name: 'RangeError', message: `buildGroup: k must be an integer 0–8, got ${bad}` });
  }
  const pools = { correct: [['a'], ['b']], wrong: [['x'], ['y']], slotOrder: [1, 2] };
  assert.throws(() => buildGroup(1, { ...pools, slotOrder: [1, 1] }), { name: 'RangeError', message: 'buildGroup: slotOrder must be a permutation of 1–2' });
  assert.throws(() => buildGroup(1, { ...pools, slotOrder: [] }), /buildGroup: slotOrder must be a non-empty array/);
  assert.throws(() => buildGroup(2, { ...pools, correct: [['a']] }), { name: 'RangeError', message: 'buildGroup: correct must hold at least 2 answers, got 1' });
  assert.throws(() => buildGroup(0, { ...pools, wrong: [['x']] }), { name: 'RangeError', message: 'buildGroup: wrong must hold at least 2 answers, got 1' });
  assert.throws(() => buildGroup(1, { ...pools, correct: [['a'], 'b'] }), { name: 'RangeError', message: 'buildGroup: correct[1] must be an array of strings' });
  assert.throws(() => buildGroup(1, { ...pools, wrong: [['x'], [3]] }), /buildGroup: wrong\[1\] must be an array of strings/);
  assert.throws(() => buildGroup(1), /buildGroup: correct must be an array of answers/);
});
