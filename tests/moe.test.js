import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ROUTER_TOY, routeTopK, gateWeights, expertLoads, imbalance, balanceStep, simulateBalancing, expertCombinations } from '../math/moe.js';
import { paramBreakdown, mlpParams, PRESETS } from '../math/params.js';

const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const nearAll = (a, b, eps) => { assert.equal(a.length, b.length); a.forEach((v, i) => near(v, b[i], eps)); };
const sum = (xs) => xs.reduce((s, x) => s + x, 0);
const BIAS_SAT = [0, 0, 0, 0, 0, -0.75, 0.5, 0];
const POPULARITY = [1, 0.5, 0, 0, 0, 0, 0, 0];
const BATCH = { tokens: 128, experts: 8, k: 2, seed: 100, popularity: POPULARITY };
const moeToy = (moe) => paramBreakdown({ ...PRESETS.toy, moe: { shared: 0, denseLayers: 0, ...moe } });

test('ROUTER_TOY is the 4 × 8 table of the storyboard, frozen', () => {
  assert.equal(ROUTER_TOY.length, 4);
  assert.ok(ROUTER_TOY.every((row) => row.length === 8));
  assert.deepEqual(ROUTER_TOY[2], [0, -0.5, 2.0, 0.25, -1, 1.5, 0.5, 0]);
  assert.ok(Object.isFrozen(ROUTER_TOY) && Object.isFrozen(ROUTER_TOY[0]));
});

test('routeTopK: the four words pick the storyboard\'s experts (0-based)', () => {
  assert.deepEqual(ROUTER_TOY.map((row) => routeTopK(row, 2)), [[1, 4], [3, 5], [2, 5], [0, 6]]);
});

test('routeTopK: the bias changes the choice (frame 9): E3 and E7', () => {
  assert.deepEqual(routeTopK(ROUTER_TOY[2], 2, BIAS_SAT), [2, 6]);
});

test('routeTopK: a zero bias equals routing without one; ties go to the lower index', () => {
  ROUTER_TOY.forEach((row) => assert.deepEqual(routeTopK(row, 2, new Array(8).fill(0)), routeTopK(row, 2)));
  assert.deepEqual(routeTopK([1, 1, 1, 0], 2), [0, 1]);
  assert.deepEqual(routeTopK([0, 3, 3, 3], 1), [1]);
});

test('routeTopK: errors', () => {
  assert.throws(() => routeTopK([1, 2], 0), /routeTopK: k must be an integer from 1 to 2/);
  assert.throws(() => routeTopK([1, 2], 3), /routeTopK: k must be an integer from 1 to 2/);
  assert.throws(() => routeTopK([], 1), /routeTopK: scores must be a non-empty array/);
  assert.throws(() => routeTopK([1, 2], 1, [0]), /routeTopK: bias must have 2 entries/);
  assert.throws(() => routeTopK([1, Number.NaN], 1), /routeTopK: scores must be finite numbers/);
});

test('gateWeights: softmax over the chosen raw scores', () => {
  nearAll(gateWeights(ROUTER_TOY[2], [2, 5]), [0.622, 0.378]);
  nearAll(gateWeights(ROUTER_TOY[2], [2, 6]), [0.818, 0.182]);
  nearAll(gateWeights(ROUTER_TOY[3], [0, 6]), [0.562, 0.438]);
});

test('gateWeights sums to 1 and never sees the bias (the frame 9 claim)', () => {
  ROUTER_TOY.forEach((row) => near(sum(gateWeights(row, routeTopK(row, 2))), 1, 1e-12));
  // The weights for the biased choice [E3, E7] come from the raw scores 2.0 and 0.5, not from the biased 2.0 and 1.0.
  const picks = routeTopK(ROUTER_TOY[2], 2, BIAS_SAT);
  nearAll(gateWeights(ROUTER_TOY[2], picks), [0.818, 0.182]);
  assert.notDeepEqual(gateWeights(ROUTER_TOY[2], picks).map((w) => w.toFixed(3)), ['0.731', '0.269']);
});

test('gateWeights: errors', () => {
  assert.throws(() => gateWeights([1, 2], []), /gateWeights: picks must be a non-empty array/);
  assert.throws(() => gateWeights([1, 2], [2]), /gateWeights: pick 2 is outside 0–1/);
  assert.throws(() => gateWeights([1, 2], [0, 0]), /gateWeights: picks must be distinct/);
});

test('expertLoads: the four words land on seven experts and one gets nothing', () => {
  assert.deepEqual(expertLoads(ROUTER_TOY, 2), [1, 1, 1, 1, 1, 2, 1, 0]);
});

test('expertLoads sums to rows · k, with and without a bias', () => {
  assert.equal(sum(expertLoads(ROUTER_TOY, 2)), 8);
  assert.equal(sum(expertLoads(ROUTER_TOY, 3, BIAS_SAT)), 12);
  assert.throws(() => expertLoads([], 2), /expertLoads: rows must be a non-empty array/);
});

test('imbalance: busiest load ÷ fair share', () => {
  near(imbalance([96, 51, 12, 16, 25, 9, 21, 26], { tokens: 128, k: 2 }), 3, 1e-12);
  near(imbalance([32, 35, 36, 33, 28, 28, 30, 34], { tokens: 128, k: 2 }), 1.125, 1e-12);
  assert.equal(imbalance([32, 32, 32, 32, 32, 32, 32, 32], { tokens: 128, k: 2 }), 1);
});

test('imbalance: errors', () => {
  assert.throws(() => imbalance([], { tokens: 128, k: 2 }), /imbalance: loads must be a non-empty array/);
  assert.throws(() => imbalance([1, 2], { tokens: 0, k: 2 }), /imbalance: tokens must be a positive integer/);
  assert.throws(() => imbalance([1, 2], { tokens: 4, k: 0 }), /imbalance: k must be a positive integer/);
});

test('balanceStep: nudge up below fair share, down above it, none at it', () => {
  nearAll(balanceStep(new Array(8).fill(0), [96, 51, 12, 16, 25, 9, 21, 26], { gamma: 0.1, tokens: 128, k: 2 }), [-0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 1e-12);
  // 4 tokens · k 2 ÷ 4 experts = fair share 2: a load of 5 is above it, 2 is at it, 1 and 0 are below it
  nearAll(balanceStep([0.5, -0.5, 0, 0], [5, 2, 1, 0], { gamma: 0.2, tokens: 4, k: 2 }), [0.3, -0.5, 0.2, 0.2], 1e-12);
});

test('balanceStep: inputs not mutated; errors', () => {
  const bias = Object.freeze([0, 0]);
  const loads = Object.freeze([6, 2]);
  assert.deepEqual(balanceStep(bias, loads, { gamma: 0.1, tokens: 4, k: 2 }), [-0.1, 0.1]);
  assert.throws(() => balanceStep([0], [1, 2], { gamma: 0.1, tokens: 4, k: 2 }), /balanceStep: bias and loads must have the same length/);
  assert.throws(() => balanceStep([0, 0], [1, 2], { gamma: -1, tokens: 4, k: 2 }), /balanceStep: gamma must be a number ≥ 0/);
});

test('simulateBalancing, gamma 0.1: imbalance by step, and the loads and biases the storyboard prints', () => {
  const run = simulateBalancing({ ...BATCH, gamma: 0.1, steps: 10 });
  assert.equal(run.length, 10);
  assert.deepEqual(run.map((r) => r.step), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(run.map((r) => r.imbalance.toFixed(2)), ['3.00', '2.31', '2.03', '1.75', '1.44', '1.44', '1.13', '1.44', '1.13', '1.44']);
  assert.deepEqual(run[0].loads, [96, 51, 12, 16, 25, 9, 21, 26]);
  assert.deepEqual(run[3].loads, [56, 35, 35, 30, 26, 26, 27, 21]);
  assert.deepEqual(run[6].loads, [32, 35, 36, 33, 28, 28, 30, 34]);
  nearAll(run[6].bias, [-0.6, -0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 1e-9);
  assert.deepEqual(run[0].bias, new Array(8).fill(0));
});

test('simulateBalancing, gamma 0 and 0.2', () => {
  const off = simulateBalancing({ ...BATCH, gamma: 0, steps: 10 });
  assert.deepEqual(off.map((r) => r.imbalance.toFixed(2)), ['3.00', '2.53', '2.72', '2.69', '2.88', '2.50', '2.88', '2.69', '2.59', '2.91']);
  assert.ok(off.every((r) => r.bias.every((b) => b === 0)), 'gamma 0 never changes the bias');
  const big = simulateBalancing({ ...BATCH, gamma: 0.2, steps: 10 });
  assert.deepEqual(big.map((r) => r.imbalance.toFixed(2)), ['3.00', '1.94', '1.22', '1.28', '1.31', '1.84', '1.28', '1.47', '1.44', '1.75']);
});

test('simulateBalancing: a shorter run is a prefix of a longer one; every step sums to tokens · k', () => {
  const long = simulateBalancing({ ...BATCH, gamma: 0.1, steps: 10 });
  const short = simulateBalancing({ ...BATCH, gamma: 0.1, steps: 3 });
  assert.deepEqual(short, long.slice(0, 3));
  assert.ok(long.every((r) => sum(r.loads) === 256));
});

test('simulateBalancing: inputs not mutated; errors', () => {
  const popularity = Object.freeze([...POPULARITY]);
  assert.equal(simulateBalancing({ ...BATCH, popularity, gamma: 0.1, steps: 2 }).length, 2);
  assert.equal(simulateBalancing({ tokens: 8, experts: 4, k: 1, gamma: 0.1, steps: 1, seed: 1 }).length, 1, 'popularity defaults to none');
  assert.throws(() => simulateBalancing({ ...BATCH, gamma: 0.1, steps: 0 }), /simulateBalancing: steps must be a positive integer/);
  assert.throws(() => simulateBalancing({ ...BATCH, gamma: 0.1, steps: 2, popularity: [1] }), /simulateBalancing: popularity must have 8 entries/);
  assert.throws(() => simulateBalancing({ ...BATCH, gamma: 0.1, steps: 2, k: 9 }), /simulateBalancing: k must be an integer from 1 to 8/);
});

test('expertCombinations: exact, as BigInt', () => {
  assert.equal(expertCombinations(8, 2), 28n);
  assert.equal(expertCombinations(16, 4), 1820n);
  assert.equal(expertCombinations(32, 8), 10_518_300n);
  assert.equal(expertCombinations(384, 6), 4_281_625_192_384n);
  assert.equal(expertCombinations(256, 8), 409_663_695_276_000n);
  assert.equal(expertCombinations(5, 0), 1n);
  assert.equal(expertCombinations(5, 5), 1n);
  assert.throws(() => expertCombinations(4, 5), /expertCombinations: k must be an integer from 0 to 4/);
  assert.throws(() => expertCombinations(0, 0), /expertCombinations: n must be a positive integer/);
});

// ---- the parameter counts the page prints come from math/params.js; these pin the storyboard's worked examples ----

test('params worked examples: dense, 8 and 16 experts, fine-grained, shared', () => {
  const dense = paramBreakdown(PRESETS.toy);
  assert.deepEqual([dense.total, dense.active], [1576, 1448]);
  const cases = [
    [{ routed: 8, topK: 2, hidden: 8 }, 4008, 1576, 64],
    [{ routed: 16, topK: 2, hidden: 8 }, 7208, 1704, 128],
    [{ routed: 16, topK: 4, hidden: 4 }, 4136, 1704, 128],
    [{ routed: 32, topK: 8, hidden: 2 }, 4392, 1960, 256],
    [{ routed: 8, shared: 1, topK: 1, hidden: 8 }, 4392, 1576, 64],
  ];
  cases.forEach(([moe, total, active, router]) => {
    const b = moeToy(moe);
    assert.deepEqual([b.total, b.active, b.parts.router / 2], [total, active, router], JSON.stringify(moe));
  });
});

test('two experts of hidden 8 equal one dense MLP of hidden 16 (lesson 17)', () => {
  assert.equal(2 * mlpParams({ kind: 'swiglu', hidden: 8 }, 8), mlpParams({ kind: 'swiglu', hidden: 16 }, 8));
  assert.equal(mlpParams({ kind: 'swiglu', hidden: 8 }, 8), 192);
  assert.equal(mlpParams({ kind: 'swiglu', hidden: 4 }, 8), 96);
});

test('fine-grained keeps the work fixed: expert parameters per token are 384 at every split (lesson 17)', () => {
  [[8, 2, 8], [16, 4, 4], [32, 8, 2]].forEach(([routed, topK, hidden]) => {
    assert.equal(topK * mlpParams({ kind: 'swiglu', hidden }, 8), 384, `${routed} experts`);
  });
});
