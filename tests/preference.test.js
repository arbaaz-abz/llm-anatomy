import { test } from 'node:test';
import assert from 'node:assert/strict';
import { klDivergence } from '../math/lm.js';
import { clippedSurrogate } from '../math/grpo.js';
import { sigmoid, bradleyTerry, dpoLoss, klPenalizedReward } from '../math/preference.js';

const at4 = (x) => Number(x.toFixed(4));
const REF = [0.40, 0.30, 0.20, 0.10];
const HONEST = [0.70, 0.15, 0.10, 0.05];
const HACKED = [0.01, 0.01, 0.01, 0.97];

test('sigmoid: storyboard §11 values, symmetric, stable far from zero', () => {
  assert.equal(sigmoid(0), 0.5);
  assert.equal(at4(sigmoid(1.5)), 0.8176);
  assert.equal(at4(sigmoid(-0.07)), 0.4825);
  assert.equal(sigmoid(-800), 0);
  assert.equal(sigmoid(800), 1);
  assert.ok(Math.abs(sigmoid(2) + sigmoid(-2) - 1) < 1e-15);
});

test('sigmoid: rejects a non-finite argument', () => {
  assert.throws(() => sigmoid(Number.NaN), RangeError);
  assert.throws(() => sigmoid(Infinity), /sigmoid: x must be a finite number/);
  assert.throws(() => sigmoid('1'), RangeError);
});

test('bradleyTerry: frame 2 (1.2 vs −0.3), the flipped pair, and a tie', () => {
  const right = bradleyTerry(1.2, -0.3);
  assert.equal(at4(right.pChosen), 0.8176);
  assert.equal(at4(right.loss), 0.2014);
  assert.equal(at4(bradleyTerry(-0.3, 1.2).loss), 1.7014);
  const tie = bradleyTerry(0, 0);
  assert.equal(tie.pChosen, 0.5);
  assert.equal(at4(tie.loss), 0.6931);
});

test('bradleyTerry: loss plus ln(pChosen) is zero', () => {
  [[1.2, -0.3], [-2, 3], [0.1, 0.1]].forEach(([a, b]) => {
    const { pChosen, loss } = bradleyTerry(a, b);
    assert.ok(Math.abs(loss + Math.log(pChosen)) < 1e-12);
  });
});

test('bradleyTerry: bad scores throw', () => {
  assert.throws(() => bradleyTerry(Number.NaN, 0), /bradleyTerry: rChosen must be a finite number/);
  assert.throws(() => bradleyTerry(0, Infinity), /bradleyTerry: rRejected must be a finite number/);
});

test('dpoLoss: the storyboard §11 table', () => {
  const at = (dChosen, dRejected, beta) => {
    const r = dpoLoss({ dChosen, dRejected, beta });
    return [r.rewardChosen, r.rewardRejected, r.margin, r.loss, r.weight].map(at4);
  };
  assert.deepEqual(at(0.5, -0.2, 0.1), [0.05, -0.02, 0.07, 0.6588, 0.4825]);
  assert.deepEqual(at(-1, -3, 0.1), [-0.1, -0.3, 0.2, 0.5981, 0.4502]);
  assert.deepEqual(at(0.5, -0.2, 0.5), [0.25, -0.1, 0.35, 0.5334, 0.4134]);
  assert.deepEqual(at(0, 0, 0.1), [0, 0, 0, 0.6931, 0.5]);
  assert.deepEqual(at(5, -5, 0.1), [0.5, -0.5, 1, 0.3133, 0.2689]);
  assert.deepEqual(at(10, -10, 0.1), [1, -1, 2, 0.1269, 0.1192]);
});

test('dpoLoss: beta defaults to 0.1', () => {
  assert.deepEqual(dpoLoss({ dChosen: 0.5, dRejected: -0.2 }), dpoLoss({ dChosen: 0.5, dRejected: -0.2, beta: 0.1 }));
});

test('dpoLoss: try-this 3 reaches the same loss with five times less drift', () => {
  const small = dpoLoss({ dChosen: 3.5, dRejected: 0, beta: 0.1 });
  const large = dpoLoss({ dChosen: 0.7, dRejected: 0, beta: 0.5 });
  assert.ok(Math.abs(small.loss - large.loss) < 1e-12);
  assert.equal(at4(small.margin), 0.35);
  assert.equal(at4(large.margin), 0.35);
});

test('dpoLoss: zero margin is ln 2 for any beta; weight and sigma add to 1', () => {
  [0.01, 0.1, 0.5, 3].forEach((beta) => {
    assert.ok(Math.abs(dpoLoss({ dChosen: 2, dRejected: 2, beta }).loss - Math.LN2) < 1e-12);
  });
  [[0.5, -0.2], [-1, -3], [10, -10]].forEach(([dChosen, dRejected]) => {
    const { margin, weight } = dpoLoss({ dChosen, dRejected });
    assert.ok(Math.abs(weight + sigmoid(margin) - 1) < 1e-12);
  });
});

test('dpoLoss: loss falls as dChosen rises and rises as dRejected rises', () => {
  const losses = (f) => [-4, -1, 0, 1, 4].map(f);
  const byChosen = losses((d) => dpoLoss({ dChosen: d, dRejected: 0 }).loss);
  const byRejected = losses((d) => dpoLoss({ dChosen: 0, dRejected: d }).loss);
  byChosen.slice(1).forEach((l, i) => assert.ok(l < byChosen[i]));
  byRejected.slice(1).forEach((l, i) => assert.ok(l > byRejected[i]));
});

test('dpoLoss: stays finite at extreme margins', () => {
  const far = dpoLoss({ dChosen: -500, dRejected: 500, beta: 5 });
  assert.ok(Number.isFinite(far.loss) && far.loss > 0);
  assert.ok(Number.isFinite(dpoLoss({ dChosen: 500, dRejected: -500, beta: 5 }).loss));
});

test('dpoLoss: bad arguments throw RangeError', () => {
  assert.throws(() => dpoLoss({ dChosen: 0, dRejected: 0, beta: 0 }), /dpoLoss: beta must be a finite number > 0/);
  assert.throws(() => dpoLoss({ dChosen: 0, dRejected: 0, beta: -1 }), RangeError);
  assert.throws(() => dpoLoss({ dChosen: Number.NaN, dRejected: 0 }), /dpoLoss: dChosen must be a finite number/);
  assert.throws(() => dpoLoss({ dChosen: 0, dRejected: Infinity }), /dpoLoss: dRejected must be a finite number/);
  assert.throws(() => dpoLoss(), RangeError);
});

test('klPenalizedReward: frame 8 (honest 0.908, hacked 0.248) and the beta crossover', () => {
  const kh = klDivergence(HONEST, REF);
  const kx = klDivergence(HACKED, REF);
  assert.equal(at4(kh), 0.1838);
  assert.equal(at4(kx), 2.1031);
  assert.equal(at4(klPenalizedReward(1.0, kh, 0.5)), 0.9081);
  assert.equal(at4(klPenalizedReward(1.3, kx, 0.5)), 0.2484);
  assert.equal(klPenalizedReward(1.3, kx, 0), 1.3);
  assert.deepEqual([0.1, 0.2].map((b) => [at4(klPenalizedReward(1.0, kh, b)), at4(klPenalizedReward(1.3, kx, b))]), [[0.9816, 1.0897], [0.9632, 0.8794]]);
});

test('klPenalizedReward: bad arguments throw RangeError', () => {
  assert.throws(() => klPenalizedReward(Number.NaN, 0, 0.5), /klPenalizedReward: reward must be a finite number/);
  assert.throws(() => klPenalizedReward(1, -0.1, 0.5), /klPenalizedReward: kl must be a finite number ≥ 0/);
  assert.throws(() => klPenalizedReward(1, 0, -0.5), /klPenalizedReward: beta must be a finite number ≥ 0/);
});

test('frame 5 imports the clip from math/grpo.js unchanged', () => {
  assert.equal(at4(clippedSurrogate(1.25, 0.2).objective), 0.24);
  assert.equal(clippedSurrogate(1.25, 0.2).clipped, true);
  assert.equal(at4(clippedSurrogate(1.1, 0.2).objective), 0.22);
  assert.equal(clippedSurrogate(1.1, 0.2).clipped, false);
});
