import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deepFreeze, softmax } from '../math/core.js';
import { tokenLoss, meanLoss, perplexity, uniformLoss, klDivergence } from '../math/lm.js';

const at4 = (x) => Number(x.toFixed(4));
const at3 = (x) => Number(x.toFixed(3));

// pretraining's running example: decoder-anatomy frame 8's logits give p(on) = 0.3903 (core.softmax).
const P_ON = softmax([2.0, 1.5, 0.5, 0.0, ...Array(12).fill(-1.0)])[0];
const SENTENCE = [0.10, 0.25, 0.30, P_ON, 0.60, 0.45, 0.80];
const withMat = (v) => SENTENCE.map((x, j) => (j === 5 ? v : x));
// rlhf-dpo §5 frames 6–8 and distillation's running example.
const REF = [0.40, 0.30, 0.20, 0.10];
const HONEST = [0.70, 0.15, 0.10, 0.05];
const HACKED = [0.01, 0.01, 0.01, 0.97];
const TEACHER = [0.90, 0.05, 0.03, 0.02];
const CHAT = [0.60, 0.20, 0.10, 0.10];
const STUDENTS = { unsure: [0.40, 0.30, 0.20, 0.10], wrong: [0.10, 0.70, 0.10, 0.10], near: [0.85, 0.07, 0.05, 0.03] };

test('the running example probabilities come from decoder-anatomy (pretraining reproducer)', () => {
  const p = softmax([2.0, 1.5, 0.5, 0.0, ...Array(12).fill(-1.0)]);
  assert.deepEqual(p.slice(0, 5).map(at4), [0.3903, 0.2367, 0.0871, 0.0528, 0.0194]);
});

test('tokenLoss: pretraining §11 signature and reproducer', () => {
  assert.equal(at4(tokenLoss(P_ON)), 0.9410);
  assert.equal(at4(tokenLoss(0.01)), 4.6052);
  assert.equal(tokenLoss(1), 0);
  assert.equal(at4(tokenLoss(0.0625)), 2.7726);
  assert.equal(at4(tokenLoss(0.99)), 0.0101);
  assert.deepEqual(SENTENCE.map((p) => at4(tokenLoss(p))), [2.3026, 1.3863, 1.2040, 0.9410, 0.5108, 0.7985, 0.2231]);
  assert.equal(at3(SENTENCE.reduce((s, p) => s + tokenLoss(p), 0)), 7.366); // frame 5: mean = 7.366 / 7
});

test('tokenLoss: distillation "traces" loss on the teacher\'s token 56 for each student', () => {
  assert.equal(at4(tokenLoss(STUDENTS.unsure[0])), 0.9163);
  assert.equal(at4(tokenLoss(STUDENTS.wrong[0])), 2.3026);
  assert.equal(at4(tokenLoss(STUDENTS.near[0])), 0.1625);
});

test('tokenLoss: strictly decreasing on (0, 1]', () => {
  const grid = Array.from({ length: 100 }, (_, i) => (i + 1) / 100);
  grid.slice(1).forEach((p, i) => assert.ok(tokenLoss(p) < tokenLoss(grid[i]), `${p}`));
  assert.ok(tokenLoss(1e-300) > tokenLoss(1e-299));
});

test('tokenLoss: throws RangeError unless 0 < p ≤ 1', () => {
  for (const bad of [0, -0.1, 1.0001, Number.NaN, Infinity, '0.5', undefined]) {
    assert.throws(() => tokenLoss(bad), { name: 'RangeError', message: `tokenLoss: p must be a number with 0 < p ≤ 1, got ${bad}` });
  }
});

test('meanLoss and perplexity: pretraining frame 5 and the mat slider', () => {
  assert.equal(at4(meanLoss(SENTENCE)), 1.0523);
  assert.equal(at4(perplexity(meanLoss(SENTENCE))), 2.8643);
  assert.equal(at3(meanLoss(SENTENCE)), 1.052);
  assert.equal(perplexity(meanLoss(SENTENCE)).toFixed(2), '2.86');
  assert.deepEqual([0.01, 1.0, 0.0625].map((v) => [at4(meanLoss(withMat(v))), at4(perplexity(meanLoss(withMat(v))))]),
    [[1.5961, 4.9339], [0.9383, 2.5555], [1.3343, 3.7975]]);
  // try-this 1: one confident mistake costs +0.544, one perfect answer earns −0.114.
  assert.equal(at3(meanLoss(withMat(0.01)) - meanLoss(SENTENCE)), 0.544);
  assert.equal(at3(meanLoss(withMat(1.0)) - meanLoss(SENTENCE)), -0.114);
});

test('meanLoss and perplexity: a uniform guess over 16 words (pretraining try-this 3)', () => {
  const uniform = SENTENCE.map(() => 1 / 16);
  assert.equal(at4(meanLoss(uniform)), 2.7726);
  assert.equal(at4(perplexity(meanLoss(uniform))), 16);
});

test('meanLoss: the masked form (pretraining §11; sft builds its mask with lossMask)', () => {
  assert.equal(at4(meanLoss([0.5, 0.25, 0.8], [false, true, true])), 0.8047);
  assert.equal(meanLoss([0.5, 0.25, 0.8], null), meanLoss([0.5, 0.25, 0.8]));
  assert.equal(meanLoss([0.5, 0.25, 0.8], [true, true, true]), meanLoss([0.5, 0.25, 0.8]));
  // sft's §5 transcript mask: 26 positions, 15 trained.
  const mask = [...'00000000111111111111100011'].map((c) => c === '1');
  assert.equal(mask.filter(Boolean).length, 15);
  const probs = mask.map((m) => (m ? 0.5 : 0.01));
  assert.ok(Math.abs(meanLoss(probs, mask) - tokenLoss(0.5)) < 1e-12);
});

test('meanLoss: of a constant array equals tokenLoss of the constant', () => {
  for (const p of [0.01, 0.3903, 0.5, 1]) {
    for (const n of [1, 7, 26]) assert.ok(Math.abs(meanLoss(Array(n).fill(p)) - tokenLoss(p)) < 1e-12);
  }
});

test('meanLoss: throws when the mask length differs, no position is kept, or an entry is bad', () => {
  assert.throws(() => meanLoss([0.5, 0.25], [true]), { name: 'RangeError', message: 'meanLoss: mask must have one entry per position (2), got 1' });
  assert.throws(() => meanLoss([0.5, 0.25], [false, false]), { name: 'RangeError', message: 'meanLoss: mask must keep at least one position' });
  assert.throws(() => meanLoss([]), { name: 'RangeError', message: 'meanLoss: probs must be a non-empty array' });
  assert.throws(() => meanLoss('0.5'), /meanLoss: probs must be a non-empty array/);
  assert.throws(() => meanLoss([0.5, 1], [true, 1]), { name: 'RangeError', message: 'meanLoss: mask[1] must be true or false' });
  assert.throws(() => meanLoss([0.5, 0], [true, true]), /tokenLoss: p must be a number with 0 < p ≤ 1, got 0/);
  assert.throws(() => meanLoss([0.5], 'yes'), { name: 'RangeError', message: 'meanLoss: mask must be null or an array of booleans' });
  // A masked-out position is still checked: a bad probability is a bad input wherever it sits.
  assert.throws(() => meanLoss([0.5, 2], [true, false]), /tokenLoss: p must be a number with 0 < p ≤ 1, got 2/);
});

test('perplexity: e^loss (pretraining §11) and its errors', () => {
  // The signature's "1.0523 → 2.8643 · 2.7726 → 16" feed it the unrounded losses: e^1.0523 itself is 2.8642.
  assert.equal(at4(perplexity(meanLoss(SENTENCE))), 2.8643);
  assert.equal(at4(perplexity(uniformLoss(16))), 16);
  assert.equal(at4(perplexity(1.0523)), 2.8642);
  assert.equal(perplexity(2.7726).toFixed(2), '16.00');
  assert.equal(perplexity(0), 1);
  for (const bad of [-0.1, Number.NaN, Infinity, '1']) {
    assert.throws(() => perplexity(bad), { name: 'RangeError', message: `perplexity: loss must be a finite number ≥ 0, got ${bad}` });
  }
});

test('uniformLoss: ln |V| (pretraining §11 and §6 reference marks)', () => {
  assert.equal(at4(uniformLoss(16)), 2.7726);
  assert.equal(at4(uniformLoss(163840)), 12.0066);
  assert.equal(at4(uniformLoss(201088)), 12.2115);
  assert.equal(at3(uniformLoss(16)), 2.773);
  assert.equal(at3(uniformLoss(163840)), 12.007);
  assert.equal(uniformLoss(1), 0);
});

test('uniformLoss: perplexity(uniformLoss(V)) = V', () => {
  for (const v of [1, 2, 16, 50257, 163840, 201088]) assert.ok(Math.abs(perplexity(uniformLoss(v)) - v) < 1e-9 * v, `${v}`);
});

test('uniformLoss: throws unless an integer ≥ 1', () => {
  for (const bad of [0, -16, 1.5, Number.NaN, Infinity, '16']) {
    assert.throws(() => uniformLoss(bad), { name: 'RangeError', message: `uniformLoss: vocabSize must be an integer ≥ 1, got ${bad}` });
  }
});

test('klDivergence: rlhf-dpo frames 7–8 (honest and hacked policy against the reference)', () => {
  assert.equal(at4(klDivergence(HONEST, REF)), 0.1838);
  assert.equal(at4(klDivergence(HACKED, REF)), 2.1031);
  assert.equal(at3(klDivergence(HONEST, REF)), 0.184);
  assert.equal(at3(klDivergence(HACKED, REF)), 2.103);
});

test('klDivergence: distillation forward KL(teacher ‖ student) and reverse KL(student ‖ teacher)', () => {
  assert.equal(at4(klDivergence(TEACHER, STUDENTS.unsure)), 0.5511);
  assert.equal(at4(klDivergence(STUDENTS.unsure, TEACHER)), 0.7535);
  assert.equal(at4(klDivergence(TEACHER, STUDENTS.wrong)), 1.7772);
  assert.equal(at4(klDivergence(STUDENTS.wrong, TEACHER)), 1.9090);
  assert.equal(at4(klDivergence(TEACHER, STUDENTS.near)), 0.0112);
  assert.equal(at4(klDivergence(STUDENTS.near, TEACHER)), 0.0127);
  assert.equal(at3(klDivergence(TEACHER, STUDENTS.unsure)), 0.551);
});

test('klDivergence: distillation\'s chat teacher and the 50/50 mix (multiTeacherLoss with one teacher is this KL)', () => {
  const chat = Object.values(STUDENTS).map((s) => at4(klDivergence(s, CHAT)));
  assert.deepEqual(chat, [0.0981, 0.6978, 0.1518]);
  const mix = Object.values(STUDENTS).map((s) => 0.5 * klDivergence(s, TEACHER) + 0.5 * klDivergence(s, CHAT));
  assert.deepEqual(mix.map(at4), [0.4258, 1.3034, 0.0822]);
});

test('klDivergence: the student\'s expected on-policy reward is minus the reverse KL (distillation frame 5)', () => {
  const s = STUDENTS.unsure;
  const expectedReward = s.reduce((sum, p, i) => sum + p * (Math.log(TEACHER[i]) - Math.log(p)), 0);
  assert.ok(Math.abs(expectedReward + klDivergence(s, TEACHER)) < 1e-12);
  assert.equal(at3(expectedReward), -0.754);
});

test('klDivergence: ≥ 0, zero only when p = q, and not symmetric', () => {
  for (const p of [REF, HONEST, HACKED, TEACHER, CHAT, ...Object.values(STUDENTS)]) {
    assert.equal(klDivergence(p, p), 0);
    for (const q of [REF, HONEST, HACKED, TEACHER, CHAT]) {
      if (q.some((v, i) => v !== p[i])) assert.ok(klDivergence(p, q) > 0, `${p} vs ${q}`);
    }
  }
  assert.notEqual(at4(klDivergence(TEACHER, STUDENTS.unsure)), at4(klDivergence(STUDENTS.unsure, TEACHER)));
});

test('klDivergence: a term with p = 0 contributes 0, even where q = 0', () => {
  assert.equal(at4(klDivergence([0.5, 0.5, 0], [0.25, 0.25, 0.5])), at4(Math.log(2)));
  assert.equal(klDivergence([1, 0], [1, 0]), 0);
});

test('klDivergence: throws on a length mismatch, q_i = 0 where p_i > 0, or a bad entry', () => {
  assert.throws(() => klDivergence([0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]), { name: 'RangeError', message: 'klDivergence: p and q must have the same length (2 vs 3)' });
  assert.throws(() => klDivergence([0.5, 0.5], [1, 0]), { name: 'RangeError', message: 'klDivergence: q[1] must be > 0 where p[1] > 0' });
  assert.throws(() => klDivergence([], []), { name: 'RangeError', message: 'klDivergence: p must be a non-empty array' });
  assert.throws(() => klDivergence([1], 'q'), /klDivergence: q must be a non-empty array/);
  for (const bad of [-0.1, 1.1, Number.NaN, '0.5']) {
    assert.throws(() => klDivergence([bad, 0.5], [0.5, 0.5]), { name: 'RangeError', message: `klDivergence: p[0] must be a probability in [0, 1], got ${bad}` });
    assert.throws(() => klDivergence([0.5, 0.5], [0.5, bad]), /klDivergence: q\[1\] must be a probability in \[0, 1\]/);
  }
});

test('no input array is mutated', () => {
  const probs = deepFreeze([...SENTENCE]);
  const mask = deepFreeze([true, false, true, true, true, true, true]);
  const p = deepFreeze([...TEACHER]);
  const q = deepFreeze([...STUDENTS.unsure]);
  assert.equal(at4(meanLoss(probs)), 1.0523);
  assert.ok(meanLoss(probs, mask) > 0);
  assert.equal(at4(klDivergence(p, q)), 0.5511);
  assert.deepEqual(q, [0.40, 0.30, 0.20, 0.10]);
});
