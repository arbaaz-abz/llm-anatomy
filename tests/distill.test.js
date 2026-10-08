import { test } from 'node:test';
import assert from 'node:assert/strict';
import { opdTokenReward, multiTeacherLoss } from '../math/distill.js';
import { klDivergence } from '../math/lm.js';

const at4 = (x) => Number(x.toFixed(4));
const TEACHER = [0.90, 0.05, 0.03, 0.02];
const CHAT = [0.60, 0.20, 0.10, 0.10];
const UNSURE = [0.40, 0.30, 0.20, 0.10];
const WRONG = [0.10, 0.70, 0.10, 0.10];
const NEAR = [0.85, 0.07, 0.05, 0.03];

test('opdTokenReward: distillation §11 worked examples', () => {
  assert.equal(at4(opdTokenReward(0.90, 0.40)), 0.8109);
  assert.equal(at4(opdTokenReward(0.05, 0.30)), -1.7918);
  assert.equal(opdTokenReward(0.97, 0.97), 0);
  assert.equal(opdTokenReward(0.05, 0.30, { clip: 1 }), -1);
  assert.equal(opdTokenReward(0.90, 0.40, { clip: 1 }), opdTokenReward(0.90, 0.40));
});

test('opdTokenReward: the try-this rewards for each student (storyboard §6, reproducer)', () => {
  assert.deepEqual(UNSURE.map((s, i) => at4(opdTokenReward(TEACHER[i], s))), [0.8109, -1.7918, -1.8971, -1.6094]);
  assert.deepEqual(WRONG.map((s, i) => at4(opdTokenReward(TEACHER[i], s))), [2.1972, -2.6391, -1.2040, -1.6094]);
  assert.deepEqual(NEAR.map((s, i) => at4(opdTokenReward(TEACHER[i], s))), [0.0572, -0.3365, -0.5108, -0.4055]);
});

test('opdTokenReward: frame 6 row 2 per-token rewards', () => {
  const pairs = [[0.5, 0.5], [0.95, 0.9], [0.98, 0.95], [0.97, 0.97], [0.05, 0.3]];
  assert.deepEqual(pairs.map(([t, s]) => at4(opdTokenReward(t, s))), [0, 0.0541, 0.0311, 0, -1.7918]);
});

test('opdTokenReward: the expected reward under the student equals minus the reverse KL', () => {
  [UNSURE, WRONG, NEAR].forEach((s) => {
    const expected = s.reduce((sum, p, i) => sum + p * opdTokenReward(TEACHER[i], p), 0);
    assert.ok(Math.abs(expected + klDivergence(s, TEACHER)) < 1e-12);
  });
  assert.equal(at4(-klDivergence(UNSURE, TEACHER)), -0.7535);
});

test('opdTokenReward: throws unless both probabilities are > 0 (and a valid clip)', () => {
  assert.throws(() => opdTokenReward(0, 0.3), RangeError);
  assert.throws(() => opdTokenReward(0.3, 0), RangeError);
  assert.throws(() => opdTokenReward(1.2, 0.3), RangeError);
  assert.throws(() => opdTokenReward('a', 0.3), RangeError);
  assert.throws(() => opdTokenReward(0.3, 0.3, { clip: 0 }), RangeError);
  assert.throws(() => opdTokenReward(0.3, 0.3, { clip: -1 }), RangeError);
});

test('multiTeacherLoss: one teacher at weight 1 is the reverse KL', () => {
  [UNSURE, WRONG, NEAR].forEach((s) => assert.equal(multiTeacherLoss(s, [TEACHER], [1]), klDivergence(s, TEACHER)));
  assert.equal(at4(multiTeacherLoss(NEAR, [TEACHER], [1])), 0.0127);
  assert.equal(at4(multiTeacherLoss(NEAR, [CHAT], [1])), 0.1518);
  assert.equal(at4(multiTeacherLoss(UNSURE, [CHAT], [1])), 0.0981);
});

test('multiTeacherLoss: the 50/50 mix (storyboard §6 try-this 3)', () => {
  assert.equal(at4(multiTeacherLoss(NEAR, [TEACHER, CHAT], [0.5, 0.5])), 0.0822);
  assert.equal(at4(multiTeacherLoss(UNSURE, [TEACHER, CHAT], [0.5, 0.5])), 0.4258);
  assert.equal(at4(multiTeacherLoss(WRONG, [TEACHER, CHAT], [0.5, 0.5])), 1.3034);
});

test('multiTeacherLoss: zero only when the student equals every teacher with weight', () => {
  assert.equal(multiTeacherLoss(TEACHER, [TEACHER, TEACHER], [0.3, 0.7]), 0);
  assert.ok(multiTeacherLoss(TEACHER, [TEACHER, CHAT], [0.5, 0.5]) > 0);
  assert.equal(multiTeacherLoss(TEACHER, [TEACHER, CHAT], [1, 0]), 0);
});

test('multiTeacherLoss: throws on a count mismatch, a negative weight or weights not summing to 1', () => {
  assert.throws(() => multiTeacherLoss(NEAR, [TEACHER, CHAT], [1]), RangeError);
  assert.throws(() => multiTeacherLoss(NEAR, [TEACHER, CHAT], [1.5, -0.5]), RangeError);
  assert.throws(() => multiTeacherLoss(NEAR, [TEACHER, CHAT], [0.5, 0.4]), RangeError);
  assert.throws(() => multiTeacherLoss(NEAR, [], []), RangeError);
  assert.throws(() => multiTeacherLoss(NEAR, [[0.5, 0.5]], [1]), RangeError);
});

test('inputs are not mutated', () => {
  const student = Object.freeze([...NEAR]);
  const teachers = Object.freeze([Object.freeze([...TEACHER]), Object.freeze([...CHAT])]);
  const weights = Object.freeze([0.5, 0.5]);
  multiTeacherLoss(student, teachers, weights);
  assert.deepEqual([...student], NEAR);
  assert.deepEqual(teachers.map((t) => [...t]), [TEACHER, CHAT]);
});
