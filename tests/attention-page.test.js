import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { LESSON } from '../architecture/concepts/attention/content.js';
import { checkWork, fmt2, fmt3, cellText, trimNumber, divisorText, workedRowTex, expansion } from '../architecture/concepts/attention/format.js';
import { CAPTIONS, CHECK_WORK } from './attention-expected.js';
import { TOY } from '../math/attention.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json') };
const graph = await read('../shared/concepts.json');

test('the lesson spec is complete', () => assert.deepEqual(validateLessonSpec(LESSON), []));
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('6 facts rows, every placeholder resolves against data/models.json', () => {
  assert.equal(LESSON.facts.rows.length, 6);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('attention')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('"Check my work" for the default state is the storyboard text', () => {
  assert.equal(checkWork({ query: 2, divisor: 2, causal: true, head: 'A' }), CHECK_WORK);
});
test('number formats: real minus, masked −∞, exact zero', () => {
  assert.deepEqual([fmt2(-1), fmt2(3), fmt3(-0.10625), fmt3(0), fmt3(-Infinity), fmt3(0.0000446)], ['−1.00', '3.00', '−0.106', '0', '−∞', '0.000']);
});

// ---- format.js beyond the storyboard's default state ----

const DEFAULT = Object.freeze({ query: 2, divisor: 2, causal: true, head: 'A' });

test('stage cells: one format per quantity, never more than five characters at NUMBER_CELL', () => {
  assert.deepEqual([-1, 3, 0.5, -0.75].map((v) => cellText(v, 'score')), ['−1', '3', '0.5', '−0.75']);
  assert.deepEqual([-0.5, 0.625, -0.375, -0.125, -Infinity].map((v) => cellText(v, 'scaled')), ['−0.5', '0.625', '−0.38', '−0.13', '−∞']);
  assert.deepEqual([0.6065, 4.4817, 0, 1, 0.09535].map((v) => cellText(v, 'exp')), ['0.607', '4.482', '0', '1', '0.095']);
  assert.deepEqual([-0.10625, 1.40669, 0, -1].map((v) => cellText(v, 'output')), ['−0.11', '1.41', '0', '−1']);
  assert.throws(() => cellText(1, 'volume'), RangeError);
});

test('trimNumber: shortest exact form with a real minus and no negative zero', () => {
  assert.deepEqual([trimNumber(-0.5), trimNumber(3), trimNumber(-0.0001), trimNumber(0.34375), trimNumber(-Infinity)], ['−0.5', '3', '0', '0.344', '−∞']);
});

test('the divisor slider names 2 as the model\'s value', () => {
  assert.equal(divisorText(2), '2 = √d_head (the model\'s value)');
  assert.equal(divisorText(0.5), '0.5');
});

test('"Check my work" keeps its layout for other states: mask off lists all four keys', () => {
  const lines = checkWork({ ...DEFAULT, causal: false }).split('\n');
  assert.equal(lines.length, 9);
  assert.match(lines[3], /^q_sat · k_down = 0·0\.5 {4}\+ 2·\(−0\.5\) \+ 0\.5·0\.5 {3}\+ 0·1 {4}= −0\.75$/);
  assert.match(lines[6], /0\.086 {3}0\.635 {4}0\.182 {4}0\.097 {5}\(adds to 1\.000\)$/);
  assert.equal(lines[7], 'output = 0.086·v_The + 0.635·v_cat + 0.182·v_sat + 0.097·v_down');
});

test('"Check my work" for query The on head B with divisor 0.5: one visible key, weight 1', () => {
  const lines = checkWork({ query: 0, divisor: 0.5, causal: true, head: 'B' }).split('\n');
  assert.deepEqual(lines.slice(1, 4), ['q_The · k_cat  = masked (−∞)', 'q_The · k_sat  = masked (−∞)', 'q_The · k_down = masked (−∞)']);
  assert.equal(lines[4], '÷ 0.5                →   3');
  assert.equal(lines[7], 'output = 1.000·v_The');
  assert.equal(lines[8], '       = [0.00, 1.00, 0.00, 0.00]');
});

test('"Check my work" for head "both" works the head A row', () => {
  assert.equal(checkWork({ ...DEFAULT, head: 'both' }), CHECK_WORK);
});

test('the math panel\'s worked row is templated from the toy state and equals the storyboard\'s block', () => {
  const storyboard = '\\text{worked row: } A_{3,:} = \\operatorname{softmax}\\big([-0.5,\\ 1.5,\\ 0.25,\\ -\\infty]\\big) = [0.095,\\ 0.703,\\ 0.202,\\ 0]';
  assert.equal(workedRowTex(DEFAULT), storyboard);
  assert.ok(LESSON.math.blocks.some((b) => b.tex === storyboard));
});

test('format.js never mutates the frozen TOY it reads', () => {
  const before = JSON.stringify(TOY);
  checkWork({ ...DEFAULT, causal: false });
  assert.equal(JSON.stringify(TOY), before);
});

test('frame 2\'s expansion line writes out the "cat" dot product exactly as the storyboard does', () => {
  const { Q, K } = TOY.heads.A;
  assert.equal(expansion(Q[2], K[1], 3), '0·0 + 2·1.5 + 0.5·0 + 0·(−0.5) = 3.0');
  assert.equal(expansion(Q[2], K[3], -0.75), '0·0.5 + 2·(−0.5) + 0.5·0.5 + 0·1 = −0.75');
});
