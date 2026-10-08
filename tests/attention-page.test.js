import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { LESSON, lessonFor } from '../architecture/concepts/attention/content.js';
import { fillText } from '../shared/claims.js';
import { checkWork, fmt2, fmt3, cellText, trimNumber, divisorText, workedRowTex, expansion, INITIAL_STATE } from '../architecture/concepts/attention/format.js';
import { CAPTIONS, CHECK_WORK } from './attention-expected.js';
import { TOY } from '../math/attention.js';
import { tryThis } from '../architecture/concepts/attention/try-this.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
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

test('the lesson filled with real data is complete (template rule 1)', () => {
  const before = JSON.stringify(data);
  const lesson = lessonFor(data);
  assert.deepEqual(validateLessonSpec(lesson), []);
  assert.equal(JSON.stringify(data), before, 'lessonFor never mutates the data');
});

test('every prose placeholder resolves and no fact or prose prints "—"', () => {
  const lesson = lessonFor(data);
  const prose = [lesson.hook, ...lesson.intuition, lesson.intuitionNote, lesson.toy.intro, lesson.facts.framing, ...lesson.facts.prose, ...lesson.takeaways];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => r.claim)].forEach((text) => assert.ok(!fillText(text, data).includes('—'), text.slice(0, 40)));
});

test('rows 4–6 print the storyboard numbers from the data', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[3], /only 24 of its 93 layers run this softmax attention, the other 69 are linear-attention layers/);
  assert.match(rows[4], /from layer 4 on, each query reads only the 16 most relevant blocks of 128 past tokens/);
  assert.match(rows[5], /only 23 of its 92 layers are softmax attention \(3 linear : 1 full\); the rest are linear-attention layers/);
});

test('the toy starts where the animation ends', () => {
  assert.deepEqual({ ...INITIAL_STATE }, { query: 2, divisor: 2, causal: true, head: 'A' });
});

test('stage cells: a masked exp or weight prints 0, a masked score −∞, a pending cell nothing', () => {
  assert.deepEqual([cellText(-Infinity, 'exp'), cellText(-Infinity, 'weight'), cellText(-Infinity, 'scaled'), cellText(null, 'output'), cellText(Number.NaN, 'score')], ['0', '0', '−∞', '', '']);
});

test('the storyboard\'s three try-this prompts print with their numbers from math/attention.js and a named insight', () => {
  const items = tryThis();
  assert.equal(items.length, 3);
  assert.deepEqual(items.map(([, insight]) => insight), [
    'The mask is the only thing that makes attention causal.',
    '√d_head is a sharpness dial.',
    'Same tokens, different pattern: multi-head is several attention patterns at once.',
  ]);
  assert.equal(items[0][0], 'Keep query = sat₃ and switch the causal mask **off** → the row becomes [0.086, 0.635, 0.182, 0.097]: 0.097 of the weight now lands on "down", a token that comes *later*, and cat\'s share drops from 0.703 to 0.635 because the row must still sum to 1');
  assert.equal(items[1][0], 'Mask back on. Drag the divisor to **0.5** → [0.000, 0.993, 0.007, 0]; now to **8** → [0.259, 0.428, 0.313, 0]');
  assert.equal(items[2][0], 'Divisor back to 2. Set query = down₄ and flip head **A → B** → head A gives [0.114, 0.656, 0.129, 0.101] (most weight on "cat"); head B gives [0.129, 0.146, 0.578, 0.146] (most weight on "sat", the previous token). Choose **both** to see the two heatmaps side by side and the [1 × 8] concat row');
  assert.match(items[1][2], /see the mono line under the slider/);
});
