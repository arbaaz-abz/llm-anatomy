import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { tokenLoss, klDivergence } from '../math/lm.js';
import { opdTokenReward, multiTeacherLoss } from '../math/distill.js';
import { GROUP_TOY, buildGroup, verifyFinalAnswer, groupAdvantages } from '../math/grpo.js';
import { LESSON, lessonFor } from '../training/concepts/distillation/content.js';
import { checkWork, fixed, signed } from '../training/concepts/distillation/format.js';
import { toyView, modelFor, tryThis, INITIAL_STATE } from '../training/concepts/distillation/toy-view.js';
import { BELOW, FACT_ROWS } from '../training/concepts/distillation/facts.js';
import * as S from '../training/concepts/distillation/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS, STAND_IN } from './distillation-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const state = (patch) => ({ ...INITIAL_STATE, ...patch });

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order; the stand-in line is the storyboard\'s', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.equal(LESSON.animation.standIn, STAND_IN);
});

test('7 facts rows, and every placeholder on the page resolves against data/*.json', () => {
  const lesson = lessonFor(data);
  assert.equal(LESSON.facts.rows.length, 7);
  assert.equal(FACT_ROWS.length, 7);
  lesson.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  const prose = [lesson.hook, ...lesson.intuition, ...Object.values(BELOW).flat()];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 60)));
  [...prose.map((t) => fillText(t, data)), ...lesson.facts.rows.map((r) => fillText(r.claim, data)), ...Object.keys(BELOW).flatMap((i) => lesson.animation.belowFor(Number(i)))]
    .forEach((text) => assert.doesNotMatch(text, /[{}—]/, text.slice(0, 60)));
  assert.equal(lesson.animation.belowFor(0).length, 0);
});

test('dated text prints the data: years come from confirmed dates, counts through the formatters (X-1)', () => {
  const lesson = lessonFor(data);
  assert.match(fillText(Object.values(BELOW).flat()[2], data), /^DeepSeek-R1 \(2025\): about 800K SFT samples/);
  assert.match(fillText(lesson.facts.rows[5].claim, data), /^DeepSeek-R1 \(2025\) was fine-tuned on about 800K SFT samples/);
  assert.match(fillText(lesson.facts.rows[6].claim, data), /about 2\.3M reasoning traces/);
  assert.match(lesson.intuition[2], /Kimi K3 uses 9,/);
  assert.match(fillText(lesson.facts.rows[4].claim, data), /about an order of magnitude or more\.$/);
  const text = [lesson.hook, ...lesson.intuition, ...lesson.facts.rows.map((r) => fillText(r.claim, data)), ...Object.values(BELOW).flat().map((t) => fillText(t, data))].join(' ');
  assert.deepEqual(text.match(/\b(19|20)\d{2}\b/g)?.sort() ?? [], ['2025', '2025', '2026']);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1): none, it closes the Recipe', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('distillation')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(LESSON.links.next, []);
});

test('the math panel names the three hl terms and its note (d) matches the stage\'s KL terms', () => {
  const tex = LESSON.math.blocks.map((b) => b.tex).join('\n');
  ['t', 's', 'r'].forEach((l) => assert.match(tex, new RegExp(`\\\\htmlClass\\{hl-${l}\\}`)));
  assert.match(LESSON.math.notes[2], /0\.90 ln 2\.25 = 0\.730 · 0\.05 ln 0\.17 = −0\.090 · 0\.03 ln 0\.15 = −0\.057 · 0\.02 ln 0\.20 = −0\.032; sum 0\.551\./);
});

test('stage stand-ins: the loss, the KL and the rewards come from math/lm.js and math/distill.js (frames 2, 3, 5)', () => {
  near(S.TRACE_LOSS, tokenLoss(0.4));
  near(S.TRACE_LOSS, 0.9163);
  near(S.FORWARD_KL, klDivergence([0.9, 0.05, 0.03, 0.02], [0.4, 0.3, 0.2, 0.1]));
  near(S.FORWARD_KL, 0.5511);
  near(S.KL_TERMS_SUM, S.FORWARD_KL, 1e-12);
  [0.730, -0.090, -0.057, -0.032].forEach((term, i) => near(S.FORWARD_KL_TERMS[i], term));
  [0.8109, -1.7918, -1.8971, -1.6094].forEach((reward, i) => near(S.REWARDS[i], reward, 5e-5));
  near(S.EXPECTED_REWARD, -0.7535, 5e-5);
  near(S.EXPECTED_REWARD, S.REWARDS.reduce((sum, r, i) => sum + S.DEFAULT_STUDENT[i] * r, 0), 1e-12);
});

test('frame 6: row 2 and its GRPO advantage are rlvr-grpo\'s (GROUP_TOY, P3-R5); the teacher\'s grades land on 54', () => {
  const group = buildGroup(2, GROUP_TOY);
  assert.deepEqual([...S.ROW_2], group[1]);
  assert.deepEqual([...S.ROW_2], ['7', '×', '8', '=', '54']);
  assert.equal(S.ROW_2_ADVANTAGE, groupAdvantages(group.map((t) => verifyFinalAnswer(t, GROUP_TOY.target)))[1]);
  assert.equal(fixed(S.ROW_2_ADVANTAGE, 2), '−0.58');
  assert.deepEqual(S.ROW_2_REWARDS.map((r) => signed(r, 3)), ['0.000', '+0.054', '+0.031', '0.000', '−1.792']);
  assert.deepEqual(S.ROW_2_REWARDS.map((r) => signed(r, 2)), ['0.00', '+0.05', '+0.03', '0.00', '−1.79']);
  assert.equal(S.FOLLOWED_TOKEN_INDEX, 4);
  assert.equal(S.REWARD_MAX_ABS, 3, 'one reward scale on this page (P3-R18)');
});

test('"Check my work" for the default state is the storyboard text', () => {
  assert.equal(toyView(INITIAL_STATE).checkWork, CHECK_WORK);
  assert.equal(checkWork(modelFor(INITIAL_STATE)), CHECK_WORK);
});

test('"Check my work" is templated for the other methods and the mix', () => {
  assert.equal(toyView(state({ method: 'traces' })).checkWork, 'loss = −ln p_student(56) = −ln 0.40 = 0.916');
  assert.equal(toyView(state({ method: 'logits' })).checkWork, [
    'KL(teacher ‖ student)',
    '  = 0.90 ln(0.90/0.40) + 0.05 ln(0.05/0.30) + 0.03 ln(0.03/0.20) + 0.02 ln(0.02/0.10)',
    '  = 0.730 − 0.090 − 0.057 − 0.032 = 0.551',
  ].join('\n'));
  const mix = toyView(state({ student: 'near', teacher: 'mix' })).checkWork;
  assert.match(mix, /^reward\(54\) = ½ × \(−0\.336\) \+ ½ × \(.+\) = /);
  assert.match(mix, /expected reward = −\(½ KL\(student ‖ math\) \+ ½ KL\(student ‖ chat\)\)/);
  assert.match(toyView(state({ method: 'logits', teacher: 'mix' })).checkWork, /^KL\(teacher ‖ student\) = ½ KL\(math ‖ student\) \+ ½ KL\(chat ‖ student\)/);
});

test('toy outputs at the default state (storyboard §6)', () => {
  const view = toyView(INITIAL_STATE);
  assert.equal(view.sampledReward, '−1.792');
  assert.equal(view.sampledToken, '54');
  assert.equal(view.expectedReward, '−0.754');
  assert.equal(view.multiLoss, '0.754');
  assert.equal(view.traceLoss, '0.916');
  assert.equal(view.forwardKl, '0.551');
  assert.deepEqual(view.rewardRows.map((r) => `${r.token} ${r.value}`), ['56 +0.811', '54 −1.792', '48 −1.897', '63 −1.609']);
  assert.equal(view.teacherNote, '');
});

test('toy outputs equal the shared functions for the same inputs (Review Focus 4)', () => {
  const student = [0.4, 0.3, 0.2, 0.1];
  const teacher = [0.9, 0.05, 0.03, 0.02];
  const model = modelFor(INITIAL_STATE);
  assert.equal(model.traceLoss, tokenLoss(student[0]));
  assert.equal(model.forwardKl, klDivergence(teacher, student));
  assert.deepEqual(model.rewards, student.map((s, i) => opdTokenReward(teacher[i], s)));
  assert.equal(model.multiLoss, multiTeacherLoss(student, [teacher], [1]));
  near(model.expectedReward, -klDivergence(student, teacher), 1e-12);
  const mix = modelFor(state({ student: 'near', teacher: 'mix' }));
  assert.equal(mix.multiLoss, multiTeacherLoss(S.STUDENTS.near, [S.TEACHERS.math, S.TEACHERS.chat], [0.5, 0.5]));
  near(mix.expectedReward, -mix.multiLoss, 1e-12);
  assert.match(toyView(state({ teacher: 'mix' })).teacherNote, /weight ½ each/);
});

test('try this: the storyboard\'s numbers and insights, in the page\'s wording', () => {
  const items = tryThis().map(({ prompt, insight, rest }) => `${prompt} → Insight: ${insight}${rest}`);
  assert.deepEqual(items, TRY_THIS);
});

test('every sampled token and student: rewards, expected reward and the per-token grades (try-this 1 and 2)', () => {
  const at = (patch) => toyView(state(patch));
  assert.deepEqual([0, 1, 2, 3].map((i) => at({ sampled: i }).sampledReward), ['+0.811', '−1.792', '−1.897', '−1.609']);
  assert.equal(at({ student: 'near' }).expectedReward, '−0.013');
  assert.equal(at({ student: 'wrong' }).sampledReward, '−2.639');
  assert.equal(at({ student: 'wrong' }).expectedReward, '−1.909');
  assert.deepEqual(['math', 'chat', 'mix'].map((teacher) => at({ student: 'near', teacher }).multiLoss), ['0.013', '0.152', '0.082']);
});

test('toyView and modelFor reject unknown states and never mutate their inputs', () => {
  assert.throws(() => toyView(state({ method: 'x' })), RangeError);
  assert.throws(() => toyView(state({ student: 'x' })), RangeError);
  assert.throws(() => toyView(state({ teacher: 'x' })), RangeError);
  assert.throws(() => toyView(state({ sampled: 4 })), RangeError);
  assert.throws(() => toyView(state({ sampled: 1.5 })), RangeError);
  const before = JSON.stringify([INITIAL_STATE, S.STUDENTS, S.TEACHERS]);
  toyView(state({ teacher: 'mix', method: 'logits' }));
  assert.equal(JSON.stringify([INITIAL_STATE, S.STUDENTS, S.TEACHERS]), before);
});

test('formatters use a real minus and never print "−0.000"', () => {
  assert.equal(fixed(-1.7918, 3), '−1.792');
  assert.equal(fixed(-0.0001, 3), '0.000');
  assert.equal(signed(0.8109, 3), '+0.811');
  assert.equal(signed(0, 3), '0.000');
  assert.equal(signed(-1e-9, 2), '0.00');
});
