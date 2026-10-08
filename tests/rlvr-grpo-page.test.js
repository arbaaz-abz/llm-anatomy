import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../training/concepts/rlvr-grpo/content.js';
import { INITIAL_STATE, fixed, signed, signed2, cellNumber } from '../training/concepts/rlvr-grpo/format.js';
import { evaluate, divisionEffect, unbiasedStd, answersFor } from '../training/concepts/rlvr-grpo/model.js';
import { tryThis } from '../training/concepts/rlvr-grpo/try-this.js';
import { CAPTIONS, TRY_THIS } from './rlvr-grpo-expected.js';
import { GROUP_TOY, buildGroup, groupAdvantages, clippedSurrogate } from '../math/grpo.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('13 facts rows; every placeholder resolves against data/models.json', () => {
  assert.equal(LESSON.facts.rows.length, 13);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});

test('every prose placeholder resolves and nothing prints "—"', () => {
  const lesson = lessonFor(data);
  const prose = [lesson.hook, ...lesson.intuition, lesson.intuitionNote, lesson.toy.intro, lesson.facts.framing, ...lesson.math.notes, ...lesson.takeaways];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => r.claim)].forEach((text) => assert.ok(!fillText(text, data).includes('—'), text.slice(0, 40)));
});

test('dated numbers come from the data: the group size, the clip bounds and the compute share', () => {
  const lesson = lessonFor(data);
  assert.match(lesson.intuition[1], /GLM-5 uses 32\)/);
  assert.match(lesson.intuition[1], /post-training compute at >10% of pretraining compute, in the report's own accounting/);
  const rows = lesson.facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /IcePop-GRPO, ε_low 0\.2 and ε_high 0\.28, KL coefficient 0 .*group size 32/);
  assert.match(rows[3], /over 1,827 synthetic environments/);
  assert.match(rows[9], /ε_high 0\.26–0\.28/);
});

test('Next lists exactly the lessons that take this one as a prereq', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('rlvr-grpo')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('no model release year is printed (rule X-1): the fact rows carry none', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data)).join(' ');
  assert.equal(rows.match(/\b20\d\d\b/g), null);
});

test('the toy opens on the storyboard default', () => {
  assert.deepEqual({ ...INITIAL_STATE }, { k: 2, norm: true, agg: 'sample', epsHigh: 0.2, row: 0, token: 4 });
});

test('default group: rewards, advantages, stats, total push and 39 tokens (10 right, 29 wrong)', () => {
  const g = evaluate(INITIAL_STATE);
  assert.deepEqual(g.rewards, [1, 0, 0, 0, 1, 0, 0, 0]);
  assert.deepEqual(g.advantages.map(signed2), ['+1.73', '−0.58', '−0.58', '−0.58', '+1.73', '−0.58', '−0.58', '−0.58']);
  assert.equal(fixed(g.stats.mean, 3), '0.250');
  assert.equal(fixed(g.stats.std, 3), '0.433');
  assert.equal(fixed(g.totalPush, 2), '6.93');
  assert.equal(g.signal, true);
  assert.equal(g.rows.reduce((n, r) => n + r.tokens.length, 0), 39);
  const right = g.rows.filter((r) => r.reward === 1).reduce((n, r) => n + r.tokens.length, 0);
  assert.equal(right, 10);
});

test('the inspector for row 1\'s 56 (default selection)', () => {
  const t = evaluate(INITIAL_STATE).rows[0].tokenRows[4];
  assert.equal(t.text, '56');
  assert.equal(signed(t.advantage, 4), '+1.7321');
  assert.equal(fixed(t.weight, 4), '0.0250');
  assert.equal(signed(t.push, 4), '+0.0433');
  assert.deepEqual([fixed(t.sampled, 3), fixed(t.now, 3), fixed(t.ratio, 3), fixed(t.objective, 3)], ['0.200', '0.250', '1.250', '2.078']);
  assert.equal(t.clipped, true);
});

test('every readout equals math/grpo.js for the same inputs', () => {
  const answers = buildGroup(2, GROUP_TOY);
  const rewards = answers.map((a) => (a.at(-1) === '56' ? 1 : 0));
  const adv = groupAdvantages(rewards);
  const g = evaluate(INITIAL_STATE);
  assert.deepEqual(g.advantages, adv);
  const t = g.rows[0].tokenRows[4];
  assert.deepEqual({ objective: t.objective, clipped: t.clipped }, clippedSurrogate(1.25, adv[0], { epsLow: 0.2, epsHigh: 0.2 }));
});

test('the sum of advantages is zero for every k, and no spread at k = 0 and 8', () => {
  for (let k = 0; k <= 8; k += 1) {
    const g = evaluate({ ...INITIAL_STATE, k });
    assert.ok(Math.abs(g.advantages.reduce((s, a) => s + a, 0)) < 1e-12, `k=${k}`);
    assert.equal(g.signal, k !== 0 && k !== 8, `k=${k}`);
  }
  assert.equal(evaluate({ ...INITIAL_STATE, k: 8 }).totalPush, 0);
});

test('cell numbers: integers bare, anything else signed with 2 decimals', () => {
  assert.deepEqual([1, 0, 0.25, -0.5774, 1.7321].map(cellNumber), ['1', '0', '+0.25', '−0.58', '+1.73']);
  assert.equal(fixed(-0.00001, 3), '0.000');
  assert.throws(() => fixed(Number.NaN, 2), RangeError);
});

test('try-this 1 numbers: k = 1, norm off, and the division effect (1 / std)', () => {
  const lone = evaluate({ ...INITIAL_STATE, k: 1 });
  assert.deepEqual([signed2(lone.advantages[0]), signed2(lone.advantages[1]), fixed(lone.totalPush, 2)], ['+2.65', '−0.38', '5.29']);
  assert.equal(signed(evaluate({ ...INITIAL_STATE, k: 1, norm: false }).advantages[0], 3), '+0.875');
  assert.deepEqual([1, 2, 4].map((k) => { const d = divisionEffect(k); return [fixed(d.withStd, 2), fixed(d.withoutStd, 2)]; }), [['5.29', '1.75'], ['6.93', '3.00'], ['8.00', '4.00']]);
});

test('try-this 2 numbers: sample vs token aggregation', () => {
  const sample = evaluate(INITIAL_STATE).rows;
  const token = evaluate({ ...INITIAL_STATE, agg: 'token' }).rows;
  assert.deepEqual([sample[3].tokenRows[0].push, sample[5].tokenRows[0].push, sample[3].pushPerAnswer, sample[5].pushPerAnswer].map((v) => signed(v, 4)), ['−0.0090', '−0.0722', '−0.0722', '−0.0722']);
  assert.deepEqual([token[3].tokenRows[0].push, token[3].pushPerAnswer, token[5].pushPerAnswer].map((v) => signed(v, 4)), ['−0.0148', '−0.1184', '−0.0148']);
});

test('try-this 3 numbers: three hatched chips at 0.20, two at 0.28', () => {
  const hatched = (epsHigh) => evaluate({ ...INITIAL_STATE, epsHigh }).rows.flatMap((r, i) => r.tokenRows.map((t, j) => (t.clipped ? [i + 1, j + 1] : null)).filter(Boolean));
  assert.deepEqual(hatched(0.2), [[1, 5], [4, 5], [5, 5]]);
  assert.deepEqual(hatched(0.28), [[4, 5], [5, 5]]);
  assert.equal(fixed(evaluate({ ...INITIAL_STATE, epsHigh: 0.28 }).rows[0].tokenRows[4].objective, 3), '2.165');
});

test('the prefix `7 × 8 =` nets to zero push in the default group', () => {
  const g = evaluate(INITIAL_STATE);
  const prefix = g.rows.filter((r) => r.tokens.join(' ').startsWith('7 × 8 =')).map((r) => r.advantage);
  assert.equal(prefix.length, 4);
  assert.ok(Math.abs(prefix.reduce((s, a) => s + a, 0)) < 1e-12);
});

test('the unbiased-std note: 1.07 times larger, +1.62 instead of +1.73', () => {
  const { factor, advantage } = unbiasedStd(2);
  assert.equal(fixed(factor, 2), '1.07');
  assert.equal(signed2(advantage), '+1.62');
  const notes = LESSON.math.notes.join(' ');
  assert.match(notes, /std 1\.07× larger and every advantage 1\.07× smaller \(\+1\.62 instead of \+1\.73\)/);
  assert.match(notes, /Σ\|o_i\| = 39 tokens/);
});

test('try-this items are the storyboard\'s, with a named insight each', () => {
  assert.deepEqual(tryThis(), TRY_THIS);
});

test('takeaways, links and the stand-in line', () => {
  assert.equal(LESSON.takeaways.length, 3);
  assert.equal(LESSON.animation.standIn, 'The eight answers and the ratios are hand-picked stand-ins; a real policy samples them and a real trainer measures the ratios.');
  assert.equal(LESSON.links.further.length, 3);
});

test('inputs are not mutated: the pools, the data and the state', () => {
  const pools = JSON.stringify(GROUP_TOY);
  const dataBefore = JSON.stringify(data);
  const state = { ...INITIAL_STATE };
  evaluate(state);
  answersFor(3);
  lessonFor(data);
  assert.equal(JSON.stringify(GROUP_TOY), pools);
  assert.equal(JSON.stringify(data), dataBefore);
  assert.deepEqual(state, INITIAL_STATE);
});
