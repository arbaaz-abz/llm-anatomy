// The sft page: lesson spec, verbatim captions, facts resolving against data/*.json, the toy's view model and Check my work.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { lossMask, maskSummary } from '../math/sft.js';
import { LESSON, lessonFor } from '../training/concepts/sft/content.js';
import { FACT_ROWS, BELOW } from '../training/concepts/sft/facts.js';
import { checkWork, INITIAL_STATE, kindRows, selectionText } from '../training/concepts/sft/format.js';
import { view, tryThis, chipsFor } from '../training/concepts/sft/toy-view.js';
import { SEGMENTS, TOTAL_TOKENS, FOLLOWED, ERROR_TOKENS, OBS_TOKENS, CALL_TOKENS } from '../training/concepts/sft/numbers.js';
import { neighbour } from '../training/concepts/sft/toy-dom.js';
import { layoutTranscript } from '../training/concepts/sft/stage.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_NO_PROMPT_MASK, CHECK_WORK_NO_OBSERVATION_MASK, CHECK_WORK_MASK_ERROR, CHECK_WORK_NOTHING_MASKED, TRY_THIS, DEFAULT_READOUTS } from './sft-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('ten facts rows; every placeholder resolves against data/*.json; no unfilled dash', () => {
  assert.equal(LESSON.facts.rows.length, 10);
  FACT_ROWS.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  lessonFor(data).facts.rows.forEach((row, i) => assert.doesNotMatch(row.claim, /—/, `row ${i + 1}`));
});
test('dated text under the stage, in the intuition and in the notes fills from data', () => {
  const lesson = lessonFor(data);
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 50)));
  const texts = [lesson.hook, ...lesson.intuition, ...lesson.math.notes, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  assert.match(lesson.intuition[2], /DeepSeek-R1 \(2025\) used about 800K examples; Olmo 3 about 2\.3M reasoning traces \(reported\)/);
  assert.match(lesson.animation.belowFor(2)[0], /Non-think \/ Think High \/ Think Max/);
  assert.match(lesson.animation.belowFor(7)[0], /about 800K SFT examples\. Olmo 3 used about 2\.3M reasoning traces \(reported\)/);
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('sft')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('no year is printed except from a confirmed release_date (X-1)', () => {
  const years = (t) => [...t.matchAll(/\((\d{4})\)/g)].map((m) => m[1]);
  const texts = [lessonFor(data).intuition.join(' '), ...FACT_ROWS.map((r) => fillText(r.claim, data)), ...BELOW.flat().map((t) => fillText(t, data))];
  texts.flatMap(years).forEach((y) => assert.ok(['2025', '2026'].includes(y), y));
  for (const [id, key] of [['deepseek-r1', 'release_date'], ['glm-5', 'release_date']]) {
    const fact = data.models.entries.find((e) => e.id === id).facts[key];
    assert.ok(fact.value && fact.confidence === 'confirmed', `${id}.${key} is confirmed`);
  }
});

test('the transcript is 26 tokens in eight segments, and the index constants point at the right tokens', () => {
  assert.equal(SEGMENTS.length, 8);
  const flat = SEGMENTS.flatMap((s) => s.tokens);
  assert.equal(flat.length, TOTAL_TOKENS);
  assert.equal(flat[FOLLOWED], '56');
  assert.equal(FOLLOWED, 24);
  assert.deepEqual(ERROR_TOKENS.map((i) => flat[i]).join(' '), '7 × 8 = 54');
  assert.equal(CALL_TOKENS.map((i) => flat[i]).join(' '), '<call> calc(7*8) </call>');
  assert.equal(OBS_TOKENS.map((i) => flat[i]).join(' '), '<obs> 56 </obs>');
  assert.equal(layoutTranscript().length, 26);
  assert.equal(new Set(layoutTranscript().map((c) => c.line)).size, 6);
});

test('Check my work: the default and every toggle state equal the storyboard text', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.equal(checkWork({ ...INITIAL_STATE, maskPrompt: false }), CHECK_WORK_NO_PROMPT_MASK);
  assert.equal(checkWork({ ...INITIAL_STATE, maskObservation: false }), CHECK_WORK_NO_OBSERVATION_MASK);
  assert.equal(checkWork({ ...INITIAL_STATE, maskError: true }), CHECK_WORK_MASK_ERROR);
  assert.equal(checkWork({ ...INITIAL_STATE, maskPrompt: false, maskObservation: false }), CHECK_WORK_NOTHING_MASKED);
});

test('toy view: the default prints the animation\'s numbers, equal to math/sft.js for the same inputs', () => {
  const v = view(INITIAL_STATE, data);
  const s = maskSummary(SEGMENTS);
  assert.deepEqual([v.trained, v.masked, v.share, v.selection], [DEFAULT_READOUTS.trained, DEFAULT_READOUTS.masked, DEFAULT_READOUTS.share, DEFAULT_READOUTS.selection]);
  assert.equal(v.trained, `${s.trained} of ${s.total}`);
  assert.equal(v.checkWork, CHECK_WORK);
  assert.deepEqual(v.kinds.map((r) => r.text), ['0/2', '0/6', '10/10', '5/5', '0/3']);
  assert.deepEqual(chipsFor(INITIAL_STATE).map((c) => c.trained), lossMask(SEGMENTS));
  assert.equal(v.chips.filter((c) => c.trained).length, 15);
});
test('toy view: the three try-this states print the storyboard counts', () => {
  const at = (patch) => view({ ...INITIAL_STATE, ...patch }, data);
  assert.deepEqual([at({ maskPrompt: false }).trained, at({ maskPrompt: false }).share], ['23 of 26', '88.5%']);
  assert.deepEqual([at({ maskObservation: false }).trained, at({ maskObservation: false }).share], ['18 of 26', '69.2%']);
  assert.deepEqual([at({ maskError: true }).trained, at({ maskError: true }).share, at({ maskError: true }).masked], ['10 of 26', '38.5%', '16']);
  assert.deepEqual(kindRows({ ...INITIAL_STATE, maskError: true }).map((r) => r.text), ['0/2', '0/6', '10/10', '0/5', '0/3']);
  assert.deepEqual(tryThis(), TRY_THIS);
});
test('template chips change only the tag text, never the counts', () => {
  const base = view(INITIAL_STATE, data);
  for (const template of ['deepseek', 'harmony']) {
    const v = view({ ...INITIAL_STATE, template }, data);
    assert.equal(v.trained, base.trained);
    assert.equal(v.checkWork, base.checkWork);
    assert.deepEqual(v.chips.map((c) => c.trained), base.chips.map((c) => c.trained));
    assert.notDeepEqual(v.chips.map((c) => c.text), base.chips.map((c) => c.text));
    assert.doesNotMatch(v.templateNote, /[{}]/);
  }
  assert.match(view({ ...INITIAL_STATE, template: 'deepseek' }, data).templateNote, /\|DSML\| XML tool calls/);
  assert.match(view({ ...INITIAL_STATE, template: 'harmony' }, data).templateNote, /harmony \(System > Developer/);
});
test('selecting a chip says what it is and whether it is trained', () => {
  const chips = chipsFor({ ...INITIAL_STATE, selected: 0 });
  assert.equal(selectionText(chips[0], chips[0].trained), 'Selected "<user>", a template tag: masked (context only).');
  assert.equal(view({ ...INITIAL_STATE, selected: 10 }, data).selection, 'Selected "×", a token of the kept mistake: trained.');
});
test('arrow keys move through the chips and between lines', () => {
  const chips = chipsFor(INITIAL_STATE);
  assert.equal(neighbour(chips, 0, 'ArrowLeft'), 0);
  assert.equal(neighbour(chips, 3, 'ArrowRight'), 4);
  assert.equal(neighbour(chips, 25, 'ArrowRight'), 25);
  assert.equal(neighbour(chips, 5, 'Home'), 0);
  assert.equal(neighbour(chips, 5, 'End'), 25);
  assert.equal(neighbour(chips, 5, 'x'), 5);
  assert.equal(chips[neighbour(chips, 1, 'ArrowDown')].line, chips[1].line + 1);
  assert.equal(neighbour(chips, 0, 'ArrowUp'), 0);
  assert.equal(neighbour(chips, 25, 'ArrowDown'), 25);
});
test('nothing here mutates its inputs', () => {
  const before = JSON.stringify(SEGMENTS);
  const state = Object.freeze({ ...INITIAL_STATE });
  view(state, data); checkWork(state); tryThis(); chipsFor(state);
  assert.equal(JSON.stringify(SEGMENTS), before);
});
