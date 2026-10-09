import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { stepTime, RUNNING_EXAMPLE, TOY_REQUESTS } from '../math/serving.js';
import { simulateStatic, simulateContinuous, scheduleTokens } from '../math/batching.js';
import { LESSON, lessonFor, BELOW } from '../serving/concepts/batching/content.js';
import { factRows } from '../serving/concepts/batching/facts.js';
import * as F from '../serving/concepts/batching/format.js';
import * as M from '../serving/concepts/batching/model.js';
import { toyView, tryThis, scenario, laneSummary, INITIAL_STATE, TOY_LIMITS } from '../serving/concepts/batching/toy-view.js';
import { TITLES } from '../serving/concepts/batching/numbers.js';
import { CAPTIONS, TRY_THIS, STAGE_TEXT } from './batching-expected.js';

const read = async (p) => JSON.parse(await readFile(new URL(p, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const R = M.requestsFor();
const LONG = M.requestsFor({ dPrompt: 4096 });
const stat = M.stepLane({ kind: 'static', requests: R });
const cont = M.stepLane({ kind: 'continuous', requests: R });

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim; caption 8 ratio and caption 9 numbers come from the model', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  const wide = M.msLane({ kind: 'continuous', requests: LONG });
  assert.equal(F.slowdown(M.longestStepMs(wide), M.stepMs(cont.schedule[1])), '19.9');
  assert.equal(F.ms(M.longestStepMs(wide)), '290 ms');
});
test('6 facts rows resolve; the filled lesson has no dash and no unfilled placeholder', () => {
  assert.equal(LESSON.facts.rows.length, 6);
  factRows(data).forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  assert.match(fillClaim(factRows(data)[3].claim, data).segments.map((s) => s.text).join(''), /up to 2\.6× .* up to 6\.9× /);
  const filled = lessonFor(data);
  [filled.hook, ...filled.intuition, ...[0, 6, 8, 9].flatMap((i) => filled.animation.belowFor(i))].forEach((t) => assert.doesNotMatch(t, /[{}—]/));
  assert.match(filled.intuition[0], /wastes 693 pad tokens/);
  assert.match(filled.intuition[1], /OSDI 2022/);
  assert.equal(Object.keys(BELOW).length, 4);
});
test('Next lists exactly the lessons that take this one as a prereq; titles printed on the stage match concepts.json', () => {
  assert.deepEqual([...LESSON.links.next].sort(), graph.concepts.filter((c) => c.prereqs.includes('batching')).map((c) => c.slug).sort());
  const title = (slug) => graph.concepts.find((c) => c.slug === slug).title;
  assert.equal(TITLES.pagedAttention, title('paged-attention'));
  assert.equal(TITLES.prefillDecode, title('prefill-decode'));
});
test('step lanes equal the simulators and continuous with budget Infinity equals scheduleTokens', () => {
  assert.deepEqual(stat.sim, simulateStatic({ requests: R, seats: 3 }));
  assert.deepEqual(cont.sim, simulateContinuous({ requests: R, seats: 3 }));
  const norm = (s) => s.map((e) => ({ step: e.step, decode: [...e.decode].sort(), prefill: e.prefill, tokens: e.tokens }));
  assert.deepEqual(norm(cont.schedule), norm(scheduleTokens({ requests: R, seats: 3 })));
  assert.deepEqual(laneSummary(stat).lastStep, stat.sim.lastStep);
  assert.equal(laneSummary(cont).busy, cont.sim.busySeatSteps);
});
test('every toy state builds, keeps its invariants and never mutates the requests', () => {
  Object.freeze(TOY_REQUESTS);
  for (const seats of TOY_LIMITS.seats) for (let c = 2; c <= 10; c += 1) for (const dPrompt of TOY_LIMITS.prompts) for (const budget of TOY_LIMITS.budgets) {
    const sc = scenario({ seats, cOutput: c, dPrompt, budget });
    const [s, k] = [laneSummary(sc.lanes.static), laneSummary(sc.lanes.continuous)];
    assert.ok(s.busy / s.seatSteps <= 1 && k.busy / k.seatSteps <= 1);
    if (budget === 0 || dPrompt === 6) assert.ok(k.lastStep <= s.lastStep, `${seats} ${c} ${dPrompt}`);
  }
  assert.throws(() => toyView({ ...INITIAL_STATE, seats: 5 }), RangeError);
  assert.throws(() => toyView({ ...INITIAL_STATE, budget: 7 }), RangeError);
});
test('the toy prints the storyboard\'s default and each try-this state', () => {
  const v = toyView(INITIAL_STATE);
  assert.deepEqual(v.summaryRows.map((r) => r.cells.map((c) => c.value)), [['10', '6'], ['57.6%', '90.5%'], ['1.36', '2.14']]);
  assert.equal(v.timingRows[0].value, '14.7 ms');
  assert.equal(v.showBudget, false);
  assert.equal(toyView({ ...INITIAL_STATE, seats: 4 }).summaryRows[1].cells[1].value, '67.9%');
  assert.equal(toyView({ ...INITIAL_STATE, cOutput: 10 }).summaryRows[1].cells.map((c) => c.value).join(), '51.1%,69.7%');
  assert.deepEqual(tryThis().map((t) => `${t.prompt} → Insight: ${t.insight}${t.rest}`), TRY_THIS);
});
test('stage readouts are the storyboard\'s', () => {
  assert.equal(F.doneLine(stat.rows, 7), STAGE_TEXT.frame2Done);
  assert.equal(F.idleLine(stat.rows, 7, M.idleAt), STAGE_TEXT.frame3Idle);
  assert.equal(F.busyLine(stat.sim, M.busyAt(stat.sim, 11), true), STAGE_TEXT.frame4Busy);
  assert.equal(M.busyAt(stat.sim, 7), 15);
  const d = cont.rows.find((r) => r.id === 'D');
  assert.equal(F.admittedLine(d, M.previousHolder(cont.rows, d), 7), STAGE_TEXT.frame5Admitted);
  assert.equal(F.summaryLine(stat.sim), STAGE_TEXT.frame6Static);
  assert.equal(F.summaryLine(cont.sim), STAGE_TEXT.frame6Continuous);
  assert.equal(F.summaryLine(M.stepLane({ kind: 'continuous', requests: R, seats: 4 }).sim), STAGE_TEXT.frame10Four);
  const mixed = cont.schedule[3];
  const t = stepTime({ ...RUNNING_EXAMPLE, tokens: mixed.tokens, seqs: mixed.decode.length, context: 16 }).timeS;
  assert.equal(F.mixedLine({ prefillTokens: 6, decoders: 2, tokens: mixed.tokens, timeS: t }), STAGE_TEXT.frame7Mixed);
  const wide = M.msLane({ kind: 'continuous', requests: LONG });
  assert.equal(F.stepLine(wide, 3), STAGE_TEXT.frame8Step);
  assert.equal(F.timesLine({ a: M.timing(wide, 'A').doneMs, c: M.timing(wide, 'C').doneMs, d: M.timing(wide, 'D').firstTokenMs }), STAGE_TEXT.frame8Times);
  assert.equal(F.shortNote(23, M.stepMs(cont.schedule[0]), M.stepMs(cont.schedule[1])), STAGE_TEXT.frame8Note);
  const small = M.msLane({ kind: 'continuous', requests: LONG, budget: 512 });
  assert.equal(F.budgetLine(512, M.longestStepMs(small)), STAGE_TEXT.frame9Budget);
  assert.equal(`${F.changedLine('A', M.timing(small, 'A').doneMs, M.timing(wide, 'A').doneMs)} · ${F.changedLine('C', M.timing(small, 'C').doneMs, M.timing(wide, 'C').doneMs)}`, STAGE_TEXT.frame9Done);
  assert.equal(F.firstTokenLine(M.timing(small, 'D').firstTokenMs, M.timing(wide, 'D').firstTokenMs), STAGE_TEXT.frame9First);
  assert.equal(F.slicesLine(small.schedule.flatMap((e) => e.prefill.filter((p) => p.id === 'D').map((p) => p.tokens))), 'D\'s prompt in slices: 510, 510, 511, 511, 512, 512, 512, 512, 6');
});
test('model: rows, seats, scrubbing purity and errors', () => {
  const d = stat.rows.find((r) => r.id === 'D');
  assert.deepEqual([d.seat, d.admitted, d.done, d.waited], [0, 7, 10, 6]);
  assert.equal(cont.rows.find((r) => r.id === 'D').seat, 1);
  assert.deepEqual(M.stepLane({ kind: 'static', requests: R }), stat);
  assert.equal(M.laneRows.length, 1);
  assert.throws(() => M.laneRows({ schedule: [], requests: R, kind: 'static', seats: 3, edges: [0] }), RangeError);
  assert.equal(F.steps(1), '1 step');
  assert.equal(F.laneTitle('continuousN', 4), 'continuous, 4 seats');
  assert.equal(F.idleLine([{ id: 'X', waited: 1, arrives: 2, admitted: 3 }], 0, () => 0), 'X waiting step 2');
  assert.equal(F.busyLine(stat.sim, 3, false), 'busy 3 of 33 seat-steps');
  assert.equal(M.widenedLane({ requests: LONG, shortRequests: R, blend: 1 }).edges.at(-1), M.msLane({ kind: 'continuous', requests: LONG }).edges.at(-1));
});
