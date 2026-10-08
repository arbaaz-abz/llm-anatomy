import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { GROUP_TOY, buildGroup, groupAdvantages, verifyFinalAnswer } from '../math/grpo.js';
import { isCorrection, mismatchRatio, rolloutSchedule } from '../math/agentic.js';
import { LESSON, lessonFor } from '../training/concepts/agentic-rl/content.js';
import { CAPTIONS as PAGE_CAPTIONS } from '../training/concepts/agentic-rl/captions.js';
import * as N from '../training/concepts/agentic-rl/numbers.js';
import { factRows, stageText } from '../training/concepts/agentic-rl/facts.js';
import { fixed2, fixed3, signed3, push2, percent1, minutes, rowsText, carriedRows, wrapText, advantage3, ofTotal } from '../training/concepts/agentic-rl/format.js';
import { INITIAL_STATE, allTokens, maskedCount, schedule, timelineView, toyView, tokenAt, tryThis } from '../training/concepts/agentic-rl/toy-view.js';
import { keyStep } from '../training/concepts/agentic-rl/stage-select.js';
import { CAPTIONS, TRY_THIS } from './agentic-rl-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const filled = lessonFor(data);
const state = (patch) => ({ ...INITIAL_STATE, ...patch });
const readout = (view, name) => view.inspector.find(([n]) => n === name)[2];

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(filled), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.deepEqual([...PAGE_CAPTIONS], CAPTIONS);
});

test('10 facts rows; every placeholder resolves against data/*.json, has a source and nothing prints "—"', () => {
  const rows = factRows();
  assert.equal(rows.length, 10);
  assert.equal(LESSON.facts.rows.length, 10);
  rows.forEach((row, i) => {
    const f = fillClaim(row.claim, data);
    assert.deepEqual(f.missing, [], `row ${i + 1}`);
    assert.ok(f.sources.length > 0, `row ${i + 1} has a source`);
    assert.doesNotMatch(f.segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`);
  });
});

test('the fact rows print the data: GLM-5, DeepSeek-V3.2, Kimi K3, Mistral', () => {
  const text = (i) => fillText(factRows()[i].claim, data);
  assert.match(text(0), /^GLM-5: async decoupled \(slime\), TITO, double-sided IS; software-engineering environments: >10,000 \(9 languages\); also thousands of Docker terminal tasks/);
  assert.match(text(1), /IcePop mask, ρ ∈ \[1\/2, 2\]\); its KL coefficient is 0\.$/);
  assert.match(text(2), /^DeepSeek-V3\.2: weak\/zero KL \(math\).*; 1,827 synthetic environments; post-training compute >10% of pretraining compute\.$/);
  assert.match(text(4), /^Kimi K3: 9 specialist experts \(3 domains × 3 effort levels\); partial rollouts \(λ\), AgentEnv microVMs/);
  assert.match(text(5), /^Nemotron 3 Super: asynchronous GRPO over 21 environments, with a separate SWE-RL stage because/);
  assert.equal(text(7), 'Mistral Large 4: asynchronous RL producing about 33B tokens per day (16B of them trainable completion tokens) on about 3K GPUs.');
  assert.doesNotMatch(text(9), /\d{4}/, 'the FP16 row prints no year (X-1, P3-R14)');
});

test('the FP16 row cites the reported paper', () => {
  const f = fillClaim(factRows().at(-1).claim, data);
  assert.equal(f.reported, true);
  assert.deepEqual(f.sources, ['https://arxiv.org/abs/2510.26788']);
});

test('prose and stage text resolve and print no "—", and no year that is not confirmed (X-1)', () => {
  const lesson = filled;
  const prose = [lesson.hook, ...lesson.intuition, lesson.intuitionNote, lesson.animation.standIn, lesson.toy.intro, lesson.facts.framing, ...lesson.math.notes, ...lesson.takeaways];
  prose.forEach((t) => assert.ok(!t.includes('—'), t.slice(0, 40)));
  const stage = stageText(data);
  [stage.beta, ...stage.table].forEach((t) => assert.ok(!t.includes('—'), t));
  assert.match(lesson.intuition[3], /GLM-5's verifiable software-engineering environments number >10,000 \(9 languages\), DeepSeek-V3\.2 synthesized 1,827 environments, and Kimi K3 trains 9 separate experts\./);
  assert.doesNotMatch([...prose, ...factRows().map((r) => fillText(r.claim, data))].join(' '), /\b2025\b/);
});

test('stage text for frames 9–10 comes from the data', () => {
  const s = stageText(data);
  assert.equal(s.beta, 'β = 0 (GLM-5, Olmo 3) · "weak or zero for math" (DeepSeek-V3.2)');
  assert.deepEqual(s.table, [
    'GLM-5: SWE environments >10,000 (9 languages)',
    'DeepSeek-V3.2: 1,827 environments',
    'Kimi K3: 9 experts (3 domains: general, general agents, coding agents × low / high / max effort)',
    'Nemotron 3 Super: 21 environments',
    'Mistral: ~33B tokens produced per day by one run on ~3K GPUs, ~16B of them trainable completion tokens',
  ]);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('agentic-rl')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the group is rlvr-grpo\'s, row for row: 39 tokens, A = +1.73 / −0.58 (P3-R5)', () => {
  assert.deepEqual(N.ROWS, buildGroup(2, GROUP_TOY));
  assert.equal(N.TOKEN_COUNT, 39);
  assert.deepEqual([...N.REWARDS], N.ROWS.map((t) => verifyFinalAnswer(t, '56')));
  assert.deepEqual([...N.ADVANTAGES], groupAdvantages([1, 0, 0, 0, 1, 0, 0, 0]));
  near(N.ADVANTAGES[0], 1.7321);
  near(N.ADVANTAGES[1], -0.5774);
  assert.deepEqual(N.ROWS[3].slice(4, 8), ['48', ',', 'so', '48']);
  assert.equal(N.ROWS[N.FOLLOWED.row][N.FOLLOWED.pos], '48');
  assert.equal(N.ROWS[5].join(' '), '63');
  assert.equal(N.ROWS[6].join(' '), '7 × 8 = 58');
});

test('the stand-in episode of frames 1–3: 11 tokens, 8 trained, 3 masked', () => {
  assert.equal(N.EPISODE.length, 11);
  assert.equal(N.POLICY_TOKENS, 8);
  assert.equal(N.OBSERVATION_TOKENS, 3);
  assert.deepEqual(N.EPISODE.slice(7, 10).map((t) => t.text), ['<obs>', '56', '</obs>']);
  assert.match(PAGE_CAPTIONS[1], /Only the model's own 8 tokens/);
});

test('the engine and trainer probabilities: ρ = 1 except the five stand-in tokens, 34 of 39 agree', () => {
  const off = allTokens(INITIAL_STATE).filter((t) => Math.abs(t.rho - 1) > 1e-9);
  assert.deepEqual(off.map((t) => [t.row + 1, t.pos + 1, Number(t.rho.toFixed(2))]), [[1, 5, 1.1], [4, 8, 3.2], [5, 5, 0.9], [6, 1, 0.4], [7, 5, 2.4]]);
  assert.equal(N.AGREE_COUNT, 34);
  assert.match(fixed3(0.05), /0\.050/);
  allTokens(INITIAL_STATE).forEach((t) => assert.ok(t.engine > 0 && t.engine <= 1 && t.trainer > 0 && t.trainer <= 1));
});

test('the default inspector: row 4\'s second 48, every readout the storyboard prints', () => {
  const view = toyView(INITIAL_STATE);
  assert.equal(readout(view, 'inspector-token'), 'row 4, token 8: 48');
  assert.equal(readout(view, 'inspector-a'), '−0.577');
  assert.equal(readout(view, 'inspector-engine'), '0.050');
  assert.equal(readout(view, 'inspector-trainer'), '0.160');
  assert.equal(readout(view, 'inspector-rho'), '3.200');
  assert.equal(readout(view, 'inspector-weight'), '1.000');
  assert.equal(readout(view, 'inspector-masked'), 'no');
  assert.equal(readout(view, 'inspector-push'), '−0.577');
  assert.equal(view.masked, '0 of 39');
  assert.deepEqual([view.timeline.iteration, view.timeline.utilization, view.timeline.carried], ['16 min', '37.5%', 'none']);
});

test('every readout equals the shared function\'s output for the same inputs (Review Focus 4)', () => {
  const t = tokenAt(state({ correction: 'tis' }), 3, 7);
  const rho = mismatchRatio(0.16 / 0.05, { precision: 'bf16' });
  assert.equal(t.rho, rho);
  assert.deepEqual({ weight: t.weight, masked: t.masked }, isCorrection(rho, { mode: 'tis', cap: 2, band: [0.5, 2] }));
  assert.equal(t.push, groupAdvantages([1, 0, 0, 0, 1, 0, 0, 0])[3] * 2);
  const sched = rolloutSchedule(N.DURATIONS, { mode: 'partial', lambda: 0.75 });
  assert.deepEqual(schedule(state({ lambda: 'six' })), sched);
  assert.equal(timelineView(state({ lambda: 'six' })).utilization, percent1(sched.busy, 8 * sched.iterationTime));
});

test('try this 1: ignore → 1, −0.577; full IS → 3.200, −1.848; truncated → 2.000, −1.155; IcePop → masked, 0, 3 of 39', () => {
  const at = (correction) => tokenAt(state({ correction }), 3, 7);
  assert.deepEqual([at('none').weight, signed3(at('none').push)], [1, '−0.577']);
  assert.deepEqual([fixed3(at('full').weight), signed3(at('full').push)], ['3.200', '−1.848']);
  assert.deepEqual([fixed3(at('tis').weight), signed3(at('tis').push)], ['2.000', '−1.155']);
  assert.deepEqual([at('icepop').masked, signed3(at('icepop').push)], [true, '0']);
  assert.equal(toyView(state({ correction: 'icepop' })).masked, '3 of 39');
  assert.equal(maskedCount(state({ correction: 'full' })), 0);
});

test('try this 2: FP16 shrinks ρ (3.200 → 1.156, 2.400 → 1.116, 0.400 → 0.892) and masks 0 of 39 (BF16: 3)', () => {
  const fp = state({ correction: 'icepop', precision: 'fp16' });
  assert.deepEqual([[3, 7], [6, 4], [5, 0]].map(([r, p]) => fixed3(tokenAt(fp, r, p).rho)), ['1.156', '1.116', '0.892']);
  assert.equal(toyView(fp).masked, '0 of 39');
  assert.equal(maskedCount(state({ correction: 'icepop' })), 3);
  near(tokenAt(fp, 3, 7).rho, 1.1565, 5e-5);
});

test('try this 3: λ steps give 37.5% → 72.9% → 87.5% and carried none → rows 4 and 7 → rows 4, 5, 7 and 8', () => {
  const views = ['all', 'six', 'four'].map((lambda) => timelineView(state({ lambda })));
  assert.deepEqual(views.map((v) => v.utilization), ['37.5%', '72.9%', '87.5%']);
  assert.deepEqual(views.map((v) => v.carried), ['none', 'rows 4 and 7', 'rows 4, 5, 7 and 8']);
  assert.deepEqual(views.map((v) => v.iteration), ['16 min', '6 min', '4 min']);
});

test('the three try-this items are the storyboard\'s wording with the numbers the functions compute', () => {
  const items = tryThis();
  assert.equal(items.length, TRY_THIS.length);
  items.forEach((item, i) => {
    assert.equal(item.prompt, TRY_THIS[i].prompt);
    assert.equal(item.insight, TRY_THIS[i].insight);
  });
  assert.match(items[0].rest, /lets one token push 3\.2× as hard; truncation caps it \(at 2 in this toy\)/);
  assert.match(items[1].rest, /\(a reported fix\)/);
  assert.doesNotMatch(items.map((i) => i.prompt + i.rest).join(' '), /\b20\d\d\b/);
});

test('the toy shows its stand-in note and FP16 chip wording', () => {
  assert.match(toyView(INITIAL_STATE).note, /^The probabilities are hand-picked stand-ins\. The FP16 setting models rounding only; mismatch from MoE routing does not shrink with precision \(DeepSeek's fix for that is Keep Routing\)\.$/);
});

test('inputs are never mutated: the frozen group, durations and probabilities survive a full sweep', () => {
  const before = JSON.stringify([N.ROWS, N.DURATIONS, N.ENGINE_P, N.TRAINER_BF16, GROUP_TOY]);
  ['none', 'full', 'tis', 'icepop'].forEach((correction) => ['bf16', 'fp16'].forEach((precision) => toyView(state({ correction, precision }))));
  ['all', 'six', 'four'].forEach((lambda) => toyView(state({ lambda })));
  assert.equal(JSON.stringify([N.ROWS, N.DURATIONS, N.ENGINE_P, N.TRAINER_BF16, GROUP_TOY]), before);
  assert.throws(() => { N.DURATIONS.push(1); }, TypeError);
});

test('format.js: real minus, exact zero, one formatter per quantity', () => {
  assert.deepEqual([fixed3(-0.5774), fixed3(0.05), fixed3(0.0001), signed3(-1.84752), signed3(1.2), signed3(-0), signed3(0)], ['−0.577', '0.050', '0.000', '−1.848', '+1.200', '0', '0']);
  assert.deepEqual([fixed2(-0.5774), fixed2(3.2), push2(-0.5774), push2(0), push2(-0)], ['−0.58', '3.20', '−0.58', '0', '0']);
  assert.equal(advantage3(1.7321), '1.732');
  assert.deepEqual([percent1(48, 128), percent1(35, 48), percent1(28, 32)], ['37.5%', '72.9%', '87.5%']);
  assert.deepEqual([minutes(16), minutes(6), minutes(0.5)], ['16 min', '6 min', '30 s']);
  assert.equal(ofTotal(3, 39), '3 of 39');
  assert.deepEqual([[], [4], [4, 7], [4, 5, 7, 8]].map(rowsText), ['none', 'row 4', 'rows 4 and 7', 'rows 4, 5, 7 and 8']);
  assert.deepEqual(carriedRows([false, false, false, true, false, false, true, false]), [4, 7]);
  assert.deepEqual(wrapText('one two three four five', 9), ['one two', 'three', 'four five']);
  assert.deepEqual(wrapText('', 10), ['']);
});

test('stage-select keys: arrows step and hold at the ends, Home and End jump, others are ignored', () => {
  assert.deepEqual([keyStep('ArrowRight', 0, 3), keyStep('ArrowDown', 2, 3), keyStep('ArrowLeft', 0, 3), keyStep('ArrowUp', 2, 3), keyStep('Home', 2, 3), keyStep('End', 0, 3), keyStep('a', 1, 3)], [1, 2, 0, 1, 0, 2, null]);
});
