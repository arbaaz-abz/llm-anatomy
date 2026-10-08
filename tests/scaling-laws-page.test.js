import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../training/concepts/scaling-laws/content.js';
import { sci, perParam, powerText, lossText, sizeText, tokensText, savingText, isExtrapolated, sliderSizes, checkWork, coefficientText, superscript, INITIAL_STATE } from '../training/concepts/scaling-laws/format.js';
import { toyView, SERVED_OPTIONS } from '../training/concepts/scaling-laws/toy-view.js';
import { tryThis } from '../training/concepts/scaling-laws/try-this.js';
import { belowFor, SERVE_BASIS_LINE, DEFINITION_LINE } from '../training/concepts/scaling-laws/facts.js';
import { MODELS, BUDGETS, MUON_SCHEDULE, MUON_COEFFICIENTS, SINGULAR_VALUES, STAGE_BUDGET } from '../training/concepts/scaling-laws/numbers.js';
import { computeOptimal, isoFlopLoss, inferenceAwareOptimum, lifetimeFlops, tokensPerParam, newtonSchulzSingular } from '../math/scaling.js';
import { trainingFlops } from '../math/scale.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './scaling-laws-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const fact = (id, key) => data.models.entries.find((e) => e.id === id).facts[key];

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('8 facts rows; every placeholder in them and in the prose resolves against data/*.json', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 8);
  lesson.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  const prose = [lesson.hook, ...lessonFor(null).intuition.map((t) => t), lesson.facts.framing, ...lesson.takeaways];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
});

test('the filled lesson prints no "—"', () => {
  const lesson = lessonFor(data);
  const all = [lesson.hook, ...lesson.intuition, lesson.facts.framing, ...lesson.facts.rows.map((r) => fillText(r.claim, data)), ...lesson.takeaways, lesson.toy.intro, lesson.animation.standIn];
  all.forEach((text) => assert.ok(!text.includes('—'), text.slice(0, 50)));
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('scaling-laws')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the facts rows print the storyboard numbers, computed from the data (README lesson 35, X-3)', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /^DeepSeek-V4-Pro: 49B active, 33T tokens \(673 tokens per active parameter; 21 per total parameter of 1\.6T\)\.$/);
  assert.match(rows[1], /^DeepSeek-V4-Flash: 13B active, 284B total, 32T tokens \(2,462 per active parameter\)\.$/);
  assert.match(rows[2], /^Nemotron 3 Super: 12B active, 120B total, 25T tokens \(2,083 per active parameter\)\.$/);
  assert.match(rows[3], /^Llama 3\.1 405B \(2024\): dense, 405B parameters, 15\.6T tokens \(39 per parameter\)\.$/);
  assert.match(rows[4], /^Olmo 3: about 5\.9T tokens for its 7B and 32B dense models \(184–843 tokens per parameter\)\.$/);
  assert.match(rows[5], /claims about 2\.5× scaling efficiency over K2/);
  assert.match(rows[6], /DeepSeek-V4-Pro Muon .*GLM-5 "Muon Split", Kimi K3 "Per-Head Muon"; MiMo-V2-Flash AdamW and Nemotron 3 Super AdamW\.$/);
  assert.equal(rows[7], "DeepSeek-V4's Newton–Schulz schedule: 8 steps (3.4445, −4.7750, 2.0315) then 2 steps (2, −1.5, 0.5).");
});

test('Olmo\'s range and Olmo and Kimi rows carry their data flags: pretrain_tokens is reported', () => {
  const rows = lessonFor(data).facts.rows;
  assert.equal(fillClaim(rows[4].claim, data).reported, true);
  assert.equal(fillClaim(rows[5].claim, data).sources.length, 1, 'the Kimi row cites its source through |cite');
});

test('the stage\'s stand-in constants equal data/models.json (two sources for one fact)', () => {
  Object.values(MODELS).forEach((m) => {
    assert.equal(fact(m.id, 'pretrain_tokens').value, m.tokens, `${m.id} tokens`);
    assert.equal(fact(m.id, 'total_params').value, m.total, `${m.id} total`);
    if (m.id !== 'llama-3.1-405b') assert.equal(fact(m.id, 'active_params').value, m.active, `${m.id} active`);
  });
  assert.equal(fact('llama-3.1-405b', 'release_date').value.slice(0, 4), '2024', 'the "(2024)" on the Llama label is a confirmed year');
  assert.equal(fact('llama-3.1-405b', 'release_date').confidence, 'confirmed');
});

test('Muon\'s stand-in schedule equals deepseek-v4-pro.muon_ns_schedule', () => {
  const text = fact('deepseek-v4-pro', 'muon_ns_schedule').value.replaceAll('−', '-');
  const [bulk, finish] = [...text.matchAll(/(\d+) steps \(([^)]+)\)/g)].map((m) => ({ steps: Number(m[1]), coefficients: m[2].split(',').map(Number) }));
  assert.deepEqual(MUON_SCHEDULE, [...Array(bulk.steps).fill(bulk.coefficients), ...Array(finish.steps).fill(finish.coefficients)]);
  assert.equal(MUON_COEFFICIENTS.finishSteps, finish.steps);
});

test('"Check my work" for the default state is the storyboard text', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.equal(toyView(INITIAL_STATE).check, CHECK_WORK);
});

test('"Check my work" follows the state: 1T at 10^24 and the 10^22 optimum', () => {
  const big = checkWork({ C: 1e24, N: 1e12 }).split('\n');
  assert.equal(big[0], 'N = 1.00 × 10¹² parameters (active)');
  assert.equal(big[2], 'tokens per parameter = D ÷ N = 0.17');
  assert.match(big[4], /= 2\.013$/);
  const small = checkWork({ C: 1e22, N: computeOptimal(1e22).N }).split('\n');
  assert.match(small[1], /^D = 10²² ÷ \(6 × 9\.0\d × 10⁹\) = 1\.84 × 10¹¹ tokens$/);
  assert.match(small[4], /= 2\.141$/);
});

test('the toy opens on the storyboard default: every readout equals the math/ function it comes from', () => {
  const view = toyView(INITIAL_STATE);
  const best = computeOptimal(STAGE_BUDGET);
  assert.deepEqual(view.cols.this, { N: sizeText(best.N), D: tokensText(best.D), ratio: perParam(best.tokensPerParam), loss: lossText(best.loss), life: sci(lifetimeFlops(best.N, best.D, 0)), extrapolated: false });
  assert.deepEqual(view.cols.opt, view.cols.this);
  assert.deepEqual([view.cols.this.N, view.cols.this.D, view.cols.this.ratio, view.cols.this.loss, view.cols.this.life], ['95.9B', '1.74T', '18.1', '1.960', '1.00 × 10²⁴']);
  assert.equal(view.savingLine, '0.0% less lifetime compute than the compute-optimal model');
  assert.equal(view.plot.refY, null, 'with nothing served there is no same-loss guide');
});

test('with 100T served the cheapest column equals inferenceAwareOptimum and the saving is 58.5%', () => {
  const state = { ...INITIAL_STATE, Dinf: 1e14 };
  const view = toyView(state);
  const best = computeOptimal(state.C);
  const pick = inferenceAwareOptimum({ targetLoss: best.loss, inferenceTokens: 1e14 });
  assert.deepEqual(view.cols.cheap, { N: sizeText(pick.N), D: tokensText(pick.D), ratio: perParam(pick.tokensPerParam), loss: lossText(best.loss), life: sci(pick.total), extrapolated: false });
  assert.deepEqual([view.cols.cheap.N, view.cols.cheap.D, view.cols.cheap.ratio, view.cols.cheap.life, view.cols.opt.life], ['29.2B', '14.4T', '494.9', '8.36 × 10²⁴', '2.02 × 10²⁵']);
  assert.equal(view.saving, savingText(1 - pick.total / lifetimeFlops(best.N, best.D, 1e14)));
  assert.equal(view.saving, '58.5%');
  assert.deepEqual(view.plot.refY, { value: best.loss, label: 'same loss 1.960' });
  assert.ok(view.plot.markers.some((m) => m.label === 'cheapest'));
});

test('a size far from the optimum is tagged extrapolated (storyboard §6 validity line)', () => {
  assert.equal(toyView({ C: 1e24, N: 1e12, Dinf: 0 }).cols.this.extrapolated, true);
  assert.equal(toyView({ C: 1e24, N: 1e9, Dinf: 0 }).cols.this.ratio, '166,666.7');
  assert.equal(toyView(INITIAL_STATE).cols.this.extrapolated, false);
  assert.deepEqual([0.5, 1, 18, 10000, 10001].map(isExtrapolated), [true, false, false, false, true]);
});

test('the slider stops include the compute-optimal size of every budget, in order, inside 1B to just past 1T', () => {
  BUDGETS.forEach((C) => {
    const stops = sliderSizes(C);
    assert.ok(stops.includes(computeOptimal(C).N), `10^${Math.log10(C)}`);
    assert.deepEqual([...stops].sort((a, b) => a - b), stops);
    assert.equal(stops[0], 1e9);
    assert.ok(stops.at(-1) >= 1e12);
  });
});

test('the plot input holds its domains: every marker and point is inside them', () => {
  BUDGETS.forEach((C) => {
    const { plot } = toyView({ C, N: computeOptimal(C).N, Dinf: 1e14 });
    const [lo, hi] = plot.yAxis.domain;
    [...plot.series[0].points.map((p) => p[1]), ...plot.markers.map((m) => m.y), plot.refY.value].forEach((y) => assert.ok(y >= lo && y <= hi, `10^${Math.log10(C)}: ${y} in [${lo}, ${hi}]`));
  });
});

test('the storyboard\'s three try-this prompts print with their numbers from math/scaling.js and a named insight', () => {
  const items = tryThis();
  assert.deepEqual(items, TRY_THIS);
  assert.deepEqual(items.map(([, insight]) => insight), ['There is a valley.', 'The more a model will be used, the smaller and longer-trained it should be.', 'Compute-optimal scales parameters and tokens together.']);
});

test('the stage numbers: 6ND for frame 1, tokens per parameter for frame 8, the optimum for frame 5', () => {
  assert.equal(sci(trainingFlops({ params: 49e9, tokens: 33e12 })), '9.70 × 10²⁴');
  assert.equal(sci(trainingFlops({ params: 405e9, tokens: 15.6e12 })), '3.79 × 10²⁵');
  const ratios = [['deepseekV4Pro', 'active'], ['deepseekV4Flash', 'active'], ['nemotron', 'active'], ['llama31', 'active'], ['deepseekV4Pro', 'total'], ['nemotron', 'total'], ['deepseekV4Flash', 'total']]
    .map(([key, side]) => perParam(tokensPerParam(MODELS[key].tokens, MODELS[key][side]), 0));
  assert.deepEqual(ratios, ['673', '2,462', '2,083', '39', '21', '208', '113']);
  assert.deepEqual([1e22, 1e24, 1e26].map((C) => { const o = computeOptimal(C); return [sizeText(o.N), tokensText(o.D), perParam(o.tokensPerParam)]; }),
    [['9.05B', '184B', '20.4'], ['95.9B', '1.74T', '18.1'], ['1.02T', '16.4T', '16.1']]);
});

test('frame 2 and 3 numbers: three splits of 10^24 and the six fitted losses', () => {
  assert.deepEqual([1e9, 96e9, 1e12].map((n) => { const r = isoFlopLoss(1e24, n); return [tokensText(r.D), perParam(r.tokensPerParam, 0)]; }), [['167T', '166,667'], ['1.74T', '18'], ['167B', '0.17']]);
  assert.deepEqual([1e9, 1e10, 3e10, 96e9, 3e11, 1e12].map((n) => lossText(isoFlopLoss(1e24, n).loss)), ['2.187', '2.008', '1.972', '1.960', '1.972', '2.013']);
});

test('frames 6 and 7: serving 100T tokens and the over-trained model', () => {
  const best = computeOptimal(1e24);
  const pick = inferenceAwareOptimum({ targetLoss: best.loss, inferenceTokens: 1e14 });
  assert.equal(sci(lifetimeFlops(best.N, best.D, 1e14) - trainingFlops({ params: best.N, tokens: best.D })), '1.92 × 10²⁵');
  assert.equal(sci(lifetimeFlops(best.N, best.D, 1e14)), '2.02 × 10²⁵');
  assert.deepEqual([sci(pick.N), sci(pick.D), sci(pick.train), sci(pick.serve), sci(pick.total)], ['2.92 × 10¹⁰', '1.44 × 10¹³', '2.53 × 10²⁴', '5.83 × 10²⁴', '8.36 × 10²⁴']);
});

test('frames 9 and 10: the singular values and the Newton–Schulz trace', () => {
  const trace = newtonSchulzSingular(SINGULAR_VALUES, MUON_SCHEDULE);
  assert.equal(trace.length, 11);
  assert.deepEqual(trace[0].map((v) => v.toFixed(3)), ['0.995', '0.100']);
  assert.deepEqual(trace[1].map((v) => v.toFixed(3)), ['0.705', '0.338']);
  assert.deepEqual(trace[2].map((v) => v.toFixed(3)), ['1.109', '0.989']);
  assert.deepEqual(trace[8].map((v) => v.toFixed(3)), ['1.092', '1.103']);
  assert.deepEqual(trace[9].map((v) => v.toFixed(3)), ['1.007', '1.009']);
  assert.deepEqual(trace[10].map((v) => v.toFixed(4)), ['1.0000', '1.0000']);
  assert.deepEqual(trace[10].map((v) => v.toFixed(6)), ['1.000027', '1.000047']);
  assert.equal((SINGULAR_VALUES[0] ** 2 + SINGULAR_VALUES[1] ** 2).toFixed(2), '9.09');
  assert.equal(coefficientText(MUON_COEFFICIENTS.bulk, 4), '(3.4445, −4.7750, 2.0315)');
  assert.equal(coefficientText(MUON_COEFFICIENTS.finish), '(2, −1.5, 0.5)');
});

test('formatters: scientific form, powers of ten, one decimal for ratios, a value below 1 keeps two figures', () => {
  assert.deepEqual([sci(9.586e10), sci(1.739e12), powerText(1e24), powerText(1.5e24), superscript(-3)], ['9.59 × 10¹⁰', '1.74 × 10¹²', '10²⁴', '1.50 × 10²⁴', '⁻³']);
  assert.deepEqual([perParam(18.14), perParam(166666.67), perParam(0.1667), perParam(0.0017), perParam(2083.33, 0)], ['18.1', '166,666.7', '0.17', '0.0017', '2,083']);
  assert.deepEqual([savingText(0), savingText(-0.001), savingText(0.0159), savingText(0.585)], ['0.0%', '0.0%', '1.6%', '58.5%']);
  assert.throws(() => sci(0), RangeError);
  assert.deepEqual(SERVED_OPTIONS.map((o) => o.label), ['0', '1T', '10T', '100T', '1,000T']);
});

test('the lines under frames 6 and 8 are the basis and the definition; other frames print none', () => {
  assert.deepEqual(belowFor(5), [SERVE_BASIS_LINE]);
  assert.deepEqual(belowFor(7), [DEFINITION_LINE]);
  assert.deepEqual([0, 1, 2, 3, 4, 6, 8, 9].map(belowFor), Array(8).fill([]));
  assert.match(DEFINITION_LINE, /pretraining tokens ÷ active parameters/);
});

test('pure view code never mutates its inputs or the data', () => {
  const before = JSON.stringify(data);
  const state = Object.freeze({ ...INITIAL_STATE, Dinf: 1e13 });
  toyView(state);
  tryThis();
  lessonFor(data);
  assert.equal(JSON.stringify(data), before);
  assert.deepEqual({ ...state }, { ...INITIAL_STATE, Dinf: 1e13 });
});
