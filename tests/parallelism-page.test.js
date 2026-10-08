import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { formatBytes, formatCount } from '../math/core.js';
import { ROUTER_TOY, routeTopK } from '../math/moe.js';
import { PRESETS, paramBreakdown } from '../math/params.js';
import { gpuCount, pipelineBubble, pipelineSchedule, ringAllReduceBytes, allToAllBytes } from '../math/parallel.js';
import { TRAINING_RECIPES, zeroPerGpuBytes, trainingBytesPerParam } from '../math/training-memory.js';
import { M_SAT } from '../architecture/concepts/decoder-anatomy/numbers.js';
import { LESSON, lessonFor } from '../training/concepts/parallelism/content.js';
import { BELOW, factRows } from '../training/concepts/parallelism/facts.js';
import { checkWork, formatStateGB, lanesFromGrid, pipelineView, presetFor, stateView, bytesExact, int, INITIAL_STATE } from '../training/concepts/parallelism/format.js';
import { toyView, tryThis } from '../training/concepts/parallelism/toy-view.js';
import * as N from '../training/concepts/parallelism/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './parallelism-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;

test('the lesson spec is complete', () => assert.deepEqual(validateLessonSpec(LESSON), []));
test('the lesson filled with real data is complete too', () => assert.deepEqual(validateLessonSpec(lessonFor(data)), []));
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('8 facts rows, every placeholder resolves against data', () => {
  assert.equal(LESSON.facts.rows.length, 8);
  factRows().forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});

test('dated text: hook, intuition and the notes under the stage fill from data, nothing missing, no unfilled dash', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, lesson.facts.framing, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(lesson.hook, /^A 405B model needs 6\.48 TB of training state and an H100 holds 80 GB \(nominal\)\. When you spread it over 8,192 GPUs/);
  assert.match(lesson.intuition[2], /8-way tensor, 16-way pipeline and 64-way data parallelism: 8 × 16 × 64 = 8,192 GPUs/);
  assert.match(lesson.intuition[2], /which comes to 7\.02 GB of state with optimizer states and gradients sharded .* its 7 tensor partners, .* its 63 data-parallel peers/);
  assert.match(lesson.animation.belowFor(10)[0], /^Llama 3\.1 405B \(2024 paper\): TP8\/CP1\/PP16\/DP64 on 8,192/);
  assert.equal(lesson.animation.belowFor(42).length, 0);
});

test('X-1: the only years printed are confirmed release dates (Kimi K3 prints none)', () => {
  ['llama-3.1-405b', 'deepseek-v3', 'kimi-k2', 'deepseek-v4-pro'].forEach((id) => assert.equal(data.models.entries.find((e) => e.id === id).facts.release_date.confidence, 'confirmed', id));
  assert.doesNotMatch(factRows().find((r) => /Kimi K3/.test(r.claim)).claim, /release_date|\b20\d\d\b/);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1), cross-track ones included', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('parallelism')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(dependents.sort(), ['cluster-topology', 'disaggregation']);
});

test('"Check my work" is the storyboard text for the default state', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.equal(toyView(INITIAL_STATE).checkWork, CHECK_WORK);
});

test('check my work fills the same template for other states; an unsharded part prints no ÷ dp', () => {
  const lines = checkWork({ ...INITIAL_STATE, stages: 8, micro: 32, tp: 1, cp: 1, pp: 1, dp: 1, zero: 0 }).split('\n');
  assert.equal(lines[0], 'bubble  = (8 − 1) ÷ (32 + 8 − 1) = 7 ÷ 39 = 17.9%');
  assert.equal(lines[1], 'GPUs    = 1 × 1 × 1 × 1 = 1');
  assert.equal(lines[3], '  weights 405B × 2 B = 810 GB · gradients 405B × 2 B = 810 GB');
  assert.equal(lines[4], '  optimizer 405B × 12 B = 4,860 GB · total 6,480 GB');
  const z3 = checkWork({ ...INITIAL_STATE, zero: 3 }).split('\n');
  assert.match(z3[3], /^  weights 3\.164B × 2 B ÷ 64 = 0\.10 GB · gradients 3\.164B × 2 B ÷ 64 = 0\.10 GB$/);
  const z1 = checkWork({ ...INITIAL_STATE, zero: 1 }).split('\n');
  assert.match(z1[3], /gradients 3\.164B × 2 B = 6\.33 GB$/);
  assert.match(z1[4], /total 13\.25 GB$/);
});

test('every toy readout equals its math function for the same inputs (Review Focus 4)', () => {
  const state = { ...INITIAL_STATE, schedule: '1f1b', stages: 8, micro: 16, tp: 4, cp: 2, pp: 8, dp: 32, zero: 3 };
  const v = toyView(state);
  const sched = pipelineSchedule({ schedule: '1f1b', stages: 8, microBatches: 16 });
  assert.equal(v.columns, sched.columns);
  assert.equal(v.stepLength, String(sched.columns));
  assert.equal(v.peak, String(sched.peakInFlight[0]));
  assert.equal(v.gpus, int(gpuCount({ tp: 4, cp: 2, pp: 8, dp: 32 })));
  const parts = zeroPerGpuBytes({ params: 405e9, recipe: TRAINING_RECIPES.adam, stage: 3, dp: 32, modelShards: 32 });
  assert.equal(v.statePerGpu, formatStateGB(parts.total));
  assert.ok(Math.abs(pipelineBubble({ stages: 8, microBatches: 16 }) - sched.idleFraction) < 1e-12, 'formula = counted idle cells');
});

test('the grid\'s idle fraction equals the formula for every p ∈ {2,3,4,8}, m ∈ 1..32, both schedules', () => {
  for (const schedule of ['gpipe', '1f1b']) for (const stages of [2, 3, 4, 8]) for (let micro = 1; micro <= 32; micro += 1) {
    const view = pipelineView({ schedule, stages, micro });
    assert.ok(Math.abs(view.bubble - view.idle) < 1e-12, `${schedule} ${stages} ${micro}`);
  }
});

test('the toy\'s lanes: one segment per grid cell, labels only up to 12 columns, idle cells hatched kind', () => {
  const v = toyView(INITIAL_STATE);
  assert.equal(v.labelled, false, '14 columns at the default are label-free');
  assert.equal(v.columns, 14);
  assert.equal(v.lanes.length, 4);
  assert.ok(v.lanes.every((l) => l.segments.length === 14));
  assert.ok(v.lanes.flatMap((l) => l.segments).every((s) => s.label === undefined));
  const small = toyView({ ...INITIAL_STATE, stages: 3, micro: 4, schedule: '1f1b' });
  assert.equal(small.labelled, true);
  assert.deepEqual(small.lanes[0].segments.map((s) => s.label ?? '.'), ['F1', 'F2', 'F3', '.', '.', 'B1', 'F4', 'B2', '.', 'B3', '.', 'B4']);
  assert.deepEqual(small.lanes[0].segments.map((s) => s.kind).slice(0, 6), ['forward', 'forward', 'forward', 'idle', 'idle', 'backward']);
  assert.equal(lanesFromGrid([['F1', '.']], { labels: true })[0].label, 'stage 1');
});

test('toy readouts at the default and at the try-this stops', () => {
  const v = toyView(INITIAL_STATE);
  assert.deepEqual([v.bubble, v.peak, v.stepLength, v.gpus, v.statePerGpu], ['42.9%', '4', '14', '8,192', '7.02 GB']);
  assert.deepEqual([4, 8, 16, 32].map((micro) => toyView({ ...INITIAL_STATE, micro }).bubble), ['42.9%', '27.3%', '15.8%', '8.6%']);
  assert.equal(toyView({ ...INITIAL_STATE, stages: 8, micro: 32 }).bubble, '17.9%');
  const g16 = toyView({ ...INITIAL_STATE, micro: 16 });
  const f16 = toyView({ ...INITIAL_STATE, micro: 16, schedule: '1f1b' });
  assert.deepEqual([g16.peak, f16.peak, g16.bubble, f16.bubble], ['16', '4', '15.8%', '15.8%']);
  assert.equal(toyView({ ...INITIAL_STATE, zero: 1 }).statePerGpu, '13.25 GB');
  assert.equal(toyView({ ...INITIAL_STATE, tp: 1, cp: 1, pp: 1, dp: 1, zero: 0 }).statePerGpu, '6,480 GB');
  assert.equal(toyView({ ...INITIAL_STATE, zero: 3 }).statePerGpu, '0.79 GB');
  assert.equal(toyView({ ...INITIAL_STATE, tp: 1, pp: 1, dp: 8192, zero: 3 }).statePerGpu, '0.79 GB');
  assert.equal(toyView({ ...INITIAL_STATE, ...N.LLAMA_PRESETS.long }).gpus, '16,384');
  assert.equal(toyView({ ...INITIAL_STATE, ...N.LLAMA_PRESETS['16k'] }).gpus, '16,384');
});

test('presets: a matching degree set names its preset, anything else is custom', () => {
  assert.deepEqual(['8k', '16k', 'long'].map((k) => presetFor({ ...N.LLAMA_PRESETS[k] })), ['8k', '16k', 'long']);
  assert.equal(presetFor({ tp: 8, cp: 1, pp: 16, dp: 32 }), 'custom');
  assert.equal(toyView({ ...INITIAL_STATE, dp: 32 }).preset, 'custom');
});

test('the try-this list is the storyboard\'s, with its numbers computed', () => {
  assert.deepEqual(tryThis().map((t) => `${t.prompt} → Insight: ${t.insight}${t.rest}`), TRY_THIS);
});

test('stage numbers: bytes, partial sums, routes and degrees follow from math/ and the stand-ins', () => {
  assert.equal(N.GRADIENT_BYTES, 3152);
  assert.equal(N.DP_SENT_BYTES, 3152);
  assert.equal(bytesExact(N.DP_SENT_BYTES), '3,152 B');
  assert.equal(formatBytes(ringAllReduceBytes(N.GPT3_GRADIENT_BYTES, N.GPT3_DP_GPUS)), '689 GB');
  assert.deepEqual([N.TP_OUTPUT_BYTES, N.TP_SENT_BYTES, N.MLP_PARAMS_PER_GPU, N.TOY.perLayer.mlp], [64, 64, 192, 384]);
  assert.deepEqual([N.HANDOFF_BYTES, N.KV_NUMBERS_PER_TOKEN, N.KV_BYTES_PER_TOKEN, N.CP_SENT_BYTES], [64, 16, 32, 64]);
  assert.equal(N.TOY.total, 1576);
  assert.equal(paramBreakdown(PRESETS.toy).total, 1576);
});

test('frame 4: the partial sums add up to decoder-anatomy\'s MLP output for sat', () => {
  assert.deepEqual(N.MLP_OUT_SAT, M_SAT);
  assert.deepEqual(N.PARTIAL_GPU1, [0.25, 0.5, -0.25, -0.25, 0.25, 0.25, 0, 0]);
  assert.deepEqual(N.PARTIAL_GPU2, [-0.25, -0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25]);
  assert.deepEqual(N.PARTIAL_GPU1.map((v, i) => v + N.PARTIAL_GPU2[i]), [...N.MLP_OUT_SAT]);
  assert.ok(N.PARTIAL_GPU2.every((v) => !Object.is(v, -0)));
});

test('frame 10: moe\'s routes, 4 of 8 copies cross GPUs, 64 B each way, loads 2 / 2 / 3 / 1', () => {
  assert.deepEqual(N.ROUTES.map((r) => r.map((e) => e + 1)), [[2, 5], [4, 6], [3, 6], [1, 7]]);
  assert.deepEqual(N.ROUTES, ROUTER_TOY.map((row) => routeTopK(row, 2)));
  assert.deepEqual(N.CROSSING.map(({ token, expert }) => [N.TOKENS[token], `E${expert + 1}`]), [['The', 'E5'], ['cat', 'E6'], ['sat', 'E3'], ['down', 'E1']]);
  assert.equal(N.ROUTES.flat().length, 8);
  assert.deepEqual([...N.COPIES_PER_GPU], [2, 2, 3, 1]);
  assert.deepEqual([N.EP_COPY_BYTES, N.EP_DISPATCH_BYTES], [16, 64]);
  assert.equal(allToAllBytes(64, 4), 48, 'uniform traffic would send 48 B; the fixed routes send 64 B');
});

test('frame 11: the real degrees equal data/models.json and the stage prints them back', () => {
  const [first] = model('llama-3.1-405b', 'parallelism').split('; ');
  assert.equal(first, `TP${N.LLAMA.tp}/CP${N.LLAMA.cp}/PP${N.LLAMA.pp}/DP${N.LLAMA.dp} on ${int(N.LLAMA.gpus)}`);
  assert.equal(gpuCount(N.LLAMA), 8192);
  Object.entries(N.LLAMA_PRESETS).forEach(([key, d], i) => {
    const entry = model('llama-3.1-405b', 'parallelism').split('; ')[i];
    assert.match(entry, new RegExp(`^TP${d.tp}/CP${d.cp}/PP${d.pp}/DP${d.dp} on ${int(gpuCount(d))}`), key);
  });
  assert.equal(model('llama-3.1-405b', 'cp_long_context'), N.LLAMA_PRESETS.long.cp);
  assert.equal(model('llama-3.1-405b', 'total_params'), N.LLAMA.params);
  assert.equal(model('deepseek-v3', 'parallelism'), `PP${N.DEEPSEEK_V3.pp} / EP${N.DEEPSEEK_V3.ep} (8 nodes) / ZeRO-1 DP, no TP`);
  assert.equal(model('deepseek-v3', 'training_gpus'), N.DEEPSEEK_V3.gpus);
  assert.equal(model('gpt-3', 'total_params') * 2, N.GPT3_GRADIENT_BYTES);
  const r = TRAINING_RECIPES.adam;
  assert.equal(trainingBytesPerParam(r).total, N.WEIGHT_BYTES_PER_PARAM);
  const s = stateView({ ...N.LLAMA, zero: N.LLAMA.zeroStage });
  assert.equal(formatBytes(N.LLAMA.params * N.WEIGHT_BYTES_PER_PARAM / (N.LLAMA.tp * N.LLAMA.pp)), '50.6 GB');
  assert.deepEqual([s.parts.weights, s.parts.grads, s.parts.optimizer].map(formatStateGB), ['6.33 GB', '0.10 GB', '0.59 GB']);
  assert.equal(formatCount(N.LLAMA.params / (N.LLAMA.tp * N.LLAMA.pp), { digits: 4 }), '3.164B');
});

test('the math panel links the counter and the bubble to the stage', () => {
  const blocks = LESSON.math.blocks.map((b) => b.tex).join('\n');
  ['comm', 'bubble'].forEach((l) => assert.match(blocks, new RegExp(`\\\\htmlClass\\{hl-${l}\\}`)));
  assert.match(blocks, /\\frac\{p-1\}\{m\+p-1\}/);
});

test('nothing here mutates its inputs', () => {
  const state = Object.freeze({ ...INITIAL_STATE, schedule: '1f1b' });
  const before = JSON.stringify(N.LLAMA_PRESETS);
  toyView(state);
  checkWork(state);
  tryThis();
  assert.equal(JSON.stringify(N.LLAMA_PRESETS), before);
});
