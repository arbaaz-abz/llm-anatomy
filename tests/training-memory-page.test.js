import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { formatBytes, formatRatio } from '../math/core.js';
import { TRAINING_RECIPES, trainingBytesPerParam, zeroPerGpuBytes, activationBytes, activationBytesPerLayer, gpusToHoldStates } from '../math/training-memory.js';
import { LESSON, lessonFor } from '../training/concepts/training-memory/content.js';
import { checkWork, gbText, trafficText, extraComputeText, scoreSplit, activationArgs } from '../training/concepts/training-memory/format.js';
import { toyView, tryThis, chipInfo, modelParams, INITIAL_STATE, NOT_MODELED } from '../training/concepts/training-memory/toy-view.js';
import { BELOW, factRows } from '../training/concepts/training-memory/facts.js';
import * as N from '../training/concepts/training-memory/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './training-memory-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;
const hw = (id, key) => data.hardware.entries.find((e) => e.id === id).facts[key].value;
const ADAM = TRAINING_RECIPES.adam;
const state = (patch = {}) => ({ ...INITIAL_STATE, ...patch });

test('the lesson spec is complete', () => assert.deepEqual(validateLessonSpec(LESSON), []));
test('the lesson filled with real data is complete too', () => assert.deepEqual(validateLessonSpec(lessonFor(data)), []));
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('10 facts rows, every placeholder resolves, nothing prints a dash', () => {
  assert.equal(LESSON.facts.rows.length, 10);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  lessonFor(data).facts.rows.forEach((row, i) => assert.doesNotMatch(fillClaim(row.claim, data).segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`));
});

test('the derived numbers in the rows come from the math and the data', () => {
  const rows = factRows(data).map((r) => fillClaim(r.claim, data).segments.map((s) => s.text).join(''));
  assert.match(rows[1], /7\.5B parameters on 64 GPUs → 120 GB \/ 31\.4 GB \/ 16\.6 GB \/ 1\.88 GB per GPU/);
  assert.match(rows[4], /1\.04T parameters; BF16 weights plus an FP32 gradient buffer are 6 bytes per parameter, about 6 TB over 256 model-parallel GPUs \(24\.4 GB per GPU\)/);
  assert.match(rows[6], /1\.6T total, 49B active \(3\.1%\)\. Its size at the 16-byte Adam recipe would be 25\.6 TB of state/);
  assert.match(rows[6], /at Muon's 12 B, 19\.2 TB/);
  assert.match(rows[3], /DeepSeek-V3 \(2024\)/);
  assert.match(rows[7], /Llama 3\.1 405B \(2024\)/);
  assert.match(rows[4], /Kimi K2 \(2025\)/);
});

test('every dated prose placeholder resolves; the GPT-3 year comes from the data', () => {
  const lesson = lessonFor(data);
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(lesson.animation.belowFor(0)[0], /GPT-3 itself \(2020\) predates BF16 training/);
  BELOW.forEach((_, i) => lesson.animation.belowFor(i).forEach((t) => assert.doesNotMatch(t, /[{}—]/)));
  assert.equal(lesson.animation.belowFor(42).length, 0);
});

test('Next lists exactly the lessons that take this one as a prereq', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('training-memory')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('stand-in constants restate the data (P3-R13)', () => {
  assert.equal(N.GPT3.params, model('gpt-3', 'total_params'));
  assert.equal(N.GPT3.layers, model('gpt-3', 'layers'));
  assert.equal(N.GPT3.hidden, model('gpt-3', 'd_model'));
  assert.equal(N.GPT3.heads, model('gpt-3', 'n_heads'));
  assert.equal(N.GPT3.seq, model('gpt-3', 'context_length'));
  assert.equal(N.V4_PRO.total, model('deepseek-v4-pro', 'total_params'));
  assert.equal(N.V4_PRO.active, model('deepseek-v4-pro', 'active_params'));
  assert.equal(N.H100_HBM, hw('h100', 'hbm_gb') * 1e9);
  assert.deepEqual(N.FRAME11_CHIPS.map((c) => c.hbm), [hw('h100', 'hbm_gb'), hw('h200', 'hbm_gb'), hw('b200', 'hbm_usable_gb'), hw('b300', 'hbm_gb')].map((g) => g * 1e9));
  assert.equal(N.MUON_BYTES, trainingBytesPerParam(TRAINING_RECIPES.muon).total);
});

test('hardware readouts follow the conventions: B200 reads usable, the others nominal, each labeled', () => {
  assert.deepEqual(['h100', 'h200', 'b200', 'b300'].map((c) => chipInfo(c, data).bytes / 1e9), [80, 141, 180, 288]);
  assert.equal(chipInfo('b200', data).basis, 'usable (192 nominal)');
  assert.equal(chipInfo('h100', data).basis, 'nominal');
  assert.deepEqual(N.FRAME11_CHIPS.map((c) => c.capacity), ['80 GB nominal', '141 GB nominal', '180 GB usable', '288 GB nominal']);
});

test('"Check my work" for the default state is the storyboard text, templated for other states', () => {
  assert.equal(toyView(INITIAL_STATE, data).checkWork, CHECK_WORK);
  const params = modelParams(INITIAL_STATE, data);
  assert.equal(checkWork(INITIAL_STATE, { params, chip: chipInfo('h100', data) }), CHECK_WORK);
  const z3 = toyView(state({ stage: 3, recompute: 'full' }), data).checkWork.split('\n');
  assert.equal(z3[0], 'weights     = 175B × 2 B ÷ 64 = 5.47 GB');
  assert.equal(z3[2], 'optimizer   = 175B × (4 + 8) B ÷ 64 = 32.81 GB');
  assert.equal(z3[3], 'activations = 96 × 2 × 2,048 × 12,288 B = 4.83 GB');
  assert.equal(z3[4], 'total       = 48.58 GB vs 80 GB (H100, nominal) → fits');
  const other = toyView(state({ model: 'deepseek-v4-pro' }), data).checkWork.split('\n');
  assert.equal(other[3], 'activations = not modeled');
  assert.match(toyView(state({ gpu: 'b200', stage: 3 }), data).checkWork, /vs 180 GB \(B200, usable \(192 nominal\)\)/);
});

test('the try-this numbers are the storyboard\'s, computed through the shared functions', () => {
  const v = (patch) => toyView(state(patch), data);
  const gb = (bytes) => gbText(bytes).replace(' GB', '');
  const zp = [0, 1, 2, 3].map((stage) => gb(zeroPerGpuBytes({ params: 7.5e9, recipe: ADAM, stage, dp: 64 }).total));
  assert.deepEqual(zp, TRY_THIS.zeroPaper);
  assert.deepEqual([0, 1, 2, 3].map((stage) => v({ model: 'zero-paper', stage }).state.total.replace(' GB', '')), TRY_THIS.zeroPaper);
  const t = TRY_THIS.gpt3Zero3;
  assert.equal(v({ stage: 3 }).state.total, `${t.state} GB`);
  assert.equal(v({ stage: 3, recompute: 'none' }).total, `${t.none} GB`);
  assert.equal(v({ stage: 3, recompute: 'selective' }).total, `${t.selective} GB`);
  assert.equal(v({ stage: 3, recompute: 'full' }).total, `${t.full} GB`);
  assert.equal(v({ stage: 3, recompute: 'none' }).verdict, 'does not fit');
  assert.equal(v({ stage: 3, recompute: 'selective' }).verdict, 'does not fit');
  assert.equal(v({ stage: 3, recompute: 'full' }).verdict, 'fits');
  assert.equal(v({ stage: 0, recompute: 'full' }).total, `${t.stage0Full} GB`);
  const k = TRY_THIS.v4Pro;
  const holders = (patch) => v({ model: 'deepseek-v4-pro', ...patch }).holders;
  assert.deepEqual([holders({}), holders({ gpu: 'b200' }), holders({ gpu: 'b300' }), holders({ recipe: 'muon', gpu: 'b300' }), v({}).holders], [k.h100, k.b200, k.b300, k.muonB300, k.gpt3H100]);
});

test('every try-this prompt fills from data with no dashes or leftover braces', () => {
  tryThis(data).forEach((t) => assert.doesNotMatch(t.prompt + t.rest, /[{}—]/));
  assert.match(tryThis(data)[2].prompt, /1\.6T total, 49B active/);
  assert.match(tryThis(data)[0].prompt, /120 → 31\.41 → 16\.64 → 1\.88 GB/);
});

test('readouts equal the shared functions for the same inputs (one definition each)', () => {
  const params = model('gpt-3', 'total_params');
  const per = trainingBytesPerParam(ADAM);
  assert.equal(toyView(INITIAL_STATE, data).perParam.totalBytes, per.total);
  const z = zeroPerGpuBytes({ params, recipe: ADAM, stage: 0, dp: 64 });
  const act = activationBytes(activationArgs(INITIAL_STATE));
  const v = toyView(INITIAL_STATE, data);
  assert.equal(v.total, gbText(z.total + act));
  assert.equal(v.activations, gbText(act));
  assert.equal(v.holders, gpusToHoldStates({ params, bytesPerParam: per.total, hbmBytes: 80e9 }));
  assert.equal(v.gpusToHold, '35 H100s (nominal)');
  assert.deepEqual(v.composition.map((p) => p.value), [z.weights, z.grads, z.optimizer, act]);
  assert.equal(trafficText(3), formatRatio(1.5));
  assert.equal(trafficText(3), '1.5×');
  assert.equal(trafficText(1), '1×');
  assert.equal(toyView(state({ stage: 3 }), data).traffic, '1.5×');
});

test('recomputation costs and the selective hover', () => {
  assert.equal(extraComputeText('full'), 'about +33%');
  assert.equal(extraComputeText('none'), '0%');
  assert.equal(toyView(state({ recompute: 'selective' }), data).extraCompute.hover, 'It recomputes only the attention-score grid.');
  assert.equal(toyView(state({ recompute: 'selective' }), data).extraCompute.text, 'small, not quantified here');
  assert.equal(toyView(state({ model: 'zero-paper' }), data).extraCompute.hover, '');
});

test('frame numbers: the activation split behind frames 5–7 and the GPU counts of frame 11', () => {
  const args = { seq: 2048, microBatch: 1, hidden: 12288, heads: 96 };
  const s = scoreSplit(args);
  assert.equal(s.total, activationBytesPerLayer({ ...args, recompute: 'none' }));
  assert.equal(Math.round((s.scores / (2048 * 12288)) * 10) / 10, 80);
  assert.equal(formatBytes(activationBytes({ layers: 96, ...args, recompute: 'none' })), '275 GB');
  const gpus = (params, bytes) => N.FRAME11_CHIPS.map((c) => gpusToHoldStates({ params, bytesPerParam: bytes, hbmBytes: c.hbm }));
  assert.deepEqual(gpus(1e12, 16), [200, 114, 89, 56]);
  assert.deepEqual(gpus(1.6e12, 16), [320, 182, 143, 89]);
  assert.deepEqual(gpus(1.6e12, 12), [240, 137, 107, 67]);
});

test('non-GPT-3 presets model no activations and say so', () => {
  const v = toyView(state({ model: 'kimi-k2' }), data);
  assert.equal(v.activations, NOT_MODELED);
  assert.equal(v.composition.some((p) => p.name === 'activations'), false);
  assert.equal(v.params, '1.04T');
});

test('inputs are not mutated', () => {
  const frozen = Object.freeze(state({ stage: 2 }));
  const before = JSON.stringify([TRAINING_RECIPES, data.models.entries.length]);
  toyView(frozen, data);
  tryThis(data);
  assert.equal(JSON.stringify([TRAINING_RECIPES, data.models.entries.length]), before);
});
