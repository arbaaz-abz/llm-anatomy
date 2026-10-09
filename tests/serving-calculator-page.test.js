import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { stepTime, RUNNING_EXAMPLE, usersAtTarget, costPerMillion, hbmFor } from '../math/serving.js';
import { batchSpeedup } from '../math/specdec.js';
import { deepFreeze } from '../math/core.js';
import { LESSON, lessonFor } from '../serving/concepts/serving-calculator/content.js';
import { BELOW, factRows } from '../serving/concepts/serving-calculator/facts.js';
import * as N from '../serving/concepts/serving-calculator/numbers.js';
import { compute } from '../serving/concepts/serving-calculator/model.js';
import { INITIAL_STATE, modelDefaults, checkWork, usd, speedupText, USER_STOPS } from '../serving/concepts/serving-calculator/format.js';
import { toyView } from '../serving/concepts/serving-calculator/toy-view.js';
import { tryThis } from '../serving/concepts/serving-calculator/try-this.js';
import { gpuPreset, hbmText, supportsWeights, modelPreset } from '../serving/concepts/serving-calculator/presets.js';
import { nextState } from '../serving/concepts/serving-calculator/toy.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './serving-calculator-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const fact = (set, id, key) => data[set].entries.find((e) => e.id === id).facts[key];
const st = (patch = {}) => ({ ...INITIAL_STATE, ...patch });
const view = (patch) => toyView(st(patch), data);
const titled = (text) => text.replace(/\[\[([a-z0-9-]+)\]\]/g, (_, slug) => graph.concepts.find((c) => c.slug === slug).title);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('10 facts rows; every placeholder resolves; the filled lesson has no dash', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 10);
  factRows().forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.doesNotMatch(fillText(row.claim, data), /—/, `row ${i + 1}`);
  });
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  [lesson.hook, ...lesson.intuition, ...lesson.takeaways, lesson.facts.framing].forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 50)));
  assert.match(fillText(lesson.facts.rows[0].claim, data), /^DeepSeek-V4-Pro: 1\.6T total, 49B active, 61 layers, 384 routed experts \(6 per token\), 1M context; d_model 7,168; expert width 3,072\.$/);
  assert.match(fillText(lesson.facts.rows[2].claim, data), /4,000–12,000 bytes/);
  assert.match(fillText(lesson.facts.rows[5].claim, data), /73\.7K input vs 14\.8K output .* \$87,072 a day at \$2 per GPU-hour; 56\.3% /);
  assert.match(lesson.intuition[2], /\$2\.65 per GPU-hour is \$0\.12 per million tokens/);
  assert.match(lesson.intuition[0], /about 865 GB \(reported\)\. A GB300 holds 288 GB, so the weights alone need at least 4 GPUs.*54\.1 GB of weights and 234 GB for users/);
});

test('this is the capstone: nothing comes after it (README lesson 1)', () => {
  assert.deepEqual(graph.concepts.filter((c) => c.prereqs.includes('serving-calculator')), []);
  assert.deepEqual(LESSON.links.next, []);
});

test('the stage constants equal the data they restate (P3-R13 pattern)', () => {
  const m = (key) => fact('models', 'deepseek-v4-pro', key).value;
  assert.equal(N.V4.totalParams, m('total_params'));
  assert.equal(N.V4.activeParams, m('active_params'));
  assert.equal(N.V4.dModel, m('d_model'));
  assert.equal(N.V4.experts, m('experts_total'));
  assert.equal(N.V4.expertsPerToken, m('experts_active'));
  assert.equal(N.V4.checkpointBytes, m('checkpoint_gb') * 1e9);
  assert.deepEqual([N.V4.kvBytesPerToken.low, N.V4.kvBytesPerToken.high], m('kv_bytes_per_token'));
  const g300 = (k) => fact('hardware', 'gb300-nvl72', k).value;
  assert.deepEqual([N.GB300.hbmBytes, N.GB300.bandwidthTBps, N.GB300.fp8Tflops], [g300('hbm_gb') * 1e9, g300('hbm_tbps'), g300('fp8_e4m3_dense_tflops')]);
  assert.equal(N.GB200_HBM_BYTES, fact('hardware', 'gb200-nvl72', 'hbm_gb').value * 1e9);
  const ix = (id, k) => fact('serving', id, k);
  assert.deepEqual(N.MEASURED.gb300, { tokSGpu: ix('inferencex-v4-pro-gb300', 'throughput_tok_s_gpu').value, tokSUser: ix('inferencex-v4-pro-gb300', 'interactivity_tok_s_user').value, peakTokSGpu: ix('inferencex-v4-pro-gb300', 'max_throughput_tok_s_gpu').value, peakTokSUser: ix('inferencex-v4-pro-gb300', 'max_throughput_tok_s_user').value, usdPerGpuHour: ix('inferencex-v4-pro-gb300', 'gpu_hour_usd').value });
  assert.deepEqual([N.MEASURED.gb200.tokSGpu, N.MEASURED.gb200.usdPerGpuHour], [ix('inferencex-v4-pro-gb200', 'throughput_tok_s_gpu').value, ix('inferencex-v4-pro-gb200', 'gpu_hour_usd').value]);
  assert.equal(N.TARGET_TOK_S_USER, N.MEASURED.gb300.tokSUser);
  assert.deepEqual([N.PRODUCTION.inputNodeTokS, N.PRODUCTION.outputNodeTokS], [ix('deepseek-v3-production', 'prefill_tok_s_node').value, ix('deepseek-v3-production', 'decode_tok_s_node').value]);
  assert.deepEqual([N.LIST_PRICES.inputUsd, N.LIST_PRICES.outputUsd, N.LIST_PRICES.anthropicRatio], [ix('pricing-deepseek-v4-pro', 'input_miss_usd_per_m').value, ix('pricing-deepseek-v4-pro', 'output_usd_per_m').value, ix('pricing-anthropic', 'output_input_ratio').value]);
  assert.match(ix('inferencex-v4-pro-gb300', 'throughput_tok_s_gpu').note, new RegExp(N.MEASURED_DATE));
  assert.equal(ix('pricing-deepseek-v4-pro', 'output_usd_per_m').last_verified, N.LIST_PRICES_DATE);
  assert.match(ix('deepseek-v3-production', 'prefill_tok_s_node').source_url, /202502/);
  assert.deepEqual(N.ISL_OSL, { input: 8192, output: 1024 });
});

test('the default toy state prints the storyboard\'s exact numbers and "Check my work"', () => {
  const v = view();
  assert.equal(v.checkWork, CHECK_WORK);
  assert.equal(checkWork(compute(st(), data), st()), CHECK_WORK);
  assert.deepEqual(v.readouts, {
    'weights-total': '865 GB', 'gpus-min': '4', hbm: '288 GB nominal', 'weights-per-gpu': '54.1 GB', 'free-per-gpu': '234 GB', users: '1,889', 'step-time': '37 ms', bound: 'compute-bound',
    'tok-user': '27 tok/s', 'tok-gpu': '51,020 tok/s', 'target-users': '1,889', 'target-limit': 'compute', 'cost-floor': '$0.0144', 'cost-gb300': '$0.119', 'cost-gb200': '$0.28',
    'prefill-tps': '51,020 tok/s', 'input-cost': '$0.0144', 'cost-ratio': '1×', 'mtp-speedup': '', 'mtp-tok-user': '', 'mtp-record': '',
    'kv-per-user': '36.9 MB (low) · 111 MB (high)', 'users-fit': '6,345 (low) · 2,115 (high)',
  });
  assert.equal(v.showMtp, false);
  assert.doesNotMatch(v.checkWork, /MTP/);
  assert.match(v.notes.ratio, /output as cheap as input/);
  assert.match(v.conditions, /GB300 NVL72, 288 GB nominal per GPU · EP 16 · 8K in \/ 1K out/);
});

test('toy outputs equal the math/ functions for the same inputs (Review Focus 1)', () => {
  const V = { activeParamsPerGpu: 49e9, weightBytesPerGpu: 865e9 / 16, dModel: 7168, actBytesPerElem: 1, peakTflops: 5000, bandwidthTBps: 8, kvBytesPerToken: 4000, context: 9216 };
  const c = compute(st(), data);
  const step = stepTime({ ...V, tokens: 1889, seqs: 1889 });
  assert.equal(c.step.timeS, step.timeS);
  assert.equal(usersAtTarget({ targetTokPerUser: 27, ...V, maxUsers: 6345 }).users, c.users);
  assert.equal(c.cost, costPerMillion(2.65, 1889 / step.timeS));
  assert.equal(c.measured.gb300.toFixed(2), '0.12');
  // The Llama preset is the course's running example: the same step time as prefill-decode's.
  const llama = compute({ ...st(), ...modelDefaults('llama-3.1-70b') }, data);
  assert.equal(llama.step.timeS, stepTime({ ...RUNNING_EXAMPLE, tokens: llama.users, seqs: llama.users, context: 9216 }).timeS);
  assert.equal(llama.maxFit, 23);
  assert.equal(usd(llama.cost), '$0.932');
});

test('try-this states (storyboard §6)', () => {
  assert.deepEqual(tryThis(data).map((t) => `${t.prompt} → Insight: ${t.insight}${titled(t.rest)}`), TRY_THIS);
  const r = (patch) => view(patch).readouts;
  assert.deepEqual([9216, 139264, 1e6].map((context) => r({ context })['users-fit']), ['6,345 (low) · 2,115 (high)', '419 (low) · 139 (high)', '58 (low) · 19 (high)']);
  assert.equal(r({ context: 1e6, hw: 'gb200-nvl72' })['users-fit'].split(' ')[0], '32');
  assert.equal(r({ users: 1 })['tok-user'], '148 tok/s');
  assert.equal(r({ users: 1 })['tok-gpu'], '148 tok/s');
  assert.equal(r({ hit: 0.563 })['cost-ratio'], '2.29×');
  assert.equal(r({ context: 139264 })['cost-ratio'], '4.46×');
  assert.equal(r({ context: 1e6 })['cost-ratio'], '31.5×');
  const mtp = (patch) => view({ mtp: true, ...patch }).readouts['mtp-speedup'];
  assert.deepEqual([mtp({ users: 256 }), mtp({}), mtp({ context: 139264 })], ['1.48×', '0.9 of the plain speed', '1.73×']);
  assert.match(view({ mtp: true }).readouts['mtp-record'], /\+87% \(1\.87×\)/);
  assert.equal(speedupText(0.904), '0.9 of the plain speed');
});

test('a model that does not fit prints words and 0 users, never a negative count or NaN (Review Focus 2)', () => {
  const v = view({ gpus: 1 });
  assert.equal(v.readouts['free-per-gpu'], 'does not fit');
  assert.equal(v.readouts.users, '0');
  assert.equal(v.readouts['step-time'], 'does not fit');
  assert.equal(v.readouts['target-limit'], 'does not fit');
  assert.match(v.checkWork, /= −577 GB free: does not fit/);
  Object.values(v.readouts).forEach((t) => assert.doesNotMatch(t, /NaN|Infinity|^-/));
  assert.equal(v.figure.memory, null);
  const tight = view({ model: 'llama-3.1-70b', weights: 'bf16', hw: 'h200', gpus: 1, context: 131072 });
  assert.equal(tight.readouts['users-fit'], '0');
  assert.equal(tight.readouts['step-time'], 'no user fits');
});

test('V4 → Llama → V4 restores the default state exactly (Review Focus 5); H200 moves NVFP4 to FP8', () => {
  const llama = nextState(st(), { model: 'llama-3.1-70b' }, data);
  assert.deepEqual([llama.weights, llama.hw, llama.gpus, llama.context], ['fp8', 'h200', 1, 9216]);
  assert.deepEqual(nextState(llama, { model: 'deepseek-v4-pro' }, data), INITIAL_STATE);
  assert.deepEqual(toyView(nextState(llama, { model: 'deepseek-v4-pro' }, data), data).readouts, view().readouts);
  assert.equal(nextState(st({ weights: 'nvfp4' }), { hw: 'h200' }, data).weights, 'fp8');
  assert.equal(supportsWeights(gpuPreset(data, 'h200'), 'nvfp4'), false);
  assert.throws(() => compute(st({ weights: 'nvfp4', hw: 'h200' }), data), RangeError);
  assert.equal(USER_STOPS.at(-1), 'target');
});

test('every hardware readout prints its basis word (Review Focus 3)', () => {
  const words = Object.fromEntries(['h200', 'b200', 'gb200-nvl72', 'gb300-nvl72'].map((id) => [id, hbmText(gpuPreset(data, id))]));
  assert.deepEqual(words, { h200: '141 GB nominal', b200: '180 GB usable (192 GB nominal)', 'gb200-nvl72': '186 GB nominal (the rack total over 72 GPUs)', 'gb300-nvl72': '288 GB nominal' });
  assert.equal(hbmFor(data.hardware.entries.find((e) => e.id === 'b200')).basis, 'usable');
  assert.equal(view({ hw: 'b200', weights: 'fp8' }).readouts.hbm, '180 GB usable (192 GB nominal)');
});

test('V4 KV ends and the Llama cache format feed the speed rows; inputs are not mutated', () => {
  assert.equal(view({ kvEnd: 'high' }).readouts['users-fit'], '6,345 (low) · 2,115 (high)');
  assert.match(view({ kvEnd: 'high' }).checkWork, /^Fit:.*\nKV:     12,000 B × 9,216 tokens = 111 MB/);
  assert.equal(modelPreset(data, 'llama-3.1-70b').kvBytes.bf16, 327680);
  assert.equal(view({ model: 'llama-3.1-70b', weights: 'fp8', hw: 'h200', gpus: 1, kvFormat: 'fp8' }).compute.chosen.bytesPerToken, 163840);
  const frozen = deepFreeze(data);
  const s = Object.freeze(st());
  assert.doesNotThrow(() => toyView(s, frozen));
  assert.equal(batchSpeedup({ alpha: 0.85, k: 1, c: 0.05, batch: 419, model: compute(st({ context: 139264 }), data).stepModel }).speedup.toFixed(2), '1.73');
});
