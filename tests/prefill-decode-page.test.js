import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { RUNNING_EXAMPLE, stepTime, freeHbmPerGpu, maxUsersPerGpu, hbmFor, prefillTokPerSecCeiling, FORWARD_FLOPS_PER_PARAM_TOKEN } from '../math/serving.js';
import { FLOPS_PER_PARAM_TOKEN } from '../math/scale.js';
import { ridgePoint, tokensToComputeBound, arithmeticIntensity } from '../math/roofline.js';
import { kvCacheBytes, kvBytesPerToken } from '../math/memory.js';
import { formatBytes, formatCount, formatDuration, formatInt, formatRatio } from '../math/core.js';
import { LESSON, lessonFor } from '../serving/concepts/prefill-decode/content.js';
import { GPUS, gpuPreset, gpuLabel, hbmText } from '../serving/concepts/prefill-decode/hardware.js';
import { analyze, checkWork, modelFor, usersStops, usersLabel, formatFlops, fixed1, INITIAL_STATE, PROMPT_STOPS, CONTEXT_STOPS } from '../serving/concepts/prefill-decode/format.js';
import { toyView, TOY_STEP_W } from '../serving/concepts/prefill-decode/toy-view.js';
import { tryThis } from '../serving/concepts/prefill-decode/try-this.js';
import { BELOW, factRows, intuition, hook, takeaways } from '../serving/concepts/prefill-decode/facts.js';
import { STATES, MAX_USERS, CROSSING, H100_CROSSING, RIDGE, STAGE_SCALE_S, decode, prefill, readParts } from '../serving/concepts/prefill-decode/model.js';
import { H200, H100, MEASURED, CROSS_TOKENS, CONTEXT, USERS_SHOWN } from '../serving/concepts/prefill-decode/numbers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_DOES_NOT_FIT, TRY_THIS } from './prefill-decode-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const entry = (set, id) => data[set].entries.find((e) => e.id === id);
const hw = (id, key) => entry('hardware', id).facts[key]?.value;
const sv = (id, key) => entry('serving', id).facts[key];
const view = (state) => toyView({ ...INITIAL_STATE, ...state }, data);
const titled = (text) => text.replace(/\[\[([a-z0-9-]+)\]\]/g, (_, slug) => graph.concepts.find((c) => c.slug === slug).title);
const cells = (v) => Object.fromEntries(v.tables.flatMap((t) => t.rows.flatMap((r) => r.cells.filter((c) => c.name).map((c) => [c.name, c.value]))));
const L = RUNNING_EXAMPLE;

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('5 facts rows; every placeholder resolves; the filled lesson has no dash', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 5);
  lesson.facts.rows.forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.doesNotMatch(fillText(row.claim, data), /—/, `row ${i + 1}`);
  });
  const texts = [lesson.hook, ...lesson.intuition, lesson.intuitionNote, ...lesson.takeaways, lesson.facts.framing, lesson.animation.standIn, lesson.toy.intro, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  [...intuition, hook, ...takeaways].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  const filled = lesson.facts.rows.map((r) => fillText(r.claim, data));
  assert.equal(filled[0], 'H200: 141 GB HBM3e (nominal), 4.8 TB/s, 1,979 TFLOP/s dense FP8 (reported: 2 × BF16); ridge 412.3 FLOPs per byte.');
  assert.equal(filled[1], 'From their dense peaks and HBM bandwidth, the H100 crosses into compute-bound at 318 tokens per weight read and the B200 at 302, in BF16 as in FP8.');
  assert.equal(filled[2], 'Llama-3.1-70B: 80 layers, 8 KV heads × 128, d_model 8,192, a 131,072-token context and 327,680 KV bytes per token in BF16; 70B parameters as its card rounds them.');
  assert.match(filled[3], /^DeepSeek-V4-Pro on GB300 NVL72 made 11,056 tok\/s\/GPU at 13\.1 tok\/s\/user \(max throughput\) and 6,182 at 27; on GB200 NVL72, 8,933 at 15\.3 and 2,189 at 27 \(ISL 8K \/ OSL 1K, FP4, InferenceX, measured 2026-05-22\); the source does not say/);
  assert.match(filled[4], /^Kimi K2\.5 on B200 \(NVFP4\) cost \$0\.14 per million tokens at 32 tok\/s\/user and \$0\.347 at 90/);
  assert.match(factRows(null)[0].claim, /ridge —/, 'without data a derived number prints a dash, never a stale one');
});

test('hook, intuition and takeaways print the running example\'s numbers, each from stepTime', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.hook, 'Why can a GPU read 217 tokens of your prompt in about the time it takes to write one token of the answer?');
  assert.match(lesson.intuition[0], /: 140 billion operations\. .* In FP8 that is 70 GB, and the H200 reads 4\.8 TB per second, so one pass over the weights takes 14\.6 ms .* takes 70\.7 µs\./);
  assert.match(lesson.intuition[1], /With 1,000 prompt tokens the arithmetic takes 70\.7 ms, while the reading .* takes 18\.1 ms: .* an H200 needs 217\. At 217 prompt tokens math and reading both take 15\.4 ms/);
  assert.match(lesson.intuition[2], /At 2,048 tokens of context the H200 is full at 105 users/);
  assert.match(lesson.takeaways[0], /\(14\.6 ms for 70 GB on an H200\)/);
  assert.match(lesson.takeaways[1], /: 217 prompt tokens on an H200/);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('prefill-decode')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the stage constants equal the data they restate (P3-R13 pattern)', () => {
  assert.deepEqual(hbmFor(entry('hardware', 'h200')), { bytes: H200.hbmBytes, basis: H200.basis });
  assert.equal(hw('h200', 'bf16_dense_tflops'), H200.peakBf16Tflops);
  assert.deepEqual([hw('h100', 'bf16_dense_tflops'), hw('h100', 'hbm_tbps')], [H100.peakTflops, H100.bandwidthTBps]);
  const gb300 = 'inferencex-v4-pro-gb300';
  assert.deepEqual(MEASURED.points.map((p) => [p.perUser, p.perGpu]), [
    [sv(gb300, 'max_throughput_tok_s_user').value, sv(gb300, 'max_throughput_tok_s_gpu').value],
    [sv(gb300, 'interactivity_tok_s_user').value, sv(gb300, 'throughput_tok_s_gpu').value],
  ]);
  ['max_throughput_tok_s_gpu', 'throughput_tok_s_gpu'].forEach((k) => assert.match(sv(gb300, k).note, /ISL 8192 \/ OSL 1024, FP4, .*measured 2026-05-22/));
  assert.equal(RUNNING_EXAMPLE.kvBytesPerToken, kvBytesPerToken({ layers: 80, kvHeads: 8, headDim: 128, bytesPerElem: 2 }));
  assert.equal(CROSS_TOKENS, Math.round(CROSSING));
});

test('the stage\'s numbers come from stepTime on RUNNING_EXAMPLE and match the storyboard (Review Focus 1)', () => {
  assert.deepEqual(STATES.decodeAlone, stepTime({ ...L, tokens: 1, seqs: 1, context: 0 }));
  assert.deepEqual(STATES.prefill, stepTime({ ...L, tokens: 1000, seqs: 0, context: 0 }));
  assert.equal(formatDuration(STATES.decodeAlone.computeS), '70.7 µs');
  assert.equal(formatDuration(STATES.decodeAlone.timeS), '14.6 ms');
  assert.deepEqual([STATES.prefill.computeS, STATES.prefill.memoryS].map(formatDuration), ['70.7 ms', '18.1 ms']);
  assert.equal(formatBytes(STATES.prefill.bytes), '87.1 GB');
  assert.deepEqual([arithmeticIntensity(STATES.decodeAlone), arithmeticIntensity(STATES.prefill)].map(fixed1), ['2.0', '1,607.5']);
  assert.deepEqual([RIDGE, CROSSING, H100_CROSSING].map(fixed1), ['412.3', '217.1', '318.2']);
  assert.deepEqual([STATES.cross.computeS, STATES.cross.memoryS].map(formatDuration), ['15.4 ms', '15.4 ms']);
  assert.equal(STATES.cross.bound, 'memory', 'by a hair at 217; 218 is compute-bound');
  assert.equal(prefill(218).bound, 'compute');
  assert.equal(MAX_USERS, maxUsersPerGpu(freeHbmPerGpu({ hbmBytes: 141e9, weightBytes: 70e9, gpus: 1 }), kvCacheBytes({ bytesPerToken: 327680, tokens: 2048 })));
  const rows = [...USERS_SHOWN, MAX_USERS].map((u) => {
    const s = STATES.batch[u];
    return [u, formatDuration(s.timeS), formatCount(1 / s.timeS), formatInt(u / s.timeS), fixed1(arithmeticIntensity(s)), formatDuration(s.kvBytes / 4.8e12)];
  });
  assert.deepEqual(rows, [
    [1, '14.7 ms', '67.9', '68', '2.0', '140 µs'], [8, '15.7 ms', '63.6', '509', '14.8', '1.12 ms'],
    [64, '23.8 ms', '42.1', '2,694', '78.6', '8.95 ms'], [105, '29.6 ms', '33.7', '3,543', '103.3', '14.7 ms'],
  ]);
  assert.equal(formatRatio((8 / STATES.batch[8].timeS) / (1 / STATES.batch[1].timeS)), '7.49×');
  assert.equal(formatBytes(STATES.batch[105].kvBytes), '70.5 GB');
  assert.equal(STAGE_SCALE_S, STATES.prefill.timeS);
  assert.deepEqual(readParts(decode(8)).map((p) => p.label), ['weights read', 'KV read', 'activations']);
  assert.equal(FORWARD_FLOPS_PER_PARAM_TOKEN, FLOPS_PER_PARAM_TOKEN / 3);
});

test('the captions\' numbers equal the functions they repeat (README lesson 29)', () => {
  assert.match(CAPTIONS[1], new RegExp(`Reading takes ${formatDuration(STATES.decodeAlone.memoryS)}, the math ${formatDuration(STATES.decodeAlone.computeS)}`));
  assert.match(CAPTIONS[2], new RegExp(`math takes ${formatDuration(STATES.prefill.computeS)} and the reading only ${formatDuration(STATES.prefill.memoryS).replace(' ms', '')}\\.`));
  assert.match(CAPTIONS[4], new RegExp(`ridge sits at ${Math.round(RIDGE)} operations`));
  assert.match(CAPTIONS[5], new RegExp(`At ${CROSS_TOKENS} prompt tokens, math and reading both take ${formatDuration(STATES.cross.timeS)}`));
  assert.match(CAPTIONS[6], new RegExp(`from ${formatDuration(STATES.batch[1].timeS).replace(' ms', '')} to ${formatDuration(STATES.batch[8].timeS)} .* makes ${formatRatio((8 / STATES.batch[8].timeS) * STATES.batch[1].timeS).replace('×', '')} times`));
  assert.match(CAPTIONS[7], new RegExp(`At 64 users that adds ${Math.round(STATES.batch[64].kvBytes / 4.8e9)} ms`));
  assert.match(CAPTIONS[8], new RegExp(`At ${MAX_USERS} users the GPU's memory is full`));
});

test('GPU presets read hbmFor with its basis word, and the dense keys (Review Focus 3)', () => {
  assert.deepEqual(GPUS.map((g) => gpuLabel(gpuPreset(data, g.id))), ['H100 · 80 GB nominal', 'H200 · 141 GB nominal', 'B200 · 180 GB usable (192 nominal)']);
  GPUS.forEach((g) => {
    const p = gpuPreset(data, g.id);
    assert.deepEqual(p.hbm, hbmFor(entry('hardware', g.id)));
    assert.deepEqual(p.peak, { bf16: hw(g.id, 'bf16_dense_tflops'), fp8: hw(g.id, 'fp8_e4m3_dense_tflops') });
    assert.equal(p.bandwidthTBps, hw(g.id, 'hbm_tbps'));
    assert.equal(cells(view({ hw: g.id })).hbm, hbmText(p));
  });
  assert.equal(cells(view({ hw: 'b200' })).hbm, '180 GB usable (192 nominal)');
  assert.deepEqual(modelFor('fp8', gpuPreset(data, 'h200')), { ...RUNNING_EXAMPLE }, 'H200 FP8 is the running example');
  assert.throws(() => gpuPreset(data, 'a100'), RangeError);
  assert.throws(() => gpuPreset({ hardware: { entries: [] } }, 'h200'), RangeError);
  assert.throws(() => modelFor('fp4', gpuPreset(data, 'h200')), RangeError);
});

test('"Check my work" for the default state is the storyboard text; every state fills a template', () => {
  assert.equal(checkWork(analyze(INITIAL_STATE, gpuPreset(data, 'h200'))), CHECK_WORK);
  assert.equal(view({}).checkWork, CHECK_WORK);
  assert.equal(view({ phase: 'prefill' }).checkWork, [
    't_math = 2 × 70e9 × 1,000 ÷ 1,979 TFLOP/s = 70.7 ms',
    't_read = (70 GB weights + 17.1 GB activations) ÷ 4.8 TB/s = 87.1 GB ÷ 4.8 TB/s = 18.1 ms',
    't_step = max(70.7 ms, 18.1 ms) = 70.7 ms, compute-bound',
  ].join('\n'));
  assert.equal(view({ hw: 'h100', weights: 'bf16' }).checkWork, CHECK_WORK_DOES_NOT_FIT);
  assert.equal(view({ weights: 'bf16', context: 8192 }).checkWork, [
    'free           = 141 GB nominal − 140 GB weights = 1 GB',
    'cache per user = 8,192 tokens × 327,680 B = 2.68 GB',
    'users that fit = 1 GB ÷ 2.68 GB, rounded down = 0',
  ].join('\n'));
});

test('toy view: the default readouts equal their math/ functions for the same inputs (Review Focus 1)', () => {
  const s = stepTime({ ...L, tokens: 8, seqs: 8, context: 2048 });
  const free = freeHbmPerGpu({ hbmBytes: 141e9, weightBytes: 70e9, gpus: 1 });
  const cache = kvCacheBytes({ bytesPerToken: 327680, tokens: 2048 });
  assert.deepEqual(cells(view({})), {
    flops: formatFlops(s.flops), 'weights-bytes': formatBytes(70e9), 'act-bytes': formatBytes(s.actBytes), 'kv-bytes': formatBytes(s.kvBytes), bytes: formatBytes(s.bytes),
    'math-time': formatDuration(s.computeS), 'read-time': formatDuration(s.memoryS), 'step-time': formatDuration(s.timeS), bound: 'memory-bound',
    intensity: fixed1(arithmeticIntensity(s)), ridge: fixed1(ridgePoint(L)), 'tokens-needed': fixed1(tokensToComputeBound({ ...L, bytesPerElem: 1, k: 8192, n: 8192 })),
    hbm: '141 GB nominal', peak: '1,979 TFLOP/s', bandwidth: '4.8 TB/s',
    'per-user': formatCount(1 / s.timeS), 'per-gpu': formatInt(8 / s.timeS), 'max-users': formatInt(maxUsersPerGpu(free, cache)), 'cache-per-user': formatBytes(cache), free: formatBytes(free),
  });
  assert.deepEqual(Object.values(cells(view({}))), ['1.12 TFLOP', '70 GB', '137 MB', '5.37 GB', '75.5 GB', '566 µs', '15.7 ms', '15.7 ms', 'memory-bound',
    '14.8', '412.3', '217.1', '141 GB nominal', '1,979 TFLOP/s', '4.8 TB/s', '63.6', '509', '105', '671 MB', '71 GB']);
  const p = cells(view({ phase: 'prefill' }));
  assert.deepEqual([p['prefill-rate'], p.ttft, p.ceiling, p.flops, p.intensity], ['14,136', '70.7 ms', formatInt(prefillTokPerSecCeiling(L)), '140 TFLOP', '1,607.5']);
  assert.equal(p['kv-bytes'], undefined, 'prefill reads no KV in this model');
});

test('toy view: try-this states print the storyboard\'s numbers', () => {
  const c = (s) => cells(view(s));
  assert.deepEqual([1, 217, 512, 8192].map((t) => c({ phase: 'prefill', promptTokens: t }).bound), ['memory-bound', 'memory-bound', 'compute-bound', 'compute-bound']);
  assert.equal(c({ phase: 'prefill', promptTokens: 8192 })['step-time'], '580 ms');
  assert.deepEqual([c({ phase: 'prefill', weights: 'bf16' })['tokens-needed'], c({ phase: 'prefill', hw: 'b200', weights: 'bf16' })['tokens-needed'], c({ phase: 'prefill', hw: 'b200' })['tokens-needed']], ['217.0', '302.0', '302.0']);
  assert.deepEqual([8192, 32768, 131072].map((context) => [c({ context, users: 1e9 })['max-users'], c({ context, users: 1e9 })['per-gpu']]), [['26', '890'], ['6', '214'], ['1', '42']]);
  const b = c({ weights: 'bf16' });
  assert.deepEqual([b.free, b['max-users'], b['per-gpu'], b['step-time']], ['1 GB', '1', '34', '29.3 ms']);
  const b200 = c({ hw: 'b200', users: 1e9 });
  assert.deepEqual([b200['max-users'], b200['step-time'], b200['per-gpu']], ['163', '22.8 ms', '7,158']);
  assert.equal(c({ hw: 'b200', users: 1 })['step-time'], '8.84 ms');
});

test('does not fit: BF16 on an H100 prints the exact text and 0 users, never a negative or NaN (Review Focus 2)', () => {
  ['decode', 'prefill'].forEach((phase) => {
    const v = view({ hw: 'h100', weights: 'bf16', phase });
    assert.equal(v.fit, 'does not fit on one GPU: see [[serving-calculator]]');
    assert.equal(titled(v.fit), 'does not fit on one GPU: see Serving a 1T model');
    assert.equal(v.stepBar, null);
    assert.equal(v.memoryBar, null);
    assert.equal(v.curve, null);
    assert.deepEqual(cells(v), { hbm: '80 GB nominal', 'weights-bytes': '140 GB', 'max-users': '0' });
    assert.doesNotMatch(JSON.stringify(v), /NaN|Infinity|−\d/);
  });
  assert.equal(view({}).fit, null);
});

test('the users slider: stops end at the max that fits; a request above it clamps with a visible note (Review Focus 5)', () => {
  assert.deepEqual(usersStops(105), [1, 2, 4, 8, 16, 32, 64, 105]);
  assert.deepEqual(usersStops(26), [1, 2, 4, 8, 16, 26]);
  assert.deepEqual(usersStops(64), [1, 2, 4, 8, 16, 32, 64]);
  assert.deepEqual(usersStops(1), [1]);
  assert.deepEqual(usersStops(0), [0]);
  assert.deepEqual([usersLabel(8, 105), usersLabel(105, 105), usersLabel(1, 1), usersLabel(0, 0)], ['8 users', '105 users, max that fits', '1 user, max that fits', '0 users, none fit']);
  const v = view({ users: 64, context: 8192 });
  assert.deepEqual([v.users.value, v.users.stops.at(-1), v.users.label], [26, 26, '26 users, max that fits']);
  assert.equal(v.usersNote, 'Users clamped to 26, the most that fit at 8,192 tokens of context.');
  assert.equal(cells(v)['per-gpu'], '890');
  assert.equal(view({ users: 64 }).usersNote, '');
  assert.ok(CONTEXT_STOPS.every((context) => GPUS.every((g) => ['bf16', 'fp8'].every((weights) => {
    const u = view({ hw: g.id, weights, context, users: 1e9 }).users;
    return u.value <= u.stops.at(-1) && u.value >= 0;
  }))), 'a clamped slider never sits above its maximum');
  const none = view({ weights: 'bf16', context: 8192 });
  assert.equal(none.usersNote, 'No user fits: one user\'s cache at 8,192 tokens (2.68 GB) is more than the 1 GB free.');
  assert.deepEqual([none.users.stops, none.users.label, none.stepBar, cells(none)['max-users']], [[0], '0 users, none fit', null, '0']);
});

test('toy figures: the step bar\'s total is stepTime\'s; the memory bar adds up to the GPU\'s memory; the curve follows the state', () => {
  const v = view({});
  const s = stepTime({ ...L, tokens: 8, seqs: 8, context: 2048 });
  assert.deepEqual(v.stepBar.reading.map((p) => p.label), ['weights read', 'KV read', 'activations']);
  assert.ok(Math.abs(v.stepBar.reading.reduce((a, p) => a + p.s, 0) - s.memoryS) < 1e-15);
  assert.equal(v.stepBar.mathS, s.computeS);
  assert.equal(v.stepBar.scaleS, s.timeS);
  assert.equal(TOY_STEP_W > 0, true);
  assert.deepEqual(v.memoryBar.parts.map((p) => p.name), ['weights', 'KV cache', 'free']);
  assert.ok(Math.abs(v.memoryBar.parts.reduce((a, p) => a + p.value, 0) - 141e9) < 1);
  assert.deepEqual(view({ weights: 'bf16', context: 8192 }).memoryBar.parts.map((p) => p.name), ['weights', 'free'], 'a zero KV part is not drawn');
  assert.equal(v.curve.markers.filter((m) => m.followed).length, 1);
  assert.equal(v.curve.markers.length, usersStops(105).length);
  assert.equal(view({ phase: 'prefill' }).curve, null);
  assert.equal(view({ phase: 'prefill' }).memoryBar, null);
});

test('analyze rejects a preset that is not the state\'s GPU; PROMPT_STOPS match the storyboard', () => {
  assert.throws(() => analyze(INITIAL_STATE, gpuPreset(data, 'h100')), RangeError);
  assert.deepEqual(PROMPT_STOPS, [1, 16, 64, 217, 512, 1000, 2048, 8192]);
  assert.deepEqual(CONTEXT_STOPS, [512, 2048, 8192, 32768, 131072]);
});

test('formatters: FLOPs at 3 s.f. in GFLOP or TFLOP; one decimal grouped', () => {
  assert.deepEqual([formatFlops(140e9), formatFlops(1.12e12), formatFlops(140e12), formatFlops(1.1469e15)], ['140 GFLOP', '1.12 TFLOP', '140 TFLOP', '1,150 TFLOP']);
  assert.deepEqual([fixed1(1607.53), fixed1(2), fixed1(412.29)], ['1,607.5', '2.0', '412.3']);
});

test('the try-this list prints the storyboard\'s text with computed numbers (README lesson 34)', () => {
  const printed = tryThis(data).map(({ prompt, insight, rest }) => titled(`${prompt} → Insight: ${insight}${rest}`));
  assert.deepEqual(printed, TRY_THIS);
});

test('the math panel names the forward-pass factor from math/serving.js', () => {
  assert.match(LESSON.math.notes.join(' '), new RegExp(`factor ${FORWARD_FLOPS_PER_PARAM_TOKEN} in t_math .* training's ${FLOPS_PER_PARAM_TOKEN}`));
  assert.ok(LESSON.math.blocks.some((b) => b.tex.includes('hl-compute')) && LESSON.math.blocks.some((b) => b.tex.includes('hl-memory')));
});

test('nothing here mutates its inputs', () => {
  const before = JSON.stringify(data);
  const state = Object.freeze({ ...INITIAL_STATE, hw: 'b200', weights: 'bf16', users: 1e9 });
  toyView(state, data);
  checkWork(analyze(state, gpuPreset(data, 'b200')));
  tryThis(data);
  lessonFor(data);
  assert.equal(JSON.stringify(data), before);
});
