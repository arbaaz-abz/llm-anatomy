import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { bitsPerElement } from '../math/roofline.js';
import { formatBytes, formatDuration, formatInt } from '../math/core.js';
import { kvCacheBytes } from '../math/memory.js';
import { weightBytes, stepTime, freeHbmPerGpu, maxUsersPerGpu, hbmFor, RUNNING_EXAMPLE } from '../math/serving.js';
import { quantizeBlocks } from '../math/quant.js';
import { LESSON, lessonFor } from '../serving/concepts/quantization/content.js';
import { CHIPS, chipPreset, stagePreset, modelFormatOptions, usableModelFormat, chipLabel, NO_FP4 } from '../serving/concepts/quantization/hardware.js';
import {
  INITIAL_STATE, MODEL_FORMATS, DOES_NOT_FIT, blockView, checkWork, shrink, memoryPlan, kvBytesFor, lineSpec, gridFor,
  scaleText, restoredText, cellText, codeText, errorText, weightsFor,
} from '../serving/concepts/quantization/format.js';
import { toyView, blockFigure } from '../serving/concepts/quantization/toy-view.js';
import { tryThis } from '../serving/concepts/quantization/try-this.js';
import { BELOW, factRows } from '../serving/concepts/quantization/facts.js';
import { STAGE_CHIPS, VLLM, MODEL_CARDS, WEIGHTS, CONTEXT, PREFILL_TOKENS } from '../serving/concepts/quantization/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS, NO_FP4_NOTE } from './quantization-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const hw = (id, key) => data.hardware.entries.find((e) => e.id === id).facts[key]?.value;
const sv = (id, key) => data.serving.entries.find((e) => e.id === id).facts[key]?.value;
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key]?.value;
const view = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);
const titled = (text) => text.replace(/\[\[([a-z0-9-]+)\]\]/g, (_, slug) => graph.concepts.find((c) => c.slug === slug).title);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('the captions\' numbers equal the functions they repeat (README lesson 29)', () => {
  const nv4 = quantizeBlocks(WEIGHTS, { format: 'nvfp4', blockSize: 4 });
  const mx4 = quantizeBlocks(WEIGHTS, { format: 'mxfp4', blockSize: 4 });
  assert.match(CAPTIONS[6], new RegExp(`error ${errorText(nv4.meanAbsErr)}, not ${errorText(mx4.meanAbsErr)}`));
  assert.match(CAPTIONS[7], new RegExp(`${bitsPerElement('nvfp4')} bits per weight`));
  const h200 = chipPreset(data, 'h200');
  const bf = shrink({ modelFormat: 'bf16', kv: 'bf16' }, h200);
  const w4 = shrink({ modelFormat: 'w4a16', kv: 'bf16' }, h200);
  assert.match(CAPTIONS[7], new RegExp(`from ${formatBytes(bf.weights)} to ${formatBytes(w4.weights)}, and the H200 fits ${w4.users} users instead of one`));
  assert.equal(bf.users, 1);
  assert.match(CAPTIONS[3], new RegExp(cellText(WEIGHTS[7])));
  assert.match(CAPTIONS[2], /Three small weights/);
  assert.equal(quantizeBlocks(WEIGHTS, { format: 'int4', blockSize: 8 }).zeroed, 3);
  assert.equal(quantizeBlocks(WEIGHTS, { format: 'int4', blockSize: 4 }).meanAbsErr < quantizeBlocks(WEIGHTS, { format: 'int4', blockSize: 8 }).meanAbsErr * 0.55, true, '"the average error halves"');
});

test('9 facts rows; every placeholder resolves; the filled lesson has no dash or brace', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 9);
  lesson.facts.rows.forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.doesNotMatch(fillText(row.claim, data), /—/, `row ${i + 1}`);
    assert.ok(fillClaim(row.claim, data).sources.length >= 1, `row ${i + 1} links a source`);
  });
  const texts = [lesson.hook, ...lesson.intuition, ...lesson.takeaways, lesson.facts.framing, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  [LESSON.hook, ...LESSON.intuition, ...LESSON.takeaways].forEach((t) => assert.doesNotMatch(t, /[{}—]/, 'the data-free lesson prints no placeholder'));
});

test('facts rows print the data, with the formats the conventions name', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[1], /^DeepSeek-V4-Pro ships FP4 experts \+ FP8 rest \(FP8 block 128×128, UE8M0 scales\): FP8 is half/);
  assert.match(rows[2], /32 per block, one 8-bit power-of-two \(E8M0\) scale\. NVFP4: 16 per block, an E4M3 scale plus a per-tensor FP32 scale, 4\.5 bits per value/);
  assert.match(rows[3], /^gpt-oss-120b shipped post-trained with MXFP4 MoE weights: 116\.83B total and 5\.13B active parameters, and it fits one 80 GB GPU\.$/);
  assert.match(rows[4], /the checkpoint is about 865 GB \(size reported\)/);
  assert.match(rows[6], /gave 2\.71–2\.95× lower dollars per million tokens, between 30 and 90 tokens per second per user/);
  assert.match(rows[7], /cost 1% or less accuracy/);
  assert.match(rows[8], /\(0\.5 of BF16's\)\. A naive kernel dropped 128K needle-in-a-haystack accuracy from 91% to 13%; the final configuration recovers 97–98% of the baseline AUC at 128K/);
  assert.match(rows[8], /Throughput on Llama rose 14\.9%; below about 7,000 tokens/);
  assert.equal(fillClaim(lessonFor(data).facts.rows[3].claim, data).reported, false);
  assert.equal(fillClaim(lessonFor(data).facts.rows[4].claim, data).reported, true, 'checkpoint size is reported');
  assert.equal(fillClaim(lessonFor(data).facts.rows[5].claim, data).reported, true, 'Kimi INT4 is reported');
  assert.equal(factRows().length, 9);
});

test('no year is printed from an unconfirmed date (X-1); no N× below 3 significant figures (X-3)', () => {
  const all = [...lessonFor(data).facts.rows.map((r) => fillText(r.claim, data)), ...BELOW.flat().map((t) => fillText(t, data)), ...lessonFor(data).intuition, ...CAPTIONS];
  assert.deepEqual(all.filter((t) => /\(20\d\d\)|in 20\d\d\b/.test(t)), []);
  assert.equal(all.filter((t) => /\d×/.test(t)).every((t) => /\d\.\d\d–\d\.\d\d×|128×128/.test(t)), true);
});

test('hook, intuition and takeaways: the computed numbers equal the functions', () => {
  const lesson = lessonFor(data);
  const h200 = chipPreset(data, 'h200');
  const users = (bits) => memoryPlan({ bits, hbmBytes: h200.hbmBytes, kvBytesPerToken: kvBytesFor('bf16') }).users;
  assert.match(lesson.intuition[0], new RegExp(`room for one user at 2,048 tokens; in a 4-bit format it leaves room for ${users(bitsPerElement('nvfp4'))}\\.$`));
  assert.equal(users(16), 1);
  assert.match(lesson.intuition[2], /\(4\.5 bits per weight instead of 4\.25\)/);
  assert.match(lesson.takeaways[1], /at 4\.5 against 4\.25 bits per weight/);
  assert.equal(lesson.takeaways.length, 3);
  assert.match(lesson.hook, /^How can you throw away three-quarters of every weight's bits/);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('quantization')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(graph.concepts.find((c) => c.slug === 'quantization').prereqs, ['gpu-primer', 'prefill-decode']);
});

test('the stage constants equal data/*.json, read through chipPreset and the data entries (P3-R13 pattern)', () => {
  Object.keys(STAGE_CHIPS).forEach((id) => {
    const live = chipPreset(data, id);
    const stage = stagePreset(id);
    assert.deepEqual({ ...stage, peak: { ...stage.peak } }, { ...live, peak: { ...live.peak } }, id);
  });
  assert.deepEqual(VLLM, { baselinePct: sv('vllm-fp8-kv', 'baseline_niah_128k_pct'), naivePct: sv('vllm-fp8-kv', 'naive_niah_128k_pct'), recoveryPct: sv('vllm-fp8-kv', 'auc_recovery_128k_pct') });
  assert.deepEqual(MODEL_CARDS, { gptOss: { totalParams: model('gpt-oss-120b', 'total_params'), activeParams: model('gpt-oss-120b', 'active_params'), fitsGpuGb: model('gpt-oss-120b', 'fits_gpu_gb') }, v4Pro: { checkpointGb: model('deepseek-v4-pro', 'checkpoint_gb') } });
  assert.equal(RUNNING_EXAMPLE.peakTflops, hw('h200', 'fp8_e4m3_dense_tflops'), 'the running example is the H200 at FP8');
  assert.equal(RUNNING_EXAMPLE.bandwidthTBps, hw('h200', 'hbm_tbps'));
  assert.deepEqual([CONTEXT, PREFILL_TOKENS], [2048, 4096]);
});

test('chip presets read hbmFor with its basis word, dense peak keys and the bandwidth key (Review Focus 3)', () => {
  const b200 = chipPreset(data, 'b200');
  assert.deepEqual([b200.hbmBytes, b200.basis, b200.nominalGb], [180e9, 'usable', 192]);
  assert.deepEqual({ ...b200.peak }, { bf16: hw('b200', 'bf16_dense_tflops'), fp8: hw('b200', 'fp8_e4m3_dense_tflops'), fp4: hw('b200', 'nvfp4_dense_tflops') });
  const h200 = chipPreset(data, 'h200');
  assert.deepEqual([h200.hbmBytes, h200.basis], [hbmFor(data.hardware.entries.find((e) => e.id === 'h200')).bytes, 'nominal']);
  assert.equal(h200.peak.fp4, null);
  assert.equal(h200.bandwidthTBps, 4.8);
  assert.equal(chipLabel(h200), 'H200 · 141 GB nominal');
  assert.equal(chipLabel(b200), 'B200 · 180 GB usable (192 nominal)');
  assert.deepEqual(CHIPS.map((c) => c.id), ['h200', 'b200']);
  assert.throws(() => chipPreset(data, 'h100'), RangeError);
  assert.throws(() => chipPreset({ hardware: { entries: [] } }, 'h200'), RangeError);
  assert.throws(() => chipPreset({ hardware: { entries: [{ id: 'h200', facts: { hbm_gb: { value: 141 } } }] } }, 'h200'), /hbm_tbps/);
});

test('the H200 disables both FP4 chips with the note (P3-R12); a chip change keeps a usable format', () => {
  const off = (id) => modelFormatOptions(chipPreset(data, id)).filter((o) => o.disabled).map((o) => [o.value, o.note]);
  assert.deepEqual(off('h200'), [['nvfp4', NO_FP4], ['mxfp4', NO_FP4]]);
  assert.deepEqual(off('b200'), []);
  assert.equal(`NVFP4, MXFP4: ${NO_FP4}`, NO_FP4_NOTE);
  assert.deepEqual(modelFormatOptions(chipPreset(data, 'b200')).map((o) => o.label), ['BF16', 'FP8', '4-bit weights, 16-bit math', 'NVFP4', 'MXFP4']);
  assert.equal(usableModelFormat(chipPreset(data, 'h200'), 'nvfp4'), 'w4a16');
  assert.equal(usableModelFormat(chipPreset(data, 'h200'), 'fp8'), 'fp8');
  assert.equal(usableModelFormat(chipPreset(data, 'b200'), 'mxfp4'), 'mxfp4');
});

test('"Check my work" for the default state is the storyboard text, and fills the same template for every state', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.equal(view({}).checkWork, CHECK_WORK);
  assert.equal(checkWork({ ...INITIAL_STATE, blockSize: 4 }).split('\n')[0], 'scale = max|w| / 7 = 0.47 / 7 = 0.0671');
  const mx = checkWork({ ...INITIAL_STATE, format: 'mxfp4', blockSize: 4 }).split('\n');
  assert.deepEqual(mx, ['scale = 2^(⌊log2 max|w|⌋ − 2) = 2^(−2 − 2) = 0.0625', 'code = nearest FP4 value to w / scale = 0.47 / 0.0625 = 7.52 (past 6, clamped to 6) → 6', 'restored = code × scale = 6 × 0.0625 = 0.375']);
  assert.equal(checkWork({ ...INITIAL_STATE, format: 'mxfp4' }).split('\n')[0], 'scale = 2^(⌊log2 max|w|⌋ − 2) = 2^(1 − 2) = 0.50');
  const nv = checkWork({ ...INITIAL_STATE, format: 'nvfp4', blockSize: 4 }).split('\n');
  assert.deepEqual(nv, ['scale = max|w| / 6, rounded to FP8 = 0.47 / 6 → 0.0781', 'code = nearest FP4 value to w / scale = 0.47 / 0.0781 = 6.02 → 6', 'restored = code × scale = 6 × 0.0781 = 0.469']);
  assert.equal(checkWork({ ...INITIAL_STATE, outlier: false }).split('\n')[0], 'scale = max|w| / 7 = 0.64 / 7 = 0.0914');
});

test('the Round-a-block figure equals quantizeBlocks for every state (inputs not mutated)', () => {
  for (const format of ['int4', 'mxfp4', 'nvfp4']) {
    for (const blockSize of [8, 4, 2]) {
      for (const outlier of [true, false]) {
        const state = { ...INITIAL_STATE, format, blockSize, outlier };
        const frozen = Object.freeze({ ...state });
        const fig = blockFigure(frozen);
        const r = quantizeBlocks(weightsFor(outlier), { format, blockSize });
        assert.deepEqual(fig.restored.map((c) => c.value), r.restored);
        assert.deepEqual(fig.codes.map((c) => c.value), r.blocks.flatMap((b) => b.codes));
        assert.equal(fig.readouts.meanError, errorText(r.meanAbsErr));
        assert.equal(fig.blocks.length, 8 / blockSize);
        assert.deepEqual(fig.blocks.map((b) => b.scale), r.blocks.map((b) => scaleText(b.scale)));
        assert.equal(fig.line.points.filter((p) => p.followed).length, 1);
        assert.equal(fig.line.points.length, 8);
      }
    }
  }
  assert.deepEqual([...WEIGHTS], [0.12, -0.31, 0.05, 0.47, -0.08, 0.22, -0.64, 2.1]);
});

test('Round-a-block default and try-this readouts (storyboard §6)', () => {
  const r = (patch) => blockFigure({ ...INITIAL_STATE, ...patch }).readouts;
  assert.deepEqual(r({}), { meanError: '0.064', zeroed: '3', clipped: '0' });
  assert.deepEqual(r({ outlier: false }).meanError, '0.024');
  assert.deepEqual(r({ blockSize: 4 }), { meanError: '0.032', zeroed: '1', clipped: '0' });
  const grid = (format, blockSize) => r({ format, blockSize }).meanError;
  assert.deepEqual([4, 8, 2].map((b) => ['int4', 'mxfp4', 'nvfp4'].map((f) => grid(f, b))), [['0.032', '0.062', '0.029'], ['0.064', '0.073', '0.049'], ['0.011', '0.054', '0.017']]);
  assert.deepEqual([r({ format: 'mxfp4', blockSize: 4 }).clipped, r({ format: 'mxfp4', blockSize: 2 }).clipped], ['1', '2']);
  assert.deepEqual(['int4', 'mxfp4', 'nvfp4'].map((f) => r({ format: f, outlier: false }).meanError), ['0.024', '0.039', '0.017']);
});

test('the number line: 15 grid values, dots at value ÷ scale (the exact position), the clipped dot past 6', () => {
  assert.equal(gridFor('int4').length, 15);
  assert.deepEqual([...gridFor('mxfp4')], [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  const l = lineSpec({ format: 'mxfp4', blockSize: 4, outlier: true });
  assert.ok(Math.abs(l.exact[3] - 7.52) < 1e-9);
  assert.equal(l.snapped[3], 6);
  const d = lineSpec(INITIAL_STATE);
  assert.ok(Math.abs(d.exact[7] - 7) < 1e-9);
  assert.deepEqual(d.snapped, [0, -1, 0, 2, 0, 1, -2, 7]);
  assert.ok(d.exact.every((v) => v >= d.lo && v <= d.hi));
});

test('Shrink-a-model readouts equal the shared functions for the same inputs (Review Focus 1)', () => {
  const m = view({}).model;
  const free = freeHbmPerGpu({ hbmBytes: 141e9, weightBytes: weightBytes({ params: 70e9, bitsPerParam: 8 }), gpus: 1 });
  const kv = kvCacheBytes({ bytesPerToken: 327680, tokens: 2048 });
  const decode = stepTime({ ...RUNNING_EXAMPLE, tokens: 1, seqs: 1, context: 2048 });
  const prefill = stepTime({ ...RUNNING_EXAMPLE, tokens: 4096, seqs: 0, context: 0 });
  assert.deepEqual(m, {
    bits: '8', weights: formatBytes(70e9), memory: { value: '141 GB nominal', sub: '' }, free: formatBytes(free), kvPerUser: formatBytes(kv),
    users: formatInt(maxUsersPerGpu(free, kv)), decode: { value: formatDuration(decode.timeS), sub: 'memory-bound' }, prefill: { value: formatDuration(prefill.timeS), sub: 'compute-bound' },
  });
  assert.deepEqual([m.weights, m.free, m.kvPerUser, m.users, m.decode.value, m.prefill.value], ['70 GB', '71 GB', '671 MB', '105', '14.7 ms', '290 ms']);
});

test('Shrink-a-model: the try-this and storyboard rows on the H200 and the B200', () => {
  const row = (patch) => { const m = view(patch).model; return [m.users, m.decode.value, m.prefill.value]; };
  assert.deepEqual(['bf16', 'fp8', 'w4a16'].map((f) => row({ modelFormat: f })), [['1', '29.3 ms', '580 ms'], ['105', '14.7 ms', '290 ms'], ['151', '8.35 ms', '580 ms']]);
  assert.deepEqual(['bf16', 'fp8', 'w4a16'].map((f) => view({ modelFormat: f }).model.weights), ['140 GB', '70 GB', '39.4 GB']);
  assert.deepEqual(['bf16', 'fp8', 'nvfp4', 'mxfp4'].map((f) => row({ hw: 'b200', modelFormat: f })), [['59', '17.6 ms', '255 ms'], ['163', '8.84 ms', '127 ms'], ['209', '5.01 ms', '63.7 ms'], ['212', '4.73 ms', '63.7 ms']]);
  assert.deepEqual(['bf16', 'fp8', 'nvfp4', 'mxfp4'].map((f) => view({ hw: 'b200', modelFormat: f, kv: 'fp8' }).model.users), ['119', '327', '419', '425']);
  assert.deepEqual(['bf16', 'fp8', 'w4a16'].map((f) => view({ modelFormat: f, kv: 'fp8' }).model.users), ['2', '211', '302']);
  assert.deepEqual(view({ hw: 'b200' }).model.memory, { value: '180 GB usable', sub: 'of 192 GB nominal' });
  assert.deepEqual(['bf16', 'fp8', 'nvfp4', 'mxfp4'].map((f) => view({ modelFormat: f, hw: 'b200' }).model.bits), ['16', '8', '4.5', '4.25']);
  assert.equal(view({ hw: 'b200', modelFormat: 'nvfp4' }).model.prefill.sub, 'compute-bound');
  assert.equal(view({ modelFormat: 'w4a16' }).model.decode.sub, 'memory-bound');
});

test('a model that does not fit prints "does not fit" and 0 users, never a negative number (Review Focus 2)', () => {
  const tiny = { ...stagePreset('h200'), hbmBytes: 100e9, hbmGb: 100 };
  const m = shrink({ modelFormat: 'bf16', kv: 'bf16' }, tiny);
  assert.equal(m.fits, false);
  assert.equal(m.users, 0);
  assert.ok(m.freeBytes < 0);
  assert.equal(DOES_NOT_FIT, 'does not fit');
  const plan = memoryPlan({ bits: 16, hbmBytes: 100e9, kvBytesPerToken: 327680 });
  assert.deepEqual([plan.fits, plan.users], [false, 0]);
});

test('shrink rejects a format the chip cannot run, and an unknown format', () => {
  assert.throws(() => shrink({ modelFormat: 'nvfp4', kv: 'bf16' }, stagePreset('h200')), /no fp4 figure/);
  assert.throws(() => shrink({ modelFormat: 'fp8e4', kv: 'bf16' }, stagePreset('h200')), RangeError);
  assert.deepEqual(Object.keys(MODEL_FORMATS), ['bf16', 'fp8', 'w4a16', 'nvfp4', 'mxfp4']);
  assert.equal(MODEL_FORMATS.w4a16.bits, MODEL_FORMATS.nvfp4.bits, '"4-bit weights, 16-bit math" is counted at NVFP4\'s 4.5 bits');
});

test('formatters: scales at 4 decimals (2 at least), cells at 2, codes and a real minus', () => {
  assert.deepEqual([0.3, 0.0671, 0.34375, 0.0625, 0.5, 0.078125].map(scaleText), ['0.30', '0.0671', '0.3438', '0.0625', '0.50', '0.0781']);
  assert.deepEqual([0.6, 0.375, 0.4688, 2.1, -0.3].map(restoredText), ['0.60', '0.375', '0.469', '2.10', '−0.30']);
  assert.deepEqual([0, -0, 2.1, -0.31, 7, -2].map(cellText), ['0', '0', '2.10', '−0.31', '7', '−2']);
  assert.deepEqual([-0.5, 1.5, 6, 0].map(codeText), ['−0.5', '1.5', '6', '0']);
  assert.equal(errorText(0.06375), '0.064');
  assert.equal(blockView(INITIAL_STATE).block, 0);
  assert.equal(blockView({ ...INITIAL_STATE, blockSize: 2 }).block, 1);
});

test('the try-this list prints the storyboard\'s four prompts with the numbers from the functions', () => {
  const printed = tryThis(data).map((t) => titled(`${t.prompt} → Insight: ${t.insight}${t.rest}`));
  assert.deepEqual(printed, TRY_THIS);
  assert.equal(tryThis(data).length, 4);
  tryThis(data).forEach((t) => assert.match(t.prompt, /[.:]$/, 'each prompt ends in a full stop or colon before the arrow'));
});

test('no "—" and no bare slug in anything the toy prints', () => {
  for (const hw_ of ['h200', 'b200']) {
    for (const modelFormat of hw_ === 'h200' ? ['bf16', 'fp8', 'w4a16'] : Object.keys(MODEL_FORMATS)) {
      const v = view({ hw: hw_, modelFormat });
      assert.doesNotMatch(JSON.stringify(v), /—|NaN|undefined|Infinity/);
    }
  }
});

test('the vLLM FP8 KV post prints its date from the key; the frame 10 note and row 9 word the recovery once (quantization-4, XS-4)', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[8], /^vLLM's FP8 KV cache \(2026-04-22\) halves KV bytes/);
  const below = fillText(BELOW[9][0], data);
  assert.match(below, /FP8 KV-cache post \(2026-04-22\)/);
  assert.match(below, /the final configuration recovers 97–98% of the baseline AUC at 128K/);
  assert.doesNotMatch(below, /the fix recovers/);
});

test('the B200 basis words read one way (XS-6) and references say frame / try-this (XS-7)', () => {
  assert.equal(fillText(BELOW[7][0], data).includes('B200: 180 GB usable (192 nominal)'), true);
  const takeaways = LESSON.takeaways.join(' ');
  assert.match(takeaways, /\(frames 2–5, try-this 1\)/);
  assert.doesNotMatch([takeaways, ...tryThis(data).map((t) => t.rest ?? '')].join(' '), /\(steps? \d/);
});
