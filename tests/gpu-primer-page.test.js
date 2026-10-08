import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { matmulCost, arithmeticIntensity, ridgePoint, attainableTflops, rooflineTime, tokensToComputeBound, bytesPerElement } from '../math/roofline.js';
import { formatDuration, formatRatio, formatBytes, formatCount } from '../math/core.js';
import { LESSON, lessonFor } from '../training/concepts/gpu-primer/content.js';
import { CHIPS, chipPreset, formatOptions, usableFormat, NO_FP4, NO_BF16_FP8 } from '../training/concepts/gpu-primer/hardware.js';
import { analyze, checkWork, verdictText, rangeText, int, fixed1, pct2, pctProse, INITIAL_STATE, TOKEN_STOPS } from '../training/concepts/gpu-primer/format.js';
import { toyView, lanesSpec, conflictNote } from '../training/concepts/gpu-primer/toy-view.js';
import { tryThis } from '../training/concepts/gpu-primer/try-this.js';
import { BELOW, factRows, h100Idle } from '../training/concepts/gpu-primer/facts.js';
import { STAGE_CHIPS, H100_SMS, SRAM_PER_SM } from '../training/concepts/gpu-primer/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './gpu-primer-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const hw = (id, key) => data.hardware.entries.find((e) => e.id === id).facts[key]?.value;
const view = (state) => toyView({ ...INITIAL_STATE, ...state }, data);
const titled = (text) => text.replace(/\[\[([a-z0-9-]+)\]\]/g, (_, slug) => graph.concepts.find((c) => c.slug === slug).title);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('12 facts rows; every placeholder resolves; the filled lesson has no dash', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 12);
  lesson.facts.rows.forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.doesNotMatch(fillText(row.claim, data), /—/, `row ${i + 1}`);
  });
  const texts = [lesson.hook, ...lesson.intuition, ...lesson.takeaways, lesson.facts.framing, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(fillText(lesson.facts.rows[0].claim, data), /^H100 SXM \(2022\): 80 GB HBM3 \(nominal\), 3\.35 TB\/s, 989 TFLOPS BF16, 1,979 FP8 \(reported: 2 × BF16\), NVLink 900 GB\/s both directions \(450 each way, .*\); ridge 295\.$/);
  assert.match(fillText(lesson.facts.rows[2].claim, data), /ridge 206:/);
  assert.match(fillText(lesson.facts.rows[7].claim, data), /HBM bandwidth 19\.2–22 TB\/s/);
  assert.match(fillText(lesson.facts.rows[10].claim, data), /a 12B model on 10T tokens in NVFP4 and matched its FP8 loss \(2025\)/);
  assert.equal(fillText(lesson.facts.rows[11].claim, data), 'FlashAttention (2022) tiles attention so the score matrix stays in on-chip SRAM instead of HBM: the same math, far fewer HBM bytes.');
  assert.match(factRows(null)[0].claim, /ridge —\./, 'without data a derived number prints a dash, never a stale one');
});

test('hook, intuition and takeaways fill from data and agree with the analysis', () => {
  const lesson = lessonFor(data);
  assert.match(lesson.hook, /rated at 989 trillion operations a second/);
  assert.match(lesson.intuition[0], /^A GPU is two machines glued together\. One is compute: 132 small processors .* An H100 can do 295 operations in the time/);
  assert.match(lesson.intuition[1], /far below 295, and the tensor cores sit idle 98\.6% of the time/);
  assert.equal(h100Idle(data), '98.6%');
  assert.match(lesson.takeaways[0], /peak ÷ bandwidth, 295 on an H100/);
  assert.match(lesson.takeaways[1], /\(1\.35% of peak at 4\)/);
  assert.match(lesson.takeaways[2], /\(4\.5 bits in NVFP4\)/);
});

test('Next lists exactly the lessons that take this one as a prereq, across tracks (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('gpu-primer')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the captions\' hardware numbers equal the data they repeat (README lesson 29)', () => {
  assert.match(CAPTIONS[0], new RegExp(`HBM holds ${hw('h100', 'hbm_gb')} GB`));
  assert.match(CAPTIONS[1], new RegExp(`at ${hw('h100', 'bf16_dense_tflops')} trillion operations a second.*only ${hw('h100', 'hbm_tbps')} trillion bytes`));
  assert.match(CAPTIONS[5], new RegExp(`as long as ${Math.round(ridgePoint({ peakTflops: hw('h100', 'bf16_dense_tflops'), bandwidthTBps: hw('h100', 'hbm_tbps') }))} FLOPs`));
  assert.match(CAPTIONS[10], new RegExp(`grew ${formatRatio(hw('b300', 'nvfp4_dense_tflops') / hw('b300', 'bf16_dense_tflops')).replace('×', '')}-fold while bytes shrank ${formatRatio(2 / bytesPerElement('nvfp4')).replace('×', '')}-fold`));
  const b300 = tokensToComputeBound({ peakTflops: hw('b300', 'nvfp4_dense_tflops'), bandwidthTBps: hw('b300', 'hbm_tbps'), bytesPerElem: bytesPerElement('nvfp4'), k: 8192, n: 8192 });
  assert.ok(Math.abs(b300 - 600) < 10, 'about 600 tokens');
});

test('the stage constants equal data/hardware.json, read through chipPreset (P3-R13 pattern)', () => {
  Object.entries(STAGE_CHIPS).forEach(([id, c]) => {
    const p = chipPreset(data, id);
    assert.equal(p.label, c.label);
    assert.deepEqual(p.bandwidths, c.bandwidths, id);
    Object.entries(c.peak).forEach(([f, v]) => assert.equal(p.peak[f], v, `${id} ${f}`));
    if (c.hbmGb) assert.equal(p.hbm.gb, c.hbmGb);
  });
  assert.equal(hw('h100', 'sm_count'), H100_SMS);
  assert.equal(data.hardware.entries.find((e) => e.id === 'memory-hierarchy').facts.sram_per_sm.value, SRAM_PER_SM);
});

test('chip presets read the dense keys the Conventions name, never a sparse one (Review Focus 5)', () => {
  const p = chipPreset(data, 'h100');
  assert.deepEqual(p.peak, { bf16: hw('h100', 'bf16_dense_tflops'), fp8: hw('h100', 'fp8_e4m3_dense_tflops'), fp4: null });
  assert.deepEqual(p.formats, { bf16: 'bf16', fp8: 'fp8_e4m3', fp4: 'nvfp4' });
  assert.deepEqual(p.hbm, { gb: 80, basis: 'nominal' });
  assert.equal(chipPreset(data, 'mi355x').peak.fp4, hw('mi355x', 'mxfp4_dense_tflops'));
  assert.equal(chipPreset(data, 'mi355x').formats.fp4, 'mxfp4');
  assert.equal(chipPreset(data, 'b300').peak.fp4, hw('b300', 'nvfp4_dense_tflops'));
  assert.deepEqual(chipPreset(data, 'rubin').bandwidths, [22, 19.2]);
  assert.deepEqual(chipPreset(data, 'rubin').peak, { bf16: null, fp8: null, fp4: 35000 });
  assert.equal(CHIPS.length, 7);
  CHIPS.forEach((c) => assert.doesNotThrow(() => chipPreset(data, c.id)));
  assert.throws(() => chipPreset(data, 'a100'), RangeError);
  assert.throws(() => chipPreset({ hardware: { entries: [] } }, 'h100'), RangeError);
});

test('disabled formats carry their reason (P3-R12); a chip change keeps a usable format', () => {
  const off = (id) => formatOptions(chipPreset(data, id)).filter((o) => o.disabled).map((o) => [o.value, o.note]);
  ['h100', 'h200', 'tpu-v7'].forEach((id) => assert.deepEqual(off(id), [['fp4', NO_FP4]], id));
  ['b200', 'b300', 'mi355x'].forEach((id) => assert.deepEqual(off(id), [], id));
  assert.deepEqual(off('rubin'), [['bf16', NO_BF16_FP8], ['fp8', NO_BF16_FP8]]);
  assert.equal(usableFormat(chipPreset(data, 'h100'), 'fp4'), 'bf16');
  assert.equal(usableFormat(chipPreset(data, 'rubin'), 'bf16'), 'fp4');
  assert.equal(usableFormat(chipPreset(data, 'b300'), 'fp8'), 'fp8');
});

test('"Check my work" for the default state is the storyboard text, and fills the same template for every state', () => {
  assert.equal(checkWork(INITIAL_STATE, chipPreset(data, 'h100')), CHECK_WORK);
  assert.equal(view({}).checkWork, CHECK_WORK);
  const rubin = checkWork({ chip: 'rubin', fmt: 'fp4', tokens: 4 }, chipPreset(data, 'rubin')).split('\n');
  assert.equal(rubin.length, 5);
  assert.equal(rubin[3], 'ridge     = 35,000 ÷ 22 = 1,590.9, or ÷ 19.2 = 1,822.9 FLOPs/byte → memory-bound');
  assert.equal(rubin[4], 'crossing  = the token count whose intensity reaches 1,590.9 or 1,822.9 = 502.3 or 586.1 tokens');
  assert.equal(checkWork({ chip: 'b300', fmt: 'fp4', tokens: 512 }, chipPreset(data, 'b300')).split('\n')[1], 'bytes     = 0.5625 × (512 × 8,192 + 8,192 × 8,192 + 512 × 8,192) = 42,467,328');
  assert.match(checkWork({ chip: 'h100', fmt: 'bf16', tokens: 512 }, chipPreset(data, 'h100')), /→ compute-bound\n/);
});

test('toy view: the default state prints the storyboard\'s numbers, each from its math/ function (Review Focus 4)', () => {
  const r = view({}).readouts;
  const cost = matmulCost({ m: 4, k: 8192, n: 8192, bytesPerElem: 2 });
  const roof = { peakTflops: hw('h100', 'bf16_dense_tflops'), bandwidthTBps: hw('h100', 'hbm_tbps') };
  const time = rooflineTime({ ...cost, ...roof });
  const I = arithmeticIntensity(cost);
  assert.deepEqual(r, {
    flops: formatCount(cost.flops), bytes: formatBytes(cost.bytes), intensity: fixed1(I), ridge: int(ridgePoint(roof)),
    tokensNeeded: fixed1(tokensToComputeBound({ ...roof, bytesPerElem: 2, k: 8192, n: 8192 })), verdict: 'memory-bound',
    attainable: `${fixed1(attainableTflops({ intensity: I, ...roof }))} TFLOPS`, peakShare: pct2(attainableTflops({ intensity: I, ...roof }), roof.peakTflops),
    memoryTime: formatDuration(time.memoryS), computeTime: formatDuration(time.computeS),
    timeRatio: { value: formatRatio(time.memoryS / time.computeS), sub: 'memory takes longer' },
  });
  assert.deepEqual([r.flops, r.bytes, r.intensity, r.ridge, r.tokensNeeded, r.attainable, r.peakShare, r.memoryTime, r.timeRatio.value],
    ['537M', '134 MB', '4.0', '295', '318.2', '13.4 TFLOPS', '1.35%', '40.1 µs', '73.9×']);
  assert.equal(view({}).lanesWidth, 'lanes share one time axis; full width = 40.1 µs');
  assert.equal(view({}).conflict, '');
});

test('the H100 compute time prints 0.543 µs (storyboard §5 frame 7, §6)', () => {
  assert.equal(view({}).readouts.computeTime, '0.543 µs');
});

test('toy view: try-this states (storyboard §6)', () => {
  const r = (s) => view(s).readouts;
  assert.deepEqual([r({ tokens: 64 }).peakShare, r({ tokens: 256 }).peakShare, r({ tokens: 512 }).peakShare], ['21.34%', '81.61%', '100.00%']);
  assert.deepEqual([r({ tokens: 256 }).verdict, r({ tokens: 512 }).verdict], ['memory-bound', 'compute-bound']);
  assert.deepEqual([r({ tokens: 4096 }).computeTime, r({ tokens: 4096 }).memoryTime, r({ tokens: 4096 }).intensity], ['556 µs', '80.1 µs', '2,048.0']);
  assert.deepEqual([r({ tokens: 256, fmt: 'fp8' }).intensity, r({ tokens: 256, fmt: 'fp8' }).ridge, r({ tokens: 256, fmt: 'fp8' }).memoryTime, r({ tokens: 256, fmt: 'fp8' }).tokensNeeded], ['481.9', '591', '21.3 µs', '318.3']);
  assert.equal(r({ tokens: 4, fmt: 'fp8' }).memoryTime, '20.1 µs');
  const b300 = (fmt) => r({ chip: 'b300', fmt, tokens: 512 });
  assert.deepEqual([b300('bf16').verdict, b300('bf16').intensity, b300('bf16').ridge, b300('bf16').tokensNeeded], ['compute-bound', '455.1', '313', '338.3']);
  assert.deepEqual([b300('fp4').verdict, b300('fp4').intensity, b300('fp4').ridge, b300('fp4').tokensNeeded], ['memory-bound', '1,618.2', '1,875', '605.3']);
  assert.deepEqual(['h100', 'b200', 'b300', 'h200'].map((chip) => r({ chip }).ridge), ['295', '281', '313', '206']);
  assert.deepEqual([r({ chip: 'h200' }).tokensNeeded, r({ chip: 'b200' }).tokensNeeded, r({ chip: 'b200', fmt: 'fp4' }).tokensNeeded], ['217.0', '302.0', '342.9']);
  assert.deepEqual([r({ chip: 'mi355x', fmt: 'fp4' }).ridge, r({ chip: 'mi355x', fmt: 'fp4' }).tokensNeeded], ['1,250', '361.3']);
  assert.deepEqual([r({ chip: 'tpu-v7' }).ridge, r({ chip: 'tpu-v7', fmt: 'fp8' }).tokensNeeded], ['313', '338.4']);
});

test('toy view: Rubin prints both bandwidth ends, and a split verdict when they disagree', () => {
  const v = view({ chip: 'rubin', fmt: 'fp4' });
  assert.deepEqual([v.readouts.ridge, v.readouts.tokensNeeded, v.readouts.verdict], ['1,591–1,823', '502.3–586.1', 'memory-bound']);
  assert.equal(v.conflict, 'bandwidth: sources conflict, 19.2 or 22 TB/s');
  assert.deepEqual(v.plot.ridgeRange.map(Math.round), [1591, 1823]);
  assert.equal(v.plot.bandwidthTBps, 22);
  assert.deepEqual(v.lanes.lanes.map((l) => l.label), ['memory, 22 TB/s', 'memory, 19.2 TB/s', 'compute']);
  assert.ok(v.lanes.lanes.every((l) => l.segments.every((s) => s.kind !== 'idle')), 'alternative lanes get no idle remainder');
  const split = view({ chip: 'rubin', fmt: 'fp4', tokens: 512 }).readouts;
  assert.equal(split.verdict, 'compute-bound at 22 TB/s, memory-bound at 19.2 TB/s');
  assert.match(split.timeRatio.sub, /compute longer at 22 TB\/s · memory longer at 19\.2 TB\/s/);
  assert.equal(conflictNote(chipPreset(data, 'h100')), '');
});

test('toy view: lanes put the idle remainder on the shorter lane; the plot follows this multiply and a residual add', () => {
  const v = view({});
  assert.deepEqual(v.lanes.lanes.map((l) => l.segments.map((s) => s.kind)), [['memory'], ['compute', 'idle']]);
  assert.deepEqual(v.lanes.lanes.map((l) => l.segments[0].label), ['40.1 µs', formatDuration(536870912 / 989e12)]);
  const big = view({ tokens: 4096 });
  assert.deepEqual(big.lanes.lanes.map((l) => l.segments.map((s) => s.kind)), [['memory', 'idle'], ['compute']]);
  assert.equal(big.lanesWidth, 'lanes share one time axis; full width = 556 µs');
  assert.deepEqual(v.plot.points.map((p) => [p.label, Number(p.intensity.toFixed(4)), p.followed ?? false]), [['4 tokens', 3.9961, true], ['residual add', 0.1667, false]]);
  assert.equal(view({ tokens: 1 }).plot.points[0].label, '1 token');
  assert.equal(v.plot.title, 'H100 · BF16');
  assert.equal(view({ chip: 'mi355x', fmt: 'fp4' }).plot.title, 'MI355X · MXFP4');
  assert.deepEqual(lanesSpec(analyze({ chip: 'h100', fmt: 'bf16', tokens: 4 }, chipPreset(data, 'h100'))).full.toFixed(3), '40.104');
});

test('analyze rejects a state the chip cannot run, and a token count off the slider', () => {
  assert.throws(() => analyze({ chip: 'h100', fmt: 'fp4', tokens: 4 }, chipPreset(data, 'h100')), RangeError);
  assert.throws(() => analyze({ chip: 'h100', fmt: 'bf16', tokens: 3 }, chipPreset(data, 'h100')), RangeError);
  assert.throws(() => analyze({ chip: 'b200', fmt: 'bf16', tokens: 4 }, chipPreset(data, 'h100')), RangeError);
  assert.equal(TOKEN_STOPS.length, 14);
  assert.deepEqual([TOKEN_STOPS[0], TOKEN_STOPS.at(-1)], [1, 8192]);
});

test('formatters: ranges, one decimal, shares at 2 decimals and at 3 significant figures', () => {
  assert.equal(rangeText([1822.9, 1590.9], int), '1,591–1,823');
  assert.equal(rangeText([4, 4.01], fixed1), '4.0');
  assert.deepEqual([fixed1(1618.17), int(-3.4), pct2(13.39, 989), pctProse(211.1, 989), pctProse(13.39, 989)], ['1,618.2', '−3', '1.35%', '21.3%', '1.35%']);
  const a = analyze({ chip: 'h100', fmt: 'bf16', tokens: 4 }, chipPreset(data, 'h100'));
  assert.equal(verdictText(a), 'memory-bound');
});

test('the try-this list prints the storyboard\'s text with computed numbers (README lesson 34)', () => {
  const printed = tryThis(data).map(({ prompt, insight, rest }) => titled(`${prompt} → Insight: ${insight}${rest}`));
  assert.deepEqual(printed, TRY_THIS);
});

test('nothing here mutates its inputs', () => {
  const before = JSON.stringify(data.hardware);
  const state = Object.freeze({ ...INITIAL_STATE, chip: 'rubin', fmt: 'fp4' });
  toyView(state, data);
  checkWork(state, chipPreset(data, 'rubin'));
  tryThis(data);
  lessonFor(data);
  assert.equal(JSON.stringify(data.hardware), before);
});
