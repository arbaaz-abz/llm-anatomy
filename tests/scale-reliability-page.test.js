import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { formatDuration } from '../math/core.js';
import { runPlan, trainingFlops, wallClockDays, mfuFrom, checkpointLossParts, FLOPS_PER_PARAM_TOKEN } from '../math/scale.js';
import { FORMATS } from '../math/roofline.js';
import { LESSON, lessonFor } from '../training/concepts/scale-reliability/content.js';
import { BELOW, factRows } from '../training/concepts/scale-reliability/facts.js';
import * as N from '../training/concepts/scale-reliability/numbers.js';
import { llamaRun, SAME_SCALE_GPU_HOURS } from '../training/concepts/scale-reliability/runs.js';
import { sci, int, pct, pctOf, bandText, gpuHours, hoursText, daysText, minutesText, dollarsM, dollarsExact, signedPct, tflops } from '../training/concepts/scale-reliability/format.js';
import { toyView, tryThis, planFor, peakFor, recordLine, barParts, CHIPS, PRESETS, PRESET_FIELDS, INITIAL_STATE, INVALID_LINE, NEVER_LINE } from '../training/concepts/scale-reliability/toy-view.js';
import { windowRun } from '../training/concepts/scale-reliability/frames-failures.js';
import { CAPTIONS, DEFAULT_READOUTS, LOSS_SPLIT, RECORD_LLAMA, RECORD_LLAMA_MFU_38, RECORD_DEEPSEEK, NO_RECORD, INVALID, DEEPSEEK_NOTE, TRY_THIS } from './scale-reliability-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key];
const hw = (id, key) => data.hardware.entries.find((e) => e.id === id).facts[key];
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a / b - 1) < eps, `${a} ≉ ${b}`);
const deepseekState = { ...INITIAL_STATE, preset: 'deepseek', ...Object.fromEntries(PRESET_FIELDS.map((k) => [k, PRESETS.deepseek[k]])) };

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('13 facts rows, every placeholder resolves, no unfilled "—", every row but none links a source', () => {
  assert.equal(LESSON.facts.rows.length, 13);
  lessonFor(data).facts.rows.forEach((row, i) => {
    const filled = fillClaim(row.claim, data);
    assert.deepEqual(filled.missing, [], `row ${i + 1}`);
    assert.ok(filled.sources.length >= 1, `row ${i + 1} has a source`);
    assert.doesNotMatch(filled.segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`);
  });
  assert.match(factRows(null)[9].claim, /— GB200|\(precision not stated/, 'without data the ratio prints a dash, never a stale number');
});

test('the filled rows print the storyboard\'s numbers', () => {
  const text = (i) => { const f = fillClaim(factRows(data)[i].claim, data); return f.segments.map((s) => s.text).join(''); };
  assert.match(text(0), /^Llama 3\.1 405B \(2024\): 15\.6T tokens, 3\.8 × 10²⁵ FLOPs, up to 16,384 H100s\.$/);
  assert.match(text(2), /MFU while training: 38–43% \(43% with TP8/);
  assert.match(text(3), /466 interruptions, 47 planned and 419 unexpected; 78% hardware, GPU issues 58\.7% of the unexpected; effective training time >90%\./);
  assert.match(text(5), /^DeepSeek-V3 \(2024\): 2\.788M H800-hours in all, 2\.664M for pre-training \(180K per trillion tokens on 2,048 GPUs\), priced at \$2 per GPU-hour: \$5\.576M/);
  assert.match(text(9), /1,648 TFLOPS per GPU .* 2\.72× GB200 NVL72's 606/);
  assert.match(text(10), /MiMo-V2-Flash trained in FP8 over 27T tokens; Nemotron 3 Super and Ultra were both pre-trained in NVFP4/);
  assert.match(text(12), /^Kimi K2 \(2025\): 0 loss spikes over 15\.5T tokens/);
});

test('Next lists exactly the lessons that take this one as a prereq (none: the GPUs & scale section ends here)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('scale-reliability')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(graph.concepts.find((c) => c.slug === 'scale-reliability').prereqs, ['cluster-topology', 'scaling-laws']);
});

test('dated text: hook, intuition and the notes under the stage fill from data, nothing missing', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  [LESSON.hook, ...LESSON.intuition, ...BELOW.flat()].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(lesson.hook, /16,384 H100s/);
  assert.match(lesson.intuition[0], /6 × 405B × 15\.6T = 3\.8 × 10²⁵.*989 TFLOPS.*30\.84M/);
  assert.match(lesson.intuition[2], /2\.788M H800-hours at \$2 each, \$5\.576M/);
  assert.equal(lesson.animation.belowFor(42).length, 0);
  assert.match(lesson.animation.belowFor(6)[0], /Behemoth trained on 32K GPUs/);
});

test('the text the data cannot fill matches the data it repeats (README lesson 29, P3-R13)', () => {
  const llama = (key) => model('llama-3.1-405b', key).value;
  assert.equal(llama('training_flops'), 3.8e25);
  near(trainingFlops({ params: llama('total_params'), tokens: llama('pretrain_tokens') }), llama('training_flops'), 0.01);
  assert.deepEqual(llama('mfu_bf16'), N.LLAMA.mfuBand);
  assert.equal(bandText(N.LLAMA.mfuBand), '38–43%');
  assert.match(model('llama-3.1-405b', 'mfu_bf16').note, /43%.*430 TFLOPS.*41%.*400.*38%.*380/);
  assert.match(llama('reliability_notes'), /3 interruptions needed manual intervention/);
  assert.equal(llama('effective_time'), `>${N.LLAMA.effective * 100}%`);
  assert.equal(daysText(wallClockDays({ gpuHours: 10.647e6, gpus: llama('training_gpus') })), '27.1 days', 'the hook says 27 days');
  assert.match(model('deepseek-v3', 'fp8_loss_error').value.replace(/\s/g, ''), new RegExp(`^${N.FP8.lossError.replace(/\s/g, '').replace('<', '<')}$`));
  assert.equal(hw('gb300-nvl72', 'megatron_dsv3_tflops_per_gpu').value, 1648);
});

test('numbers.js restates the data and the page pins each constant to its entry (P3-R13)', () => {
  const llama = (key) => model('llama-3.1-405b', key).value;
  const ds = (key) => model('deepseek-v3', key).value;
  assert.deepEqual([N.LLAMA.params, N.LLAMA.tokens, N.LLAMA.gpus, N.LLAMA.gpuHours], [llama('total_params'), llama('pretrain_tokens'), llama('training_gpus'), llama('training_gpu_hours')]);
  assert.deepEqual([N.LLAMA.interruptions, N.LLAMA.hardwareShare, N.LLAMA.gpuShare], [llama('interruptions_54d'), llama('interruptions_hardware_share'), llama('interruptions_gpu_share')]);
  assert.deepEqual([N.DEEPSEEK.params, N.DEEPSEEK.tokens, N.DEEPSEEK.gpus], [ds('active_params'), ds('pretrain_tokens'), ds('training_gpus')]);
  assert.deepEqual([N.DEEPSEEK.pretrainGpuHours, N.DEEPSEEK.totalGpuHours], [ds('pretrain_gpu_hours'), ds('training_gpu_hours')]);
  assert.equal(N.DEEPSEEK.dollarsPerGpuHour, ds('training_cost_usd_reported') / ds('training_gpu_hours'));
  assert.equal(`${N.DEEPSEEK.commSms} of ${N.DEEPSEEK.smsPerGpu}`, ds('comm_sms'));
  assert.equal(N.DEEPSEEK.smsPerGpu, hw('h100', 'sm_count').value);
  assert.equal(N.BEHEMOTH.tflopsPerGpu, model('llama-4-behemoth', 'achieved_tflops_per_gpu').value);
  assert.deepEqual([N.H100.bf16, N.H100.fp8], [hw('h100', 'bf16_dense_tflops').value, hw('h100', 'fp8_e4m3_dense_tflops').value]);
  assert.equal(N.LLAMA.windowDays, 54);
  assert.equal(model('llama-3.1-405b', 'interruptions_54d').note.includes('54-day'), true);
  assert.equal(N.PER_GPU_MTBF_H, 54 * 24 / 419 * 16384);
  assert.equal(CHIPS.every((c) => peakFor(c.value, data) === hw(c.entry, c.key).value), true);
  assert.deepEqual(CHIPS.map((c) => peakFor(c.value, data)), [989, 1979, 2250, 5000]);
  assert.equal(N.STOPS.perGpuMtbfH.includes(Math.round(N.PER_GPU_MTBF_H)), true);
  [[N.STOPS.params, [N.LLAMA.params, N.DEEPSEEK.params]], [N.STOPS.tokens, [N.LLAMA.tokens, N.DEEPSEEK.tokens]], [N.STOPS.gpus, [N.LLAMA.gpus, N.DEEPSEEK.gpus, 32768, N.BIG_GPUS]]]
    .forEach(([stops, values]) => values.forEach((v) => assert.ok(stops.includes(v), `${v} is a slider stop`)));
  assert.ok(N.STOPS.intervalMin.includes(5) && N.STOPS.intervalMin.includes(60) && N.STOPS.saveS.includes(15) && N.STOPS.restartMin.includes(30));
});

test('the stage numbers: 6ND, GPU-hours at peak, the shared scale and the frame 8 failures', () => {
  const flops = trainingFlops({ params: N.LLAMA.params, tokens: N.LLAMA.tokens });
  near(flops, 3.7908e25);
  assert.equal(FLOPS_PER_PARAM_TOKEN, 6);
  assert.equal(sci(flops), '3.79 × 10²⁵');
  assert.equal(gpuHours(SAME_SCALE_GPU_HOURS), '36.97M');
  assert.equal(N.FAILURE_DAYS.length, 419);
  assert.ok(N.FAILURE_DAYS.every((d, i) => d >= 0 && d < 54 && (i === 0 || d >= N.FAILURE_DAYS[i - 1])), 'sorted, inside the 54 days');
  assert.deepEqual(N.FAILED_GPUS.map((g) => Number.isInteger(g) && g >= 0 && g < 8), [true, true]);
  assert.equal(N.OTHER_NODES, 2046);
  const rp = mfuFrom({ flops, gpuHours: N.LLAMA.gpuHours, peakTflops: N.H100.bf16 });
  assert.equal(pct(rp), '34.5%');
  assert.equal(pct(rp / N.LLAMA.effective), '38.4%');
  assert.equal(pctOf(100, 140), '71%');
  assert.equal(pctOf(20, 132), '15%');
  assert.equal(dollarsExact(5.576e6), '$5.576M');
  assert.equal(daysText(wallClockDays({ gpuHours: N.DEEPSEEK.pretrainGpuHours, gpus: 2048 })), '54.2 days');
});

test('frame 9\'s window: saves every 13.6 min, the work since the last one turns lost at the failure, then a restart gap', () => {
  const interval = llamaRun().intervalH * 60;
  const run = windowRun(N.WINDOW_MIN, interval, N.STAND_IN.restartMin);
  assert.deepEqual(run.ticks.filter((t) => t.label === 'save').map((t) => Math.round(t.t * 10) / 10), [13.6, 27.2, 40.9, 54.5, 68.1, 89.6]);
  assert.deepEqual(run.segments.map((s) => s.kind), ['compute', 'compute', 'compute', 'compute', 'compute', 'lost', 'compute']);
  assert.deepEqual(run.gaps, [{ from: 73, to: 76, label: 'restart' }]);
  const early = windowRun(50, interval, 3);
  assert.deepEqual([early.failed, early.gaps.length, early.segments.at(-1).to], [false, 0, 50]);
  assert.equal(windowRun(70, interval, 3).segments.at(-1).kind, 'compute', 'before the failure the last stretch is still compute');
  const parts = checkpointLossParts({ intervalH: interval / 60, saveH: 30 / 3600, restartH: 3 / 60, mtbfH: llamaRun().mtbfH });
  assert.deepEqual([pct(parts.save), pct(parts.lostWork), pct(parts.restart), pct(parts.total)], ['3.7%', '3.7%', '1.6%', '9.0%']);
});

test('toy view: the default state prints the storyboard\'s numbers and record line', () => {
  const v = toyView(INITIAL_STATE, data);
  Object.entries(DEFAULT_READOUTS).forEach(([name, text]) => assert.equal(v[{ 'useful-hours': 'usefulHours', 'gpu-hours': 'gpuHours', 'run-average-mfu': 'runAverageMfu' }[name] ?? name], text, name));
  assert.equal(v.lossSplit, LOSS_SPLIT);
  assert.equal(v.record, RECORD_LLAMA);
  assert.equal(v.invalid, '');
  assert.deepEqual(v.bar.map((p) => p.name), ['useful (MFU)', 'below peak', 'lost to failures']);
  assert.equal(v.bar[2].hatched, true);
  near(v.bar.reduce((s, p) => s + p.value, 0), llamaRun().gpuHours, 1e-12);
});

test('every toy readout equals the shared function\'s output for the same inputs (Review Focus 4)', () => {
  const plan = planFor(INITIAL_STATE, data);
  const reference = runPlan({ params: 405e9, tokens: 15.6e12, gpus: 16384, peakTflops: 989, mfu: 0.4, perGpuMtbfHours: 54 * 24 / 419 * 16384, saveH: 30 / 3600, restartH: 3 / 60, dollarsPerGpuHour: 2 });
  assert.deepEqual({ ...plan, peakTflops: undefined }, { ...reference, peakTflops: undefined });
  const v = toyView(INITIAL_STATE, data);
  assert.equal(v.gpuHours, gpuHours(reference.gpuHours));
  assert.equal(v.days, formatDuration(reference.days * 86400));
  assert.equal(v.mtbf, formatDuration(reference.mtbfH * 3600));
  assert.equal(v.cost, dollarsM(reference.cost));
  assert.equal(v.runAverageMfu, `${pct(mfuFrom({ flops: reference.flops, gpuHours: reference.gpuHours, peakTflops: 989 }))} of the H100 BF16 peak`);
  assert.equal(INITIAL_STATE.mfu, 0.4);
});

test('toy view: presets, record lines and the stand-in peak note', () => {
  const ds = toyView(deepseekState, data);
  assert.deepEqual([ds.gpuHours, ds.days, ds.usefulHours, ds.loss], ['2.664M', '54.2 days', '2.589M', '2.8%']);
  assert.equal(ds.record, RECORD_DEEPSEEK);
  assert.equal(ds.presetNote, DEEPSEEK_NOTE);
  assert.equal(toyView({ ...deepseekState, mfu: 0.36 }, data).record, 'Reported pre-training: 2.664M GPU-hours. This run: 2.638M (−1.0%).'.replace('2.638M', toyView({ ...deepseekState, mfu: 0.36 }, data).gpuHours));
  assert.equal(toyView({ ...INITIAL_STATE, mfu: 0.38 }, data).record, RECORD_LLAMA_MFU_38);
  assert.equal(toyView({ ...INITIAL_STATE, gpus: 32768 }, data).record, NO_RECORD, 'a hypothetical run is not compared with a record');
  assert.equal(toyView(INITIAL_STATE, data).presetNote, '');
  assert.equal(toyView({ ...INITIAL_STATE, chip: 'b200-bf16' }, data).runAverageMfu.endsWith('of the B200 BF16 peak'), true, 'the percentage names the peak it divides by');
});

test('toy view: try-this figures are the storyboard\'s', () => {
  assert.deepEqual(tryThis(data).map((t) => t.prompt), TRY_THIS);
  tryThis(data).forEach((t) => assert.doesNotMatch(t.prompt, /[{}—]|NaN|undefined/));
  const at = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);
  assert.deepEqual([at({ gpus: 32768 }).gpuHours, at({ gpus: 100000 }).days, at({ gpus: 100000 }).loss], ['30.81M', '15.4 days', '28.0%']);
  assert.deepEqual([at({ auto: false, intervalMin: 5 }).loss, at({ auto: false, intervalMin: 60 }).loss, at({ saveS: 15 }).interval, at({ saveS: 15 }).loss], ['13.0%', '18.6%', '9.63 min', '6.8%']);
  assert.equal(at({ saveS: 15, gpus: 100000 }).loss, '22.7%');
  assert.equal(at({ auto: false, intervalMin: 5 }).intervalSub, 'chosen');
});

test('toy view: outside the loss formula\'s range the readouts are replaced by the stated line (README lesson 22)', () => {
  const v = toyView({ ...INITIAL_STATE, gpus: 200000, restartMin: 30 }, data);
  assert.equal(v.invalid, INVALID);
  assert.deepEqual([v.gpuHours, v.days, v.cost, v.runAverageMfu, v.bar], ['—', '—', '—', '—', null]);
  assert.equal(toyView({ ...INITIAL_STATE, gpus: 100000 }, data).invalid, '', 'the 100,000-GPU case stays inside');
  const never = toyView({ ...INITIAL_STATE, auto: false, intervalMin: 1, saveS: 600 }, data);
  assert.equal(never.invalid, NEVER_LINE);
  assert.equal(INVALID_LINE, INVALID);
  assert.equal(recordLine({ ...INITIAL_STATE, gpus: 200000 }, { valid: false }), NO_RECORD);
});

test('the toy rejects what it cannot print', () => {
  assert.throws(() => peakFor('h100-fp4', data), RangeError);
  assert.throws(() => peakFor('h100-bf16', { hardware: { entries: [] } }), RangeError);
  assert.throws(() => planFor({ ...INITIAL_STATE, chip: 'nope' }, data), RangeError);
  assert.deepEqual(barParts(llamaRun(), 989).map((p) => p.hue), [1, 2, 3]);
});

test('the formatters', () => {
  assert.equal(sci(3.2856e24), '3.29 × 10²⁴');
  assert.throws(() => sci(0), RangeError);
  assert.deepEqual([int(16384), int(-5.4), tflops(342.6), hoursText(3.093), minutesText(13.62), dollarsM(58.47e6), signedPct(-0.052), signedPct(0.0123)], ['16,384', '−5', '343', '3.09 h', '13.6 min', '$58.5M', '−5.2%', '+1.2%']);
  assert.equal(pctOf(40, 100, 1), '40.0%');
  assert.equal(FORMATS.fp8_e4m3.layout, '1/4/3', 'frame 6 draws E4M3 from the formats table');
  assert.equal(FORMATS.bf16.layout, '1/8/7');
});

test('nothing here mutates its inputs', () => {
  const state = Object.freeze({ ...INITIAL_STATE, mfu: 0.38 });
  const before = JSON.stringify([PRESETS, CHIPS, N.LLAMA, N.STOPS]);
  toyView(state, data);
  tryThis(data);
  lessonFor(data);
  assert.equal(JSON.stringify([PRESETS, CHIPS, N.LLAMA, N.STOPS]), before);
});
