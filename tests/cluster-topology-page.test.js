import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, lookupFact } from '../shared/claims.js';
import { tpCommRatio, ppCommRatio, dpCommRatio, epCommRatio, epMinLinkGBps } from '../math/topology.js';
import { ringAllReduceBytes } from '../math/parallel.js';
import { formatBytes } from '../math/core.js';
import { LESSON, lessonFor } from '../training/concepts/cluster-topology/content.js';
import { checkWork, ratioFor, tensorLayerWork, pct1, int, fitsInside, DEGREE_STOPS, TOKEN_STOPS } from '../training/concepts/cluster-topology/format.js';
import { toyView, tryThis, whereOptions, effectiveState, lanesFor, snapDegree, systemFromData, INITIAL_STATE, NO_FIT_NOTE, BASIS_LINE } from '../training/concepts/cluster-topology/toy-view.js';
import { belowFor, factRows } from '../training/concepts/cluster-topology/facts.js';
import * as S from '../training/concepts/cluster-topology/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS, BASIS_LINE as EXPECTED_BASIS } from './cluster-topology-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const hw = (id, key) => lookupFact(data.hardware, id, key).value;
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const view = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('11 facts rows, every placeholder resolves, none prints a stale dash once filled', () => {
  assert.equal(LESSON.facts.rows.length, 11);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  factRows(data).forEach((row, i) => assert.doesNotMatch(row.claim, /—/, `row ${i + 1}`));
  assert.match(factRows(data)[0].claim, /900, 1,800 and 3,000–3,600 GB\/s|900, 1,800 and 3,600 GB\/s|both-directions totals/);
  assert.match(factRows(null)[6].claim, /\(—\)/, 'without data a ratio prints a dash, never a stale number');
});
test('hook, intuition and the notes under the stage fill from data with nothing missing', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, ...CAPTIONS.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  [...LESSON.intuition, ...CAPTIONS.flatMap((_, i) => belowFor(null, i))].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(lesson.intuition[0], /50 GB\/s each way, nine times less than NVLink's 450/);
  assert.match(lesson.intuition[1], /costs 28% of the compute time over NVLink/);
  assert.match(lesson.intuition[1], /about 1\.5% of the compute even over the network/);
  assert.match(lesson.animation.belowFor(0)[0], /NVIDIA quotes 900 GB\/s per GPU for H100 NVLink, counting both directions/);
  assert.match(lesson.animation.belowFor(3)[0], /stand-in \(262,144\); the ratio rises to 79\.2% at 16,384 tokens/);
  assert.match(lesson.animation.belowFor(5)[0], /\(5\.0% against 1\.5%\)/);
  assert.match(lesson.animation.belowFor(6)[0], /3,072 GPUs.*24,576 GPUs/);
  assert.match(lesson.animation.belowFor(9)[0], /72 ÷ 8 = 9 servers/);
  assert.match(lesson.animation.belowFor(9)[0], /needs 407 GB\/s.*45\.2%.*406\.9%.*75\.4%/);
  assert.equal(lesson.animation.belowFor(42).length, 0);
});
test('no printed year the data does not confirm (README X-1)', () => {
  const years = [...lessonFor(data).facts.rows.map((r) => fillClaim(r.claim, data).text ?? r.claim)].join(' ').match(/\((?:20\d\d)[^)]*\)/g) ?? [];
  years.forEach((y) => assert.match(y, /^\((2024 paper|2024|2025|2026)/));
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('cluster-topology')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(graph.concepts.find((c) => c.slug === 'cluster-topology').prereqs, ['parallelism']);
});

test('"Check my work" for the default state is the storyboard text', () => {
  assert.equal(checkWork(INITIAL_STATE, S.SYSTEMS.h100), CHECK_WORK);
  assert.equal(view({}).check, CHECK_WORK);
});
test('"Check my work" fills the same template for every cut and system', () => {
  const pipeline = checkWork({ ...INITIAL_STATE, cut: 'pipeline', degree: 16, where: 'network' }, S.SYSTEMS.h100).split('\n');
  assert.match(pipeline[0], /^bytes per boundary, one micro-batch, full step = 2 × 2,048 × 12,288 × 2 B = 100\.66 MB$/);
  assert.match(pipeline[2], /= 1\.5%$/);
  const dp = checkWork({ ...INITIAL_STATE, cut: 'data', degree: 64, where: 'network' }, S.SYSTEMS.h100).split('\n');
  assert.match(dp[0], /= 3\.9375 B$/);
  assert.match(dp[2], /= 5\.0%$/);
  const ep = checkWork({ ...INITIAL_STATE, system: 'gb200', cut: 'expert', where: 'network' }, S.SYSTEMS.gb200).split('\n');
  assert.equal(ep[0], 'compute ÷ link = 2,500 TFLOPS ÷ 100 GB/s = 25,000.0 FLOPs per byte');
  assert.match(ep[1], /= 406\.9%$/);
  const gb = checkWork({ ...INITIAL_STATE, system: 'gb200' }, S.SYSTEMS.gb200);
  assert.match(gb, /÷ 900 GB\/s\) ÷ \(2,783\.1 GFLOP ÷ 2,500 TFLOPS\) = 35\.2%$/);
});

test('the toy opens on the storyboard default', () => {
  const v = view({});
  assert.deepEqual([v.ratioText, v.bytes, v.hide, v.linkNeeded, v.lanes.note, v.lanes.cap], ['27.8%', '352 MB', 'no: on the critical path', null, 'both lanes in the same time units', null]);
  assert.equal(v.systemLine, 'H100 HGX: NVLink 450 GB/s each way, network 50 GB/s each way, 989 TFLOPS dense BF16 per GPU.');
  assert.equal(EXPECTED_BASIS, BASIS_LINE.replace('([[scale-reliability]])', '(scale-reliability)'));
});
test('the try-this list is the storyboard\'s, every number computed', () => {
  assert.deepEqual(tryThis(data).map(({ text, insight }) => `${text} → Insight: ${insight}`), TRY_THIS);
});
test('every readout equals the math/topology.js function for the same inputs (Review Focus 4)', () => {
  const sys = systemFromData('h100', data);
  const input = { hidden: 12288, peakTflops: 989, linkGBps: 450 };
  near(ratioFor({ ...INITIAL_STATE }, sys), tpCommRatio({ tp: 8, ...input }), 1e-12);
  near(ratioFor({ cut: 'pipeline', degree: 16, where: 'network' }, sys), ppCommRatio({ pp: 16, layers: 96, ...input, linkGBps: 50 }), 1e-12);
  near(ratioFor({ cut: 'data', degree: 64, where: 'network', tokens: 262144 }, sys), dpCommRatio({ dp: 64, tokensPerReplica: 262144, peakTflops: 989, linkGBps: 50 }), 1e-12);
  near(ratioFor({ cut: 'expert', degree: 8, where: 'inside' }, sys), epCommRatio({ peakTflops: 989, linkGBps: 450 }), 1e-12);
  assert.equal(view({ cut: 'expert' }).linkNeeded, `${int(epMinLinkGBps({ peakTflops: 989 }))} GB/s`);
  const { bytes } = tensorLayerWork(8);
  assert.equal(bytes, 4 * ringAllReduceBytes(2048 * 12288 * 2, 8));
  assert.equal(view({}).bytes, formatBytes(bytes));
  assert.equal(pct1(1.0), '100.0%');
});
test('every hardware readout reads the key the Conventions name (Review Focus 5)', () => {
  const h = systemFromData('h100', data);
  assert.deepEqual([h.peakTflops, h.nvlinkGBps, h.networkGBps], [hw('h100', 'bf16_dense_tflops'), hw('h100', 'nvlink_gb_s_each_way'), hw('network-400g', 'gb_s_each_way')]);
  const g = systemFromData('gb200', data);
  assert.deepEqual([g.peakTflops, g.nvlinkGBps, g.networkGBps], [hw('gb200-nvl72', 'bf16_dense_tflops'), hw('gb200-nvl72', 'nvlink_gb_s_each_way'), hw('network-800g', 'gb_s_each_way')]);
  assert.match(view({ system: 'gb200' }).systemLine, /each way/);
});
test('page constants restate the data they copy (P3-R13)', () => {
  Object.values(S.SYSTEMS).forEach((s) => {
    const live = systemFromData(s.id, data);
    assert.deepEqual([live.peakTflops, live.nvlinkGBps, live.networkGBps], [s.peakTflops, s.nvlinkGBps, s.networkGBps], s.id);
  });
  const meta = (k) => hw('meta-llama3-cluster', k);
  assert.deepEqual([meta('gpus_per_rack'), meta('racks_per_pod'), meta('gpus_per_pod'), meta('pods'), meta('oversubscription'), meta('per_gpu_gbps')],
    [S.META.gpusPerRack, S.META.racksPerPod, S.META.gpusPerPod, S.META.pods, S.META.oversubscription, S.META.perGpuGbps]);
  assert.equal(S.META.gpusPerRack * S.META.racksPerPod, S.META.gpusPerPod);
  const model = (id, k) => lookupFact(data.models, id, k).value;
  assert.deepEqual([model('deepseek-v3', 'nvlink_effective_gb_s'), model('deepseek-v3', 'ib_gb_s'), model('deepseek-v3', 'max_nodes_per_token')], [S.DEEPSEEK_V3.nvlinkGBps, S.DEEPSEEK_V3.ibGBps, S.DEEPSEEK_V3.maxNodesPerToken]);
  assert.equal(hw('gb200-nvl72', 'rack_nvlink_tbps'), S.NVL72.rackNvlinkTbps);
  assert.match(hw('gb200-nvl72', 'scale_up_domain'), new RegExp(`${S.NVL72.gpus} GPUs \\+ ${S.NVL72.graceCpus} Grace CPUs`));
  assert.equal(S.SYSTEMS.gb200.domain / S.SYSTEMS.h100.domain, 9);
  assert.deepEqual(model('llama-3.1-405b', 'parallelism_order').toString().toLowerCase().includes('tp'), true);
  assert.equal(S.LLAMA3_LAYOUT.tp * S.LLAMA3_LAYOUT.pp * S.LLAMA3_LAYOUT.dp, 8192);
  assert.equal(S.CAP_UNITS, 520 / 1.2);
});
test('storyboard §6 worked values for every try-this state', () => {
  const r = (p) => view(p).ratioText;
  assert.deepEqual([r({ where: 'network' }), r({ degree: 16, where: 'network' }), r({ system: 'gb200' }), r({ system: 'gb200', degree: 16 })], ['250.4%', '536.6%', '35.2%', '75.4%']);
  assert.deepEqual([r({ cut: 'data', degree: 64, where: 'network' }), r({ cut: 'data', degree: 64, where: 'network', tokens: 16384 }), r({ cut: 'pipeline', degree: 16, where: 'network' })], ['5.0%', '79.2%', '1.5%']);
  assert.deepEqual([r({ cut: 'expert' }), r({ cut: 'expert', where: 'network' }), r({ cut: 'expert', system: 'gb200' }), r({ cut: 'expert', system: 'gb200', where: 'network' })], ['35.8%', '321.9%', '45.2%', '406.9%']);
  assert.equal(view({ cut: 'expert', system: 'gb200' }).linkNeeded, '407 GB/s');
  assert.equal(r({ degree: 64, where: 'network' }), '2,253.6%');
});
test('"inside" is disabled past the NVLink domain, with the visible note (P3-R12); expert has no degree to exceed', () => {
  const h = S.SYSTEMS.h100;
  assert.deepEqual(whereOptions(INITIAL_STATE, h).map((o) => o.disabled ?? false), [false, false]);
  const over = whereOptions({ ...INITIAL_STATE, degree: 16 }, h);
  assert.deepEqual([over[0].disabled, over[0].note, over[1].disabled], [true, NO_FIT_NOTE, undefined]);
  assert.equal(whereOptions({ ...INITIAL_STATE, degree: 16 }, S.SYSTEMS.gb200)[0].disabled, undefined);
  assert.equal(whereOptions({ ...INITIAL_STATE, degree: 64 }, S.SYSTEMS.gb200)[0].disabled, undefined);
  assert.equal(whereOptions({ ...INITIAL_STATE, cut: 'expert', degree: 64 }, h)[0].disabled, undefined);
  assert.equal(effectiveState({ ...INITIAL_STATE, degree: 16 }, h).where, 'network');
  assert.equal(fitsInside(h, 'tensor', 8), true);
  assert.equal(fitsInside(h, 'tensor', 16), false);
});
test('lanes: compute is 100 units; a comm lane past 4.33× compute is cut with its rounded percentage', () => {
  assert.deepEqual(lanesFor(0.2782).cap, null);
  const capped = lanesFor(22.536);
  assert.deepEqual([capped.compute, capped.cap.at, capped.cap.label], [100, S.CAP_UNITS, 'continues: 2,254%']);
  assert.equal(capped.note, 'both lanes in the same time units; comm lane cut at 4.33× compute');
  assert.equal(lanesFor(5.366).cap.label, 'continues: 537%');
});
test('degree stops: snapping keeps a stop and moves an off-range degree to the nearest stop', () => {
  assert.equal(snapDegree('tensor', 8), 8);
  assert.equal(snapDegree('pipeline', 64), 32);
  assert.equal(snapDegree('expert', 8), 8);
  assert.deepEqual(DEGREE_STOPS.tensor, [2, 4, 8, 16, 32, 64]);
  assert.equal(TOKEN_STOPS[6], 262144);
  assert.equal(TOKEN_STOPS[2], 16384);
});
test('inputs are never mutated', () => {
  const before = JSON.stringify(INITIAL_STATE);
  toyView(INITIAL_STATE, data);
  checkWork(INITIAL_STATE, S.SYSTEMS.h100);
  assert.equal(JSON.stringify(INITIAL_STATE), before);
  assert.ok(Object.isFrozen(S.SYSTEMS) && Object.isFrozen(S.GPT3));
});
test('the page prints ratios through formatRatio and no ratio below 1× (README lesson 35)', () => {
  const all = [lessonFor(data).intuition.join(' '), ...CAPTIONS.flatMap((_, i) => belowFor(data, i)), ...factRows(data).map((r) => r.claim)].join(' ');
  assert.doesNotMatch(all, /\b0\.\d+×/);
  assert.match(factRows(data)[6].claim, /\(3\.2×\)/);
});
