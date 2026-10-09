import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { formatBytes, formatCount, formatDuration, formatInt, formatRatio } from '../math/core.js';
import { kvCacheBytes, kvBytesPerTokenMla, sharePct } from '../math/memory.js';
import { matmulCost, arithmeticIntensity, ridgePoint, tokensToComputeBound, bytesPerElement } from '../math/roofline.js';
import { RUNNING_EXAMPLE, TOY_REQUESTS, stepTime, kvTransferTime, tokensPerExpert, freeHbmPerGpu, hbmFor } from '../math/serving.js';
import { LESSON, lessonFor } from '../serving/concepts/disaggregation/content.js';
import { CAPTIONS as PAGE_CAPTIONS } from '../serving/concepts/disaggregation/captions.js';
import { INITIAL_STATE, PROMPT_STOPS, EP_STOPS, USER_PRESETS, setupFrom, shipAnalysis, expertAnalysis, expertVerdict, checkWork, prefillCrossover, kvBytesPerToken, pct1 } from '../serving/concepts/disaggregation/format.js';
import { toyView, TOY_NOTE, NOT_FIT, SHORT_NOTE } from '../serving/concepts/disaggregation/toy-view.js';
import { tryThis } from '../serving/concepts/disaggregation/try-this.js';
import { BELOW, factRows, nvlinkVsNetwork } from '../serving/concepts/disaggregation/facts.js';
import { branchSchedule, laneOf, colocatedMetrics, separatedTimes, prefillEndMs, decodeSteps, BRANCH_REQUESTS } from '../serving/concepts/disaggregation/colocated.js';
import * as N from '../serving/concepts/disaggregation/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './disaggregation-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const fact = (set, id, key) => data[set].entries.find((e) => e.id === id).facts[key];
const setup = setupFrom(data);
const view = (state) => toyView({ ...INITIAL_STATE, ...state }, data);
const titled = (text) => text.replace(/\[\[([a-z0-9-]+)\]\]/g, (_, slug) => graph.concepts.find((c) => c.slug === slug).title);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order; the page copy equals the expected copy', () => {
  assert.deepEqual(PAGE_CAPTIONS, CAPTIONS);
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.equal(LESSON.animation.steps.length, 10);
});

test('11 facts rows; every placeholder resolves; the filled lesson has no dash; years only from a confirmed date (X-1)', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 11);
  lesson.facts.rows.forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.doesNotMatch(fillText(row.claim, data), /—/, `row ${i + 1}`);
  });
  const texts = [lesson.hook, ...lesson.intuition, ...lesson.takeaways, lesson.facts.framing, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  const rows = lesson.facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[1], /^DistServe \(2024\) framed serving as goodput: 7\.4× more requests or 12\.6× tighter latency targets \(the paper's abstract\)\. Splitwise \(2023\): 1\.4× throughput at 20% lower cost, or 2\.35× at the same cost/);
  assert.match(rows[3], /^vLLM on GB200 \(DeepSeek-R1, NVFP4, 2026-02-03\): the best layout was 4 prefill instances of 2 GPUs feeding one 8-GPU decode instance; 26\.2K prompt tokens per second per prefill GPU and 10\.1K output tokens per second per decode GPU/);
  assert.match(rows[5], /prefill on EP32 over 4 nodes \(9 routed \+ 1 shared expert per GPU\); decode on EP144 over 18 nodes \(2 routed \+ 1 shared per GPU\), with 32 redundant routed experts; 8 of 256 routed experts active per token\.$/);
  assert.match(rows[8], /^DeepSeek-V4-Pro has 384 routed experts, 6 active per token, and ships as about 865 GB\.$/);
  assert.match(rows[9], /^GB300 NVL72: 72 GPUs in one NVLink domain, about 130 TB\/s of aggregate NVLink .*2\.83× per-GPU throughput over GB200 at 27 tokens per second per user to its 288 GB of memory \(nominal\)/);
  assert.match(rows[10], /^Link speeds per GPU, each way: NVLink5 900 GB\/s \(NVIDIA quotes 1\.8 TB\/s counting both directions\); network ports 400 Gb\/s = 50 GB\/s and 800 Gb\/s = 100 GB\/s, so NVLink is 9× the 800 Gb\/s port on Blackwell\.$/);
  assert.equal(nvlinkVsNetwork(data), formatRatio(900 / 100));
  assert.equal(nvlinkVsNetwork(null), '—', 'without data a derived number prints a dash, never a stale one');
  for (const id of ['distserve', 'splitwise']) assert.equal(fact('serving', id, 'release_date').confidence, 'confirmed', `${id} prints a year only because its date is confirmed`);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('disaggregation')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('lesson titles, never slugs, in text (README lesson 33)', () => {
  const lesson = lessonFor(data);
  const prose = [lesson.hook, ...lesson.intuition, ...lesson.takeaways, ...BELOW.flat()].map((t) => fillText(t, data));
  prose.forEach((t) => assert.doesNotMatch(titled(t), /\b(paged-attention|prefill-decode|cluster-topology|serving-calculator|kv-compression)\b/));
});

test('the stage constants equal data/*.json (P3-R13 pattern)', () => {
  assert.equal(N.V4_PRO.expertsTotal, fact('models', 'deepseek-v4-pro', 'experts_total').value);
  assert.equal(N.V4_PRO.expertsActive, fact('models', 'deepseek-v4-pro', 'experts_active').value);
  assert.equal(N.V4_PRO.checkpointBytes, fact('models', 'deepseek-v4-pro', 'checkpoint_gb').value * 1e9);
  assert.equal(N.V4_PRO.dModel, fact('models', 'deepseek-v4-pro', 'd_model').value);
  assert.equal(N.V4_PRO.expertHidden, fact('models', 'deepseek-v4-pro', 'expert_hidden').value);
  assert.equal(N.GB300.hbmBytes, hbmFor(data.hardware.entries.find((e) => e.id === 'gb300-nvl72')).bytes);
  assert.equal(N.GB300.peakTflops, fact('hardware', 'gb300-nvl72', 'nvfp4_dense_tflops').value);
  assert.equal(N.GB300.bandwidthTBps, fact('hardware', 'gb300-nvl72', 'hbm_tbps').value);
  assert.equal(N.GB300.rackNvlinkTbps, fact('hardware', 'gb300-nvl72', 'rack_nvlink_tbps').value);
  assert.equal(N.EXPERT_COUNT.total, N.V4_PRO.expertsTotal);
  assert.equal(N.EXPERT_COUNT.rows * N.EXPERT_COUNT.columns, N.EXPERT_COUNT.total);
  assert.deepEqual(N.MLA, { layers: fact('models', 'deepseek-v3', 'layers').value, dLatent: fact('models', 'deepseek-v3', 'mla_kv_rank').value, dRope: fact('models', 'deepseek-v3', 'mla_rope_dim').value });
  assert.equal(kvBytesPerTokenMla({ ...N.MLA, bytesPerElem: 2 }), fact('models', 'deepseek-v3', 'kv_bytes_per_token').value, 'MLA KV: 70,272 B/token');
  assert.equal(RUNNING_EXAMPLE.kvBytesPerToken, fact('models', 'llama-3.1-70b', 'kv_bytes_per_token').value);
  const [nvlink, n800, n400] = N.LINK_IDS.map((id) => N.STAGE_LINKS[id]);
  assert.deepEqual([nvlink.gbPerS, n800.gbPerS, n400.gbPerS], [fact('hardware', 'b200', 'nvlink_gb_s_each_way').value, fact('hardware', 'network-800g', 'gb_s_each_way').value, fact('hardware', 'network-400g', 'gb_s_each_way').value]);
  const g = (key) => fact('serving', 'vllm-gb200-dsr1', key).value;
  assert.deepEqual(N.VLLM_GB200, { prefillGroups: g('prefill_instances'), prefillGpusEach: g('prefill_gpus_each'), decodeGpus: g('decode_gpus'), prefillTokSGpu: g('prefill_tok_s_gpu'), decodeTokSGpu: g('decode_tok_s_gpu') });
  const d = (key) => fact('serving', 'deepseek-v3-production', key).value;
  assert.equal(N.DEEPSEEK.prefillEp, d('prefill_ep'));
  assert.equal(N.DEEPSEEK.decodeEp, d('decode_ep'));
  assert.equal(N.DEEPSEEK.redundant, d('redundant_experts'));
  assert.equal(N.DEEPSEEK.prefillRoutedPerGpu, d('prefill_routed_experts_per_gpu'));
  assert.equal(N.DEEPSEEK.prefillSharedPerGpu, d('prefill_shared_experts_per_gpu'));
  assert.equal(N.DEEPSEEK.routedExperts, fact('models', 'deepseek-v3', 'experts_total').value);
  assert.equal(N.DEEPSEEK.activePerToken, fact('models', 'deepseek-v3', 'experts_active').value);
  assert.equal((N.DEEPSEEK.routedExperts + N.DEEPSEEK.redundant) / N.DEEPSEEK.decodeEp, 2, '288 ÷ 144 = 2 routed experts per decode GPU');
  assert.equal(`${N.DEEPSEEK.decodeEp / N.DEEPSEEK.gpusPerNode} nodes`, '18 nodes');
  assert.equal(N.DEEPSEEK.prefillEp / N.DEEPSEEK.gpusPerNode, 4);
  assert.equal(N.GB300_GAIN, fact('serving', 'inferencex-v4-pro-gb300', 'per_gpu_gain_vs_gb200').value);
  assert.equal(N.PAPERS.distserve.year, fact('serving', 'distserve', 'release_date').value.slice(0, 4));
  assert.equal(N.PAPERS.splitwise.year, fact('serving', 'splitwise', 'release_date').value.slice(0, 4));
  assert.equal(N.PAPERS.distserve.goodputGain, fact('serving', 'distserve', 'goodput_gain').value);
  assert.equal(N.PAPERS.distserve.sloGain, fact('serving', 'distserve', 'slo_gain').value);
  assert.equal(N.PAPERS.splitwise.throughputGain, fact('serving', 'splitwise', 'throughput_gain').value);
  assert.equal(N.PAPERS.splitwise.costCutPct, fact('serving', 'splitwise', 'cost_cut_pct').value);
});

test('the toy reads link speeds each way, V4-Pro and the GB300 from data (setupFrom), with the HBM basis', () => {
  assert.deepEqual(setup.links.map((l) => [l.id, l.label, l.gbPerS, l.bytesPerSecond]), [
    ['nvlink', 'NVLink5 900 GB/s', 900, 900e9], ['net800', 'network 800 Gb/s (100 GB/s)', 100, 100e9], ['net400', 'network 400 Gb/s (50 GB/s)', 50, 50e9]]);
  assert.deepEqual(setup.v4, N.V4_PRO);
  assert.deepEqual({ ...setup.gb300 }, { hbmBytes: 288e9, hbmBasis: 'nominal', peakTflops: 15000, bandwidthTBps: 8 });
  assert.throws(() => setupFrom({ hardware: { entries: [] }, models: { entries: [] } }), RangeError);
  assert.throws(() => shipAnalysis({ ...INITIAL_STATE, link: 'wifi' }, setup), RangeError);
  assert.throws(() => shipAnalysis({ ...INITIAL_STATE, kv: 'fp4' }, setup), RangeError);
});

test('frames 1–3: batching\'s branch timed from scheduleTokens and stepTime (storyboard §5 frame 1, §2)', () => {
  const none = branchSchedule(Infinity);
  const chunked = branchSchedule(N.CHUNK_BUDGET);
  const long = none.find((s) => s.prefill.some((p) => p.id === 'D'));
  assert.equal(formatDuration(long.ms / 1000), '290 ms', 'no chunks: one 290 ms step');
  assert.equal(long.step, 3, 'D\'s prompt arrives in step 3 (after B has finished)');
  const slices = chunked.filter((s) => s.prefill.some((p) => p.id === 'D'));
  assert.equal(slices.length, 9, 'nine slices, the ninth is 6 tokens');
  assert.deepEqual(slices.map((s) => s.prefill[0].tokens), [510, 510, 511, 511, 512, 512, 512, 512, 6]);
  const full = slices.filter((s) => s.tokens === 512);
  assert.equal(full.length, 8, 'D\'s prompt takes eight steps of 36.2 ms');
  full.forEach((s) => assert.equal(formatDuration(s.ms / 1000), '36.2 ms'));
  assert.equal(chunked.filter((s) => s.step >= 3 && s.decode.includes('A') && s.tokens === 512).length, 2, 'A sits through two of them');
  assert.equal(chunked.filter((s) => s.step >= 3 && s.decode.includes('C') && s.tokens === 512).length, 4, 'C sits through four');
  assert.equal(formatDuration(none.find((s) => s.step === 1).ms / 1000), '14.6 ms', 'before the branch a step is 14.6 ms');
  const dNone = laneOf(none, 'D');
  assert.deepEqual(dNone.steps.map((s) => s.kind), ['queue', 'prefill', 'decode', 'decode', 'decode'], 'D waits for a seat, then prefills, then decodes 3 tokens');
  assert.equal(dNone.arrivesMs, none[1].fromMs, 'D arrives at step 1');
  assert.equal(prefillEndMs(none, 'D'), long.fromMs + long.ms);
  assert.equal(decodeSteps(none, 'A').length, TOY_REQUESTS.find((r) => r.id === 'A').output);
  assert.equal(decodeSteps(none, 'C').length, TOY_REQUESTS.find((r) => r.id === 'C').output);
  assert.deepEqual(BRANCH_REQUESTS.map((r) => r.prompt), [8, 5, 10, N.BRANCH_PROMPT]);
  assert.throws(() => { BRANCH_REQUESTS.push({}); }, TypeError);
  assert.equal(TOY_REQUESTS.find((r) => r.id === 'D').prompt, 6, 'the shared request list is not mutated');
});

test('frame 2: the goodput table (illustrative targets 400 ms and 15 ms) never lets one budget meet both for all three', () => {
  const rows = N.GOODPUT_BUDGETS.map((b) => colocatedMetrics(b));
  assert.deepEqual(rows.map((m) => m.meeting), [1, 1, 1, 2]);
  assert.ok(rows.every((m) => m.meeting < m.total), 'no setting meets both targets for every request');
  const at = (b) => Object.fromEntries(colocatedMetrics(b).rows.map((r) => [r.id, r]));
  assert.equal(formatDuration(at(Infinity).D.ttftMs / 1000), '319 ms');
  assert.equal(formatDuration(at(Infinity).A.tpotMs / 1000), '83.4 ms');
  assert.equal(formatDuration(at(512).C.tpotMs / 1000), '29 ms');
  assert.equal(formatDuration(at(128).D.ttftMs / 1000), '525 ms');
  assert.equal(at(128).A.okTpot && at(128).C.okTpot, true, 'a 128-token budget keeps A and C under 15 ms');
  assert.equal(at(128).D.okTtft, false, '… and pushes D\'s first token past 400 ms');
  assert.equal(N.TTFT_TARGET_MS, 400);
  assert.equal(N.TPOT_TARGET_MS, 15);
});

test('captions\' numbers equal the functions they repeat (README lesson 29)', () => {
  const sep = separatedTimes();
  assert.match(CAPTIONS[0], new RegExp(`stalled A and C for ${formatDuration(branchSchedule(Infinity).find((s) => s.step === 3).ms / 1000).replace(' ms', '')} ms`));
  assert.match(CAPTIONS[0], /stretched to 36 ms/);
  assert.equal(Math.round(branchSchedule(N.CHUNK_BUDGET).find((s) => s.step === 3).ms), 36);
  assert.match(CAPTIONS[2], new RegExp(`keep their ${formatDuration(sep.decodeMs / 1000)} steps`));
  assert.equal(formatDuration(sep.prefillMs / 1000), '290 ms');
  assert.equal(sep.stepsLong.toFixed(1), '19.9', '"D\'s bar is 19.9 of A\'s steps long"');
  const a = shipAnalysis({ ...INITIAL_STATE, link: 'net400' }, setup);
  const n = shipAnalysis({ ...INITIAL_STATE, link: 'nvlink' }, setup);
  assert.match(CAPTIONS[3], new RegExp(`${formatBytes(a.kvBytes)}, about ${formatDuration(a.transferS)} over a 400 Gb/s network port and ${formatDuration(n.transferS)} over NVLink`));
  assert.match(CAPTIONS[4], new RegExp(`four ${N.VLLM_GB200.prefillGpusEach}-GPU prefill groups feeding one ${N.VLLM_GB200.decodeGpus}-GPU decode group`));
  assert.equal(N.VLLM_GB200.prefillGroups * N.VLLM_GB200.prefillGpusEach + N.VLLM_GB200.decodeGpus, 16);
  assert.match(CAPTIONS[5], new RegExp(`With ${N.V4_PRO.expertsTotal} experts and ${N.V4_PRO.expertsActive} per token, ${N.EXPERT_COUNT && N.USERS_SHOWN} users`));
  assert.match(CAPTIONS[6], new RegExp(`over ${N.EP_SHOWN} GPUs.*gives each expert ${formatCount(expertAnalysis({ ep: N.EP_SHOWN, users: N.USERS_SHOWN }, setup).tokens)} tokens per step`));
  assert.match(CAPTIONS[7], new RegExp(`over ${N.DEEPSEEK.prefillEp} GPUs, decode over ${N.DEEPSEEK.decodeEp}`));
});

test('"Ship the KV" for the storyboard\'s worked examples (§6), each from its math/ function', () => {
  const row = (prompt, link, kv = 'bf16') => shipAnalysis({ ...INITIAL_STATE, prompt, link, kv }, setup);
  const printed = (a) => [formatDuration(a.transferS), formatDuration(a.prefillS), pct1(a.transferS, a.prefillS)];
  assert.deepEqual(printed(row(512, 'net400')), ['3.36 ms', '36.2 ms', '9.3%']);
  assert.deepEqual(printed(row(4096, 'net400')), ['26.8 ms', '290 ms', '9.3%']);
  assert.deepEqual(printed(row(131072, 'net400')), ['859 ms', '9.27 s', '9.3%']);
  assert.deepEqual(printed(row(128, 'net400')), ['839 µs', '15 ms', '5.6%']);
  assert.deepEqual(printed(row(4096, 'net800')), ['13.4 ms', '290 ms', '4.6%']);
  assert.deepEqual(printed(row(4096, 'nvlink')), ['1.49 ms', '290 ms', '0.5%']);
  assert.deepEqual(['net400', 'net800', 'nvlink'].map((l) => pct1(row(4096, l, 'fp8').transferS, row(4096, l, 'fp8').prefillS)), ['4.6%', '2.3%', '0.3%']);
  assert.equal(row(4096, 'net400', 'fp8').bytesPerToken, 163840);
  assert.equal(row(4096, 'net400').bytesPerToken, 327680);
  const mla = row(4096, 'net400').mla;
  assert.equal(mla.bytesPerToken, 70272);
  assert.deepEqual([formatBytes(mla.kvBytes), formatDuration(mla.transferS)], ['288 MB', '5.76 ms']);
  assert.equal(formatBytes(row(4096, 'net400').kvBytes), '1.34 GB');
  assert.equal(prefillCrossover(), 217, 'the crossover where prefill stops being one weight read');
  [100, 128, 200, 217].forEach((t) => assert.equal(row(t, 'net400').belowCrossover, t < 217));
  assert.equal(row(4096, 'net400').belowCrossover, false);
  const ratios = [512, 4096, 131072].map((p) => row(p, 'net400')).map((a) => sharePct(a.transferS, a.prefillS, { decimals: 3 }));
  assert.ok(Math.max(...ratios) - Math.min(...ratios) < 0.01, 'past the crossover the ratio is fixed by the model and the link');
});

test('"Feed the experts" for the storyboard\'s worked examples (§6)', () => {
  const at = (ep, users = 64) => expertAnalysis({ ...INITIAL_STATE, ep, users }, setup);
  assert.deepEqual(EP_STOPS.map((ep) => formatCount(at(ep).tokens)), ['1', '8', '16', '32', '72']);
  assert.deepEqual(EP_STOPS.map((ep) => formatBytes(at(ep).weightsBytes)), ['865 GB', '108 GB', '54.1 GB', '27 GB', '12 GB']);
  assert.deepEqual(EP_STOPS.map((ep) => at(ep).fits), [false, true, true, true, true]);
  assert.deepEqual(EP_STOPS.map((ep) => formatBytes(at(ep).freeBytes)).slice(1), ['180 GB', '234 GB', '261 GB', '276 GB']);
  assert.deepEqual(EP_STOPS.map((ep) => formatCount(at(ep).intensity)), ['3.55', '28.3', '56.5', '112', '248']);
  assert.equal(formatCount(at(72, 256).tokens), '288');
  assert.equal(formatCount(at(72, 256).intensity), '903');
  assert.equal(formatCount(at(16, 16).tokens), '4');
  assert.equal(at(16).ridge, 1875);
  assert.equal(formatInt(Math.ceil(at(16).tokensNeeded)), '699');
  assert.equal(tokensPerExpert({ usersPerGpu: 64, epSize: 1, expertsActive: 6, expertsTotal: 384 }), 1, 'EP 1: users × k ÷ E');
  assert.equal(at(1).freeBytes, freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: 1 }));
  assert.equal(expertVerdict(at(72, 256)), 'memory-bound', 'even 288 tokens per expert stay under 699');
  assert.equal(at(1, 16).tokens, 0.25, 'a fractional count prints through formatCount');
  assert.equal(formatCount(at(72).expertsPerGpu), '5.33', '384 ÷ 72 is not a whole number');
  const m = at(16);
  assert.equal(m.intensity, arithmeticIntensity(matmulCost({ m: 16, k: 7168, n: 3072, bytesPerElem: bytesPerElement('nvfp4') })));
  assert.equal(m.ridge, ridgePoint({ peakTflops: 15000, bandwidthTBps: 8 }));
  assert.equal(m.tokensNeeded, tokensToComputeBound({ peakTflops: 15000, bandwidthTBps: 8, bytesPerElem: 0.5625, k: 7168, n: 3072 }));
});

test('"Check my work" for the default state is the storyboard text, and fills the same template for every state', () => {
  assert.equal(checkWork(INITIAL_STATE, setup), CHECK_WORK);
  assert.equal(view({}).checkWork, CHECK_WORK);
  const l = checkWork({ prompt: 131072, link: 'nvlink', kv: 'fp8', ep: 72, users: 256 }, setup).split('\n');
  assert.equal(l[0], 'transfer = prompt × KV bytes ÷ link = 131,072 × 163,840 B ÷ 900 GB/s = 21.5 GB ÷ 900 GB/s = 23.9 ms');
  assert.equal(l[1], 'tokens per expert = users × EP × k ÷ E = 256 × 72 × 6 ÷ 384 = 288');
});

test('toy view: the default state prints each readout from its math/ function (Review Focus 1), HBM with its basis (Focus 3)', () => {
  const v = view({});
  const prefill = stepTime({ ...RUNNING_EXAMPLE, tokens: 4096, seqs: 0, context: 0 }).timeS;
  const transfer = kvTransferTime(4096, 327680, 50e9);
  assert.deepEqual(v.ship, {
    kvBytes: formatBytes(kvCacheBytes({ bytesPerToken: 327680, tokens: 4096 })), transfer: formatDuration(transfer), prefill: formatDuration(prefill),
    ratio: `${sharePct(transfer, prefill).toFixed(1)}%`, mlaKv: '288 MB', mlaTransfer: '5.76 ms', mlaBytesPerToken: '70,272', bytesPerToken: '327,680',
    linkLabel: 'network 400 Gb/s (50 GB/s)', shortNote: '', prompt: 4096,
  });
  assert.deepEqual([v.ship.kvBytes, v.ship.transfer, v.ship.prefill, v.ship.ratio], ['1.34 GB', '26.8 ms', '290 ms', '9.3%']);
  const { bar, ...rest } = v.experts;
  assert.deepEqual(rest, {
    expertsPerGpu: '24', weights: '54.1 GB', free: '234 GB', hbmNote: '288 GB HBM per GPU (nominal); weights 865 GB as shipped (reported)',
    tokens: '16', intensity: '56.5', ridge: '1,875', tokensNeeded: '699', verdict: 'memory-bound',
  });
  assert.deepEqual(bar.parts.map((p) => p.name), ['weights 54.1 GB', 'free 234 GB']);
  assert.match(rest.hbmNote, /nominal/);
  assert.equal(TOY_NOTE, 'Floors from bytes, bandwidth and FLOPs; real transfers add start-up latency, and real all-to-all adds communication time not modeled here.');
});

test('toy view: "does not fit" prints the exact text and no negative number; the short-prompt note shows below the crossover', () => {
  const v = view({ ep: 1 });
  assert.equal(v.experts.free, NOT_FIT);
  assert.equal(NOT_FIT, 'does not fit');
  assert.equal(v.experts.bar, null);
  assert.equal(v.experts.tokens, '1');
  assert.equal(v.experts.weights, '865 GB');
  assert.doesNotMatch(JSON.stringify({ ...v.experts, bar: null }), /−|-\d|NaN/);
  assert.equal(view({ prompt: 128 }).ship.shortNote, SHORT_NOTE(217));
  assert.equal(SHORT_NOTE(217), 'below 217 tokens prefill is one weight read, so the ratio is smaller here; real transfers add a fixed start-up cost the toy does not model, which is why short prompts gain least');
  assert.equal(view({ prompt: 512 }).ship.shortNote, '');
});

test('every stop and chip of every control renders without error and inputs are not mutated', () => {
  const frozen = Object.freeze({ ...INITIAL_STATE });
  PROMPT_STOPS.forEach((prompt) => ['nvlink', 'net800', 'net400'].forEach((link) => ['bf16', 'fp8'].forEach((kv) => assert.doesNotThrow(() => toyView({ ...frozen, prompt, link, kv }, data)))));
  EP_STOPS.forEach((ep) => USER_PRESETS.forEach((users) => assert.doesNotThrow(() => toyView({ ...frozen, ep, users }, data))));
  assert.equal(kvBytesPerToken('fp8'), RUNNING_EXAMPLE.kvBytesPerToken / 2);
  assert.deepEqual(toyView(frozen, data), toyView(frozen, data), 'the same state twice gives deepEqual output');
});

test('the try-this list is the storyboard\'s, with every number computed (§6)', () => {
  const printed = tryThis(setup).map(({ prompt, insight, rest }) => titled(`${prompt} → Insight: ${insight}${rest}`));
  assert.deepEqual(printed, TRY_THIS);
  tryThis(setup).forEach(({ prompt }) => assert.match(prompt, /[.:]$/, 'each prompt ends in a full stop or colon before " → "'));
});
