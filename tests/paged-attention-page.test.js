import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { formatBytes } from '../math/core.js';
import { kvCacheBytes, sharePct } from '../math/memory.js';
import { simulatePaged, kvBytesPerBlock, reserveMaxBytes } from '../math/paging.js';
import { TOY_REQUESTS, hbmFor } from '../math/serving.js';
import { LESSON, lessonFor } from '../serving/concepts/paged-attention/content.js';
import { BELOW, factRows, intuition, HOOK } from '../serving/concepts/paged-attention/facts.js';
import {
  INITIAL_STATE, shareText, shareCell, plural, exactBytes, bytesPair, statusText, stepText, meanWaste,
} from '../serving/concepts/paged-attention/format.js';
import { toyView, tryThis, scaleView, simulate } from '../serving/concepts/paged-attention/toy-view.js';
import * as N from '../serving/concepts/paged-attention/numbers.js';
import { slotsOf, mixSlots, poolGeometry } from '../serving/concepts/paged-attention/pool.js';
import { CAPTIONS, TRY_THIS } from './paged-attention-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const view = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('12 facts rows; every placeholder resolves; nothing is left unfilled', () => {
  assert.equal(LESSON.facts.rows.length, 12);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  factRows(data).forEach((row, i) => assert.doesNotMatch(fillClaim(row.claim, data).segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`));
});
test('Next lists exactly the lessons that take this one as a prereq', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('paged-attention')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('there is no "Check my work" box: this page is a simulation', () => {
  assert.doesNotMatch(JSON.stringify(lessonFor(data)), /Check my work/);
});

test('hook, intuition, takeaways and the note under frame 9 fill from data with nothing missing', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, ...lesson.takeaways, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  [HOOK, ...intuition(), ...BELOW.flat()].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.equal(lesson.hook, 'Why did early LLM servers run out of KV memory while 60–80% of it held nothing?');
  assert.match(lesson.intuition[1], /worst case \(52\.1%\) is lower than the measured 60–80%\. The vLLM team measured that only 20\.4–38\.2% of KV memory/);
  assert.match(lesson.intuition[1], /up to a point \(see \[\[batching\]\]\)/);
  assert.match(lesson.intuition[2], /\(16 in vLLM, 4 in this page's toy\)/);
  assert.match(lesson.intuition[3], /vLLM V1 preempts by recompute/);
  assert.match(lesson.animation.belowFor(8)[0], /caches and shares only full blocks, keyed by content/);
  [0, 1, 2].forEach((i) => assert.match(lesson.animation.belowFor(i)[0], /^Prefill ends with a request's first token; each tick after it is one decode step\.$/));
  assert.equal(lesson.animation.belowFor(3).length, 0);
});

test('facts rows print the storyboard\'s derived figures with their basis words', () => {
  const claims = factRows(data).map((r) => fillClaim(r.claim, data).segments.map((s) => s.text).join(''));
  assert.match(claims[0], /Only 20\.4–38\.2% of KV cache memory .* waste at 60–80%, and under 4% after paging\./);
  assert.match(claims[1], /by a factor of 2–4 over FasterTransformer/);
  assert.match(claims[2], /20–26% slower/);
  assert.match(claims[3], /default block size is 16 tokens/);
  assert.match(claims[5], /Llama-3\.1-70B 327,680 B \(≈ 328 kB\) per token \(GQA, 2 \(K and V\) × 80 layers × 8 KV heads × 128 × 2 B\): 5\.24 MB per 16-token block, 42\.9 GB reserved for one request at its 131,072-token context; DeepSeek-V3 70,272 B \(≈ 70\.3 kB\).*1\.12 MB per 16-token block, 9\.21 GB .*; GPT-3 4,718,592 B \(≈ 4\.72 MB\).*75\.5 MB per 16-token block, 9\.66 GB reserved for one request at its 2,048-token context\. The Llama figure is 53\.7% of an H100 \(80 GB nominal\)\./);
  assert.match(claims[6], /Kimi K3 has 24 gated-MLA layers among 69 KDA layers, Qwen3\.8 alternates layers in a 3 linear : 1 full pattern/);
  assert.match(claims[7], /4,989 KV tokens per request \(V3\/R1, Feb 2025\), a 16-token block wastes at most 15 slots, 0\.3%; the toy's 8\.0%–16\.7% comes from/);
  assert.match(claims[9], /^As of 2026-09-10, vLLM can spill blocks down the tiers HBM → host DRAM → storage/);
  assert.match(claims[10], /BF16's, vLLM, 2026-04-22\)/);
  assert.match(claims[11], /about 85% of the cap \(LMSYS, 2026-02-19\)/);
  assert.equal(2 * 80 * 8 * 128 * 2, 327_680, 'the printed Llama recipe multiplies to the per-token bytes it explains');
  assert.match(claims[8], /56\.3% of DeepSeek's input tokens/);
  assert.match(claims[10], /halves the bytes per block \(0\.5 of BF16's, vLLM, 2026-04-22\).*to 13%/);
  assert.match(claims[11], /cap of 40 requests per GPU on GB300 NVL72 \(288 GB per GPU, nominal\) against 24 on GB200 NVL72 \(186 GB per GPU, the rack total over 72, nominal\); LMSYS's practical target is 36 and 20/);
  assert.doesNotMatch(claims.join(' '), /\(20\d\d\)/);
});

test('the captions\' numbers equal the simulators and the data', () => {
  assert.match(CAPTIONS[2], /39\.6% empty/);
  assert.equal(shareText(N.contiguousAt(2).waste, N.POOL_SLOTS), '39.6%');
  assert.match(CAPTIONS[3], /blocks of 4 tokens \(vLLM uses 16\)/);
  assert.equal(data.serving.entries.find((e) => e.id === 'vllm').facts.default_block_size.value, 16);
  assert.match(CAPTIONS[4], /A's third block landed at 7/);
  assert.deepEqual(N.pagedAt(1).live[0].table, [0, 1, 7]);
  assert.deepEqual(N.addressOf([0, 1, 7], 9, 4), { block: 7, slot: 0 });
  assert.deepEqual(N.addressOf([4, 5, 6], 11, 4), { block: 6, slot: 2 });
  assert.deepEqual(N.addressOf(N.pagedAt(1).live[1].table, 6, 4), { block: 3, slot: 1 });
  assert.equal(N.finishStep('before', 'D'), 6);
  assert.equal(N.finishStep('after', 'D'), 4);
  assert.deepEqual(N.TOY_REQUESTS, TOY_REQUESTS);
});

test('every stage figure is the simulator\'s output for the same inputs (frames 2 to 7)', () => {
  assert.deepEqual([N.contiguousAt(0).useful, N.contiguousAt(0).waste, N.contiguousAt(0).free], [23, 25, 0]);
  assert.equal(shareText(25, 48), '52.1%');
  assert.deepEqual([shareText(23, 48), shareText(5, 48), shareText(20, 48), shareText(8, 48), shareText(4, 48)], ['47.9%', '10.4%', '41.7%', '16.7%', '8.3%']);
  const s3 = N.pagedAt(3);
  assert.deepEqual([s3.useful, s3.waste, s3.free, s3.blocksUsed], [32, 4, 12, 9]);
  assert.equal(JSON.stringify(simulatePaged({ requests: TOY_REQUESTS, poolSlots: 48, blockSize: 4, step: 3 })), JSON.stringify(s3));
});

test('forks (frames 8 and 9): step 1 shares blocks 8 and 9 with ref 2; step 2 copies 9 to 10', () => {
  const one = N.forkAt(1);
  assert.deepEqual([one.d1, one.d2, one.copied], [[8, 9], [8, 9], null]);
  assert.deepEqual(N.refCounts(one.entries, 12).slice(8, 11), [2, 2, 0]);
  assert.equal(N.blocksInUse(one.entries), 10);
  const two = N.forkAt(2);
  assert.deepEqual([two.d1, two.d2, two.copied], [[8, 10], [8, 9], { from: 9, to: 10 }]);
  assert.deepEqual(N.refCounts(two.entries, 12).slice(8, 11), [2, 1, 1]);
  assert.equal(N.blocksInUse(two.entries), 11);
  assert.throws(() => N.forkAt(3), RangeError);
});

test('pool drawing: slots from tables, progressive flips, and the geometry the marks use', () => {
  const geo = { blocks: 12, blockSize: 4 };
  const slots = slotsOf(N.entriesOfPaged(N.pagedAt(0)), geo);
  assert.equal(slots.length, 48);
  assert.deepEqual([slots[0], slots[11].state, slots[19].state, slots[47].state], [{ owner: 'A', state: 'filled' }, 'filled', 'filled', 'free']);
  const free = slotsOf([], geo);
  assert.deepEqual(mixSlots(free, slots, 0), free);
  assert.deepEqual(mixSlots(free, slots, 1), slots);
  assert.deepEqual(mixSlots(free, slots, 0.5), mixSlots(free, slots, 0.5));
  const g = poolGeometry({ blocks: 12, blockSize: 4, cell: 12, perRow: 6 });
  assert.deepEqual([g.blockW, g.blockH, g.width, g.height], [57, 26, 392, 82]);
  assert.deepEqual(g.blockAt(7), { x: 67, y: 56, w: 57, h: 26 });
  const shared = slotsOf([{ id: 'A', table: [0], tokens: 4 }, { id: 'B', table: [0], tokens: 4 }], geo);
  assert.equal(shared[0].owner, 'A');
});

test('toy at its default: step 3, block size 4', () => {
  const v = view({});
  assert.deepEqual([v.before.useful, v.before.wasted, v.before.free], [{ value: '62.5%', sub: '30 slots' }, { value: '37.5%', sub: '18 slots' }, { value: '0.0%', sub: '0 slots' }]);
  assert.deepEqual([v.after.useful.value, v.after.wasted.value, v.after.free.value, v.after.blocks], ['66.7%', '8.3%', '25.0%', '9 of 12 blocks']);
  assert.deepEqual([v.before.dStart, v.before.dFinish, v.after.dStart, v.after.dFinish, v.before.average, v.after.average], ['step 3', 'step 6', 'step 1', 'step 4', '34.2%', '8.0%']);
  assert.deepEqual(v.follow.rows.map((r) => r.physical), ['4', '5', '6', '2']);
  assert.deepEqual(v.saved, { value: '0', sub: '0 blocks' });
  assert.equal(v.scale, null);
  assert.deepEqual(v.requests.find((r) => r.id === 'B').after, { status: 'done', tokens: '—', blocks: '—', wasted: '—' });
  assert.deepEqual(v.requests.find((r) => r.id === 'D').before, { status: 'running', tokens: '6', wasted: '10' });
});

test('toy: every readout equals the simulator for the same inputs', () => {
  [2, 4, 8, 16].forEach((blockSize) => [0, 1, 2, 3, 4, 5, 6].forEach((step) => [false, true].forEach((sharedPrefix) => {
    const v = view({ blockSize, step, sharedPrefix });
    const sim = simulatePaged({ requests: TOY_REQUESTS, poolSlots: 48, blockSize, step, sharedPrefix: sharedPrefix ? 4 : 0 });
    assert.equal(v.after.wasted.value, shareText(sim.waste, 48));
    assert.equal(v.after.blocks, `${sim.blocksUsed} of ${sim.poolBlocks} blocks`);
    assert.equal(v.saved.value, String(sim.blocksSaved));
  })));
  assert.equal(view({ step: 0 }).before.wasted.value, '52.1%');
  assert.equal(view({ step: 2 }).requests[3].before.status, 'waiting');
  assert.equal(view({ step: 0 }).requests[3].after.status, 'not arrived');
  assert.equal(view({ step: 1 }).after.wasted.value, '16.7%');
});

test('toy: following a request that holds no blocks says so', () => {
  assert.equal(view({ step: 0, follow: 'D' }).follow.empty, 'D holds no blocks at step 0 (not arrived).');
  assert.equal(view({ step: 6, follow: 'A' }).follow.empty, 'A holds no blocks at step 6 (done).');
  assert.equal(view({ step: 6, follow: 'C' }).follow.empty, null);
});

test('toy: the scale-up readout for each model, with the H100\'s basis word', () => {
  const llama = view({ model: 'llama' }).scale;
  assert.deepEqual([llama.perToken, llama.block.value, llama.reserve.value, llama.share], [{ value: '327,680 B', sub: '≈ 328 kB' }, '5.24 MB', '42.9 GB', { value: '53.7%', sub: 'of an H100 (80 GB nominal)' }]);
  assert.deepEqual([scaleView('v3', data).block.value, scaleView('v3', data).reserve.value, scaleView('gpt3', data).block.value, scaleView('gpt3', data).reserve.value], ['1.12 MB', '9.21 GB', '75.5 MB', '9.66 GB']);
  const h100 = hbmFor(data.hardware.entries.find((e) => e.id === 'h100'));
  assert.equal(h100.basis, 'nominal');
  assert.equal(llama.share.value, `${sharePct(reserveMaxBytes(327_680, 131_072), h100.bytes).toFixed(1)}%`);
  assert.equal(reserveMaxBytes(327_680, 131_072), kvCacheBytes({ bytesPerToken: 327_680, tokens: 131_072 }));
  assert.equal(formatBytes(kvBytesPerBlock(327_680, 16)), '5.24 MB');
  assert.throws(() => scaleView('llama', { ...data, models: { entries: [] } }), RangeError);
  assert.throws(() => scaleView('llama', { ...data, hardware: { entries: [] } }), RangeError);
});

test('the try-this list is the storyboard\'s, every number computed', () => {
  assert.deepEqual(tryThis().map(({ text, insight }) => `${text} → Insight: ${insight}`), TRY_THIS);
  tryThis().forEach(({ text }) => assert.match(text, /[.:]$/));
  assert.match(tryThis()[2].rest, /\[\[prefix-caching\]\]/);
});

test('formatters: shares at one decimal, exact integers, real minus, plurals', () => {
  assert.deepEqual([shareText(25, 48), shareCell(30, 48), plural(1, 'slot'), plural(2, 'block'), exactBytes(327_680), bytesPair(70_272)], ['52.1%', { value: '62.5%', sub: '30 slots' }, '1 slot', '2 blocks', '327,680 B', { value: '70,272 B', sub: '≈ 70.3 kB' }]);
  assert.deepEqual([stepText(null), stepText(3), meanWaste([25, 22], 48)], ['not started', 'step 3', '49.0%']);
  assert.equal(statusText({ running: false, done: false }, 1, 0), 'not arrived');
  assert.equal(statusText({ running: false, done: false }, 1, 1), 'waiting');
  assert.deepEqual(simulate({ blockSize: 4, step: 3, sharedPrefix: false }).all.after.length, 7);
});

test('inputs are not mutated', () => {
  const frozen = JSON.stringify(data);
  Object.freeze(INITIAL_STATE);
  toyView(INITIAL_STATE, data);
  tryThis();
  lessonFor(data);
  assert.equal(JSON.stringify(data), frozen);
  assert.equal(JSON.stringify(TOY_REQUESTS), JSON.stringify(N.TOY_REQUESTS));
});

test('no bare slug or operator reaches a caption (lesson names, README 33)', () => {
  CAPTIONS.forEach((c) => assert.doesNotMatch(c, /[=+×÷]|paged-attention|prefix-caching/));
});
