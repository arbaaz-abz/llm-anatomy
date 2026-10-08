import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { formatBytes } from '../math/core.js';
import { kvBytesPerToken, kvBytesPerTokenMla, kvCacheBytes, decodeWork, sharePct } from '../math/memory.js';
import { LESSON, lessonFor } from '../architecture/concepts/kv-cache/content.js';
import { BELOW, factRows, intuition, ratioRange } from '../architecture/concepts/kv-cache/facts.js';
import {
  int, exactBytes, exactSpan, plural, ratioText, compactSub, spanText, shareText, fitText, workFor, checkWorkA, checkWorkB, INITIAL_STATE, SLIDER_VALUES,
} from '../architecture/concepts/kv-cache/format.js';
import { toyView, tryThis, snapShape, shapeOf, bytesPerTokenOf, MODEL_CHIPS, fact } from '../architecture/concepts/kv-cache/toy-view.js';
import * as N from '../architecture/concepts/kv-cache/numbers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_A, CHECK_WORK_B } from './kv-cache-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;
const view = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('6 facts rows, every placeholder resolves, no derived number is left unfilled', () => {
  assert.equal(LESSON.facts.rows.length, 6);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  factRows(data).forEach((row, i) => assert.doesNotMatch(row.claim, /—/, `row ${i + 1} has an unfilled computed number`));
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('kv-cache')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('"Check my work" for the default state is the storyboard text', () => {
  const v = view({});
  assert.equal(v.checkA, CHECK_WORK_A);
  assert.equal(v.checkB, CHECK_WORK_B);
  assert.equal(`${v.checkA}\n${v.checkB}`, CHECK_WORK);
});

// ---- dated text ----

test('hook, intuition and the notes under the stage fill from data with nothing missing', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(lesson.intuition[2], /In GPT-3 \(2020\) that came to 4,718,592 bytes \(4\.72 MB\) for every single token/);
  assert.match(lesson.intuition[2], /roughly 400 to 1,200 times less than GPT-3/);
  assert.match(lesson.animation.belowFor(8)[0], /GPT-3 \(2020\) 4,718,592 B; Llama-3\.1-70B 327,680 B; DeepSeek-V3 \(2024\) 70,272 B; DeepSeek-V4-Pro \(2026\) 4,000–12,000 B/);
  assert.equal(lesson.animation.belowFor(42).length, 0);
  assert.deepEqual(ratioRange(data), [400, 1200]);
  assert.equal(ratioRange(null), null);
});

test('facts rows print the storyboard\'s derived figures', () => {
  const claims = factRows(data).map((r) => fillClaim(r.claim, data).segments.map((s) => s.text).join(''));
  assert.match(claims[0], /^GPT-3 \(2020\): 96 layers × 96 heads × 128, .*4,718,592 B \(≈ 4\.72 MB\) per token at 2 bytes; 9\.66 GB for its 2,048-token context\.$/);
  assert.match(claims[1], /Llama-3\.1-70B: 80 layers, 8 KV heads × 128: 327,680 B \(≈ 328 kB\) per token; 42\.9 GB at its 131,072-token context, 53\.7% of an 80 GB H100\./);
  assert.match(claims[2], /70,272 B \(≈ 70\.3 kB\) per token, derived from its config; 73\.7 GB at 1,048,576 tokens\./);
  assert.match(claims[3], /about 4,000–12,000 B per token, a formula-derived estimate .*about 4\.19 GB–12\.6 GB for a 1,048,576-token conversation\./);
  assert.match(claims[4], /only its 18 full-attention layers grow a cache, 36,864 B \(≈ 36\.9 kB\) per token; its 18 sliding-window layers hold a fixed ~4\.72 MB per conversation\./);
});

test('the numbers the captions and the stage print equal the data (README lesson 29)', () => {
  assert.deepEqual([N.GPT3.layers, N.GPT3.kvHeads, N.GPT3.headDim, N.GPT3.context], ['layers', 'n_kv_heads', 'head_dim', 'context_length'].map((k) => model('gpt-3', k)));
  assert.deepEqual([N.LLAMA.layers, N.LLAMA.kvHeads, N.LLAMA.headDim, N.LLAMA.context], ['layers', 'n_kv_heads', 'head_dim', 'context_length'].map((k) => model('llama-3.1-70b', k)));
  assert.deepEqual([N.V3.layers, N.V3.dLatent, N.V3.dRope], ['layers', 'mla_kv_rank', 'mla_rope_dim'].map((k) => model('deepseek-v3', k)));
  assert.deepEqual(N.V4_PRO_BYTES, model('deepseek-v4-pro', 'kv_bytes_per_token'));
  assert.equal(N.GPU_BYTES, data.hardware.entries.find((e) => e.id === 'h100').facts.hbm_gb.value * 1e9);
  assert.equal(N.BYTES_PER_TOKEN.gpt3, model('gpt-3', 'kv_bytes_per_token'));
  assert.equal(N.BYTES_PER_TOKEN.llama, model('llama-3.1-70b', 'kv_bytes_per_token'));
  assert.equal(N.BYTES_PER_TOKEN.v3, model('deepseek-v3', 'kv_bytes_per_token'));
  assert.match(CAPTIONS[6], new RegExp(`stored ${formatBytes(N.BYTES_PER_TOKEN.gpt3)} for every token, because all ${N.GPT3.kvHeads} heads in all ${N.GPT3.layers} layers`));
  assert.match(CAPTIONS[6], new RegExp(`${int(N.GPT3.context)}-token context needed ${formatBytes(N.CACHE.gpt3Context)}`));
  assert.match(CAPTIONS[7], new RegExp(`needs ${formatBytes(N.CACHE.llamaOne)}, so a single ${N.GPU_BYTES / 1e9} GB GPU cannot hold two`));
  assert.match(CAPTIONS[5], /32 numbers, 64 bytes per token/);
  assert.equal(kvBytesPerToken({ ...N.TOY_SHAPE, bytesPerElem: 1 }), 32);
  assert.equal(N.BYTES_PER_TOKEN.toy, 64);
});

test('the stage\'s byte figures are the storyboard\'s, to the byte', () => {
  assert.equal(N.CACHE.gpt3Context, 9_663_676_416);
  assert.equal(N.CACHE.llamaOne, 42_949_672_960);
  assert.equal(N.CACHE.llamaTwo, 85_899_345_920);
  assert.equal(N.CACHE.toyFive, 320);
  assert.equal(N.AT_STOP.llama, 343_597_383_680);
  assert.equal(N.AT_STOP.v3, 73_685_532_672);
  assert.deepEqual([N.AT_STOP.v4Low, N.AT_STOP.v4High], [4_194_304_000, 12_582_912_000]);
  assert.deepEqual([N.gpuShare(N.CACHE.llamaOne), N.gpuShare(N.AT_STOP.v3), N.gpuShare(N.AT_STOP.v4Low), N.gpuShare(N.AT_STOP.v4High)], [53.7, 92.1, 5.2, 15.7]);
  assert.equal(N.GPU_FILL(N.AT_STOP.llama), 1);
  assert.deepEqual([4, 1, 2, 3, 4].slice(1).map((g) => N.passTotals(g, false).positions), [4, 9, 15, 22]);
  assert.deepEqual([1, 2, 3, 4].map((g) => N.passTotals(g, true).positions), [4, 5, 6, 7]);
  assert.deepEqual([1, 2, 3, 4].map((g) => N.passTotals(g, true).keyReads), [10, 15, 21, 28]);
  assert.deepEqual([1, 2, 3, 4].map((g) => N.passTotals(g, false).keyReads), [10, 25, 46, 74]);
});

// ---- format.js ----

test('number formats: separators, real minus, plurals and three-figure ratios', () => {
  assert.deepEqual([int(1_499_500), int(-3), exactBytes(131_072), plural(1, 'block'), plural(2, 'KV head'), plural(96, 'number')], ['1,499,500', '−3', '131,072 B', '1 block', '2 KV heads', '96 numbers']);
  assert.deepEqual([3.1428, 750.12, 583.6, 1179.6, 1, 12.3].map(ratioText), ['3.14×', '750×', '584×', '1,180×', '1×', '12.3×']);
  assert.throws(() => ratioText(0), RangeError);
  assert.deepEqual([compactSub(9999), compactSub(1_499_500), compactSub(1_166_666_500)], ['', '1.5M', '1.17B']);
  assert.equal(exactSpan([4000, 12000]), '4,000–12,000 B');
  assert.equal(spanText([4000, 12000], int), '4,000–12,000');
  assert.equal(spanText(7, int), '7');
});

test('shares and fit use sharePct, with "<0.1%" for a tiny cache', () => {
  assert.deepEqual([shareText(N.CACHE.llamaOne, N.GPU_BYTES), shareText(N.CACHE.llamaTwo, N.GPU_BYTES), shareText(131_072, N.GPU_BYTES)], ['53.7%', '107.4%', '<0.1%']);
  assert.equal(shareText(0, N.GPU_BYTES), '0.0%');
  assert.equal(fitText(N.CACHE.llamaTwo, N.GPU_BYTES), 'does not fit (before the weights)');
  assert.equal(fitText(N.CACHE.llamaOne, N.GPU_BYTES), 'fits (before the weights)');
  assert.equal(fitText([N.GPU_BYTES / 2, N.GPU_BYTES * 2], N.GPU_BYTES), 'may not fit at the high end (before the weights)');
});

test('"Check my work" keeps its layout for other states', () => {
  assert.equal(checkWorkA({ prompt: 4, reply: 1 }), 'no cache: 4                              = 4 positions\ncache:    4 (prefill)                    = 4 positions');
  assert.equal(checkWorkA({ prompt: 16, reply: 2 }).split('\n')[0], 'no cache: 16 + 17                        = 33 positions');
  assert.equal(checkWorkA({ prompt: 1000, reply: 1000 }), 'no cache: 1,000 + 1,001 + … + 1,999      = 1,499,500 positions\ncache:    1,000 (prefill) + 999 × 1      =     1,999 positions');
  const gpt3 = view({ model: 'gpt3', context: 131_072 });
  assert.equal(gpt3.checkB, ['bytes per token = 2 (K and V) × 96 blocks × 96 KV heads × 128 numbers × 2 bytes', '                = 4,718,592 B', '× 131,072 tokens                         = 618,475,290,624 B = 618 GB'].join('\n'));
  assert.match(view({ model: 'llama', sequences: 8 }).checkB, /× 2,048 tokens × 8 conversations\s+= 5,368,709,120 B = 5\.37 GB$/);
  assert.equal(view({ model: 'v3' }).checkB.split('\n')[0], 'bytes per token = 61 blocks × (512 latent + 64 position key) numbers × 2 bytes');
  assert.equal(view({ model: 'v4pro', context: 1_048_576 }).checkB.split('\n').at(-1), '× 1,048,576 tokens                       = 4,194,304,000–12,582,912,000 B = 4.19 GB–12.6 GB');
  assert.equal(view({ model: 'toy', bytes: 1, kvHeads: 1 }).checkB.split('\n')[0], 'bytes per token = 2 (K and V) × 2 blocks × 1 KV head × 4 numbers × 1 byte');
});

// ---- toy-view.js: the storyboard's try-this numbers ----

test('panel A: the animation\'s reply, and the 1,000-token worked example', () => {
  const v = view({});
  assert.deepEqual([v.work.current.positions.value, v.work.flipped.positions.value, v.work.current.keyReads.value, v.work.flipped.keyReads.value], ['7', '22', '28', '74']);
  assert.deepEqual([v.work.positionsRatio, v.work.keyReadsRatio], ['3.14×', '2.64×']);
  const off = view({ cache: false });
  assert.deepEqual([off.work.current.positions.value, off.work.flipped.positions.value, off.work.positionsRatio], ['22', '7', '3.14×']);
  const big = view({ prompt: 1000, reply: 1000 });
  assert.deepEqual([big.work.current.positions, big.work.flipped.positions], [{ value: '1,999', sub: '' }, { value: '1,499,500', sub: '1.5M' }]);
  assert.deepEqual([big.work.current.keyReads.value, big.work.flipped.keyReads.value, big.work.positionsRatio, big.work.keyReadsRatio], ['1,999,000', '1,166,666,500', '750×', '584×']);
  assert.deepEqual(workFor({ prompt: 4, reply: 4, cache: true }), { current: decodeWork({ prompt: 4, generated: 4, cache: true }), flipped: decodeWork({ prompt: 4, generated: 4, cache: false }) });
});

test('panel B: the toy at its own context, GPT-3, Llama-3.1-70B, DeepSeek-V3, DeepSeek-V4-Pro', () => {
  const toy = view({});
  assert.deepEqual([toy.bytesPerToken.value, toy.cacheOne.value, toy.cacheOne.sub, toy.gpus[0].share, toy.contextNote], ['64 B', '131 kB', '131,072 B', '<0.1%', 'The toy has no context limit of its own.']);
  const llama = view({ model: 'llama', context: 131_072 });
  assert.deepEqual([llama.bytesPerToken.value, llama.bytesPerToken.sub, llama.cacheOne.value, llama.gpus[0].share, llama.gpus[1].share], ['327,680 B', '328 kB', '42.9 GB', '53.7%', '14.9%']);
  assert.equal(llama.contextNote, 'within this model\'s own 131,072-token context');
  const two = view({ model: 'llama', context: 131_072, sequences: 2 });
  assert.deepEqual([two.cacheAll.value, two.gpus[0].share, two.gpus[0].fit, two.gpus[0].fill], ['85.9 GB', '107.4%', 'does not fit (before the weights)', 1]);
  const gpt3 = view({ model: 'gpt3', context: 2048, sequences: 8 });
  assert.deepEqual([gpt3.cacheAll.value, gpt3.gpus[0].share], ['77.3 GB', '96.6%']);
  const gpt3Long = view({ model: 'gpt3', context: 131_072 });
  assert.deepEqual([gpt3Long.cacheOne.value, gpt3Long.contextNote], ['618 GB', 'beyond this model\'s 2,048-token context: a what-if at its shape']);
  const v3 = view({ model: 'v3', context: 131_072 });
  assert.deepEqual([v3.bytesPerToken.value, v3.bytesPerToken.sub, v3.cacheOne.value, v3.shapeRows.map((r) => r.value)], ['70,272 B', '70.3 kB', '9.21 GB', ['61', '512', '64', '2']]);
  assert.match(v3.formulaNote, /latent/);
  const v4 = view({ model: 'v4pro', context: 1_048_576 });
  assert.deepEqual([v4.bytesPerToken.value, v4.cacheOne.value, v4.gpus[0].share, v4.gpus[1].share, v4.shapeRows], ['4,000–12,000 B', '4.19 GB–12.6 GB', '5.2%–15.7%', '1.5%–4.4%', []]);
  assert.match(v4.contextNote, /beyond this model's 1,000,000-token context/);
  assert.equal(v4.formulaNote, 'formula-derived estimate; the layer mix is uncertain');
});

test('try this 3: KV heads 96 → 8 on GPT-3\'s shape; the chips snap to the preset', () => {
  const snapped = snapShape('gpt3', data);
  assert.deepEqual(snapped, { layers: 96, kvHeads: 96, headDim: 128, bytes: 2 });
  assert.equal(snapShape('toy', data), null);
  assert.equal(snapShape('v3', data), null);
  const eight = view({ model: 'toy', ...snapped, kvHeads: 8, context: 131_072 });
  assert.deepEqual([eight.bytesPerToken.value, eight.bytesPerToken.sub, eight.cacheOne.value], ['393,216 B', '393 kB', '51.5 GB']);
  assert.equal(bytesPerTokenOf(shapeOf({ ...INITIAL_STATE, model: 'gpt3' }, data)), kvBytesPerToken({ layers: 96, kvHeads: 96, headDim: 128, bytesPerElem: 2 }));
  assert.equal(bytesPerTokenOf(shapeOf({ ...INITIAL_STATE, model: 'v3' }, data)), kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 }));
});

test('the try-this text prints the storyboard\'s numbers, computed from the same functions', () => {
  const [one, two, three] = tryThis(data).map((t) => t.text);
  assert.match(one, /22 positions\. Cache on: 7\. .* 1,499,500 against 1,999, 750× fewer\. .* 1,166,666,500 against 1,999,000/);
  assert.match(two, /42\.9 GB, 53\.7% of an H100\. .* 85\.9 GB \(107\.4%\), "does not fit \(before the weights\)"\. .* 77\.3 GB, 96\.6%\./);
  assert.match(three, /4\.72 MB, 328 kB, then 70\.3 kB per token \(618 GB, 42\.9 GB, then 9\.21 GB per conversation\)\..*4\.72 MB becomes 393 kB\./);
  assert.match(tryThis(data)[2].rest, /\[\[kv-compression\]\]/);
});

test('lessons are named in running text, never as a bare slug or a possessive (cross-lesson X-4)', () => {
  const lesson = lessonFor(data);
  assert.match(lesson.facts.framing, /fewer key\/value sets or a compressed latent \(\[\[kv-compression\]\]\), and with windows/);
  assert.deepEqual(lesson.animation.belowFor(7), ['Two conversations do not fit even before the weights; [[prefill-decode]] adds those.']);
  assert.equal(tryThis(data)[2].rest, ' How models do that without losing quality: [[kv-compression]].');
  assert.doesNotMatch(view({ model: 'v3' }).formulaNote, /kv-compression/);
  assert.doesNotMatch(lesson.math.blocks.map((b) => b.tex).join(' '), /kv-compression/);
});

test('slider stops, chips and missing data', () => {
  assert.deepEqual(MODEL_CHIPS.map((c) => c.value), ['toy', 'gpt3', 'llama', 'v3', 'v4pro']);
  assert.deepEqual(SLIDER_VALUES.context, [2048, 8192, 32_768, 131_072, 262_144, 1_048_576]);
  assert.equal(fact(data, 'gpt-3', 'layers'), 96);
  assert.throws(() => fact(data, 'gpt-3', 'no_such_key'), RangeError);
  assert.throws(() => toyView({ ...INITIAL_STATE, model: 'gpt3' }, { models: { entries: [] }, hardware: data.hardware }), RangeError);
  assert.throws(() => toyView(INITIAL_STATE, { models: data.models, hardware: { entries: [] } }), RangeError);
  assert.throws(() => checkWorkA({ prompt: 4, reply: 0 }), RangeError);
});

test('inputs are not mutated', () => {
  const state = Object.freeze({ ...INITIAL_STATE, model: 'llama', context: 131_072, sequences: 2 });
  const frozen = JSON.stringify(data);
  toyView(state, data);
  tryThis(data);
  lessonFor(data);
  assert.equal(JSON.stringify(data), frozen);
  assert.equal(sharePct(N.CACHE.llamaOne, N.GPU_BYTES), 53.7);
  assert.equal(kvCacheBytes({ bytesPerToken: 64, tokens: 2048 }), 131_072);
  assert.equal(checkWorkB({ shape: { kind: 'mha', layers: 2, kvHeads: 2, headDim: 4, bytesPerElem: 2 }, bytesPerToken: 64, tokens: 2048, sequences: 1, total: 131_072 }), CHECK_WORK_B);
  assert.equal(intuition(null).length, 3);
});
