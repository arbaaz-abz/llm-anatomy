import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { kvBytesPerToken, kvBytesPerTokenMla, kvCacheBytes, kvGroups } from '../math/memory.js';
import { formatBytes } from '../math/core.js';
import { TOY } from '../math/attention.js';
import { LESSON, lessonFor } from '../architecture/concepts/kv-compression/content.js';
import { modelShapes, requireShapes } from '../architecture/concepts/kv-compression/shapes.js';
import { numbersPerLayer, bytesPerToken, timesSmaller, storedHeads, wiring } from '../architecture/concepts/kv-compression/scheme.js';
import { ladder, mlaRatio, realHeadBaseline, compare } from '../architecture/concepts/kv-compression/ladder.js';
import { int, timesText, weightText, checkWork, groupsLine } from '../architecture/concepts/kv-compression/format.js';
import { INITIAL_STATE, toyView, presetPatch, stopsFor, divisors } from '../architecture/concepts/kv-compression/toy-view.js';
import { tryThis } from '../architecture/concepts/kv-compression/try-this.js';
import { patternFor } from '../architecture/concepts/kv-compression/pattern.js';
import { absorptionCheck, byHand } from '../architecture/concepts/kv-compression/absorb.js';
import { v3Ratio } from '../architecture/concepts/kv-compression/facts.js';
import { CAPTIONS, CHECK_WORK, PATTERN_A, PATTERN_B_SHARED, PATTERN_B_OWN } from './kv-compression-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const shapes = requireShapes(data);
const fact = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;

// ---- the lesson spec ----

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
});

test('the captions\' real-model numbers agree with the data: "8 KV heads" (frame 3) and "67 times" (frame 9)', () => {
  assert.equal(fact('llama-3.1-70b', 'n_kv_heads'), 8);
  assert.match(CAPTIONS[2], /8 KV heads became a common choice/);
  const ratio = fact('gpt-3', 'kv_bytes_per_token') / fact('deepseek-v3', 'kv_bytes_per_token');
  assert.equal(Math.round(ratio), 67);
  assert.match(CAPTIONS[8], /by 67 times/);
  assert.match(CAPTIONS[1], /8 times smaller/); // toy: 8 query heads share 1
});

test('9 facts rows, every placeholder resolves against data/models.json', () => {
  assert.equal(LESSON.facts.rows.length, 9);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});

test('the rows print the storyboard\'s numbers from the data', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.equal(rows[0], 'GPT-3 (2020): MHA, 96 KV heads × 128, 4,718,592 B per token.');
  assert.equal(rows[1], 'Llama-3.1-70B: GQA with 8 KV heads × 128, 327,680 B per token.');
  assert.equal(rows[2], 'gpt-oss-120b (2025): 64 query heads share 8 KV heads, each 64 wide.');
  assert.equal(rows[3], 'MiniMax-M3 (2026): 64 query heads, 4 KV heads × 128, 122,880 B per token (derived).');
  assert.equal(rows[4], 'Qwen3.8 (2026): its attention layers use 64 query heads and 4 KV heads.');
  assert.equal(rows[5], 'DeepSeek-V3 (2024): MLA, latent 512 + position key 64 per layer: 70,272 B per token, 56.9× less than MHA at its 128 heads.');
  assert.equal(rows[6], 'MLA is also used by Kimi K3 (in its 24 full-attention layers) and GLM-5.3 (latent 512, query latent 2,048).');
  assert.match(rows[7], /^The cost: MLA's decode compute is high\. GLM-5 \(2026\) reworked it: MLA-256: head dim 192 → 256, fewer heads; Muon Split needed to match GQA-8\.$/);
  assert.equal(rows[8], 'DeepSeek-V4-Pro (2026): 1 KV head, 512 wide, shared by all query heads, then compressed ([[long-context-attention]]).');
});

test('every prose placeholder resolves and nothing prints "—"', () => {
  const lesson = lessonFor(data);
  const below = [0, 1, 2, 3, 4, 5, 6, 7, 8].flatMap((i) => lesson.animation.belowFor(i));
  const prose = [lesson.hook, ...lesson.intuition, lesson.facts.framing, lesson.animation.standIn, lesson.toy.intro, ...lesson.takeaways, ...below];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => fillText(r.claim, data)), ...lesson.math.blocks.map((b) => b.tex)].forEach((text) => assert.ok(!fillText(text, data).includes('—'), text.slice(0, 40)));
});

test('the hook and framing are the storyboard\'s, with the data\'s numbers', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.hook, 'In GPT-3, every one of the 96 heads in every layer stored its own keys and values. Do the heads really need separate copies, and how did DeepSeek-V3 store 57 times less per layer than it would with a key and value per head, without taking those separate keys and values away?');
  assert.match(lesson.facts.framing, /^Almost no 2026 model uses plain MHA\. Most use GQA with 4 or 8 KV heads or MLA;/);
  assert.equal(Math.round(v3Ratio(data)), 57);
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('kv-compression')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the page below the stage names the real numbers of frames 3, 6, 8 and 9', () => {
  const below = lessonFor(data).animation.belowFor;
  assert.match(below(2)[0], /^8 KV heads in Llama-3\.1-70B and 8 in gpt-oss-120b \(2025\)/);
  assert.match(below(5)[0], /^DeepSeek-V3 \(2024\): 512 \+ 64 = 576 numbers per token per layer\./);
  assert.match(below(7)[0], /MLA stores 576 numbers per layer where two shared KV heads would store 512/);
  const lines = below(8);
  assert.equal(lines[0], 'GPT-3 (2020, MHA): 4,718,592 B per token (4.72 MB); 618 GB at 131,072 tokens.');
  assert.equal(lines[1], 'Llama-3.1-70B (GQA-8): 327,680 B per token (328 kB); 42.9 GB at 131,072 tokens.');
  assert.equal(lines[2], 'MiniMax-M3 (2026, GQA-4): 122,880 B per token (123 kB); 16.1 GB at 131,072 tokens.');
  assert.equal(lines[3], 'DeepSeek-V3 (2024, MLA): 70,272 B per token (70.3 kB); 9.21 GB at 131,072 tokens.');
  assert.equal(lines[4], 'GPT-3 at 131,072 tokens is a what-if at its shape; its own context was 2,048.');
  assert.match(lines[5], /^2026 moved on: DeepSeek-V4-Pro keeps one wide KV head and compresses it/);
});

// ---- the shapes read from the data ----

test('each real shape is read from the data and reproduces the data\'s bytes per token', () => {
  assert.deepEqual(['gpt3', 'llama', 'minimax', 'v3'].map((k) => shapes[k].scheme), ['mha', 'gqa', 'gqa', 'mla']);
  const bytes = (s) => bytesPerToken(s, { layers: s.layers, bytesPerElem: 2 });
  assert.equal(bytes(shapes.gpt3), fact('gpt-3', 'kv_bytes_per_token'));
  assert.equal(bytes(shapes.llama), fact('llama-3.1-70b', 'kv_bytes_per_token'));
  assert.equal(bytes(shapes.minimax), fact('minimax-m3', 'kv_bytes_per_token'));
  assert.equal(bytes(shapes.v3), fact('deepseek-v3', 'kv_bytes_per_token'));
});

test('a missing fact makes a shape null and requireShapes names it', () => {
  assert.deepEqual(Object.values(modelShapes(null)), [null, null, null, null]);
  assert.throws(() => requireShapes(null), (e) => e instanceof RangeError && /gpt-3, llama-3\.1-70b, minimax-m3, deepseek-v3/.test(e.message));
  const broken = { models: { entries: data.models.entries.filter((e) => e.id !== 'minimax-m3') } };
  assert.equal(modelShapes(broken).minimax, null);
  assert.notEqual(modelShapes(broken).gpt3, null);
});

// ---- the numbers (storyboard §6 worked examples, to the byte) ----

test('toy numbers per token per layer: MHA 64, GQA-2 16, MQA 8, MLA 10', () => {
  const toy = { queryHeads: 8, headDim: 4, dLatent: 8, dRope: 2 };
  assert.deepEqual(['mha', 'mqa', 'mla'].map((scheme) => numbersPerLayer({ ...toy, scheme })), [64, 8, 10]);
  assert.equal(numbersPerLayer({ ...toy, scheme: 'gqa', kvHeads: 2 }), 16);
  assert.equal(kvBytesPerToken({ layers: 1, kvHeads: 2, headDim: 4, bytesPerElem: 1 }), 16);
  assert.equal(kvBytesPerTokenMla({ layers: 1, dLatent: 8, dRope: 2, bytesPerElem: 1 }), 10);
});

test('real byte counts: GPT-3 shape as GQA-8 and MQA, MiniMax-M3, DeepSeek-V3 as MHA and as MLA', () => {
  const gpt3 = shapes.gpt3;
  const at = (shape, scheme, kvHeads) => bytesPerToken({ ...shape, scheme, kvHeads }, { layers: shape.layers, bytesPerElem: 2 });
  assert.equal(at(gpt3, 'gqa', 8), 393_216);
  assert.equal(at(gpt3, 'mqa'), 49_152);
  assert.equal(at(shapes.minimax, 'gqa', 4), 122_880);
  assert.equal(at(shapes.v3, 'mha'), 3_997_696);
  assert.equal(at(shapes.v3, 'mla'), 70_272);
  assert.equal(kvCacheBytes({ bytesPerToken: 122_880, tokens: 131_072 }), 16_106_127_360);
});

test('"times smaller than MHA" equals queryHeads ÷ kvHeads for every GQA setting (README lesson 16)', () => {
  [shapes.gpt3, shapes.llama, shapes.minimax, shapes.v3, { queryHeads: 8, headDim: 4, layers: 1 }].forEach((shape) => {
    divisors(shape.queryHeads).forEach((kvHeads) => {
      const ratio = timesSmaller({ ...shape, scheme: 'gqa', kvHeads }, { layers: shape.layers, bytesPerElem: 2 });
      assert.equal(ratio, shape.queryHeads / kvHeads, `${shape.queryHeads} heads, ${kvHeads} KV heads`);
    });
  });
  assert.equal(timesText(timesSmaller({ ...shapes.v3, scheme: 'mla' }, { layers: 61, bytesPerElem: 2 })), '56.9×');
});

test('MLA sits between MQA and GQA-2 in the toy, and between GQA-2 and GQA-4 at DeepSeek-V3\'s shape (frame 8, lesson 17)', () => {
  const toy = ladder({ queryHeads: 8, headDim: 4, dLatent: 8, dRope: 2 });
  assert.deepEqual(toy.map((r) => [r.label, r.value]), [['MHA', 64], ['GQA-8', null], ['GQA-2', 16], ['MLA', 10], ['MQA', 8]]);
  const real = ladder(shapes.v3);
  assert.deepEqual(real.map((r) => [r.label, r.value]), [['MHA', 32_768], ['GQA-8', 2048], ['GQA-2', 512], ['MLA', 576], ['MQA', 256]]);
  const gqa4 = numbersPerLayer({ ...shapes.v3, scheme: 'gqa', kvHeads: 4 });
  assert.ok(512 < 576 && 576 < gqa4, `GQA-4 is ${gqa4}`);
  assert.ok(8 < 10 && 10 < 16);
  assert.equal(timesText(mlaRatio(real)), '56.9×');
  assert.equal(realHeadBaseline(shapes.v3).toFixed(1), '71.1');
  assert.equal(shapes.v3.qkWidth, 192);
});

test('frame 9: four models, bytes per token and the cache at 131,072 tokens; GPT-3 is the only what-if', () => {
  const rows = ['gpt3', 'llama', 'minimax', 'v3'].map((k) => compare(shapes[k]));
  assert.deepEqual(rows.map((r) => r.scheme), ['MHA', 'GQA-8', 'GQA-4', 'MLA']);
  assert.deepEqual(rows.map((r) => formatBytes(r.perToken)), ['4.72 MB', '328 kB', '123 kB', '70.3 kB']);
  assert.deepEqual(rows.map((r) => formatBytes(r.atContext)), ['618 GB', '42.9 GB', '16.1 GB', '9.21 GB']);
  assert.deepEqual(rows.map((r) => r.isWhatIf), [true, false, false, false]);
  assert.equal(timesText(rows[0].perToken / rows[3].perToken), '67.1×');
});

test('kvGroups wiring: eight entries, every KV head serves queryHeads ÷ kvHeads, non-divisors throw', () => {
  assert.deepEqual(kvGroups({ queryHeads: 8, kvHeads: 2 }), [0, 0, 0, 0, 1, 1, 1, 1]);
  assert.deepEqual(wiring({ scheme: 'mqa', queryHeads: 8 }), [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(wiring({ scheme: 'mha', queryHeads: 8 }), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.equal(storedHeads({ scheme: 'gqa', queryHeads: 8, kvHeads: 4 }), 4);
  assert.throws(() => storedHeads({ scheme: 'mla', queryHeads: 8 }), RangeError);
  assert.throws(() => kvGroups({ queryHeads: 8, kvHeads: 3 }), /kvHeads must divide queryHeads/);
});

test('format helpers: whole ratios print without .0, weights print 0 and 1 as integers, groups pluralize', () => {
  assert.deepEqual([timesText(12), timesText(56.89), timesText(6.4), timesText(1)], ['12×', '56.9×', '6.4×', '1×']);
  assert.deepEqual([weightText(0), weightText(1), weightText(0.6789), weightText(-Infinity), weightText(null), weightText(Number.NaN)], ['0', '1', '0.679', '0', '', '']);
  assert.equal(int(4718592), '4,718,592');
  assert.equal(int(-3), '−3');
  assert.equal(groupsLine({ scheme: 'mqa', queryHeads: 96, stored: 1 }), '96 query heads read 1 KV head: 96 query heads per KV head');
  assert.equal(groupsLine({ scheme: 'mla', queryHeads: 8, stored: 1 }), '8 query heads rebuild their own keys and values from 1 latent per token');
});

// ---- frame 4 and the pattern toggle ----

test('the frame-4 pattern is attention\'s two heads: head B on A\'s keys and on its own (rows to 3 d.p.)', () => {
  const { a, shared, own } = patternFor();
  const cells = (w) => w.flat().map(weightText);
  assert.deepEqual(cells(a.weights), PATTERN_A);
  assert.deepEqual(cells(shared.weights), PATTERN_B_SHARED);
  assert.deepEqual(cells(own.weights), PATTERN_B_OWN);
  assert.equal(shared.weights[1][0].toFixed(3), '0.731'); // from "cat", head B puts 0.731 on "The"
  assert.equal(a.weights[1][1].toFixed(3), '0.679'); // head A puts 0.679 on "cat"
});

test('the absorption check: both orders give 0.412632, and the by-hand line gives 55 both ways', () => {
  const check = absorptionCheck();
  assert.equal(check.rebuilt.toFixed(6), '0.412632');
  assert.equal(check.folded.toFixed(6), '0.412632');
  assert.deepEqual(byHand(), { rebuilt: 55, folded: 55 });
  assert.ok(LESSON.math.blocks.some((b) => b.tex.includes('0.412632 = 0.412632')));
});

test('the math panel\'s DeepSeek-V3 block is templated from the data', () => {
  const tex = lessonFor(data).math.blocks.at(-1).tex;
  assert.match(tex, /61 \\cdot \(512 \+ 64\) \\cdot 2 = 70\{,\}272/);
  assert.match(tex, /= 56\.9$/);
  assert.match(tex, /hl-lat/);
  assert.ok(LESSON.math.blocks.every((b) => !/DeepSeek-V3/.test(b.tex)), 'no dated numbers without data');
});

// ---- the toy ----

test('the toy opens on the animation\'s first numbers and prints the storyboard\'s "Check my work"', () => {
  assert.deepEqual({ ...INITIAL_STATE }, { model: 'toy', scheme: 'mha', kvHeads: 2, latent: 8, context: 131_072, shareKeys: true });
  const view = toyView(INITIAL_STATE, shapes);
  assert.equal(view.checkWork, CHECK_WORK);
  assert.deepEqual([view.perLayer, view.bytes, view.ratio, view.cache], ['64', '128 B', '1×', '16.8 MB']);
  assert.equal(view.groups, '8 query heads read 8 KV heads: 1 query head per KV head');
  assert.equal(view.whatIf, '');
  assert.deepEqual(view.wiring.groups, [0, 1, 2, 3, 4, 5, 6, 7]);
});

test('toy views for the other schemes and the real chips', () => {
  const gqa = toyView({ ...INITIAL_STATE, scheme: 'gqa' }, shapes);
  assert.deepEqual([gqa.perLayer, gqa.ratio], ['16', '4×']);
  assert.equal(gqa.checkWork.split('\n')[0], 'numbers per token per layer: 2 (K and V) × 2 KV heads × 4 numbers = 16');
  const mla = toyView({ ...INITIAL_STATE, scheme: 'mla' }, shapes);
  assert.deepEqual([mla.perLayer, mla.bytes, mla.ratio], ['10', '20 B', '6.4×']);
  assert.equal(mla.checkWork.split('\n')[0], 'numbers per token per layer: 8 (latent) + 2 (position key) = 10');
  const v3 = toyView({ ...INITIAL_STATE, ...presetPatch('v3', shapes) }, shapes);
  assert.deepEqual([v3.perLayer, v3.bytes, v3.bytesSub, v3.ratio, v3.cache, v3.whatIf, v3.wiring], ['576', '70,272 B', '70.3 kB', '56.9×', '9.21 GB', '', null]);
  assert.equal(v3.spec, 'DeepSeek-V3: 61 layers, 128 query heads of 128 numbers. The position key stays at 64 numbers.');
  const gpt3 = toyView({ ...INITIAL_STATE, ...presetPatch('gpt3', shapes) }, shapes);
  assert.deepEqual([gpt3.bytes, gpt3.cache, gpt3.whatIf], ['4,718,592 B', '618 GB', 'a what-if at this shape; its own context was 2,048']);
});

test('model chips set the real scheme, a GQA starting point that divides the heads, and the real latent', () => {
  assert.deepEqual(presetPatch('toy', shapes), { model: 'toy', scheme: 'mha', kvHeads: 2, latent: 8 });
  assert.deepEqual(presetPatch('gpt3', shapes), { model: 'gpt3', scheme: 'mha', kvHeads: 8, latent: 512 });
  assert.deepEqual(presetPatch('llama', shapes), { model: 'llama', scheme: 'gqa', kvHeads: 8, latent: 512 });
  assert.deepEqual(presetPatch('minimax', shapes), { model: 'minimax', scheme: 'gqa', kvHeads: 4, latent: 512 });
  assert.deepEqual(presetPatch('v3', shapes), { model: 'v3', scheme: 'mla', kvHeads: 8, latent: 512 });
  ['toy', 'gpt3', 'llama', 'minimax', 'v3'].forEach((m) => {
    const patch = presetPatch(m, shapes);
    const stops = stopsFor(m, shapes);
    assert.ok(stops.kvHeads.includes(patch.kvHeads), `${m} KV heads ${patch.kvHeads}`);
    assert.ok(stops.latent.includes(patch.latent), `${m} latent ${patch.latent}`);
  });
  assert.deepEqual(stopsFor('toy', shapes).kvHeads, [1, 2, 4, 8]);
  assert.deepEqual(stopsFor('v3', shapes).kvHeads, [1, 2, 4, 8, 16, 32, 64, 128]);
});

test('the three "Try this" prompts print the storyboard\'s numbers', () => {
  const [one, two, three] = tryThis(shapes);
  assert.equal(one.prompt, 'Pick GPT-3 with MHA: 4,718,592 B per token. Switch to GQA with 8 KV heads: 393,216 B (12× smaller); MQA: 49,152 B (96×).');
  assert.match(one.insight, /^the cache shrinks exactly in proportion to KV heads; the 96 query heads, and the 96 patterns, stay\.$/);
  assert.equal(two.prompt, 'Switch the pattern: head B\'s row for "cat" goes from [0.798, 0.202] to [0.731, 0.269] when it reads head A\'s keys, and its row for "down" from most weight on "sat" (0.578) to 0.366 on "sat" and 0.285 on itself; head A\'s rows stay the same.');
  assert.equal(three.prompt, 'Pick DeepSeek-V3 (it opens on MLA): 576 numbers per token per layer, 70,272 B per token, 56.9× smaller than MHA at its 128 heads. Switch to GQA with 2 KV heads: 512 numbers; MQA: 256.');
});

test('checkWork pluralizes one head and one layer', () => {
  const lines = checkWork({ scheme: 'mqa', stored: 1, headDim: 128, dLatent: 512, dRope: 64, perLayer: 256, layers: 1, bytesPerElem: 2, bytes: 512, mhaPerLayer: 32_768, ratio: 128 }).split('\n');
  assert.equal(lines[0], 'numbers per token per layer: 2 (K and V) × 1 KV head × 128 numbers = 256');
  assert.equal(lines[1], 'bytes per token: 256 numbers × 1 layer × 2 bytes = 512 B');
  assert.equal(lines[2], 'compared with MHA at this shape (32,768 numbers per layer): 128× smaller');
});

test('nothing here mutates the data or the frozen toy', () => {
  const before = JSON.stringify(data);
  const toyBefore = JSON.stringify(TOY);
  lessonFor(data);
  toyView({ ...INITIAL_STATE, scheme: 'gqa' }, shapes);
  tryThis(shapes);
  patternFor();
  assert.equal(JSON.stringify(data), before);
  assert.equal(JSON.stringify(TOY), toyBefore);
});
