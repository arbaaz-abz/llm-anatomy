import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../architecture/concepts/long-context-attention/content.js';
import { belowFor } from '../architecture/concepts/long-context-attention/facts.js';
import { int, trimNumber, cellText, formatFor, windowScoresFor, sinkSplit } from '../architecture/concepts/long-context-attention/format.js';
import { INITIAL_STATE, toyView, visibleControls, patternView, sinkView, linearView, realView, realOptions, tryThis, linearNote } from '../architecture/concepts/long-context-attention/toy-view.js';
import { gptOssCache, qwenCache, minimaxCache, v4Estimate, layerColumns } from '../architecture/concepts/long-context-attention/real-scale.js';
import { CAPTIONS } from './long-context-attention-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const lookup = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;
const state = (patch) => ({ ...INITIAL_STATE, ...patch });

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('12 facts rows; every placeholder in rows, prose and notes resolves, and nothing prints "—"', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 12);
  lesson.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  const below = CAPTIONS.flatMap((_, i) => lesson.animation.belowFor(i));
  const prose = [lesson.hook, ...lesson.intuition, lesson.toy.intro, lesson.animation.standIn, lesson.facts.framing, ...lesson.takeaways, ...below];
  prose.forEach((t) => assert.ok(!t.includes('—'), t.slice(0, 50)));
  [...lesson.facts.rows.map((r) => r.claim), ...LESSON.intuition].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 40)));
  lesson.facts.rows.forEach((r) => assert.ok(!fillText(r.claim, data).includes('—'), r.claim.slice(0, 40)));
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('long-context-attention')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('the storyboard\'s row numbers come out of the data', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /window-128 layers \(1 full : 1 window\).*36,864 B per token plus about 4\.72 MB fixed/);
  assert.match(rows[4], /top 2,048/);
  assert.match(rows[5], /blocks of 128, top 16 per group, from layer 4; MiniMax claims 28\.4× less/);
  assert.match(rows[6], /every 4 tokens merged, then the top 1,024 read \(CSA\), or every 128 merged \(HCA\).*128-token window.*0\.27× .* 0\.1× /);
  assert.match(rows[7], /about 4 kB–12 kB per token/);
  assert.match(rows[8], /15 full layers of 60 with 2 KV heads × 256: 30,720 B per token/);
  assert.match(rows[9], /69 Kimi Delta Attention layers \+ 24 MLA layers.*48B \/ 3B.*75% less KV cache and about 6×/);
});

test('the notes under the stage print the storyboard\'s real-scale numbers', () => {
  const filled = belowFor(data).map((paragraphs) => paragraphs.map((p) => fillText(p, data)));
  assert.match(filled[0][0], /reads 1,048,576 keys.*keeps 1,048,576 entries/);
  assert.match(filled[5][0], /reads 2,048 entries and stores 1,048,576/);
  assert.match(filled[6][0], /CSA stores 262,144 merged entries \(every 4\) and reads 1,024 \+ 128 window; HCA merges every 128 and reads all 8,192/);
  assert.match(filled[8][0], /Qwen3\.8: 3 linear : 1 full, 23 groups of four, 92 layers .* Kimi K3: 69 linear \+ 24 full \(MLA\) = 93 layers/);
  assert.match(filled[8][1], /15 of 60 layers are full, 30,720 B per token \(32\.2 GB at 1M\)/);
  assert.deepEqual(filled[9].slice(1, 3), ['Full attention: reads 1,048,576, stores 1,048,576.', 'Window 128: reads 128, stores 128.']);
  assert.deepEqual(filled[9].slice(3), [
    'DSA, top 2,048: reads 2,048 (plus the indexer\'s pass), stores 1,048,576.',
    'MSA, 16 blocks of 128: reads about 2,048 (the newest block may be extra), stores 1,048,576.',
    'CSA (merge 4, top 1,024): reads 1,024 + 128, stores 262,144 + 128.',
    'HCA (merge 128): reads 8,192 + 128, stores 8,192 + 128.',
    'Linear layer: a fixed state; nothing grows.',
  ]);
  assert.deepEqual(belowFor(null)[9], ['—']);
});

test('formats: real minus, trimmed numbers, one format per quantity', () => {
  assert.equal(int(1048576), '1,048,576');
  assert.equal(int(-3), '−3');
  assert.deepEqual([-0.5, 3, 0, 0.25, -Infinity].map((v) => trimNumber(v)), ['−0.5', '3', '0', '0.25', '−∞']);
  assert.deepEqual([0.6, null, Number.NaN].map((v) => cellText(v, 'weight')), ['0.600', '', '']);
  assert.equal(formatFor('output')(-1.5), '−1.5');
  assert.throws(() => cellText(1, 'volume'), RangeError);
  assert.throws(() => sinkSplit([], 1), RangeError);
});

test('the sink split: 0.600 at logit 1, 0.069 at −2; the four window scores are the storyboard\'s', () => {
  assert.deepEqual(windowScoresFor(16, 4), [-1, -0.5, -1, -0.75]);
  assert.deepEqual(windowScoresFor(2, 4), [-1, -0.75]);
  const none = sinkSplit(windowScoresFor(16, 4), 1);
  assert.deepEqual(none.plain.map((v) => v.toFixed(3)), ['0.203', '0.334', '0.203', '0.260']);
  assert.equal(none.sink.toFixed(3), '0.600');
  assert.deepEqual(none.window.map((v) => v.toFixed(3)), ['0.081', '0.134', '0.081', '0.104']);
  assert.equal(sinkSplit(windowScoresFor(16, 4), -2).sink.toFixed(3), '0.069');
  assert.equal(sinkView(state({ pattern: 'sink' })).sumText, 'Σ = 1.000');
});

test('toy: opens on full attention; sparse, compressed and window counters equal the storyboard', () => {
  const v = (patch) => patternView(state(patch));
  assert.deepEqual([v({}).reads, v({}).cells, v({}).stored], ['16', '136', '16']);
  assert.deepEqual([v({ pattern: 'sparse' }).reads, v({ pattern: 'sparse' }).cells, v({ pattern: 'sparse' }).stored], ['4', '58', '16']);
  const c = v({ pattern: 'compressed', topK: 1 });
  assert.deepEqual([c.reads, c.stored, c.readsSub], ['5', '8', '1 entry = 4 tokens']);
  assert.equal(v({ pattern: 'window', query: 2 }).reads, '2');
  assert.equal(patternView(state({ pattern: 'linear' })), null);
});

test('toy: controls that do not apply are hidden', () => {
  assert.deepEqual(visibleControls(state({})), { window: false, topK: false, merge: false, sinkLogit: false, gate: false, query: true });
  assert.deepEqual(visibleControls(state({ pattern: 'compressed' })), { window: true, topK: true, merge: true, sinkLogit: false, gate: false, query: true });
  assert.equal(visibleControls(state({ pattern: 'linear' })).query, false);
  assert.equal(toyView(state({ pattern: 'sink' }), data).sink.values.length, 5);
});

test('toy: linear attention gives the storyboard output at gate 1 and 0.5', () => {
  assert.deepEqual(linearView(state({ pattern: 'linear' })).output, [-1.5, 6, 1.5, 3.25]);
  assert.deepEqual(linearView(state({ pattern: 'linear', gate: 0.5 })).output, [-0.75, 3, 0.75, 1.75]);
});

test('real scale: per-layer counts, and the whole-model caches equal the data\'s own derived rows', () => {
  const first = (preset, context = 1_048_576) => layerColumns(preset, data, context).columns;
  assert.deepEqual([first('glm-5.3')[0].reads, first('glm-5.3')[0].stored], ['2,048', '1,048,576']);
  assert.deepEqual(first('deepseek-v4-pro').map((c) => [c.reads, c.stored]), [['1,152', '262,272'], ['8,320', '8,320']]);
  assert.equal(first('qwen3.5')[1].reads, 'fixed state');
  assert.equal(gptOssCache(data, 131_072).total, 4_836_556_800);
  assert.equal(gptOssCache(data, 131_072).perToken, lookup('gpt-oss-120b', 'kv_bytes_per_token'));
  assert.equal(gptOssCache(data, 131_072).fixed, lookup('gpt-oss-120b', 'kv_fixed_bytes'));
  assert.equal(qwenCache(data, 1_048_576).perToken, lookup('qwen3.5-397b', 'kv_bytes_per_token'));
  assert.equal(qwenCache(data, 1_048_576).growing, 32_212_254_720);
  assert.equal(minimaxCache(data, 1).perToken, lookup('minimax-m3', 'kv_bytes_per_token'));
  const v4 = v4Estimate(data, 1_000_000);
  assert.deepEqual([v4.low, v4.high], [3_960_000_000, 11_640_000_000]);
  const [lo, hi] = lookup('deepseek-v4-pro', 'kv_bytes_per_token');
  assert.ok(v4.low / 1_000_000 >= lo * 0.98 && v4.high / 1_000_000 <= hi * 1.0, 'the estimate sits inside the data\'s reported range');
});

test('real scale view: V4-Pro is labeled an estimate; beyond-context gets a what-if note; a missing model falls back', () => {
  const v4 = realView(state({ real: 'deepseek-v4-pro' }), data);
  assert.match(v4.cache.sub, /^estimate/);
  assert.match(v4.note, /Beyond this model's 1,000,000-token context/);
  assert.equal(realView(state({ real: 'gpt-oss', context: 131_072 }), data).cache.value, '4.84 GB');
  assert.equal(realView(state({ real: 'gpt-oss', context: 131_072 }), data).note, '');
  assert.equal(realView(state({ real: 'qwen3.5' }), data).cache.value, '32.2 GB');
  assert.equal(realView(state({ real: 'glm-5.3' }), data).rows.at(-1).cells[0].value, '1,048,576');
  assert.equal(realView(state({ real: 'gpt-oss' }), null).rows.length, 0);
  assert.throws(() => layerColumns('moe', data, 8), RangeError);
  assert.deepEqual(realOptions(data).map((o) => o.label).slice(1, 3), ['gpt-oss (window 128)', 'GLM-5.3 (DSA 2,048)']);
});

test('the lesson never mutates the data', () => {
  const before = JSON.stringify(data);
  lessonFor(data).animation.belowFor(9);
  toyView(state({ real: 'deepseek-v4-pro' }), data);
  assert.equal(JSON.stringify(data), before);
});

test('the three try-this prompts and insights are the storyboard\'s, with numbers from the math and the data (X-5)', () => {
  const [one, two, three] = tryThis(data);
  assert.equal(one.prompt, 'Pattern sparse top-k, k = 4: token 16 reads 4 entries, stores 16. Switch to compressed (merge 4, top 1): reads 5, stores 8. At real scale, tap GLM-5.3 then DeepSeek-V4-Pro: DSA reads 2,048 and stores 1,048,576; CSA reads 1,152 and stores 262,272.');
  assert.equal(one.insight, 'picking what to read saves compute; only merging, windows or a fixed state save memory.');
  assert.equal(two.prompt, 'Pattern window + sink, sink logit 1: the sink takes 0.600 and the four tokens share 0.400. Set the sink logit to −2: [0.189, 0.311, 0.189, 0.242] + sink 0.069, almost the no-sink row.');
  assert.match(two.insight, /^the sink is a learned "nothing here" option;/);
  assert.equal(three.prompt, 'Pattern linear, gate 1: the output for "sat" is [−1.5, 6, 1.5, 3.25], weighted by the raw scores −1, 3, 0.5 with no softmax. Set the gate to 0.5: the state halves before each new token, output [−0.75, 3, 0.75, 1.75]. The state stays 16 numbers however long the text.');
  assert.match(three.insight, /^a linear layer trades exact lookup for a fixed-size memory/);
});

test('the linear note follows the gate (README lesson 5)', () => {
  assert.equal(linearNote(state({ gate: 1 })), 'The state has 16 numbers however long the text; gate 1 keeps every addition.');
  assert.equal(linearNote(state({ gate: 0.5 })), 'The state has 16 numbers however long the text; gate 0.5 halves the state before each new token.');
});

test('no fact row prints a reported release year (X-1): only gpt-oss, DeepSeek-V4-Pro and Qwen3.8 years remain', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.ok(!/(GLM-5\.3|MiniMax-M3|Kimi K3) \(20\d\d\)/.test(rows.join('\n')));
  assert.match(rows[4], /^DeepSeek V3\.2's DSA, as GLM-5\.3 uses it:/);
  assert.match(rows[5], /^MiniMax-M3: GQA/);
  assert.match(rows[9], /^Kimi K3: 69/);
});
