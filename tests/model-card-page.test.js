import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { decodeCard, activeShare } from '../math/card.js';
import { LESSON, lessonFor } from '../architecture/concepts/model-card/content.js';
import { stageModel, EXIT_TITLES } from '../architecture/concepts/model-card/stage-model.js';
import { sceneAt, ENDS, FRAME_COUNT, SCENE_KEYS } from '../architecture/concepts/model-card/scene.js';
import { typedLayers } from '../architecture/concepts/model-card/frames-diagram.js';
import { belowFor, factRows } from '../architecture/concepts/model-card/facts.js';
import { mathBlocks } from '../architecture/concepts/model-card/math-blocks.js';
import { toyView, tryThis, tokensFor, CARDS, INITIAL_STATE } from '../architecture/concepts/model-card/toy-view.js';
import * as F from '../architecture/concepts/model-card/format.js';
import { CAPTIONS } from './model-card-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const entry = (id) => data.models.entries.find((e) => e.id === id);
const model = (id, key) => entry(id).facts[key].value;

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  const before = JSON.stringify(data);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
  assert.equal(JSON.stringify(data), before, 'lessonFor never mutates the data');
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('the numbers printed in the captions are the data\'s', () => {
  const v4 = (key) => model('deepseek-v4-pro', key);
  assert.match(CAPTIONS[1], new RegExp(`${v4('total_params') / 1e12} trillion parameters and uses ${v4('active_params') / 1e9} billion per token, about ${Math.round(activeShare(entry('deepseek-v4-pro')) * 100)}%`));
  assert.match(CAPTIONS[2], new RegExp(`Here there are ${v4('layers')}\\.$`));
  assert.match(CAPTIONS[3], new RegExp(`picks ${v4('experts_active')} of ${v4('experts_total')} small experts`));
  assert.match(CAPTIONS[4], new RegExp(`one shared ${v4('head_dim')}-wide key/value set`));
  assert.equal(v4('n_kv_heads'), 1);
  assert.match(CAPTIONS[5], /about 4 to 12 kB per token/);
  assert.deepEqual(v4('kv_bytes_per_token'), [4000, 12000]);
  assert.match(CAPTIONS[6], /^"1M context"/);
  assert.equal(v4('context_length'), 1_000_000);
  assert.equal(v4('modalities'), 'text');
});

test('9 facts rows, every placeholder resolves against the data files', () => {
  const rows = LESSON.facts.rows;
  assert.equal(rows.length, 9);
  lessonFor(data).facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  assert.equal(factRows(data).length, 9);
});

test('the cache row carries the "reported" chip and both sources it names', () => {
  const filled = fillClaim(lessonFor(data).facts.rows[1].claim, data);
  assert.equal(filled.reported, true);
  assert.equal(filled.sources.length, 2);
});

test('Next is empty: no lesson takes model-card as a prerequisite (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('model-card')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(dependents, []);
});

test('every prose placeholder resolves and nothing prints "—"', () => {
  const lesson = lessonFor(data);
  const below = CAPTIONS.flatMap((_, i) => lesson.animation.belowFor(i));
  const prose = [lesson.hook, ...lesson.intuition, lesson.toy.intro, lesson.facts.framing, ...lesson.facts.prose, ...lesson.takeaways, lesson.animation.standIn];
  [...prose, ...below, ...lesson.facts.rows.map((r) => r.claim)].forEach((text) => {
    assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 50));
    assert.ok(!fillText(text, data).includes('—'), text.slice(0, 50));
  });
  lesson.math.blocks.forEach((b) => assert.ok(!b.tex.includes('—'), 'no dash in a worked equation'));
});

test('the hook reads the V4-Pro spec line from the data', () => {
  assert.equal(lessonFor(data).hook.split('"')[1], '1.6T total / 49B active, 61 layers, 384 + 1 experts, top-6, compressed attention with one 512-wide KV head, 1M context, text-only.');
});

test('belowFor has a list for each frame and none for a frame that does not exist', () => {
  CAPTIONS.forEach((_, i) => assert.ok(belowFor(i, data).length > 0));
  assert.throws(() => belowFor(10, data), RangeError);
  assert.match(belowFor(5, data).join(' '), /An estimate from the config: about 4 kB per token if .* 1 : 1 \(FP8\), about 12 kB if the mix is 3 : 1 \(BF16\)/);
  assert.match(belowFor(9, data).join(' '), /Its config says 78 layers; the GLM-5 paper says 80\./);
  assert.match(belowFor(9, data).join(' '), /Mistral's card claims 1M tokens; independent evaluators reportedly measure about 512K\./);
  assert.match(belowFor(9, data).join(' '), /49B counts the routed experts only; 52B includes the embeddings\./);
});

// ---- the stage model ----

test('stage model: the ten chips, in the storyboard\'s words, from the data', () => {
  const m = stageModel(data);
  assert.deepEqual(m.chips.map((c) => c.label), ['1.6T total', '49B active', '61 layers', '384 + 1 experts, top-6', '1 KV head × 512, compressed', '1M context', 'text', 'Muon', '33T tokens', 'MIT']);
  assert.deepEqual(m.extras, { frame4: ['expert hidden 3,072'], frame7: ['RoPE base 10,000, YaRN ×16', 'stages 4K → 16K → 64K → 1M'], frame9: ['MTP depth 1', 'FP4 experts + FP8 rest'] });
  assert.equal(stageModel(null), null);
});

test('stage model: frame 2, 4, 5 and 6 numbers are the storyboard\'s', () => {
  const m = stageModel(data);
  assert.equal(m.active.formula, '49B ÷ 1.6T = 3.06%');
  assert.deepEqual({ total: m.experts.total, used: m.experts.used, others: m.experts.others, routed: m.experts.routedText, hidden: m.experts.hidden }, { total: 384, used: 6, others: 378, routed: '1.56%', hidden: '3,072' });
  assert.deepEqual(m.kv, { merge: 4, topK: '1,024', hcaMerge: 128, window: 128, heads: 1, headDim: 512 });
  assert.equal(m.cache.perTokenText, '4–12 kB');
  assert.equal(m.cache.conversationText, '4–12 GB');
  assert.equal(m.cache.scaleText, '328 GB');
  assert.equal(m.cache.gpuLabel, '80 GB GPU');
  assert.deepEqual([m.cache.lowShare, m.cache.highShare, m.cache.paperRatioText], ['5.0', '15.0', '10%']);
  assert.deepEqual(m.context, { positions: '1,000,000', ropeBase: '10,000', ropeBaseCompressed: '160,000', yarn: 16, stages: '4K → 16K → 64K → 1M' });
  assert.deepEqual(m.modality, { text: 'text-only', confidence: 'confirmed' });
});

test('stage model: the four conflicts of frame 10, with the shares', () => {
  const [glm, context, cache, active] = stageModel(data).conflicts;
  assert.deepEqual([glm.low, glm.high], ['78', '80']);
  assert.deepEqual([context.low, context.high], ['512K', '1M']);
  assert.deepEqual([cache.low, cache.high], ['4 kB', '12 kB']);
  assert.deepEqual([active.low, active.high, active.shares], ['49B', '52B', '4.67% · 4.95% of 1.05T']);
});

test('frame 9\'s exits name the lessons by title, pinned to shared/concepts.json', () => {
  const titles = Object.fromEntries(graph.concepts.map((c) => [c.slug, c.title]));
  assert.deepEqual(stageModel(data).exits.map((e) => e.title), ['Scaling laws', 'Pretraining', 'Picking the next token', 'Quantization']);
  Object.entries(EXIT_TITLES).forEach(([slug, title]) => assert.equal(title, titles[slug], slug));
});

test('toy view: every card\'s active share, gpt-oss-120b at 4.39% (5.13B of 116.83B)', () => {
  const shares = CARDS.map((c) => toyView({ ...INITIAL_STATE, left: c.id, right: 'none' }, data).columns[0].costs.activeShare);
  assert.deepEqual(shares, ['3.06%', '3.75%', '3.96%', '5.31%', '5.37%', '4.67%', '4.39%']);
});

test('the conflict source labels agree with the data notes (a reworded note fails here, not on the page)', () => {
  const note = (id, key) => entry(id).facts[key].note;
  assert.match(note('glm-5.3', 'layers'), /config\.json.*says 78 layers.*GLM-5 paper.*says 80/);
  assert.match(note('mistral-large-4', 'context_length'), /model card reportedly claims 1M.*evaluators reportedly measure about 512K/);
  assert.match(note('deepseek-v4-pro', 'kv_bytes_per_token'), /config\.json.*alternating.*about 4 KB\/token.*3:1.*about 12 KB\/token/s);
  assert.equal(entry('mistral-large-4').facts.active_params_with_embeddings.value, 52e9);
  const [glm, context, cache, active] = stageModel(data).conflicts;
  assert.deepEqual([glm.lowSource, glm.highSource], ['config.json', 'GLM-5 paper']);
  assert.deepEqual([context.lowSource, context.highSource], ['evaluators', 'Mistral card']);
  assert.deepEqual([cache.lowSource, cache.highSource], ['1 : 1 mix (config)', '3 : 1 (blog summaries)']);
  assert.deepEqual([active.lowSource, active.highSource], ['routed', 'with embeddings']);
});

test('learner text never carries the data file\'s maintainer notes: no URLs, sourcing chatter or binary-looking units', () => {
  const lesson = lessonFor(data);
  const texts = [...CAPTIONS.flatMap((_, i) => lesson.animation.belowFor(i)), ...toyView({ ...INITIAL_STATE, left: 'glm-5.3', right: 'mistral-large-4' }, data).columns.flatMap((c) => c.notes.map((n) => n.text)),
    ...toyView(INITIAL_STATE, data).columns.flatMap((c) => c.notes.map((n) => n.text))];
  texts.forEach((t) => assert.doesNotMatch(t, /https?:|primary, used here|kingy|on fetch|manual re-read|\bKB\b|\bKiB\b/, t.slice(0, 60)));
});

// ---- scenes: every frame starts where the last one ended ----

test('scenes: ten frames, every key a number 0–1, each frame starts at the previous frame\'s end', () => {
  assert.equal(FRAME_COUNT, 10);
  ENDS.forEach((end) => SCENE_KEYS.forEach((k) => assert.ok(end[k] >= 0 && end[k] <= 1, k)));
  for (let i = 1; i < FRAME_COUNT; i += 1) assert.deepEqual(sceneAt(i, 0), sceneAt(i - 1, 1), `frame ${i + 1} start`);
  assert.deepEqual(sceneAt(0, 1), { ...ENDS[0] });
  assert.equal(sceneAt(0, 0).chips, 0, 'frame 1 starts with an empty card column');
  assert.deepEqual(sceneAt(3, 0.37), sceneAt(3, 0.37), 'pure');
  assert.throws(() => sceneAt(10, 0), RangeError);
  assert.throws(() => sceneAt(-1, 0), RangeError);
});

test('scenes: the followed field is marked the same way in every frame (one chip, or the storyboard\'s four in frame 9)', () => {
  const marked = (end) => Object.entries(end).filter(([k, v]) => /^s[0-9a-b]$/.test(k) && v === 1).map(([k]) => k).sort();
  assert.deepEqual(marked(ENDS[1]), ['s0', 's1']);
  assert.deepEqual(marked(ENDS[2]), ['s2']);
  assert.deepEqual(marked(ENDS[8]), ['s7', 's8', 'sa', 'sb']);
  assert.deepEqual(marked(ENDS[9]), []);
});

test('the stack\'s count label types the model\'s layer count', () => {
  assert.deepEqual([0, 0.4, 1].map((t) => typedLayers('61', t)), ['N', '6', '61']);
});

// ---- math blocks ----

test('math blocks: three, with the worked numbers from the data', () => {
  const blocks = mathBlocks(data);
  assert.equal(blocks.length, 3);
  assert.match(blocks[0].tex, /\\htmlClass\{hl-act\}/);
  assert.match(blocks[0].tex, /\\frac\{\\text\{49B\}\}\{\\text\{1\.6T\}\} = 3\.06\\%/);
  assert.match(blocks[0].tex, /\\frac\{6\}\{384\} = 1\.56\\%/);
  assert.match(blocks[1].tex, /\\htmlClass\{hl-kv\}/);
  assert.match(blocks[1].tex, /2 \\cdot 60 \\cdot 4 \\cdot 128 \\cdot 2 \\times 10\^6 = \\text\{123 GB\}/);
  assert.match(blocks[2].tex, /\[4\{,\}000,\\ 12\{,\}000\]\\ \\text\{B\} \\times 10\^6 = \[4\.0,\\ 12\.0\]\\ \\text\{GB\}/);
  assert.deepEqual(mathBlocks(null).map((b) => b.tex.includes('—')), [true, true, true]);
});

// ---- toy view ----

const view = (patch = {}) => toyView({ ...INITIAL_STATE, ...patch }, data);
const cost = (v, side, key) => v.columns.find((c) => c.side === side).costs[key];

test('toy view: the default state is V4-Pro against Kimi K3 (try this 1)', () => {
  const v = view();
  assert.deepEqual(v.columns.map((c) => c.name), ['DeepSeek-V4-Pro', 'Kimi K3']);
  assert.deepEqual(['left', 'right'].map((s) => cost(v, s, 'activeShare')), ['3.06%', '3.75%']);
  assert.deepEqual(['left', 'right'].map((s) => cost(v, s, 'routedShare')), ['1.56%', '1.79%']);
  assert.equal(cost(v, 'left', 'cacheToken'), '4,000–12,000 B ≈ 4–12 kB');
  assert.equal(cost(v, 'left', 'cacheConversation'), '4–12 GB');
  assert.equal(cost(v, 'left', 'gpuShare'), '5.0–15.0%');
  assert.equal(cost(v, 'right', 'cacheToken'), 'not in our data');
  assert.equal(v.columns[0].rows.length, 13);
  assert.equal(v.hbmGb, 80);
});

test('toy view: the cache lines of try this 2', () => {
  assert.equal(cost(view({ right: 'minimax-m3' }), 'right', 'cacheToken'), '122,880 B ≈ 123 kB');
  assert.equal(cost(view({ right: 'minimax-m3' }), 'right', 'cacheConversation'), '129 GB');
  assert.equal(cost(view({ right: 'minimax-m3' }), 'right', 'gpuShare'), '161.1%');
  assert.equal(cost(view({ right: 'minimax-m3' }), 'right', 'gpuSub'), 'more than one GPU\'s memory');
  assert.equal(cost(view({ right: 'gpt-oss-120b' }), 'right', 'cacheToken'), '36,864 B ≈ 36.9 kB + 4.72 MB fixed');
  assert.equal(cost(view({ right: 'gpt-oss-120b' }), 'right', 'cacheConversation'), '4.84 GB');
  const short = view({ right: 'gpt-oss-120b', context: '131072' });
  assert.equal(cost(short, 'left', 'cacheConversation'), '524 MB–1.57 GB');
  assert.equal(cost(short, 'left', 'gpuShare'), '0.7–2.0%');
});

test('toy view: GLM-5.3 against Mistral Large 4 keeps ranges and the labs\' second figure (try this 3)', () => {
  const v = view({ left: 'glm-5.3', right: 'mistral-large-4' });
  const row = (side, key) => v.columns.find((c) => c.side === side).rows.find((r) => r.key === key);
  assert.equal(row('left', 'layers').value, '78–80');
  assert.equal(row('right', 'context_length').value, '512K–1M');
  assert.equal(row('right', 'context_length').detail, '', 'a range has no single exact value');
  assert.equal(row('right', 'layers').value, 'not in our data');
  assert.equal(row('left', 'active_params').tag, 'reported');
  assert.equal(row('left', 'layers').tag, '');
  assert.equal(cost(v, 'right', 'activeEmbedding'), '4.95%');
  assert.equal(v.showEmbedding, true);
  assert.equal(view().showEmbedding, false);
  const notes = v.columns.flatMap((c) => c.notes).map((n) => n.text).join(' ');
  assert.match(notes, /Its config says 78 layers; the GLM-5 paper says 80\./);
  assert.match(notes, /52B includes the embeddings/);
  assert.match(notes, /carried over from GLM-5/);
});

test('toy view: "none" is one column; the own context of a card without one is not in our data', () => {
  assert.equal(view({ right: 'none' }).columns.length, 1);
  assert.equal(tokensFor({ facts: {} }, 'own'), null);
  assert.deepEqual(tokensFor(entry('mistral-large-4'), 'own'), [512000, 1000000]);
  assert.equal(tokensFor(entry('kimi-k3'), '131072'), 131_072);
  assert.equal(view().columns[0].rows.find((r) => r.key === 'context_length').detail, '1,000,000 tokens');
});

test('toy view: every card chip has an entry; a missing entry or hardware figure throws a clear error', () => {
  CARDS.forEach((c) => assert.doesNotThrow(() => view({ left: c.id })));
  assert.throws(() => view({ left: 'nope' }), /no entry "nope"/);
  assert.throws(() => toyView(INITIAL_STATE, { ...data, hardware: { entries: [] } }), /hbm_gb/);
});

test('toy view: the active-share bars exist when both figures do, and are null for a missing one', () => {
  assert.equal(view().columns[0].bars.formula, 'active ÷ total = 3.06%');
  const bare = { ...data, models: { ...data.models, entries: data.models.entries.map((e) => (e.id === 'qwen3.8' ? { ...e, facts: { ...e.facts, active_params: undefined } } : e)) } };
  assert.equal(toyView({ ...INITIAL_STATE, right: 'qwen3.8' }, bare).columns[1].bars, null);
});

test('try this: three prompts, each with its insight, numbers from the cards', () => {
  const items = tryThis(data);
  assert.equal(items.length, 3);
  assert.match(items[0].prompt, /1\.6T \/ 49B \(3\.06%\) against 2\.78T \/ 104\.2B \(3\.75%\); routed experts per token 1\.56% against 1\.79%/);
  assert.match(items[1].prompt, /4–12 GB.*129 GB.*123 GB at exactly one million tokens.*36,864 B ≈ 36\.9 kB \+ 4\.72 MB fixed, 4\.84 GB at its 131,072 tokens/);
  assert.match(items[2].prompt, /layers 78–80.*512K–1M.*Mistral active 49B.*GLM active 40B/);
  items.forEach((i) => assert.match(i.insight, /^[a-z]/));
});

test('the toy never mutates the data', () => {
  const before = JSON.stringify(data);
  view({ left: 'glm-5.3', right: 'mistral-large-4' });
  tryThis(data); stageModel(data); mathBlocks(data);
  assert.equal(JSON.stringify(data), before);
});

// ---- format.js ----

test('format: shares, cache texts and sizes', () => {
  assert.equal(F.int(122_880), '122,880');
  assert.equal(F.int(-5), '−5');
  assert.equal(F.sharePercent(0.0306), '3.06%');
  assert.equal(F.gpuShareText(null, 80e9), 'not in our data');
  assert.equal(F.gpuShareText(4e9, 80e9), '5.0%');
  assert.equal(F.gpuShareText([4e9, 12e9], 80e9), '5.0–15.0%');
  assert.equal(F.cacheTokenText({ kind: 'not in our data' }), 'not in our data');
  assert.equal(F.cacheTokenText({ kind: 'reported', bytes: 327_680 }), '327,680 B ≈ 328 kB');
  assert.equal(F.cacheConversationText(null), 'not in our data');
  assert.equal(F.cacheConversationText(327_680_000_000), '328 GB');
  assert.equal(F.provenanceText('derived'), 'derived from the config');
  assert.equal(F.provenanceText('unknown'), '');
  assert.equal(F.exactTokens([1, 2]), '');
  assert.equal(F.exactTokens(1_048_576), '1,048,576 tokens');
  assert.deepEqual({ ...F.INITIAL_STATE }, { left: 'deepseek-v4-pro', right: 'kimi-k3', context: 'own' });
});

test('every field row in the toy has its gloss and lesson from FIELD_GUIDE, and decodeCard agrees with the table', () => {
  const rows = decodeCard(entry('deepseek-v4-pro'));
  const shown = view().columns[0].rows;
  rows.forEach((r) => assert.equal(shown.find((s) => s.key === r.key).label, r.label));
});
