import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { FIELD_GUIDE, decodeCard, activeShare, routedShare, activeWithEmbeddings, cachePerToken, cacheForConversation, formatByteRange } from '../math/card.js';
import { deepFreeze } from '../math/core.js';
import { kvBytesPerToken, kvCacheBytes } from '../math/memory.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const models = await read('../data/models.json');
const graph = await read('../shared/concepts.json');
const entry = (id) => models.entries.find((e) => e.id === id);
const row = (id, key) => decodeCard(entry(id)).find((r) => r.key === key);
const near = (a, b, eps = 5e-5) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

const CARDS = ['deepseek-v4-pro', 'kimi-k3', 'qwen3.8', 'glm-5.3', 'minimax-m3', 'mistral-large-4', 'gpt-oss-120b'];

test('FIELD_GUIDE: 13 fields, each with a label, a one-line gloss and a lesson that exists (or null)', () => {
  const slugs = new Set(graph.concepts.map((c) => c.slug));
  const entries = Object.entries(FIELD_GUIDE);
  assert.equal(entries.length, 13);
  for (const [key, field] of entries) {
    assert.ok(field.label && field.gloss, key);
    assert.ok(field.gloss.length <= 80, `${key}: a gloss is one short line`);
    [field.lesson].flat().filter(Boolean).forEach((slug) => assert.ok(slugs.has(slug), `${key} → ${slug}`));
  }
  assert.deepEqual(FIELD_GUIDE.context_length.lesson, ['long-context-attention', 'rope']);
  assert.equal(FIELD_GUIDE.license.lesson, null);
  assert.ok(Object.isFrozen(FIELD_GUIDE) && Object.isFrozen(FIELD_GUIDE.layers));
});

test('decodeCard(DeepSeek-V4-Pro): 13 rows in FIELD_GUIDE order, with the storyboard displays', () => {
  const rows = decodeCard(entry('deepseek-v4-pro'));
  assert.deepEqual(rows.map((r) => r.key), Object.keys(FIELD_GUIDE));
  const display = Object.fromEntries(rows.map((r) => [r.key, r.display]));
  assert.equal(display.layers, '61');
  assert.equal(display.total_params, '1.6T');
  assert.equal(display.active_params, '49B');
  assert.equal(display.experts_total, '384');
  assert.equal(display.experts_active, '6');
  assert.equal(display.context_length, '1M');
  assert.equal(display.modalities, 'text');
  assert.equal(display.optimizer, 'Muon');
  assert.equal(display.pretrain_tokens, '33T');
  assert.equal(display.license, 'MIT');
  assert.equal(display.release_date, '2026-04-24');
  const kv = rows.find((r) => r.key === 'kv_bytes_per_token');
  assert.deepEqual({ display: kv.display, isRange: kv.isRange, confidence: kv.confidence, value: kv.value }, { display: '4–12 kB', isRange: true, confidence: 'reported', value: [4000, 12000] });
  assert.match(kv.sourceUrl, /^https:\/\//);
  assert.equal(rows[0].label, 'Total parameters');
  assert.equal(rows[0].lesson, 'moe');
});

test('decodeCard keeps the conflicts as ranges: GLM-5.3 layers 78–80, Mistral Large 4 context 512K–1M', () => {
  const layers = row('glm-5.3', 'layers');
  assert.deepEqual({ display: layers.display, isRange: layers.isRange, confidence: layers.confidence, value: layers.value }, { display: '78–80', isRange: true, confidence: 'confirmed', value: [78, 80] });
  assert.match(layers.note, /config\.json.*says 78 layers.*GLM-5 paper.*says 80/);
  const context = row('mistral-large-4', 'context_length');
  assert.deepEqual({ display: context.display, isRange: context.isRange, confidence: context.confidence }, { display: '512K–1M', isRange: true, confidence: 'reported' });
  assert.equal(row('mistral-large-4', 'layers'), undefined, 'Mistral Large 4 publishes no layer count: no row, not a guess');
});

test('decodeCard prints a published value at the precision the data holds', () => {
  assert.equal(row('kimi-k3', 'active_params').display, '104.2B');
  assert.equal(row('gpt-oss-120b', 'total_params').display, '116.83B');
  assert.equal(row('gpt-oss-120b', 'active_params').display, '5.13B');
  assert.equal(row('mistral-large-4', 'total_params').display, '1.05T');
  assert.equal(row('kimi-k3', 'pretrain_tokens').display, 'not disclosed');
  assert.equal(row('kimi-k3', 'context_length').display, '1.05M');
  assert.equal(row('qwen3.8', 'context_length').display, '262K');
  assert.equal(row('minimax-m3', 'kv_bytes_per_token').display, '123 kB');
});

test('decodeCard never averages or collapses a range: every array in the data stays an array with both ends printed', () => {
  for (const e of models.entries) {
    for (const r of decodeCard(e)) {
      const stored = e.facts[r.key].value;
      assert.deepEqual(r.value, stored, `${e.id}.${r.key}`);
      assert.equal(r.isRange, Array.isArray(stored), `${e.id}.${r.key}`);
      if (r.isRange) assert.ok(r.display.includes('–'), `${e.id}.${r.key}: ${r.display}`);
      assert.equal(r.confidence, e.facts[r.key].confidence);
    }
  }
});

test('decodeCard rejects something that is not an entry', () => {
  assert.throws(() => decodeCard(null), TypeError);
  assert.throws(() => decodeCard({ id: 'x' }), /entry/);
});

test('activeShare = published active ÷ published total, 2 d.p. (the one definition)', () => {
  const expected = { 'deepseek-v4-pro': 0.0306, 'kimi-k3': 0.0375, 'qwen3.8': 0.0396, 'glm-5.3': 0.0531, 'minimax-m3': 0.0537, 'mistral-large-4': 0.0467, 'gpt-oss-120b': 0.0439 };
  for (const [id, share] of Object.entries(expected)) near(activeShare(entry(id)), share);
});

test('activeShare is null for a range or a missing figure', () => {
  const base = entry('deepseek-v4-pro');
  const withRange = { ...base, facts: { ...base.facts, total_params: { ...base.facts.total_params, value: [1.5e12, 1.6e12] } } };
  const { active_params: _omitted, ...rest } = base.facts;
  assert.equal(activeShare(withRange), null);
  assert.equal(activeShare({ ...base, facts: rest }), null);
});

test('routedShare = experts per token ÷ routed experts: 1.56% and 1.79%, null when not in the data', () => {
  near(routedShare(entry('deepseek-v4-pro')), 0.0156);
  near(routedShare(entry('kimi-k3')), 0.0179);
  near(routedShare(entry('gpt-oss-120b')), 0.0313);
  near(routedShare(entry('glm-5.3')), 0.0313);
  assert.equal(routedShare(entry('llama-3.1-70b')), null);
});

test('activeWithEmbeddings reads Mistral\'s 52B from its own data key, and is null for cards without one', () => {
  assert.equal(activeWithEmbeddings(entry('mistral-large-4')), 52e9);
  near(activeWithEmbeddings(entry('mistral-large-4')) / entry('mistral-large-4').facts.total_params.value, 0.0495);
  CARDS.filter((id) => id !== 'mistral-large-4').forEach((id) => assert.equal(activeWithEmbeddings(entry(id)), null, id));
});

test('cachePerToken: reported for DeepSeek-V4-Pro (a range), derived for MiniMax-M3 and gpt-oss-120b, not in our data for the rest', () => {
  assert.deepEqual(cachePerToken(entry('deepseek-v4-pro')), { kind: 'reported', bytes: [4000, 12000] });
  assert.deepEqual(cachePerToken(entry('minimax-m3')), { kind: 'derived', bytes: 122_880 });
  assert.deepEqual(cachePerToken(entry('gpt-oss-120b')), { kind: 'derived', bytes: 36_864, fixed: 4_718_592 });
  ['kimi-k3', 'glm-5.3', 'qwen3.8', 'mistral-large-4'].forEach((id) => assert.deepEqual(cachePerToken(entry(id)), { kind: 'not in our data' }, id));
});

test('the derived values equal the data file\'s own derivations and the memory module\'s formula', () => {
  assert.equal(cachePerToken(entry('minimax-m3')).bytes, entry('minimax-m3').facts.kv_bytes_per_token.value);
  assert.equal(cachePerToken(entry('gpt-oss-120b')).bytes, entry('gpt-oss-120b').facts.kv_bytes_per_token.value);
  assert.equal(cachePerToken(entry('gpt-oss-120b')).fixed, entry('gpt-oss-120b').facts.kv_fixed_bytes.value);
  assert.equal(cachePerToken(entry('minimax-m3')).bytes, kvBytesPerToken({ layers: 60, kvHeads: 4, headDim: 128, bytesPerElem: 2 }));
});

test('a plain GQA stack with no window mix derives like kvBytesPerToken; an unknown layer mix is not guessed', () => {
  const base = entry('minimax-m3');
  const withPattern = (layer_pattern) => ({ ...base, facts: { ...base.facts, layer_pattern: { value: layer_pattern } } });
  assert.equal(cachePerToken({ ...base, facts: { ...base.facts, layers: { value: 40 } } }).bytes, 2 * 40 * 4 * 128 * 2);
  assert.equal(cachePerToken(withPattern('3 linear : 1 full')).kind, 'reported', 'falls back to the data\'s own figure');
  const odd = { ...entry('gpt-oss-120b'), facts: { ...entry('gpt-oss-120b').facts, layers: { value: 35 } } };
  assert.notEqual(cachePerToken(odd).kind, 'derived', '35 layers cannot split 1 full : 1 window');
});

test('a card with no cache figure and no derivable stack is "not in our data", even with a reported figure missing', () => {
  const { kv_bytes_per_token: _dropped, ...facts } = entry('minimax-m3').facts;
  const bare = { ...entry('minimax-m3'), facts: { ...facts, attention: { value: 'MLA' } } };
  assert.deepEqual(cachePerToken(bare), { kind: 'not in our data' });
});

test('cacheForConversation: worked lines from the storyboard (§6)', () => {
  assert.equal(cacheForConversation(entry('minimax-m3'), 1_000_000), 122_880_000_000);
  assert.equal(cacheForConversation(entry('minimax-m3'), 1_048_576), 122_880 * 1_048_576);
  assert.equal(cacheForConversation(entry('gpt-oss-120b'), 131_072), 36_864 * 131_072 + 4_718_592);
  assert.deepEqual(cacheForConversation(entry('deepseek-v4-pro'), 1_000_000), [4_000_000_000, 12_000_000_000]);
  assert.equal(cacheForConversation(entry('llama-3.1-70b'), 1_000_000), 327_680_000_000);
  assert.equal(cacheForConversation(entry('kimi-k3'), 1_000_000), null);
});

test('cacheForConversation: a range of tokens gives a range of bytes, and agrees with kvCacheBytes', () => {
  assert.deepEqual(cacheForConversation(entry('minimax-m3'), [512_000, 1_000_000]), [122_880 * 512_000, 122_880 * 1_000_000]);
  assert.equal(cacheForConversation(entry('minimax-m3'), 131_072), kvCacheBytes({ bytesPerToken: 122_880, tokens: 131_072 }));
  assert.throws(() => cacheForConversation(entry('minimax-m3'), 0), RangeError);
  assert.throws(() => cacheForConversation(entry('minimax-m3'), [10, 5]), RangeError);
});

test('window layers store at most `window` entries: a short conversation pays less than the fixed bytes', () => {
  assert.equal(cacheForConversation(entry('gpt-oss-120b'), 64), 36_864 * 64 + 18 * 64 * 2048);
});

test('formatByteRange: a shared unit is printed once; different units keep both', () => {
  assert.equal(formatByteRange([4000, 12000]), '4–12 kB');
  assert.equal(formatByteRange([4e9, 12e9]), '4–12 GB');
  assert.equal(formatByteRange([524_288_000, 1_572_864_000]), '524 MB–1.57 GB');
});

test('inputs are never mutated (the data is frozen)', () => {
  const frozen = deepFreeze(models);
  const e = frozen.entries.find((x) => x.id === 'gpt-oss-120b');
  const before = JSON.stringify(frozen);
  decodeCard(e); activeShare(e); routedShare(e); cachePerToken(e); cacheForConversation(e, 131_072); activeWithEmbeddings(e);
  assert.equal(JSON.stringify(frozen), before);
});
