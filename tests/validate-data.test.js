import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { DATA_FILES, validateDataset } from '../scripts/validate-data.js';

const fact = (over = {}) => ({
  value: 1.6e12, unit: 'params', source_url: 'https://example.org/report',
  confidence: 'confirmed', last_verified: '2026-10-07', ...over,
});
const dataset = (facts) => ({ as_of: '2026-10-07', entries: [{ id: 'm', name: 'M', facts }] });

test('a well-formed dataset has no errors', () => {
  assert.deepEqual(validateDataset(dataset({ total_params: fact() }), 't'), []);
  assert.deepEqual(validateDataset(dataset({ ctx: fact({ value: [512000, 1000000], note: 'A says 1M, B measures 512K' }) }), 't'), []);
});

test('rejects missing source, bad confidence, bad date, bad value', () => {
  const errs = validateDataset(dataset({
    a: fact({ source_url: '' }),
    b: fact({ confidence: 'unverified' }),
    c: fact({ last_verified: '07/10/2026' }),
    d: fact({ value: { x: 1 } }),
    e: fact({ value: [3, 1], note: 'n' }),
  }), 't');
  assert.equal(errs.length, 5, errs.join('\n'));
});

test('a range needs a note, ids must be unique, as_of required', () => {
  assert.equal(validateDataset(dataset({ a: fact({ value: [1, 2] }) }), 't').length, 1);
  const dup = { as_of: '2026-10-07', entries: [{ id: 'x', name: 'X', facts: {} }, { id: 'x', name: 'Y', facts: {} }] };
  assert.match(validateDataset(dup, 't').join(), /duplicate id/);
  assert.match(validateDataset({ entries: [] }, 't').join(), /as_of/);
});

test('malformed input yields errors instead of throwing', () => {
  assert.ok(validateDataset(null, 't').length > 0);
  assert.ok(validateDataset('text', 't').length > 0);
  assert.ok(validateDataset([], 't').length > 0);
  assert.match(validateDataset({ as_of: '2026-10-07', entries: {} }, 't').join(), /entries/);
  assert.match(validateDataset({ as_of: '2026-10-07', entries: [null] }, 't').join(), /entry/);
  assert.match(validateDataset(dataset({ a: null }), 't').join(), /fact/);
  assert.match(validateDataset(dataset({ a: 5 }), 't').join(), /fact/);
  assert.match(validateDataset({ as_of: '2026-10-07', entries: [{ id: 'x', name: 'X', facts: [] }] }, 't').join(), /facts/);
});

test('entries need a non-empty string id and name', () => {
  const bad = { as_of: '2026-10-07', entries: [{ name: 'X', facts: {} }, { id: 'y', name: '', facts: {} }, { id: 3, name: 'Z', facts: {} }] };
  const errs = validateDataset(bad, 't');
  assert.equal(errs.filter((e) => /\bid\b/.test(e)).length, 2, errs.join('\n'));
  assert.equal(errs.filter((e) => /\bname\b/.test(e)).length, 1, errs.join('\n'));
});

test('an empty-string value is an error', () => {
  assert.equal(validateDataset(dataset({ a: fact({ value: '' }) }), 't').length, 1);
  assert.equal(validateDataset(dataset({ a: fact({ value: '   ' }) }), 't').length, 1);
});

test('dates must be real calendar dates', () => {
  for (const bad of ['2026-99-99', '2026-02-30', '2026-00-10', '2026-13-01']) {
    assert.equal(validateDataset(dataset({ a: fact({ last_verified: bad }) }), 't').length, 1, bad);
    assert.match(validateDataset({ as_of: bad, entries: [] }, 't').join(), /as_of/, bad);
  }
  assert.deepEqual(validateDataset({ as_of: '2024-02-29', entries: [] }, 't'), []);
});

test('the validator covers models, hardware, serving and papers', () => {
  assert.deepEqual(DATA_FILES, ['models', 'hardware', 'serving', 'papers']);
});

const readData = async (file) => JSON.parse(await readFile(new URL(`../data/${file}.json`, import.meta.url), 'utf8'));

test('serving.json measured anchors state their counting convention', async () => {
  const serving = await readData('serving');
  const throughputOrCost = /tok_s|cost_per_m|usd_per_m/;
  for (const entry of serving.entries) {
    for (const [key, fact] of Object.entries(entry.facts)) {
      if (!throughputOrCost.test(key)) continue;
      const text = `${fact.unit ?? ''} ${fact.note ?? ''}`;
      assert.match(text, /input|output|total|per user/i, `${entry.id}.${key} must say which tokens it counts`);
    }
  }
});

test('serving.json and papers.json hold the ids the storyboards and brief name', async () => {
  const ids = async (file) => new Set((await readData(file)).entries.map((e) => e.id));
  const serving = await ids('serving');
  for (const id of ['inferencex-v4-pro-gb300', 'inferencex-v4-pro-gb200', 'deepseek-v3-production', 'vllm', 'pagedattention']) {
    assert.ok(serving.has(id), id);
  }
  const papers = await readData('papers');
  for (const id of ['chinchilla-refit-2024', 'fp16-mismatch-2025', 'on-policy-distillation-2025']) {
    assert.ok(papers.entries.some((e) => e.id === id), id);
  }
  const refit = papers.entries.find((e) => e.id === 'chinchilla-refit-2024').facts;
  assert.deepEqual(['E', 'A', 'B', 'alpha', 'beta'].map((k) => refit[k].value), [1.8172, 482.01, 2085.43, 0.3478, 0.3658]);
});

test('the shipped data files are valid', async () => {
  for (const file of DATA_FILES) {
    const json = JSON.parse(await readFile(new URL(`../data/${file}.json`, import.meta.url), 'utf8'));
    assert.deepEqual(validateDataset(json, file), []);
  }
});
