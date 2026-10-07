import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateDataset } from '../scripts/validate-data.js';

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

test('the shipped data files are valid', async () => {
  for (const file of ['models', 'hardware']) {
    const json = JSON.parse(await readFile(new URL(`../data/${file}.json`, import.meta.url), 'utf8'));
    assert.deepEqual(validateDataset(json, file), []);
  }
});
