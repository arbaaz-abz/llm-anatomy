import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatFact } from '../shared/facts.js';

const base = { source_url: 'https://example.org', confidence: 'confirmed', last_verified: '2026-10-07' };

test('formats scalars with an optional formatter and unit', () => {
  assert.deepEqual(formatFact({ ...base, value: 49e9 }, (v) => `${v / 1e9}B`), {
    text: '49B', href: 'https://example.org', reported: false, note: '',
  });
  assert.equal(formatFact({ ...base, value: 8, unit: 'KV heads' }).text, '8 KV heads');
});

test('formats ranges and flags reported facts', () => {
  const f = formatFact({ ...base, value: [512000, 1000000], confidence: 'reported', note: 'Mistral says 1M; evaluators measure ~512K' }, (v) => `${v / 1000}K`);
  assert.equal(f.text, '512K–1000K');
  assert.equal(f.reported, true);
  assert.match(f.note, /evaluators/);
});
