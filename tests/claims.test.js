import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fillClaim, lookupFact, CLAIM_FORMATS } from '../shared/claims.js';

const fact = (value, extra = {}) => ({ value, source_url: 'https://example.org/a', confidence: 'confirmed', last_verified: '2026-10-07', ...extra });
const DATA = {
  models: { entries: [
    { id: 'gpt-3', facts: { layers: fact(96), total_params: fact(175e9), release_date: fact('2020-05-28') } },
    { id: 'qwen3.8', facts: { layers: fact(92, { source_url: 'https://example.org/q' }) } },
    { id: 'glm-5.3', facts: { layers: fact([78, 80], { note: 'sources conflict', confidence: 'reported' }) } },
  ] },
  hardware: { entries: [{ id: 'h100', facts: { hbm_gb: fact(80) } }] },
};
const text = (r) => r.segments.map((s) => s.text).join('');

test('placeholders fill from models.json with the default and named formats', () => {
  const r = fillClaim('GPT-3 ({gpt-3.release_date|year}): {gpt-3.layers} layers, {gpt-3.total_params|count} parameters.', DATA);
  assert.equal(text(r), 'GPT-3 (2020): 96 layers, 175B parameters.');
  assert.deepEqual(r.sources, ['https://example.org/a']);
  assert.equal(r.reported, false);
  assert.deepEqual(r.missing, []);
});

test('entry ids with dots, hardware refs, ranges and the reported flag', () => {
  const r = fillClaim('{qwen3.8.layers} · {hw:h100.hbm_gb} GB · {glm-5.3.layers}', DATA);
  assert.equal(text(r), '92 · 80 GB · 78–80');
  assert.deepEqual(r.sources, ['https://example.org/q', 'https://example.org/a']);
  assert.equal(r.reported, true);
});

test('a missing entry or key is reported, shown as a dash, never thrown', () => {
  const r = fillClaim('{gpt-3.d_model} and {nope.layers}', DATA);
  assert.equal(text(r), '— and —');
  assert.deepEqual(r.missing, ['gpt-3.d_model', 'nope.layers']);
});

test('an unknown format name is a programming error', () => {
  assert.throws(() => fillClaim('{gpt-3.layers|fancy}', DATA), /unknown format "fancy"/);
});

test('formats print real minus signs and grouping', () => {
  assert.equal(CLAIM_FORMATS.int(-1234), '−1,234');
  assert.equal(CLAIM_FORMATS.int(4718592), '4,718,592');
  assert.equal(CLAIM_FORMATS.bytes(70272), '70.3 kB');
  assert.equal('kib' in CLAIM_FORMATS, false);
  assert.equal(lookupFact(DATA.models, 'gpt-3', 'layers').value, 96);
  assert.equal(lookupFact(undefined, 'gpt-3', 'layers'), null);
});

test('sv: reads serving.json and paper: reads papers.json; models and hw: are unchanged', () => {
  const data = {
    ...DATA,
    serving: { entries: [{ id: 'deepseek-v3-mtp', facts: { acceptance_pct: fact([85, 90], { source_url: 'https://example.org/s' }) } }] },
    papers: { entries: [{ id: 'chinchilla-refit-2024', facts: { E: fact(1.8172, { source_url: 'https://example.org/p', confidence: 'reported' }) } }] },
  };
  const r = fillClaim('MTP accepts {sv:deepseek-v3-mtp.acceptance_pct}%; E = {paper:chinchilla-refit-2024.E|raw}; {gpt-3.layers} · {hw:h100.hbm_gb}', data);
  assert.equal(text(r), 'MTP accepts 85–90%; E = 1.8172; 96 · 80');
  assert.deepEqual(r.sources, ['https://example.org/s', 'https://example.org/p', 'https://example.org/a']);
  assert.equal(r.reported, true);
  assert.deepEqual(fillClaim('{sv:deepseek-v3-mtp.acceptance_pct}', DATA).missing, ['sv:deepseek-v3-mtp.acceptance_pct']);
});
