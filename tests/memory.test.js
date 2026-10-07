import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { kvBytesPerToken, kvBytesPerTokenMla, sharePct } from '../math/memory.js';

test('kvBytesPerToken: toy, GPT-3 (MHA), Llama-3.1-70B (GQA-8), GPT-3 shape with 8 KV heads', () => {
  assert.equal(kvBytesPerToken({ layers: 2, kvHeads: 2, headDim: 4, bytesPerElem: 2 }), 64);
  assert.equal(kvBytesPerToken({ layers: 96, kvHeads: 96, headDim: 128, bytesPerElem: 2 }), 4_718_592);
  assert.equal(kvBytesPerToken({ layers: 80, kvHeads: 8, headDim: 128, bytesPerElem: 2 }), 327_680);
  assert.equal(kvBytesPerToken({ layers: 96, kvHeads: 8, headDim: 128, bytesPerElem: 2 }), 393_216);
});

test('MHA, GQA and MQA are one formula: kvHeads = heads, groups, 1', () => {
  const at = (kvHeads) => kvBytesPerToken({ layers: 96, kvHeads, headDim: 128, bytesPerElem: 2 });
  assert.equal(at(96), 2 * 96 * 96 * 128 * 2);
  assert.equal(at(1), 49_152);
  assert.equal(at(96) / at(1), 96);
});

test('kvBytesPerTokenMla: DeepSeek-V3 = 61 × 576 × 2 = 70,272 B, with no factor of 2 for K and V', () => {
  assert.equal(kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 }), 70_272);
  assert.equal(kvBytesPerTokenMla({ layers: 1, dLatent: 8, dRope: 2, bytesPerElem: 1 }), 10);
  assert.equal(kvBytesPerToken({ layers: 61, kvHeads: 128, headDim: 128, bytesPerElem: 2 }) / 70_272, 3_997_696 / 70_272);
});

test('fractional bytes per element are allowed (FP4 with scales), counts are not', () => {
  assert.equal(kvBytesPerToken({ layers: 1, kvHeads: 1, headDim: 8, bytesPerElem: 0.5625 }), 9);
  for (const bad of [0, -1, 1.5, Number.NaN, '2', undefined]) {
    assert.throws(() => kvBytesPerToken({ layers: bad, kvHeads: 1, headDim: 1, bytesPerElem: 2 }), /kvBytesPerToken: layers must be a positive integer/);
    assert.throws(() => kvBytesPerTokenMla({ layers: 1, dLatent: bad, dRope: 1, bytesPerElem: 2 }), /kvBytesPerTokenMla: dLatent must be a positive integer/);
  }
  assert.throws(() => kvBytesPerToken({ layers: 1, kvHeads: 1, headDim: 1, bytesPerElem: 0 }), /bytesPerElem must be a positive finite number/);
});

test('inputs are not mutated', () => {
  const args = Object.freeze({ layers: 2, kvHeads: 2, headDim: 4, bytesPerElem: 2 });
  assert.equal(kvBytesPerToken(args), 64);
});

test('sharePct: kv-cache §6 worked examples, one decimal by default', () => {
  assert.equal(sharePct(42_949_672_960, 80e9), 53.7);
  assert.equal(sharePct(85_899_345_920, 80e9), 107.4);
  assert.equal(sharePct(77_309_411_328, 80e9), 96.6);
  assert.equal(sharePct(1_296, 1_048_576), 0.1);
  assert.equal(sharePct(1_843_200, 1_048_576), 175.8);
  assert.equal(sharePct(5, 5), 100);
  assert.equal(sharePct(2 * 1_296, 1_048_576), 0.2);
});

test('sharePct: more decimals on request (share bars below 1 %), and its errors', () => {
  assert.equal(sharePct(0.00818, 1, { decimals: 2 }), 0.82);
  assert.equal(sharePct(768, 1576), 48.7);
  for (const whole of [0, -1, Number.NaN, Infinity]) assert.throws(() => sharePct(1, whole), /sharePct: whole must be a finite number > 0/);
  assert.throws(() => sharePct(Number.NaN, 1), /sharePct: part must be a finite number/);
  assert.throws(() => sharePct(1, 2, { decimals: 1.5 }), /decimals must be an integer 0–6/);
});

test('the data file\'s derived KV bytes come from these formulas', async () => {
  const models = JSON.parse(await readFile(new URL('../data/models.json', import.meta.url), 'utf8'));
  const f = (id, key) => models.entries.find((e) => e.id === id).facts[key].value;
  assert.equal(f('deepseek-v3', 'kv_bytes_per_token'), kvBytesPerTokenMla({ layers: f('deepseek-v3', 'layers'), dLatent: f('deepseek-v3', 'mla_kv_rank'), dRope: f('deepseek-v3', 'mla_rope_dim'), bytesPerElem: 2 }));
  assert.equal(f('gpt-3', 'kv_bytes_per_token'), kvBytesPerToken({ layers: f('gpt-3', 'layers'), kvHeads: f('gpt-3', 'n_kv_heads'), headDim: f('gpt-3', 'head_dim'), bytesPerElem: 2 }));
});
