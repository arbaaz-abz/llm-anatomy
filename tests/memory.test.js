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

// ---- Shared patch S1 (plan Task 15): kv-cache, kv-compression and long-context-attention additions ----
import { kvCacheBytes, decodeWork, kvGroups, stackKvBytes, linearStateBytes } from '../math/memory.js';

test('kvCacheBytes: kv-cache §6 worked examples to the byte', () => {
  assert.equal(kvCacheBytes({ bytesPerToken: 64, tokens: 2_048 }), 131_072);
  assert.equal(kvCacheBytes({ bytesPerToken: 4_718_592, tokens: 2_048 }), 9_663_676_416);
  assert.equal(kvCacheBytes({ bytesPerToken: 4_718_592, tokens: 2_048, sequences: 8 }), 77_309_411_328);
  assert.equal(kvCacheBytes({ bytesPerToken: 327_680, tokens: 131_072 }), 42_949_672_960);
  assert.equal(kvCacheBytes({ bytesPerToken: 327_680, tokens: 1_048_576 }), 343_597_383_680);
  assert.equal(kvCacheBytes({ bytesPerToken: 70_272, tokens: 131_072 }), 9_210_691_584);
  assert.equal(kvCacheBytes({ bytesPerToken: 70_272, tokens: 1_048_576 }), 73_685_532_672);
  assert.equal(kvCacheBytes({ bytesPerToken: 327_680, tokens: 10_000 }), 3_276_800_000);
  assert.equal(sharePct(kvCacheBytes({ bytesPerToken: 4_718_592, tokens: 2_048, sequences: 8 }), 80e9), 96.6);
});

test('kvCacheBytes: kv-compression, model-card and Serving examples', () => {
  assert.equal(kvCacheBytes({ bytesPerToken: 122_880, tokens: 131_072 }), 16_106_127_360);
  assert.equal(kvCacheBytes({ bytesPerToken: 122_880, tokens: 1_000_000 }), 122_880_000_000);
  assert.deepEqual([4_000, 12_000].map((b) => kvCacheBytes({ bytesPerToken: b, tokens: 1_000_000 })), [4e9, 12e9]);
  assert.equal(kvCacheBytes({ bytesPerToken: 327_680, tokens: 2 * 131_072 }), 2 * 42_949_672_960);
});

test('kvCacheBytes is linear in tokens and sequences', () => {
  const at = (tokens, sequences) => kvCacheBytes({ bytesPerToken: 327_680, tokens, sequences });
  assert.equal(at(4_096, 1), 2 * at(2_048, 1));
  assert.equal(at(2_048, 6), 6 * at(2_048, 1));
  assert.equal(at(2_048, 6), 3 * at(2_048, 2));
});

test('kvCacheBytes: sizes are positive finite numbers, counts positive integers', () => {
  assert.equal(kvCacheBytes({ bytesPerToken: 3_960.5, tokens: 2 }), 7_921);
  for (const bad of [0, -1, Number.NaN, Infinity, '64', undefined]) {
    assert.throws(() => kvCacheBytes({ bytesPerToken: bad, tokens: 1 }), (e) => e instanceof RangeError && /kvCacheBytes: bytesPerToken must be a positive finite number/.test(e.message));
  }
  for (const bad of [0, -1, 1.5, Number.NaN, '2', undefined]) {
    assert.throws(() => kvCacheBytes({ bytesPerToken: 64, tokens: bad }), /kvCacheBytes: tokens must be a positive integer/);
    if (bad !== undefined) assert.throws(() => kvCacheBytes({ bytesPerToken: 64, tokens: 1, sequences: bad }), /kvCacheBytes: sequences must be a positive integer/);
  }
});

test('decodeWork: kv-cache §6 worked examples', () => {
  assert.deepEqual(decodeWork({ prompt: 4, generated: 1, cache: true }), { positions: 4, keyReads: 10 });
  assert.deepEqual(decodeWork({ prompt: 4, generated: 2, cache: true }), { positions: 5, keyReads: 15 });
  assert.deepEqual(decodeWork({ prompt: 4, generated: 2, cache: false }), { positions: 9, keyReads: 25 });
  assert.deepEqual(decodeWork({ prompt: 4, generated: 4, cache: false }), { positions: 22, keyReads: 74 });
  assert.deepEqual(decodeWork({ prompt: 4, generated: 4, cache: true }), { positions: 7, keyReads: 28 });
  assert.deepEqual(decodeWork({ prompt: 1000, generated: 1000, cache: false }), { positions: 1_499_500, keyReads: 1_166_666_500 });
  assert.deepEqual(decodeWork({ prompt: 1000, generated: 1000, cache: true }), { positions: 1_999, keyReads: 1_999_000 });
});

test('decodeWork: the prefill is the same with and without a cache; the cache computes prompt + generated − 1 positions', () => {
  for (const prompt of [1, 4, 16, 1000]) {
    assert.deepEqual(decodeWork({ prompt, generated: 1, cache: true }), decodeWork({ prompt, generated: 1, cache: false }));
    for (const generated of [1, 2, 4, 128]) assert.equal(decodeWork({ prompt, generated, cache: true }).positions, prompt + generated - 1);
  }
});

test('decodeWork: errors name the argument; inputs are not mutated', () => {
  for (const bad of [0, -1, 1.5, Number.NaN, '4', undefined]) {
    assert.throws(() => decodeWork({ prompt: bad, generated: 1, cache: true }), /decodeWork: prompt must be a positive integer/);
    assert.throws(() => decodeWork({ prompt: 4, generated: bad, cache: true }), /decodeWork: generated must be a positive integer/);
  }
  for (const bad of [undefined, 1, 'yes', null]) assert.throws(() => decodeWork({ prompt: 4, generated: 1, cache: bad }), /decodeWork: cache must be true or false/);
  const args = Object.freeze({ prompt: 4, generated: 4, cache: true });
  assert.deepEqual(decodeWork(args), { positions: 7, keyReads: 28 });
});

test('kvGroups: kv-compression §6 wiring for MHA, GQA-2 and MQA', () => {
  assert.deepEqual(kvGroups({ queryHeads: 8, kvHeads: 8 }), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(kvGroups({ queryHeads: 8, kvHeads: 2 }), [0, 0, 0, 0, 1, 1, 1, 1]);
  assert.deepEqual(kvGroups({ queryHeads: 8, kvHeads: 1 }), [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.throws(() => kvGroups({ queryHeads: 8, kvHeads: 3 }), (e) => e instanceof RangeError && e.message === 'kvGroups: kvHeads must divide queryHeads');
});

test('kvGroups: one entry per query head, each KV head serves queryHeads / kvHeads of them', () => {
  for (const [queryHeads, kvHeads] of [[8, 4], [64, 8], [96, 8], [128, 1], [64, 64]]) {
    const groups = kvGroups({ queryHeads, kvHeads });
    assert.equal(groups.length, queryHeads);
    for (let g = 0; g < kvHeads; g += 1) assert.equal(groups.filter((x) => x === g).length, queryHeads / kvHeads);
  }
  for (const bad of [0, -2, 1.5, Number.NaN]) {
    assert.throws(() => kvGroups({ queryHeads: bad, kvHeads: 1 }), /kvGroups: queryHeads must be a positive integer/);
    assert.throws(() => kvGroups({ queryHeads: 8, kvHeads: bad }), /kvGroups: kvHeads must be a positive integer/);
  }
});

test('kv-compression: the frozen formulas give every toy and real worked number, and "times smaller" is queryHeads / kvHeads', () => {
  const perLayer = (kvHeads) => kvBytesPerToken({ layers: 1, kvHeads, headDim: 4, bytesPerElem: 1 });
  assert.deepEqual([8, 4, 2, 1].map(perLayer), [64, 32, 16, 8]);
  const mlaToy = kvBytesPerTokenMla({ layers: 1, dLatent: 8, dRope: 2, bytesPerElem: 1 });
  assert.equal(mlaToy, 10);
  assert.ok(mlaToy > perLayer(1) && mlaToy < perLayer(2)); // between MQA and GQA-2 (lesson 17)
  const v3 = (kvHeads) => kvBytesPerToken({ layers: 1, kvHeads, headDim: 128, bytesPerElem: 1 });
  assert.ok(576 > v3(2) && 576 < v3(4));
  assert.equal(kvBytesPerToken({ layers: 96, kvHeads: 1, headDim: 128, bytesPerElem: 2 }), 49_152);
  assert.equal(kvBytesPerToken({ layers: 60, kvHeads: 4, headDim: 128, bytesPerElem: 2 }), 122_880);
  assert.equal(kvBytesPerToken({ layers: 61, kvHeads: 128, headDim: 128, bytesPerElem: 2 }), 3_997_696);
  assert.equal((3_997_696 / 70_272).toFixed(1), '56.9');
  for (const kvHeads of [1, 2, 4, 8, 16, 32, 96]) {
    const mha = kvBytesPerToken({ layers: 96, kvHeads: 96, headDim: 128, bytesPerElem: 2 });
    if (96 % kvHeads === 0) assert.equal(mha / kvBytesPerToken({ layers: 96, kvHeads, headDim: 128, bytesPerElem: 2 }), 96 / kvHeads);
  }
});

const GPT_OSS = Object.freeze([
  Object.freeze({ layers: 18, kind: 'full', bytesPerTokenPerLayer: 2048 }),
  Object.freeze({ layers: 18, kind: 'window', window: 128, bytesPerTokenPerLayer: 2048 }),
]);

test('stackKvBytes: gpt-oss-120b (18 full + 18 window-128 layers) at 131,072 tokens', () => {
  assert.equal(2048, kvBytesPerToken({ layers: 1, kvHeads: 8, headDim: 64, bytesPerElem: 2 }));
  assert.deepEqual(stackKvBytes({ groups: GPT_OSS, tokens: 131_072 }), { perToken: 36_864, fixed: 4_718_592, total: 4_836_556_800 });
});

test('stackKvBytes: Qwen3.5-397B, an unknown linear state makes fixed and total null (ruling S1-R1)', () => {
  const groups = [{ layers: 15, kind: 'full', bytesPerTokenPerLayer: 2048 }, { layers: 45, kind: 'linear', stateBytes: null }];
  const out = stackKvBytes({ groups, tokens: 1_048_576 });
  assert.deepEqual(out, { perToken: 30_720, fixed: null, total: null });
  assert.equal(kvCacheBytes({ bytesPerToken: out.perToken, tokens: 1_048_576 }), 32_212_254_720);
  const known = stackKvBytes({ groups: [groups[0], { layers: 45, kind: 'linear', stateBytes: 32 }], tokens: 10 });
  assert.deepEqual(known, { perToken: 30_720, fixed: 45 * 32, total: 307_200 + 1_440 });
});

test('stackKvBytes: DeepSeek-V4-Pro estimate, both layer mixes, 1,000,000 tokens (brief 04 §8.2)', () => {
  const mix = (csa, hca, b) => stackKvBytes({ tokens: 1_000_000, groups: [
    { layers: csa, kind: 'compressed', merge: 4, bytesPerEntry: 512 * b },
    { layers: hca, kind: 'compressed', merge: 128, bytesPerEntry: 512 * b },
  ] });
  assert.equal(mix(30, 30, 1).total, 3_960_000_000);
  assert.equal(mix(30, 30, 2).total, 7_920_000_000);
  assert.equal(mix(45, 15, 1).total, 5_820_000_000);
  assert.equal(mix(45, 15, 2).total, 11_640_000_000);
  assert.equal(mix(30, 30, 1).perToken, 3_960);
  assert.equal(mix(45, 15, 2).perToken, 11_640);
  assert.equal(mix(30, 30, 1).fixed, 0);
});

test('stackKvBytes: totals grow linearly for full layers and stay constant for window and linear layers', () => {
  const at = (groups, tokens) => stackKvBytes({ groups, tokens }).total;
  const full = [{ layers: 2, kind: 'full', bytesPerTokenPerLayer: 64 }];
  const win = [{ layers: 2, kind: 'window', window: 4, bytesPerTokenPerLayer: 64 }];
  const lin = [{ layers: 3, kind: 'linear', stateBytes: 32 }];
  assert.equal(at(full, 200), 2 * at(full, 100));
  assert.equal(at(win, 200), at(win, 100));
  assert.equal(at(lin, 200), at(lin, 100));
  assert.equal(at(win, 2), 2 * 2 * 64); // fewer tokens than the window: only those are stored (ruling S1-R2)
});

test('stackKvBytes: bad groups throw RangeError naming the field; inputs are not mutated', () => {
  const bad = (groups, re) => assert.throws(() => stackKvBytes({ groups, tokens: 8 }), (e) => e instanceof RangeError && re.test(e.message));
  bad([], /stackKvBytes: groups must be a non-empty array/);
  bad(undefined, /stackKvBytes: groups must be a non-empty array/);
  bad([{ layers: 1, kind: 'mamba' }], /stackKvBytes: group 0 kind must be/);
  bad([{ layers: 0, kind: 'full', bytesPerTokenPerLayer: 2 }], /stackKvBytes: group 0 layers must be a positive integer/);
  bad([{ layers: 1, kind: 'full', bytesPerTokenPerLayer: 0 }], /stackKvBytes: group 0 bytesPerTokenPerLayer must be a positive finite number/);
  bad([{ layers: 1, kind: 'window', window: 1.5, bytesPerTokenPerLayer: 2 }], /stackKvBytes: group 0 window must be a positive integer/);
  bad([{ layers: 1, kind: 'compressed', merge: 0, bytesPerEntry: 2 }], /stackKvBytes: group 0 merge must be a positive integer/);
  bad([{ layers: 1, kind: 'compressed', merge: 4, bytesPerEntry: -1 }], /stackKvBytes: group 0 bytesPerEntry must be a positive finite number/);
  bad([{ layers: 1, kind: 'linear', stateBytes: 0 }], /stackKvBytes: group 0 stateBytes must be a positive finite number or null/);
  assert.throws(() => stackKvBytes({ groups: GPT_OSS, tokens: 0 }), /stackKvBytes: tokens must be a positive integer/);
  assert.equal(stackKvBytes(Object.freeze({ groups: GPT_OSS, tokens: 1 })).perToken, 36_864);
});

test('linearStateBytes: the toy state is 16 numbers, 32 bytes; it scales with every factor', () => {
  assert.equal(linearStateBytes({ layers: 1, heads: 1, dKey: 4, dValue: 4, bytesPerElem: 2 }), 32);
  assert.equal(linearStateBytes({ layers: 45, heads: 2, dKey: 4, dValue: 8, bytesPerElem: 0.5 }), 45 * 2 * 4 * 8 * 0.5);
  assert.throws(() => linearStateBytes({ layers: 0, heads: 1, dKey: 4, dValue: 4, bytesPerElem: 2 }), /linearStateBytes: layers must be a positive integer/);
  assert.throws(() => linearStateBytes({ layers: 1, heads: 1, dKey: 4, dValue: 4, bytesPerElem: 0 }), /linearStateBytes: bytesPerElem must be a positive finite number/);
  assert.equal(linearStateBytes(Object.freeze({ layers: 1, heads: 1, dKey: 4, dValue: 4, bytesPerElem: 2 })), 32);
});
