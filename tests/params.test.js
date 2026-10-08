import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { attentionParams, mlpParams, paramBreakdown, partialBreakdown, publishedGap, PRESETS, V4_PRO_PARTIAL } from '../math/params.js';

const toyMoe = (routed) => ({ ...PRESETS.toy, moe: { routed, shared: 0, topK: 2, hidden: 8, denseLayers: 0 } });
const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);
const ALL = [PRESETS.toy, toyMoe(8), toyMoe(16), PRESETS.gpt3, PRESETS.gptOss120b, PRESETS.deepseekV3];

test('attentionParams: GQA and MLA, with and without biases', () => {
  assert.equal(attentionParams({ kind: 'gqa', nHeads: 2, nKvHeads: 2, dHead: 4 }, 8), 256);
  assert.equal(attentionParams({ kind: 'gqa', nHeads: 96, nKvHeads: 96, dHead: 128 }, 12288, { biases: true }), 604_028_928);
  assert.equal(attentionParams({ kind: 'gqa', nHeads: 64, nKvHeads: 8, dHead: 64 }, 2880, { biases: true }), 26_550_080);
  assert.equal(attentionParams(PRESETS.deepseekV3.attention, 7168), 187_107_328);
  assert.throws(() => attentionParams({ kind: 'mqa' }, 8), /kind must be "gqa" or "mla"/);
  assert.throws(() => attentionParams({ kind: 'gqa', nHeads: 0, nKvHeads: 1, dHead: 4 }, 8), /nHeads must be a positive integer/);
});

test('mlpParams: SwiGLU 3·d·h, GELU 2·d·h, plus biases', () => {
  assert.equal(mlpParams({ kind: 'swiglu', hidden: 16 }, 8), 384);
  assert.equal(mlpParams({ kind: 'swiglu', hidden: 8 }, 8), 192);
  assert.equal(mlpParams({ kind: 'gelu', hidden: 49152 }, 12288, { biases: true }), 1_208_020_992);
  assert.equal(mlpParams({ kind: 'swiglu', hidden: 2048 }, 7168), 44_040_192);
  assert.equal(mlpParams({ kind: 'swiglu', hidden: 2880 }, 2880, { biases: true }), 24_891_840);
  assert.throws(() => mlpParams({ kind: 'relu', hidden: 4 }, 8), /kind must be "swiglu" or "gelu"/);
});

test('two toy experts of hidden 8 are exactly one dense MLP of hidden 16 (lesson 17)', () => {
  assert.equal(2 * mlpParams({ kind: 'swiglu', hidden: 8 }, 8), mlpParams({ kind: 'swiglu', hidden: 16 }, 8));
});

test('toy: the animation\'s 656 per block and the "Check my work" totals', () => {
  const r = paramBreakdown(PRESETS.toy);
  assert.deepEqual(r.parts, { embedding: 128, positional: 0, attention: 512, mlp: 768, experts: 0, router: 0, norms: 40, head: 128 });
  assert.deepEqual([r.total, r.active, r.activeWithEmbedding], [1576, 1448, 1576]);
  assert.equal(r.perLayer.norms, 16);
  assert.equal(r.perLayer.attention + r.perLayer.mlp + r.perLayer.norms, 656);
  assert.equal(r.parts.norms - r.perLayer.norms * 2, 8); // the final norm
  assert.equal(r.share.mlp.toFixed(4), '0.4873');
  assert.equal(r.share.attention.toFixed(4), '0.3249');
});

test('toy try-this numbers: blocks 1, blocks 8 × d 64, experts 8 and 16', () => {
  const one = paramBreakdown({ ...PRESETS.toy, layers: 1 });
  assert.equal(one.total, 920);
  assert.equal(((one.parts.embedding + one.parts.head) / one.total).toFixed(4), '0.2783');
  const big = paramBreakdown({ ...PRESETS.toy, layers: 8, dModel: 64, attention: { kind: 'gqa', nHeads: 2, nKvHeads: 2, dHead: 32 }, mlp: { kind: 'swiglu', hidden: 128 } });
  assert.equal(big.total, 330_816);
  const e8 = paramBreakdown(toyMoe(8));
  assert.deepEqual([e8.total, e8.active, e8.parts.experts, e8.parts.router, e8.parts.mlp, e8.perLayer.expert], [4008, 1576, 3072, 128, 0, 192]);
  const e16 = paramBreakdown(toyMoe(16));
  assert.deepEqual([e16.total, e16.active], [7208, 1704]);
});

test('real presets reproduce the storyboard to the digit', () => {
  const g3 = paramBreakdown(PRESETS.gpt3);
  assert.deepEqual([g3.total, g3.active], [174_604_259_328, 174_604_259_328]); // tied: the shared matrix is multiplied
  assert.equal(g3.perLayer.norms, 4 * 12288); // two LayerNorms per block, weight + bias each
  assert.equal((g3.share.mlp * 100).toFixed(2), '66.42');
  assert.equal((g3.share.attention * 100).toFixed(2), '33.21');
  assert.equal((g3.share.embedding * 100).toFixed(2), '0.35');
  const oss = paramBreakdown(PRESETS.gptOss120b);
  assert.deepEqual([oss.total, oss.active, oss.activeWithEmbedding], [116_829_149_760, 5_132_842_560, 5_711_976_000]);
  assert.equal((oss.share.experts * 100).toFixed(2), '98.18');
  assert.equal((oss.share.attention * 100).toFixed(2), '0.82');
  assert.equal(oss.parts.head, 579_133_440);
  assert.equal((oss.activeShare.head * 100).toFixed(1), '11.3');
  const v3 = paramBreakdown(PRESETS.deepseekV3);
  assert.deepEqual([v3.total, v3.active, v3.activeWithEmbedding], [671_026_404_352, 36_625_603_584, 37_552_282_624]);
  assert.equal((v3.share.experts * 100).toFixed(2), '97.83');
  assert.equal((v3.share.attention * 100).toFixed(2), '1.70');
  assert.equal(v3.parts.mlp, 1_189_085_184);
});

test('invariants on every preset: parts sum to total, active ≤ total, shares sum to 1', () => {
  for (const config of ALL) {
    const r = paramBreakdown(config);
    assert.equal(sum(r.parts), r.total);
    assert.ok(r.active <= r.total);
    assert.ok(Math.abs(sum(r.share) - 1) < 1e-12);
    assert.ok(Math.abs(sum(r.activeShare) - 1) < 1e-12);
  }
});

test('one definition of active: untied dense = total − embedding; tied dense = total; dense uses every block parameter', () => {
  const toy = paramBreakdown(PRESETS.toy);
  assert.equal(toy.active, toy.total - toy.parts.embedding);
  const g3 = paramBreakdown(PRESETS.gpt3);
  assert.equal(g3.active, g3.total);
  for (const r of [toy, g3]) assert.equal(r.active - r.parts.head - (r === g3 ? r.parts.embedding + r.parts.positional : 0), r.parts.attention + r.parts.mlp + r.parts.norms);
});

test('partialBreakdown: known parts plus a "not published" remainder', () => {
  const v4 = partialBreakdown({ known: V4_PRO_PARTIAL.known, publishedTotal: V4_PRO_PARTIAL.publishedTotal });
  assert.equal(v4.parts.experts, 1_551_425_863_680);
  assert.equal(v4.parts.unknown, 48_574_136_320);
  assert.equal(v4.share.experts.toFixed(4), '0.9696');
  assert.equal(V4_PRO_PARTIAL.activeKnown.experts, 28_207_742_976);
  // Kimi K3 (vocab 163,840, d 7,168, data/models.json): 2.35B of 2.78T, the page's "0.08%".
  const k3 = partialBreakdown({ known: { embedding: 163_840 * 7168, head: 163_840 * 7168 }, publishedTotal: 2.78e12 });
  assert.equal(k3.parts.embedding + k3.parts.head, 2_348_810_240);
  assert.equal(((k3.parts.embedding + k3.parts.head) / k3.total).toFixed(6), '0.000845');
  assert.throws(() => partialBreakdown({ known: { experts: 2 }, publishedTotal: 1 }), /exceed publishedTotal/);
});

test('publishedGap: signed, relative to the published figure', () => {
  assert.equal(publishedGap(174_604_259_328, 175e9).toFixed(4), '-0.0023');
  assert.ok(Math.abs(publishedGap(116_829_149_760, 116.8e9) - 0.00025) < 1e-5);
  assert.equal(publishedGap(671_026_404_352, 671e9).toFixed(5), '0.00004');
  assert.throws(() => publishedGap(1, 0), /published must be > 0/);
});

test('presets are frozen and paramBreakdown never mutates its input', () => {
  assert.ok(Object.isFrozen(PRESETS.deepseekV3.moe));
  assert.throws(() => paramBreakdown({ ...PRESETS.toy, layers: 0 }), /layers must be a positive integer/);
  assert.throws(() => paramBreakdown({ ...PRESETS.toy, moe: { routed: 2, shared: 0, topK: 3, hidden: 8, denseLayers: 0 } }), /topK \(3\) exceeds moe.routed/);
});

test('presets match data/models.json (one source of truth for every config number)', async () => {
  const models = JSON.parse(await readFile(new URL('../data/models.json', import.meta.url), 'utf8'));
  const f = (id, key) => models.entries.find((e) => e.id === id)?.facts[key]?.value;
  const { gpt3, gptOss120b: oss, deepseekV3: v3 } = PRESETS;
  assert.deepEqual([gpt3.vocab, gpt3.dModel, gpt3.layers, gpt3.attention.nHeads, gpt3.attention.dHead, gpt3.maxPositions],
    [f('gpt-3', 'vocab_size'), f('gpt-3', 'd_model'), f('gpt-3', 'layers'), f('gpt-3', 'n_heads'), f('gpt-3', 'head_dim'), f('gpt-3', 'context_length')]);
  assert.deepEqual([oss.vocab, oss.dModel, oss.layers, oss.attention.nHeads, oss.attention.nKvHeads, oss.attention.dHead, oss.moe.routed, oss.moe.topK, oss.moe.hidden],
    [f('gpt-oss-120b', 'vocab_size'), f('gpt-oss-120b', 'd_model'), f('gpt-oss-120b', 'layers'), f('gpt-oss-120b', 'n_heads'), f('gpt-oss-120b', 'n_kv_heads'), f('gpt-oss-120b', 'head_dim'), f('gpt-oss-120b', 'experts_total'), f('gpt-oss-120b', 'experts_active'), f('gpt-oss-120b', 'expert_hidden')]);
  assert.deepEqual([v3.vocab, v3.dModel, v3.layers, v3.attention.nHeads, v3.attention.qLoraRank, v3.attention.kvLoraRank, v3.attention.qkNopeDim, v3.attention.qkRopeDim, v3.attention.vHeadDim, v3.mlp.hidden, v3.moe.routed, v3.moe.shared, v3.moe.topK, v3.moe.hidden, v3.moe.denseLayers],
    ['vocab_size', 'd_model', 'layers', 'n_heads', 'mla_q_rank', 'mla_kv_rank', 'mla_nope_dim', 'mla_rope_dim', 'mla_v_dim', 'mlp_hidden', 'experts_total', 'experts_shared', 'experts_active', 'expert_hidden', 'dense_layers'].map((k) => f('deepseek-v3', k)));
});
