// Parameter counts for decoder models (decoder-anatomy §6). Pure: no DOM, inputs never mutated.
import { deepFreeze } from './core.js';

const isPositiveInt = (n) => Number.isInteger(n) && n >= 1;
const isCount = (n) => Number.isInteger(n) && n >= 0;

function requirePositive(fn, name, value) {
  if (!isPositiveInt(value)) throw new RangeError(`${fn}: ${name} must be a positive integer, got ${value}`);
}
function requireCount(fn, name, value) {
  if (!isCount(value)) throw new RangeError(`${fn}: ${name} must be an integer ≥ 0, got ${value}`);
}

function gqaParams({ nHeads, nKvHeads, dHead }, d, biases) {
  [['nHeads', nHeads], ['nKvHeads', nKvHeads], ['dHead', dHead]].forEach(([n, v]) => requirePositive('attentionParams', n, v));
  const q = nHeads * dHead;
  const kv = nKvHeads * dHead;
  return d * q + 2 * d * kv + q * d + (biases ? q + 2 * kv + d : 0);
}

// d·q_r + q_r·nH·(nope+rope) + d·(kv_r+rope) + kv_r·nH·(nope+v) + nH·v·d + the two latent norms (q_r + kv_r).
function mlaParams({ nHeads, qLoraRank: qr, kvLoraRank: kvr, qkNopeDim: nope, qkRopeDim: rope, vHeadDim: v }, d) {
  [['nHeads', nHeads], ['qLoraRank', qr], ['kvLoraRank', kvr], ['qkNopeDim', nope], ['qkRopeDim', rope], ['vHeadDim', v]]
    .forEach(([n, val]) => requirePositive('attentionParams', n, val));
  return d * qr + qr * nHeads * (nope + rope) + d * (kvr + rope) + kvr * nHeads * (nope + v) + nHeads * v * d + qr + kvr;
}

export function attentionParams(attention, dModel, { biases = false } = {}) {
  requirePositive('attentionParams', 'dModel', dModel);
  if (attention?.kind === 'gqa') return gqaParams(attention, dModel, biases);
  if (attention?.kind === 'mla') return mlaParams(attention, dModel);
  throw new RangeError(`attentionParams: attention.kind must be "gqa" or "mla", got ${attention?.kind}`);
}

const MLP_MATRICES = { swiglu: 3, gelu: 2 };

export function mlpParams({ kind, hidden }, dModel, { biases = false } = {}) {
  const matrices = MLP_MATRICES[kind];
  if (!matrices) throw new RangeError(`mlpParams: kind must be "swiglu" or "gelu", got ${kind}`);
  requirePositive('mlpParams', 'hidden', hidden);
  requirePositive('mlpParams', 'dModel', dModel);
  const biasCount = biases ? (kind === 'swiglu' ? 2 * hidden : hidden) + dModel : 0;
  return matrices * dModel * hidden + biasCount;
}

const sum = (values) => values.reduce((acc, v) => acc + v, 0);
const shares = (parts, whole) => Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, whole > 0 ? v / whole : 0]));

function checkConfig({ vocab, dModel, layers, moe }) {
  requirePositive('paramBreakdown', 'vocab', vocab);
  requirePositive('paramBreakdown', 'dModel', dModel);
  requirePositive('paramBreakdown', 'layers', layers);
  if (!moe) return;
  requireCount('paramBreakdown', 'moe.routed', moe.routed);
  requireCount('paramBreakdown', 'moe.shared', moe.shared);
  requireCount('paramBreakdown', 'moe.denseLayers', moe.denseLayers);
  requirePositive('paramBreakdown', 'moe.topK', moe.topK);
  if (moe.topK > moe.routed) throw new RangeError(`paramBreakdown: moe.topK (${moe.topK}) exceeds moe.routed (${moe.routed})`);
  if (moe.denseLayers > layers) throw new RangeError(`paramBreakdown: moe.denseLayers (${moe.denseLayers}) exceeds layers (${layers})`);
}

function layerCounts({ dModel: d, layers, biases = false, attention, mlp, moe }) {
  const moeLayers = moe ? layers - moe.denseLayers : 0;
  return {
    moeLayers,
    denseLayers: layers - moeLayers,
    attention: attentionParams(attention, d, { biases }),
    mlp: mlpParams(mlp, d, { biases }),
    expert: moe ? mlpParams({ kind: mlp.kind, hidden: moe.hidden }, d, { biases }) : 0,
  };
}

// One definition of active (README lesson 16): active = total − unused experts − (tied ? 0 : embedding table).
// The lookup table is left out (a lookup is not a multiplication) unless it is also the unembedding (tied).
export function paramBreakdown(config) {
  checkConfig(config);
  const { vocab, dModel: d, layers, norm = 'rmsnorm', positional, maxPositions = 0, tiedEmbeddings = false, moe } = config;
  const per = layerCounts(config);
  const normSize = (norm === 'layernorm' ? 2 : 1) * d;
  const parts = {
    embedding: vocab * d,
    positional: positional === 'learned' ? maxPositions * d : 0,
    attention: layers * per.attention,
    mlp: per.denseLayers * per.mlp,
    experts: moe ? per.moeLayers * (moe.routed + moe.shared) * per.expert : 0,
    router: moe ? per.moeLayers * d * moe.routed : 0,
    norms: layers * 2 * normSize + normSize,
    head: tiedEmbeddings ? 0 : vocab * d,
  };
  const activeParts = {
    ...parts,
    embedding: tiedEmbeddings ? parts.embedding : 0,
    experts: moe ? per.moeLayers * (moe.topK + moe.shared) * per.expert : 0,
  };
  const total = sum(Object.values(parts));
  const active = sum(Object.values(activeParts));
  return {
    parts,
    total,
    active,
    activeWithEmbedding: active + (tiedEmbeddings ? 0 : parts.embedding),
    perLayer: { attention: per.attention, mlp: per.mlp, expert: per.expert, norms: 2 * normSize },
    share: shares(parts, total),
    activeShare: shares(activeParts, active),
  };
}

export function partialBreakdown({ known, publishedTotal }) {
  if (!(publishedTotal > 0)) throw new RangeError(`partialBreakdown: publishedTotal must be > 0, got ${publishedTotal}`);
  const knownSum = sum(Object.values(known));
  if (knownSum > publishedTotal) throw new RangeError(`partialBreakdown: known parts (${knownSum}) exceed publishedTotal (${publishedTotal})`);
  const parts = { ...known, unknown: publishedTotal - knownSum };
  return { parts, total: publishedTotal, share: shares(parts, publishedTotal) };
}

export function publishedGap(computed, published) {
  if (!(published > 0)) throw new RangeError(`publishedGap: published must be > 0, got ${published}`);
  return (computed - published) / published;
}

// Every number maps to a data/models.json key (tests/params.test.js cross-checks them). GPT-3's tied
// embeddings are a modeling choice (GPT-2's convention), printed on the page, not a data fact.
export const PRESETS = deepFreeze({
  toy: {
    vocab: 16, dModel: 8, layers: 2, norm: 'rmsnorm', positional: 'rope', tiedEmbeddings: false, biases: false,
    attention: { kind: 'gqa', nHeads: 2, nKvHeads: 2, dHead: 4 }, mlp: { kind: 'swiglu', hidden: 16 }, moe: null,
  },
  gpt3: {
    vocab: 50257, dModel: 12288, layers: 96, norm: 'layernorm', positional: 'learned', maxPositions: 2048, tiedEmbeddings: true, biases: true,
    attention: { kind: 'gqa', nHeads: 96, nKvHeads: 96, dHead: 128 }, mlp: { kind: 'gelu', hidden: 49152 }, moe: null,
  },
  gptOss120b: {
    vocab: 201088, dModel: 2880, layers: 36, norm: 'rmsnorm', positional: 'rope', tiedEmbeddings: false, biases: true,
    attention: { kind: 'gqa', nHeads: 64, nKvHeads: 8, dHead: 64 }, mlp: { kind: 'swiglu', hidden: 2880 },
    moe: { routed: 128, shared: 0, topK: 4, hidden: 2880, denseLayers: 0 },
  },
  deepseekV3: {
    vocab: 129280, dModel: 7168, layers: 61, norm: 'rmsnorm', positional: 'rope', tiedEmbeddings: false, biases: false,
    attention: { kind: 'mla', nHeads: 128, qLoraRank: 1536, kvLoraRank: 512, qkNopeDim: 128, qkRopeDim: 64, vHeadDim: 128 },
    mlp: { kind: 'swiglu', hidden: 18432 }, moe: { routed: 256, shared: 1, topK: 8, hidden: 2048, denseLayers: 3 },
  },
});

// DeepSeek-V4-Pro from confirmed facts only (384 + 1 experts of hidden 3,072, d 7,168, 61 layers).
export const V4_PRO_PARTIAL = deepFreeze({
  known: { experts: 385 * 61 * 3 * 7168 * 3072 },
  activeKnown: { experts: 7 * 61 * 3 * 7168 * 3072 },
  publishedTotal: 1.6e12,
  publishedActive: 49e9,
  assumption: 'if all 61 layers are MoE',
});
