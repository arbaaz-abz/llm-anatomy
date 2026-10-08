// The toy's "at real scale" numbers (pure): per-layer reads and stored entries from math/longctx.js, whole-model caches from
// math/memory.js. Every dated figure is read from data/models.json (ctx.data); a missing key makes the preset return null.
import { lookupFact } from '@shared/claims.js';
import { readsAndStores } from '@math/longctx.js';
import { kvBytesPerToken, kvCacheBytes, stackKvBytes } from '@math/memory.js';
import { BYTES_PER_NUMBER, V4_MIXES, V4_ENTRY_BYTES } from './numbers.js';
import { int } from './format.js';

export const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;

// { key: value, … } for the keys of one entry, or null when any is missing.
function facts(data, id, keys) {
  const values = keys.map((key) => fact(data, id, key));
  return values.every((v) => v != null) ? Object.fromEntries(keys.map((key, i) => [key, values[i]])) : null;
}

const GPT_OSS_PATTERN = '1 full : 1 window';

// gpt-oss-120b: half its layers full, half window; 2 · KV heads · head size · bytes per number per layer (config, BF16).
export function gptOssCache(data, tokens) {
  const f = facts(data, 'gpt-oss-120b', ['layers', 'window', 'n_kv_heads', 'head_dim', 'layer_pattern']);
  if (!f || f.layer_pattern !== GPT_OSS_PATTERN) return null;
  const half = f.layers / 2;
  const bytesPerTokenPerLayer = kvBytesPerToken({ layers: 1, kvHeads: f.n_kv_heads, headDim: f.head_dim, bytesPerElem: BYTES_PER_NUMBER });
  const groups = [{ layers: half, kind: 'full', bytesPerTokenPerLayer }, { layers: half, kind: 'window', window: f.window, bytesPerTokenPerLayer }];
  return { ...stackKvBytes({ groups, tokens }), fullLayers: half, windowLayers: half, bytesPerTokenPerLayer, window: f.window };
}

// Qwen3.5-397B: some full layers with a growing cache; the state of the linear layers is not published (ruling S1-R1).
export function qwenCache(data, tokens) {
  const f = facts(data, 'qwen3.5-397b', ['layers', 'full_attention_layers', 'n_kv_heads', 'head_dim']);
  if (!f) return null;
  const perLayer = kvBytesPerToken({ layers: 1, kvHeads: f.n_kv_heads, headDim: f.head_dim, bytesPerElem: BYTES_PER_NUMBER });
  const groups = [{ layers: f.full_attention_layers, kind: 'full', bytesPerTokenPerLayer: perLayer }, { layers: f.layers - f.full_attention_layers, kind: 'linear', stateBytes: null }];
  const { perToken } = stackKvBytes({ groups, tokens });
  return { perToken, growing: kvCacheBytes({ bytesPerToken: perToken, tokens }), fullLayers: f.full_attention_layers, layers: f.layers };
}

// MiniMax-M3: GQA, every token's K and V stored whatever the sparse reads pick (its data row is the same product, reported).
export function minimaxCache(data, tokens) {
  const f = facts(data, 'minimax-m3', ['layers', 'n_kv_heads', 'head_dim']);
  if (!f) return null;
  const perToken = kvBytesPerToken({ layers: f.layers, kvHeads: f.n_kv_heads, headDim: f.head_dim, bytesPerElem: BYTES_PER_NUMBER });
  return { perToken, total: kvCacheBytes({ bytesPerToken: perToken, tokens }) };
}

// DeepSeek-V4-Pro: a formula estimate over both CSA : HCA layer mixes and 1 or 2 bytes per number (brief 04 §8.2).
export function v4Estimate(data, tokens) {
  const f = facts(data, 'deepseek-v4-pro', ['csa_merge', 'hca_merge', 'n_kv_heads', 'head_dim']);
  if (!f) return null;
  const entry = f.n_kv_heads * f.head_dim;
  const totals = V4_MIXES.flatMap((mix) => V4_ENTRY_BYTES.map((bytes) => stackKvBytes({
    groups: [
      { layers: mix.csaLayers, kind: 'compressed', merge: f.csa_merge, bytesPerEntry: entry * bytes },
      { layers: mix.hcaLayers, kind: 'compressed', merge: f.hca_merge, bytesPerEntry: entry * bytes },
    ],
    tokens,
  }).total));
  return { low: Math.min(...totals), high: Math.max(...totals), entry };
}

const fixedState = { reads: 'fixed state', stored: 'fixed state', readsSub: '', storedSub: '' };
const counted = (rs, subs = {}) => ({ reads: int(rs.reads), stored: int(rs.stored), readsSub: subs.reads ?? '', storedSub: subs.stored ?? '' });

function compressedColumn(label, rs, { topK, window, entries }) {
  const reads = topK === Infinity ? `all ${int(entries)} + ${window} window` : `${int(topK)} + ${window} window`;
  return { label, ...counted(rs, { reads, stored: `${int(entries)} + ${window} window` }) };
}

// The per-layer columns of one real-scale preset at `context` tokens; null when the data lacks a key.
export function layerColumns(preset, data, context) {
  const full = readsAndStores({ kind: 'full', n: context });
  if (preset === 'full') return { columns: [{ label: 'Full attention layer', ...counted(full, { reads: 'every stored key', stored: 'one per token' }) }], indexed: null };
  if (preset === 'gpt-oss') {
    const window = fact(data, 'gpt-oss-120b', 'window');
    if (window == null) return null;
    return { columns: [{ label: 'Full layers', ...counted(full) }, { label: 'Window layers', ...counted(readsAndStores({ kind: 'window', n: context, window })) }], indexed: null };
  }
  if (preset === 'glm-5.3') {
    const topK = fact(data, 'glm-5.3', 'sparse_top_k');
    if (topK == null) return null;
    const rs = readsAndStores({ kind: 'sparse', n: context, topK });
    return { columns: [{ label: 'DSA layer', ...counted(rs, { reads: 'top-k by the indexer', stored: 'every token' }) }], indexed: int(rs.indexed) };
  }
  if (preset === 'minimax-m3') {
    const f = facts(data, 'minimax-m3', ['sparse_top_blocks', 'sparse_block']);
    if (!f) return null;
    const rs = readsAndStores({ kind: 'msa', n: context, topBlocks: f.sparse_top_blocks, blockSize: f.sparse_block });
    return { columns: [{ label: 'MSA layer', ...counted(rs, { reads: 'about: the newest block may be extra', stored: 'every token' }) }], indexed: int(rs.indexed) };
  }
  if (preset === 'deepseek-v4-pro') {
    const f = facts(data, 'deepseek-v4-pro', ['csa_merge', 'csa_top_k', 'hca_merge', 'window']);
    if (!f) return null;
    const csa = readsAndStores({ kind: 'compressed', n: context, merge: f.csa_merge, topK: f.csa_top_k, window: f.window });
    const hca = readsAndStores({ kind: 'compressed', n: context, merge: f.hca_merge, topK: Infinity, window: f.window });
    return {
      columns: [
        compressedColumn('CSA layer', csa, { topK: f.csa_top_k, window: f.window, entries: Math.ceil(context / f.csa_merge) }),
        compressedColumn('HCA layer', hca, { topK: Infinity, window: f.window, entries: Math.ceil(context / f.hca_merge) }),
      ],
      indexed: null,
    };
  }
  if (preset === 'qwen3.5') return { columns: [{ label: 'Full layers', ...counted(full) }, { label: 'Linear layers', ...fixedState }], indexed: null };
  throw new RangeError(`layerColumns: unknown preset "${preset}"`);
}

// The data entry whose context length bounds a preset (null: no published limit on this page).
export const PRESET_MODEL = Object.freeze({ full: null, 'gpt-oss': 'gpt-oss-120b', 'glm-5.3': 'glm-5.3', 'minimax-m3': 'minimax-m3', 'deepseek-v4-pro': 'deepseek-v4-pro', 'qwen3.5': null });
