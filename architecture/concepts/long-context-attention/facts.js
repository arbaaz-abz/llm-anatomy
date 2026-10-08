// long-context-attention's dated text (pure, no DOM): the §8 rows, the framing paragraph and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; the counts on this page's tables are
// computed from those same entries with math/longctx.js and math/memory.js. Rows keep their placeholders so the scaffold
// adds each row's source link and "reported" chip. Two models' dates are left out: the data has no release date for them.
import { lookupFact } from '@shared/claims.js';
import { readsAndStores } from '@math/longctx.js';
import { formatBytes } from '@math/core.js';
import { ONE_M } from './numbers.js';
import { qwenCache } from './real-scale.js';
import { int } from './format.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const MISSING = '—';

export const FRAMING = 'Every 2026 model that reaches about 1M tokens combines staged length training ([[midtraining]]) with attention that is cheap at length; the softmax route also raises its RoPE base ([[rope]]), while Kimi K3\'s full layers use no position encoding at all. Which cheap attention is where the labs split. All of them still run on FlashAttention-style kernels, which compute exact attention tile by tile on-chip; they make any pattern fast but do not change what is read or stored.';

// The 12 rows of storyboard §8, in order.
export const ROWS = Object.freeze([
  { claim: 'gpt-oss-120b ({gpt-oss-120b.release_date|year}): alternating full and window-{gpt-oss-120b.window} layers ({gpt-oss-120b.layer_pattern}) with a learned sink per head; {gpt-oss-120b.kv_bytes_per_token|int} B per token plus about {gpt-oss-120b.kv_fixed_bytes|bytes} fixed.' },
  { claim: 'Gemma 3: {gemma-3.layer_pattern} layers, window {gemma-3.window}.' },
  { claim: 'MiMo-V2-Flash: {mimo-v2-flash.layer_pattern}, window {mimo-v2-flash.window}, with a learned sink bias.' },
  { claim: 'A learned sink logit per head: gpt-oss-120b (sink: {gpt-oss-120b.attention_sink}), MiMo-V2-Flash (sink: {mimo-v2-flash.attention_sink}) and DeepSeek-V4-Pro (sink: {deepseek-v4-pro.attention_sink}).' },
  { claim: 'DeepSeek V3.2\'s DSA, as GLM-5.3 uses it: an indexer scores every past token; attention reads the top {glm-5.3.sparse_top_k}; the cache still holds every token.' },
  { claim: 'MiniMax-M3: GQA with {minimax-m3.n_kv_heads} KV heads plus block-sparse attention, blocks of {minimax-m3.sparse_block}, top {minimax-m3.sparse_top_blocks} per group, from layer {minimax-m3.sparse_from_layer}; MiniMax claims {minimax-m3.msa_compute_claim|raw}× less attention compute at 1M on a smaller test model; the cache is still stored.' },
  { claim: 'DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): every {deepseek-v4-pro.csa_merge} tokens merged, then the top {deepseek-v4-pro.csa_top_k} read (CSA), or every {deepseek-v4-pro.hca_merge} merged (HCA), alternating, each with a {deepseek-v4-pro.window}-token window; at 1M tokens it uses {deepseek-v4-pro.v32_flops_ratio_1m|raw}× the per-token compute and {deepseek-v4-pro.v32_kv_ratio_1m|raw}× the KV cache of DeepSeek-V3.2.' },
  { claim: 'DeepSeek-V4-Pro\'s cache is about {deepseek-v4-pro.kv_bytes_per_token|bytes} per token: a formula estimate; sources disagree on the CSA : HCA layer mix (1 : 1 from its config, 3 : 1 in blog summaries), so both are shown.' },
  { claim: 'Qwen3.5 and Qwen3.8 ({qwen3.8.release_date|year}): {qwen3.8.layer_pattern}; Qwen3.5-397B keeps {qwen3.5-397b.full_attention_layers} full layers of {qwen3.5-397b.layers} with {qwen3.5-397b.n_kv_heads} KV heads × {qwen3.5-397b.head_dim}: {qwen3.5-397b.kv_bytes_per_token} B per token.' },
  { claim: 'Kimi K3: {kimi-k3.linear_attention_layers} Kimi Delta Attention layers + {kimi-k3.full_attention_layers} MLA layers; Kimi Linear (a {kimi-linear.total_params|count} / {kimi-linear.active_params|count} test model) reported up to {kimi-linear.kv_reduction_pct}% less KV cache and about {kimi-linear.decode_speedup_1m}× decode throughput at 1M (as reported; the unit is not stated).' },
  { claim: 'The split: DeepSeek, GLM and MiniMax chose sparse or compressed softmax attention; Qwen and Moonshot chose linear hybrids.', derived: true },
  { claim: 'The cost of retrofitting (GLM-5 report): {glm-5.swa_retrofit_note}.' },
]);

// Real-scale numbers the notes under the stage print, from the data's own entries.
function sparseNumbers(data) {
  const topK = fact(data, 'glm-5.3', 'sparse_top_k');
  return topK ? readsAndStores({ kind: 'sparse', n: ONE_M, topK }) : null;
}

function v4Numbers(data) {
  const keys = ['csa_merge', 'csa_top_k', 'hca_merge', 'window'];
  const [csaMerge, csaTopK, hcaMerge, window] = keys.map((k) => fact(data, 'deepseek-v4-pro', k));
  if (![csaMerge, csaTopK, hcaMerge, window].every(Number.isFinite)) return null;
  return {
    csaMerge, hcaMerge, window, csaTopK,
    csa: readsAndStores({ kind: 'compressed', n: ONE_M, merge: csaMerge, topK: csaTopK, window }),
    hca: readsAndStores({ kind: 'compressed', n: ONE_M, merge: hcaMerge, topK: Infinity, window }),
  };
}

// Page text under the stage, one list per frame (index 0 = frame 1). {…} placeholders are filled by fillText.
export function belowFor(data) {
  const full = readsAndStores({ kind: 'full', n: ONE_M });
  const sparse = sparseNumbers(data);
  const v4 = v4Numbers(data);
  const qwen = qwenCache(data, ONE_M);
  return [
    [`At 1M tokens, one query head in one layer reads ${int(full.reads)} keys for every new token, and the layer keeps ${int(full.stored)} entries.`],
    ['Real windows: {gpt-oss-120b.window} tokens in gpt-oss-120b, {mimo-v2-flash.window} in MiMo-V2-Flash and {gemma-3.window} in Gemma 3.'],
    [
      'MiMo-V2-Flash: {mimo-v2-flash.layer_pattern}, window {mimo-v2-flash.window} · gpt-oss-120b: {gpt-oss-120b.layer_pattern} over {gpt-oss-120b.layers} layers, window {gpt-oss-120b.window}.',
      'A cost of the trick, in the GLM-5 report: {glm-5.swa_retrofit_note}.',
    ],
    ['gpt-oss-120b, MiMo-V2-Flash and DeepSeek-V4-Pro each learn one sink logit per head.'],
    ['DSA reads the top {glm-5.3.sparse_top_k} of every past token; its indexer still scores every stored key, cheaply.'],
    [sparse ? `At 1M tokens a DSA layer reads ${int(sparse.reads)} entries and stores ${int(sparse.stored)}.` : MISSING],
    [v4 ? `DeepSeek-V4-Pro at 1M tokens: CSA stores ${int(v4.csa.stored - v4.window)} merged entries (every ${v4.csaMerge}) and reads ${int(v4.csaTopK)} + ${v4.window} window; HCA merges every ${v4.hcaMerge} and reads all ${int(v4.hca.reads - v4.window)}.` : MISSING],
    [],
    [
      'Qwen3.8: {qwen3.8.layer_pattern}, {qwen3.8.full_attention_layers} groups of four, {qwen3.8.layers} layers · Kimi K3: {kimi-k3.linear_attention_layers} linear + {kimi-k3.full_attention_layers} full (MLA) = {kimi-k3.layers} layers.',
      qwen ? `Qwen3.5-397B: {qwen3.5-397b.full_attention_layers} of {qwen3.5-397b.layers} layers are full, ${int(qwen.perToken)} B per token (${formatBytes(qwen.growing)} at 1M) plus a fixed state per linear layer.` : MISSING,
    ],
    tableLines(data, { sparse, v4 }),
  ];
}

// Frame 10's table, one line per attention kind: reads and stored entries per layer per token at 1,048,576 tokens.
function tableLines(data, { sparse, v4 }) {
  const msaArgs = ['sparse_top_blocks', 'sparse_block'].map((k) => fact(data, 'minimax-m3', k));
  const windowGptOss = fact(data, 'gpt-oss-120b', 'window');
  if (!sparse || !v4 || !msaArgs.every(Number.isFinite) || !Number.isFinite(windowGptOss)) return [MISSING];
  const [topBlocks, blockSize] = msaArgs;
  const window = readsAndStores({ kind: 'window', n: ONE_M, window: windowGptOss });
  const msa = readsAndStores({ kind: 'msa', n: ONE_M, topBlocks, blockSize });
  const full = readsAndStores({ kind: 'full', n: ONE_M });
  return [
    `Reads and stored entries per attention layer, per token, at ${int(ONE_M)} tokens:`,
    `Full attention: reads ${int(full.reads)}, stores ${int(full.stored)}.`,
    `Window ${windowGptOss}: reads ${int(window.reads)}, stores ${int(window.stored)}.`,
    `DSA, top ${int(sparse.reads)}: reads ${int(sparse.reads)} (plus the indexer's pass), stores ${int(sparse.stored)}.`,
    `MSA, ${topBlocks} blocks of ${blockSize}: reads about ${int(msa.reads)} (the newest block may be extra), stores ${int(msa.stored)}.`,
    `CSA (merge ${v4.csaMerge}, top ${int(v4.csaTopK)}): reads ${int(v4.csaTopK)} + ${v4.window}, stores ${int(v4.csa.stored - v4.window)} + ${v4.window}.`,
    `HCA (merge ${v4.hcaMerge}): reads ${int(v4.hca.reads - v4.window)} + ${v4.window}, stores ${int(v4.hca.stored - v4.window)} + ${v4.window}.`,
    'Linear layer: a fixed state; nothing grows.',
  ];
}
