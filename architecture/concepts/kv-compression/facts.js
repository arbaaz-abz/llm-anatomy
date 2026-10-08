// kv-compression's dated text (pure, no DOM): the hook, the §8 rows and framing, and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; ratios are computed from those same entries
// through the same math the stage and toy use. Rows keep their placeholders so the scaffold adds each row's source link and "reported" chip.
import { formatBytes } from '@math/core.js';
import { modelShapes } from './shapes.js';
import { ladder, mlaRatio, compare } from './ladder.js';
import { int, timesText } from './format.js';
import { DEFAULT_CONTEXT } from './numbers.js';

const DASH = '—';

// MHA ÷ MLA at DeepSeek-V3's shape: the "56.9×" of the V3 row and the "57 times" of the hook. null when the data lacks the shape.
export function v3Ratio(data) {
  const v3 = modelShapes(data).v3;
  return v3 ? mlaRatio(ladder(v3)) : null;
}

export function hook(data) {
  const ratio = v3Ratio(data);
  return `In GPT-3, every one of the {gpt-3.n_heads} heads in every layer stored its own keys and values. Do the heads really need separate copies, and how did DeepSeek-V3 store ${ratio === null ? DASH : Math.round(ratio)} times less per layer than it would with a key and value per head, without taking those separate keys and values away?`;
}

export const FRAMING = 'Almost no 2026 model uses plain MHA. Most use GQA with {minimax-m3.n_kv_heads} or {gpt-oss-120b.n_kv_heads} KV heads or MLA; DeepSeek-V4 went back to one very wide shared KV head and compresses it further ([[long-context-attention]]).';

// The nine rows of storyboard §8, in order.
export function factRows(data) {
  const ratio = v3Ratio(data);
  return [
    { claim: 'GPT-3 ({gpt-3.release_date|year}): MHA, {gpt-3.n_kv_heads} KV heads × {gpt-3.head_dim}, {gpt-3.kv_bytes_per_token|int} B per token.' },
    { claim: 'Llama-3.1-70B: GQA with {llama-3.1-70b.n_kv_heads} KV heads × {llama-3.1-70b.head_dim}, {llama-3.1-70b.kv_bytes_per_token|int} B per token.' },
    { claim: 'gpt-oss-120b ({gpt-oss-120b.release_date|year}): {gpt-oss-120b.n_heads} query heads share {gpt-oss-120b.n_kv_heads} KV heads, each {gpt-oss-120b.head_dim} wide.' },
    { claim: 'MiniMax-M3: {minimax-m3.n_heads} query heads, {minimax-m3.n_kv_heads} KV heads × {minimax-m3.head_dim}, {minimax-m3.kv_bytes_per_token|int} B per token (derived).' },
    { claim: 'Qwen3.8 ({qwen3.8.release_date|year}): its attention layers use {qwen3.8.n_heads} query heads and {qwen3.8.n_kv_heads} KV heads.' },
    { claim: `DeepSeek-V3 ({deepseek-v3.release_date|year}): MLA, latent {deepseek-v3.mla_kv_rank} + position key {deepseek-v3.mla_rope_dim} per layer: {deepseek-v3.kv_bytes_per_token|int} B per token, ${ratio === null ? DASH : timesText(ratio)} less than MHA at its {deepseek-v3.n_heads} heads.` },
    { claim: 'MLA is also used by Kimi K3 (in its {kimi-k3.full_attention_layers} full-attention layers) and GLM-5.3 (latent {glm-5.3.mla_kv_rank}, query latent {glm-5.3.mla_q_rank}).' },
    { claim: 'The cost: MLA\'s decode compute is high. GLM-5 ({glm-5.release_date|year}) reworked it: {glm-5.attention_note}.' },
    { claim: 'DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): {deepseek-v4-pro.n_kv_heads} KV head, {deepseek-v4-pro.head_dim} wide, shared by all query heads, then compressed ([[long-context-attention]]).' },
  ];
}

// "{gpt-3...}" numbers for one model's frame-9 line: bytes per token from the data, the cache at 131,072 tokens from the same math.
function modelLine(label, id, shape) {
  const c = compare(shape);
  return `${label}: {${id}.kv_bytes_per_token|int} B per token ({${id}.kv_bytes_per_token|bytes}); ${formatBytes(c.atContext)} at ${int(DEFAULT_CONTEXT)} tokens.`;
}

// Page text under the stage, one list per frame: the dated notes (filled from data) and the frame → lesson hand-offs.
export function belowFor(data) {
  const s = modelShapes(data);
  const v3 = s.v3;
  const rows = v3 ? ladder(v3) : null;
  const value = (key) => (rows ? int(rows.find((r) => r.key === key).value) : DASH);
  const all = s.gpt3 && s.llama && s.minimax && v3;
  const lines = all ? [
    modelLine('GPT-3 ({gpt-3.release_date|year}, MHA)', 'gpt-3', s.gpt3),
    modelLine('Llama-3.1-70B (GQA-{llama-3.1-70b.n_kv_heads})', 'llama-3.1-70b', s.llama),
    modelLine('MiniMax-M3 (GQA-{minimax-m3.n_kv_heads})', 'minimax-m3', s.minimax),
    modelLine('DeepSeek-V3 ({deepseek-v3.release_date|year}, MLA)', 'deepseek-v3', v3),
    `GPT-3 at ${int(DEFAULT_CONTEXT)} tokens is a what-if at its shape; its own context was {gpt-3.context_length|int}.`,
  ] : [];
  return [
    [],
    [],
    ['{llama-3.1-70b.n_kv_heads} KV heads in Llama-3.1-70B and {gpt-oss-120b.n_kv_heads} in gpt-oss-120b ({gpt-oss-120b.release_date|year}); the rest of the 2026 table is under "In today\'s models" below.'],
    ['Head A and head B are the two heads of [[attention]], with the same numbers.'],
    [],
    [`DeepSeek-V3 ({deepseek-v3.release_date|year}): {deepseek-v3.mla_kv_rank} + {deepseek-v3.mla_rope_dim} = ${v3 ? int(v3.dLatent + v3.dRope) : DASH} numbers per token per layer. Why the position key is kept apart from the latent: [[rope]].`],
    ['The folding is exact: a worked check is in the math panel below.'],
    [`At ${v3 ? v3.name : 'DeepSeek-V3'}'s shape MLA stores ${value('mla')} numbers per layer where two shared KV heads would store ${value('gqa2')}: close in memory, with a key and a value for every head.`],
    [...lines, '2026 moved on: DeepSeek-V4-Pro keeps one wide KV head and compresses it ([[long-context-attention]]).'],
  ];
}
