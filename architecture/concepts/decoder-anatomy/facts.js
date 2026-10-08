// decoder-anatomy's dated text (pure, no DOM): the §8 rows and framing, and the frame notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; ratios are computed from those same
// entries. Rows keep their placeholders so the scaffold adds each row's source link and "reported" chip.
import { paramBreakdown, PRESETS } from '@math/params.js';
import { sharePct } from '@math/memory.js';
import { formatCount } from '@math/core.js';
import { fillClaim, lookupFact } from '@shared/claims.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
// The table's Mixture-of-Experts entries: the basis for every total ÷ active ratio and the "blocks deep" range.
export const MOE_TABLE = Object.freeze(['gpt-oss-120b', 'deepseek-v4-pro', 'kimi-k3', 'qwen3.8', 'glm-5.3', 'minimax-m3', 'mistral-large-4']);

const activeShare = (data, id) => {
  const [total, active] = [fact(data, id, 'total_params'), fact(data, id, 'active_params')];
  return total && active ? sharePct(active, total) : null;
};
const pct = (data, id) => (activeShare(data, id) == null ? '—' : `${activeShare(data, id).toFixed(1)}%`);

function range(values, format = (v) => String(Math.round(v))) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? `${format(Math.min(...finite))}–${format(Math.max(...finite))}` : '—';
}

const ratios = (data) => MOE_TABLE.map((id) => fact(data, id, 'total_params') / fact(data, id, 'active_params'));
const depths = (data) => MOE_TABLE.flatMap((id) => [fact(data, id, 'layers')].flat());

export function framing(data) {
  return `Most 2026 frontier models are this picture with four knobs turned: the MLP is a Mixture of Experts (so total and active parameters differ by ${range(ratios(data))}×), `
    + `attention stores far less per position than GPT-3 did, some take images into the same stream, and the stack is ${range(depths(data))} blocks deep. `
    + 'The dense models in the table, GPT-3 and Llama 3.1 405B, are the baseline. Every ratio below is total ÷ active, counted per forward pass.';
}

const count = (n, digits) => formatCount(n, { digits });
const OURS = Object.freeze({
  gpt3: paramBreakdown(PRESETS.gpt3),
  gptOss: paramBreakdown(PRESETS.gptOss120b),
  v3: paramBreakdown(PRESETS.deepseekV3),
});

function normRow(data) {
  const shares = MOE_TABLE.map((id) => activeShare(data, id));
  return `2026 norm: about ${range(shares)}% of parameters active per token (the table's entries run ${range(shares, (v) => v.toFixed(1))}%), `
    + 'a much smaller share than in earlier Mixture-of-Experts models such as Mixtral 8x7B and Qwen3-235B.';
}

function v3Row(data) {
  const [total, mtp] = [fact(data, 'deepseek-v3', 'total_params'), fact(data, 'deepseek-v3', 'mtp_params')];
  const checkpoint = total && mtp ? formatCount(total + mtp) : '—';
  return `DeepSeek-V3 ({deepseek-v3.release_date|year}): {deepseek-v3.total_params|count} / {deepseek-v3.active_params|count}, {deepseek-v3.layers} blocks ({deepseek-v3.dense_layers} dense), `
    + '{deepseek-v3.experts_total} + {deepseek-v3.experts_shared} experts top-{deepseek-v3.experts_active}, MLA; '
    + `our count ${count(OURS.v3.total, 5)} / ${count(OURS.v3.active, 3)}; the Hugging Face checkpoint is ${checkpoint} because it adds a {deepseek-v3.mtp_params|count} multi-token-prediction module.`;
}

// The 13 rows of storyboard §8, in order.
export function factRows(data) {
  return [
    { claim: `GPT-3 {gpt-3.total_params|count} ({gpt-3.release_date|year}): {gpt-3.layers} blocks, d_model {gpt-3.d_model}, {gpt-3.n_heads} heads × {gpt-3.head_dim}, vocab {gpt-3.vocab_size}, {gpt-3.context_length} positions, dense; our count ${count(OURS.gpt3.total, 4)}.` },
    { claim: `gpt-oss-120b ({gpt-oss-120b.release_date|year}): {gpt-oss-120b.total_params|count4} total / {gpt-oss-120b.active_params|count4} active (${pct(data, 'gpt-oss-120b')}), {gpt-oss-120b.layers} blocks, {gpt-oss-120b.experts_total} experts top-{gpt-oss-120b.experts_active}; config d {gpt-oss-120b.d_model}, {gpt-oss-120b.n_heads} Q / {gpt-oss-120b.n_kv_heads} KV heads × {gpt-oss-120b.head_dim}, vocab {gpt-oss-120b.vocab_size}, expert hidden {gpt-oss-120b.expert_hidden}; our count ${count(OURS.gptOss.total, 5)} / ${count(OURS.gptOss.active, 3)}.` },
    { claim: v3Row(data) },
    { claim: `DeepSeek-V4-Pro: {deepseek-v4-pro.total_params|count} / {deepseek-v4-pro.active_params|count} (${pct(data, 'deepseek-v4-pro')}), {deepseek-v4-pro.layers} blocks, {deepseek-v4-pro.experts_total} + {deepseek-v4-pro.experts_shared} experts top-{deepseek-v4-pro.experts_active}, expert hidden {deepseek-v4-pro.expert_hidden}, d {deepseek-v4-pro.d_model}; input: {deepseek-v4-pro.modalities} only.` },
    { claim: `Kimi K3: {kimi-k3.total_params|count} / {kimi-k3.active_params|count4} (${pct(data, 'kimi-k3')}), {kimi-k3.layers} blocks, {kimi-k3.experts_total} + {kimi-k3.experts_shared} experts top-{kimi-k3.experts_active}, vocab {kimi-k3.vocab_size}, d {kimi-k3.d_model}; native multimodal with a {kimi-k3.vision_encoder_params|count}-parameter vision encoder.` },
    { claim: `Qwen3.8-2.4T-A95B: {qwen3.8.total_params|count} / {qwen3.8.active_params|count} (${pct(data, 'qwen3.8')}), {qwen3.8.layers} blocks, {qwen3.8.experts_total} routed experts plus a shared one, top-{qwen3.8.experts_active}; input: {qwen3.8.modalities} only.` },
    { claim: `GLM-5.3: {glm-5.3.total_params|count} / {glm-5.3.active_params|count} (${pct(data, 'glm-5.3')}), {glm-5.3.layers} blocks (sources conflict, shown as a range).` },
    { claim: `MiniMax-M3: about {minimax-m3.total_params|count} / about {minimax-m3.active_params|count} (${pct(data, 'minimax-m3')}), {minimax-m3.layers} blocks; {minimax-m3.modalities}.` },
    { claim: 'Mistral Large 4: {mistral-large-4.total_params|count} total / {mistral-large-4.active_params|count} routed-active, more if the embedding table is counted: labs differ on whether the embedding counts as active.' },
    { claim: 'Llama 3.1 405B: {llama-3.1-405b.total_params|count}, dense, so every block parameter is used for every token; a dense baseline beside GPT-3.' },
    { claim: normRow(data), derived: true },
    { claim: 'KV per position: GPT-3 ({gpt-3.release_date|year}, at 2 bytes per number) {gpt-3.kv_bytes_per_token|int} B ≈ {gpt-3.kv_bytes_per_token|bytes} → DeepSeek-V3 ({deepseek-v3.release_date|year}) {deepseek-v3.kv_bytes_per_token|int} B ≈ {deepseek-v3.kv_bytes_per_token|bytes}, derived from its confirmed config; 2026 designs go to a few kB: [[kv-cache]].' },
    { claim: 'Patch size {kimi-k3.patch_size} × {kimi-k3.patch_size} pixels in Kimi K3 ({minimax-m3.patch_size} in MiniMax-M3); Kimi K3\'s vision encoder is {kimi-k3.vision_encoder_layers} layers / {kimi-k3.vision_encoder_params|count} parameters.' },
  ];
}

// Page text under the stage, one list per frame: the dated notes (filled from data) and the frame → lesson hand-offs.
export const BELOW = Object.freeze([
  ['A patch is {kimi-k3.patch_size} × {kimi-k3.patch_size} pixels in Kimi K3 and MiniMax-M3 ({kimi-k3.release_date|year}); 4 × 4 here.'],
  ['Kimi K3\'s vision encoder ({kimi-k3.release_date|year}): {kimi-k3.vision_encoder_layers} layers, {kimi-k3.vision_encoder_params|count} parameters, patch {kimi-k3.patch_size}. The image road is opened in [[multimodal]].'],
  ['d_model is the row width: 8 here, {gpt-3.d_model} in GPT-3 ({gpt-3.release_date|year}), {deepseek-v4-pro.d_model} in DeepSeek-V4-Pro and Kimi K3 ({deepseek-v4-pro.release_date|year}).'],
  ['The attention box is opened in [[attention]]; how a row knows its position is [[rope]]. The norm\'s arithmetic is in the math panel below.'],
  ['What changed in each box since GPT-3, and why: [[decoder-recap]].'],
  ['DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): {deepseek-v4-pro.experts_total} routed experts, {deepseek-v4-pro.experts_active} used per token. The router and load balancing: [[moe]].'],
  [
    'N on a model card: toy 2 · GPT-3 ({gpt-3.release_date|year}) {gpt-3.layers} · gpt-oss-120b ({gpt-oss-120b.release_date|year}) {gpt-oss-120b.layers} · DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}) {deepseek-v4-pro.layers} · Kimi K3 ({kimi-k3.release_date|year}) {kimi-k3.layers}',
    'Real stacks mix a few block types: dense and MoE, full and sliding-window or linear attention ([[decoder-recap]], [[long-context-attention]]).',
  ],
  ['How a token is picked from these probabilities (temperature, top-k, top-p): [[sampling]].'],
  ['Per position, GPT-3 ({gpt-3.release_date|year}, at 2 bytes per number) stored {gpt-3.kv_bytes_per_token|int} B ≈ {gpt-3.kv_bytes_per_token|bytes}; DeepSeek-V3 ({deepseek-v3.release_date|year}) {deepseek-v3.kv_bytes_per_token|int} B ≈ {deepseek-v3.kv_bytes_per_token|bytes}, derived from its config; 2026 designs go to a few kB: [[kv-cache]]. Sliding-window and linear-attention layers store less: [[long-context-attention]].'],
]);

// Lesson text with {entry.key|format} placeholders → plain text; a missing fact prints "—" and is logged.
export function fillText(text, data) {
  const filled = fillClaim(text, data);
  if (filled.missing.length && data) console.error(`decoder-anatomy: fact missing from data/*.json: ${filled.missing.join(', ')}`);
  return filled.segments.map((s) => s.text).join('');
}
