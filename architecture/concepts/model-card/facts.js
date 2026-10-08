// model-card's dated text (pure, no DOM): the §8 rows, the framing line, and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; the few computed figures come from
// math/card.js and math/memory.js. Rows keep their placeholders so the scaffold adds each source link and "reported" chip.
import { FIELD_GUIDE, formatByteRange } from '@math/card.js';
import { lookupFact } from '@shared/claims.js';

const DS = 'deepseek-v4-pro';
const fact = (data, id, key) => lookupFact(data?.models, id, key) ?? null;

export const FRAMING = 'Every figure on this page comes from the entries below, in the course\'s data file, each with its source link. A figure the sources do not confirm carries a "reported" chip, and where sources disagree the page shows the range, never an average.';

const kvRange = (data) => {
  const value = fact(data, DS, 'kv_bytes_per_token')?.value;
  return Array.isArray(value) ? formatByteRange(value) : '—';
};

// The 9 rows of storyboard §8, in order.
export function factRows(data) {
  return [
    { claim: `DeepSeek-V4-Pro (preview {${DS}.release_date}): {${DS}.total_params|count} / {${DS}.active_params|count}, {${DS}.layers} layers, {${DS}.experts_total} routed experts (+ {${DS}.experts_shared} shared), top-{${DS}.experts_active}; CSA (merge {${DS}.csa_merge}, top {${DS}.csa_top_k|int}) + HCA (merge {${DS}.hca_merge}) + window {${DS}.window}, with {${DS}.n_kv_heads} KV head of {${DS}.head_dim}; {${DS}.context_length|int}-token context; {${DS}.modalities} only; {${DS}.optimizer}; {${DS}.pretrain_tokens|count} tokens; {${DS}.license}.` },
    { claim: `DeepSeek-V4-Pro cache: {${DS}.kv_bytes_per_token|int} bytes per token (about ${kvRange(data)}), a formula-derived estimate with an uncertain layer mix, not a published figure. For scale, Llama-3.1-70B stores {llama-3.1-70b.kv_bytes_per_token|int} bytes per token (GQA, {llama-3.1-70b.n_kv_heads} KV heads).` },
    { claim: `Fields read from the paper and config, not the card: expert hidden size {${DS}.expert_hidden|int}; RoPE base {${DS}.rope_theta|int} ({${DS}.rope_theta_compressed|int} for the compressed streams) and YaRN factor {${DS}.yarn_factor}; context stages {${DS}.context_stages}; MTP depth {${DS}.mtp_depth}; weights: {${DS}.weight_formats}.` },
    { claim: 'Kimi K3: {kimi-k3.total_params|count} / {kimi-k3.active_params|count4}, {kimi-k3.layers} layers ({kimi-k3.linear_attention_layers} linear-attention + {kimi-k3.full_attention_layers} MLA), {kimi-k3.experts_total} routed experts + {kimi-k3.experts_shared} shared, top-{kimi-k3.experts_active}; {kimi-k3.context_length|int}-token context; {kimi-k3.modalities}.' },
    { claim: 'Qwen3.8-2.4T-A95B: {qwen3.8.total_params|count} / {qwen3.8.active_params|count}, {qwen3.8.layers} layers, {qwen3.8.experts_total} routed experts plus a shared one, top-{qwen3.8.experts_active}; {qwen3.8.context_length|int}-token native context; {qwen3.8.modalities} only.' },
    { claim: 'GLM-5.3: {glm-5.3.total_params|count} / {glm-5.3.active_params|count} (active carried over from GLM-5), {glm-5.3.layers} layers (the config and the GLM-5 paper disagree, so both ends are shown), {glm-5.3.attention}; {glm-5.3.context_length|int}-token context; {glm-5.3.modalities} only.' },
    { claim: 'MiniMax-M3: about {minimax-m3.total_params|count} / about {minimax-m3.active_params|count}, {minimax-m3.layers} layers, GQA with {minimax-m3.n_heads} query and {minimax-m3.n_kv_heads} KV heads plus sparse attention; {minimax-m3.context_length|int}-token context; {minimax-m3.modalities}. Its {minimax-m3.kv_bytes_per_token|int} bytes of cache per token are derived from its config, as the data entry says.' },
    { claim: 'Mistral Large 4 (preview {mistral-large-4.release_date}): {mistral-large-4.total_params|count} total / {mistral-large-4.active_params|count} routed-active (a higher figure when the embedding is counted); context {mistral-large-4.context_length|int} tokens (the model card claims the high end, independent evaluators reportedly measure the low end); {mistral-large-4.modalities}; open weights promised, not yet out.' },
    { claim: 'gpt-oss-120b: {gpt-oss-120b.total_params|count5} / {gpt-oss-120b.active_params|count4}, {gpt-oss-120b.layers} layers alternating full and window-{gpt-oss-120b.window} attention, {gpt-oss-120b.experts_total} experts, top-{gpt-oss-120b.experts_active}; {gpt-oss-120b.context_length|int}-token context; its cache is {gpt-oss-120b.kv_bytes_per_token|int} bytes per token plus {gpt-oss-120b.kv_fixed_bytes|int} bytes fixed, derived from the config.' },
  ];
}

// "→ lesson" hand-offs as lesson links, for a field's one-line gloss.
const lessons = (key) => [FIELD_GUIDE[key].lesson].flat().filter(Boolean).map((slug) => `[[${slug}]]`).join(', ');
export const fieldLine = (key) => `${FIELD_GUIDE[key].label}: ${FIELD_GUIDE[key].gloss}${lessons(key) ? ` (${lessons(key)})` : ''}`;

const noteOf = (data, id, key) => fact(data, id, key)?.note ?? '';

// Page text under the stage, one list per frame: the field's gloss and lesson link (README lesson 5: nothing behind
// hover), the data's own notes, and the paper / config figures the stage mentions.
export function belowFor(index, data) {
  const lines = [
    [`DeepSeek-V4-Pro, preview of {${DS}.release_date}: every chip is one field of its card in the data file. Each frame lights one field and the part of the decoder it describes.`],
    [fieldLine('total_params'), fieldLine('active_params'), 'Per-token work follows the active number; memory follows the total.'],
    [fieldLine('layers')],
    [fieldLine('experts_total'), fieldLine('experts_active'), 'From the paper / config: each routed expert is an MLP of hidden size {deepseek-v4-pro.expert_hidden|int}.'],
    [fieldLine('attention'), 'From the paper / config: CSA merges every {deepseek-v4-pro.csa_merge} tokens into one entry and reads the top {deepseek-v4-pro.csa_top_k|int} entries; HCA merges every {deepseek-v4-pro.hca_merge}; a {deepseek-v4-pro.window}-token window stays whole on every compressed layer. The tiles are a few tokens standing for many, not to scale.'],
    [fieldLine('kv_bytes_per_token'), `The data file's note: ${noteOf(data, DS, 'kv_bytes_per_token')}`],
    [fieldLine('context_length'), 'From the paper / config: reaching this length took a stretched position encoding ([[rope]]), staged training ([[midtraining]]) and cheap attention ([[long-context-attention]]).'],
    [fieldLine('modalities'), `The data file's note: ${noteOf(data, DS, 'modalities')} Kimi K3: {kimi-k3.modalities}. MiniMax-M3: {minimax-m3.modalities}.`],
    [fieldLine('optimizer'), fieldLine('pretrain_tokens'), 'From the paper / config: the multi-token-prediction depth leads to [[sampling]]; the weight formats lead to [[quantization]].'],
    [`GLM-5.3 layers: ${noteOf(data, 'glm-5.3', 'layers')}`, `Mistral Large 4 context: ${noteOf(data, 'mistral-large-4', 'context_length')}`, `DeepSeek-V4-Pro cache per token: ${noteOf(data, DS, 'kv_bytes_per_token')}`, `Mistral Large 4 active parameters: ${noteOf(data, 'mistral-large-4', 'active_params')}`],
  ];
  const text = lines[index];
  if (!text) throw new RangeError(`model-card: no frame ${index + 1}`);
  return text;
}
