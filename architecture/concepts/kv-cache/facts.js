// kv-cache's dated text (pure, no DOM): the §8 rows and framing, the hook and intuition numbers, and the notes under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/*.json; the few figures computed from them (a cache size,
// a ratio, a GPU share) are built here from math/memory.js and ride inside the claim, so the scaffold still adds each row's source.
import { formatBytes } from '@math/core.js';
import { kvCacheBytes, sharePct } from '@math/memory.js';
import { lookupFact } from '@shared/claims.js';

const MISSING = '—';
const SLIDER_STOP = 1_048_576; // 2²⁰ tokens, the page's round-number context stop
const GPU_KEY = 'h100';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const cacheBytes = (data, id, tokens) => {
  const perToken = fact(data, id, 'kv_bytes_per_token');
  return perToken == null ? null : [perToken].flat().map((b) => kvCacheBytes({ bytesPerToken: b, tokens }));
};
const sizeText = (bytes) => (bytes == null ? MISSING : bytes.map(formatBytes).join('–'));

function h100Bytes(data) {
  const gb = lookupFact(data?.hardware, GPU_KEY, 'hbm_gb')?.value;
  return gb == null ? null : gb * 1e9;
}

// GPT-3's per-token cache ÷ the 2026 range, rounded to the nearest hundred: "400 to 1,200" (an estimate).
export function ratioRange(data) {
  const gpt3 = fact(data, 'gpt-3', 'kv_bytes_per_token');
  const range = fact(data, 'deepseek-v4-pro', 'kv_bytes_per_token');
  if (gpt3 == null || range == null) return null;
  return [gpt3 / range[1], gpt3 / range[0]].map((r) => Math.round(r / 100) * 100);
}
const ratioText = (data) => (ratioRange(data) ?? [MISSING, MISSING]).map((r) => (r === MISSING ? r : r.toLocaleString('en-US'))).join(' to ');

export const HOOK = 'Every new token has to look at every earlier token\'s keys and values. Why does a model store them instead of recomputing them, and why did that storage, not the arithmetic, become the limit on long context?';

export function intuition(data) {
  return [
    'When a model writes a reply, it produces one token per pass. To pick the token after "on", the new token "on" has to go through every block, and in each attention layer its query is compared with the keys of all five positions and blends their values. The keys and values of "The", "cat", "sat" and "down" were already computed on the previous pass, and they come out the same every time: the causal mask means a position only ever looks backwards, so nothing that comes later can change them.',
    'So the model keeps them. Each attention layer stores a K row and a V row per position, and every decode step adds one more of each and reads all of them. That stored table is the KV cache. It turns "rerun the whole conversation for every word" into "run one token, then read the past", which is why chat replies stream at all.',
    'The price is memory, and it grows with everything: one K and one V row per position, per key/value head, per layer, per conversation. '
      + 'In GPT-3 ({gpt-3.release_date|year}) that came to {gpt-3.kv_bytes_per_token|int} bytes ({gpt-3.kv_bytes_per_token|bytes}) for every single token. '
      + 'Each decode step also has to read the whole cache back from GPU memory, so a long conversation is slow as well as big. '
      + `That is why most attention redesigns since 2023 attack this one number: by 2026 the largest open models store a few kB per token, roughly ${ratioText(data)} times less than GPT-3 (an estimate; see [[model-card]], [[kv-compression]] and [[long-context-attention]]).`,
  ];
}

export const FRAMING = 'Every model in this table keeps a KV cache; what changed between 2020 and 2026 is how many bytes each token costs. The 2026 models get there with fewer key/value sets ([[kv-compression]]), a compressed latent ([[kv-compression]]), and windows, compression or fixed-size states in some layers ([[long-context-attention]]).';

// The 6 rows of storyboard §8, in order. Rows keep their placeholders (source link, "reported" chip); computed figures are literal.
export function factRows(data) {
  const gpu = h100Bytes(data);
  const llama = cacheBytes(data, 'llama-3.1-70b', fact(data, 'llama-3.1-70b', 'context_length') ?? 1);
  const llamaShare = llama && gpu ? `${sharePct(llama[0], gpu).toFixed(1)}%` : MISSING;
  const layers = fact(data, 'gpt-oss-120b', 'layers');
  const half = layers == null ? MISSING : String(layers / 2);
  return [
    { claim: `GPT-3 ({gpt-3.release_date|year}): {gpt-3.layers} layers × {gpt-3.n_heads} heads × {gpt-3.head_dim}, each head with its own K and V: {gpt-3.kv_bytes_per_token|int} B (≈ {gpt-3.kv_bytes_per_token|bytes}) per token at 2 bytes; ${sizeText(cacheBytes(data, 'gpt-3', fact(data, 'gpt-3', 'context_length') ?? 1))} for its {gpt-3.context_length|int}-token context.` },
    { claim: `Llama-3.1-70B: {llama-3.1-70b.layers} layers, {llama-3.1-70b.n_kv_heads} KV heads × {llama-3.1-70b.head_dim}: {llama-3.1-70b.kv_bytes_per_token|int} B (≈ {llama-3.1-70b.kv_bytes_per_token|bytes}) per token; ${sizeText(llama)} at its {llama-3.1-70b.context_length|int}-token context, ${llamaShare} of an {hw:h100.hbm_gb} GB H100.` },
    { claim: `DeepSeek-V3 ({deepseek-v3.release_date|year}): one {deepseek-v3.mla_kv_rank}-number latent plus a {deepseek-v3.mla_rope_dim}-number position key per layer, {deepseek-v3.layers} layers: {deepseek-v3.kv_bytes_per_token|int} B (≈ {deepseek-v3.kv_bytes_per_token|bytes}) per token, derived from its config; ${sizeText(cacheBytes(data, 'deepseek-v3', SLIDER_STOP))} at ${SLIDER_STOP.toLocaleString('en-US')} tokens.` },
    { claim: `DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): about {deepseek-v4-pro.kv_bytes_per_token|int} B per token, a formula-derived estimate (the layer mix of its compressed attention is uncertain); about ${sizeText(cacheBytes(data, 'deepseek-v4-pro', SLIDER_STOP))} for a ${SLIDER_STOP.toLocaleString('en-US')}-token conversation.` },
    { claim: `gpt-oss-120b ({gpt-oss-120b.release_date|year}): only its ${half} full-attention layers grow a cache, {gpt-oss-120b.kv_bytes_per_token|int} B (≈ {gpt-oss-120b.kv_bytes_per_token|bytes}) per token; its ${half} sliding-window layers hold a fixed ~{gpt-oss-120b.kv_fixed_bytes|bytes} per conversation.` },
    { claim: 'Reusing a cache across requests that share a prompt prefix is standard in serving ([[prefix-caching]]).', derived: true },
  ];
}

// Page text under the stage, one list per frame (0-based): dated notes filled from data and the hand-offs to other lessons.
export const BELOW = Object.freeze([
  [],
  [],
  [],
  [],
  [],
  ['Layers with a window or a fixed-size state store less: [[long-context-attention]].'],
  ['The GPT-3 paper ({gpt-3.release_date|year}) gives these sizes; the cache is stored at 2 bytes per number.'],
  ['Two conversations do not fit even before the weights, which [[prefill-decode]] adds.'],
  [
    'Per token, with the year of each model: GPT-3 ({gpt-3.release_date|year}) {gpt-3.kv_bytes_per_token|int} B; Llama-3.1-70B {llama-3.1-70b.kv_bytes_per_token|int} B; DeepSeek-V3 ({deepseek-v3.release_date|year}) {deepseek-v3.kv_bytes_per_token|int} B; DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}) {deepseek-v4-pro.kv_bytes_per_token|int} B, a reported estimate.',
    '1,048,576 (2²⁰) is a slider stop; each model\'s own context comes from its data entry.',
    'Some 2026 layers keep a fixed-size state or a window instead of one row per token: [[long-context-attention]].',
  ],
]);
