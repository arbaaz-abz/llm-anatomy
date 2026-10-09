// prefix-caching's dated text (pure, no DOM): the 10 §8 rows and framing, the dated text drawn on the stage (frames 1, 8, 10, 11) and
// the three price lists the toy and frame 11 share. Dated numbers are {sv:entry.key|format} placeholders filled from data/serving.json;
// rows keep their placeholders so the scaffold adds each source link and chip. A missing fact prints "—" (and the page test fails).
import { lookupFact, fillText } from '@shared/claims.js';

export const PREFILL_TITLE = 'Prefill vs decode'; // shared/concepts.json title of prefill-decode (README lesson 33); the page test pins it
export const PAGED_TITLE = 'PagedAttention';

const sv = (data, id, key) => lookupFact(data?.serving, id, key)?.value ?? null;

export const FRAMING = 'Every mechanism below is a 2026 serving feature or a published price. The prices are list prices on one day and they change; '
  + 'the sizes (a block of 16, a cache hit rate) are engine defaults and one production system, not constants.';

// The 10 rows of storyboard §8, in order.
export function factRows() {
  return [
    { claim: 'vLLM automatic prefix caching hashes each block by its parent block\'s hash, its token ids and any extra keys. It caches {sv:vllm-prefix-cache.granularity} only, and evicts unused blocks through an {sv:vllm-prefix-cache.eviction}.' },
    { claim: 'vLLM\'s default block size is {sv:vllm.default_block_size} tokens.' },
    { claim: 'SGLang\'s RadixAttention keeps cached KV in a radix tree keyed by token sequences, with LRU leaf eviction.{sv:sglang.radix_attention|cite}' },
    { claim: 'DeepSeek\'s production inference system (V3/R1, {sv:deepseek-v3-production.date|date}): {sv:deepseek-v3-production.kv_hit_rate_pct|raw}% of its {sv:deepseek-v3-production.input_tokens_per_day|count} input tokens per day hit the KV cache.' },
    { claim: 'KV-aware routing: NVIDIA Dynamo\'s router sends requests "to GPUs that already have the most relevant short-term memory from earlier steps"{sv:dynamo.kv_aware_router|cite}; llm-d\'s scheduler routes by prefix{sv:llm-d.prefix_aware_routing|cite}; vLLM engines emit KV events so routers know what each replica holds{sv:vllm-tiered-kv.kv_events|cite}.' },
    { claim: 'Mooncake (Kimi): a global KV pool with a cache-aware scheduler; {sv:mooncake.capacity_gain_pct}% more effective request capacity under its latency targets.' },
    { claim: 'vLLM tiered KV offloading ({sv:vllm-tiered-kv.date|date}): {sv:vllm-tiered-kv.tiers}. With Qwen-35B on 2 × H100, up to {sv:vllm-tiered-kv.hbm_conversations} conversations fit in HBM, {sv:vllm-tiered-kv.cpu_offload_range} need CPU offload, and throughput with storage offload was {sv:vllm-tiered-kv.storage_gain|raw} conversations.' },
    { claim: 'Anthropic: a cache read costs {sv:pricing-anthropic.cache_read_mult|raw}× the input price for most models, a cache write {sv:pricing-anthropic.cache_write_5m_mult|raw}× (5-minute entry) or {sv:pricing-anthropic.cache_write_1h_mult|raw}× (1-hour entry). Sonnet 5.5 input is ${sv:pricing-anthropic-sonnet-5.5.input_usd_per_m|raw} per million tokens; Opus 5.5 is ${sv:pricing-anthropic-opus-5.5.input_usd_per_m|raw}, with reads at {sv:pricing-anthropic-opus-5.5.cache_read_mult|raw}×.' },
    { claim: 'DeepSeek V4-Pro, off-peak: ${sv:pricing-deepseek-v4-pro.input_miss_usd_per_m|raw} per million input tokens on a miss and ${sv:pricing-deepseek-v4-pro.input_hit_usd_per_m|raw} on a hit, with no cache-write fee; peak rates are {sv:pricing-deepseek-v4-pro.peak_mult|raw}×.' },
    { claim: 'Why a write costs more and a read less is not published. This page\'s reasoning (a hit skips prefill math, and the KV must be held until reuse) is an explanation, not a vendor statement.', derived: true },
  ];
}

// Prices per million input tokens for the three providers, as the toy and frame 11 use them. `read` and `write` are multipliers of
// `base`; DeepSeek publishes a hit price, so its read multiplier is hit ÷ miss and it has no write fee.
export function providersFor(data) {
  const a = (key) => sv(data, 'pricing-anthropic', key);
  const rows = [
    { id: 'sonnet', label: 'Anthropic Sonnet 5.5', name: 'Sonnet 5.5', base: sv(data, 'pricing-anthropic-sonnet-5.5', 'input_usd_per_m'), read: a('cache_read_mult'), write: a('cache_write_5m_mult'), hasWrite: true },
    { id: 'opus', label: 'Anthropic Opus 5.5', name: 'Opus 5.5', base: sv(data, 'pricing-anthropic-opus-5.5', 'input_usd_per_m'), read: sv(data, 'pricing-anthropic-opus-5.5', 'cache_read_mult'), write: a('cache_write_5m_mult'), hasWrite: true },
    { id: 'deepseek', label: 'DeepSeek V4-Pro (off-peak)', name: 'DeepSeek V4-Pro', base: sv(data, 'pricing-deepseek-v4-pro', 'input_miss_usd_per_m'), read: null, write: 1, hasWrite: false, hitUsd: sv(data, 'pricing-deepseek-v4-pro', 'input_hit_usd_per_m') },
  ];
  const complete = rows.every((r) => Number.isFinite(r.base) && Number.isFinite(r.write) && (r.id === 'deepseek' ? Number.isFinite(r.hitUsd) : Number.isFinite(r.read)));
  return complete ? rows.map((r) => (r.id === 'deepseek' ? { ...r, read: r.hitUsd / r.base } : r)) : null;
}

export const deepseekHitRate = (data) => sv(data, 'deepseek-v3-production', 'kv_hit_rate_pct');

// The date the prices were read, from the pricing entry's own date key (printed as "prices read 2026-10-07; they change").
export const readOn = (data) => fillText('{sv:pricing-anthropic.date|date}', data);

// The hit-rate chip that sets DeepSeek's production figure: "DeepSeek Feb 2025", the month from the data's date key.
export const deepseekChip = (data) => `DeepSeek ${fillText('{sv:deepseek-v3-production.date|date}', data)}`;

// Dated text drawn on the stage (frames 1, 8, 10, 11), filled from the data; "—" where the data is missing.
export function stageText(data) {
  const fill = (text) => fillText(text, data);
  return Object.freeze({
    frame1: Object.freeze([
      `1 slot = one token's K and V, for every layer (from ${PAGED_TITLE})`,
      fill('hand-picked prompts; blocks of 4 (vLLM uses {sv:vllm.default_block_size})'),
      'tokens count from 1; blocks from 0',
    ]),
    deepseek: fill('DeepSeek production ({sv:deepseek-v3-production.date|date}): {sv:deepseek-v3-production.kv_hit_rate_pct|raw}%'),
    tiers: Object.freeze([
      fill('vLLM tiered offload ({sv:vllm-tiered-kv.date|date}), Qwen-35B on 2 × H100:'),
      fill('up to {sv:vllm-tiered-kv.hbm_conversations} conversations fit in HBM; {sv:vllm-tiered-kv.cpu_offload_range} need CPU offload;'),
      fill('with storage offload, throughput was {sv:vllm-tiered-kv.storage_gain|raw} conversations.'),
    ]),
    providers: providersFor(data),
    readOn: readOn(data),
  });
}
