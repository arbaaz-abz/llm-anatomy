// paged-attention's dated and derived text (pure, no DOM): the hook, the intuition, the §8 rows and the notes under the
// stage. Dated numbers are {entry.key|format} placeholders filled from data/*.json; the few figures computed from them
// (a block's weight, a share of a GPU) are built here from math/ and ride inside the claim, so each row keeps its source.
import { formatBytes } from '@math/core.js';
import { kvCacheBytes } from '@math/memory.js';
import { kvBytesPerBlock, reserveMaxBytes } from '@math/paging.js';
import { hbmFor } from '@math/serving.js';
import { lookupFact } from '@shared/claims.js';
import { shareText, exactBytes } from './format.js';
import { contiguousAt, pagedAt, POOL_SLOTS, LAST_STEP } from './numbers.js';

const MISSING = '—';
const STEPS = Array.from({ length: LAST_STEP + 1 }, (_, i) => i);
const sumWaste = (sims) => sims.reduce((a, s) => a + s.waste, 0);

export const HOOK = 'Why did early LLM servers run out of KV memory while {sv:pagedattention.waste_before_pct|raw}% of it held nothing?';

// The toy's worst case before paging (step 0) and its paged waste over the six steps, from the simulators.
export const WORST_BEFORE = shareText(Math.max(...STEPS.map((s) => contiguousAt(s).waste)), POOL_SLOTS);
const pagedMean = shareText(sumWaste(STEPS.map((s) => pagedAt(s))), POOL_SLOTS * STEPS.length);
const pagedPeak = shareText(Math.max(...STEPS.map((s) => pagedAt(s).waste)), POOL_SLOTS);

export function intuition() {
  return [
    'In its attention layers, a request keeps a K and a V tile for each token it has seen, and it gets one more pair per decode step. (Hybrid models\' linear layers keep a fixed-size state instead; see [[long-context-attention]].) Nobody knows in advance how many: a reply can be 3 tokens or 3,000. Early servers solved this the blunt way: the moment a request arrived they reserved one contiguous strip big enough for the longest reply the model could give.',
    `Most of that strip stays empty for most of the request's life, and it cannot be lent to anyone else. Strips of different sizes also leave gaps between them that are too small for the next request. The toy below shows only the first kind of waste, so its worst case (${WORST_BEFORE}) is lower than the measured {sv:pagedattention.waste_before_pct|raw}%. The vLLM team measured that only {sv:pagedattention.kv_useful_before_pct|raw}% of KV memory in such systems held real tokens. The number of requests you can run at once is set by how many fit in memory, so that waste caps the batch, and in decode a bigger batch means more tokens per second, up to a point (see [[batching]]). These are the same four requests as [[batching]]: its 3-seat lane is this page's before-lane and its 4-seat lane the after-lane, so here the seats become memory.`,
    'PagedAttention borrows the operating system\'s answer to the same problem. RAM is handed out in fixed-size pages, and each program has a page table that maps its own numbering to wherever its pages actually landed. Here the "page" is a block of KV for a fixed number of tokens ({sv:vllm.default_block_size} in vLLM, 4 in this page\'s toy), handed out only when a request fills its current block, and each request\'s block table tells the attention kernel where every block is.',
    'Two things fall out. Waste shrinks to at most one partly filled block per request, and finished blocks go straight back to the pool for anyone. And a block becomes something several sequences can point at: two continuations of one prompt share its blocks, with a reference count and copy-on-write. That sharing is the bridge to [[prefix-caching]]. The price is that paging over-commits: reserving the maximum guaranteed every admitted request could finish, blocks on demand do not, so if the pool runs dry mid-reply the scheduler preempts a request, freeing its blocks and recomputing them later (vLLM V1 preempts by {sv:vllm.v1_preemption|raw}; [[batching]] names the knob).',
  ];
}

export const FRAMING = 'These are the numbers behind the idea: what the 2023 paper measured, what one block weighs for three real models, and where blocks go next. Dated figures come with their source; anything not confirmed by a primary source is marked "reported".';

const fact = (data, dataset, id, key) => lookupFact(data?.[dataset], id, key)?.value ?? null;
const sizeOr = (bytes) => (bytes == null ? MISSING : formatBytes(bytes));

// "Llama-3.1-70B 327,680 B (≈ 328 kB) per token: 5.24 MB per 16-token block, 42.9 GB for one request at its 131,072-token context".
function modelClause(data, { id, name, kind, blockSize }) {
  const perToken = fact(data, 'models', id, 'kv_bytes_per_token');
  const context = fact(data, 'models', id, 'context_length');
  if (perToken == null || context == null) return `${name}: ${MISSING}`;
  const reserve = reserveMaxBytes(perToken, context);
  return `${name}{${id}.kv_bytes_per_token|cite} ${exactBytes(perToken)} (≈ ${formatBytes(perToken)}) per token (${kind}): ${formatBytes(kvBytesPerBlock(perToken, blockSize))} per ${blockSize}-token block, ${formatBytes(reserve)} reserved for one request at its ${context.toLocaleString('en-US')}-token context`;
}

export function factRows(data) {
  const block = fact(data, 'serving', 'vllm', 'default_block_size') ?? 16;
  const h100Entry = data?.hardware?.entries?.find((e) => e.id === 'h100');
  const h100 = h100Entry ? hbmFor(h100Entry) : null;
  const llamaReserve = (() => {
    const perToken = fact(data, 'models', 'llama-3.1-70b', 'kv_bytes_per_token');
    const context = fact(data, 'models', 'llama-3.1-70b', 'context_length');
    return perToken == null || context == null ? null : kvCacheBytes({ bytesPerToken: perToken, tokens: context });
  })();
  const llamaShare = llamaReserve && h100 ? `${shareText(llamaReserve, h100.bytes)} of an H100 (${formatBytes(h100.bytes)} ${h100.basis})` : MISSING;
  const avgKv = fact(data, 'serving', 'deepseek-v3-production', 'avg_kv_length_tokens');
  const avgWaste = avgKv ? shareText(block - 1, avgKv) : MISSING;
  return [
    { claim: 'Only {sv:pagedattention.kv_useful_before_pct|raw}% of KV cache memory held actual tokens in the existing systems the 2023 paper profiled (its Fig. 2); the vLLM launch post puts the waste at {sv:pagedattention.waste_before_pct|raw}%, and under {sv:pagedattention.waste_after_pct_max|raw}% after paging.' },
    { claim: 'PagedAttention raised throughput by a factor of {sv:pagedattention.throughput_gain|raw} over FasterTransformer and Orca at the same latency, measured as served requests per second at the same normalized latency (the 2023 paper).' },
    { claim: 'In the 2023 paper its attention kernel was {sv:pagedattention.kernel_overhead_2023_pct|raw}% slower than FasterTransformer\'s: the win was memory, not speed per token.' },
    { claim: 'vLLM\'s default block size is {sv:vllm.default_block_size} tokens; too small starves GPU parallelism, too large fragments memory and shares less.' },
    { claim: 'Paged KV is the default memory model in {sv:pagedattention.engines|raw}.' },
    { claim: `What one block weighs, per token, per ${block}-token block and for one request at its whole context (decimal units): ${[
      { id: 'llama-3.1-70b', name: 'Llama-3.1-70B', kind: 'GQA, 2 (K and V) × 80 layers × 8 KV heads × 128 × 2 B' },
      { id: 'deepseek-v3', name: 'DeepSeek-V3', kind: 'MLA, 61 layers × 576 × 2 B' },
      { id: 'gpt-3', name: 'GPT-3', kind: 'MHA' },
    ].map((m) => modelClause(data, { ...m, blockSize: block })).join('; ')}. The Llama figure is ${llamaShare}.{hw:h100.hbm_gb|cite}` },
    { claim: 'Hybrid models page only their attention layers: Kimi K3 has {kimi-k3.full_attention_layers} gated-MLA layers among {kimi-k3.linear_attention_layers} KDA layers, Qwen3.8 alternates layers in a {qwen3.8.layer_pattern} pattern; the linear layers keep a fixed-size state instead of per-token KV.' },
    { claim: `At DeepSeek's production average of {sv:deepseek-v3-production.avg_kv_length_tokens|int} KV tokens per request (V3/R1, {sv:deepseek-v3-production.date|date}), a ${block}-token block wastes at most ${block - 1} slots, ${avgWaste}; the toy's ${pagedMean}–${pagedPeak} comes from 5–16-token sequences.` },
    { claim: 'Bridge: {sv:deepseek-v3-production.kv_hit_rate_pct|raw}% of DeepSeek\'s input tokens hit the KV cache (prefix reuse), on the same production day.' },
    { claim: 'As of {sv:vllm-tiered-kv.date|date}, vLLM can spill blocks down the tiers {sv:vllm-tiered-kv.tiers}: the numbers are in [[prefix-caching]].' },
    { claim: 'FP8 KV halves the bytes per block ({sv:vllm-fp8-kv.bytes_factor|raw} of BF16\'s, vLLM, {sv:vllm-fp8-kv.date|date}), but a naive version dropped 128K needle-in-a-haystack accuracy to {sv:vllm-fp8-kv.naive_niah_128k_pct}%: details in [[quantization]].' },
    { claim: 'Concurrency is free HBM divided by KV per sequence. DeepSeek-R1 (NVFP4) at 128K input and 8K output has a theoretical cap of {hw:gb300-nvl72.concurrent_128k_per_gpu} requests per GPU on GB300 NVL72 ({hw:gb300-nvl72.hbm_gb} GB per GPU, nominal) against {hw:gb200-nvl72.concurrent_128k_per_gpu} on GB200 NVL72 ({hw:gb200-nvl72.hbm_gb} GB per GPU, the rack total over 72, nominal); LMSYS\'s practical target is {hw:gb300-nvl72.concurrent_128k_per_gpu_target} and {hw:gb200-nvl72.concurrent_128k_per_gpu_target}, about 85% of the cap (LMSYS, {sv:lmsys-gb300-longctx.date|date}).' },
  ];
}

// Page text under the stage, one list per frame (0-based).
export const STEP_RULE = 'The prefill step ends with a request\'s first token; each tick after it is one decode step, one more token.';
export const BELOW = Object.freeze([
  [STEP_RULE],
  [STEP_RULE],
  [STEP_RULE],
  [],
  [],
  [],
  [],
  [],
  ['This is the 2023 paper\'s mechanism. Today\'s vLLM caches and shares only {sv:vllm-prefix-cache.granularity}, keyed by content (vLLM design doc); [[prefix-caching]] covers the 2026 engines.'],
]);

