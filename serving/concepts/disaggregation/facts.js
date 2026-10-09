// disaggregation's dated text (pure, no DOM): the hook, intuition and takeaways (storyboard §3, §9), the §8 rows, and the notes
// printed under the stage. Dated numbers are {entry.key} placeholders filled from data/*.json; derived ones are computed from it.
import { formatRatio } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';

const DASH = '—';
const hw = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;

export const hook = 'Why would you run the first second of a request on different GPUs from the rest of it?';

export const intuition = () => [
  'In [[batching]] one long prompt stalled everyone\'s stream, and chunked prefill only spread the stall out. The deeper problem is that prefill and decode want different things. Prefill is compute-bound and likes big chunks of tokens; decode is memory-bound and needs short, steady steps, because users feel every gap between tokens. Users judge a server by two numbers, time to first token and time per output token, and on shared GPUs improving one tends to hurt the other. The measure that matters is goodput: how many requests per second meet both targets.',
  'Disaggregation gives each phase its own GPUs. A prefill pool reads prompts in big compute-bound batches. When a prompt is done, its KV cache is shipped to a decode pool, which runs nothing but short decode steps. Each pool gets its own size, its own parallelism, even its own hardware. The price is the shipping and the bookkeeping: the KV cache has to cross a link, and there are now two pools to keep in balance as traffic shifts. Over a fast link the transfer is a few percent of the prefill it follows.',
  'Mixture-of-experts models add a second reason to spread out. Each token uses only a few of hundreds of experts, so on one GPU every expert sees a token or two per step and is read from memory for almost no math. Wide expert parallelism spreads the experts over many GPUs and sends each token to its experts\' GPU (an all-to-all exchange at every MoE layer), so tokens from all of those GPUs\' users pool at each expert. That all-to-all is why the 72-GPU NVLink rack matters: it keeps every exchange on the fastest link.',
];

export const takeaways = () => [
  'On shared GPUs prefill and decode fight over every step, so improving time to first token tends to hurt time per token; disaggregation gives each phase its own pool and judges the result by goodput.',
  'The KV cache must cross a link, but past the 217-token crossover its transfer time scales with the prompt just like prefill, so the cost is a fixed fraction set by the model and the link: 0.5% over NVLink, 9.3% over a 400 Gb/s port for the running example.',
  'Wide expert parallelism spreads a MoE\'s experts over many GPUs so each expert sees enough tokens per step and each GPU holds less; it needs fast all-to-all, which is why NVL72 racks host it, and it adds pools, traffic and imbalance to manage.',
];

export const framing = () => 'Disaggregation and wide expert parallelism in production, with the papers behind them. Link speeds are per GPU, each way; the ratios and counts derived from them are computed from the same entries.';

// NVLink over the network, on Blackwell: 900 ÷ 100 GB/s (the 800 Gb/s port), computed from the entries.
export function nvlinkVsNetwork(data) {
  const [nvlink, net] = [hw(data, 'b200', 'nvlink_gb_s_each_way'), hw(data, 'network-800g', 'gb_s_each_way')];
  return nvlink == null || net == null ? DASH : formatRatio(nvlink / net);
}

// The 11 sourced rows of storyboard §8.
export function factRows(data) {
  return [
    { claim: 'Prefill/decode disaggregation: {sv:disaggregation.note}.' },
    { claim: 'DistServe ({sv:distserve.release_date|year}) framed serving as goodput: {sv:distserve.goodput_gain|raw}× more requests or {sv:distserve.slo_gain|raw}× tighter latency targets (the paper\'s abstract). Splitwise ({sv:splitwise.release_date|year}): {sv:splitwise.throughput_gain|raw}× throughput at {sv:splitwise.cost_cut_pct|raw}% lower cost, or {sv:splitwise.throughput_gain_same_cost|raw}× at the same cost (the paper\'s abstract).' },
    { claim: 'NIXL, the library that moves KV between pools: {sv:nixl.transports}.' },
    { claim: 'vLLM on GB200 (DeepSeek-R1, NVFP4, 2026-02-03): the best layout was {sv:vllm-gb200-dsr1.prefill_instances} prefill instances of {sv:vllm-gb200-dsr1.prefill_gpus_each} GPUs feeding one {sv:vllm-gb200-dsr1.decode_gpus}-GPU decode instance; {sv:vllm-gb200-dsr1.prefill_tok_s_gpu|count} prompt tokens per second per prefill GPU and {sv:vllm-gb200-dsr1.decode_tok_s_gpu|count} output tokens per second per decode GPU, at 2K tokens in and 2K out.' },
    { claim: '{sv:vllm-large-scale.disagg_moe_note} (vLLM).' },
    { claim: 'DeepSeek V3/R1 production (Feb 2025): prefill on EP{sv:deepseek-v3-production.prefill_ep} over 4 nodes ({sv:deepseek-v3-production.prefill_routed_experts_per_gpu} routed + {sv:deepseek-v3-production.prefill_shared_experts_per_gpu} shared expert per GPU); decode on EP{sv:deepseek-v3-production.decode_ep} over 18 nodes ({sv:deepseek-v3-production.decode_experts_per_gpu} per GPU), with {sv:deepseek-v3-production.redundant_experts} redundant routed experts; {deepseek-v3.experts_active} of {deepseek-v3.experts_total} routed experts active per token.' },
    { claim: '{sv:vllm-large-scale.wide_ep_note} (vLLM).' },
    { claim: 'SGLang\'s open reproduction of this layout (prefill/decode disaggregation plus expert parallelism): up to {sv:sglang-large-ep.output_gain_vs_tp16|raw}× the output throughput of TP16, and expert load balancing (EPLB) speeds prefill {sv:sglang-large-ep.eplb_prefill_gain|raw}× and decode {sv:sglang-large-ep.eplb_decode_gain|raw}×.' },
    { claim: 'DeepSeek-V4-Pro has {deepseek-v4-pro.experts_total} routed experts, {deepseek-v4-pro.experts_active} active per token, and ships as about {deepseek-v4-pro.checkpoint_gb|raw} GB.' },
    { claim: 'GB300 NVL72: 72 GPUs in one NVLink domain, about {hw:gb300-nvl72.rack_nvlink_tbps} TB/s of aggregate NVLink (NVIDIA\'s figure; it counts both directions). InferenceX credits GB300\'s {sv:inferencex-v4-pro-gb300.per_gpu_gain_vs_gb200|raw}× per-GPU throughput over GB200 at {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user} tokens per second per user to its {hw:gb300-nvl72.hbm_gb} GB of memory (nominal) allowing EP 16 and larger prefill batches (V4-Pro, 2026-05).' },
    { claim: `Link speeds per GPU, each way: NVLink5 {hw:b200.nvlink_gb_s_each_way} GB/s (NVIDIA quotes 1.8 TB/s counting both directions); network ports {hw:network-400g.gbps} Gb/s = {hw:network-400g.gb_s_each_way} GB/s and {hw:network-800g.gbps} Gb/s = {hw:network-800g.gb_s_each_way} GB/s, so NVLink is ${nvlinkVsNetwork(data)} the 800 Gb/s port on Blackwell.` },
  ];
}

// Page text under the stage, one list per frame: dated notes (filled from data) and hand-offs to other lessons.
export const BELOW = Object.freeze([
  ['The branch is [[batching]]\'s: three seats, D\'s prompt 4,096 tokens, every step timed for Llama-3.1-70B in FP8 on one H200.'],
  ['Goodput is DistServe\'s measure ({sv:distserve.release_date|year}): the highest request rate at which enough requests meet both targets. The 400 ms and 15 ms targets here are illustrative, not a published service level.'],
  [],
  ['Link speeds are per GPU, each way. NIXL moves the KV over them: {sv:nixl.transports}.'],
  ['Prefill GPUs count prompt tokens, decode GPUs count output tokens, so the two rates are not comparable. Source: vLLM, 2026-02-03.'],
  ['DeepSeek-V4-Pro has {deepseek-v4-pro.experts_total} routed experts and {deepseek-v4-pro.experts_active} active per token. How a router picks them: [[moe]].'],
  ['Expert parallelism and the all-to-all exchange are taught in [[parallelism]]; the five cuts and what each sends.'],
  ['Source: DeepSeek, Inference System Overview (Feb 2025): prefill EP{sv:deepseek-v3-production.prefill_ep}, decode EP{sv:deepseek-v3-production.decode_ep}, {sv:deepseek-v3-production.redundant_experts} redundant routed experts.'],
  ['NVIDIA\'s 130 TB/s counts both directions: 72 GPUs at 1.8 TB/s each. The links per GPU, each way, are in [[cluster-topology]].'],
  [],
]);
