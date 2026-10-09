// serving-calculator's dated text (pure, no DOM): the hook, the intuition and takeaways, the §8 rows and framing, and the notes under
// the stage. Dated numbers are {entry.key|format} placeholders filled from data/*.json; the worked example's derived numbers
// (54.1 GB, 234 GB, 1,889 users, ...) come from figures.js, never typed.
import { formatBytes, formatInt, formatCount } from '@math/core.js';
import { V4, CONTEXT, EP_SIZE } from './numbers.js';
import { MIN_GPUS, WEIGHTS_PER_GPU, FREE_PER_GPU, kvPerUser, usersFit, atTarget, tokSUser, tokSGpu } from './figures.js';

const kb = (bytes) => formatBytes(bytes);
// "36.9 to 111 MB": one unit when both ends share it.
function range(lowBytes, highBytes) {
  const [low, high] = [kb(lowBytes), kb(highBytes)];
  const unit = (text) => text.split(' ')[1];
  return unit(low) === unit(high) ? `${low.split(' ')[0]} to ${high}` : `${low} to ${high}`;
}

export const hook = 'How many GPUs does it take to serve a 1.6-trillion-parameter model, and what does a million output tokens cost?';

export function intuition() {
  const per = range(kvPerUser(CONTEXT.short, 'low'), kvPerUser(CONTEXT.short, 'high'));
  const long = range(kvPerUser(CONTEXT.long, 'low'), kvPerUser(CONTEXT.long, 'high'));
  const fitLong = `${formatInt(usersFit(CONTEXT.long, 'high'))} to ${formatInt(usersFit(CONTEXT.long, 'low'))}`;
  return [
    `Start with the weights. DeepSeek-V4-Pro ships its experts in FP4 and everything else in FP8, about {deepseek-v4-pro.checkpoint_gb} GB (reported). A GB300 holds {hw:gb300-nvl72.hbm_gb} GB, so the weights alone need at least ${MIN_GPUS} GPUs. In practice one copy of the model, a replica, spreads its experts over ${EP_SIZE} GPUs (wide expert parallelism: see [[disaggregation]]), leaving each GPU ${kb(WEIGHTS_PER_GPU)} of weights and ${kb(FREE_PER_GPU)} for users.`,
    `Then the users. Each one's KV cache lives on the GPU for the whole answer. V4-Pro's compressed attention keeps that small, about ${range(V4.kvBytesPerToken.low, V4.kvBytesPerToken.high)} per token, so at 8K tokens a user costs ${per} and thousands fit. At 1M tokens a user costs ${long} and only ${fitLong} fit per GPU. Once memory is full of KV, every decode step has to read most of the GPU's {hw:gb300-nvl72.hbm_gb} GB, which in this floor caps each user near ${Math.round(tokSUser(usersFit(CONTEXT.long), CONTEXT.long))} tokens per second (sparse attention reads less, so real long-context models can do better).`,
    `Speed and cost come last. The floor from [[prefill-decode]] says one user could get ${formatCount(tokSUser(1, CONTEXT.short))} tokens per second; a {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user} tokens-per-second target leaves room, in the ideal, for ${formatInt(atTarget(CONTEXT.short).users)} users decoding at once per GPU. Measured systems get far less, because communication, attention and scheduling eat most of each step. InferenceX measured {sv:inferencex-v4-pro-gb300.throughput_tok_s_gpu} tokens per second per GPU at {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user} per user, which at \${sv:inferencex-v4-pro-gb300.gpu_hour_usd|raw} per GPU-hour is \${sv:inferencex-v4-pro-gb300.cost_per_m|raw} per million tokens, input and output counted together. That is far below DeepSeek's list price (\${sv:pricing-deepseek-v4-pro.input_miss_usd_per_m|raw} input, \${sv:pricing-deepseek-v4-pro.output_usd_per_m|raw} output off-peak), and the gap between input and output prices follows from everything in this track: input tokens are read many at a time and often come from cache, while output tokens come one per step and hold memory while they do. List prices also pay for what that figure leaves out: a server is sized for its peak traffic, so its GPUs sit partly idle in quiet hours.`,
  ];
}

export function takeaways() {
  return [
    `Weights set the minimum GPUs (${formatBytes(V4.checkpointBytes)} needs at least ${MIN_GPUS} GB300s; ${EP_SIZE} in a practical replica), but the KV cache sets how many users fit: thousands per GPU at 8K, a few dozen at 1M, where a full GPU caps each user near ${Math.round(tokSUser(usersFit(CONTEXT.long), CONTEXT.long))} tokens per second (frames 2–5, try-this 1).`,
    `The roofline floor says what is possible (${formatCount(tokSUser(1, CONTEXT.short))} tok/s for one user; ${formatInt(atTarget(CONTEXT.short).users)} users at {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user} tok/s in the ideal); measured systems sit far below it, so cost is measured throughput priced per GPU-hour: about \${sv:inferencex-v4-pro-gb300.cost_per_m|raw} per million input and output tokens for V4-Pro on GB300 (frames 6–9, try-this 2).`,
    'Output costs more than input because GPUs make far fewer output tokens per second than they read input, and long contexts widen the gap; cache hits cheapen only the input side, and MTP, guessing a second token per step, helps only where decode is memory-bound (frame 10, try-this 3–4).',
  ];
}

export const framing = () => 'Every number below comes from data/*.json; those the sources did not confirm are labeled "reported". The floor and the measurements are compared only in the same unit, as a range: whether InferenceX counts input tokens in tok/s/GPU is not stated.';

// The 10 sourced rows of storyboard §8.
export function factRows() {
  return [
    { claim: 'DeepSeek-V4-Pro: {deepseek-v4-pro.total_params|count} total, {deepseek-v4-pro.active_params|count} active, {deepseek-v4-pro.layers} layers, {deepseek-v4-pro.experts_total} routed experts ({deepseek-v4-pro.experts_active} per token), {deepseek-v4-pro.context_length|count} context; d_model {deepseek-v4-pro.d_model}; expert width {deepseek-v4-pro.expert_hidden}.' },
    { claim: 'Weights: {deepseek-v4-pro.weight_formats}; the checkpoint is about {deepseek-v4-pro.checkpoint_gb} GB.' },
    { claim: 'KV per token is {deepseek-v4-pro.kv_bytes_per_token} bytes (low to high): formula-derived from the config, and the sources disagree on the layer mix.' },
    { claim: 'GB300 NVL72, per GPU: {hw:gb300-nvl72.hbm_gb} GB HBM (nominal; no usable figure is in data), {hw:gb300-nvl72.hbm_tbps|raw} TB/s, {hw:gb300-nvl72.fp8_e4m3_dense_tflops} TFLOPS dense FP8, {hw:gb300-nvl72.nvfp4_dense_tflops} TFLOPS dense FP4. GB200 NVL72: {hw:gb200-nvl72.hbm_gb} GB per GPU (the rack total, {hw:gb200-nvl72.rack_hbm_tb|raw} TB, over 72 GPUs), {hw:gb200-nvl72.hbm_tbps|raw} TB/s.' },
    { claim: 'InferenceX, DeepSeek-V4-Pro, 8K in / 1K out, FP4, disaggregated Dynamo + vLLM, measured {sv:inferencex-v4-pro-gb300.date|date}: GB300 {sv:inferencex-v4-pro-gb300.throughput_tok_s_gpu} tok/s/GPU at {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user} tok/s/user, ${sv:inferencex-v4-pro-gb300.cost_per_m|raw}/M at ${sv:inferencex-v4-pro-gb300.gpu_hour_usd|raw} per GPU-hour; GB200 {sv:inferencex-v4-pro-gb200.throughput_tok_s_gpu} at {sv:inferencex-v4-pro-gb200.interactivity_tok_s_user}, ${sv:inferencex-v4-pro-gb200.cost_per_m|raw}/M at ${sv:inferencex-v4-pro-gb200.gpu_hour_usd|raw}; peak {sv:inferencex-v4-pro-gb300.max_throughput_tok_s_gpu} (GB300) and {sv:inferencex-v4-pro-gb200.max_throughput_tok_s_gpu} (GB200) at {sv:inferencex-v4-pro-gb300.max_throughput_tok_s_user|raw} and {sv:inferencex-v4-pro-gb200.max_throughput_tok_s_user|raw} tok/s/user. Whether tok/s/GPU includes input tokens is not stated.' },
    { claim: 'DeepSeek production (V3/R1, {sv:deepseek-v3-production.date|date}): about {sv:deepseek-v3-production.prefill_tok_s_node|count} input vs {sv:deepseek-v3-production.decode_tok_s_node|count} output tokens per second per H800 node; ${sv:deepseek-v3-production.cost_per_day_usd} a day at ${sv:deepseek-v3-production.gpu_hour_usd|raw} per GPU-hour; {sv:deepseek-v3-production.kv_hit_rate_pct|raw}% of input tokens hit the cache; average KV length {sv:deepseek-v3-production.avg_kv_length_tokens} tokens.' },
    { claim: 'List prices (read {sv:pricing-deepseek-v4-pro.date|date}): DeepSeek V4-Pro off-peak ${sv:pricing-deepseek-v4-pro.input_miss_usd_per_m|raw} per M input tokens (cache miss) and ${sv:pricing-deepseek-v4-pro.output_usd_per_m|raw} per M output; Anthropic charges {sv:pricing-anthropic.output_input_ratio}× as much for output as for input across tiers; its batch API is {sv:pricing-anthropic.batch_discount_pct}% off.' },
    { claim: 'Why output costs more (analysis, not a source claim): input tokens are processed many per weight read, at high intensity, and often come from the cache; output tokens come one per step per user and each user\'s KV stays in memory for the whole answer; and providers keep latency headroom, with smaller batches, for output.', derived: true },
    { claim: 'LMSYS ({sv:lmsys-gb300-longctx.date|date}; DeepSeek-R1, GB300 NVL72, 128K in / 8K out): multi-token prediction raised per-user speed {sv:lmsys-gb300-longctx.mtp_per_user_gain_pct}%. Theoretical caps of {hw:gb300-nvl72.concurrent_128k_per_gpu} concurrent 128K requests per GPU on GB300 vs {hw:gb200-nvl72.concurrent_128k_per_gpu} on GB200; the practical targets are {hw:gb300-nvl72.concurrent_128k_per_gpu_target} and {hw:gb200-nvl72.concurrent_128k_per_gpu_target}.' },
    { claim: 'Claude models from 4.6 on charge one flat per-token price for a 1M-token context{sv:pricing-anthropic.flat_1m_context|cite}.' },
  ];
}

// Page text under the stage, one list per frame.
export const BELOW = Object.freeze([
  ['The {deepseek-v4-pro.checkpoint_gb} GB checkpoint size is reported, not confirmed; BF16 and FP8 sizes are parameters times bits per number.'],
  ['GB200 NVL72\'s {hw:gb200-nvl72.hbm_gb} GB is the rack\'s {hw:gb200-nvl72.rack_hbm_tb|raw} TB over 72 GPUs; GB300\'s {hw:gb300-nvl72.hbm_gb} GB is nominal, because no usable figure is in data.'],
  [],
  ['V4-Pro\'s KV per token is a reported estimate, {deepseek-v4-pro.kv_bytes_per_token} bytes from low to high; both ends are shown. The toy lets you pick one for the speed numbers.'],
  [],
  [],
  [],
  ['InferenceX does not say whether its tok/s/GPU counts input tokens, so the floor and the measurement are compared as a range. The listed causes are the brief\'s estimate, not measured shares.'],
  ['List prices are off-peak: ${sv:pricing-deepseek-v4-pro.input_miss_usd_per_m|raw} per M input tokens on a cache miss, ${sv:pricing-deepseek-v4-pro.output_usd_per_m|raw} per M output.'],
  ['Per node vs per GPU: the two bars and the floor ratio compare shapes, not sizes. Anthropic\'s ratio is across its tiers: {sv:pricing-anthropic.output_input_ratio}×.'],
]);
