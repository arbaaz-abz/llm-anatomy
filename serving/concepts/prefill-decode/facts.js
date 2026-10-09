// prefill-decode's dated text (pure, no DOM): the §8 rows and framing, the hook, intuition and takeaways, and the notes
// printed under the stage. Hardware figures are {hw:…} placeholders filled from data/*.json; the running example's
// times and counts come from model.js (stepTime on RUNNING_EXAMPLE, which tests pin to the same data), never typed.
import { tokensToComputeBound, ridgePoint } from '@math/roofline.js';
import { FORWARD_FLOPS_PER_PARAM_TOKEN } from '@math/serving.js';
import { formatBytes, formatDuration, formatInt } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';
import { MODEL, STATES, WEIGHTS_READ_S, CROSSING, MAX_USERS } from './model.js';
import { fixed1 } from './format.js';
import { CONTEXT, PREFILL_TOKENS, CROSS_TOKENS } from './numbers.js';

const DASH = '—';
const hw = (data, id, key) => lookupFact(data?.hardware, id, key)?.value ?? null;

// A number derived from data, or a dash when the data is absent (LESSON = lessonFor(null) is for validation only).
function derived(data, compute) {
  try {
    return data?.hardware ? compute() : DASH;
  } catch {
    return DASH;
  }
}

const crossingOf = (data, id, peakKey, bytesPerElem) => tokensToComputeBound({ peakTflops: hw(data, id, peakKey), bandwidthTBps: hw(data, id, 'hbm_tbps'), bytesPerElem, k: MODEL.dModel, n: MODEL.dModel });
const h100Cross = (data) => derived(data, () => formatInt(crossingOf(data, 'h100', 'bf16_dense_tflops', 2)));
const b200Cross = (data) => derived(data, () => {
  const [bf16, fp8] = [crossingOf(data, 'b200', 'bf16_dense_tflops', 2), crossingOf(data, 'b200', 'fp8_e4m3_dense_tflops', 1)].map(formatInt);
  return bf16 === fp8 ? bf16 : `${bf16} (BF16) or ${fp8} (FP8)`;
});
const h200Ridge = (data) => derived(data, () => fixed1(ridgePoint({ peakTflops: hw(data, 'h200', 'fp8_e4m3_dense_tflops'), bandwidthTBps: hw(data, 'h200', 'hbm_tbps') })));

const n = Object.freeze({
  opsPerToken: `${Number(((FORWARD_FLOPS_PER_PARAM_TOKEN * MODEL.activeParamsPerGpu) / 1e9).toPrecision(3))} billion`,
  weights: formatBytes(MODEL.weightBytesPerGpu),
  read: formatDuration(WEIGHTS_READ_S),
  math1: formatDuration(STATES.decodeAlone.computeS),
  prefillMath: formatDuration(STATES.prefill.computeS),
  prefillRead: formatDuration(STATES.prefill.memoryS),
  cross: formatInt(CROSSING),
  crossTime: formatDuration(STATES.cross.timeS),
  prompt: formatInt(PREFILL_TOKENS),
  context: formatInt(CONTEXT),
  maxUsers: formatInt(MAX_USERS),
});

export const hook = `Why can a GPU read ${formatInt(CROSS_TOKENS)} tokens of your prompt in about the time it takes to write one token of the answer?`;

export const intuition = Object.freeze([
  `Every token that passes through a 70B model does about two arithmetic operations per weight (a multiply and an add, counted as two): ${n.opsPerToken} operations. To do them, the GPU must bring every weight from its high-bandwidth memory (HBM) into its compute units. In FP8 that is ${n.weights}, and the H200 reads {hw:h200.hbm_tbps|raw} TB per second, so one pass over the weights takes ${n.read} no matter how few tokens use it. The arithmetic for one token takes ${n.math1}. Decode makes one token per request per step, so it spends almost the whole step waiting for memory.`,
  `Prefill is the opposite case. All prompt tokens go through together, so each weight, once read, is used for every one of them. With ${n.prompt} prompt tokens the arithmetic takes ${n.prefillMath}, while the reading (weights plus the tokens' own activations) takes ${n.prefillRead}: now the GPU waits on math. Counted in tokens per weight read, as in [[gpu-primer]], an H200 needs ${n.cross}. At ${n.cross} prompt tokens math and reading both take ${n.crossTime}, about as long as one decode step. That is the answer to the hook: reading a prompt reuses each weight hundreds of times, writing an answer uses it once.`,
  `This is why servers batch decode. Eight users decoding together share one read of the weights, so the step barely grows and the GPU makes nearly eight times more tokens. The catch is the KV cache: the new token attends to every earlier one, so each user's cached keys and values must also be read every step, and they take memory. At ${n.context} tokens of context the H200 is full at ${n.maxUsers} users, long before the batch would reach the ridge. More users per GPU means cheaper tokens but a slower stream for each user. Real servers pick a point on that curve.`,
]);

export const intuitionNote = 'This page builds on [[serving-overview]] (prefill, the decode step, TTFT and TPOT) and [[gpu-primer]] (HBM, arithmetic intensity, the roofline and its ridge point).';

export const takeaways = Object.freeze([
  `Decode reads every weight to make one token per request, so it waits on memory (${n.read} for ${n.weights} on an H200); prefill reuses each weight for every prompt token, so it waits on math.`,
  `The crossover is the GPU's ridge, counted as tokens per weight read: ${n.cross} prompt tokens on an H200 take about as long as one decode step, which is why reading a prompt is cheap per token and writing an answer is not.`,
  'Batching decode shares the weight read, raising tokens per GPU while slowing each user; KV memory usually caps the batch before the ridge, and long contexts shrink it fast.',
]);

export const framing = 'GPU figures are vendor dense peaks, with HBM capacity marked nominal or usable and bandwidth in decimal TB/s; the measured serving points are InferenceX runs, and each row names its model, hardware and conditions.';

// The 5 sourced rows of storyboard §8.
export function factRows(data) {
  return [
    { claim: `H200: {hw:h200.hbm_gb} GB HBM3e (nominal), {hw:h200.hbm_tbps|raw} TB/s, {hw:h200.fp8_e4m3_dense_tflops} TFLOP/s dense FP8 (reported: 2 × BF16); ridge ${h200Ridge(data)} FLOPs per byte.` },
    { claim: `From their dense peaks and HBM bandwidth, the H100 crosses into compute-bound at ${h100Cross(data)} tokens per weight read and the B200 at ${b200Cross(data)}, in BF16 as in FP8{hw:h100.bf16_dense_tflops|cite}{hw:h100.hbm_tbps|cite}{hw:b200.bf16_dense_tflops|cite}{hw:b200.fp8_e4m3_dense_tflops|cite}{hw:b200.hbm_tbps|cite}.` },
    { claim: 'Llama-3.1-70B: {llama-3.1-70b.layers} layers, {llama-3.1-70b.n_kv_heads} KV heads × {llama-3.1-70b.head_dim}, d_model {llama-3.1-70b.d_model}, a {llama-3.1-70b.context_length}-token context and {llama-3.1-70b.kv_bytes_per_token} KV bytes per token in BF16; {llama-3.1-70b.total_params|count} parameters as its card rounds them.' },
    { claim: 'DeepSeek-V4-Pro on GB300 NVL72 made {sv:inferencex-v4-pro-gb300.max_throughput_tok_s_gpu} tok/s/GPU at {sv:inferencex-v4-pro-gb300.max_throughput_tok_s_user|raw} tok/s/user (max throughput) and {sv:inferencex-v4-pro-gb300.throughput_tok_s_gpu} at {sv:inferencex-v4-pro-gb300.interactivity_tok_s_user|raw}; on GB200 NVL72, {sv:inferencex-v4-pro-gb200.max_throughput_tok_s_gpu} at {sv:inferencex-v4-pro-gb200.max_throughput_tok_s_user|raw} and {sv:inferencex-v4-pro-gb200.throughput_tok_s_gpu} at {sv:inferencex-v4-pro-gb200.interactivity_tok_s_user|raw} (ISL 8K / OSL 1K, FP4, InferenceX, measured {sv:inferencex-v4-pro-gb300.date}); the source does not say whether tok/s/GPU counts input tokens too.' },
    { claim: 'Kimi K2.5 on B200 (NVFP4) cost ${sv:inferencex-kimi-k2.5-b200.cost_per_m_at_32|raw} per million tokens at 32 tok/s/user and ${sv:inferencex-kimi-k2.5-b200.cost_per_m_at_90|raw} at 90 tok/s/user (InferenceX); the source does not say whether those tokens include input.' },
  ];
}

// Page text under the stage, one list per frame: dated notes (filled from data) and frame → lesson hand-offs.
export const BELOW = Object.freeze([
  ['Weights in FP8 take 1 byte each; the KV cache stays in BF16 (2 bytes per number). What fewer bits cost in quality: see [[quantization]].'],
  ['Every time on this page is a floor from bytes and FLOPs: real engines reach less than the HBM bandwidth, so real steps take longer.'],
  [],
  ['Arithmetic intensity and the ridge point come from [[gpu-primer]], where the same multiply is drawn one layer at a time.'],
  ['The H200\'s FP8 peak ({hw:h200.fp8_e4m3_dense_tflops} TFLOP/s dense) is reported: half of NVIDIA\'s with-sparsity figure, 2 × BF16. The ridge inherits that.'],
  [],
  ['Keeping a decode batch full while requests arrive and finish at different times is the job of [[batching]].'],
  [],
  ['Models with a few kB of KV per token, such as DeepSeek-V4-Pro, can batch far enough to reach the ridge: see [[serving-calculator]].'],
  ['InferenceX does not say whether its tok/s/GPU counts input tokens; this page\'s own curve counts output tokens only.'],
]);
