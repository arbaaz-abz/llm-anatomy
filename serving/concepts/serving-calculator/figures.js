// serving-calculator stage numbers (pure): every figure the frames print, computed once from numbers.js through the math/ functions
// (README lesson 16: one function per metric; the toy reads the same functions through model.js).
import { stepTime, weightBytes, freeHbmPerGpu, maxUsersPerGpu, usersAtTarget, costPerMillion, minGpusForWeights, prefillTokPerSecCeiling } from '@math/serving.js';
import { kvCacheBytes } from '@math/memory.js';
import { bitsPerElement } from '@math/roofline.js';
import { V4, GB300, GB200_HBM_BYTES, EP_SIZE, STAGE_MODEL, TARGET_TOK_S_USER, MEASURED, ISL_OSL } from './numbers.js';

export const WEIGHTS = V4.checkpointBytes;
export const WEIGHTS_PER_GPU = STAGE_MODEL.weightBytesPerGpu;
export const FREE_PER_GPU = freeHbmPerGpu({ hbmBytes: GB300.hbmBytes, weightBytes: WEIGHTS, gpus: EP_SIZE });
export const MIN_GPUS = minGpusForWeights(WEIGHTS, GB300.hbmBytes);
export const MIN_GPUS_GB200 = minGpusForWeights(WEIGHTS, GB200_HBM_BYTES);
export const WEIGHTS_BF16 = weightBytes({ params: V4.totalParams, bitsPerParam: bitsPerElement('bf16') });
export const WEIGHTS_FP8 = weightBytes({ params: V4.totalParams, bitsPerParam: bitsPerElement('fp8_e4m3') });

// KV for one user of `tokens` at the low or high end of the reported range, and how many fit beside the weights.
export const kvPerUser = (tokens, end = 'low') => kvCacheBytes({ bytesPerToken: V4.kvBytesPerToken[end], tokens });
export const usersFit = (tokens, end = 'low') => maxUsersPerGpu(FREE_PER_GPU, kvPerUser(tokens, end));

// One decode step with `users` users each holding `tokens` of context.
export const decodeStep = (users, tokens, end = 'low') => stepTime({ ...STAGE_MODEL, kvBytesPerToken: V4.kvBytesPerToken[end], tokens: users, seqs: users, context: tokens });
export const tokSUser = (users, tokens, end = 'low') => 1 / decodeStep(users, tokens, end).timeS;
export const tokSGpu = (users, tokens, end = 'low') => users / decodeStep(users, tokens, end).timeS;

// The most users at the 27 tok/s target when `fit` users fit.
export const atTarget = (tokens, end = 'low') => usersAtTarget({
  targetTokPerUser: TARGET_TOK_S_USER, ...STAGE_MODEL, context: tokens, kvBytesPerToken: V4.kvBytesPerToken[end], maxUsers: usersFit(tokens, end),
});

export const WEIGHT_READ_S = WEIGHTS_PER_GPU / (GB300.bandwidthTBps * 1e12); // the weights-only floor of one step
export const IMPLIED_S = 1 / TARGET_TOK_S_USER;
export const FLOOR_TO_IMPLIED = IMPLIED_S / WEIGHT_READ_S; // 5.48
export const PREFILL_CEILING = prefillTokPerSecCeiling({ activeParamsPerGpu: STAGE_MODEL.activeParamsPerGpu, peakTflops: STAGE_MODEL.peakTflops });

// The measured row against the floor at the target: output-only (8.25×) and, if InferenceX counts input tokens too, ÷ 9 (74.3×).
export const TOKENS_PER_OUTPUT = (ISL_OSL.input + ISL_OSL.output) / ISL_OSL.output;
export const MEASURED_OUTPUT_ONLY = MEASURED.gb300.tokSGpu / TOKENS_PER_OUTPUT;
export const costOfMeasured = (row) => costPerMillion(row.usdPerGpuHour, row.tokSGpu);
