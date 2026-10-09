// prefill-decode's running-example states (pure, no DOM): every number the stage and the prose print comes from
// stepTime / maxUsersPerGpu on RUNNING_EXAMPLE (Llama-3.1-70B FP8 on one H200, math/serving.js), never retyped.
import { RUNNING_EXAMPLE as L, stepTime, freeHbmPerGpu, maxUsersPerGpu } from '@math/serving.js';
import { kvCacheBytes } from '@math/memory.js';
import { ridgePoint, tokensToComputeBound, arithmeticIntensity } from '@math/roofline.js';
import { H200, H100, CONTEXT, PREFILL_TOKENS, CROSS_TOKENS, USERS_SHOWN } from './numbers.js';

const TB = 1e12;
const ROOF = Object.freeze({ peakTflops: L.peakTflops, bandwidthTBps: L.bandwidthTBps });
const bytesPerWeight = L.weightBytesPerGpu / L.activeParamsPerGpu; // FP8: 1

export const prefill = (tokens) => stepTime({ ...L, tokens, seqs: 0, context: 0 });
// Decode for `users` users, each with `context` tokens cached (frame 2: one user, no context).
export const decode = (users, context = CONTEXT) => stepTime({ ...L, tokens: users, seqs: users, context });
export const intensityOf = (step) => arithmeticIntensity(step);

// The reading parts of a step, in seconds at the GPU's bandwidth: weights, KV, activations (zero parts dropped).
export function readParts(step, { weightBytes = L.weightBytesPerGpu, bandwidthTBps = L.bandwidthTBps, act = 'activations' } = {}) {
  const at = (bytes) => bytes / (bandwidthTBps * TB);
  return [
    { label: 'weights read', s: at(weightBytes) },
    { label: 'KV read', s: at(step.kvBytes) },
    { label: act, s: at(step.actBytes) },
  ].filter((p) => p.s > 0);
}

export const WEIGHTS_READ_S = L.weightBytesPerGpu / (L.bandwidthTBps * TB);
export const RIDGE = ridgePoint(ROOF);
export const CROSSING = tokensToComputeBound({ ...ROOF, bytesPerElem: bytesPerWeight, k: L.dModel, n: L.dModel });
export const H100_CROSSING = tokensToComputeBound({ ...H100, k: L.dModel, n: L.dModel });
export const FREE_BYTES = freeHbmPerGpu({ hbmBytes: H200.hbmBytes, weightBytes: L.weightBytesPerGpu, gpus: 1 });
export const CACHE_PER_USER = kvCacheBytes({ bytesPerToken: L.kvBytesPerToken, tokens: CONTEXT });
export const MAX_USERS = maxUsersPerGpu(FREE_BYTES, CACHE_PER_USER); // 105 at 2,048 tokens

// The states the frames step through (each a real stepTime output).
export const STATES = Object.freeze({
  decodeAlone: decode(1, 0),
  prefill: prefill(PREFILL_TOKENS),
  cross: prefill(CROSS_TOKENS),
  batch: Object.freeze(Object.fromEntries([...USERS_SHOWN, MAX_USERS].map((u) => [u, decode(u)]))),
});

// The stage's one seconds scale (P4-R8): the longest step it draws, the 1,000-token prefill.
export const STAGE_SCALE_S = STATES.prefill.timeS;

export const perUser = (step) => 1 / step.timeS;
export const perGpu = (users, step) => users / step.timeS;
export { L as MODEL };
