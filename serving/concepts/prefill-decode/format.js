// prefill-decode pure helpers (no DOM): the toy's one analysis of a state on a GPU preset, its stops and formatters, and
// the "Check my work" text. Every number comes from math/serving.js, math/roofline.js and math/memory.js.
import { stepTime, freeHbmPerGpu, maxUsersPerGpu, weightBytes, prefillTokPerSecCeiling, FORWARD_FLOPS_PER_PARAM_TOKEN } from '@math/serving.js';
import { bitsPerElement, bytesPerElement, ridgePoint, tokensToComputeBound, arithmeticIntensity } from '@math/roofline.js';
import { kvCacheBytes } from '@math/memory.js';
import { formatBytes, formatDuration, formatInt } from '@math/core.js';
import { WEIGHT_FORMATS, hbmText } from './hardware.js';
import { MODEL } from './model.js';

export const INITIAL_STATE = Object.freeze({ phase: 'decode', hw: 'h200', weights: 'fp8', promptTokens: 1000, users: 8, context: 2048 });
export const PROMPT_STOPS = Object.freeze([1, 16, 64, 217, 512, 1000, 2048, 8192]);
export const CONTEXT_STOPS = Object.freeze([512, 2048, 8192, 32768, 131072]);
const USER_POWERS = Object.freeze(Array.from({ length: 11 }, (_, i) => 2 ** i)); // 1 … 1,024

const ONE_DECIMAL = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const THREE_SIG = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 3 });
// One decimal, grouped: intensity, ridge and tokens per weight read ("14.8", "412.3", "1,607.5").
export const fixed1 = (n) => ONE_DECIMAL.format(n);
// Operations at 3 significant figures in GFLOP below a trillion, else TFLOP ("140 GFLOP", "1.12 TFLOP", "1,150 TFLOP").
export const formatFlops = (f) => (f < 1e12 ? `${THREE_SIG.format(f / 1e9)} GFLOP` : `${THREE_SIG.format(f / 1e12)} TFLOP`);
export const tbps = (bw) => `${bw} TB/s`;
export const tflops = (peak) => `${formatInt(peak)} TFLOP/s`;
// The parameter count as the math block writes it: 70e9.
export const paramsText = (n) => `${n / 1e9}e9`;
const plural = (n, word) => `${formatInt(n)} ${word}${n === 1 ? '' : 's'}`;

// The users slider's stops: powers of two below the max that fits, then the max itself; [0] when no user fits.
export const usersStops = (maxUsers) => (maxUsers === 0 ? [0] : [...USER_POWERS.filter((p) => p < maxUsers), maxUsers]);
export function usersLabel(users, maxUsers) {
  if (users === 0) return '0 users, none fit';
  return `${plural(users, 'user')}${users === maxUsers ? ', max that fits' : ''}`;
}

// The step-time model of Llama-3.1-70B in a weight format on a GPU (RUNNING_EXAMPLE's model, the GPU's numbers).
export function modelFor(weights, preset) {
  const fmt = WEIGHT_FORMATS[weights];
  if (!fmt) throw new RangeError(`modelFor: weights must be one of ${Object.keys(WEIGHT_FORMATS).join(', ')}, got ${weights}`);
  return Object.freeze({
    activeParamsPerGpu: MODEL.activeParamsPerGpu,
    weightBytesPerGpu: weightBytes({ params: MODEL.activeParamsPerGpu, bitsPerParam: bitsPerElement(fmt.key) }),
    dModel: MODEL.dModel,
    actBytesPerElem: bytesPerElement(fmt.key),
    kvBytesPerToken: MODEL.kvBytesPerToken,
    peakTflops: preset.peak[weights],
    bandwidthTBps: preset.bandwidthTBps,
  });
}

function stepFor(state, model, users) {
  if (state.phase === 'prefill') return stepTime({ ...model, tokens: state.promptTokens, seqs: 0, context: 0 });
  return users > 0 ? stepTime({ ...model, tokens: users, seqs: users, context: state.context }) : null;
}

// Everything the toy, the check box and the try-this list read for one state on one GPU preset (hardware.js).
// fits: the weights fit in the GPU's memory; users: the requested users clamped to the max that fits.
export function analyze(state, preset) {
  if (preset?.id !== state.hw) throw new RangeError(`analyze: the preset is ${preset?.id}, the state's GPU is ${state.hw}`);
  const model = modelFor(state.weights, preset);
  const free = freeHbmPerGpu({ hbmBytes: preset.hbm.bytes, weightBytes: model.weightBytesPerGpu, gpus: 1 });
  const fits = free >= 0;
  const cachePerUser = kvCacheBytes({ bytesPerToken: model.kvBytesPerToken, tokens: state.context });
  const maxUsers = maxUsersPerGpu(free, cachePerUser);
  const users = Math.min(state.users, maxUsers);
  const step = fits ? stepFor(state, model, users) : null;
  return {
    phase: state.phase, model, preset, free, fits, cachePerUser, maxUsers, users, step,
    clamped: fits && state.phase === 'decode' && maxUsers > 0 && state.users > maxUsers,
    intensity: step ? arithmeticIntensity(step) : null,
    ridge: ridgePoint(model),
    crossing: tokensToComputeBound({ peakTflops: model.peakTflops, bandwidthTBps: model.bandwidthTBps, bytesPerElem: model.actBytesPerElem, k: model.dModel, n: model.dModel }),
    ceiling: prefillTokPerSecCeiling(model),
    tokens: state.phase === 'prefill' ? state.promptTokens : users,
    context: state.context,
  };
}

const bytes = formatBytes;
const BOUND_WORD = Object.freeze({ memory: 'memory-bound', compute: 'compute-bound' });
export const boundText = (step) => BOUND_WORD[step.bound];

function stepLines(a) {
  const { step, model } = a;
  const bw = tbps(model.bandwidthTBps);
  const kv = step.kvBytes > 0 ? ` + ${bytes(step.kvBytes)} KV` : '';
  return [
    `t_math = ${FORWARD_FLOPS_PER_PARAM_TOKEN} × ${paramsText(model.activeParamsPerGpu)} × ${formatInt(a.tokens)} ÷ ${tflops(model.peakTflops)} = ${formatDuration(step.computeS)}`,
    `t_read = (${bytes(model.weightBytesPerGpu)} weights + ${bytes(step.actBytes)} activations${kv}) ÷ ${bw} = ${bytes(step.bytes)} ÷ ${bw} = ${formatDuration(step.memoryS)}`,
    `t_step = max(${formatDuration(step.computeS)}, ${formatDuration(step.memoryS)}) = ${formatDuration(step.timeS)}, ${boundText(step)}`,
  ];
}

function noFitLines(a) {
  const w = bytes(a.model.weightBytesPerGpu);
  return [
    `weights    = ${paramsText(a.model.activeParamsPerGpu)} × ${a.model.actBytesPerElem} bytes = ${w}`,
    `GPU memory = ${hbmText(a.preset)}`,
    `${w} of weights > ${bytes(a.preset.hbm.bytes)} of memory → does not fit on one GPU`,
  ];
}

function noRoomLines(a) {
  return [
    `free           = ${hbmText(a.preset)} − ${bytes(a.model.weightBytesPerGpu)} weights = ${bytes(a.free)}`,
    `cache per user = ${formatInt(a.context)} tokens × ${formatInt(a.model.kvBytesPerToken)} B = ${bytes(a.cachePerUser)}`,
    `users that fit = ${bytes(a.free)} ÷ ${bytes(a.cachePerUser)}, rounded down = 0`,
  ];
}

// The "Check my work" box (storyboard §6): t_step = max(t_math, t_read) with both substituted; when the weights do not
// fit, the comparison that says so; when no user's cache fits, the division that gives 0.
export function checkWork(a) {
  if (!a.fits) return noFitLines(a).join('\n');
  if (!a.step) return noRoomLines(a).join('\n');
  return stepLines(a).join('\n');
}

const MIN_PART_PX = 4; // a memory-bar part thinner than this is named under the bar instead of drawn

// The memory bar's parts (weights, KV cache, free) on a bar `w` px wide: parts thinner than 4 px are left out of the bar
// and returned in `thin`, so the drawn bar never needs a zoomed tail and every byte is still printed.
export function memoryParts({ weights, kv, total, w }) {
  const all = [
    { name: 'weights', value: weights, hue: 1 },
    { name: 'KV cache', value: kv, hue: 2 },
    { name: 'free', value: Math.max(0, total - weights - kv), hue: 3 },
  ].filter((p) => p.value > 0);
  const isThin = (p) => (p.value / total) * w < MIN_PART_PX;
  const parts = all.filter((p) => !isThin(p));
  return { parts, thin: all.filter(isThin), drawnTotal: parts.reduce((a, p) => a + p.value, 0) };
}

// "not drawn, too thin: free 536 MB" (or '' when every part is drawn).
export const thinText = (thin) => (thin.length ? `too thin to draw: ${thin.map((p) => `${p.name} ${formatBytes(p.value)}`).join(', ')}` : '');
