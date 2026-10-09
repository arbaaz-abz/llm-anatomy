// Serving math shared by the nine Serving lessons (Plan 4, P4-R3). Pure: no DOM, inputs never mutated.
// One function per metric and one frozen running example, so a number is computed one way everywhere
// (README lesson 16). Owned by the serving track; FROZEN after Plan 4's shared prep (S6): change only
// through a shared patch.
// Units: bytes, FLOPs, seconds; TFLOPS and TB/s in (as math/roofline.js takes them).

import { deepFreeze } from './core.js';
import { rooflineTime } from './roofline.js';

// A forward pass costs 2 FLOPs per parameter per token: the forward share of training's 6 (math/scale.js
// FLOPS_PER_PARAM_TOKEN; tests/serving.test.js pins the relation) (P4-R4).
export const FORWARD_FLOPS_PER_PARAM_TOKEN = 2;

// The course-wide running example: Llama-3.1-70B, FP8 weights, one H200 (prefill-decode §1).
// tests/serving.test.js pins every field against data/models.json and data/hardware.json (P4-R21, P3-R13).
export const RUNNING_EXAMPLE = deepFreeze({
  activeParamsPerGpu: 70e9,
  weightBytesPerGpu: 70e9,
  dModel: 8192,
  actBytesPerElem: 1,
  kvBytesPerToken: 327680,
  peakTflops: 1979,
  bandwidthTBps: 4.8,
});

// The four toy requests A–D of paged-attention §5, shared by batching, paged-attention and serving-overview
// frame 9 (P4-R3). `arrives` is the step a request arrives; `prompt` and `output` are token counts.
export const TOY_REQUESTS = deepFreeze([
  { id: 'A', arrives: 0, prompt: 8, output: 4 },
  { id: 'B', arrives: 0, prompt: 5, output: 2 },
  { id: 'C', arrives: 0, prompt: 10, output: 6 },
  { id: 'D', arrives: 1, prompt: 6, output: 3 },
]);

const isNumber = (x) => typeof x === 'number' && Number.isFinite(x);

function requirePositive(fn, name, value) {
  if (!isNumber(value) || value <= 0) throw new RangeError(`${fn}: ${name} must be a positive finite number, got ${value}`);
}

function requireNonNegative(fn, name, value) {
  if (!isNumber(value) || value < 0) throw new RangeError(`${fn}: ${name} must be a finite number ≥ 0, got ${value}`);
}

function requireFinite(fn, name, value) {
  if (!isNumber(value)) throw new RangeError(`${fn}: ${name} must be a finite number, got ${value}`);
}

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 0) throw new RangeError(`${fn}: ${name} must be an integer ≥ 0, got ${value}`);
}

function requirePositiveInt(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer, got ${value}`);
}

// Usable HBM when the data has it, else the nominal figure; every page prints the basis word with the number
// (P4-R21, Review Focus 3). `entry` is a data/hardware.json entry; its facts are { value, … } objects.
export function hbmFor(entry) {
  const facts = entry && entry.facts;
  if (!facts || typeof facts !== 'object') throw new RangeError('hbmFor: entry must be a hardware entry with facts');
  const usable = facts.hbm_usable_gb;
  const nominal = facts.hbm_gb;
  const [claim, basis] = usable ? [usable, 'usable'] : [nominal, 'nominal'];
  if (!claim) throw new RangeError(`hbmFor: entry ${entry.id} has neither hbm_usable_gb nor hbm_gb`);
  requirePositive('hbmFor', 'hbm gigabytes', claim.value);
  return { bytes: claim.value * 1e9, basis };
}

// params · bits / 8.
export function weightBytes({ params, bitsPerParam }) {
  requirePositive('weightBytes', 'params', params);
  requirePositive('weightBytes', 'bitsPerParam', bitsPerParam);
  return (params * bitsPerParam) / 8;
}

// One step of the model on one GPU, as a roofline (prefill-decode §6).
// prefill: tokens = prompt, seqs = 0 · decode: tokens = seqs = users · spec-decode verify: tokens = users · (k + 1), seqs = users.
export function stepTime({
  activeParamsPerGpu, weightBytesPerGpu, dModel, actBytesPerElem, tokens, seqs, context, kvBytesPerToken, peakTflops, bandwidthTBps,
}) {
  requirePositive('stepTime', 'activeParamsPerGpu', activeParamsPerGpu);
  requireNonNegative('stepTime', 'weightBytesPerGpu', weightBytesPerGpu);
  requirePositive('stepTime', 'dModel', dModel);
  requirePositive('stepTime', 'actBytesPerElem', actBytesPerElem);
  requirePositive('stepTime', 'tokens', tokens);
  requireNonNegative('stepTime', 'seqs', seqs);
  requireNonNegative('stepTime', 'context', context);
  requireNonNegative('stepTime', 'kvBytesPerToken', kvBytesPerToken);
  requirePositive('stepTime', 'peakTflops', peakTflops);
  requirePositive('stepTime', 'bandwidthTBps', bandwidthTBps);
  const flops = FORWARD_FLOPS_PER_PARAM_TOKEN * activeParamsPerGpu * tokens;
  const actBytes = (FORWARD_FLOPS_PER_PARAM_TOKEN * activeParamsPerGpu * actBytesPerElem * tokens) / dModel;
  const kvBytes = seqs * context * kvBytesPerToken;
  const bytes = weightBytesPerGpu + actBytes + kvBytes;
  return { flops, bytes, actBytes, kvBytes, ...rooflineTime({ flops, bytes, peakTflops, bandwidthTBps }) };
}

// HBM left per GPU after the weights are spread over `gpus` (negative: the model does not fit).
export function freeHbmPerGpu({ hbmBytes, weightBytes: weights, gpus }) {
  requirePositive('freeHbmPerGpu', 'hbmBytes', hbmBytes);
  requireNonNegative('freeHbmPerGpu', 'weightBytes', weights);
  requirePositiveInt('freeHbmPerGpu', 'gpus', gpus);
  return hbmBytes - weights / gpus;
}

// Whole users whose caches fit in the free bytes; 0 (never negative) when nothing is free.
export function maxUsersPerGpu(freeBytes, cacheBytesPerUser) {
  requireFinite('maxUsersPerGpu', 'freeBytes', freeBytes);
  requirePositive('maxUsersPerGpu', 'cacheBytesPerUser', cacheBytesPerUser);
  return freeBytes <= 0 ? 0 : Math.floor(freeBytes / cacheBytesPerUser);
}

// Prefill throughput if the math were the only limit: peak / (2 · active) tokens per second (P4-R5).
export function prefillTokPerSecCeiling({ activeParamsPerGpu, peakTflops }) {
  requirePositive('prefillTokPerSecCeiling', 'activeParamsPerGpu', activeParamsPerGpu);
  requirePositive('prefillTokPerSecCeiling', 'peakTflops', peakTflops);
  return (peakTflops * 1e12) / (FORWARD_FLOPS_PER_PARAM_TOKEN * activeParamsPerGpu);
}

// Where one request's time goes (serving-overview §6). prefill is one stepTime (no second definition);
// the first token ends the prefill, then each further token takes one decode step.
export function requestTimeline({ queueS = 0, promptTokens, model, outputTokens, decodeTokPerS }) {
  requireNonNegative('requestTimeline', 'queueS', queueS);
  requirePositive('requestTimeline', 'promptTokens', promptTokens);
  requirePositiveInt('requestTimeline', 'outputTokens', outputTokens);
  requirePositive('requestTimeline', 'decodeTokPerS', decodeTokPerS);
  if (!model || typeof model !== 'object') throw new RangeError('requestTimeline: model must be a step-time model object');
  const prefillS = stepTime({ ...model, tokens: promptTokens, seqs: 0, context: 0 }).timeS;
  const ttftS = queueS + prefillS;
  const tpotS = 1 / decodeTokPerS;
  const decodeS = (outputTokens - 1) * tpotS;
  const e2eS = ttftS + decodeS;
  return { queueS, prefillS, ttftS, tpotS, e2eS, decodeShare: decodeS / e2eS };
}

// Seconds to move a prompt's KV cache over a link (disaggregation §6).
export function kvTransferTime(promptTokens, kvBytesPerToken, linkBytesPerSecond) {
  requirePositive('kvTransferTime', 'promptTokens', promptTokens);
  requirePositive('kvTransferTime', 'kvBytesPerToken', kvBytesPerToken);
  requirePositive('kvTransferTime', 'linkBytesPerSecond', linkBytesPerSecond);
  return (promptTokens * kvBytesPerToken) / linkBytesPerSecond;
}

// Tokens each expert sees per step when `epSize` GPUs each hold `usersPerGpu` users (disaggregation §6).
export function tokensPerExpert({ usersPerGpu, epSize, expertsActive, expertsTotal }) {
  requirePositive('tokensPerExpert', 'usersPerGpu', usersPerGpu);
  requirePositive('tokensPerExpert', 'epSize', epSize);
  requirePositive('tokensPerExpert', 'expertsActive', expertsActive);
  requirePositive('tokensPerExpert', 'expertsTotal', expertsTotal);
  return (usersPerGpu * epSize * expertsActive) / expertsTotal;
}

// Fewest GPUs whose HBM holds the weights, before any KV cache: ⌈weights / hbm⌉.
export function minGpusForWeights(weights, hbmBytes) {
  requirePositive('minGpusForWeights', 'weightBytes', weights);
  requirePositive('minGpusForWeights', 'hbmBytes', hbmBytes);
  return Math.ceil(weights / hbmBytes);
}

// Most users one GPU can serve while every user still gets `targetTokPerUser` (serving-calculator §6).
// Per user a decode step moves context · kvBytesPerToken of cache plus its activations; the step must fit in T = 1 / target.
// limit: the smallest of the compute cap, the bandwidth cap and the memory-capacity cap (`maxUsers`); a tie goes to
// compute, then bandwidth. A target no user can meet gives users 0.
export function usersAtTarget({
  targetTokPerUser, activeParamsPerGpu, weightBytesPerGpu, dModel, actBytesPerElem, context, kvBytesPerToken, peakTflops, bandwidthTBps, maxUsers,
}) {
  requirePositive('usersAtTarget', 'targetTokPerUser', targetTokPerUser);
  requirePositive('usersAtTarget', 'activeParamsPerGpu', activeParamsPerGpu);
  requireNonNegative('usersAtTarget', 'weightBytesPerGpu', weightBytesPerGpu);
  requirePositive('usersAtTarget', 'dModel', dModel);
  requirePositive('usersAtTarget', 'actBytesPerElem', actBytesPerElem);
  requireNonNegative('usersAtTarget', 'context', context);
  requireNonNegative('usersAtTarget', 'kvBytesPerToken', kvBytesPerToken);
  requirePositive('usersAtTarget', 'peakTflops', peakTflops);
  requirePositive('usersAtTarget', 'bandwidthTBps', bandwidthTBps);
  requireCount('usersAtTarget', 'maxUsers', maxUsers);
  const stepBudgetS = 1 / targetTokPerUser;
  const bytesPerUser = context * kvBytesPerToken + (FORWARD_FLOPS_PER_PARAM_TOKEN * activeParamsPerGpu * actBytesPerElem) / dModel;
  const byMem = Math.max(0, Math.floor((stepBudgetS * bandwidthTBps * 1e12 - weightBytesPerGpu) / bytesPerUser));
  const byCompute = Math.floor((stepBudgetS * peakTflops * 1e12) / (FORWARD_FLOPS_PER_PARAM_TOKEN * activeParamsPerGpu));
  const users = Math.min(byMem, byCompute, maxUsers);
  let limit = 'memory capacity';
  if (maxUsers >= Math.min(byMem, byCompute)) limit = byCompute <= byMem ? 'compute' : 'bandwidth';
  return { users, byMem, byCompute, maxUsers, limit };
}

// Dollars per million output tokens: $ / (tok/s · 3600) · 1e6 (the InferenceX convention).
export function costPerMillion(dollarsPerGpuHour, tokPerSecPerGpu) {
  requireNonNegative('costPerMillion', 'dollarsPerGpuHour', dollarsPerGpuHour);
  requirePositive('costPerMillion', 'tokPerSecPerGpu', tokPerSecPerGpu);
  return (dollarsPerGpuHour / (tokPerSecPerGpu * 3600)) * 1e6;
}
