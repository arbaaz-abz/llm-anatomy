// cluster-topology's pure formatters, the toy's state and its "Check my work" text (storyboard §6). No DOM.
import { sharePct } from '@math/memory.js';
import { ringAllReduceBytes } from '@math/parallel.js';
import { FLOPS_PER_PARAM_TOKEN } from '@math/scale.js';
import { tpCommRatio, ppCommRatio, dpCommRatio, epCommRatio, EP_HIDING_FLOPS_PER_BYTE } from '@math/topology.js';
import { GPT3, SYSTEMS } from './numbers.js';

const MINUS = '−';
const MB = 1e6;
const GFLOP = 1e9;

export const CUTS = Object.freeze(['tensor', 'pipeline', 'data', 'expert']);
export const DEGREE_STOPS = Object.freeze({
  tensor: Object.freeze([2, 4, 8, 16, 32, 64]),
  pipeline: Object.freeze([2, 4, 8, 16, 32]),
  data: Object.freeze([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
});
export const TOKEN_STOPS = Object.freeze([4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]);

// Where the toy opens: tensor parallelism, degree 8, on an H100 HGX server's NVLink (storyboard §6).
export const INITIAL_STATE = Object.freeze({ system: 'h100', cut: 'tensor', degree: 8, where: 'inside', tokens: 262144 });

export const int = (n) => Math.round(n).toLocaleString('en-US').replace(/^-/, MINUS);
export const plainNumber = (n, digits = 2) => n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
// Comm as a percentage of compute: sharePct is the course's one % definition; "2,253.6%" keeps its separator.
export const pct1 = (ratio) => `${sharePct(ratio, 1).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
export const pct0 = (ratio) => `${int(sharePct(ratio, 1, { decimals: 0 }))}%`;
export const mbText = (bytes) => `${plainNumber(bytes / MB)} MB`;
export const gflopText = (flops) => `${plainNumber(flops / GFLOP, 1)} GFLOP`;
export const plural = (n, one, many = `${one}s`) => `${int(n)} ${n === 1 ? one : many}`;

export const linkFor = (system, where) => (where === 'inside' ? system.nvlinkGBps : system.networkGBps);
export const fitsInside = (system, cut, degree) => cut === 'expert' || degree <= system.domain;

// One layer, one full step, per GPU: 4 ring all-reduces of seq × batch × hidden BF16 numbers vs 72 · s · b · h² ÷ t FLOPs.
export function tensorLayerWork(degree) {
  const { seq, microBatch, hidden, bytesPerNumber } = GPT3;
  const size = seq * microBatch * hidden * bytesPerNumber;
  return { size, bytes: 4 * ringAllReduceBytes(size, degree), flops: (72 * seq * microBatch * hidden * hidden) / degree };
}

// The ratio of the state's cut on `sys` ({ peakTflops, nvlinkGBps, networkGBps }), from math/topology.js.
export function ratioFor({ cut, degree, where, tokens }, sys) {
  const base = { peakTflops: sys.peakTflops, linkGBps: linkFor(sys, where) };
  if (cut === 'tensor') return tpCommRatio({ tp: degree, hidden: GPT3.hidden, ...base });
  if (cut === 'pipeline') return ppCommRatio({ pp: degree, hidden: GPT3.hidden, layers: GPT3.layers, ...base });
  if (cut === 'data') return dpCommRatio({ dp: degree, tokensPerReplica: tokens, ...base });
  return epCommRatio(base);
}

const tensorCheck = (state, sys, link) => {
  const { bytes, flops } = tensorLayerWork(state.degree);
  const { seq, hidden } = GPT3;
  return [
    `bytes per GPU, one layer, full step = 4 × ring all-reduce of ${int(seq)} × ${int(hidden)} × 2 B over ${int(state.degree)} GPUs = ${mbText(bytes)}`,
    `FLOPs per GPU, one layer, full step = 72 × ${int(seq)} × ${int(hidden)}² ÷ ${int(state.degree)} = ${gflopText(flops)}`,
    `comm ÷ compute = (${mbText(bytes)} ÷ ${int(link)} GB/s) ÷ (${gflopText(flops)} ÷ ${int(sys.peakTflops)} TFLOPS) = ${pct1(ratioFor(state, sys))}`,
  ].join('\n');
};

const pipelineCheck = (state, sys, link) => {
  const { seq, hidden, layers } = GPT3;
  const bytes = 2 * seq * hidden * 2;
  const flops = (72 * seq * hidden * hidden * layers) / state.degree;
  return [
    `bytes per boundary, one micro-batch, full step = 2 × ${int(seq)} × ${int(hidden)} × 2 B = ${mbText(bytes)}`,
    `FLOPs per stage, one micro-batch, full step = 72 × ${int(seq)} × ${int(hidden)}² × (${int(layers)} ÷ ${int(state.degree)}) = ${gflopText(flops)}`,
    `comm ÷ compute = (${mbText(bytes)} ÷ ${int(link)} GB/s) ÷ (${gflopText(flops)} ÷ ${int(sys.peakTflops)} TFLOPS) = ${pct1(ratioFor(state, sys))}`,
  ].join('\n');
};

const dataCheck = (state, sys, link) => {
  const bytes = ringAllReduceBytes(2, state.degree);
  const flops = FLOPS_PER_PARAM_TOKEN * state.tokens;
  return [
    `bytes per parameter, one step = ring all-reduce of 2 B over ${int(state.degree)} replicas = ${plainNumber(bytes, 4)} B`,
    `FLOPs per parameter, one step = ${FLOPS_PER_PARAM_TOKEN} × ${int(state.tokens)} tokens = ${int(flops)}`,
    `comm ÷ compute = (${plainNumber(bytes, 4)} B ÷ ${int(link)} GB/s) ÷ (${int(flops)} FLOPs ÷ ${int(sys.peakTflops)} TFLOPS) = ${pct1(ratioFor(state, sys))}`,
  ].join('\n');
};

const expertCheck = (state, sys, link) => {
  const perByte = (sys.peakTflops * 1e12) / (link * 1e9);
  return [
    `compute ÷ link = ${int(sys.peakTflops)} TFLOPS ÷ ${int(link)} GB/s = ${plainNumber(perByte, 1)} FLOPs per byte`,
    `DeepSeek-V4 hides expert traffic at ${int(EP_HIDING_FLOPS_PER_BYTE)} FLOPs per byte or fewer: ${plainNumber(perByte, 1)} ÷ ${int(EP_HIDING_FLOPS_PER_BYTE)} = ${pct1(ratioFor(state, sys))}`,
  ].join('\n');
};

// "Check my work" for any state: the cut's bytes and FLOPs per GPU, then the same ratio math/topology.js returns.
export function checkWork(state, sys = SYSTEMS[state.system]) {
  const link = linkFor(sys, state.where);
  const build = { tensor: tensorCheck, pipeline: pipelineCheck, data: dataCheck, expert: expertCheck }[state.cut];
  return build(state, sys, link);
}
