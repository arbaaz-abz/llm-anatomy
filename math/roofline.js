// The roofline: FLOPs, bytes and which of the two sets the time (gpu-primer §6). Pure: no DOM, inputs never mutated.
// Owned by gpu-primer; imported by prefill-decode, quantization, disaggregation, serving-calculator and math/serving.js.
// Units: TFLOPS and TB/s in, so peakTflops / bandwidthTBps is FLOPs per byte directly; times in seconds.
// FROZEN after Plan 3's shared prep (S3): change only through a shared patch.

import { deepFreeze } from './core.js';

// Keys name the exact format, never a bare 'fp8'. Layout is sign/exponent/mantissa. Block-scaled formats add one
// scale of `scaleBits` per `blockSize` numbers (NVFP4: an E4M3 scale; its per-tensor FP32 scale is ignored).
// tests/roofline.test.js pins these against data/hardware.json `formats` (ruling P3-R13).
export const FORMATS = deepFreeze({
  fp32: { bits: 32, layout: '1/8/23' },
  bf16: { bits: 16, layout: '1/8/7' },
  fp16: { bits: 16, layout: '1/5/10' },
  fp8_e4m3: { bits: 8, layout: '1/4/3' },
  fp8_e5m2: { bits: 8, layout: '1/5/2' },
  mxfp4: { bits: 4, layout: '1/2/1', blockSize: 32, scaleBits: 8 },
  nvfp4: { bits: 4, layout: '1/2/1', blockSize: 16, scaleBits: 8 },
});

const FORMAT_NAMES = Object.keys(FORMATS).join(', ');

function requirePositive(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${fn}: ${name} must be a positive finite number, got ${value}`);
  }
}

function requireNonNegative(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`${fn}: ${name} must be a finite number ≥ 0, got ${value}`);
  }
}

function requireFormat(fn, format) {
  if (typeof format !== 'string' || !Object.hasOwn(FORMATS, format)) {
    throw new RangeError(`${fn}: format must be one of ${FORMAT_NAMES} (never a bare 'fp8'), got ${format}`);
  }
}

function formatBits(format) {
  const { bits, blockSize, scaleBits } = FORMATS[format];
  return blockSize ? bits + scaleBits / blockSize : bits;
}

// Bits stored per number, scales included: 'bf16' → 16 · 'nvfp4' → 4.5 · 'mxfp4' → 4.25.
export function bitsPerElement(format) {
  requireFormat('bitsPerElement', format);
  return formatBits(format);
}

export function bytesPerElement(format) {
  requireFormat('bytesPerElement', format);
  return formatBits(format) / 8;
}

// X [m × k] · W [k × n] → Y [m × n]: 2mkn FLOPs; read X and W once, write Y once. m may be fractional
// (tokensToComputeBound returns the real crossing), so sizes are positive numbers, not integers.
export function matmulCost({ m, k, n, bytesPerElem }) {
  requirePositive('matmulCost', 'm', m);
  requirePositive('matmulCost', 'k', k);
  requirePositive('matmulCost', 'n', n);
  requirePositive('matmulCost', 'bytesPerElem', bytesPerElem);
  return { flops: 2 * m * k * n, bytes: bytesPerElem * (m * k + k * n + m * n) };
}

// An element-wise op: reads `inputs` tensors of `elements` numbers and writes one.
export function elementwiseCost({ elements, inputs = 2, flopsPerElem = 1, bytesPerElem }) {
  requirePositive('elementwiseCost', 'elements', elements);
  if (!Number.isInteger(inputs) || inputs < 0) throw new RangeError(`elementwiseCost: inputs must be an integer ≥ 0, got ${inputs}`);
  requireNonNegative('elementwiseCost', 'flopsPerElem', flopsPerElem);
  requirePositive('elementwiseCost', 'bytesPerElem', bytesPerElem);
  return { flops: elements * flopsPerElem, bytes: bytesPerElem * elements * (inputs + 1) };
}

// FLOPs per byte moved.
export function arithmeticIntensity({ flops, bytes }) {
  requireNonNegative('arithmeticIntensity', 'flops', flops);
  requirePositive('arithmeticIntensity', 'bytes', bytes);
  return flops / bytes;
}

// The intensity where the memory roof meets the compute roof, in FLOPs per byte.
export function ridgePoint({ peakTflops, bandwidthTBps }) {
  requirePositive('ridgePoint', 'peakTflops', peakTflops);
  requirePositive('ridgePoint', 'bandwidthTBps', bandwidthTBps);
  return peakTflops / bandwidthTBps;
}

// min(peak, intensity · bandwidth), in TFLOPS.
export function attainableTflops({ intensity, peakTflops, bandwidthTBps }) {
  requireNonNegative('attainableTflops', 'intensity', intensity);
  requirePositive('attainableTflops', 'peakTflops', peakTflops);
  requirePositive('attainableTflops', 'bandwidthTBps', bandwidthTBps);
  // Compare against the ridge rather than taking min(), so the roof is exactly the peak from the ridge on.
  return intensity >= peakTflops / bandwidthTBps ? peakTflops : intensity * bandwidthTBps;
}

// Time if the math and the memory traffic overlap perfectly: the slower one sets it. A tie counts as compute-bound,
// so the bound flips to 'compute' exactly at tokensToComputeBound.
export function rooflineTime({ flops, bytes, peakTflops, bandwidthTBps }) {
  requireNonNegative('rooflineTime', 'flops', flops);
  requireNonNegative('rooflineTime', 'bytes', bytes);
  requirePositive('rooflineTime', 'peakTflops', peakTflops);
  requirePositive('rooflineTime', 'bandwidthTBps', bandwidthTBps);
  const computeS = flops / (peakTflops * 1e12);
  const memoryS = bytes / (bandwidthTBps * 1e12);
  const bound = computeS >= memoryS ? 'compute' : 'memory';
  return { computeS, memoryS, timeS: Math.max(computeS, memoryS), bound };
}

// The token count m at which matmulCost({ m, k, n })'s intensity reaches the ridge (not rounded):
// m = R·b·k·n / (2kn − R·b·(k + n)). Infinity when the intensity can never get there (it tends to 2kn / (b(k + n))).
export function tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k, n }) {
  requirePositive('tokensToComputeBound', 'peakTflops', peakTflops);
  requirePositive('tokensToComputeBound', 'bandwidthTBps', bandwidthTBps);
  requirePositive('tokensToComputeBound', 'bytesPerElem', bytesPerElem);
  requirePositive('tokensToComputeBound', 'k', k);
  requirePositive('tokensToComputeBound', 'n', n);
  const rb = ridgePoint({ peakTflops, bandwidthTBps }) * bytesPerElem;
  const denominator = 2 * k * n - rb * (k + n);
  return denominator > 0 ? (rb * k * n) / denominator : Infinity;
}
