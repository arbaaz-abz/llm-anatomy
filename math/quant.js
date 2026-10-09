// Block quantization for the quantization lesson (storyboard §6). Pure: no DOM, inputs never mutated.
// Formats, bits per value and bytes per value are not defined here: FORMATS, bitsPerElement and bytesPerElement live in
// math/roofline.js (gpu-primer) and weightBytes in math/serving.js. This module only rounds a row of weights.

import { deepFreeze } from './core.js';

// E2M1 magnitudes (sign stored separately), from NVIDIA's NVFP4 post.
export const E2M1_GRID = deepFreeze([0, 0.5, 1, 1.5, 2, 3, 4, 6]);

const INT4_MAX = 7; // symmetric −7…7
const E2M1_MAX = 6;
const E2M1_EMAX = 2; // exponent of E2M1's largest value (6 = 1.5 · 2²), MX paper Algorithm 1
const E4M3_MAX = 448;
const E4M3_MIN_NORMAL = 2 ** -6;
const E4M3_MANTISSA_BITS = 3;
const RESTORED_DECIMALS = 4;
const FORMAT_NAMES = Object.freeze(['int4', 'mxfp4', 'nvfp4']);

const isNumber = (x) => typeof x === 'number' && Number.isFinite(x);
const noNegativeZero = (x) => (x === 0 ? 0 : x);
const roundTo = (x, decimals) => noNegativeZero(Number(x.toFixed(decimals)));

// Nearest grid magnitude with the sign kept; a tie goes toward zero (the number line drops ties the same way).
export function roundToGrid(x, grid) {
  if (!isNumber(x)) throw new RangeError(`roundToGrid: x must be a finite number, got ${x}`);
  if (!Array.isArray(grid) || grid.length === 0) throw new RangeError('roundToGrid: grid must be a non-empty array of magnitudes');
  const a = Math.abs(x);
  let best = grid[0];
  for (const g of grid) {
    const d = Math.abs(a - g);
    const bestD = Math.abs(a - best);
    if (d < bestD || (d === bestD && g < best)) best = g;
  }
  return noNegativeZero(Math.sign(x) * best);
}

// Round to an FP8 E4M3 value (3 mantissa bits). 0 stays 0; |x| below E4M3's normal range or above 448 is outside the
// range a block scale can take, so it throws.
export function roundE4M3(x) {
  if (!isNumber(x)) throw new RangeError(`roundE4M3: x must be a finite number, got ${x}`);
  if (x === 0) return 0;
  const a = Math.abs(x);
  if (a < E4M3_MIN_NORMAL) throw new RangeError(`roundE4M3: |x| must be at least 2^-6 (E4M3's normal range) or 0, got ${x}`);
  if (a > E4M3_MAX) throw new RangeError(`roundE4M3: |x| must be at most ${E4M3_MAX}, got ${x}`);
  const step = 2 ** (Math.floor(Math.log2(a)) - E4M3_MANTISSA_BITS);
  return Math.sign(x) * Math.round(a / step) * step;
}

// How each format turns a block's largest magnitude into a scale and a value into a code.
const FORMAT_RULES = Object.freeze({
  int4: {
    scale: (max) => max / INT4_MAX,
    code: (r) => roundToGrid(Math.max(-INT4_MAX, Math.min(INT4_MAX, r)), Array.from({ length: INT4_MAX + 1 }, (_, i) => i)),
    clipLimit: INT4_MAX + 0.5, // midway from 7 to a missing 8
  },
  mxfp4: {
    scale: (max) => 2 ** (Math.floor(Math.log2(max)) - E2M1_EMAX),
    code: (r) => roundToGrid(Math.max(-E2M1_MAX, Math.min(E2M1_MAX, r)), E2M1_GRID),
    clipLimit: E2M1_MAX + 1, // midway from 6 to a missing 8
  },
  nvfp4: {
    scale: (max) => roundE4M3(max / E2M1_MAX), // a labeled stand-in rule; the per-tensor scale is 1
    code: (r) => roundToGrid(Math.max(-E2M1_MAX, Math.min(E2M1_MAX, r)), E2M1_GRID),
    clipLimit: E2M1_MAX + 1,
  },
});

function quantizeBlock(block, rules) {
  const max = Math.max(...block.map(Math.abs));
  if (max === 0) return { scale: 0, codes: block.map(() => 0), restored: block.map(() => 0), clipped: 0 };
  const scale = rules.scale(max);
  const ratios = block.map((v) => v / scale);
  const codes = ratios.map(rules.code);
  return {
    scale,
    codes,
    restored: codes.map((c) => roundTo(c * scale, RESTORED_DECIMALS)),
    clipped: ratios.filter((r) => Math.abs(r) > rules.clipLimit).length,
  };
}

function validate(values, format, blockSize) {
  if (!Array.isArray(values) && !ArrayBuffer.isView(values)) throw new RangeError('quantizeBlocks: values must be a non-empty array of finite numbers');
  if (values.length === 0 || ![...values].every(isNumber)) throw new RangeError('quantizeBlocks: values must be a non-empty array of finite numbers');
  if (!FORMAT_NAMES.includes(format)) throw new RangeError(`quantizeBlocks: format must be one of ${FORMAT_NAMES.join(', ')}, got ${format}`);
  if (!Number.isInteger(blockSize) || blockSize < 1) throw new RangeError(`quantizeBlocks: blockSize must be a positive integer, got ${blockSize}`);
}

// Cut `values` into blocks of `blockSize` (the last may be shorter), give each block its own scale and round.
// zeroed: weights that were not 0 and came back as 0. clipped: weights past the grid's last rounding boundary.
export function quantizeBlocks(values, { format, blockSize }) {
  validate(values, format, blockSize);
  const rules = FORMAT_RULES[format];
  const row = [...values];
  const blocks = [];
  for (let i = 0; i < row.length; i += blockSize) blocks.push(quantizeBlock(row.slice(i, i + blockSize), rules));
  const restored = blocks.flatMap((b) => b.restored);
  const err = restored.map((r, i) => noNegativeZero(r - row[i]));
  return {
    blocks: blocks.map(({ scale, codes, restored: r }) => ({ scale, codes, restored: r })),
    restored,
    err,
    meanAbsErr: err.reduce((s, e) => s + Math.abs(e), 0) / row.length,
    zeroed: restored.filter((r, i) => r === 0 && row[i] !== 0).length,
    clipped: blocks.reduce((s, b) => s + b.clipped, 0),
  };
}
