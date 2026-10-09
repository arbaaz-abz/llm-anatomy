// quantization pure helpers (no DOM): the eight-weight row's views, the model-scale "shrink a model" analysis, the formatters
// and the "Check my work" text. Every number comes from math/quant.js, math/serving.js and math/roofline.js; one formatter
// per quantity (README lesson 35): scales at 4 decimals (2 at least), cell values at 2 decimals, errors at 3, durations through
// formatDuration, bytes through formatBytes, counts of users through formatInt.
import { bitsPerElement } from '@math/roofline.js';
import { weightBytes, stepTime, freeHbmPerGpu, maxUsersPerGpu, RUNNING_EXAMPLE } from '@math/serving.js';
import { kvCacheBytes } from '@math/memory.js';
import { quantizeBlocks, E2M1_GRID } from '@math/quant.js';
import { formatBytes, formatDuration, formatInt } from '@math/core.js';
import { WEIGHTS, OUTLIER, CALM, FOLLOWED, CONTEXT, PREFILL_TOKENS, SCALE_RULE } from './numbers.js';

export const INITIAL_STATE = Object.freeze({ format: 'int4', blockSize: 8, outlier: true, modelFormat: 'fp8', hw: 'h200', kv: 'bf16' });
export const FORMAT_LABEL = Object.freeze({ int4: 'INT4', mxfp4: 'MXFP4', nvfp4: 'NVFP4' });
export const DOES_NOT_FIT = 'does not fit';

// The toy's model-weight choices. `bits` is a FORMATS key (bitsPerElement counts the scales); `math` names the peak the
// arithmetic runs at; activations stay at least one byte (a 4-bit weight width is not an activation width). "4-bit weights,
// 16-bit math" is counted at NVFP4's 4.5 bits, so it differs from the NVFP4 chip only in where the math runs.
export const MODEL_FORMATS = Object.freeze({
  bf16: Object.freeze({ label: 'BF16', bits: 'bf16', math: 'bf16', actBytesPerElem: 2 }),
  fp8: Object.freeze({ label: 'FP8', bits: 'fp8_e4m3', math: 'fp8', actBytesPerElem: 1 }),
  w4a16: Object.freeze({ label: '4-bit weights, 16-bit math', bits: 'nvfp4', math: 'bf16', actBytesPerElem: 2 }),
  nvfp4: Object.freeze({ label: 'NVFP4', bits: 'nvfp4', math: 'fp4', actBytesPerElem: 1 }),
  mxfp4: Object.freeze({ label: 'MXFP4', bits: 'mxfp4', math: 'fp4', actBytesPerElem: 1 }),
});

const realMinus = (s) => s.replace(/^-/, '−');
const trimmed = (v) => (Object.is(v, -0) || v === 0 ? 0 : v);

// A weight, code-times-scale or error cell: whole numbers as they are, otherwise two decimals ("−0.31", "2.10", "0").
export const cellText = (v) => {
  const x = trimmed(v);
  const text = Number.isInteger(x) ? String(x) : x.toFixed(2);
  return realMinus(text === '-0.00' ? '0.00' : text); // a tiny negative error prints as 0.00, never "−0.00"
};
// An integer or E2M1 code: "−5", "1.5", "0.5".
export const codeText = (v) => realMinus(String(trimmed(v)));
// Fixed decimals with trailing zeros dropped down to two: a scale prints at 4 ("0.30", "0.0671", "0.3438", "0.0625"), a
// restored value in a note at 3 ("0.60", "0.375", "0.469", "2.10").
function trimmedFixed(v, decimals) {
  let t = trimmed(v).toFixed(decimals);
  while (t.endsWith('0') && /\.\d{3,}$/.test(t)) t = t.slice(0, -1);
  return realMinus(t);
}
export const scaleText = (s) => trimmedFixed(s, 4);
export const restoredText = (v) => trimmedFixed(v, 3);
// A mean error: three decimals ("0.064").
export const errorText = (e) => e.toFixed(3);

export const weightsFor = (outlier) => [...WEIGHTS.slice(0, -1), outlier ? OUTLIER : CALM];

// The eight-weight row quantized for a toy state, with the followed weight's block.
export function blockView({ format, blockSize, outlier }) {
  const weights = weightsFor(outlier);
  const result = quantizeBlocks(weights, { format, blockSize });
  const block = Math.floor(FOLLOWED / blockSize);
  return { weights, result, block, followedBlock: result.blocks[block] };
}

// The number line the weights land on: 15 integers for INT4, the 15 signed E2M1 values for FP4. The domain holds one grid step
// of room each side, so a dot that overshoots the last value (7.52 in front of 6) still sits on the line.
export const LINE_DOMAIN = Object.freeze({ lo: -8, hi: 8 });
export const INT4_GRID = Object.freeze(Array.from({ length: 2 * SCALE_RULE.int4 + 1 }, (_, i) => i - SCALE_RULE.int4));
export const E2M1_SIGNED = Object.freeze([...E2M1_GRID.slice(1).reverse().map((v) => -v), ...E2M1_GRID]);
export const gridFor = (format) => (format === 'int4' ? INT4_GRID : E2M1_SIGNED);

// Each weight's position on the line (value ÷ its block's scale) and the grid value it snaps to (its code), in row order.
export function lineSpec({ format, blockSize, outlier }) {
  const { weights, result } = blockView({ format, blockSize, outlier });
  const exact = weights.map((w, i) => w / result.blocks[Math.floor(i / blockSize)].scale);
  const snapped = result.blocks.flatMap((b) => b.codes);
  return { ...LINE_DOMAIN, grid: gridFor(format), exact, snapped };
}

// The "Check my work" box (storyboard §6): the followed weight's scale, code and restored value, templated per format.
export function checkWork(state) {
  const { weights, result, block, followedBlock } = blockView(state);
  const inBlock = weights.slice(block * state.blockSize, (block + 1) * state.blockSize);
  const max = cellText(Math.max(...inBlock.map(Math.abs)));
  const w = weights[FOLLOWED];
  const scale = followedBlock.scale;
  const code = followedBlock.codes[FOLLOWED - block * state.blockSize];
  const restored = result.restored[FOLLOWED];
  const ratio = (w / scale).toFixed(2);
  const rules = {
    int4: [`scale = max|w| / ${SCALE_RULE.int4} = ${max} / ${SCALE_RULE.int4} = ${scaleText(scale)}`,
      `code = round(w / scale) = round(${cellText(w)} / ${scaleText(scale)}) = ${codeText(code)}`],
    mxfp4: [`scale = 2^(⌊log2 max|w|⌋ − ${SCALE_RULE.mxOffset}) = 2^(${realMinus(String(Math.floor(Math.log2(Number(max)))))} − ${SCALE_RULE.mxOffset}) = ${scaleText(scale)}`,
      `code = nearest FP4 value to w / scale = ${cellText(w)} / ${scaleText(scale)} = ${ratio}${Math.abs(w / scale) > SCALE_RULE.e2m1 ? ' (past 6, clamped to 6)' : ''} → ${codeText(code)}`],
    nvfp4: [`scale = max|w| / ${SCALE_RULE.e2m1}, rounded to FP8 = ${max} / ${SCALE_RULE.e2m1} → ${scaleText(scale)}`,
      `code = nearest FP4 value to w / scale = ${cellText(w)} / ${scaleText(scale)} = ${ratio} → ${codeText(code)}`],
  };
  return [...rules[state.format], `restored = code × scale = ${codeText(code)} × ${scaleText(scale)} = ${restoredText(restored)}`].join('\n');
}

// Memory only: weights, what is left, the cache of one user and how many users fit. bits is bits per weight with scales.
export function memoryPlan({ bits, hbmBytes, kvBytesPerToken }) {
  const weights = weightBytes({ params: RUNNING_EXAMPLE.activeParamsPerGpu, bitsPerParam: bits });
  const freeBytes = freeHbmPerGpu({ hbmBytes, weightBytes: weights, gpus: 1 });
  const kvPerUser = kvCacheBytes({ bytesPerToken: kvBytesPerToken, tokens: CONTEXT });
  return { bits, weights, freeBytes, fits: freeBytes >= 0, kvPerUser, users: maxUsersPerGpu(freeBytes, kvPerUser) };
}

export const kvBytesFor = (kv) => RUNNING_EXAMPLE.kvBytesPerToken * (kv === 'fp8' ? 0.5 : 1);

// "Shrink a model" on one chip: memory plus the decode step (1 user, 2,048 context) and the prefill (4,096 tokens), each one
// stepTime. Throws when the chip has no peak for the format's math (the toy disables that choice first).
export function shrink({ modelFormat, kv }, preset) {
  const f = MODEL_FORMATS[modelFormat];
  if (!f) throw new RangeError(`shrink: modelFormat must be one of ${Object.keys(MODEL_FORMATS).join(', ')}, got ${modelFormat}`);
  const peakTflops = preset.peak[f.math];
  if (peakTflops == null) throw new RangeError(`shrink: ${preset.id} has no ${f.math} figure in data`);
  const kvBytesPerToken = kvBytesFor(kv);
  const memory = memoryPlan({ bits: bitsPerElement(f.bits), hbmBytes: preset.hbmBytes, kvBytesPerToken });
  const model = { ...RUNNING_EXAMPLE, weightBytesPerGpu: memory.weights, actBytesPerElem: f.actBytesPerElem, peakTflops, bandwidthTBps: preset.bandwidthTBps, kvBytesPerToken };
  return {
    ...memory,
    decode: stepTime({ ...model, tokens: 1, seqs: 1, context: CONTEXT }),
    prefill: stepTime({ ...model, tokens: PREFILL_TOKENS, seqs: 0, context: 0 }),
  };
}

// Bits per weight with scales, as the formats count them: 16, 8, 4.5, 4.25 (stage and toy print the same).
export const bitsText = (bits) => String(bits);
export const freeText = (m) => (m.fits ? formatBytes(m.freeBytes) : DOES_NOT_FIT);
export const usersText = (m) => formatInt(m.users);
export const timeText = (step) => formatDuration(step.timeS);
export const memoryText = (preset) => `${formatBytes(preset.hbmBytes)} ${preset.basis}`;
