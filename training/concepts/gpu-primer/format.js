// gpu-primer pure helpers (no DOM): the one analysis of "X [tokens × 8,192] · W [8,192 × 8,192] on a chip in a format",
// the number formats the page prints, and the "Check my work" text. Every number comes from math/roofline.js.
import { matmulCost, elementwiseCost, arithmeticIntensity, ridgePoint, attainableTflops, rooflineTime, tokensToComputeBound, bytesPerElement } from '@math/roofline.js';
import { sharePct } from '@math/memory.js';
import { D_REAL } from './numbers.js';

export const INITIAL_STATE = Object.freeze({ chip: 'h100', fmt: 'bf16', tokens: 4 });
export const TOKEN_STOPS = Object.freeze(Array.from({ length: 14 }, (_, i) => 2 ** i)); // 1 … 8,192, powers of 2
export const TOKEN_PRESETS = Object.freeze([4, 256, 4096]);

const GROUPED = Object.freeze({ int: { maximumFractionDigits: 0 }, one: { minimumFractionDigits: 1, maximumFractionDigits: 1 } });
const realMinus = (s) => s.replace(/^-/, '−');
// 536,870,912 · 295 (rounded) · a real minus.
export const int = (n) => realMinus(Math.round(n).toLocaleString('en-US', GROUPED.int));
// One decimal, grouped: 4.0 · 318.2 · 1,618.2.
export const fixed1 = (n) => realMinus(n.toLocaleString('en-US', GROUPED.one));
// "% of peak" at 2 decimals through sharePct (the one % definition): 1.35% · 21.34% · 100.00%.
export const pct2 = (part, whole) => `${sharePct(part, whole, { decimals: 2 }).toFixed(2)}%`;
// The same share as prose prints it, 3 significant figures: 1.35% · 21.3% · 81.6%.
export const pctProse = (part, whole) => `${Number(sharePct(part, whole, { decimals: 4 }).toPrecision(3))}%`;
// Several values (one per bandwidth end) as one string: "295" or "1,591–1,823", low → high, equal texts printed once.
export function rangeText(values, format) {
  const texts = [...values].sort((a, b) => a - b).map(format);
  return [...new Set(texts)].join('–');
}

// Everything the toy and the try-this list read for one state on one chip preset (hardware.js chipPreset).
// ends: one per published bandwidth (Rubin has two), fastest first.
export function analyze({ chip, fmt, tokens }, preset) {
  if (preset?.id !== chip) throw new RangeError(`analyze: the preset is ${preset?.id}, the state's chip is ${chip}`);
  const peakTflops = preset.peak[fmt];
  if (peakTflops == null) throw new RangeError(`analyze: ${chip} has no ${fmt} figure in data`);
  if (!TOKEN_STOPS.includes(tokens)) throw new RangeError(`analyze: tokens must be a power of 2 from 1 to 8,192, got ${tokens}`);
  const format = preset.formats[fmt];
  const bytesPerElem = bytesPerElement(format);
  const cost = matmulCost({ m: tokens, k: D_REAL, n: D_REAL, bytesPerElem });
  const intensity = arithmeticIntensity(cost);
  const ends = preset.bandwidths.map((bandwidthTBps) => {
    const roof = { peakTflops, bandwidthTBps };
    return {
      bandwidthTBps,
      ridge: ridgePoint(roof),
      crossing: tokensToComputeBound({ ...roof, bytesPerElem, k: D_REAL, n: D_REAL }),
      attainable: attainableTflops({ intensity, ...roof }),
      time: rooflineTime({ ...cost, ...roof }),
    };
  });
  const resid = elementwiseCost({ elements: tokens * D_REAL, bytesPerElem });
  return { tokens, format, bytesPerElem, peakTflops, cost, intensity, ends, residIntensity: arithmeticIntensity(resid) };
}

const BOUND_WORD = Object.freeze({ memory: 'memory-bound', compute: 'compute-bound' });
const tb = (bw) => `${bw} TB/s`;

// "memory-bound", or per bandwidth end when the two ends disagree (Rubin at 512 tokens).
export function verdictText(a) {
  const words = a.ends.map((e) => BOUND_WORD[e.time.bound]);
  if (new Set(words).size === 1) return words[0];
  return a.ends.map((e, i) => `${words[i]} at ${tb(e.bandwidthTBps)}`).join(', ');
}

const crossingText = (c) => (Number.isFinite(c) ? `${fixed1(c)} tokens` : 'never at this matrix size');

// The "Check my work" box (storyboard §6): five lines templated from matmulCost, arithmeticIntensity, ridgePoint and
// tokensToComputeBound; Rubin prints both bandwidth ends on the ridge and crossing lines.
export function checkWork(state, preset) {
  const a = analyze(state, preset);
  const m = int(a.tokens);
  const k = int(D_REAL);
  const b = String(a.bytesPerElem);
  const ridges = a.ends.map((e) => `${int(a.peakTflops)} ÷ ${e.bandwidthTBps} = ${fixed1(e.ridge)}`);
  const ridgeLine = a.ends.length === 1 ? `${ridges[0]} FLOPs/byte` : `${ridges[0]}, or ÷ ${a.ends[1].bandwidthTBps} = ${fixed1(a.ends[1].ridge)} FLOPs/byte`;
  const reaches = a.ends.map((e) => fixed1(e.ridge)).join(' or ');
  const crossings = a.ends.map((e) => crossingText(e.crossing).replace(/ tokens$/, '')).join(' or ');
  const unit = a.ends.every((e) => Number.isFinite(e.crossing)) ? ' tokens' : '';
  return [
    `FLOPs     = 2 × ${m} × ${k} × ${k} = ${int(a.cost.flops)}`,
    `bytes     = ${b} × (${m} × ${k} + ${k} × ${k} + ${m} × ${k}) = ${int(a.cost.bytes)}`,
    `intensity = ${int(a.cost.flops)} ÷ ${int(a.cost.bytes)} = ${fixed1(a.intensity)} FLOPs/byte`,
    `ridge     = ${ridgeLine} → ${verdictText(a)}`,
    `crossing  = the token count whose intensity reaches ${reaches} = ${crossings}${unit}`,
  ].join('\n');
}
