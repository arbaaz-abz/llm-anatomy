// moe pure helpers (no DOM): the toy's configuration, number formats, expert names and the combination counts.
// Every number comes from math/moe.js, math/params.js or math/memory.js; this file only arranges and prints them.
import { PRESETS } from '@math/params.js';
import { sharePct } from '@math/memory.js';

export const TOY_LIMITS = Object.freeze({ routed: [0, 2, 4, 8, 16, 32], split: [1, 2, 4], gamma: [0, 0.05, 0.1, 0.2], step: [0, 9] });
export const INITIAL_STATE = Object.freeze({ real: 'toy', routed: 8, split: 1, shared: false, gamma: 0.1, step: 0 });
export const BATCH = Object.freeze({ tokens: 128, experts: 8, k: 2, seed: 100, popularity: Object.freeze([1, 0.5, 0, 0, 0, 0, 0, 0]) });

const TOY_TOP_K = 2; // the animation's top-k; fine-grained splits pick `split` times as many
const TOY_HIDDEN = 8; // the animation's expert hidden size
const SUPERSCRIPT = Object.freeze({ 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' });
const EXACT_BELOW = 1e15; // combination counts print exactly up to here, then as a power of ten

export const int = (n) => (n < 0 ? '−' : '') + Math.abs(n).toLocaleString('en-US');
export const expertName = (index) => `E${index + 1}`;
export const picksText = (picks) => picks.map(expertName).join(', ');
export const gateText = (w) => w.toFixed(3);
export const gatesText = (weights) => weights.map(gateText).join(' · ');
export const imbalanceText = (x) => x.toFixed(2);
// Bias with two decimals and a real minus; a −0.00 from rounding prints as 0.00.
export const biasText = (b) => { const t = b.toFixed(2); return t === '-0.00' ? '0.00' : t.replace('-', '−'); };

// Score cells: whole numbers print one decimal ("2.0", "−1.0"), zero prints "0", everything else its shortest exact form.
export function scoreText(v) {
  if (v === 0) return '0';
  const text = Number.isInteger(v) ? v.toFixed(1) : String(v);
  return text.replace('-', '−');
}

// The toy's mixture of experts for the sliders. `routed` is the slider (0 = the dense MLP of hidden 16); `split` cuts each
// expert into `split` pieces of hidden 8 / split and picks `split` times as many; `shared` adds one expert of the routed size
// and takes one of the top-k slots, so the experts a token runs stay 2 · split and the work per token stays 384.
export function toyMoe({ routed, split, shared }) {
  if (!TOY_LIMITS.routed.includes(routed)) throw new RangeError(`toyMoe: routed must be one of ${TOY_LIMITS.routed.join(', ')}, got ${routed}`);
  if (!TOY_LIMITS.split.includes(split)) throw new RangeError(`toyMoe: split must be one of ${TOY_LIMITS.split.join(', ')}, got ${split}`);
  if (routed === 0) return { dense: true, config: PRESETS.toy, experts: 0, topK: 0, shared: 0, hidden: PRESETS.toy.mlp.hidden };
  const experts = routed * split;
  const sharedCount = shared ? 1 : 0;
  const topK = Math.min(TOY_TOP_K * split - sharedCount, experts);
  const hidden = TOY_HIDDEN / split;
  const moe = { routed: experts, shared: sharedCount, topK, hidden, denseLayers: 0 };
  return { dense: false, config: { ...PRESETS.toy, moe }, experts, topK, shared: sharedCount, hidden };
}

const MANTISSA_DIGITS = 2;

// "28", "1,820", "10,518,300": exact below 10^15, then "≈ 1.4 × 10³⁵" (the exponent from the digit count).
export function combosText(count) {
  if (typeof count !== 'bigint') throw new TypeError('combosText: count must be a BigInt');
  if (count < BigInt(EXACT_BELOW)) return Number(count).toLocaleString('en-US');
  return combosApprox(count);
}

// Always scientific: "≈ 4.3 × 10¹²" (two significant figures; 9.96 × 10¹² rounds up to 1.0 × 10¹³).
export function combosApprox(count) {
  const digits = count.toString();
  let exponent = digits.length - 1;
  let mantissa = Number(`${digits[0]}.${digits.slice(1, 1 + MANTISSA_DIGITS + 1)}`).toPrecision(MANTISSA_DIGITS);
  if (Number(mantissa) >= 10) { mantissa = '1.0'; exponent += 1; }
  return `≈ ${mantissa} × 10${[...String(exponent)].map((d) => SUPERSCRIPT[d]).join('')}`;
}

// "6 of 384" and its share "1.6%", from a routed-expert pair (data.models experts_active / experts_total).
export const routedText = (active, total) => `${int(active)} of ${int(total)}`;
export const routedShare = (active, total) => `${sharePct(active, total).toFixed(1)}%`;
