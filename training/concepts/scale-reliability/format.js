// scale-reliability pure formatters (no DOM): one formatter per quantity (README lesson 35). GPU-hours print through
// formatCount at 4 significant figures, durations through formatDuration, ratios through formatRatio, shares through sharePct.
import { formatCount, formatDuration } from '@math/core.js';
import { sharePct } from '@math/memory.js';

const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_DAY = 86400;
const SUPERSCRIPT = Object.freeze({ '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' });

export const int = (n) => (n < 0 ? '−' : '') + Math.abs(Math.round(n)).toLocaleString('en-US');

// 3.79 × 10²⁵ (a count of FLOPs in scientific notation, `sig` significant figures).
export function sci(x, sig = 3) {
  if (!(Number.isFinite(x) && x > 0)) throw new RangeError(`sci: x must be a finite number > 0, got ${x}`);
  const [mantissa, exponent] = x.toExponential(sig - 1).split('e');
  const power = String(Number(exponent)).split('').map((c) => SUPERSCRIPT[c]).join('');
  return `${Number(mantissa)} × 10${power}`;
}

export const gpuHours = (hours) => formatCount(hours, { digits: 4 });
export const hoursText = (hours) => formatDuration(hours * SECONDS_PER_HOUR);
export const daysText = (days) => formatDuration(days * SECONDS_PER_DAY);
export const minutesText = (minutes) => formatDuration(minutes * 60);
export const dollarsM = (dollars) => `$${(dollars / 1e6).toFixed(1)}M`;
export const dollarsExact = (dollars) => `$${formatCount(dollars, { digits: 4 })}`;
export const tflops = (v) => int(v);

// A fraction as a percentage with the given decimals, through sharePct (the course's one % definition).
export const pct = (fraction, decimals = 1) => `${sharePct(fraction, 1, { decimals }).toFixed(decimals)}%`;

// The share of a whole, e.g. pctOf(100, 140) = '71%'.
export const pctOf = (part, whole, decimals = 0) => `${sharePct(part, whole, { decimals }).toFixed(decimals)}%`;

// "38–43%" from a [low, high] fraction pair.
export const bandText = ([low, high]) => `${pct(low, 0).slice(0, -1)}–${pct(high, 0)}`;

// A signed percentage with a real minus: signedPct(-0.052) = '−5.2%'.
export const signedPct = (fraction, decimals = 1) => {
  const value = sharePct(fraction, 1, { decimals });
  return `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(decimals)}%`;
};
