// rlvr-grpo's pure formatters and the toy's opening state. No DOM. Real minus and a leading zero everywhere (README).
import { EPS_HIGH_PPO, DEFAULT_K } from './numbers.js';

const MINUS = '−';

// The toy opens where the animation ends: k = 2 of 8 right, std division on, sample-level loss, PPO clip, row 1's `56`.
// `row` and `token` are 0-based indexes; the page prints 1-based positions.
export const INITIAL_STATE = Object.freeze({ k: DEFAULT_K, norm: true, agg: 'sample', epsHigh: EPS_HIGH_PPO, row: 0, token: 4 });

const isZeroAt = (v, digits) => Math.abs(v) < 0.5 * 10 ** -digits;

// Fixed decimals with a real minus; a value that rounds to zero prints "0.000", never "−0.000".
export function fixed(v, digits) {
  if (!Number.isFinite(v)) throw new RangeError(`fixed: v must be finite, got ${v}`);
  if (isZeroAt(v, digits)) return (0).toFixed(digits);
  return v.toFixed(digits).replace('-', MINUS);
}

// Signed: a plus on a positive value that does not round to zero ("+1.73", "−0.58", "0.00").
export function signed(v, digits) {
  const text = fixed(v, digits);
  return v > 0 && !isZeroAt(v, digits) ? `+${text}` : text;
}

export const fmt2 = (v) => fixed(v, 2);
export const fmt3 = (v) => fixed(v, 3);
export const fmt4 = (v) => fixed(v, 4);
export const signed2 = (v) => signed(v, 2);
export const signed4 = (v) => signed(v, 4);

// A stage cell counting between two real states: integers print bare ("1", "0"); anything else as signed 2 d.p.
export const cellNumber = (v) => (Number.isInteger(v) ? String(v) : signed2(v));

export const yesNo = (flag) => (flag ? 'yes' : 'no');
