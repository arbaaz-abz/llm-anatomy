// Pure formatters and small derived numbers for the long-context-attention page. No DOM.
// One format per quantity (README lesson 7): counts with thousands separators, weights at 3 decimals, state and output cells exact up to 2.
import { softmax } from '@math/core.js';
import { WINDOW_SCORES_BY_OFFSET } from './numbers.js';

const MINUS = '−';
const realMinus = (s) => s.replace(/^-/, MINUS);

// 1048576 → "1,048,576".
export const int = (n) => realMinus(Math.round(n).toLocaleString('en-US'));

// The shortest exact form up to `digits` decimals ("−0.5", "1.25", "6"), real minus, never "−0".
export function trimNumber(v, digits = 2) {
  if (v === -Infinity) return `${MINUS}∞`;
  const s = v.toFixed(digits).replace(/\.?0+$/, '');
  return s === '-0' || s === '' ? '0' : realMinus(s);
}

const KINDS = Object.freeze({
  weight: (v) => v.toFixed(3),
  state: (v) => trimNumber(v, 2),
  output: (v) => trimNumber(v, 2),
});

// What a stage or toy cell prints, per quantity. A cell not computed yet (null, NaN) prints nothing.
export function cellText(v, kind) {
  const format = KINDS[kind];
  if (!format) throw new RangeError(`cellText: unknown kind "${kind}" (use ${Object.keys(KINDS).join(', ')})`);
  return v == null || Number.isNaN(v) ? '' : format(v);
}

export const formatFor = (kind) => (v) => cellText(v, kind);

// The window scores of the followed query (1-based position `query`), oldest first: tokens query − window + 1 … query.
export function windowScoresFor(query, window) {
  const first = Math.max(1, query - window + 1);
  return Array.from({ length: query - first + 1 }, (_, i) => WINDOW_SCORES_BY_OFFSET[query - (first + i)]);
}

// Softmax over the window scores, without and with one learned sink logit. `sink` is the weight the sink takes.
export function sinkSplit(scores, sinkLogit) {
  if (!Array.isArray(scores) || scores.length === 0) throw new RangeError('sinkSplit: scores must be a non-empty array');
  const withSink = softmax([sinkLogit, ...scores]);
  return { plain: softmax(scores), sink: withSink[0], window: withSink.slice(1) };
}

export const sumOf = (xs) => xs.reduce((acc, x) => acc + x, 0);
