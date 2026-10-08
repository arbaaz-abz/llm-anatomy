// agentic-rl pure formatters (no DOM): one formatter per quantity (README lesson 35), a real minus everywhere.
import { formatDuration } from '@math/core.js';
import { sharePct } from '@math/memory.js';

const realMinus = (s) => s.replace(/^-/, '−');

// Probabilities, ratios and weights: 3 decimals.
export const fixed3 = (v) => realMinus(Math.abs(v) < 0.0005 ? '0.000' : v.toFixed(3));
// A push: signed, 3 decimals; a masked token (even a negative zero) pushes exactly "0".
export const signed3 = (v) => (Math.abs(v) < 0.0005 ? '0' : `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`);
// The advantage in the inspector and the stage: a real minus, 2 decimals on the stage (formatCell) and 3 in the inspector.
export const advantage3 = (v) => realMinus(v.toFixed(3));

// busy ÷ (G × iteration), the page's one utilization definition: sharePct at 1 decimal.
export const percent1 = (part, whole) => `${sharePct(part, whole, { decimals: 1 }).toFixed(1)}%`;
export const minutes = (m) => formatDuration(m * 60);
export const ofTotal = (n, total) => `${n} of ${total}`;

// Row numbers (1-based on screen): [4, 7] → "rows 4 and 7"; [4, 5, 7, 8] → "rows 4, 5, 7 and 8"; [] → "none".
export function rowsText(rows) {
  if (rows.length === 0) return 'none';
  if (rows.length === 1) return `row ${rows[0]}`;
  return `rows ${rows.slice(0, -1).join(', ')} and ${rows.at(-1)}`;
}
export const carriedRows = (carried) => carried.map((c, i) => (c ? i + 1 : 0)).filter(Boolean);

// Stage numbers at 2 decimals (the clip line prints its marker the same way), with a real minus.
export const fixed2 = (v) => realMinus(v.toFixed(2));
// A push on the stage: 2 decimals, and a masked token's exact "0".
export const push2 = (v) => (Math.abs(v) < 0.005 ? '0' : fixed2(v));
// A line of stage text split at word boundaries into lines of at most `max` characters.
export function wrapText(text, max) {
  return text.split(' ').reduce((lines, word) => {
    const last = lines.at(-1);
    return last !== undefined && last.length + 1 + word.length <= max ? [...lines.slice(0, -1), `${last} ${word}`] : [...lines, word];
  }, []);
}
