// The scaling-laws stage kit: layout, timing helpers, text marks, and the curve every curve frame (3-7) shares.
// Pure: nothing touches the DOM at import time. Numbers come from math/scaling.js, math/scale.js and numbers.js.
import * as G from '@shared/glyphs.js';
import { isoFlopLoss } from '@math/scaling.js';
import { STAGE_BUDGET } from './numbers.js';
import { sizeText, powerText } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const PLOT = Object.freeze({ x: 4, y: 4, w: 372, h: 290 }); // the whole curvePlot figure, axis labels inside
export const COLUMN = Object.freeze({ x: 392, y: 22, lineH: 16, chars: 27 }); // the right column beside the plot (frames 3, 4, 6, 7)
export const UNDER = Object.freeze({ x: 8, y: 322, lineH: 17 }); // full-width notes under the plot

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
export const logLerp = (a, b, t) => 10 ** lerp(Math.log10(a), Math.log10(b), t);

// ---- drawing helpers ----
export function fade(node, opacity) {
  if (opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(svg, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

// Body-ink text (a readout, not a caption-grey label).
export const ink = (svg, x, y, str, opts = {}) => label(svg, x, y, str, { ...opts, cls: 'g-readout' });

export const select = (svg, x, y, w, h, opacity = 1) => fade(G.selectionMark(svg, { x, y, w, h }), opacity);

// An invisible rect.g-frame the math-panel hover outlines, then the glyph(s) inside it.
export function linked(svg, letter, box) {
  const g = G.svgEl('g', { 'data-link': letter }, svg);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// Greedy word wrap by character count (the stage font is monospace, so characters are columns).
export function wrap(str, chars) {
  return str.split(' ').reduce((lines, word) => {
    const last = lines[lines.length - 1];
    if (last !== undefined && last.length + 1 + word.length <= chars) return [...lines.slice(0, -1), `${last} ${word}`];
    return [...lines, word];
  }, []);
}

// Lines of text from (x, y), each `lineH` apart; a line is a string or { text, ink, opacity }. Returns the y after the last line.
export function lines(svg, x, y, items, { lineH = COLUMN.lineH, opacity = 1 } = {}) {
  items.forEach((item, i) => {
    const { text, bold = false, opacity: own = 1 } = typeof item === 'string' ? { text: item } : item;
    (bold ? ink : label)(svg, x, y + i * lineH, text, { opacity: opacity * own });
  });
  return y + items.length * lineH;
}

// "9.70 × 10²⁴" counting up: the mantissa counts, the exponent holds (only the final value is exact).
export function countedSci(final, t, formatted) {
  if (t >= 1) return formatted;
  const [mantissa, rest] = formatted.split(' × ');
  return `${(Number(mantissa) * t).toFixed(2)} × ${rest}`;
}

// ---- the shared curve (frames 3-7): fitted loss at STAGE_BUDGET against model size, x log 1B-1T, y 1.94-2.20 ----
export const X_DOMAIN = Object.freeze([1e9, 1e12]);
export const Y_DOMAIN = Object.freeze([1.92, 2.2]); // wider than the storyboard's 1.94 so the optimum's label has room below it
const X_TICKS = Object.freeze([1e9, 1e10, 1e11, 1e12].map((value) => ({ value, label: sizeText(value) })));
const Y_TICKS = Object.freeze([1.95, 2, 2.05, 2.1, 2.15, 2.2].map((value) => ({ value, label: value.toFixed(2) })));
const CURVE_SEGMENTS = 60;

export const lossAt = (N) => isoFlopLoss(STAGE_BUDGET, N).loss;

// The curve drawn from the left up to fraction t of its log-x length.
export function curvePoints(t) {
  const total = CURVE_SEGMENTS * clamp01(t);
  const whole = Math.floor(total);
  const xs = Array.from({ length: whole + 1 }, (_, i) => logLerp(X_DOMAIN[0], X_DOMAIN[1], i / CURVE_SEGMENTS));
  const tail = total > whole ? [logLerp(X_DOMAIN[0], X_DOMAIN[1], total / CURVE_SEGMENTS)] : [];
  return [...xs, ...tail].map((x) => [x, lossAt(x)]);
}

// A marker on the curve (or at a given loss); followed ones draw their own selection mark inside curvePlot.
export const marker = (x, text, { followed = false, y = lossAt(x) } = {}) => ({ x, y, label: text, followed });

export function drawCurve(svg, { drawn = 1, markers = [], refY = null, seriesLabel = null }) {
  const points = curvePoints(drawn);
  return G.curvePlot(svg, {
    ...PLOT,
    xAxis: { label: 'active parameters N', log: true, domain: [...X_DOMAIN], ticks: X_TICKS },
    yAxis: { label: `fitted loss at ${powerText(STAGE_BUDGET)} FLOPs`, domain: [...Y_DOMAIN], ticks: Y_TICKS },
    series: points.length >= 2 ? [{ points, label: seriesLabel ?? '', labelAt: 'end' }] : [],
    markers,
    refY,
    label: 'Fitted loss against model size at a fixed budget',
  });
}
