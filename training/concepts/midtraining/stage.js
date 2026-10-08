// The midtraining stage's shared kit: layout constants, timing helpers, the learning-rate plot and the drawing helpers.
// Pure: nothing touches the DOM at import time.
import * as G from '@shared/glyphs.js';
import { lrCurve } from '@math/schedule.js';
import { CURVE_SAMPLES } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
// Frames 1-5 share one plot, so only what the frame changes moves.
export const PLOT = Object.freeze({ x: 4, y: 44, w: 556, h: 200 });
export const TEXT_Y = Object.freeze([262, 280, 298, 316, 334]); // the lines under the plot
export const TOP_Y = 16; // the line above the plot

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// ---- drawing helpers ----
export function fade(node, opacity) {
  if (opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(svg, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

// A plain text line, word-wrapped to `max` characters (one stage line holds about 76 of them); returns the lines drawn.
export function note(svg, x, y, str, { max = 76, lineHeight = 17, ...rest } = {}) {
  const lines = wrap(str, max);
  lines.forEach((line, i) => label(svg, x, y + i * lineHeight, line, rest));
  return lines.length;
}

export const select = (svg, box, opacity = 1) => fade(G.selectionMark(svg, box), opacity);

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines.
export function linked(svg, letter, box) {
  const g = G.svgEl('g', { 'data-link': letter }, svg);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// A plain line in the muted ink (marks that are not glyphs).
export function rule(svg, x1, y1, x2, y2, opacity = 1) {
  const line = G.svgEl('line', { x1, y1, x2, y2 }, svg);
  line.style.stroke = 'var(--ink-muted)';
  line.style.strokeWidth = '1.5';
  line.style.strokeLinecap = 'round';
  return fade(line, opacity);
}

// The previous frame's end state, fading out over the first part of this one (a layout change is never a hard cut).
export function outgoing(svg, drawPrevious, opacity) {
  if (opacity <= 0) return;
  const g = G.svgEl('g', {}, svg);
  drawPrevious(g, 1);
  fade(g, opacity);
}

// Greedy word wrap for the stage's plain text lines.
export function wrap(text, max) {
  return text.split(' ').reduce((lines, word) => {
    const last = lines[lines.length - 1];
    if (last !== undefined && `${last} ${word}`.length <= max) return [...lines.slice(0, -1), `${last} ${word}`];
    return [...lines, word];
  }, []);
}

// ---- the learning-rate plot: x = share of the run, y = fraction of the peak, both 0-1 ----
const pct = (v) => `${Math.round(v * 100)}%`;
const X_TICKS = Object.freeze([0, 0.25, 0.5, 0.75, 1].map((value) => ({ value, label: pct(value) })));
const Y_TICKS = Object.freeze([0, 0.25, 0.5, 0.75, 1].map((value) => ({ value, label: value === 0 || value === 1 ? String(value) : String(value) })));

export function plotSpec({ series = [], markers = [], bands = [], label: name = 'learning rate over the run' } = {}) {
  return {
    w: PLOT.w, h: PLOT.h,
    xAxis: { label: 'share of the run', domain: [0, 1], ticks: X_TICKS },
    yAxis: { label: 'learning rate (fraction of the peak)', domain: [0, 1], ticks: Y_TICKS },
    series, markers, bands, label: name,
  };
}

export const drawPlot = (svg, extra) => G.curvePlot(svg, { x: PLOT.x, y: PLOT.y, ...plotSpec(extra) });

// A band's box in stage coordinates (for the selection outline and math links), from the same layout the plot draws.
export function bandBox(bands, index) {
  const L = G.curvePlotLayout(plotSpec({ bands }));
  const b = L.bands[index];
  return { x: PLOT.x + b.x, y: PLOT.y + L.plot.top, w: b.width, h: L.plot.bottom - L.plot.top };
}

// lrAt sampled from 0 (or `from`) up to `upto` for a partly drawn curve; [] until there is something to draw.
export function partialCurve(options, upto, { from = 0, samples = CURVE_SAMPLES } = {}) {
  if (!(upto > from)) return [];
  return lrCurve(options, Math.max(2, Math.ceil(samples * (upto - from))), { from, to: upto });
}
