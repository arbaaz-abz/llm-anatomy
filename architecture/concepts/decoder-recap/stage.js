// decoder-recap stage kit: the layout every frame shares (two block stacks on the left, the evidence panel on the right),
// timing helpers and small drawing helpers built on the glyph library. Pure: nothing touches the DOM at import time.
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const STACK_W = 120;
export const LEFT_X = 2; // GPT-3 (2020)
export const RIGHT_X = 128; // 2026: keeps this place from frame 1 to frame 10
export const PANEL = Object.freeze({ x: 262, w: 314 }); // the frame's evidence: opened boxes, rows, readouts
export const ROWS_X = 254; // frame 2: eight NUMBER_CELL cells (320 px) right of the right stack

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
// Each frame starts from the previous frame's end state: what leaves fades out and what arrives fades in during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);
const bump = (t) => Math.sin(Math.PI * clamp01(t)); // 0 → 1 → 0

// ---- drawing helpers ----
// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15), styled like glyph labels. y is the text baseline.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// Several lines of one note, 14 px apart.
export function lines(parent, x, y, list, options = {}) {
  list.forEach((str, i) => note(parent, x, y + i * 14, str, options));
}

// A math-linked row of cells: the wrapper carries data-link and an invisible frame for the math panel's hover outline.
export function linkedRow(parent, letter, { x, y, values, cell = CELL, maxAbs, format, label }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: values.length * cell + 2, height: cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  G.vector(g, { x, y, values, cell, orient: 'row', maxAbs, format, label });
  return g;
}

// A row whose first `shown` cells have typed in (cells are never scaled up from 0, template rule 7).
export const typed = (values, t) => values.slice(0, Math.ceil(values.length * t));

// The selection outline around a rect, at the given opacity (0 draws nothing).
export function select(parent, rect, opacity = 1) {
  if (opacity <= 0) return null;
  return G.selectionMark(layer(parent, opacity), rect);
}

export const pulse = (p, from, to) => bump(seg(p, from, to));

// A text-only tile for the expert grid and kv groups: a block frame without a label.
export function tile(parent, { x, y, w, h, state = 'idle' }) {
  const g = G.svgEl('g', { class: `glyph g-block g-block--${state}`, transform: `translate(${x} ${y})` }, parent);
  G.svgEl('rect', { class: 'g-frame', width: w, height: h, rx: 2 }, g);
  return g;
}
