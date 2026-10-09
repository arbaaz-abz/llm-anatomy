// quantization stage layout: fixed positions shared by the block frames, the timing helpers and the drawing helpers on the glyph
// library. Every frame is a pure function of (index, progress): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL;
export const LEFT = 8; // the left text column's x
export const LINE = 16; // text line pitch
export const ROW_X = 200; // the weight row's x: eight cells of 40 px end at 520
export const ROW_W = 8 * CELL;
export const LABEL_X = ROW_X - 8; // row labels end here
export const NEUTRAL = 1e15; // a cell's value-scale maxAbs: codes stay neutral (the number is the reading)
export const MAX_WEIGHT = 2.1; // the value scale of weights and restored values
export const MAX_ERROR = 0.3; // the value scale of the error row
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF] (template rule 4)
// The block frames' rows (frames 1–6).
export const ROWS = Object.freeze({ chips: 6, weights: 36, line: 82, codes: 134, restored: 176, tags: 228, errors: 234, notes: 294 });

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15). cls 'g-label' is muted; '' prints in ink (a value the learner reads).
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start', pre = false } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor }, g);
  if (pre) t.style.whiteSpace = 'pre'; // aligned columns keep their spaces
  t.textContent = str;
  return g;
}

// Several lines from (x, y) down, one per LINE.
export function lines(parent, x, y, texts, { cls = '' } = {}) {
  texts.forEach((t, i) => note(parent, x, y + i * LINE, t, { cls }));
}

// A row label, right-aligned at the label column and centered on a row of cells.
export const rowLabel = (parent, y, str) => note(parent, LABEL_X, y + CELL / 2 + 4, str, { anchor: 'end' });

// A math-panel link target (template rule 8): <g data-link> with an invisible frame the hover outlines.
export function linkGroup(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// A plain 1 px rule in the muted ink (a block boundary).
export function rule(parent, x, y1, y2) {
  const line = G.svgEl('line', { class: 'glyph g-rule', x1: x, y1, x2: x, y2 }, parent);
  line.style.stroke = 'var(--ink-muted)';
  line.style.strokeWidth = '1';
  return line;
}

// Cross-fade between two states of one element: unchanged draws once; a new one fades in; a changed one fades the old out and
// the new in over the handoff (or over the `enter` / `exit` windows a frame gives it). `same` decides whether two states match.
export function fadePair(parent, prev, cur, p, draw, { same = (a, b) => JSON.stringify(a) === JSON.stringify(b), enter = arriving, exit = leaving } = {}) {
  if (cur == null && prev == null) return;
  if (cur != null && prev != null && same(prev, cur)) { draw(parent, cur); return; }
  if (prev != null && exit(p) > 0) draw(layer(parent, exit(p)), prev);
  if (cur != null && enter(p) > 0) draw(layer(parent, enter(p)), cur);
}
