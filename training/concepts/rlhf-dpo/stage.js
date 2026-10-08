// rlhf-dpo stage layout: timing helpers and small drawing helpers built on the glyph library. Every frame is a pure
// function of (parent, progress); no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { fixed } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const CHAR_W = 6.6; // the 11 px label size, in px per character
export const CHIP_H = 24; // the token glyph's height
export const CHIP_GAP = 4;
export const LABEL_X = 16; // the "A" / "B" letter beside a chip row
export const ROW_X = 34; // where a chip row starts

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
// Each frame starts from the previous frame's end state: the previous end state fades out during [0, HANDOFF].
export const HANDOFF = 0.15;
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);
export const typed = (text, t) => text.slice(0, Math.round(text.length * clamp01(t)));

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15); cls '' prints in ink instead of muted.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

export function select(parent, box, opacity = 1) {
  if (opacity > 0) G.selectionMark(layer(parent, opacity), box);
}

// A group a math term can name (data-link letter after "hl-"), with an invisible frame for the hover outline.
export function linked(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: w + 2, height: h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

export const rowWidth = (tokens) => tokens.reduce((sum, t) => sum + G.tokenWidth(t), 0) + CHIP_GAP * (tokens.length - 1);

// A row of token chips from (x, y). shown(i) is each chip's opacity; fill(i) its fill; hatchedLast hatches the last chip.
export function chipRow(parent, { x = ROW_X, y, tokens, shown = () => 1, fill = () => undefined, hatchedLast = false }) {
  let cx = x;
  tokens.forEach((text, i) => {
    const o = shown(i);
    if (o > 0) G.token(layer(parent, o), { x: cx, y, text, fill: fill(i), hatched: hatchedLast && i === tokens.length - 1 });
    cx += G.tokenWidth(text) + CHIP_GAP;
  });
}

// "A" or "B" beside a chip row, the row, and (when given) the math link letter around both.
export function answerRow(parent, { y, letter, tokens, link = null, shown }) {
  const box = { x: LABEL_X - 2, y: y - 2, w: ROW_X - LABEL_X + rowWidth(tokens) + 4, h: CHIP_H + 4 };
  const host = link ? linked(parent, link, box) : parent;
  note(host, LABEL_X, y + 16, letter, { cls: '' });
  chipRow(host, { y, tokens, shown });
  return box;
}

// The selection outline around an answer row (its letter and its chips).
export const rowMark = (parent, y, tokens, opacity = 1) => select(parent, { x: LABEL_X - 4, y: y - 3, w: ROW_X - LABEL_X + rowWidth(tokens) + 8, h: CHIP_H + 6 }, opacity);

// Arrows along a polyline: the dot travels the whole route as t goes 0 → 1 (each segment gets a flow, all same carry).
export function route(parent, points, { carry = 'token', t = 1 } = {}) {
  const lengths = points.slice(1).map((pt, i) => Math.hypot(pt[0] - points[i][0], pt[1] - points[i][1]));
  const total = lengths.reduce((s, l) => s + l, 0);
  let start = 0;
  lengths.forEach((len, i) => {
    const local = clamp01((t * total - start) / len);
    G.flow(parent, { from: points[i], to: points[i + 1], carry, progress: local });
    start += len;
  });
}

export const signed = (v, digits = 1) => `${v > 0 ? '+' : ''}${fixed(v, digits)}`;
