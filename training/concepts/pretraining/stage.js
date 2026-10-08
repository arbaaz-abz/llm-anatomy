// pretraining stage layout: fixed positions shared by every frame, the timing helpers, and small drawing helpers built on
// the glyph library. The chip row keeps one anchor in frames 1–5 and the toy; the loss strip sits under its target chips.
import * as G from '@shared/glyphs.js';
import { TOKENS, LOSS_MAX_ABS } from './numbers.js';
import { formatLoss } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const SLOT = 46; // one token's column: its chip above, its loss cell below (chips are ≤ 43 px wide)
export const CHIPS = Object.freeze({ x: 130, y: 64 });
export const CHIP_H = 24; // the token glyph's height
export const STRIP = Object.freeze({ y: 118, pY: 110 }); // frames 4–5: loss cells, and the p printed above each
export const DIM = 0.4; // the opacity of a waiting chip (the token glyph's own dim opacity)
export const CHAR_W = 6.6; // JetBrains Mono at the 11 px label size

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
// Each frame starts from the previous frame's end state (template rule 4): what leaves fades out and what arrives fades in
// during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);
// The first n characters of a readout that types in (template rule 7: shown, never scaled up from 0).
export const typed = (text, t) => text.slice(0, Math.round(text.length * clamp01(t)));

export const slotX = (i) => CHIPS.x + i * SLOT;
export const slotCenter = (i) => slotX(i) + SLOT / 2;
export const chipX = (i) => slotCenter(i) - G.tokenWidth(TOKENS[i]) / 2;
export const cellX = (i) => slotX(i) + (SLOT - CELL) / 2;

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain); null when invisible.
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15), styled like glyph labels; cls '' prints in ink instead of muted.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// The eight token chips at their slots. opacity(i) fades a chip; follow marks chip indices with the selection outline.
export function chips(parent, { opacity = () => 1, follow = [], selOpacity = 1, fill = () => undefined } = {}) {
  TOKENS.forEach((text, i) => {
    const o = opacity(i);
    if (o > 0) G.token(layer(parent, o), { x: chipX(i), y: CHIPS.y, text, index: i + 1, fill: fill(i) });
  });
  follow.forEach((i) => {
    if (selOpacity > 0) G.selectionMark(layer(parent, selOpacity), { x: chipX(i), y: CHIPS.y, w: G.tokenWidth(TOKENS[i]), h: CHIP_H });
  });
}

// One loss cell (a one-cell vector at NUMBER_CELL, value-colored on the page's fixed loss scale, three decimals).
export function lossCell(parent, { x, y, loss, size = CELL }) {
  return G.vector(parent, { x, y, values: [loss], cell: size, orient: 'row', maxAbs: LOSS_MAX_ABS, format: formatLoss });
}

// A group that a math term can name: the wrapper carries data-link and an invisible frame for the hover outline.
export function linked(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: w + 2, height: h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

export const select = (parent, box, opacity = 1) => {
  if (opacity > 0) G.selectionMark(layer(parent, opacity), box);
};
