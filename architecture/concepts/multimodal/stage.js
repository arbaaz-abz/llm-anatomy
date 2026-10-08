// The multimodal stage's shared kit: layout constants, timing helpers and the small drawing helpers every frame uses.
// Pure: nothing touches the DOM at import time. Positions are module constants, never computed from progress unless
// the storyboard says the item moves.
import * as G from '@shared/glyphs.js';
import { PATCHES, FOLLOWED, GRID_N } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const SMALL = 18; // hover-only vectors (README visual constraints)

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF]
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

// ---- layout ----
export const GRID = Object.freeze({ x: 16, y: 40, size: 24, gapX: 8, gapY: 12 }); // the 4 × 4 patch grid, frames 1–5
export const Z = Object.freeze({ x: 408, y: 44, cell: 16 }); // the 16 × 8 patch vectors, frames 2–4
export const CENTER = Object.freeze({ x: 152, w: 176 }); // the patch-embedding block, frame 2
export const ENCODER = Object.freeze({ x: 152, w: 232 }); // the vision encoder column, frame 3
export const ROW = Object.freeze({ y0: 40, stride: 34, cell: SMALL }); // rows of the sequence, frames 4–6
export const MERGED = Object.freeze({ cell: 6, slotW: 48, rightX: 536 }); // merged 32-cell vectors, right edge at 536 in frame 4
export const MERGED_LEFT = 14; // frame 5: merged vectors slide here
export const PROJECTOR = Object.freeze({ x: 232, w: 132, y: ROW.y0 }); // frame 5
export const STREAM_X = 392; // the sequence column, frames 5–6
export const NOTE_Y = Object.freeze([330, 348]);

export const rowY = (i) => ROW.y0 + i * ROW.stride; // top of sequence row i
export const mergedX = (slot) => MERGED.rightX - 4 * MERGED.slotW + slot * MERGED.slotW;

// ---- drawing helpers ----
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return g;
}

// Several note lines, 18 px apart, fading in together.
export function noteLines(parent, x, y, lines, opacity = 1) {
  lines.forEach((line, i) => note(parent, x, y + i * 18, line, { opacity }));
}

// The one selection outline: the followed patch and the token it ends up in.
export const mark = (parent, x, y, w, h, opacity = 1) => {
  if (opacity <= 0) return null;
  const g = layer(parent, opacity);
  G.selectionMark(g, { x, y, w, h });
  return g;
};

// A neutral guide line (a plain mark, in the text color so both themes read it).
export function guide(parent, x1, y1, x2, y2, opacity = 0.4) {
  if (opacity <= 0 || (x1 === x2 && y1 === y2)) return null;
  return G.svgEl('line', { x1, y1, x2, y2, stroke: 'currentColor', 'stroke-opacity': opacity.toFixed(3), 'stroke-width': 1 }, parent);
}

// Wraps a glyph in a math-linked group: the invisible rect.g-frame the math panel outlines on hover.
export function linked(parent, letter, box) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// ---- the patch grid ----
export const patchX = (col, gap = 1) => GRID.x + col * (GRID.size + GRID.gapX * gap);
export const patchY = (row, gap = 1) => GRID.y + row * (GRID.size + GRID.gapY * gap);

// The 4 × 4 patches; `gap` 0 = the tiled image, 1 = separated. `numbers` is the opacity of the 1–16 labels.
export function patchGrid(parent, { gap = 1, numbers = 1, opacity = 1 } = {}) {
  const g = layer(parent, opacity);
  PATCHES.forEach((pixels, i) => {
    const [row, col] = [Math.floor(i / GRID_N), i % GRID_N];
    G.patch(g, { x: patchX(col, gap), y: patchY(row, gap), pixels, state: 'idle', size: GRID.size });
    note(g, patchX(col, gap) + GRID.size, patchY(row, gap) + GRID.size + 9, String(i + 1), { anchor: 'end', cls: 'g-sub', opacity: numbers });
  });
  mark(g, patchX(FOLLOWED % GRID_N, gap), patchY(Math.floor(FOLLOWED / GRID_N), gap), GRID.size, GRID.size);
  return g;
}

// Cross lines between the four 2 × 2 groups of the separated grid (frame 4), drawn to `t` of their length.
export function groupLines(parent, t, opacity = 1) {
  const midX = patchX(2) - GRID.gapX / 2;
  const midY = patchY(2) - GRID.gapY / 2;
  const [left, right] = [patchX(0), patchX(3) + GRID.size];
  const [top, bottom] = [patchY(0), patchY(3) + GRID.size];
  guide(parent, midX, top, midX, lerp(top, bottom, t), 0.5 * opacity);
  guide(parent, left, midY, lerp(left, right, t), midY, 0.5 * opacity);
}

// ---- vector rows ----
// Z or its encoded twin: 16 rows of 8, row i at Z.y + i · cell; `visible(i)` says which rows are drawn.
export function zMatrix(parent, values, { scale, label = 'patch vectors Z', visible = () => true, opacity = 1, link } = {}) {
  const g = layer(parent, opacity);
  const rows = values.map((row, i) => (visible(i) ? row : row.map(() => Number.NaN)));
  const host = link ? linked(g, link, { x: Z.x, y: Z.y, w: 8 * Z.cell, h: 16 * Z.cell }) : g;
  G.matrix(host, { x: Z.x, y: Z.y, values: rows, cell: Z.cell, maxAbs: scale, label });
  return g;
}
