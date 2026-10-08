// The stage's shared kit: layout, the toy patterns every frame draws (from math/longctx.js), timing helpers and the few
// drawing helpers the frames share. Pure: nothing touches the DOM at import time.
import * as G from '@shared/glyphs.js';
import { attentionPattern } from '@math/longctx.js';
import { TOKENS, WINDOW, TOP_K, MERGE, INDEXER } from './numbers.js';
import { formatFor } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = 18; // the pattern grid: read or not read, no numbers in cells (README: small grids are hover-only)
export const TILE = 13; // the 16-tile cache stack: 16 × 15 px must fit the right column
export const WIDE_TILE = 22; // a merged entry stands for 4 tokens
export const GRID = Object.freeze({ x: 30, y: 44 }); // 16 × 16 at 18 px: 288 px square
export const GRID_SIDE = TOKENS * CELL;
export const ROW_Y = (row) => GRID.y + row * CELL;
export const RIGHT = 334; // left edge of the right column (the grid ends at 318)
export const CHIP = Object.freeze({ x: RIGHT, y: 8 });
export const STACK = Object.freeze({ x: RIGHT, y: 66 });
export const COUNTER_Y = Object.freeze([150, 168, 186]);
export const NOTE_Y = 214; // free text area in the right column, below the counters
export const NOTE_STEP = 16;

// ---- the toy patterns (storyboard §6; every cell comes from attentionPattern) ----
const pattern = (spec) => attentionPattern({ n: TOKENS, ...spec });
export const PATTERNS = Object.freeze({
  full: pattern({ kind: 'full' }),
  window: pattern({ kind: 'window', window: WINDOW }),
  sparse: pattern({ kind: 'sparse', topK: TOP_K, scores: INDEXER }),
  compressed: pattern({ kind: 'compressed', window: WINDOW, topK: 1, merge: MERGE, scores: INDEXER }),
});
export const FOLLOWED_ROW = TOKENS - 1;
const rowKeys = (p) => p.mask[FOLLOWED_ROW].map((read, j) => (read ? j : -1)).filter((j) => j >= 0);
export const PICKS = Object.freeze(rowKeys(PATTERNS.sparse)); // 0-based: tokens 1, 5, 6, 13
export const countMask = (mask) => mask.reduce((n, row) => n + row.filter(Boolean).length, 0);

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// ---- drawing helpers ----
export function fade(node, opacity) {
  if (opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}

// A group for one scene; the frames fade a whole scene in or out at its edges.
export const scene = (svg, opacity = 1) => fade(G.svgEl('g', {}, svg), opacity);

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(parent, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

export function lines(parent, x, y, texts, opacity = 1) {
  texts.forEach((text, i) => label(parent, x, y + i * NOTE_STEP, text, { opacity }));
}

export const select = (parent, x, y, w, h, opacity = 1) => fade(G.selectionMark(parent, { x, y, w, h }), opacity);

// Glyph values: a cell not computed yet is null (blank, neutral fill).
export const glyphValues = (values) => values.map((v) => (v == null ? Number.NaN : v));

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines, then the glyph inside it.
export function linked(parent, letter, box) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// The followed token's chip, marked the same way in every frame it appears.
export function chip(parent, opacity = 1) {
  const node = G.token(parent, { x: CHIP.x, y: CHIP.y, text: 'token 16' });
  fade(node, opacity);
  select(parent, CHIP.x, CHIP.y, G.tokenWidth('token 16'), 24, opacity);
  label(parent, CHIP.x + 84, CHIP.y + 12, 'the newest token', { opacity });
}

// ---- the pattern grid: a read cell is a light tint (a state, never an amount); anything else is hatched ----
const READ_SCALE = 2.5; // a read cell is the value 1 on a 2.5 scale: a pale tint
const readText = (v) => (v === -Infinity ? 'not read' : 'read');
const INDEX_LABELS = Object.freeze(Array.from({ length: TOKENS }, (_, i) => String(i + 1)));

// `mask`: read cells (true) in a 16 × 16 matrix; the first `rows` rows are drawn, the rest stay blank until they are.
export function patternGrid(parent, { mask, rows = TOKENS, opacity = 1 }) {
  const drawn = (i) => i < rows;
  const values = mask.map((row, i) => row.map(() => (drawn(i) ? 1 : Number.NaN)));
  const visible = mask.map((row, i) => (drawn(i) ? row : row.map(() => true)));
  const holder = linked(parent, 'read', { x: GRID.x, y: GRID.y, w: GRID_SIDE, h: GRID_SIDE });
  const g = G.heatmap(holder, { x: GRID.x, y: GRID.y, values, mask: visible, cell: CELL, maxAbs: READ_SCALE, rowLabels: INDEX_LABELS, colLabels: INDEX_LABELS, format: readText });
  return fade(g, opacity);
}

export function gridAxes(parent, opacity = 1, row = FOLLOWED_ROW) {
  label(parent, 2, 14, 'queries ↓', { opacity });
  label(parent, GRID.x + GRID_SIDE, 14, 'keys →', { anchor: 'end', opacity });
  select(parent, GRID.x, ROW_Y(row), GRID_SIDE, CELL, opacity);
}

// A mask that moves from `from` to `to` as t goes 0 → 1: each cell flips at its own `order` in (0, 1], so the end is exact.
export function morph(from, to, t, order) {
  return from.map((row, i) => row.map((v, j) => (t >= order(i, j) ? to[i][j] : v)));
}

// ---- the cache stack: 16 tiles, with a veil over any tile that has left the cache or is not read ----
export function tileStack(parent, { veils, label: title = 'stored in this layer', opacity = 1 }) {
  const holder = linked(parent, 'store', { x: STACK.x, y: STACK.y, w: TOKENS * (TILE + 2), h: 2 * TILE + 2 });
  const g = fade(G.kvStack(holder, { x: STACK.x, y: STACK.y, count: TOKENS, tile: TILE, label: title }), opacity);
  const stride = TILE + 2;
  veils.forEach((level, i) => {
    if (level <= 0) return;
    const veil = G.svgEl('rect', { x: STACK.x + i * stride - 1, y: STACK.y - 1, width: TILE + 2, height: 2 * TILE + 4, rx: 3, 'aria-hidden': 'true' }, parent);
    veil.style.fill = 'var(--surface)';
    veil.style.opacity = String(Math.min(level, 1) * opacity);
  });
  return g;
}

export const visibleTiles = (veils) => veils.filter((v) => v < 1).length;

// The three counters of the followed token (the legend "reads" = compute, "stored" = memory).
export function counters(parent, { reads, stored, cells, readsNote = '', cellsNote = '', opacity = 1 }) {
  label(parent, RIGHT, COUNTER_Y[0], `reads (compute): ${reads} entries${readsNote}`, { opacity });
  label(parent, RIGHT, COUNTER_Y[1], `stored (memory): ${stored} entries`, { opacity });
  label(parent, RIGHT, COUNTER_Y[2], `cells read, all rows: ${cells}${cellsNote}`, { opacity });
}

// One row of NUMBER_CELL numbers; each cell prints its quantity's format (format.js cellText).
export function numberRow(parent, { x, y, values, kind, maxAbs, label: name, link, cell = G.NUMBER_CELL }) {
  const holder = link ? linked(parent, link, { x, y, w: values.length * cell, h: cell }) : parent;
  return G.vector(holder, { x, y, values: glyphValues(values), cell, orient: 'row', maxAbs, label: name, format: formatFor(kind) });
}
