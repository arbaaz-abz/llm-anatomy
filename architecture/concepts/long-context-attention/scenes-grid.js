// The grid scene of frames 1–7 and 8's fade-out: the followed chip, the 16 × 16 pattern, the cache stack and the counters,
// plus the sink row (frame 4) and the indexer row (frame 5). Every function draws into `parent` at an opacity, so a frame
// can cross-fade the previous frame's end state out while its own comes in.
import * as G from '@shared/glyphs.js';
import { INDEXER, DEFAULT_SINK_LOGIT, WINDOW, TOKENS } from './numbers.js';
import { windowScoresFor, sinkSplit, sumOf, trimNumber } from './format.js';
import {
  CELL, TILE, WIDE_TILE, GRID, GRID_SIDE, RIGHT, STACK, NOTE_Y, FOLLOWED_ROW, PICKS, PATTERNS, countMask, ease, seg, lerp,
  fade, scene, label, lines, select, chip, linked, patternGrid, gridAxes, tileStack, visibleTiles, counters, numberRow,
} from './stage.js';

const NO_VEIL = Object.freeze(Array(TOKENS).fill(0));
export const row16 = (mask) => mask[FOLLOWED_ROW].filter(Boolean).length;

// Tiles 1–12 slide out of a window layer, oldest first: `t` runs 0 → 1.
export const leavingVeils = (t) => Array.from({ length: TOKENS }, (_, i) => (i < TOKENS - WINDOW ? Math.min(Math.max(t * (TOKENS - WINDOW) - i, 0), 1) : 0));
// ... and come back, newest first.
export const returningVeils = (t) => Array.from({ length: TOKENS }, (_, i) => (i < TOKENS - WINDOW ? 1 - Math.min(Math.max(t * (TOKENS - WINDOW) - (TOKENS - WINDOW - 1 - i), 0), 1) : 0));
export const dimVeils = (t) => Array.from({ length: TOKENS }, (_, i) => (PICKS.includes(i) ? 0 : 0.65 * t));

// Pattern, 16-tile stack and counters; `stats` overrides the numbers a mask cannot give (entries read in a compressed layer).
export function gridView(parent, { mask, rows = TOKENS, veils = NO_VEIL, stats = {}, statsOpacity = 1, opacity = 1 }) {
  const holder = scene(parent, opacity);
  chip(holder);
  gridAxes(holder);
  patternGrid(holder, { mask, rows });
  tileStack(holder, { veils });
  counters(holder, { reads: row16(mask), stored: visibleTiles(veils), cells: countMask(mask), ...stats, opacity: statsOpacity });
  return holder;
}

export const END = Object.freeze({
  1: { mask: PATTERNS.full.mask, veils: NO_VEIL },
  2: { mask: PATTERNS.window.mask, veils: leavingVeils(1) },
  5: { mask: PATTERNS.sparse.mask, veils: NO_VEIL },
  6: { mask: PATTERNS.sparse.mask, veils: dimVeils(1) },
});

// ---- frame 4: the sink ----
const SINK_ROW = Object.freeze({ x: RIGHT, y: 236 });
const SINK_STRIDE = G.NUMBER_CELL + 3;
const WEIGHT_SCALE = 1;

// Frame 4's weights of the followed token over its window; `p` runs 0 → 1 (scores shown, sink slides in, weights redraw).
export function sinkBlock(parent, p, opacity = 1) {
  const holder = scene(parent, opacity);
  const scores = windowScoresFor(TOKENS, WINDOW);
  const { plain, sink, window } = sinkSplit(scores, DEFAULT_SINK_LOGIT);
  const slide = ease(seg(p, 0.35, 0.55));
  const redraw = ease(seg(p, 0.55, 1));
  const shown = seg(p, 0, 0.2);
  lines(holder, RIGHT, 206, [`window scores ${scores.map((s) => trimNumber(s)).join(', ')}`, 'all low: nothing here matters'], shown);
  const weights = plain.map((w, k) => lerp(w, window[k], redraw));
  numberRow(holder, { x: SINK_ROW.x + SINK_STRIDE, y: SINK_ROW.y, values: weights, kind: 'weight', maxAbs: WEIGHT_SCALE, label: undefined });
  if (slide > 0) {
    const sinkValue = slide < 1 ? null : sink * redraw;
    fade(numberRow(holder, { x: SINK_ROW.x - 20 * (1 - slide), y: SINK_ROW.y, values: [sinkValue], kind: 'weight', maxAbs: WEIGHT_SCALE, link: 'sink' }), slide);
  }
  scores.forEach((_, k) => label(holder, SINK_ROW.x + SINK_STRIDE * (k + 1) + 20, SINK_ROW.y + 52, String(TOKENS - WINDOW + 1 + k), { anchor: 'middle' }));
  label(holder, SINK_ROW.x + 20, SINK_ROW.y + 52, 'sink', { anchor: 'middle', opacity: slide });
  label(holder, SINK_ROW.x, SINK_ROW.y + 70, `Σ = ${sumOf([...weights, slide < 1 ? 0 : sink * redraw]).toFixed(3)}`);
  const sinkX = SINK_ROW.x - 20 * (1 - slide);
  const outlineX = lerp(SINK_ROW.x + SINK_STRIDE, sinkX, slide);
  select(holder, outlineX, SINK_ROW.y, SINK_ROW.x + 5 * SINK_STRIDE - 3 - outlineX, G.NUMBER_CELL);
  return holder;
}

// ---- frame 5: the indexer's scores for the followed token, and the four it picks ----
const INDEXER_Y = 346;
const MARKER_Y = 340;
const indexerValues = (t) => INDEXER[FOLLOWED_ROW].map((v, j) => (t * TOKENS >= j + 1 - 1e-9 ? v : null));

// `fill` types the 16 scores in (0 → 1); `picks` raises the four markers and the text (0 → 1).
export function indexerBlock(parent, { fill, picks, opacity = 1 }) {
  const holder = scene(parent, opacity);
  if (fill <= 0) return holder;
  const values = indexerValues(fill).map((v) => (v == null ? Number.NaN : v));
  G.vector(holder, { x: GRID.x, y: INDEXER_Y, values, cell: CELL, orient: 'row', maxAbs: 1, format: (v) => v.toFixed(2) });
  label(holder, GRID.x + GRID_SIDE + 8, INDEXER_Y + CELL / 2, 'indexer scores for token 16', { opacity: seg(fill, 0, 0.2) });
  PICKS.forEach((j) => label(holder, GRID.x + j * CELL + CELL / 2, MARKER_Y, '▲', { anchor: 'middle', opacity: picks }));
  lines(holder, RIGHT, NOTE_Y, ['indexer picks for token 16:', PICKS.map((j) => j + 1).join(', ')], picks);
  lines(holder, RIGHT, NOTE_Y + 40, ['the indexer still scores', 'every stored key, cheaply'], picks);
  return holder;
}

// ---- frame 7: merged entries ----
const MERGED_ENTRIES = PATTERNS.compressed.stored - Math.min(TOKENS, WINDOW);
const MERGED_STRIDE = WIDE_TILE + 2;
const WINDOW_STACK_X = STACK.x + MERGED_ENTRIES * MERGED_STRIDE + 36;

// 4 merged tiles (1 entry = 4 tokens) and the 4 window tiles; merged tiles 2–4 are not read by the followed token.
export function compressedStacks(parent, opacity = 1) {
  const holder = scene(parent, opacity);
  const store = linked(holder, 'store', { x: STACK.x, y: STACK.y, w: WINDOW_STACK_X - STACK.x + WINDOW * (TILE + 2), h: 2 * WIDE_TILE + 2 });
  G.kvStack(store, { x: STACK.x, y: STACK.y, count: MERGED_ENTRIES, tile: WIDE_TILE, label: '1 entry = 4 tokens' });
  G.kvStack(store, { x: WINDOW_STACK_X, y: STACK.y, count: WINDOW, tile: TILE, label: 'window' });
  for (let i = 1; i < MERGED_ENTRIES; i += 1) {
    const veil = G.svgEl('rect', { x: STACK.x + i * MERGED_STRIDE - 1, y: STACK.y - 1, width: WIDE_TILE + 2, height: 2 * WIDE_TILE + 4, rx: 3, 'aria-hidden': 'true' }, holder);
    veil.style.fill = 'var(--surface)';
    veil.style.opacity = '0.65';
  }
  return holder;
}

// One line over each group of 4 columns: those 4 keys are one entry.
export function groupBrackets(parent, opacity = 1) {
  const holder = scene(parent, opacity);
  for (let g = 0; g < MERGED_ENTRIES; g += 1) {
    const x1 = GRID.x + g * 4 * CELL + 2;
    const line = G.svgEl('line', { x1, y1: 27, x2: x1 + 4 * CELL - 4, y2: 27 }, holder);
    line.style.stroke = 'var(--ink-muted)';
  }
  return holder;
}

const COMPRESSED_STATS = Object.freeze({ reads: PATTERNS.compressed.readsPerRow[FOLLOWED_ROW], stored: PATTERNS.compressed.stored, cells: PATTERNS.compressed.cellsRead });

// Frame 7's end state: compressed pattern, merged stacks, brackets, counters (also frame 8's fade-out).
export function compressedScene(parent, opacity = 1) {
  const holder = scene(parent, opacity);
  chip(holder);
  gridAxes(holder);
  patternGrid(holder, { mask: PATTERNS.compressed.mask });
  groupBrackets(holder);
  compressedStacks(holder);
  counters(holder, { ...COMPRESSED_STATS, readsNote: ' (1 merged + 4)', cellsNote: ' token cells' });
  return holder;
}
