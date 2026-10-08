// Frames 1, 2, 4, 5, 6 and 7 of storyboard §5: the 16 × 16 pattern of the followed token's layer. Each frame is a pure function of
// its progress p (0 → 1); the end of frame n is the start of frame n + 1 (frame 3 shows the layer stack in between).
import { TOKENS } from './numbers.js';
import {
  PATTERNS, RIGHT, FOLLOWED_ROW, countMask, seg, ease, morph, lines, scene, chip, gridAxes, patternGrid, tileStack, counters,
} from './stage.js';
import {
  END, gridView, row16, leavingVeils, returningVeils, dimVeils, sinkBlock, indexerBlock, compressedStacks, groupBrackets, COMPRESSED_NOTES,
} from './scenes-grid.js';
import { sixLayers } from './scenes-layers.js';

const FULL = PATTERNS.full.mask;
const WINDOW_MASK = PATTERNS.window.mask;
const SPARSE = PATTERNS.sparse.mask;
const COMPRESSED = PATTERNS.compressed.mask;
const NO_VEIL = Object.freeze(Array(TOKENS).fill(0));
const distanceBack = (i, j) => i - j;
const PROMPT_Y = 306;

// ---- frame 1: the full causal grid fills row by row, the stack grows to 16 ----
export function drawFrame1(svg, p) {
  const rows = Math.floor(p * TOKENS + 1e-9);
  const veils = NO_VEIL.map((_, i) => (i < rows ? 0 : 1));
  gridView(svg, { mask: FULL, rows, veils, stats: { stored: PATTERNS.full.stored }, statsOpacity: seg(p, 0.85, 1) });
}

// ---- frame 2: a band of width 4; the 12 oldest tiles leave the stack ----
export function drawFrame2(svg, p) {
  const t = ease(p);
  const mask = morph(FULL, WINDOW_MASK, t, (i, j) => (TOKENS - distanceBack(i, j)) / (TOKENS - 4));
  gridView(svg, { mask, veils: leavingVeils(t) });
}

// ---- frame 4: frame 3's layer stack cross-fades into the window grid; the sink slides in ----
export function drawFrame4(svg, p) {
  const come = seg(p, 0, 0.3);
  if (come < 1) sixLayers(svg, 1, 1 - come);
  gridView(svg, { ...END[2], opacity: come });
  sinkBlock(svg, seg(p, 0.3, 1), come);
}

// ---- frame 5: the full grid returns, the indexer scores every key, the grid keeps each row's top 4 ----
export function drawFrame5(svg, p) {
  const back = ease(seg(p, 0, 0.25));
  const sparse = ease(seg(p, 0.7, 1));
  const restored = morph(WINDOW_MASK, FULL, back, (i, j) => (distanceBack(i, j) - 3) / (TOKENS - 4));
  const mask = sparse > 0 ? morph(FULL, SPARSE, sparse, (i) => (TOKENS - i) / TOKENS) : restored;
  gridView(svg, { mask, veils: sparse > 0 ? NO_VEIL : returningVeils(back) });
  if (back < 1) sinkBlock(svg, 1, 1 - back);
  indexerBlock(svg, { fill: seg(p, 0.25, 0.55), picks: seg(p, 0.55, 0.7) });
}

// ---- frame 6: reading fewer keys does not shrink the cache ----
export function drawFrame6(svg, p) {
  gridView(svg, { ...END[5], veils: dimVeils(ease(seg(p, 0.35, 0.8))) });
  indexerBlock(svg, { fill: 1, picks: 1 });
  lines(svg, RIGHT, PROMPT_Y, ['Did the cache shrink?'], 1 - seg(p, 0.25, 0.35));
  lines(svg, RIGHT, PROMPT_Y, ['stored: 16.', 'The indexer needs them all.'], seg(p, 0.75, 0.9));
}

// ---- frame 7: merging every 4 tokens ----
export function drawFrame7(svg, p) {
  const out = 1 - seg(p, 0, 0.15);
  const merged = seg(p, 0.35, 0.6);
  const holder = scene(svg);
  chip(holder);
  gridAxes(holder);
  patternGrid(holder, { mask: morph(SPARSE, COMPRESSED, ease(seg(p, 0.1, 0.6)), (i) => (TOKENS - i) / TOKENS) });
  if (merged < 1) tileStack(holder, { veils: dimVeils(1), opacity: 1 - merged });
  if (merged > 0) compressedStacks(holder, merged);
  groupBrackets(holder, seg(p, 0.5, 0.7));
  const compressed = { reads: PATTERNS.compressed.readsPerRow[FOLLOWED_ROW], stored: PATTERNS.compressed.stored, cells: PATTERNS.compressed.cellsRead, notes: COMPRESSED_NOTES };
  counters(holder, p >= 0.6 ? compressed : { reads: row16(SPARSE), stored: TOKENS, cells: countMask(SPARSE) });
  if (out > 0) {
    indexerBlock(svg, { fill: 1, picks: 1, opacity: out });
    lines(svg, RIGHT, PROMPT_Y, ['stored: 16.', 'The indexer needs them all.'], out);
  }
}
