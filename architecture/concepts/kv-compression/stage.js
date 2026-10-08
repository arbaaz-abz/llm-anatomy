// The kv-compression stage kit: layout constants, timing helpers and the few drawing helpers every frame shares.
// Pure: nothing touches the DOM at import time. Positions are module constants; a frame moves only what its caption names.
import * as G from '@shared/glyphs.js';
import { weightText } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads (frame 4's heatmaps)

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// ---- layout: the query row, the stored heads under it, the text lines under both ----
export const Q = Object.freeze({ w: 60, h: 28, y: 52, x0: 29, pitch: 66 }); // 8 blocks, 8 × 66 = 528 px
export const QUERY_HEADS = 8;
export const qx = (h) => Q.x0 + h * Q.pitch;
export const qcx = (h) => qx(h) + Q.w / 2;
export const STACK = Object.freeze({ tile: 9, tiles: 4, y: 150, w: 42, h: 20 }); // a kvStack: 4 tokens, K above V
export const HEAD_CENTERS = Object.freeze(Array.from({ length: QUERY_HEADS }, (_, h) => qcx(h)));
export const GROUP_CENTERS = Object.freeze([(qcx(0) + qcx(3)) / 2, (qcx(4) + qcx(7)) / 2]); // centered under Q1–Q4 and Q5–Q8
export const SINGLE_CENTER = (qcx(0) + qcx(7)) / 2;
export const HEADER = Object.freeze({ x: 29, y: 20 });
export const READOUT = Object.freeze({ x: 551, y: 20 });
export const LINES = Object.freeze({ x: 29, y: [282, 304, 326, 348], share: 262 }); // the "Numbers shown" lines under the figure (share: frames 1–3)
export const REST_DOT = 0.62; // where a flow's dot rests: clear of both ends

// ---- drawing helpers ----
// A dim block's own opacity (.4) lives in the theme, so a fade goes on a wrapper group, never on the block.
export function dimBlock(svg, spec, opacity = 1) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', {}, svg);
  G.block(g, { ...spec, state: 'dim' });
  return fade(g, opacity);
}

export function fade(node, opacity) {
  if (opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
// cls 'g-label' is the muted face; any other class name keeps the ink color.
export function label(svg, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

// The stage's mono face at 11 px is about 6.6 px a character.
export const textWidth = (str) => Math.round(str.length * 6.6);
export const key = (svg, x, y, str, opts = {}) => label(svg, x, y, str, { ...opts, cls: 'g-key' });
export const select = (svg, x, y, w, h, opacity = 1) => fade(G.selectionMark(svg, { x, y, w, h }), opacity);

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines, then the glyph inside it (template rule 8).
export function linked(svg, letter, box) {
  const g = G.svgEl('g', { 'data-link': letter }, svg);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// A crossfade of two texts at one place: `from` fades out over [a, b], `to` fades in over [b, c].
export function swapKey(svg, x, y, from, to, p, [a, b, c], opts = {}) {
  if (from) key(svg, x, y, from, { ...opts, opacity: 1 - seg(p, a, b) });
  if (to) key(svg, x, y, to, { ...opts, opacity: seg(p, b, c) });
}

// A thin bracket under a group: a rule with two short ticks, drawn with the theme's line color.
export function bracket(svg, { x1, x2, y, opacity = 1 }) {
  const g = G.svgEl('g', { class: 'glyph g-note', 'aria-hidden': 'true' }, svg);
  G.svgEl('path', { d: `M ${x1} ${y - 5} V ${y} H ${x2} V ${y - 5}`, fill: 'none', style: 'stroke: var(--line); stroke-width: 1.5' }, g);
  return fade(g, opacity);
}

// ---- the query row and the stored heads ----
export function queryRow(svg, opacity = 1) {
  for (let h = 0; h < QUERY_HEADS; h += 1) fade(G.block(svg, { x: qx(h), y: Q.y, w: Q.w, h: Q.h, label: `Q${h + 1}` }), opacity);
  select(svg, qx(0), Q.y, Q.w, Q.h, opacity); // Q1, the head we follow, in every frame it is drawn
}

// One stored KV head: a kvStack of the four tokens' keys (top) and values (bottom), centered at `cx`.
export function kvHead(svg, cx, opacity = 1) {
  const x = cx - STACK.w / 2;
  const g = linked(svg, 'kvh', { x: x - 16, y: STACK.y - 2, w: STACK.w + 16, h: STACK.h + 4 });
  G.kvStack(g, { x, y: STACK.y, count: STACK.tiles, tile: STACK.tile });
  return fade(g, opacity);
}

// The wire from a stored head up to the query head that reads it.
export function wire(svg, { fromX, head, opacity = 1, progress = REST_DOT }) {
  if (opacity <= 0) return null;
  const g = G.flow(svg, { from: [fromX, STACK.y - 3], to: [qcx(head), Q.y + Q.h + 3], carry: 'kv', progress });
  return fade(g, opacity);
}

// ---- frame 4's heatmaps ----
// A cell not computed yet is null (blank, neutral fill); a masked weight is exactly 0, so it prints "0" and is hatched (excluded).
const glyphValues = (values) => values.map((v) => (v == null ? Number.NaN : v));
export function weightGrid(svg, { x, y, weights, mask, rowsShown = 4, rowLabels, colLabels }) {
  const values = weights.map((row, i) => glyphValues(row.map((w, j) => (i < rowsShown || !mask[i][j] ? w : null))));
  const hatch = mask.map((row) => row.map((visible) => !visible));
  return G.heatmap(svg, { x, y, values, hatch, cell: CELL, maxAbs: 1, rowLabels, colLabels, format: weightText });
}

// Draws `draw(group, 1)` (a finished frame) as a ghost that fades out: the way a frame leaves the one before it.
export function ghost(svg, draw, opacity) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', {}, svg);
  draw(g, 1);
  return fade(g, opacity);
}
