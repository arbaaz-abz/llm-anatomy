// kv-cache stage kit: fixed layout, timing helpers and the few drawing helpers every frame shares. Pure: nothing touches
// the DOM at import time. Every helper takes the <svg> or a <g> inside it, so a frame can draw the previous scene at a
// lower opacity while it hands over (template rule 4).
import * as G from '@shared/glyphs.js';
import { K_ROWS, V_ROWS, ROW_LABELS, MAX_ABS, ON_ROW, TOKENS } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const GRID = 20; // K and V: values on hover only (the followed row is lifted out at NUMBER_CELL)
const CHAR_W = 6.6; // one mono character at the 11 px label size

// ---- layout, frames 1–5 (K and V keep their place from frame 1 to frame 5; the strips sit below them) ----
export const TOKEN_Y = 8;
export const CHIP_X = Object.freeze([50, 94, 138, 182]);
export const ON_X = 233;
export const ON_FROM = 520; // "on₅" slides in from the right
export const W_BLOCK = Object.freeze({ x: 50, y: 56, w: 213, h: 24 });
export const MAT_Y = 104;
export const MAT_X = Object.freeze({ k: 50, v: 316 });
export const LANE = Object.freeze({ x: 134, keyEnd: 190, valueStart: 312, valueEnd: 266, center: 228 });
export const BRANCH = Object.freeze({ x: 142, y: 94 });
export const STRIP = Object.freeze({ k: { x: 50, y: 240 }, v: { x: 276, y: 240 }, labelY: 230 });
export const RIGHT = Object.freeze({ x: 412, end: 576, y: 104 });
export const KV_STACK = Object.freeze({ x: 50, y: 316 });
export const NOTE_Y = 355;

// ---- timing ----
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
// Each frame starts from the previous frame's end state: what leaves fades out and what arrives fades in during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

// ---- drawing helpers ----
export function fade(node, opacity) {
  if (opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(parent, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

// Several lines of plain text, `gap` px apart; `cls` '' prints in ink, 'g-label' in the muted face.
export function textBlock(parent, x, y, lines, { cls = 'g-label', gap = 16, opacity = 1 } = {}) {
  lines.forEach((line, i) => label(parent, x, y + i * gap, line, { cls, opacity }));
}

export const select = (parent, x, y, w, h, opacity = 1) => (opacity > 0 ? fade(G.selectionMark(parent, { x, y, w, h }), opacity) : null);

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines, then whatever `draw` puts inside.
export function linked(parent, letter, box, draw) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  draw(g);
  return g;
}

// A linked line of text: the frame is sized from the character count.
export function linkedText(parent, letter, { x, y, str, anchor = 'start', cls = 'g-label', opacity = 1 }) {
  if (opacity <= 0) return null;
  const w = str.length * CHAR_W;
  const left = anchor === 'end' ? x - w : x;
  const g = linked(parent, letter, { x: left - 2, y: y - 9, w: w + 4, h: 18 }, (inner) => label(inner, x, y, str, { anchor, cls }));
  return fade(g, opacity);
}

// What a K or V cell prints (hover only at 20 px; at NUMBER_CELL the strips print it): the shortest exact form, real minus.
export const cellText = (v) => (v == null || Number.isNaN(v) ? '' : String(v).replace('-', '−'));
const blank = (row) => row.map(() => Number.NaN);

// ---- the token chips ----
export function promptChips(parent, opacity = 1) {
  TOKENS.forEach((t, i) => fade(G.token(parent, { x: CHIP_X[i], y: TOKEN_Y, text: t, index: i + 1 }), opacity));
}

// "on₅", the followed token: slides in from the right (slide 0 → 1); the selection mark travels with it.
export function onChip(parent, { slide = 1, opacity = 1 } = {}) {
  const x = lerp(ON_FROM, ON_X, slide);
  fade(G.token(parent, { x, y: TOKEN_Y, text: 'on', index: 5 }), opacity);
  select(parent, x, TOKEN_Y, G.tokenWidth('on'), 24, opacity);
}

export function wBlock(parent, opacity = 1) {
  return fade(G.block(parent, { ...W_BLOCK, label: 'W_Q, W_K, W_V' }), opacity);
}

export const chipCenters = Object.freeze([...CHIP_X.map((x, i) => x + G.tokenWidth(TOKENS[i]) / 2), ON_X + G.tokenWidth('on') / 2]);

// One activation dot per chip, running down into the W block.
export function chipFlows(parent, count, progress, opacity = 1) {
  if (progress <= 0 || opacity <= 0) return;
  const g = layer(parent, opacity);
  chipCenters.slice(0, count).forEach((cx) => G.flow(g, { from: [cx, TOKEN_Y + 26], to: [cx, W_BLOCK.y - 2], carry: 'activation', progress }));
}

// ---- K and V ----
// rows: rows drawn; solid: leading rows filled in; fading: [{ row, opacity }] rows drawn on top at their own opacity
// (a row that is arriving or leaving); follow: opacity of the selection mark on "on" (row 5), 0 hides it.
export function kvMatrices(parent, { rows, solid, fading = [], follow = 1 }) {
  const names = [['k', K_ROWS], ['v', V_ROWS]];
  names.forEach(([name, source]) => {
    const values = source.slice(0, rows).map((row, i) => (i < solid ? row : blank(row)));
    const g = G.matrix(parent, {
      x: MAT_X[name], y: MAT_Y, values, cell: GRID, maxAbs: MAX_ABS, label: name.toUpperCase(), format: cellText,
      rowLabels: name === 'k' ? ROW_LABELS.slice(0, rows) : [],
    });
    g.dataset.link = name;
    fading.forEach(({ row, opacity }) => fade(G.vector(parent, { x: MAT_X[name], y: MAT_Y + row * GRID, values: source[row], cell: GRID, orient: 'row', maxAbs: MAX_ABS, format: cellText }), opacity));
    select(parent, MAT_X[name], MAT_Y + ON_ROW * GRID, 4 * GRID, GRID, follow);
  });
}

// ---- the lifted k_on / v_on strips (frames 2 and 3) ----
export const typeIn = (row, t) => row.map((v, j) => (t * row.length >= j + 1 - 1e-9 ? v : Number.NaN));

// lift 0 → 1 moves each strip from below the matrices into row 5 (cell 40 → 20); fill types the numbers in left to right.
export function strips(parent, { fill = 1, lift = 0, opacity = 1 }) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  [['k', K_ROWS[ON_ROW], 'k_on'], ['v', V_ROWS[ON_ROW], 'v_on']].forEach(([name, row, text]) => {
    const cell = lerp(CELL, GRID, lift);
    const x = lerp(STRIP[name].x, MAT_X[name], lift);
    const y = lerp(STRIP[name].y, MAT_Y + ON_ROW * GRID, lift);
    linked(g, name, { x, y, w: 4 * cell, h: cell }, (host) => G.vector(host, { x, y, values: typeIn(row, fill), cell, orient: 'row', maxAbs: MAX_ABS, format: cellText }));
    label(g, STRIP[name].x, STRIP.labelY, text, { opacity: 1 - lift });
    select(g, x, y, 4 * cell, cell);
  });
}

// ---- the KV cache stack (frames 3–7): "stored rows", linked to the math term hl-cache ----
export function cacheStackAt(parent, { x, y, count, highlight = [], opacity = 1, title }) {
  const box = { x: x - 14, y: y - 22, w: Math.max(count, 5) * 16 + 24, h: 54 };
  const g = linked(parent, 'cache', box, (host) => G.kvStack(host, { x, y, count, highlight, label: title }));
  fade(g, opacity);
}

export const cacheStack = (parent, opts) => cacheStackAt(parent, { ...KV_STACK, title: 'KV cache', ...opts });

// Draw the previous frame's last picture while it fades out over [0, HANDOFF]; nothing once it is gone.
export function handOff(parent, p, drawPrevious) {
  if (p < HANDOFF) drawPrevious(layer(parent, leaving(p)), 1);
}

// ---- a counter in the right zone: muted label left, value right, optional sub-line ----
export function counter(parent, y, name, value, sub, opacity = 1) {
  if (opacity <= 0) return;
  label(parent, RIGHT.x, y, name, { opacity });
  label(parent, RIGHT.end, y, String(value), { anchor: 'end', cls: '', opacity });
  if (sub) label(parent, RIGHT.x, y + 16, sub, { opacity });
}
