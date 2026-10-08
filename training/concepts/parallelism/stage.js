// The parallelism stage's shared kit: layout constants, timing helpers and the few drawing helpers every frame uses.
// Pure: nothing touches the DOM at import time. Numbers come from numbers.js / format.js, never from here.
import * as G from '@shared/glyphs.js';
import { TOKENS, SAT } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CHAR_W = 6.6; // the stage's 11 px mono face
export const COUNTER_Y = 352;
export const RIGHT = 572;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

export function fade(node, opacity) {
  if (opacity < 1) node.style.opacity = String(Math.max(opacity, 0));
  return node;
}

// A plain text mark inside a .glyph group (the stage's one label face and size). cls '' = ink, 'g-label' = muted.
export function label(svg, x, y, str, { anchor = 'start', cls = '', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

export const select = (svg, x, y, w, h, opacity = 1) => fade(G.selectionMark(svg, { x, y, w, h }), opacity);

// A math-linked group: an invisible-until-hovered rect.g-frame (the math panel outlines it), then the content.
export function linked(svg, name, box) {
  const g = G.svgEl('g', { 'data-link': name }, svg);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 3, y: box.y - 3, width: box.w + 6, height: box.h + 6, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// The "sent per GPU" counter, bottom right in every frame (hl-comm in the math panel).
export function counter(svg, text) {
  const w = Math.ceil(text.length * CHAR_W);
  const g = linked(svg, 'comm', { x: RIGHT - w, y: COUNTER_Y - 7, w, h: 14 });
  label(g, RIGHT, COUNTER_Y, text, { anchor: 'end' });
}

// The four token chips of "The cat sat down" at (x, y); "sat" gets the selection mark. Returns each chip's box.
const GAP = 4;
export function chipBoxes(x) {
  let at = x;
  return TOKENS.map((t) => {
    const box = { x: at, w: G.tokenWidth(t) };
    at += box.w + GAP;
    return box;
  });
}
export const chipsWidth = (x = 0) => { const b = chipBoxes(x); return b[3].x + b[3].w - x; };

export function chipRow(svg, x, y, { opacity = 1, tokens = TOKENS, mark = true } = {}) {
  const boxes = chipBoxes(x);
  tokens.forEach((t, i) => { if (t != null) fade(G.token(svg, { x: boxes[i].x, y, text: t, index: i + 1 }), opacity); });
  if (mark && tokens[SAT] != null) select(svg, boxes[SAT].x, y, boxes[SAT].w, 24, opacity);
  return boxes;
}

// A solid 1 px --ink-muted line (a cut line or a rule; never dashed: P3-R11).
export function rule(svg, x1, y1, x2, y2) {
  const line = G.svgEl('line', { x1, y1, x2, y2 }, G.svgEl('g', { class: 'glyph g-note' }, svg));
  line.style.stroke = 'var(--ink-muted)';
  line.style.strokeWidth = '1';
  return line;
}

// A small legend swatch: the tint a GPU's pieces take in matrices (cell gpu tint).
export function gpuSwatch(svg, x, y, gpuNumber) {
  G.cell(svg, { x, y, size: 16, v: 0, maxAbs: 1, gpu: gpuNumber, format: () => '' });
  label(svg, x + 20, y + 8, `GPU ${gpuNumber}`, { cls: 'g-label' });
}

export const noNumber = (v) => (Number.isFinite(v) ? String(v).replace('-', '−') : '');
export const BLANK = () => '';
export const zeros = (rows, cols) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
