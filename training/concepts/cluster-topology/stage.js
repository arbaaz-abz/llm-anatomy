// cluster-topology stage kit: layout constants, timing helpers and the drawing helpers every frame shares. Pure: nothing
// touches the DOM at import time. Every helper takes the <svg> or a <g> inside it (template rules 3–4).
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
const CHAR_W = 6.6; // one mono character at the 11 px label size

// ---- the rack glyph's built geometry (P3-R8): 16 px GPU tiles, 10 px gaps ----
export const TILE = 16;
export const TILE_GAP = 10;
export const rackSize = (gpus, cols) => ({ w: cols * (TILE + TILE_GAP) + TILE_GAP, h: Math.ceil(gpus / cols) * (TILE + TILE_GAP) + TILE_GAP });
export const RACK8 = Object.freeze(rackSize(8, 4)); // 114 × 62
export const RACK16 = Object.freeze(rackSize(16, 4)); // 114 × 114
export const RACK72 = Object.freeze(rackSize(72, 9)); // 244 × 218

// Top-left corner of GPU i (0-based) in a rack drawn at (x, y), and its centre.
export function tileAt(x, y, i, cols) {
  const r = Math.floor(i / cols);
  const c = i % cols;
  const left = x + TILE_GAP + c * (TILE + TILE_GAP);
  const top = y + TILE_GAP + r * (TILE + TILE_GAP);
  return { x: left, y: top, cx: left + TILE / 2, cy: top + TILE / 2 };
}

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

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(parent, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

export function textBlock(parent, x, y, lines, { cls = 'g-label', gap = 16, opacity = 1, anchor = 'start' } = {}) {
  lines.forEach((line, i) => label(parent, x, y + i * gap, line, { cls, opacity, anchor }));
}

export const select = (parent, x, y, w, h, opacity = 1) => (opacity > 0 ? fade(G.selectionMark(parent, { x, y, w, h }), opacity) : null);

// The followed GPU: GPU 1 of server 1 (tile i = 0 of the rack at (x, y)).
export function selectGpu(parent, x, y, cols, i = 0, opacity = 1) {
  const t = tileAt(x, y, i, cols);
  return select(parent, t.x, t.y, TILE, TILE, opacity);
}

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines, then whatever `draw` puts inside.
export function linked(parent, letter, box, draw) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  draw?.(g);
  return g;
}

// A linked line of text: the frame is sized from the character count.
export function linkedText(parent, letter, { x, y, str, anchor = 'start', cls = 'g-label', opacity = 1 }) {
  if (opacity <= 0) return null;
  const w = str.length * CHAR_W;
  const left = anchor === 'end' ? x - w : anchor === 'middle' ? x - w / 2 : x;
  return fade(linked(parent, letter, { x: left - 2, y: y - 9, w: w + 4, h: 18 }, (inner) => label(inner, x, y, str, { anchor, cls })), opacity);
}

// A thin network line (a port or an uplink): a plain 1.5 px line in the communication color.
export function wire(parent, x1, y1, x2, y2, opacity = 1) {
  const line = G.svgEl('line', { x1, y1, x2, y2, 'stroke-width': 1.5, 'stroke-linecap': 'round' }, parent);
  line.style.stroke = 'var(--sem-comm)';
  return fade(line, opacity);
}

// A faint filled region with no outline (outlines mean selection): a pod, an expert group.
export function region(parent, { x, y, w, h, tint = 'var(--surface-2)', opacity = 1 }) {
  const rect = G.svgEl('rect', { x, y, width: w, height: h, rx: 6 }, parent);
  rect.style.fill = tint;
  rect.style.stroke = 'none';
  return fade(rect, opacity);
}

// Flow dots: one flow from → to at progress t (0 → 1); nothing is drawn before it starts.
export function hop(parent, from, to, carry, t, opacity = 1) {
  if (t <= 0 || opacity <= 0) return null;
  return fade(G.flow(parent, { from, to, carry, progress: t }), opacity);
}

// Lane timeline over the lane overlays: transparent g-frame rects over chosen lanes so the math panel can outline them.
export function laneLinks(parent, { x, y, w, lanes, scale, cap = null, ticks = [] }, letters) {
  const L = G.laneTimelineLayout({ w, lanes, scale, cap, ticks });
  letters.forEach((letter, i) => {
    if (letter) linked(parent, letter, { x: x + L.gutter, y: y + L.lanes[i].y, w: L.track, h: 24 });
  });
  return L;
}
