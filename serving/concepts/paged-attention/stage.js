// paged-attention stage kit: layout, timing helpers and the plain-text marks every frame shares. Pure: nothing touches the
// DOM at import time. Every helper takes the <svg> or a <g> inside it (template rule 4).
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const BARS = Object.freeze({ x: 34, y: 8, gap: 14, unit: 6, h: 10 }); // the A–D request bars on top
export const LANE_TITLE_Y = 82;
export const POOL_Y = 92;
export const TABLE_X = 6; // after-lane frames: the followed block table on the left, the pool on the right
export const TABLE_Y = 112;
export const AFTER_X = 184;
export const BEFORE_X = 10;
export const BAR_X = 10; // the usage bar
export const BAR_W = 500;
export const CHAR_W = 6.6; // one label character at the 11 px size

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, from, to) => clamp01((p - from) / (to - from)); // progress p remapped to the sub-phase [from, to]
export const lerp = (a, b, t) => a + (b - a) * t;
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? Math.max(opacity, 0).toFixed(3) : null }, parent);

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(parent, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return g;
}

export function textBlock(parent, x, y, lines, { cls = 'g-label', gap = 15, opacity = 1 } = {}) {
  lines.forEach((line, i) => label(parent, x, y + i * gap, line, { cls, opacity }));
}

export function select(parent, x, y, w, h, opacity = 1) {
  if (opacity <= 0) return null;
  const g = G.selectionMark(parent, { x, y, w, h });
  if (opacity < 1) g.setAttribute('opacity', opacity.toFixed(3));
  return g;
}

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines, then whatever `draw` puts inside.
export function linked(parent, name, box, draw) {
  const g = G.svgEl('g', { 'data-link': name }, parent);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  draw(g);
  return g;
}
