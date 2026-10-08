// model-card stage layout: fixed positions shared by every frame, and small drawing helpers built on the glyph library.
// The card column and the diagram keep their places in frames 1–9 (storyboard §5 "Determinism").
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const COL = Object.freeze({ x: 8, titleY: 12, y: 20, stride: 27, chipH: 24, extraLabelY: 304, extraY: [310, 337] });
export const STACK = Object.freeze({ x: 236, y: 24, w: 104, h: 212 }); // blockStack: block 1, "⋮ × N", block N
export const PANEL = Object.freeze({ x: 346, w: 226 }); // the diagram's right panel: experts, KV cache, bars, readouts
export const LANE = Object.freeze({ x: 236, labelY: 268, y: 276, words: ['The', 'cat', 'sat', 'down'], gap: 6, patchX: 420, patchSize: 24, barY: 314, endX: 572 });
export const TOKEN_H = 24;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;

export const chipY = (i) => COL.y + i * COL.stride;
export const laneWidth = () => LANE.words.reduce((sum, w) => sum + G.tokenWidth(w), 0) + LANE.gap * (LANE.words.length - 1);

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// A plain rule in the flow path style (no arrowhead, no dot): range connectors, brackets, the stretched lane.
export function rule(parent, x1, y1, x2, y2) {
  const g = G.svgEl('g', { class: 'g-flow' }, parent);
  G.svgEl('line', { class: 'g-path', x1, y1, x2, y2 }, g);
  return g;
}

// The selection outline on one chip (the followed field, or the part it describes), at the given opacity.
export function mark(parent, opacity, box) {
  if (opacity > 0) G.selectionMark(layer(parent, opacity), box);
}
