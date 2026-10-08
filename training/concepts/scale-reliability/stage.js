// scale-reliability stage: the fixed 580 × 366 canvas, timing helpers (seg / lerp / ease) and the small drawing helpers
// the frames share. Every frame is a pure function of (index, progress): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const MARGIN = 30;
export const FULL_W = STAGE.w - 2 * MARGIN; // 520: the width of a bar that stands for SAME_SCALE_GPU_HOURS

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// A line of text that types in: the first `t` of its characters (t = 1 prints it whole).
export const typed = (str, t) => str.slice(0, Math.ceil(str.length * clamp01(t)));

// A group a math term can name (theme.css outlines its .g-frame on hover); the frame is invisible at rest.
export function linked(parent, name, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': name }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// A single plain tick on a bar's scale (page-drawn: 1 px ink, a label above), e.g. "model card: 30.84M".
export function tick(parent, { x, y, h, label, opacity = 1, dy = 0 }) {
  const g = layer(parent, opacity);
  G.svgEl('line', { x1: x, y1: y + dy, x2: x, y2: y + h + dy, stroke: 'var(--ink)', 'stroke-width': 1 }, g);
  note(g, x, y - 5 + dy, label, { anchor: 'middle' });
  return g;
}

// The bar for `hours` GPU-hours on the shared scale: `fullHours` fills FULL_W (frames 2, 3 and 10 print "same scale").
export const barWidth = (hours, fullHours) => (FULL_W * hours) / fullHours;
