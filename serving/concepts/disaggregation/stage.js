// disaggregation stage layout and drawing helpers: fixed positions shared by the frames, the timing helpers, and the
// handoff that makes every frame start exactly where the previous one ended (template rule 4).
import * as G from '@shared/glyphs.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL;
export const LEFT = 8; // the left text column's x
export const LINE = 16; // text line pitch
export const CHAR_W = 6.6; // the monospace stage text's advance, px per character

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a));
export const lerp = (a, b, t) => a + (b - a) * t;
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF]
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark. cls 'g-label' is muted; '' prints in ink (a value the learner reads).
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// Several lines from (x, y) down, one per LINE.
export function lines(parent, x, y, texts, { cls = '' } = {}) {
  texts.forEach((t, i) => note(parent, x, y + i * LINE, t, { cls }));
}

// A plain 1 px muted connector (never dashed, P4-R15).
export function connector(parent, [x1, y1], [x2, y2]) {
  return G.svgEl('line', { class: 'g-link-plain', x1, y1, x2, y2, stroke: 'var(--ink-muted)', 'stroke-width': 1 }, G.svgEl('g', { class: 'glyph g-connector' }, parent));
}

// A math-panel link target: <g data-link> with an invisible frame the hover outlines.
export function linkGroup(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// The previous frame's end state fades out while this frame fades in: at p = 0 the stage is exactly the previous frame at p = 1.
// `prev` draws into a layer (a glyph builder never needs the <svg> itself); `draw(parent)` is this frame's content.
export function scene(svg, p, prev, draw) {
  if (prev && p < HANDOFF) prev(layer(svg, leaving(p)), 1);
  draw(layer(svg, prev ? arriving(p) : 1));
}

// Request lanes on a shared time axis. `lane` is { arrivesMs, steps: [{ fromMs, toMs, kind }] } (colocated.js laneOf);
// only what has happened by `upToMs` is drawn: a prefill or a wait grows, a decode step appears when it completes.
export function drawLane(parent, { x, y, id, lane, pxPerMs, upToMs = Infinity }) {
  const steps = lane.steps.flatMap((s) => {
    if (s.fromMs >= upToMs) return [];
    const end = Math.min(s.toMs, upToMs);
    if (s.kind === 'decode' && end < s.toMs) return [];
    return [{ from: s.fromMs * pxPerMs, to: end * pxPerMs, kind: s.kind }];
  });
  if (steps.length === 0) return null;
  return G.request(parent, { x, y, owner: id, label: id, steps });
}

// The extent of a lane's revealed steps in px (for a selection mark), or null before anything is drawn.
export function laneExtent(lane, pxPerMs, upToMs = Infinity) {
  const shown = lane.steps.filter((s) => s.fromMs < upToMs);
  if (shown.length === 0) return null;
  const last = shown.at(-1);
  const to = last.kind === 'decode' && last.toMs > upToMs ? shown.at(-2)?.toMs ?? shown[0].fromMs : Math.min(last.toMs, upToMs);
  return { from: shown[0].fromMs * pxPerMs, to: to * pxPerMs };
}
