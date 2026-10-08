// moe stage layout: fixed positions shared by every frame, the timing helpers and small drawing helpers built on the
// glyph library. The header (sat row → router → scores) and the expert row keep their place in frames 1–3, 5–7 and 9;
// the load bars keep theirs in frames 4, 8 and 10.
import * as G from '@shared/glyphs.js';
import { X_SAT, SCORE_MAX } from './numbers.js';
import { scoreText, expertName } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const SMALL = 18; // hover-only vectors
export const HALF = 20; // the fine-grained score cells (hover-only)
export const TINY = 14; // expert outputs (hover-only)
export const X0 = 56; // left edge of every eight-column row; column e spans X0 + 40e … X0 + 40e + 40
export const PITCH = CELL;
export const WIDTH8 = 8 * PITCH;
export const NOTES_X = X0 + WIDTH8 + 12; // the notes column right of the rows

export const SAT_ROW = Object.freeze({ x: X0, y: 8 });
export const ROUTER = Object.freeze({ x: X0, y: 44, w: 144, h: 28 });
export const SHARED = Object.freeze({ x: 232, y: 44, w: 100, h: 28 });
export const SCORES_Y = 112;
export const ARROW_X = X0 + 2 * PITCH; // between the E2 and E3 columns, clear of the column labels
export const EXPERTS = Object.freeze({ y: 164, w: 36, h: 28 });
export const LINES_Y = 238; // the readout lines of frames 5–7
export const LINE_GAP = 18;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
// Each frame starts from the previous frame's end state: what leaves fades out and what arrives fades in during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);
export const dip = (t) => 1 - 0.55 * Math.sin(Math.PI * clamp01(t)); // "dim and brighten" for a pulse

export const colLeft = (e) => X0 + e * PITCH;
export const colCenter = (e) => colLeft(e) + PITCH / 2;
export const expertLeft = (e) => colLeft(e) + (PITCH - EXPERTS.w) / 2;

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// Draws into a group at `opacity`; nothing at all when it is fully transparent.
export function fade(parent, opacity, draw) {
  if (opacity <= 0) return;
  draw(layer(parent, opacity));
}

// A plain labeled text mark (README lesson 15), styled like glyph labels; cls '' gives body ink.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return g;
}

// Scene change: the previous frame's end state fades out while the new scene fades in (template rule 4).
export const fadeOutOf = (svg, draw, p) => fade(svg, leaving(p), (g) => draw(g, 1));

// A row of cells that a math term can name: the wrapper carries data-link and an invisible frame for the hover outline.
export function linkedCells(parent, letter, { x, y, values, cell, maxAbs, format }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: values.length * cell + 2, height: cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  G.vector(g, { x, y, values, cell, orient: 'row', maxAbs, format });
  return g;
}

// The followed row "sat" (a hover-only vector) with the one selection outline.
export function satRow(parent, { opacity = 1 } = {}) {
  const g = layer(parent, opacity);
  G.vector(g, { ...SAT_ROW, values: X_SAT, cell: SMALL, orient: 'row', maxAbs: SCORE_MAX, label: 'x′_sat' });
  G.selectionMark(g, { x: SAT_ROW.x, y: SAT_ROW.y, w: X_SAT.length * SMALL, h: SMALL });
  return g;
}

// sat → router → the score row: the two arrows (each with a dot at `intoRouter`, `outOfRouter` progress) and the router block.
export function header(parent, { intoRouter = 1, outOfRouter = 1, opacity = 1 } = {}) {
  const g = layer(parent, opacity);
  satRow(g);
  G.flow(g, { from: [ARROW_X, SAT_ROW.y + SMALL + 3], to: [ARROW_X, ROUTER.y - 2], carry: 'activation', progress: intoRouter });
  G.block(g, { x: ROUTER.x, y: ROUTER.y, w: ROUTER.w, h: ROUTER.h, label: 'router [8 × 8]' }).setAttribute('data-link', 'score');
  G.flow(g, { from: [ARROW_X, ROUTER.y + ROUTER.h + 2], to: [ARROW_X, SCORES_Y - 12], carry: 'activation', progress: outOfRouter });
  return g;
}

// One row of router scores at NUMBER_CELL: `count` cells typed in so far, `hatch[i]` true = not chosen (still printed).
export function scoreRow(parent, { values, count = values.length, hatch = null, y = SCORES_Y, link = 'score', labels = true, rowLabel = null, maxAbs = SCORE_MAX }) {
  const shown = values.slice(0, count);
  const g = G.heatmap(parent, {
    x: X0, y, values: [shown], cell: CELL, maxAbs, format: scoreText,
    colLabels: labels ? shown.map((_, e) => expertName(e)) : [], rowLabels: rowLabel ? [rowLabel] : [],
    hatch: hatch ? [hatch.slice(0, count)] : undefined,
  });
  if (link) g.setAttribute('data-link', link);
  return g;
}

// Eight expert blocks under their score columns. `active` lists the lit ones; `activeOpacity` pulses them.
export function expertRow(parent, { y = EXPERTS.y, active = [], activeOpacity = 1 } = {}) {
  const g = G.svgEl('g', {}, parent);
  for (let e = 0; e < 8; e += 1) {
    const lit = active.includes(e);
    const host = lit ? layer(g, activeOpacity) : g;
    G.block(host, { x: expertLeft(e), y, w: EXPERTS.w, h: EXPERTS.h, label: expertName(e), state: lit ? 'active' : 'idle' });
  }
  return g;
}

