// The rope stage's shared kit: layout constants, timing helpers and the drawing helpers every frame uses.
// Pure: nothing touches the DOM at import time.
import * as G from '@shared/glyphs.js';
import { TOY } from '@math/attention.js';
import { rotatePairs } from '@math/rope.js';
import { cellText, trimNumber, Q_SAT } from './format.js';
import { FREQS, Q_PAIRS, K_PAIRS, K_CAT } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const TOKENS = TOY.tokens;
export const QUERY = 2; // "sat", the item we follow in every frame
export const KEY = 1; // "cat"
export const SCALE = Object.freeze({ vector: 2, score: 3.5 }); // storyboard §4 "Color": vector cells on a maxAbs-2 scale

// ---- layout: chips on top; frames 1-5 share one q/k layout; frames 6-9 share the offset row ----
export const TOKEN_Y = 8;
export const TOKEN_X = Object.freeze([14, 62, 110, 158]);
export const GHOST_X = 398; // the slot for position 9
export const TAG_Y = 46; // "query" and "key" under the chips
export const NAME_Y = 66;
export const VEC_Y = 78;
export const PAIR_W = 2 * CELL;
export const PAIR_GAP = 56; // the divider that opens between a vector's two pairs
export const BLOCK_X = Object.freeze({ q: 14, k: 262 });
export const DIAL_R = 28;
export const DIAL_Y = 172;
export const TEXT_Y = Object.freeze([232, 246]); // speed line, angle line
export const NOTE_Y = 264;
export const READOUT = Object.freeze({ label: 238, value: 250, y: [286, 303, 322], was: 330 });
export const PROMPT_Y = Object.freeze([342, 356]);
export const SCORE_ROW = Object.freeze({ x: 150, y: 176 });

export const ROW = Object.freeze({ x: 130, y: 76, n: 8 });
export const REACH_DIAL = Object.freeze({ r: 34, y: 228, x: Object.freeze([190, 380]) });
export const REACH_TEXT_Y = Object.freeze([282, 298]);
export const BOTTOM_X = 14;
export const BOTTOM_Y = Object.freeze([326, 344]);

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

// Plain text marks live in a .glyph group so the stage's one label face and size apply (spec §5.4).
export function label(svg, x, y, str, { anchor = 'start', cls = 'g-label', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

export const select = (svg, x, y, w, h, opacity = 1) => fade(G.selectionMark(svg, { x, y, w, h }), opacity);

// A math-linked group: an invisible rect.g-frame the math-panel hover outlines, then the content inside it.
export function linked(svg, letter, box) {
  const g = G.svgEl('g', { 'data-link': letter }, svg);
  G.svgEl('rect', { class: 'g-frame', x: box.x - 1, y: box.y - 1, width: box.w + 2, height: box.h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// A plain stroke in the muted ink (marks that are not glyphs): set through style so the theme's variables apply.
export function strokeStyle(node, { width = 1.5, opacity = 1 } = {}) {
  node.style.stroke = 'var(--ink-muted)';
  node.style.strokeWidth = String(width);
  node.style.strokeLinecap = 'round';
  if (opacity < 1) node.style.opacity = String(opacity);
  return node;
}

const glyphValues = (values) => values.map((v) => (v == null ? Number.NaN : v));

// A row of NUMBER_CELL numbers; null cells are not computed yet (blank).
export function numberRow(svg, { x, y, values, maxAbs = SCALE.vector, format = cellText, link, cell = CELL }) {
  const parent = link ? linked(svg, link, { x, y, w: values.length * cell, h: cell }) : svg;
  return G.vector(parent, { x, y, values: glyphValues(values), cell, orient: 'row', maxAbs, format });
}

// The four token chips; "sat" carries the selection mark wherever it appears. `ghost` is the copy of "cat" at another slot.
export function tokenRow(svg, { opacity = 1, ghost = null } = {}) {
  TOKENS.forEach((t, i) => fade(G.token(svg, { x: TOKEN_X[i], y: TOKEN_Y, text: t, index: i + 1 }), opacity));
  select(svg, TOKEN_X[QUERY], TOKEN_Y, G.tokenWidth(TOKENS[QUERY]), 24, opacity);
  if (ghost) fade(G.token(svg, { x: ghost.x, y: TOKEN_Y, text: TOKENS[KEY], index: ghost.index, state: 'dim' }), ghost.opacity);
}

export const tagsRow = (svg, opacity = 1) => {
  label(svg, TOKEN_X[QUERY] + G.tokenWidth(TOKENS[QUERY]) / 2, TAG_Y, 'query', { anchor: 'middle', opacity });
  label(svg, TOKEN_X[KEY] + G.tokenWidth(TOKENS[KEY]) / 2, TAG_Y, 'key', { anchor: 'middle', opacity });
};

// ---- frames 1-5: a query or key vector as two pairs, each with its dial ----
const VEC = Object.freeze({ q: Q_SAT, k: K_CAT });
const PAIRS = Object.freeze({ q: Q_PAIRS, k: K_PAIRS });
export const pairCenterX = (who, i, gap = 1) => BLOCK_X[who] + i * (PAIR_W + PAIR_GAP * gap) + PAIR_W / 2;

// Draws one vector (q or k) at position `pos` (fractional while it turns): its two pairs of cells, the divider `gap`
// (0 = joined, 1 = open), the names, and the two dials with their speed and angle lines.
// `parts`: { dials, speed, angle } opacities; `angleAt` the whole position the angle line prints.
export function pairBlock(svg, { who, pos, gap, name, parts = {}, angleAt = Math.round(pos), short = false }) {
  const values = rotatePairs(VEC[who], pos, FREQS);
  const x = BLOCK_X[who];
  [0, 1].forEach((i) => numberRow(svg, { x: x + i * (PAIR_W + PAIR_GAP * gap), y: VEC_Y, values: values.slice(2 * i, 2 * i + 2) }));
  label(svg, x, NAME_Y, name);
  if (who === 'q') select(svg, x, VEC_Y, 2 * PAIR_W + PAIR_GAP * gap, CELL);
  const { dials = 0, speed = 0, angle = 0 } = parts;
  [0, 1].forEach((i) => {
    const cx = pairCenterX(who, i);
    if (dials > 0) fade(G.dial(svg, { x: cx, y: DIAL_Y, r: DIAL_R, vector: PAIRS[who][i], angle: pos * FREQS[i], scale: SCALE.vector, label: `pair ${i + 1}` }), dials);
    if (speed > 0) label(svg, cx, TEXT_Y[0], `${trimNumber(FREQS[i])} per token`, { anchor: 'middle', opacity: speed });
    if (angle > 0) label(svg, cx, TEXT_Y[1], short ? `${trimNumber(angleAt * FREQS[i])} rad` : `${angleAt} × ${trimNumber(FREQS[i])} = ${trimNumber(angleAt * FREQS[i])} rad`, { anchor: 'middle', opacity: angle });
  });
}

// ---- frames 6-9: the offset row ----
export function offsetRow(svg, { values, title, opacity = 1 }) {
  label(svg, ROW.x, ROW.y - 14, title, { opacity });
  label(svg, ROW.x - 10, ROW.y + CELL / 2, 'score', { anchor: 'end', opacity });
  fade(numberRow(svg, { x: ROW.x, y: ROW.y, values, maxAbs: SCALE.score }), opacity);
  values.forEach((_, j) => label(svg, ROW.x + j * CELL + CELL / 2, ROW.y + CELL + 12, String(j), { anchor: 'middle', opacity }));
  label(svg, ROW.x - 10, ROW.y + CELL + 12, 'offset', { anchor: 'end', opacity });
}

// A key dial for frames 6-9: its hand has turned by `angle` between the two tokens.
export function reachDial(svg, i, { angle, seen = null, reached = null, opacity = 1 }) {
  return fade(G.dial(svg, { x: REACH_DIAL.x[i], y: REACH_DIAL.y, r: REACH_DIAL.r, angle, seen, reached, label: `pair ${i + 1}` }), opacity);
}

// Cells revealed left to right as the offset grows: cell j shows once the offset t has reached j.
export const revealed = (values, offset) => values.map((v, j) => (offset >= j - 1e-9 ? v : null));

// Two rows' cells blended: every cell interpolates between two computed states.
export const blendRows = (from, to, t) => from.map((v, j) => lerp(v, to[j], t));
