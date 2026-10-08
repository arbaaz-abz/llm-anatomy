// agentic-rl stage layout: fixed positions shared by every frame, the timing helpers, and the drawing helpers built on
// the glyph library (the episode strip, the group table, the rollout timeline). Every frame is a pure function of (index, progress).
import * as G from '@shared/glyphs.js';
import { ADVANTAGES, A_MAX_ABS, DURATIONS, EPISODE, FOLLOWED, REWARDS, ROWS } from './numbers.js';
import { minutes } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL;
export const CHIP_H = 24;
export const CHIP_GAP = 4;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
// Each frame starts from the previous frame's end state: what leaves fades out and what arrives fades in during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// A bare G.cell has no .glyph ancestor, so theme.css would not size its text (11px); this wraps it in one.
export const numberCell = (parent, opts) => G.cell(G.svgEl('g', { class: 'glyph' }, parent), opts);

// A group a math term can name: data-link plus an invisible frame for the hover outline (theme.css lists the letters).
export function linked(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: w + 2, height: h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

export const widthOf = (words) => words.map((w) => G.tokenWidth(w));
export const chipXs = (words, x0, gap = CHIP_GAP) => widthOf(words).map((_, i) => x0 + widthOf(words).slice(0, i).reduce((s, w) => s + w + gap, 0));
export const chipFill = (advantage) => G.valueColor(advantage, A_MAX_ABS);

// The followed token's selection outline, drawn after the chip it marks at the same geometry.
export const selectChip = (parent, x, y, text) => G.selectionMark(parent, { x, y, w: G.tokenWidth(text), h: CHIP_H });

// ---- the group table (frame 3, the toy): rows of token chips, a verdict, the R and A cells ----
export const TABLE = Object.freeze({ x: 8, y: 22, stride: 43, chipsX: 24, verdictX: 303, rX: 317, aX: 361 });
const rowTop = (row) => TABLE.y + row * TABLE.stride;

// chip(row, pos) → { fill?, hatched?, state? }; `shown(row)` → { verdict, r, a } opacities (1 when omitted).
export function groupTable(parent, { chip = () => ({}), shown = () => ({ verdict: 1, r: 1, a: 1 }), selected = FOLLOWED, onChip = null } = {}) {
  note(parent, TABLE.rX + CELL / 2, TABLE.y - 6, 'R', { anchor: 'middle' });
  note(parent, TABLE.aX + CELL / 2, TABLE.y - 6, 'A', { anchor: 'middle' });
  ROWS.forEach((tokens, row) => {
    const y = rowTop(row);
    note(parent, TABLE.x, y + CELL / 2 + 4, String(row + 1));
    const xs = chipXs(tokens, TABLE.chipsX);
    const chipY = y + (CELL - CHIP_H) / 2;
    tokens.forEach((text, pos) => {
      const g = G.token(parent, { x: xs[pos], y: chipY, text, ...chip(row, pos) });
      onChip?.(g, row, pos);
      if (selected && selected.row === row && selected.pos === pos) selectChip(parent, xs[pos], chipY, text);
    });
    const s = shown(row);
    if (s.verdict > 0) G.verdict(layer(parent, s.verdict), { x: TABLE.verdictX, y: y + CELL / 2, ok: REWARDS[row] === 1 });
    if (s.r > 0) numberCell(layer(parent, s.r), { x: TABLE.rX, y, size: CELL, v: REWARDS[row], maxAbs: 1, fill: REWARDS[row] === 1 ? 'ok' : 'bad' });
    if (s.a > 0) numberCell(linked(layer(parent, s.a), 'a', { x: TABLE.aX, y, w: CELL, h: CELL }), { x: TABLE.aX, y, size: CELL, v: ADVANTAGES[row], maxAbs: A_MAX_ABS });
  });
}

// The box of one chip of the group table (row, pos 0-based): where the toy draws a token and its selection mark.
export function tokenBox(row, pos) {
  const x = chipXs(ROWS[row], TABLE.chipsX)[pos];
  return { x, y: rowTop(row) + (CELL - CHIP_H) / 2, w: G.tokenWidth(ROWS[row][pos]), h: CHIP_H };
}

// ---- the episode (frames 1–3) ----
// The 11 tokens of row 1 as a tool episode, in reading order: policy turn 1, the observation, policy turn 2.
export const episodeWords = EPISODE.map((t) => t.text);

// ---- the rollout timeline (frames 5–6, the toy): one request bar per episode on a shared minute axis ----
export const TIMELINE = Object.freeze({ x: 56, y: 30, stride: 30, unit: 24, minutes: 16 });
export const timelineY = (row) => TIMELINE.y + row * TIMELINE.stride;
export const timelineX = (min, t = TIMELINE) => t.x + min * t.unit;

// Bars grown to `reach` minutes (each stops at its own length), the axis under them. The request glyph's label
// says "prefill 0 tokens, decode N tokens": here the bar is an episode and a tick is a minute, so the group is relabeled.
export function timelineBars(parent, { reach = TIMELINE.minutes, durations = DURATIONS, t = TIMELINE } = {}) {
  durations.forEach((d, row) => {
    const g = G.request(parent, { x: t.x, y: t.y + row * t.stride, prefill: 0, decode: Math.min(d, reach), unit: t.unit, label: String(row + 1) });
    g.setAttribute('aria-label', `episode ${row + 1}: ${minutes(Math.min(d, reach))} of ${minutes(d)}`);
  });
  const axisY = t.y + durations.length * t.stride - 6;
  const axis = G.svgEl('g', { class: 'glyph g-note' }, parent);
  G.svgEl('line', { class: 'g-axis', x1: t.x, y1: axisY, x2: t.x + t.minutes * t.unit, y2: axisY, stroke: 'var(--ink-muted)', 'stroke-width': 1 }, axis);
  [0, 4, 8, 12, 16].forEach((m) => {
    G.svgEl('line', { x1: timelineX(m, t), y1: axisY, x2: timelineX(m, t), y2: axisY + 4, stroke: 'var(--ink-muted)', 'stroke-width': 1 }, axis);
    note(axis, timelineX(m, t), axisY + 16, m === 0 ? '0' : minutes(m), { anchor: 'middle' });
  });
  return axisY;
}

// The plain "idle" label after a bar that has finished (no hatch: that is the Serving glyph option, ruling D8).
export const idleLabel = (parent, row, minute, t = TIMELINE) => note(parent, timelineX(minute, t) + 6, timelineY(row) + 8, 'idle', { cls: 'g-label' });

// A block that fades between its resting state and `active` (lit 0 → 1); the two layers cross-fade.
export function litBlock(parent, geom, label, lit, rest = 'dim') {
  if (lit < 1) G.block(layer(parent, 1 - lit), { ...geom, label, state: rest });
  if (lit > 0) G.block(layer(parent, lit), { ...geom, label, state: 'active' });
}

export const TRAINER = Object.freeze({ x: 472, y: 130, w: 96, h: 40 });
export const trainerBlock = (parent, lit) => litBlock(parent, TRAINER, 'trainer', lit);

// Two or three plain readout lines at the foot of a stage (the numbers a frame shows).
export function readout(parent, lines, { y = 330, opacity = 1, x = 14, gap = 17 } = {}) {
  const g = layer(parent, opacity);
  lines.forEach((line, i) => note(g, x, y + i * gap, line, { cls: 'g-readout' }));
  return g;
}
