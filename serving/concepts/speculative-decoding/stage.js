// speculative-decoding stage layout: fixed positions shared by every frame, the timing helpers and small drawing helpers
// built on the glyph library. The chip row, the two blocks and the two step bars keep their places across frames 1–5.
import * as G from '@shared/glyphs.js';
import { PROMPT, GUESSES, STEP_SCALE_S } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CHIPS = Object.freeze({ x: 24, y: 8, gap: 8, h: 24 });
export const VERDICT_Y = 46;
export const PICK_Y = 62; // "target's pick" label baseline, under the corrected chip
export const BLOCK = Object.freeze({ y: 78, h: 40 });
export const DRAFTER = Object.freeze({ x: 70, w: 96 });
export const TARGET = Object.freeze({ x: 300, w: 150 });
export const NOTE_Y = 142; // the "Numbers shown" line of frames 1–5
export const BAR = Object.freeze({ y: 188, w: 220, titleY: 174, left: 24, right: 300 });

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

// x of each chip in a row of words, left to right from CHIPS.x.
export function chipXs(words) {
  const xs = [];
  words.reduce((x, w) => { xs.push(x); return x + G.tokenWidth(w) + CHIPS.gap; }, CHIPS.x);
  return xs;
}

export const ROW = Object.freeze([...PROMPT, ...GUESSES]); // the chip slots: the prompt, then one slot per guess
export const ROW_XS = Object.freeze(chipXs(ROW));
export const chipCenter = (i) => ROW_XS[i] + G.tokenWidth(ROW[i]) / 2;
export const TARGET_CX = TARGET.x + TARGET.w / 2;
export const DRAFTER_CX = DRAFTER.x + DRAFTER.w / 2;

// One chip of ROW at its place; `state` 'draft' draws the dashed unverified guess.
// `text` puts another word in the slot (the target's pick takes the failed guess's place).
export function chip(parent, i, { opacity = 1, state = 'idle', dy = 0, text = ROW[i] } = {}) {
  if (opacity <= 0) return;
  G.token(layer(parent, opacity), { x: ROW_XS[i], y: CHIPS.y + dy, text, state });
}

// The followed item's outline around chip i (template rule 5), at the given opacity.
export function followChip(parent, i, opacity = 1, text = ROW[i]) {
  if (opacity <= 0) return;
  G.selectionMark(layer(parent, opacity), { x: ROW_XS[i], y: CHIPS.y, w: G.tokenWidth(text), h: CHIPS.h });
}

export function blocks(parent, { drafter = 1, target = 1 } = {}) {
  if (target > 0) G.block(layer(parent, target), { x: TARGET.x, y: BLOCK.y, w: TARGET.w, h: BLOCK.h, label: 'target (70B)' });
  if (drafter > 0) G.block(layer(parent, drafter), { x: DRAFTER.x, y: BLOCK.y, w: DRAFTER.w, h: BLOCK.h, label: 'drafter' });
}

// A line from a chip's foot to a block's top with a dot at progress t.
export function flowTo(parent, from, to, t) {
  if (t <= 0 || t >= 1) return;
  G.flow(parent, { from, to, carry: 'token', progress: t });
}

// Dateless "stand-in" line printed on frames 2–5 (README lesson 10).
export const STAND_IN = 'guesses, probabilities and drafter cost are hand-picked stand-ins';
export const standIn = (parent, opacity = 1) => note(layer(parent, opacity), CHIPS.x, 338, STAND_IN);

// A titled step bar on the stage's one seconds scale; `t` (0–1) scales the seconds so the bar draws.
export function stepBarAt(parent, { x, y = BAR.y, titleY = BAR.titleY, parts, t = 1, title }) {
  note(parent, x, titleY, title);
  G.stepBar(layer(parent), {
    x, y, w: BAR.w, scaleS: STEP_SCALE_S, label: title,
    reading: parts.reading.map((r) => ({ label: r.label, s: r.s * t })), mathS: parts.mathS * t,
  });
}

// Several lines of plain text, 13 px apart (the 11 px label size at 580 px holds about 80 characters).
export function textLines(parent, x, y, lines, opts = {}) {
  lines.forEach((str, i) => note(parent, x, y + i * 13, str, opts));
}

// A row of cells a math term can name: the wrapper carries data-link and an invisible frame for the hover outline.
export function linkedRow(parent, letter, { x, y, values, cell, label, format }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: values.length * cell + 2, height: cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  G.vector(g, { x, y, values, cell, orient: 'row', maxAbs: 1, label, format });
  return g;
}
