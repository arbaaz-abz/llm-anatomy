// training-pipeline stage layout: fixed positions shared by every frame, small timing helpers and drawing helpers.
// Every frame is a pure function of (index, progress): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { STAGES } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const ROW = Object.freeze({ x: 5, y: 58, h: 40, gap: 6 });
export const CHIP_Y = 20; // the checkpoint chip rides above the row
export const CHIP_H = 24; // the token glyph's height
export const REPLY = Object.freeze({ y: 322, x: 6 });
export const SPEC = Object.freeze({ y: 130, w: 148, h: 40, gap: 14 });
export const MERGED = Object.freeze({ y: 232, w: 148, h: 40 });
export const MERGED_X = (STAGE.w - MERGED.w) / 2;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
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

// Block x positions: cumulative widths plus gaps.
export const blockX = (i) => ROW.x + STAGES.slice(0, i).reduce((sum, s) => sum + s.w + ROW.gap, 0);
export const blockCenter = (i) => blockX(i) + STAGES[i].w / 2;

// The six stage blocks. `states[i]` is idle | active | dim; `opacity` per block draws the row in left to right (frame 1).
export function stageRow(parent, states, { opacity = () => 1 } = {}) {
  STAGES.forEach((s, i) => {
    const o = opacity(i);
    if (o > 0) G.block(layer(parent, o), { x: blockX(i), y: ROW.y, w: s.w, h: ROW.h, label: s.label, state: states[i] });
  });
}

// The one selection outline on stage block i.
export function selectStage(parent, i, opacity = 1) {
  if (opacity > 0) G.selectionMark(layer(parent, opacity), { x: blockX(i), y: ROW.y, w: STAGES[i].w, h: ROW.h });
}

// The checkpoint chip above the row, at fractional block position `at` (travels between block centers).
export function checkpointChip(parent, at, opacity = 1) {
  if (opacity <= 0) return;
  const lo = Math.floor(at);
  const hi = Math.min(lo + 1, STAGES.length - 1);
  const cx = lerp(blockCenter(lo), blockCenter(hi), at - lo);
  const w = G.tokenWidth('checkpoint');
  G.token(layer(parent, opacity), { x: cx - w / 2, y: CHIP_Y, text: 'checkpoint' });
}

// A row of token chips from x; `opacity(i)` fades each.
export function chipRow(parent, words, { x, y, opacity = () => 1, gap = 8 }) {
  let cx = x;
  words.forEach((text, i) => {
    const o = opacity(i);
    if (o > 0) G.token(layer(parent, o), { x: cx, y, text });
    cx += G.tokenWidth(text) + gap;
  });
  return cx - gap;
}

// The prompt chip, an arrow, and the reply chips: the reply row at its fixed place (every frame).
export function replyRow(parent, { words, opacity = () => 1, label = 'reply (illustrative)', arrowProgress = 1 }) {
  const promptW = G.tokenWidth('What is 7 × 8?');
  note(parent, REPLY.x, REPLY.y - 8, label);
  G.token(parent, { x: REPLY.x, y: REPLY.y, text: 'What is 7 × 8?' });
  const arrowX = REPLY.x + promptW + 6;
  G.flow(parent, { from: [arrowX, REPLY.y + CHIP_H / 2], to: [arrowX + 30, REPLY.y + CHIP_H / 2], carry: 'token', progress: arrowProgress });
  chipRow(parent, words, { x: arrowX + 40, y: REPLY.y, opacity });
}

// The text of a reveal: the first `t` of the string (a typed-in label).
export const typed = (str, t) => str.slice(0, Math.ceil(str.length * clamp01(t)));
