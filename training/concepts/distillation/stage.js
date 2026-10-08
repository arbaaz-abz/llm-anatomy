// distillation stage layout: fixed positions shared by every frame, timing helpers and small drawing helpers built on the
// glyph library. The teacher column is left, the student column right; frames 1–5 keep both at these anchors.
import * as G from '@shared/glyphs.js';
import { CANDIDATES, PROMPT, PROB_MAX_ABS, REWARD_MAX_ABS } from './numbers.js';
import { signed } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const CHIP_H = 24; // the token glyph's height
export const PROMPT_AT = Object.freeze({ x: 14, y: 6, gap: 8 });
export const COL = Object.freeze({ teacher: 14, student: 360 }); // x of each column's block and 4-cell row
export const ROW = Object.freeze({ block: 40, vec: 92, under: 136 }); // `under`: the row that sits directly below the student's
export const BLOCK = Object.freeze({ w: 170, h: 32 });
export const ROW_W = 4 * CELL;
export const SAMPLE = Object.freeze({ chipY: 150, cellY: 182 }); // the student's own sample (frames 4–5): chip, then its grade

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
export const cellX = (col, i) => col + i * CELL;
export const cellCenter = (col, i) => cellX(col, i) + CELL / 2;

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  G.svgEl('text', { x, y, class: 'g-label', 'text-anchor': anchor }, g).textContent = str;
  return g;
}

const chipWidth = (text) => G.tokenWidth(text);
const promptX = (i) => PROMPT_AT.x + PROMPT.slice(0, i).reduce((sum, t) => sum + chipWidth(t) + PROMPT_AT.gap, 0);

// The prompt `7 × 8 =` as four chips along the top.
export function promptChips(parent, opacity = 1) {
  const g = layer(parent, opacity);
  PROMPT.forEach((text, i) => G.token(g, { x: promptX(i), y: PROMPT_AT.y, text }));
  return g;
}

export const teacherBlock = (parent, { state = 'idle', opacity = 1, x = COL.teacher } = {}) => (
  G.block(layer(parent, opacity), { x, y: ROW.block, w: BLOCK.w, h: BLOCK.h, label: 'teacher: math specialist', state }));
export const studentBlock = (parent, { state = 'idle', opacity = 1 } = {}) => (
  G.block(layer(parent, opacity), { x: COL.student, y: ROW.block, w: BLOCK.w, h: BLOCK.h, label: 'student', state }));

// The four candidate tokens above a row: `56 · 54 · 48 · 63`.
export function candidateHeads(parent, col, opacity = 1, y = ROW.vec - 8) {
  const g = layer(parent, opacity);
  CANDIDATES.forEach((c, i) => note(g, cellCenter(col, i), y, c, { anchor: 'middle' }));
  return g;
}

// One row of four probabilities. `link` is the math-panel letter (t = teacher, s = student); `cellOpacity(i)` fades single
// cells in or out; `follow` is the index that takes the selection outline.
export function probRow(parent, { x, y, values, link, cellOpacity = () => 1, opacity = 1, follow = null, selOpacity = 1 }) {
  const g = G.svgEl('g', { 'data-link': link }, layer(parent, opacity));
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: ROW_W + 2, height: CELL + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  values.forEach((v, i) => {
    const o = cellOpacity(i);
    if (o > 0) G.cell(layer(g, o), { x: cellX(x, i), y, size: CELL, v, maxAbs: PROB_MAX_ABS });
  });
  if (follow != null && selOpacity > 0) G.selectionMark(layer(parent, opacity * selOpacity), { x: cellX(x, follow), y, w: CELL, h: CELL });
  return g;
}

// One reward cell on the page's one reward scale; the format is signed, 2 decimals ("+0.81", "−1.79", "0.00").
export const rewardFormat = (v) => signed(v, 2);
export function rewardCell(parent, { x, y, v, opacity = 1, link = 'r' }) {
  const g = G.svgEl('g', { 'data-link': link }, layer(parent, opacity));
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: CELL + 2, height: CELL + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  G.cell(g, { x, y, size: CELL, v, maxAbs: REWARD_MAX_ABS, format: rewardFormat });
  return g;
}

// A chip coloured on the reward scale; the colour is the reward at `reach` (0 = neutral, 1 = its grade).
export const rewardChip = (parent, { x, y, text, v, reach = 1, link = 's' }) => (
  G.token(G.svgEl('g', { 'data-link': link }, parent), { x, y, text, fill: G.valueColor(v * reach, REWARD_MAX_ABS) }));

// Types a string in: the first n characters of `str` for progress t (a typewriter, pure in t).
export const typed = (str, t) => str.slice(0, Math.ceil(clamp01(t) * str.length));
