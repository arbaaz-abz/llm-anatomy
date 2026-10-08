// The rlvr-grpo stage kit: layout constants, the two groups the frames draw (all numbers from model.js / math/grpo.js),
// timing helpers and the drawing helpers every frame shares. Pure: nothing touches the DOM at import time.
import * as G from '@shared/glyphs.js';
import { evaluate } from './model.js';
import { INITIAL_STATE, signed2 } from './format.js';
import { ADV_MAX_ABS, GROUP_SIZE } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL;

// ---- the two groups: k = 2 (frames 1–5, the toy's default) and k = 8 (frame 6, no spread) ----
export const GROUP = evaluate(INITIAL_STATE);
export const ALL_RIGHT = evaluate({ ...INITIAL_STATE, k: GROUP_SIZE });
export const FOLLOWED = Object.freeze({ row: 0, token: 4 }); // row 1's final `56`, followed in every frame it appears

// ---- layout: header band 46 px + 8 rows × 40 px (= NUMBER_CELL, so the R and A columns line up) = 366 px ----
export const TOP = 46;
export const ROW_H = CELL;
export const rowY = (i) => TOP + i * ROW_H;
export const CHIP_H = 24;
export const CHIP_X0 = 34;
export const CHIP_GAP = 4;
export const GUTTER_X = 22; // frame 1's token flow runs down this gutter to the row it fills
export const INDEX_X = 8;
export const VERDICT_X = 308;
export const R_COL = Object.freeze({ x: 322, y: TOP });
export const A_COL = Object.freeze({ x: 366, y: TOP });
export const COL_BOTTOM = TOP + GROUP_SIZE * ROW_H;
export const MARGIN_X = 416; // the right margin: readouts, the checker block and its footnote
export const PROMPT = Object.freeze({ x: CHIP_X0, y: 3 });
export const HEADER_BLOCK = Object.freeze({ x: 204, y: 3, w: 204, h: 24 }); // policy (frames 1–2); the branch label of frame 6 sits here too
export const RULE_X = Object.freeze({ from: R_COL.x - 6, to: A_COL.x + CELL + 4 });

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
export function label(svg, x, y, str, { anchor = 'start', opacity = 1 } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const t = G.svgEl('text', { x, y, class: 'g-label', 'text-anchor': anchor, 'dominant-baseline': 'central' }, g);
  t.textContent = str;
  return fade(g, opacity);
}

export const select = (svg, x, y, w, h, opacity = 1) => fade(G.selectionMark(svg, { x, y, w, h }), opacity);

// A group of text lines in the right margin, one label per line.
export function marginLines(svg, lines, { y = 84, step = 18, opacity = 1 } = {}) {
  lines.forEach((line, i) => label(svg, MARGIN_X, y + i * step, line, { opacity }));
}

// ---- chips ----
export const advantageFill = (a) => G.valueColor(a, ADV_MAX_ABS);
export const chipX = (tokens, j) => CHIP_X0 + tokens.slice(0, j).reduce((x, t) => x + G.tokenWidth(t) + CHIP_GAP, 0);
export const chipY = (row) => rowY(row) + 3; // chips sit high in the row, leaving a lane below them for frame 5's gradient flow

// One answer's chips on row `row`. shown = how many chips are drawn; fill = a color for every chip, or (j) → color | null;
// the followed chip carries the selection mark.
export function chipRow(svg, tokens, row, { shown = tokens.length, fill = null, hatched = false, opacity = 1, link = null } = {}) {
  const parent = link ? G.svgEl('g', { 'data-link': link }, svg) : svg; // a linked chip's frame is what the math panel outlines
  tokens.slice(0, shown).forEach((text, j) => {
    const color = typeof fill === 'function' ? fill(j) : fill;
    fade(G.token(parent, { x: chipX(tokens, j), y: chipY(row), text, fill: color, hatched }), opacity);
    if (row === FOLLOWED.row && j === FOLLOWED.token) select(svg, chipX(tokens, j), chipY(row), G.tokenWidth(text), CHIP_H, opacity);
  });
}

export function rowIndexes(svg, opacity = 1) {
  for (let i = 0; i < GROUP_SIZE; i += 1) label(svg, INDEX_X, rowY(i) + ROW_H / 2, String(i + 1), { anchor: 'middle', opacity });
}

export function promptChips(svg, opacity = 1) {
  ['7', '×', '8', '=', '?'].reduce((x, text) => {
    fade(G.token(svg, { x, y: PROMPT.y, text }), opacity);
    return x + G.tokenWidth(text) + CHIP_GAP;
  }, PROMPT.x);
}

// ---- number columns (R, A): a pending cell is null (blank, neutral); fills are 'ok' / 'bad' / null per cell ----
const glyphValues = (values) => values.map((v) => (v == null ? Number.NaN : v));
// The R column prints whole numbers; the advantage column always prints two decimals, "0.00" included.
const printers = { reward: (v) => (Number.isNaN(v) ? '' : String(Math.round(v))), advantage: (v) => (Number.isNaN(v) ? '' : signed2(v)) };

export function column(svg, { origin, values, header, fills = null, opacity = 1, link, kind = fills ? 'reward' : 'advantage' }) {
  const parent = link ? G.svgEl('g', { 'data-link': link }, svg) : svg;
  if (link) G.svgEl('rect', { class: 'g-frame', x: origin.x + 1.5, y: origin.y + 1.5, width: CELL - 3, height: values.length * CELL - 3, rx: 3, fill: 'none', stroke: 'none' }, parent);
  const v = G.vector(parent, { x: origin.x, y: origin.y, values: glyphValues(values), cell: CELL, maxAbs: ADV_MAX_ABS, label: header, format: printers[kind], fill: fills });
  return fade(v, opacity);
}

export const rewardFills = (rewards, shown = rewards.length) => rewards.map((r, i) => (i < shown ? (r === 1 ? 'ok' : 'bad') : null));

// The mean rule across the R and A columns: a horizontal line at the group mean on a 0 → 1 scale up the column.
export const ruleY = (value) => COL_BOTTOM - 8 - value * 300;
export function meanRule(svg, value, text, opacity = 1) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  const line = G.svgEl('line', { x1: RULE_X.from, x2: RULE_X.to, y1: ruleY(value), y2: ruleY(value), class: 'g-rule' }, g);
  line.style.stroke = 'var(--ink)';
  line.style.strokeWidth = '2';
  fade(g, opacity);
  label(svg, RULE_X.to + 8, ruleY(value), text, { opacity });
}
