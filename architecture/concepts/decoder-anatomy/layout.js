// decoder-anatomy stage layout: fixed positions shared by every frame, and small drawing helpers built on the glyph
// library. The stream keeps its anchor from frame 3 to 9, the block lane from 3 to 7, the lifted strip in 4, 5, 6 and 8.
import * as G from '@shared/glyphs.js';
import { TOKENS, MAX_ABS } from './stream.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL; // numbers the learner reads
export const SMALL = 18; // hover-only cells (README visual constraints)
export const CHIPS = Object.freeze({ x: 64, y: 6, gap: 8 });
export const STREAM = Object.freeze({ x: 64, y: 58 });
export const LANE = Object.freeze({ x: 424, y: 40, w: 152 });
export const DETAIL = Object.freeze({ x: 234, y: 34, w: 182, h: 116 });
export const STRIP = Object.freeze({ x: 64, a: 166, b: 222, c: 278 });
export const PATCH = Object.freeze({ x: 404, y: 190 });
export const STACK_H = Object.freeze({ two: 176, folded: 212 }); // blockStack heights: count 2, and the "⋮ × N" fold
const ROW_LABEL_GAP = 10;

// Cat's-ear crop (4 × 4 greys, image content, never the value scale): the same patch the gallery draws.
export const EAR = Object.freeze([0.18, 0.22, 0.78, 0.9, 0.2, 0.55, 0.86, 0.95, 0.42, 0.76, 0.9, 0.84, 0.7, 0.86, 0.8, 0.62]);

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
export const dip = (t) => 1 - 0.55 * Math.sin(Math.PI * clamp01(t)); // "dim and brighten" for the norm stage

export const chipWidths = (words) => words.map((w) => G.tokenWidth(w));
export const chipX = (i, words = TOKENS) => CHIPS.x + chipWidths(words).slice(0, i).reduce((s, w) => s + w + CHIPS.gap, 0);

// A group at the given opacity (attribute omitted when fully opaque, so frames at rest stay plain).
export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

export function chips(parent, words = TOKENS, { opacity = () => 1, dx = () => 0 } = {}) {
  words.forEach((text, i) => G.token(layer(parent, opacity(i)), { x: chipX(i, words) + dx(i), y: CHIPS.y, text, index: i + 1 }));
}

// A row of cells that a math term can name: the wrapper carries data-link and an invisible frame for the hover outline.
export function linkedRow(parent, letter, { x, y, values, cell, maxAbs = MAX_ABS, label }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: values.length * cell + 2, height: cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  G.vector(g, { x, y, values, cell, orient: 'row', maxAbs, label });
  return g;
}

// The residual stream X [n × 8] at its fixed anchor; `selected` rows carry the one selection outline.
export function stream(parent, rows, { cell = SMALL, selected = [], words = TOKENS } = {}) {
  const g = G.matrix(parent, { x: STREAM.x, y: STREAM.y, values: rows, cell, maxAbs: MAX_ABS, label: cell >= CELL ? 'residual stream X' : 'stream X', rowLabels: words.slice(0, rows.length) });
  g.setAttribute('data-link', 'x');
  selected.forEach((r) => G.selectionMark(parent, { x: STREAM.x, y: STREAM.y + r * cell, w: rows[0].length * cell, h: cell }));
  return g;
}

export const streamRowY = (r, cell = SMALL) => STREAM.y + r * cell;
export const rowLabelX = (x) => x - ROW_LABEL_GAP;

// The block lane: blockStack at its fixed place. fold = true draws "block 1, ⋮ × N, block N".
export function lane(parent, { halves = ['attention', 'MLP'], active = null, fold = false, opacity = 1 } = {}) {
  const g = G.blockStack(layer(parent, opacity), { x: LANE.x, y: LANE.y, w: LANE.w, count: fold ? 3 : 2, shown: fold ? 1 : 2, halves, active, countLabel: '⋮ × N' });
  if (fold) {
    // blockStack prints "block 3" for the last of three; on this stage the last block is block N (shared request filed).
    const last = [...g.querySelectorAll('text.g-sub')].find((t) => t.textContent === 'block 3');
    if (last) last.textContent = 'block N';
  }
  return g;
}

// Where the lane's blocks sit (blockStack geometry: 16 px header, 26 px halves, 6 px between, 8 px pad, 12 px gaps).
export const laneBlockTop = (index, fold) => LANE.y + (index === 0 ? 0 : (fold ? 130 : 94));

// A small "sheet" of the four rows travelling down the lane (frames 3 and 7): tiny hover-only cells.
export function sheet(parent, rows, y, opacity) {
  const g = layer(parent, opacity);
  G.matrix(g, { x: LANE.x + 2, y, values: rows, cell: 5, maxAbs: MAX_ABS });
  return g;
}

// A frame of the given size with a label: an empty slot (frame 1's residual stream before any row arrives).
export function emptyFrame(parent, { x, y, w, h, label, rowLabels = [], cell = CELL }) {
  const g = G.svgEl('g', { class: 'glyph g-matrix g-empty', transform: `translate(${x} ${y})` }, parent);
  G.svgEl('rect', { class: 'g-frame', x: -1, y: -1, width: w + 2, height: h + 2, rx: 3 }, g);
  const t = G.svgEl('text', { x: 0, y: -10, class: 'g-label' }, g);
  t.textContent = label;
  rowLabels.forEach((r, i) => {
    const rl = G.svgEl('text', { x: -ROW_LABEL_GAP, y: i * cell + cell / 2, class: 'g-label', 'text-anchor': 'end', 'dominant-baseline': 'central' }, g);
    rl.textContent = r;
  });
  return g;
}
