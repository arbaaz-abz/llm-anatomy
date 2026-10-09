// quantization frames 1–7: the eight-weight row, quantized (storyboard §5). Frames 1–6 draw a scene (scenes.js) and cross-fade what
// differs from the previous frame's scene; frame 7 lays MXFP4 and NVFP4 side by side.
import * as G from '@shared/glyphs.js';
import { WEIGHTS, FOLLOWED } from './numbers.js';
import { cellText, codeText } from './format.js';
import { sceneFor, compareScene } from './scenes.js';
import {
  CELL, LEFT, ROW_X, ROW_W, ROWS, NEUTRAL, MAX_WEIGHT, MAX_ERROR, seg, lerp, ease, arriving, leaving, layer, note, rowLabel, linkGroup, rule, fadePair,
} from './stage.js';

const DOT_MOVE = [0.25, 0.85]; // when a dot slides to its grid value inside a frame

// A row of eight cells at NUMBER_CELL; `shown` draws the first cells and leaves empty slots for the rest.
function cellRow(parent, y, cells, { maxAbs, format, shown = cells.length }) {
  const g = layer(parent);
  G.vector(g, { x: ROW_X, y, values: cells.slice(0, shown).map((c) => c.value), cell: CELL, orient: 'row', maxAbs, format });
  for (let i = shown; i < cells.length; i += 1) {
    const slot = G.svgEl('g', { class: 'glyph g-empty' }, g);
    G.svgEl('rect', { class: 'g-frame', x: ROW_X + i * CELL + 1.5, y: y + 1.5, width: CELL - 3, height: CELL - 3, rx: 3 }, slot);
  }
  return g;
}

const weightCells = () => WEIGHTS.map((value) => ({ value, text: cellText(value) }));

function followedMark(parent, y) {
  G.selectionMark(parent, { x: ROW_X + FOLLOWED * CELL + 1.5, y: y + 1.5, w: CELL - 3, h: CELL - 3 });
}

// The weights row, always on stage; in frame 1 its cells fill left to right and a byte counter counts 2 bytes per cell.
function drawWeights(svg, scene, p) {
  const filling = scene.counter === true;
  const shown = filling ? Math.ceil(seg(p, 0.1, 0.8) * WEIGHTS.length - 1e-9) : WEIGHTS.length;
  cellRow(svg, ROWS.weights, weightCells(), { maxAbs: MAX_WEIGHT, format: cellText, shown });
  rowLabel(svg, ROWS.weights, 'weights');
  if (shown > FOLLOWED) followedMark(svg, ROWS.weights);
  if (filling) {
    note(svg, ROW_X + ROW_W + 16, ROWS.weights - 4, 'bytes');
    G.cell(svg, { x: ROW_X + ROW_W + 16, y: ROWS.weights, size: CELL, v: 2 * shown, maxAbs: NEUTRAL, format: String });
  }
}

function drawChips(parent, chips) {
  chips.forEach((c) => {
    const w = G.tokenWidth(c.text);
    const x = ROW_X + ((c.from + c.to + 1) / 2) * CELL - w / 2;
    const link = linkGroup(parent, 'scale', { x, y: ROWS.chips, w, h: 24 });
    G.token(link, { x, y: ROWS.chips, text: c.text });
  });
}

function drawCodes(parent, cells) {
  const link = linkGroup(parent, 'code', { x: ROW_X, y: ROWS.codes, w: ROW_W, h: CELL });
  cellRow(link, ROWS.codes, cells, { maxAbs: NEUTRAL, format: codeText });
  rowLabel(parent, ROWS.codes, 'codes');
}

function drawRestored(parent, cells) {
  cellRow(parent, ROWS.restored, cells, { maxAbs: MAX_WEIGHT, format: cellText });
  rowLabel(parent, ROWS.restored, 'restored');
}

function drawErrors(parent, cells) {
  cellRow(parent, ROWS.errors, cells, { maxAbs: MAX_ERROR, format: cellText });
  rowLabel(parent, ROWS.errors, 'error');
}

function drawTags(parent, indexes) {
  indexes.forEach((i) => note(parent, ROW_X + i * CELL + CELL / 2, ROWS.tags, 'became 0', { anchor: 'middle' }));
}

// The number line with its dots at t (0 = each dot at `a`, 1 = at `b`).
function drawLine(parent, line, t) {
  const points = line.points.map((q) => ({ value: lerp(q.a, q.b, t), snapped: true, followed: q.followed }));
  G.numberLine(parent, { x: ROW_X, y: ROWS.line, w: ROW_W, lo: line.lo, hi: line.hi, grid: line.grid, points, label: line.label });
  note(parent, ROW_X - 8, ROWS.line + 28, line.label, { anchor: 'end' });
}

function lineParts(parent, prev, cur, p) {
  const c = cur.line;
  const q = prev?.line;
  if (c && q && c.key === q.key) { drawLine(parent, c, 1); return; }
  if (q && leaving(p) > 0) drawLine(layer(parent, leaving(p)), q, 1);
  if (c && arriving(p) > 0) drawLine(layer(parent, arriving(p)), c, ease(seg(p, ...DOT_MOVE)));
}

const late = (a, b) => ({ enter: (p) => seg(p, a, b) });

// The block boundary between cells 4 and 5, broken where the "became 0" labels sit.
function blockRule(g) {
  const x = ROW_X + ROW_W / 2;
  rule(g, x, ROWS.chips, ROWS.tags - 14);
  rule(g, x, ROWS.errors, ROWS.errors + CELL);
}

function drawScene(svg, cur, prev, p) {
  drawWeights(svg, cur, p);
  fadePair(svg, prev?.rule ? prev : null, cur.rule ? cur : null, p, (g) => blockRule(g), { same: () => true });
  fadePair(svg, prev?.chips, cur.chips, p, drawChips);
  lineParts(svg, prev, cur, p);
  fadePair(svg, prev?.codes, cur.codes, p, drawCodes, cur.line && !prev?.line ? late(0.6, 0.95) : {});
  fadePair(svg, prev?.restored, cur.restored, p, drawRestored, late(0.15, 0.6));
  fadePair(svg, prev?.errors, cur.errors, p, drawErrors, late(0.5, 0.9));
  fadePair(svg, prev?.tags?.length ? prev.tags : null, cur.tags?.length ? cur.tags : null, p, drawTags, late(0.8, 1));
  fadePair(svg, prev?.outlierTag ? prev : null, cur.outlierTag ? cur : null, p, (g) => note(g, ROW_X + (WEIGHTS.length - 0.5) * CELL, ROWS.weights - 6, 'outlier', { anchor: 'middle', cls: '' }), { same: () => true });
  fadePair(svg, prev?.notes, cur.notes, p, (g, notes) => notes.forEach((t, i) => note(g, LEFT, ROWS.notes + i * 16, t, { cls: '' })));
}

// Frames 1–6.
export function drawBlockFrame(svg, index, p) {
  drawScene(svg, sceneFor(index), index > 0 ? sceneFor(index - 1) : null, p);
}

// ---- frame 7: MXFP4 against NVFP4, blocks of four ----
const CMP = Object.freeze({ weights: 6, mxChips: 50, mxCodes: 76, mxRestored: 118, nvChips: 164, nvCodes: 190, nvRestored: 232, line: 278, notes: 344 });

function compareSection(parent, rows, y, header) {
  note(parent, LEFT, y.chips + 17, header, { cls: '' });
  drawChipsAt(parent, rows.chips, y.chips);
  cellRow(parent, y.codes, rows.codes, { maxAbs: NEUTRAL, format: codeText });
  rowLabel(parent, y.codes, 'codes');
  cellRow(parent, y.restored, rows.restored, { maxAbs: MAX_WEIGHT, format: cellText });
  rowLabel(parent, y.restored, 'restored');
}

function drawChipsAt(parent, chips, y) {
  chips.forEach((c) => {
    const w = G.tokenWidth(c.text);
    G.token(parent, { x: ROW_X + ((c.from + c.to + 1) / 2) * CELL - w / 2, y, text: c.text });
  });
}

function errorLabel(parent, y, text) {
  note(parent, ROW_X + ROW_W + 8, y + 16, 'error');
  note(parent, ROW_X + ROW_W + 8, y + 31, text, { cls: '' });
}

function drawCompareScene(parent, p) {
  const s = compareScene();
  cellRow(parent, CMP.weights, weightCells(), { maxAbs: MAX_WEIGHT, format: cellText });
  rowLabel(parent, CMP.weights, 'weights');
  followedMark(parent, CMP.weights);
  compareSection(parent, s.mx, { chips: CMP.mxChips, codes: CMP.mxCodes, restored: CMP.mxRestored }, 'MXFP4 (power-of-two scale)');
  compareSection(parent, s.nv, { chips: CMP.nvChips, codes: CMP.nvCodes, restored: CMP.nvRestored }, 'NVFP4 (8-bit float scale)');
  errorLabel(parent, CMP.mxRestored, s.errors.mx);
  errorLabel(parent, CMP.nvRestored, s.errors.nv);
  rule(parent, ROW_X + ROW_W / 2, CMP.weights, CMP.nvRestored + CELL);
  const slide = ease(seg(p, 0.2, 0.7)); // the MX dot slides from where the NVFP4 scale put it to where the power-of-two scale puts it
  const points = [{ value: lerp(s.nvRatio, s.mxRatio, slide), snapped: true, followed: true }, { value: s.nvRatio, snapped: true, followed: true }];
  G.numberLine(parent, { x: ROW_X, y: CMP.line, w: ROW_W, lo: s.lo, hi: s.hi, grid: s.grid, points, label: 'the followed weight, ÷ scale' });
  note(parent, ROW_X - 8, CMP.line + 28, 'weight ÷ scale', { anchor: 'end' });
  s.notes.forEach((t, i) => note(parent, LEFT, CMP.notes + i * 16, t, { cls: '' }));
}

export const drawCompareEnd = (parent) => drawCompareScene(parent, 1);

export function drawFrame7(svg, p) {
  const out = leaving(p);
  if (out > 0) drawScene(layer(svg, out), sceneFor(5), null, 1);
  drawCompareScene(layer(svg, arriving(p)), p);
}
