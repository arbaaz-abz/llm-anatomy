// Frames 8–10: every row of head A, a second head, then the two heads joined and mixed by W_O (storyboard §5).
import * as G from '@shared/glyphs.js';
import { TOY, multiHead } from '@math/attention.js';
import {
  CELL, QUERY, TOKENS, HEAD_A, HEAD_B, SCALE, HEAT, OUT_ROW, O_GRID, HEADS_Y, HEAD_X, O_ROW_Y, CONCAT, W_O,
  phase, lerp, ease, fade, label, select, numberRow, numberGrid, tokenRow, qkvGrids, wBlock,
} from './stage.js';
import { axisLabels, divideLabel, strip, valueChips, outputRow } from './frames-row.js';

const CONCAT_SAT = multiHead([TOY.heads.A, TOY.heads.B]).concat[QUERY];
const ROW_ONE_NOTE = 'row 1 sees one key, so its weight is 1.0 whatever its score';
const NOTE_POS = { x: HEAT.x + 4 * CELL, y: 272 };
const BLOCK = { left: 46, top: 38, w: 218, h: O_ROW_Y + CELL + 6 - HEADS_Y + 38 }; // a head block: its heatmap and "sat" row
const maskOut = (r) => r.weights.map((row, i) => row.map((v, j) => (r.mask[i][j] ? v : -Infinity)));
const WEIGHTS_A = maskOut(HEAD_A);
const WEIGHTS_B = maskOut(HEAD_B);

function oGrid(svg, { rowFill, satShown, opacity = 1 }) {
  const values = HEAD_A.output.map((row, i) => row.map((v) => {
    if (i === QUERY) return satShown ? v : null;
    return rowFill > 0 ? v * rowFill : null;
  }));
  const g = numberGrid(svg, { ...O_GRID, values, kind: 'output', maxAbs: SCALE.out, label: 'O_A = A·V', link: 'o' });
  if (satShown) select(svg, O_GRID.x, O_GRID.y + QUERY * CELL, 4 * CELL, CELL, opacity);
  return fade(g, opacity);
}

// Frame 8: the masked-score grid becomes the weight heatmap A, row 3 first, then rows 1, 2 and 4 together.
function weightGrid(svg, p) {
  const tRow3 = ease(phase(p, 0, 0.3));
  const tRest = ease(phase(p, 0.3, 1));
  const t = (i) => (i === QUERY ? tRow3 : tRest);
  const cellOf = (scaledToo) => (i, j) => {
    if (!HEAD_A.mask[i][j]) return -Infinity;
    const from = scaledToo ? HEAD_A.scaled[i][j] / SCALE.score : HEAD_A.scaled[i][j];
    return lerp(from, HEAD_A.weights[i][j], t(i));
  };
  const grid = (fn) => TOKENS.map((_, i) => TOKENS.map((__, j) => fn(i, j)));
  numberGrid(svg, {
    ...HEAT, values: grid(cellOf(false)), colors: grid(cellOf(true)), kind: p < 1 ? 'scaled' : 'weight', maxAbs: SCALE.weight,
    mask: HEAD_A.mask, label: 'A', rowLabels: TOKENS, colLabels: TOKENS, link: 'a',
  });
  select(svg, HEAT.x, HEAT.y + QUERY * CELL, 4 * CELL, CELL);
}

export function drawFrame8(svg, p) {
  const stay = 1 - phase(p, 0, 0.3);
  tokenRow(svg);
  if (stay > 0) {
    wBlock(svg, stay);
    qkvGrids(svg, { opacity: stay });
    divideLabel(svg, stay);
    strip(svg, { opacity: stay });
    valueChips(svg, stay);
  }
  weightGrid(svg, p);
  axisLabels(svg);
  const dock = ease(phase(p, 0, 0.3));
  oGrid(svg, { rowFill: phase(p, 0.3, 1), satShown: dock >= 1, opacity: phase(p, 0.2, 0.35) });
  if (dock < 1) outputRow(svg, { x: lerp(OUT_ROW.x, O_GRID.x, dock), y: lerp(OUT_ROW.y, O_GRID.y + QUERY * CELL, dock), labelOpacity: 1 - phase(p, 0, 0.1) });
  label(svg, NOTE_POS.x, NOTE_POS.y, ROW_ONE_NOTE, { anchor: 'end', opacity: phase(p, 0.3, 0.6) });
}

// One head block: its weight heatmap and, unless it is still on its way (frame 9), its "sat" row of O.
function headBlock(svg, name, { x, y, weights, mask, output = null, opacity = 1, frame = opacity, rowFill = () => 1, outFill = 1 }) {
  fade(G.block(svg, { x: x - BLOCK.left, y: y - BLOCK.top, w: BLOCK.w, h: BLOCK.h, label: '' }), frame);
  const g = numberGrid(svg, { x, y, values: weights, kind: 'weight', maxAbs: SCALE.weight, mask, label: `head ${name}: weights`, rowLabels: TOKENS, colLabels: TOKENS, link: 'a' });
  [...g.querySelectorAll('.g-cell')].forEach((c, k) => {
    if (mask[Math.floor(k / 4)][k % 4]) fade(c, rowFill(Math.floor(k / 4)));
  });
  fade(g, opacity);
  select(svg, x, y + QUERY * CELL, 4 * CELL, CELL, opacity);
  if (!output) return;
  const rowY = O_ROW_Y + (y - HEADS_Y);
  const values = output.map((v) => (outFill > 0 ? v * outFill : null));
  fade(numberRow(svg, { x, y: rowY, values, kind: 'output', maxAbs: SCALE.out, link: 'o' }), opacity);
  label(svg, x - 10, rowY + CELL / 2, 'o_sat', { anchor: 'end', opacity });
  select(svg, x, rowY, 4 * CELL, CELL, opacity);
}

const otherRowsNote = (svg, opacity) => label(svg, HEAD_X.A, 274, '3 other rows not shown (see the toy)', { opacity });

function headB(svg, { appear = 1, fillT = 1, outFill = 1 } = {}) {
  headBlock(svg, 'B', {
    x: HEAD_X.B, y: HEADS_Y, weights: WEIGHTS_B, mask: HEAD_B.mask, output: HEAD_B.output[QUERY], opacity: appear,
    rowFill: (i) => phase(fillT * 4, i, i + 1), outFill,
  });
}

// Frame 9 moves head A up, then left, with its "sat" row of O travelling the same way just below it, so the
// two never cross; then head B appears beside it and fills.
export function drawFrame9(svg, p) {
  const out = 1 - phase(p, 0, 0.25);
  const up = ease(phase(p, 0.1, 0.3));
  const left = ease(phase(p, 0.3, 0.6));
  if (out > 0) {
    tokenRow(svg, out);
    axisLabels(svg, out);
    label(svg, NOTE_POS.x, NOTE_POS.y, ROW_ONE_NOTE, { anchor: 'end', opacity: out });
    oGrid(svg, { rowFill: 1, satShown: false, opacity: out });
  }
  const x = lerp(HEAT.x, HEAD_X.A, left);
  const y = lerp(HEAT.y, HEADS_Y, up);
  headBlock(svg, 'A', { x, y, weights: WEIGHTS_A, mask: HEAD_A.mask, frame: phase(p, 0.55, 0.7) });
  const satX = lerp(O_GRID.x, HEAD_X.A, left);
  const satY = lerp(O_GRID.y + QUERY * CELL, O_ROW_Y, up);
  numberRow(svg, { x: satX, y: satY, values: HEAD_A.output[QUERY], kind: 'output', maxAbs: SCALE.out, link: 'o' });
  label(svg, satX - 10, satY + CELL / 2, 'o_sat', { anchor: 'end', opacity: left });
  select(svg, satX, satY, 4 * CELL, CELL);
  otherRowsNote(svg, phase(p, 0.55, 0.7));
  headB(svg, { appear: phase(p, 0.6, 0.7), fillT: phase(p, 0.7, 0.9), outFill: phase(p, 0.9, 1) });
}

export function drawFrame10(svg, p) {
  headBlock(svg, 'A', { x: HEAD_X.A, y: HEADS_Y, weights: WEIGHTS_A, mask: HEAD_A.mask, output: HEAD_A.output[QUERY] });
  headB(svg);
  otherRowsNote(svg, 1);
  const appear = phase(p, 0, 0.1);
  const slide = ease(phase(p, 0.1, 0.5));
  const values = CONCAT_SAT.map((v) => (slide >= 1 ? v : null));
  fade(numberRow(svg, { ...CONCAT, values, kind: 'output', maxAbs: SCALE.out, link: 'o' }), appear);
  label(svg, CONCAT.x - 10, CONCAT.y + CELL / 2, 'sat, joined', { anchor: 'end', opacity: appear });
  select(svg, CONCAT.x, CONCAT.y, 8 * CELL, CELL, appear);
  fade(G.block(svg, { ...W_O, label: 'W_O [8 × 8]' }), appear);
  if (p >= 0.5) G.flow(svg, { from: [CONCAT.x + 8 * CELL + 4, CONCAT.y + CELL / 2], to: [W_O.x - 2, W_O.y + W_O.h / 2], carry: 'activation', progress: phase(p, 0.5, 0.8) });
  const words = phase(p, 0.7, 1);
  label(svg, CONCAT.x, 338, 'each of the 8 outputs is a weighted mix of all 8 joined numbers', { opacity: words });
  label(svg, W_O.x + W_O.w, 354, '→ added to the residual stream', { anchor: 'end', opacity: words });
  const ghost = 1 - phase(p, 0.45, 0.55);
  if (slide <= 0 || ghost <= 0) return;
  [[HEAD_X.A, HEAD_A], [HEAD_X.B, HEAD_B]].forEach(([fromX, r], h) => {
    const gx = lerp(fromX, CONCAT.x + h * 4 * CELL, slide);
    const gy = lerp(O_ROW_Y, CONCAT.y, slide);
    fade(numberRow(svg, { x: gx, y: gy, values: r.output[QUERY], kind: 'output', maxAbs: SCALE.out }), ghost);
  });
}
