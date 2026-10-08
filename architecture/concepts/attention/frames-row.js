// Frames 1–7: one row of attention for the query "sat" (storyboard §5). Each frame is a pure function
// of its progress p (0 → 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { expansion } from './format.js';
import {
  CELL, GRID, QUERY, TOKENS, HEAD_A, QKV, SCALE, TOKEN_X, TOKEN_Y, MAT_Y, MAT_X, STRIP_X, STRIP_Y, SIDE_X, NOTE_Y,
  HEAT, SCORE_ROW, CHIP_X, OUT_ROW, phase, lerp, ease, fade, label, select, numberRow, numberGrid, tokenRow, qkvGrids,
  wBlock, maskUpTo,
} from './stage.js';

const ROW = HEAD_A.scores[QUERY].map((_, j) => j);
const VISIBLE = ROW.filter((j) => HEAD_A.mask[QUERY][j]);
const HEAT_ROW_Y = HEAT.y + QUERY * CELL;
const STRIP_H = STRIP_Y[2] + CELL - STRIP_Y[0];

// The parts every frame from 1 to 7 keeps: chips, the W block and Q, K, V.
function base(svg, rowsShown = 4) {
  tokenRow(svg);
  wBlock(svg);
  qkvGrids(svg, { rowsShown });
}

export function axisLabels(svg, opacity = 1) {
  label(svg, HEAT.x - 10, HEAT.y - 12, 'queries ↓', { anchor: 'end', opacity });
  label(svg, HEAT.x + 4 * CELL, HEAT.y - 28, 'keys →', { anchor: 'end', opacity });
}

export const divideLabel = (svg, opacity = 1) => label(svg, HEAT.x, 50, '÷ √d_head = ÷ √4 = ÷ 2', { opacity });

const maskedScaled = (mask) => HEAD_A.scaled.map((row, i) => row.map((v, j) => (mask[i][j] ? v : -Infinity)));

// The score heatmap S (frames 3–7); its row 3 is "sat".
function scoreGrid(svg, { values, kind, title, mask }) {
  numberGrid(svg, { ...HEAT, values, kind, maxAbs: SCALE.score, mask, label: title, rowLabels: TOKENS, colLabels: TOKENS, link: 's' });
  select(svg, HEAT.x, HEAT_ROW_Y, 4 * CELL, CELL);
}

export function drawFrame1(svg, p) {
  base(svg, p * 4);
  if (p >= 1) return;
  const i = Math.min(Math.floor(p * 4), 3);
  const cx = TOKEN_X[i] + G.tokenWidth(TOKENS[i]) / 2;
  G.flow(svg, { from: [cx, TOKEN_Y + 26], to: [cx, 72], carry: 'activation', progress: p * 4 - i });
}

const SWEEP = { from: 0.16, span: 0.17 }; // four keys, one per span, after the lift
const keysDone = (p) => Math.min(Math.max(Math.floor((p - SWEEP.from) / SWEEP.span + 1e-9), 0), 4);

function liftedQuery(svg, lift, opacity = 1) {
  const cell = lerp(GRID, CELL, lift);
  const x = lerp(MAT_X.q, STRIP_X, lift);
  const y = lerp(MAT_Y + QUERY * GRID, STRIP_Y[0], lift);
  fade(numberRow(svg, { x, y, values: QKV.Q[QUERY], kind: 'score', maxAbs: SCALE.qkv, link: 'q', cell }), opacity);
  label(svg, STRIP_X - 10, STRIP_Y[0] + CELL / 2, 'q_sat', { anchor: 'end', opacity: lift * opacity });
  select(svg, x, y, 4 * cell, cell, opacity);
}

function keyRow(svg, j, opacity = 1) {
  fade(numberRow(svg, { x: STRIP_X, y: STRIP_Y[1], values: QKV.K[j], kind: 'score', maxAbs: SCALE.qkv, link: 'k' }), opacity);
  label(svg, STRIP_X - 10, STRIP_Y[1] + CELL / 2, `k_${TOKENS[j]}`, { anchor: 'end', opacity });
}

const CAT = 1;
const catExpansion = () => `q_sat · k_cat = ${expansion(QKV.Q[QUERY], QKV.K[CAT], HEAD_A.scores[QUERY][CAT])}`;
const expansionLine = (svg, opacity) => label(svg, MAT_X.q, STRIP_Y[2] + CELL / 2, catExpansion(), { opacity });

export function drawFrame2(svg, p) {
  base(svg);
  liftedQuery(svg, ease(phase(p, 0, SWEEP.from)));
  const done = keysDone(p);
  const sweeping = p >= SWEEP.from && done < 4;
  const j = sweeping ? done : CAT;
  if (p >= SWEEP.from) keyRow(svg, j);
  const scores = HEAD_A.scores[QUERY].map((v, k) => (k < done ? v : null));
  numberRow(svg, { ...SCORE_ROW, values: scores, kind: 'score', maxAbs: SCALE.score, label: 'q·k', link: 's' });
  select(svg, SCORE_ROW.x, SCORE_ROW.y, 4 * CELL, CELL);
  if (sweeping) {
    const t = (p - SWEEP.from - j * SWEEP.span) / SWEEP.span;
    G.flow(svg, { from: [MAT_X.k + 4 * GRID + 3, MAT_Y + j * GRID + GRID / 2], to: [SCORE_ROW.x + j * CELL + CELL / 2, SCORE_ROW.y - 3], carry: 'activation', progress: t });
  }
  expansionLine(svg, phase(p, 0.84, 1));
}

export function drawFrame3(svg, p) {
  base(svg);
  const leave = 1 - phase(p, 0, 0.3);
  if (leave > 0) {
    liftedQuery(svg, 1, leave);
    keyRow(svg, CAT, leave);
    expansionLine(svg, leave);
  }
  const dock = ease(phase(p, 0, 0.3));
  const cols = phase(p, 0.3, 1) * 4;
  const values = HEAD_A.scores.map((row, i) => row.map((v, j) => {
    if (i === QUERY) return dock >= 1 ? v : null;
    return cols >= j + 1 - 1e-9 ? v : null;
  }));
  numberGrid(svg, { ...HEAT, values, kind: 'score', maxAbs: SCALE.score, label: 'S = QKᵀ', rowLabels: TOKENS, colLabels: TOKENS, link: 's' });
  axisLabels(svg, phase(p, 0.3, 1));
  if (dock >= 1) {
    select(svg, HEAT.x, HEAT_ROW_Y, 4 * CELL, CELL);
    return;
  }
  const x = lerp(SCORE_ROW.x, HEAT.x, dock);
  const y = lerp(SCORE_ROW.y, HEAT_ROW_Y, dock);
  numberRow(svg, { x, y, values: HEAD_A.scores[QUERY], kind: 'score', maxAbs: SCALE.score, link: 's' });
  select(svg, x, y, 4 * CELL, CELL);
}

export function drawFrame4(svg, p) {
  base(svg);
  const t = ease(p);
  const values = HEAD_A.scores.map((row, i) => row.map((v, j) => lerp(v, HEAD_A.scaled[i][j], t)));
  scoreGrid(svg, { values, kind: 'scaled', title: 'S/√d_head' });
  axisLabels(svg);
  divideLabel(svg, phase(p, 0, 0.3));
}

export function drawFrame5(svg, p) {
  base(svg);
  const mask = maskUpTo(Math.floor(p * 5 + 1e-9));
  scoreGrid(svg, { values: maskedScaled(mask), kind: 'scaled', title: 'S/√d_head', mask });
  axisLabels(svg);
  divideLabel(svg);
}

// ---- frames 6–7: the row strip (scaled, exp, weight), then the weighted sum of the values ----
const EXP_TEXTS = ROW.map((j) => (VISIBLE.includes(j) ? undefined : '0')); // the masked cell shows 0, hatched
const shownBy = (t, k) => t * VISIBLE.length >= k + 1 - 1e-9;
const stripRow = (row, t) => row.map((v, j) => {
  if (!VISIBLE.includes(j)) return -Infinity;
  return shownBy(t, VISIBLE.indexOf(j)) ? v : null;
});

function stripLabels(svg, opacity) {
  ['scaled', 'exp', 'weight'].forEach((name, r) => label(svg, STRIP_X - 10, STRIP_Y[r] + CELL / 2, name, { anchor: 'end', opacity }));
}

// The strip with exp filled to `expT` and weights to `wT` (0..1); `lift` moves the scaled row out of S.
export function strip(svg, { lift = 1, expT = 1, wT = 1, opacity = 1 }) {
  const x = lerp(HEAT.x, STRIP_X, lift);
  const y = lerp(HEAT_ROW_Y, STRIP_Y[0], lift);
  fade(numberRow(svg, { x, y, values: HEAD_A.masked[QUERY], kind: 'scaled', maxAbs: SCALE.score }), opacity);
  stripLabels(svg, lift * opacity);
  const rows = [
    { y: STRIP_Y[1], values: stripRow(HEAD_A.exps[QUERY], expT), kind: 'exp', maxAbs: SCALE.score },
    { y: STRIP_Y[2], values: stripRow(HEAD_A.weights[QUERY], wT), kind: 'weight', maxAbs: SCALE.weight, link: 'a' },
  ];
  rows.forEach((r) => fade(numberRow(svg, { x: STRIP_X, ...r, texts: EXP_TEXTS }), lift * opacity));
  const weightSum = VISIBLE.reduce((s, j, k) => s + (shownBy(wT, k) ? HEAD_A.weights[QUERY][j] : 0), 0);
  label(svg, SIDE_X, STRIP_Y[1] + CELL / 2, `sum ${HEAD_A.expSums[QUERY].toFixed(3)}`, { opacity: expT >= 1 ? opacity : 0 });
  label(svg, SIDE_X, STRIP_Y[2] + CELL / 2, `Σ = ${weightSum.toFixed(3)}`, { opacity: wT > 0 ? opacity : 0 });
  label(svg, MAT_X.q, NOTE_Y, 'three stages of one row, each grid on its own scale; read the numbers', { opacity: lift * opacity });
  if (lift < 1) select(svg, x, y, 4 * CELL, CELL, opacity);
  else select(svg, STRIP_X, STRIP_Y[0], 4 * CELL, STRIP_H, opacity);
}

export function maskedGrid(svg, title = 'S/√d_head') {
  scoreGrid(svg, { values: HEAD_A.masked, kind: 'scaled', title, mask: HEAD_A.mask });
}

export function drawFrame6(svg, p) {
  base(svg);
  maskedGrid(svg);
  axisLabels(svg);
  divideLabel(svg);
  strip(svg, { lift: ease(phase(p, 0, 0.2)), expT: phase(p, 0.2, 0.6), wT: phase(p, 0.6, 1) });
}

// Frame 7 pieces that frame 8 fades out: the weight chips beside V and the hatched (masked) V row.
export function valueChips(svg, opacity = 1) {
  VISIBLE.forEach((j) => fade(numberRow(svg, { x: CHIP_X, y: MAT_Y + j * GRID, values: [HEAD_A.weights[QUERY][j]], kind: 'weight', maxAbs: SCALE.weight, cell: GRID }), opacity));
  ROW.filter((j) => !VISIBLE.includes(j)).forEach((j) => {
    const g = fade(G.svgEl('g', { class: 'glyph g-excluded' }, svg), opacity);
    const box = { x: MAT_X.v, y: MAT_Y + j * GRID, width: 4 * GRID, height: GRID, rx: 3 };
    G.svgEl('rect', box, g).style.fill = 'var(--surface)';
    G.svgEl('rect', { ...box, class: 'g-hatch', fill: G.hatchFill(svg) }, g);
  });
}

export function outputRow(svg, { x = OUT_ROW.x, y = OUT_ROW.y, fill = 1, opacity = 1, labelOpacity = opacity } = {}) {
  const values = HEAD_A.output[QUERY].map((v) => (fill > 0 ? v * fill : null));
  fade(numberRow(svg, { x, y, values, kind: 'output', maxAbs: SCALE.out, link: 'o' }), opacity);
  label(svg, x, y - 10, 'o_sat', { opacity: labelOpacity });
  select(svg, x, y, 4 * CELL, CELL, opacity);
}

export function drawFrame7(svg, p) {
  base(svg);
  maskedGrid(svg);
  axisLabels(svg);
  divideLabel(svg);
  strip(svg, {});
  if (p < 0.3) G.flow(svg, { from: [STRIP_X + 4 * CELL + 4, STRIP_Y[2] - 4], to: [CHIP_X + GRID / 2, MAT_Y + 4 * GRID + 4], carry: 'activation', progress: phase(p, 0, 0.3) });
  valueChips(svg, phase(p, 0.1, 0.3));
  outputRow(svg, { fill: phase(p, 0.6, 1) });
  const slide = ease(phase(p, 0.3, 0.8));
  const ghost = 1 - phase(p, 0.7, 0.85);
  if (slide <= 0 || ghost <= 0) return;
  VISIBLE.forEach((j) => {
    const x = lerp(MAT_X.v, OUT_ROW.x, slide);
    const y = lerp(MAT_Y + j * GRID, OUT_ROW.y + (CELL - GRID) / 2, slide);
    fade(G.vector(svg, { x, y, values: QKV.V[j], cell: GRID, orient: 'row', maxAbs: SCALE.qkv }), ghost);
  });
}
