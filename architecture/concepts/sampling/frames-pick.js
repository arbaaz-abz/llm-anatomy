// Frames 1–3: the logits and their probabilities, greedy decoding, and a random draw from the strip (storyboard §5).
// Each frame is a pure function of its progress p (0 → 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { probText, scoreText, fmt3 } from './format.js';
import {
  CELL, ROW_X, ROW_W, RIGHT_X, CHIP_Y, FIVE_Y, NOTE_PAD, SCORE_CELLS, OTHERS, WORDS, HANDOFF, seg, lerp, ease, leaving, layer, note,
  chips, slotChip, wordLabels, fiveRow, cellMark, othersNote, cellsAt, probsAt, tick, SLOT_X,
} from './stage.js';

const PROBS = cellsAt(1);
const TOTAL = probsAt(1).reduce((s, v) => s + v, 0);
const TRAVEL3 = Object.freeze({ from: 0.8, to: 0.95 });
const ON = 0; // the followed token's cell in the five-cell row
const PROB_ROW_BOTTOM = FIVE_Y.probs + CELL;

// Cells type in left to right as t goes 0 → 1 (null = not computed yet); never scaled up from 0.
const typeIn = (values, t) => values.map((v, j) => (t * values.length >= j + 1 - 1e-9 ? v : null));

// ---- frame 1: scores, then the probabilities softmax makes from them ----
// The body (everything but the chips) so frame 2 can carry it as it is.
export function rowsBody(svg, p) {
  const scoresT = seg(p, 0, 0.35);
  const probsT = seg(p, 0.5, 0.9);
  wordLabels(svg, FIVE_Y.scores - 8, scoresT > 0 ? 1 : 0);
  fiveRow(svg, { y: FIVE_Y.scores, values: typeIn(SCORE_CELLS, scoresT), kind: 'score', name: 'scores', link: 'z' });
  note(svg, RIGHT_X, FIVE_Y.scores + CELL / 2, 'these scores are the logits', { opacity: seg(p, 0.3, 0.4) });
  if (scoresT >= 1) note(svg, ROW_X + ROW_W, FIVE_Y.scores + CELL + NOTE_PAD, `${scoreText(SCORE_CELLS[4])} each`, { anchor: 'end' });
  const flowT = seg(p, 0.35, 0.55);
  if (flowT > 0) {
    G.flow(svg, { from: [ROW_X + ROW_W / 2, FIVE_Y.scores + CELL + 28], to: [ROW_X + ROW_W / 2, FIVE_Y.probs - 6], carry: 'activation', progress: flowT });
    note(svg, ROW_X + ROW_W / 2 + 12, (FIVE_Y.scores + CELL + FIVE_Y.probs) / 2 + 10, 'softmax', { opacity: flowT });
  }
  if (probsT <= 0) return;
  fiveRow(svg, { y: FIVE_Y.probs, values: typeIn(PROBS, probsT), kind: 'prob', name: 'probs', link: 'p' });
  if (probsT >= 1) {
    note(svg, ROW_X + ROW_W, PROB_ROW_BOTTOM + NOTE_PAD, othersNote(probsAt(1)), { anchor: 'end' });
    note(svg, RIGHT_X, FIVE_Y.probs + CELL / 2, `Σ = ${TOTAL.toFixed(3)}`);
  }
  note(svg, ROW_X, 292, 'top: scores, any size · bottom: probabilities, add to 1', { opacity: seg(p, 0.9, 1) });
}

export function drawFrame1(svg, p) {
  chips(svg, [{ slot: 'five', text: '?' }]);
  rowsBody(svg, p);
}

// ---- frame 2: greedy decoding takes the top cell ----
const TRAVEL = Object.freeze({ from: 0.25, to: 0.8 });
const onCell = () => ({ x: ROW_X + (CELL - G.tokenWidth('on')) / 2, y: FIVE_Y.probs + (CELL - 24) / 2 });

export function greedyBody(svg, p, opacity = 1) {
  const inner = layer(svg, opacity);
  rowsBody(inner, 1);
  cellMark(inner, FIVE_Y.probs, ON, seg(p, 0, 0.2));
  note(inner, RIGHT_X, FIVE_Y.probs + CELL / 2 + 26, 'greedy: always the top', { opacity: seg(p, 0, 0.2) });
  note(inner, ROW_X, 314, `pick: ${WORDS[ON]} (${probText(PROBS[ON])}), every time`, { opacity: seg(p, 0.8, 1) });
}

export function drawFrame2(svg, p) {
  const travel = ease(seg(p, TRAVEL.from, TRAVEL.to));
  chips(svg, [{ slot: 'five', text: p >= TRAVEL.to ? 'on' : '?' }]);
  greedyBody(svg, p);
  if (travel > 0 && p < TRAVEL.to) {
    const from = onCell();
    slotChip(svg, 'five', 'on', { at: { x: lerp(from.x, SLOT_X.five, travel), y: lerp(from.y, CHIP_Y, travel) } });
  }
}

// ---- frame 3: the same probabilities as a strip, and a random number that lands in one slice ----
const BAR = Object.freeze({ x: 10, y: 112, w: 560, h: 20 });
const U = 0.55; // hand-picked stand-in (storyboard §4)
const HIT = 1; // "." owns the slice that u lands in
const PARTS = Object.freeze(PROBS.map((value, i) => ({ name: WORDS[i], value: i < 4 ? value : value * OTHERS, hue: i + 1 })));
const ticks = () => {
  let running = 0;
  return PROBS.slice(0, 4).map((v) => { running += v; return running; });
};
const BOUNDARIES = Object.freeze(ticks());
const barX = (share) => BAR.x + share * BAR.w;

// Boundary labels alternate between two heights, so "0.714" and "0.767" (29 px apart) never touch.
const TICK_DEPTH = Object.freeze([26, 50]);
const LABEL_GAP = 10;

function strip(svg, opacity) {
  const inner = layer(svg, opacity);
  G.shareBar(inner, { ...BAR, parts: PARTS, format: (share) => fmt3(share), label: 'probability strip' });
  BOUNDARIES.forEach((b, i) => {
    const depth = TICK_DEPTH[i % 2];
    tick(inner, barX(b), BAR.y + BAR.h, BAR.y + BAR.h + depth);
    note(inner, barX(b), BAR.y + BAR.h + depth + LABEL_GAP, fmt3(b), { anchor: 'middle' });
  });
}

// The slice the marker lands in: the strip's boundaries 0.390 and 0.627, minus the 2 px gap shareBar leaves.
const sliceMark = (svg, opacity) => {
  if (opacity <= 0) return;
  G.selectionMark(layer(svg, opacity), { x: barX(BOUNDARIES[0]), y: BAR.y, w: PROBS[HIT] * BAR.w - 2, h: BAR.h });
};

function marker(svg, p) {
  const u = U * ease(seg(p, 0.3, 0.7));
  const x = barX(u);
  const visible = seg(p, 0.05, 0.25) * (1 - seg(p, 0.75, 0.85));
  if (visible <= 0) return;
  const g = layer(svg, visible);
  note(g, x, BAR.y - 8, '▼', { anchor: 'middle' });
  note(g, Math.min(Math.max(x, BAR.x + 36), BAR.x + BAR.w - 36), BAR.y - 26, `u = ${u.toFixed(2)}`, { anchor: 'middle' });
}

// Frame 3's body (everything but the chips): the strip, the marker, the slice it lands in and the two lines under it.
export function barBody(svg, p, opacity = 1) {
  const inner = layer(svg, opacity);
  strip(inner, seg(p, 0.05, 0.25));
  marker(inner, p);
  sliceMark(inner, seg(p, 0.7, 0.8));
  const line = seg(p, 0.7, 0.85);
  note(inner, BAR.x, 290, `u = ${U} lands in the slice from ${fmt3(BOUNDARIES[0])} to ${fmt3(BOUNDARIES[1])}, which is "."`, { opacity: line });
  note(inner, BAR.x, 310, 'each boundary prints the running total of the probabilities', { opacity: line });
}

export function drawFrame3(svg, p) {
  const swap = seg(p, 0, HANDOFF);
  chips(svg, [
    { slot: 'five', text: 'on', opacity: 1 - swap, mark: 0 },
    { slot: 'five', text: p >= TRAVEL3.to ? '.' : '?', opacity: swap },
  ]);
  if (p < HANDOFF) greedyBody(svg, 1, leaving(p));
  barBody(svg, p);
  const travel = ease(seg(p, TRAVEL3.from, TRAVEL3.to));
  if (travel > 0 && p < TRAVEL3.to) {
    const from = { x: barX(BOUNDARIES[0]) + 6, y: BAR.y - 4 };
    slotChip(svg, 'five', '.', { at: { x: lerp(from.x, SLOT_X.five, travel), y: lerp(from.y, CHIP_Y, travel) } });
  }
}
