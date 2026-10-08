// moe frames 8–10: a batch of 128 tokens drifts toward a few experts; a selection-only bias on one token (frame 9) and over
// six steps (frame 10) evens it out. Nothing is followed in the batch frames (8 and 10); frame 9 returns to "sat".
import * as G from '@shared/glyphs.js';
import { SAT, SCORES_SAT, BIAS_SAT, CHOOSE_SAT, PICKS, GATES, PICKS_BIASED, GATES_BIASED, RUN, MAX_LOAD, FAIR_SHARE, LAST_STEP } from './numbers.js';
import {
  X0, CELL, WIDTH8, NOTES_X, SCORES_Y, EXPERTS,
  seg, lerp, arriving, layer, note, fade, fadeOutOf, header, scoreRow, expertRow, linkedCells, colCenter, expertLeft,
} from './stage.js';
import { drawFrame7 } from './frames-count.js';
import { gateText, gatesText, picksText, scoreText, biasText, imbalanceText, expertName } from './format.js';

const BARS_BASE = 312;
const BARS_H = 170;
const BIAS_Y = BARS_BASE + 30;
const NAMES = SCORES_SAT.map((_, e) => expertName(e));
const GATE_MAX = 1;
const SAT_PICKS = PICKS[SAT];
const ROW_STRIDE = CELL + 4; // frame 9's three rows (scores, bias, choose) sit one cell apart
const BIAS_ROW_Y = SCORES_Y + ROW_STRIDE;
const CHOOSE_ROW_Y = BIAS_ROW_Y + ROW_STRIDE;
const EXPERTS_Y_9 = CHOOSE_ROW_Y + CELL + 16;
const GATE_Y_9 = EXPERTS_Y_9 + EXPERTS.h + 12;
const SWITCH_AT = 0.72; // frame 9: the bias changes the picks

const ease = (t) => t * t * (3 - 2 * t);
const roundText = (v) => String(Math.round(v));

// Eight load bars on the fixed 0–96 scale with the dashed fair-share line (same place in frames 8 and 10).
function batchBars(parent, values) {
  G.bars(parent, {
    x: X0, y: BARS_BASE - BARS_H, w: WIDTH8, h: BARS_H, values, labels: NAMES, max: MAX_LOAD,
    reference: { value: FAIR_SHARE, label: `fair share ${FAIR_SHARE}` }, format: roundText, label: 'tokens per expert, a batch of 128',
  });
}

const busiest = (loads) => Math.max(...loads);
const NOTE_LINES_Y = 62;
const LINE = 20;

// Frame 8: across a batch the loads are far from even: the busiest expert has three times its fair share.
export function drawFrame8(svg, p) {
  fadeOutOf(svg, drawFrame7, p);
  const g = layer(svg, arriving(p));
  const loads = RUN[0].loads;
  note(g, X0, 28, 'a batch of 128 tokens, not one word', { cls: '' });
  batchBars(g, loads.map((v) => v * ease(seg(p, 0.15, 0.8))));
  const readout = layer(g, seg(p, 0.8, 1));
  note(readout, NOTES_X, NOTE_LINES_Y, 'busiest ÷ fair share');
  note(readout, NOTES_X, NOTE_LINES_Y + LINE, `${busiest(loads)} ÷ ${FAIR_SHARE} = ${imbalanceText(RUN[0].imbalance)}`, { cls: '' });
  note(readout, NOTES_X, NOTE_LINES_Y + 2 * LINE, `fair share: 128 × 2 ÷ 8 = ${FAIR_SHARE}`);
}

// A row of gate weights under the experts that were chosen.
function gates(parent, picks, weights, opacity) {
  fade(parent, opacity, (g) => {
    picks.forEach((pick, i) => linkedCells(g, 'gate', { x: expertLeft(pick) - 2, y: GATE_Y_9, values: [weights[i]], cell: CELL, maxAbs: GATE_MAX, format: gateText }));
  });
}

// Frame 9 (a branch back to "sat"): a bias row is added to the scores for choosing only; the gate weights keep the raw scores.
export function drawFrame9(svg, p) {
  fadeOutOf(svg, drawFrame8, p);
  const g = layer(svg, arriving(p));
  header(g);
  scoreRow(g, { values: SCORES_SAT, rowLabel: 'scores' });
  const slide = seg(p, 0.15, 0.4);
  fade(g, slide, (h) => scoreRow(h, { values: BIAS_SAT, y: lerp(SCORES_Y + 8, BIAS_ROW_Y, slide), link: 'bias', labels: false, rowLabel: 'bias' }));
  const sum = seg(p, 0.45, 0.65);
  fade(g, sum, (h) => scoreRow(h, { values: CHOOSE_SAT, y: CHOOSE_ROW_Y, link: null, labels: false, rowLabel: 'choose', hatch: CHOOSE_SAT.map((_, e) => !PICKS_BIASED.includes(e)) }));
  const switched = p >= SWITCH_AT;
  expertRow(g, { y: EXPERTS_Y_9, active: switched ? PICKS_BIASED : SAT_PICKS });
  const swap = seg(p, SWITCH_AT, 0.9);
  gates(g, SAT_PICKS, GATES[SAT], 1 - swap);
  gates(g, PICKS_BIASED, GATES_BIASED, swap);
  note(g, expertLeft(SAT_PICKS[0]) - 10, GATE_Y_9 + CELL / 2, 'gate weight', { anchor: 'end' });
  const readout = layer(g, seg(p, SWITCH_AT, 0.9));
  note(readout, NOTES_X, SCORES_Y + 12, `picks: ${picksText(PICKS_BIASED)}`, { cls: '' });
  note(readout, NOTES_X, GATE_Y_9 + 6, 'gates from raw scores:');
  note(readout, NOTES_X, GATE_Y_9 + 6 + LINE, `softmax([${PICKS_BIASED.map((e) => scoreText(SCORES_SAT[e])).join(', ')}])`);
  note(readout, NOTES_X, GATE_Y_9 + 6 + 2 * LINE, `= [${GATES_BIASED.map(gateText).join(', ')}]`);
}

// The step the bars are at (0 – 6, fractional) and the nearest whole step, for the printed counter, biases and imbalance.
function stepAt(p) {
  const k = LAST_STEP * seg(p, 0.2, 1);
  const i = Math.min(Math.floor(k), LAST_STEP - 1);
  const f = k - i;
  return { i, f, nearest: i + (f >= 0.5 ? 1 : 0) };
}

// Frame 10 (back in the batch): six steps of "raise the idle, lower the busy"; the bars flatten toward the fair-share line.
export function drawFrame10(svg, p) {
  fadeOutOf(svg, drawFrame9, p);
  const g = layer(svg, arriving(p));
  const { i, f, nearest } = stepAt(p);
  const now = RUN[nearest];
  note(g, X0, 28, 'a batch of 128 tokens, step by step', { cls: '' });
  batchBars(g, RUN[i].loads.map((v, e) => lerp(v, RUN[i + 1].loads[e], ease(f))));
  note(g, X0 - 8, BIAS_Y, 'bias', { anchor: 'end' });
  now.bias.forEach((b, e) => note(g, colCenter(e), BIAS_Y, biasText(b), { anchor: 'middle', cls: '' }));
  note(g, NOTES_X, NOTE_LINES_Y, `step ${nearest} of ${LAST_STEP}`, { cls: '' });
  note(g, NOTES_X, NOTE_LINES_Y + LINE, 'busiest ÷ fair share');
  note(g, NOTES_X, NOTE_LINES_Y + 2 * LINE, `${busiest(now.loads)} ÷ ${FAIR_SHARE} = ${imbalanceText(now.imbalance)}`, { cls: '' });
}
