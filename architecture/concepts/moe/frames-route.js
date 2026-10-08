// moe frames 1–4: one token through the router (scores, top-2, gate weights), then all four words and their loads.
// Each draw is a pure function of progress; the end state of each is the start of the next (a scene change fades).
import * as G from '@shared/glyphs.js';
import { ROUTER_TOY } from '@math/moe.js';
import { SAT, SCORES_SAT, PICKS, GATES, E3_OUT, E6_OUT, SCORE_MAX, GATE_MAX, LOADS_FOUR } from './numbers.js';
import {
  X0, CELL, TINY, NOTES_X, SCORES_Y, EXPERTS, ROUTER, WIDTH8,
  seg, lerp, arriving, leaving, layer, note, fadeOutOf, header, scoreRow, expertRow, linkedCells, colCenter, expertLeft,
} from './stage.js';
import { routingGrid, loadBars, GRID, BARS } from './figures.js';
import { gateText, picksText, scoreText, expertName } from './format.js';

const SAT_PICKS = PICKS[SAT];
const HATCH_SAT = SCORES_SAT.map((_, e) => !SAT_PICKS.includes(e));
const SKIPPED = SCORES_SAT.map((_, e) => e).filter((e) => !SAT_PICKS.includes(e));
const GATE_Y = EXPERTS.y + EXPERTS.h + 12;
const OUT_Y = GATE_Y + CELL + 14;
const ADDER = Object.freeze({ x: (colCenter(SAT_PICKS[0]) + colCenter(SAT_PICKS[1])) / 2, y: OUT_Y + TINY + 36 });
const MERGED_Y = ADDER.y + 18;
const OUT_W = 8 * TINY;
const outLeft = (pick) => colCenter(pick) - OUT_W / 2;
const SAT_OUTS = [E3_OUT, E6_OUT];
const scaled = (vec, w) => vec.map((v) => v * w);
const MERGED = SAT_OUTS[0].map((v, i) => GATES[SAT][0] * v + GATES[SAT][1] * SAT_OUTS[1][i]);
const NOTE_Y = SCORES_Y + CELL / 2;
const ROUTER_NOTE_Y = ROUTER.y + ROUTER.h / 2;

// Frame 1: sat enters the router; the eight scores type in left to right.
export function drawFrame1(svg, p) {
  header(svg, { intoRouter: seg(p, 0, 0.25), outOfRouter: seg(p, 0.2, 0.45) });
  const typed = Math.ceil(8 * seg(p, 0.4, 0.95));
  if (typed > 0) scoreRow(svg, { values: SCORES_SAT, count: typed });
  expertRow(svg);
  note(layer(svg, seg(p, 0.55, 0.85)), ROUTER.x + ROUTER.w + 12, ROUTER_NOTE_Y, '8 × 8 = 64 parameters per block');
}

// The note beside the score row once the top two are chosen (frame 2's end state, which frame 3 starts from).
function topTwoNote(parent, opacity) {
  const g = layer(parent, opacity);
  note(g, NOTES_X, NOTE_Y - 8, `top-2: ${SAT_PICKS.map((e) => `${expertName(e)} (${scoreText(SCORES_SAT[e])})`).join(', ')}`);
  note(g, NOTES_X, NOTE_Y + 10, `${SKIPPED.length} experts skipped`);
}

// Frame 2: the six lowest cells hatch over and the two chosen experts light.
export function drawFrame2(svg, p) {
  header(svg);
  const hatched = SKIPPED.filter((_, i) => p >= 0.1 + i * 0.08);
  scoreRow(svg, { values: SCORES_SAT, hatch: SCORES_SAT.map((_, e) => hatched.includes(e)) });
  expertRow(svg, { active: p >= 0.65 ? SAT_PICKS : [] });
  note(layer(svg, leaving(p)), ROUTER.x + ROUTER.w + 12, ROUTER_NOTE_Y, '8 × 8 = 64 parameters per block');
  topTwoNote(svg, seg(p, 0.5, 0.8));
}

// The gate cells under the chosen experts (softmax of the two chosen raw scores).
function gateCells(parent, opacity) {
  const g = layer(parent, opacity);
  SAT_PICKS.forEach((pick, i) => linkedCells(g, 'gate', { x: expertLeft(pick) - 2, y: GATE_Y, values: [GATES[SAT][i]], cell: CELL, maxAbs: GATE_MAX, format: gateText }));
  note(g, expertLeft(SAT_PICKS[0]) - 10, GATE_Y + CELL / 2, 'gate weight', { anchor: 'end' });
  return g;
}

// The scaled expert outputs, the adder and the blended row of frame 3.
function blend(parent, { shrink, toAdder, merged }) {
  SAT_PICKS.forEach((pick, i) => {
    const w = lerp(1, GATES[SAT][i], shrink);
    G.vector(parent, { x: outLeft(pick), y: OUT_Y, values: scaled(SAT_OUTS[i], w), cell: TINY, orient: 'row', maxAbs: SCORE_MAX });
    if (toAdder > 0) G.flow(parent, { from: [colCenter(pick), OUT_Y + TINY + 3], to: [ADDER.x + (i === 0 ? -6 : 6), ADDER.y - 10], carry: 'activation', progress: toAdder });
  });
  if (merged <= 0) return;
  const g = layer(parent, merged);
  G.adder(g, ADDER);
  G.vector(g, { x: ADDER.x - OUT_W / 2, y: MERGED_Y, values: MERGED, cell: TINY, orient: 'row', maxAbs: SCORE_MAX });
  note(g, ADDER.x + OUT_W / 2 + 10, MERGED_Y + TINY / 2, 'layer output');
}

// Frame 3: softmax over the chosen two gives the gate weights; both experts run and their outputs are blended at the adder.
export function drawFrame3(svg, p) {
  header(svg);
  scoreRow(svg, { values: SCORES_SAT, hatch: HATCH_SAT });
  expertRow(svg, { active: SAT_PICKS });
  topTwoNote(svg, leaving(p));
  note(layer(svg, seg(p, 0, 0.25)), NOTES_X, NOTE_Y - 8, `softmax([${SAT_PICKS.map((e) => scoreText(SCORES_SAT[e])).join(', ')}])`);
  note(layer(svg, seg(p, 0, 0.25)), NOTES_X, NOTE_Y + 10, `= [${GATES[SAT].map(gateText).join(', ')}]`);
  const run = seg(p, 0.25, 0.5);
  SAT_PICKS.forEach((pick) => G.flow(svg, { from: [colCenter(pick), EXPERTS.y + EXPERTS.h + 2], to: [colCenter(pick), OUT_Y - 2], carry: 'activation', progress: run }));
  gateCells(svg, seg(p, 0, 0.25));
  if (run < 1) return;
  blend(svg, { shrink: seg(p, 0.55, 0.8), toAdder: seg(p, 0.8, 0.95), merged: seg(p, 0.8, 1) });
}

const LOAD_ROW_REVEAL = 0.17;
const rowStart = (t) => 0.15 + t * LOAD_ROW_REVEAL;

// Frame 4's scene at progress p; drawn on its own layer so the next frame can fade it out again.
function routingScene(parent, p) {
  const progress = ROUTER_TOY.map((_, t) => seg(p, rowStart(t), rowStart(t) + LOAD_ROW_REVEAL));
  const rowsShown = progress.filter((x) => x > 0).length;
  const hatchRows = progress.filter((x) => x >= 0.5).length;
  routingGrid(parent, { picks: PICKS, rowsShown, hatchRows });
  PICKS.forEach((picks, t) => {
    if (progress[t] > 0) note(layer(parent, progress[t]), X0 + WIDTH8 + 8, GRID.y + t * CELL + CELL / 2, `→ ${picksText(picks)}`);
  });
  const loads = LOADS_FOUR.map((_, e) => PICKS.reduce((sum, picks, t) => sum + (picks.includes(e) ? progress[t] : 0), 0));
  note(parent, X0, BARS.base - BARS.h - 22, 'load: tokens per expert');
  loadBars(parent, { values: loads });
}

// Frame 4: all four words are routed; each chosen cell drops a token into its expert's bar.
export function drawFrame4(svg, p) {
  fadeOutOf(svg, drawFrame3, p);
  routingScene(layer(svg, arriving(p)), p);
}
