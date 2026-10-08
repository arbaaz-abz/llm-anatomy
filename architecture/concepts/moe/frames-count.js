// moe frames 5–7: what is stored and what runs. Every expert counts as total, only the chosen ones as active; fine-grained
// experts split each one in two; a shared expert runs for every token. The readout lines are typed in, never counted up.
import * as G from '@shared/glyphs.js';
import { mlpParams, paramBreakdown, PRESETS } from '@math/params.js';
import { expertCombinations } from '@math/moe.js';
import { SAT, SCORES_SAT, PICKS, SAT_FINE, FINE_PICKS, TOP1_SAT, SCORE_MAX } from './numbers.js';
import {
  X0, SMALL, HALF, ROUTER_LABEL, ROUTER_LABEL_FINE, NOTES_X, SCORES_Y, EXPERTS, SHARED, SAT_ROW, LINES_Y, LINE_GAP, CELL,
  seg, lerp, arriving, leaving, dip, layer, fade, note, fadeOutOf, header, scoreRow, expertRow, colLeft, colCenter,
} from './stage.js';
import { drawFrame4 } from './frames-route.js';
import { int, expertName, combosText, scoreText, toyMoe } from './format.js';

const SAT_PICKS = PICKS[SAT];
const HATCH_SAT = SCORES_SAT.map((_, e) => !SAT_PICKS.includes(e));
const HATCH_TOP1 = SCORES_SAT.map((_, e) => !TOP1_SAT.includes(e));
const HATCH_FINE = SAT_FINE.map((_, k) => !FINE_PICKS.includes(k));
const TOY_D = PRESETS.toy.dModel;
const EXPERT_PARAMS = mlpParams({ kind: 'swiglu', hidden: 8 }, TOY_D); // 192
const FINE_PARAMS = mlpParams({ kind: 'swiglu', hidden: 4 }, TOY_D); // 96
const toyCounts = (state) => paramBreakdown(toyMoe(state).config);
const WITH_SHARED = toyCounts({ routed: 8, split: 1, shared: true });
const COUNT_Y = EXPERTS.y + EXPERTS.h + 12;
const FINE_W = HALF - 2;
const LINE_STARTS_5 = [0.35, 0.55, 0.75];
const LINE_STARTS_6 = [0.6, 0.72, 0.86];
const LINE_STARTS_7 = [0.6, 0.78];

const LINES_5 = [
  `all 8 experts: 8 × ${EXPERT_PARAMS} = ${int(8 * EXPERT_PARAMS)} parameters`,
  `used by this token: 2 × ${EXPERT_PARAMS} = ${int(2 * EXPERT_PARAMS)}`,
  `= one dense MLP of hidden 16 (also ${int(mlpParams({ kind: 'swiglu', hidden: 16 }, TOY_D))})`,
];
const LINES_6 = [
  `16 experts of hidden 4: ${FINE_PARAMS} parameters each`,
  `used by this token: 4 × ${FINE_PARAMS} = ${int(4 * FINE_PARAMS)}, the same work`,
  `choices: ${combosText(expertCombinations(8, 2))} → ${combosText(expertCombinations(16, 4))}`,
];
const LINES_7 = [
  `shared ${EXPERT_PARAMS} + top-1 routed ${EXPERT_PARAMS} = ${int(2 * EXPERT_PARAMS)} active per block`,
  `whole toy: ${int(WITH_SHARED.total)} total / ${int(WITH_SHARED.active)} active`,
];

// Readout lines in the page's body ink: line i appears once p reaches its start.
function readoutLines(parent, lines, starts, p) {
  lines.forEach((line, i) => { if (p >= starts[i]) note(parent, X0, LINES_Y + i * LINE_GAP, line, { cls: '' }); });
}
const allLines = (parent, lines) => readoutLines(parent, lines, lines.map(() => 0), 1);

function expertCosts(parent) {
  note(parent, X0 - 6, COUNT_Y, 'params', { anchor: 'end' });
  for (let e = 0; e < 8; e += 1) note(parent, colCenter(e), COUNT_Y, String(EXPERT_PARAMS), { anchor: 'middle' });
}

// Frame 5's scene at progress p (drawn again, at p = 1, as the start of frame 6).
function countScene(parent, p) {
  header(parent);
  scoreRow(parent, { values: SCORES_SAT, hatch: HATCH_SAT });
  expertRow(parent, { active: SAT_PICKS, activeOpacity: dip(seg(p, 0.5, 0.9)) });
  expertCosts(parent);
  readoutLines(parent, LINES_5, LINE_STARTS_5, p);
}

// Frame 5: every expert is stored (8 × 192), but this token runs two (2 × 192), the size of one dense MLP.
export function drawFrame5(svg, p) {
  fadeOutOf(svg, drawFrame4, p);
  countScene(layer(svg, arriving(p)), p);
}

// The expert row while it splits (t from 0 to 1): each block becomes two half-size blocks, a and b. `active` lists fine indices.
function splitRow(parent, { t, active }) {
  const g = G.svgEl('g', {}, parent);
  [1, 0].forEach((half) => { // right halves first, so the left halves' labels stay on top while the two overlap
    for (let e = 0; e < 8; e += 1) {
      const k = 2 * e + half;
      const label = t >= 0.5 ? (half ? 'b' : 'a') : (half ? '' : expertName(e));
      G.block(g, { x: colLeft(e) + 2 + half * lerp(0, HALF, t), y: EXPERTS.y, w: lerp(EXPERTS.w, FINE_W, t), h: EXPERTS.h, label, state: active.includes(k) ? 'active' : 'idle' });
    }
  });
  const names = layer(g, seg(t, 0.5, 1));
  for (let e = 0; e < 8; e += 1) note(names, colCenter(e), EXPERTS.y - 9, expertName(e), { anchor: 'middle' });
}

// The 16 fine-grained scores (hover-only 20 px cells); the unchosen hatch.
function fineScores(parent, hatch) {
  const row = G.heatmap(parent, { x: X0, y: SCORES_Y, values: [SAT_FINE], cell: HALF, maxAbs: SCORE_MAX, format: scoreText, hatch: [hatch] });
  row.setAttribute('data-link', 'score');
}

const halvesOf = (coarse) => coarse.flatMap((e) => [2 * e, 2 * e + 1]);

// Frame 6's scene at progress p (drawn again, at p = 1, as the start of frame 7).
function fineScene(parent, p) {
  const swap = seg(p, 0.1, 0.4);
  const t = seg(p, 0.15, 0.55);
  header(parent, { routerLabel: swap >= 0.5 ? ROUTER_LABEL_FINE : ROUTER_LABEL });
  fade(parent, 1 - swap, (g) => scoreRow(g, { values: SCORES_SAT, hatch: HATCH_SAT }));
  fade(parent, swap, (g) => fineScores(g, HATCH_FINE.map((hatched) => hatched && p >= 0.6)));
  if (t === 0) expertRow(parent, { active: SAT_PICKS });
  else splitRow(parent, { t, active: p < 0.6 ? halvesOf(SAT_PICKS) : FINE_PICKS });
  fade(parent, leaving(p), (g) => { expertCosts(g); allLines(g, LINES_5); });
  readoutLines(parent, LINES_6, LINE_STARTS_6, p);
}

// Frame 6: each expert splits in two; the router now scores 16 and picks four, the same work with far more combinations.
export function drawFrame6(svg, p) {
  fineScene(svg, p);
}

// Frame 7's shared block: arrives idle, then fills and stays filled.
function sharedBlock(parent, p) {
  const g = layer(parent, seg(p, 0.45, 0.65));
  G.block(g, { ...SHARED, label: 'shared expert', state: p >= 0.7 ? 'active' : 'idle' });
  const flow = seg(p, 0.55, 0.8);
  if (flow > 0) G.flow(g, { from: [SAT_ROW.x + 8 * SMALL + 4, SAT_ROW.y + SMALL / 2], to: [SHARED.x + 12, SHARED.y - 2], carry: 'activation', progress: flow });
}

// Frame 7: back to eight routed experts, the router picks one, and a shared expert runs beside it.
export function drawFrame7(svg, p) {
  const swap = seg(p, 0.1, 0.4);
  const t = 1 - seg(p, 0.15, 0.55);
  header(svg, { routerLabel: swap < 0.5 ? ROUTER_LABEL_FINE : ROUTER_LABEL });
  fade(svg, 1 - swap, (g) => fineScores(g, HATCH_FINE));
  fade(svg, swap, (g) => scoreRow(g, { values: SCORES_SAT, hatch: HATCH_TOP1 }));
  if (t === 0) expertRow(svg, { active: p >= 0.8 ? TOP1_SAT : [] });
  else splitRow(svg, { t, active: t === 1 ? FINE_PICKS : [] });
  sharedBlock(svg, p);
  fade(svg, leaving(p), (g) => allLines(g, LINES_6));
  note(layer(svg, seg(p, 0.8, 0.95)), NOTES_X, SCORES_Y + CELL / 2, `top-1: ${expertName(TOP1_SAT[0])} (${scoreText(SCORES_SAT[TOP1_SAT[0]])})`);
  readoutLines(svg, LINES_7, LINE_STARTS_7, p);
}
