// moe figures drawn on both the stage (frame 4) and the toy (panel A): the 4 × 8 router-score heatmap with the unchosen cells
// hatched, and the load bars under it. Pure drawing; the callers pass how much of each is revealed.
import * as G from '@shared/glyphs.js';
import { ROUTER_TOY } from '@math/moe.js';
import { TOKENS, SAT, SCORE_MAX } from './numbers.js';
import { X0, CELL, WIDTH8, layer } from './stage.js';
import { scoreText, expertName } from './format.js';

export const GRID = Object.freeze({ y: 44 });
export const BARS = Object.freeze({ base: 312, h: 76, max: 4 }); // max 4: all four words could choose one expert

// The 4 × 8 heatmap. Rows past `rowsShown` are empty (typed in later); a row's unchosen cells hatch once `hatchRows` reach it.
// `picks[t]` lists row t's chosen experts (0-based).
export function routingGrid(parent, { x = X0, y = GRID.y, picks, rowsShown = 4, hatchRows = rowsShown }) {
  const values = ROUTER_TOY.map((row, t) => (t < rowsShown ? row : row.map(() => Number.NaN)));
  const hatch = ROUTER_TOY.map((row, t) => row.map((_, e) => t < hatchRows && !picks[t].includes(e)));
  const g = G.heatmap(parent, {
    x, y, values, cell: CELL, maxAbs: SCORE_MAX, label: 'router scores', format: (v) => (Number.isNaN(v) ? '' : scoreText(v)),
    rowLabels: TOKENS, colLabels: ROUTER_TOY[0].map((_, e) => expertName(e)), hatch,
  });
  g.setAttribute('data-link', 'score');
  if (rowsShown > SAT) G.selectionMark(layer(parent), { x, y: y + SAT * CELL, w: WIDTH8, h: CELL });
  return g;
}

// Load bars under the grid: one bar per expert, tokens per expert (values may be fractional while the loads grow).
export function loadBars(parent, { x = X0, base = BARS.base, h = BARS.h, values }) {
  return G.bars(parent, {
    x, y: base - h, w: WIDTH8, h, values, labels: values.map((_, e) => expertName(e)), max: BARS.max,
    format: (v) => String(Math.round(v)), label: 'tokens per expert',
  });
}

