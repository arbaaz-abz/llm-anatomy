// speculative-decoding toy DOM helpers: readout tables, the two curve plots and the position panel's vectors.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { K_RANGE, P, Q, ZOOM_WORDS } from './numbers.js';
import { cellText, ratioText } from './format.js';

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

const cell = (value, name) => ({ value, name });

export function roundTable(view) {
  return readoutTable({
    name: 'round', head: ['Per round', 'Value'],
    rows: [
      { label: 'Tokens per round', cells: [cell(view.tokens, 'tokens-per-round')] },
      { label: 'Speedup at batch 1, ignoring the batch', cells: [cell(view.simple, 'speedup-simple')] },
    ],
  });
}

export function batchTable(view) {
  const table = readoutTable({
    name: 'batch', head: ['At the chosen batch', 'Value'],
    rows: [
      { label: view.batchLabel, cells: [cell(view.batchSpeedup, 'speedup-batch')] },
      { label: 'Plain step', cells: [cell(view.plain, 'plain-step')] },
      { label: 'Verify pass', cells: [cell(view.verify, 'verify-pass')] },
      { label: 'Verify pass is', cells: [cell(view.verifyBound, 'verify-bound')] },
    ],
  });
  table.querySelector('[data-readout="verify-bound"]').classList.add(view.verifyBound.startsWith('memory') ? 'sem-text--memory' : 'sem-text--compute');
  return table;
}

export function positionTable(view, guess) {
  const p = view.position;
  return readoutTable({
    name: 'position', head: ['At the zoomed position', 'Value'],
    rows: [
      { label: `Keep chance for ${guess}`, cells: [cell(p.keep, 'keep-chance')] },
      { label: 'Acceptance rate of the drafter', cells: [cell(p.acceptance, 'acceptance-rate')] },
      { label: 'Leftover', sub: ZOOM_WORDS.join(' · '), cells: [cell(p.leftoverText, 'leftover')] },
      { label: 'Result', sub: 'equals p', cells: [cell(p.resultText, 'result')] },
    ],
  });
}

const PLOT_W = 420;
const PLOT_H = 190;

// Tokens per round and speedup against guesses at the current α and c; the current k is framed on the speedup.
export function paintKPlot(svg, view, state) {
  svg.replaceChildren();
  const speedup = view.kSeries.speedup;
  G.curvePlot(svg, {
    x: 0, y: 0, w: PLOT_W, h: PLOT_H, label: 'tokens per round and speedup against guesses',
    xAxis: { label: 'guesses per round k', domain: [1, 8], ticks: K_RANGE },
    yAxis: { label: 'tokens per round · speedup', domain: [1, 7], ticks: [1, 2, 3, 4, 5, 6, 7] },
    series: [{ points: view.kSeries.tokens, label: 'tokens', style: 'muted', tone: 'ink' }, { points: speedup, label: 'speedup', style: 'solid', tone: 'ink' }],
    markers: [{ x: state.k, y: speedup[state.k - 1][1], label: ratioText(speedup[state.k - 1][1]), followed: true }],
  });
  G.fitViewBox(svg, 6);
}

// Speedup against users at the current α, k and c; the chosen batch is framed.
export function paintBatchPlot(svg, view, state, max) {
  svg.replaceChildren();
  const marked = view.batchSeries.find(([users]) => users === state.batch);
  const ys = view.batchSeries.map(([, y]) => y);
  const lo = Math.min(1, Math.floor(Math.min(...ys) * 4) / 4);
  const hi = Math.ceil(Math.max(...ys) * 4) / 4;
  const ticks = [lo, hi];
  G.curvePlot(svg, {
    x: 0, y: 0, w: PLOT_W, h: PLOT_H, label: 'speedup against users in the batch',
    xAxis: { label: 'users in the batch', domain: [1, max], ticks: [1, 64, 128, max] },
    yAxis: { label: 'speedup', domain: [lo, hi], ticks },
    series: [{ points: view.batchSeries, style: 'solid', tone: 'ink' }],
    markers: marked ? [{ x: marked[0], y: marked[1], label: ratioText(marked[1]), followed: true }] : [],
  });
  G.fitViewBox(svg, 6);
}

const CELL = G.NUMBER_CELL;
const GRID = Object.freeze({ x: 130, rows: [24, 74, 124, 174] });

// The position panel: p, q, the leftover and the result as four rows of cells, the guessed word outlined in p and q.
export function paintPosition(svg, view, guess) {
  svg.replaceChildren();
  ZOOM_WORDS.forEach((w, j) => G.svgEl('text', { x: GRID.x + j * CELL + CELL / 2, y: 12, class: 'g-label', 'text-anchor': 'middle' }, svg).append(w));
  [['p (target)', P], ['q (drafter)', Q], ['leftover', view.position.leftover], ['result', view.position.result]].forEach(([name, values], i) => {
    G.svgEl('text', { x: GRID.x - 8, y: GRID.rows[i] + CELL / 2 + 4, class: 'g-label', 'text-anchor': 'end' }, svg).append(name);
    G.vector(svg, { x: GRID.x, y: GRID.rows[i], values, cell: CELL, orient: 'row', maxAbs: 1, format: cellText });
  });
  const column = view.position.guessIndex;
  [0, 1].forEach((row) => G.selectionMark(svg, { x: GRID.x + column * CELL, y: GRID.rows[row], w: CELL, h: CELL }));
  G.fitViewBox(svg, 8);
}

