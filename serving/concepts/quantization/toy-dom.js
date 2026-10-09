// quantization toy DOM helpers: the Round-a-block figure and the two readout tables, repainted from toyView.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { CELL, MAX_WEIGHT, MAX_ERROR, NEUTRAL, note } from './stage.js';

const FIG = Object.freeze({ x: 70, chips: 0, weights: 30, line: 76, codes: 128, restored: 170, errors: 212, width: 8 * CELL, labelGap: 8 });

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

const text = (parent, x, y, str, props = {}) => note(parent, x, y, str, { anchor: props['text-anchor'] ?? 'start' });

function row(svg, y, cells, { maxAbs, label }) {
  G.vector(svg, { x: FIG.x, y, values: cells.map((c) => c.value), cell: CELL, orient: 'row', maxAbs, format: (v) => cells.find((c) => c.value === v).text });
  text(svg, FIG.x - FIG.labelGap, y + CELL / 2 + 4, label, { 'text-anchor': 'end' });
}

// One scale chip per block, centered over its weights.
function chips(svg, blocks) {
  blocks.forEach((b) => {
    const label = `scale ${b.scale}`;
    const w = G.tokenWidth(label);
    G.token(svg, { x: FIG.x + ((b.from + b.to + 1) / 2) * CELL - w / 2, y: FIG.chips, text: label });
  });
}

// The eight weights, their scale chips, the number line they land on, and the codes, restored and error rows. The viewBox is
// fitted to what was drawn, so the figure scrolls inside its box at 400 px.
export function paintFigure(svg, fig) {
  svg.replaceChildren();
  chips(svg, fig.blocks);
  row(svg, FIG.weights, fig.weights, { maxAbs: MAX_WEIGHT, label: 'weights' });
  G.numberLine(svg, { x: FIG.x, y: FIG.line, w: FIG.width, lo: fig.line.lo, hi: fig.line.hi, grid: fig.line.grid, points: fig.line.points, label: 'weight ÷ scale' });
  text(svg, FIG.x - FIG.labelGap, FIG.line + 28, 'weight ÷ scale', { 'text-anchor': 'end' });
  row(svg, FIG.codes, fig.codes, { maxAbs: NEUTRAL, label: 'codes' });
  row(svg, FIG.restored, fig.restored, { maxAbs: MAX_WEIGHT, label: 'restored' });
  row(svg, FIG.errors, fig.errors, { maxAbs: MAX_ERROR, label: 'error' });
  G.fitViewBox(svg, 6);
  svg.setAttribute('aria-label', `The eight weights quantized: mean error ${fig.readouts.meanError}, ${fig.readouts.zeroed} rounded to zero`);
}

export function roundTable(r) {
  return readoutTable({
    head: ['Rounding', ''], name: 'round-readouts',
    rows: [
      { label: 'Mean error', sub: 'average distance from the original', cells: [{ value: r.meanError, name: 'mean-error' }] },
      { label: 'Weights rounded to zero', cells: [{ value: r.zeroed, name: 'zeroed' }] },
      { label: 'Weights clipped', sub: 'past the grid\'s last rounding boundary', cells: [{ value: r.clipped, name: 'clipped' }] },
    ],
  });
}

export function modelTable(m) {
  return readoutTable({
    head: ['Llama-3.1-70B on one GPU', ''], name: 'model-readouts',
    rows: [
      { label: 'Bits per weight', sub: 'scales included', cells: [{ value: m.bits, name: 'bits' }] },
      { label: 'Weights', cells: [{ value: m.weights, name: 'weights' }] },
      { label: 'GPU memory', cells: [{ value: m.memory.value, sub: m.memory.sub, name: 'memory' }] },
      { label: 'Free after the weights', cells: [{ value: m.free, name: 'free' }] },
      { label: 'KV cache per user', sub: '2,048 tokens', cells: [{ value: m.kvPerUser, name: 'kv-user' }] },
      { label: 'Users that fit', sub: 'at 2,048 tokens', cells: [{ value: m.users, name: 'users' }] },
      { label: 'Decode step', sub: '1 user, 2,048 tokens of context', cells: [{ value: m.decode.value, sub: m.decode.sub, name: 'decode' }] },
      { label: 'Prefill', sub: '4,096 tokens', cells: [{ value: m.prefill.value, sub: m.prefill.sub, name: 'prefill' }] },
    ],
  });
}
