// DOM helpers for the long-context-attention toy: a labelled row of number cells (the sink split, the linear output) and the
// figure drawn into the toy's <svg>. Layout comes from theme.css (.toy-row, .toy-row-label, .cell); the only inline style is a
// cell's value color.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { cellText, formatFor } from './format.js';
import { STACK, TILE, patternGrid, gridAxes, tileStack, glyphValues, linked, scene } from './stage.js';

// <output class="toy-row" data-readout="name"> with one .cell per value; set(cells, label) repaints it.
export function numberRowView(name, text) {
  const out = el('output', { className: 'toy-row' });
  out.dataset.readout = name;
  const label = el('span', { className: 'toy-row-label', textContent: text });
  const set = (values, kind, scale, rowLabel = text) => {
    label.textContent = rowLabel;
    out.replaceChildren(label, ...values.map((v) => {
      const span = el('span', { className: 'cell', textContent: cellText(v, kind) });
      span.style.background = G.valueColor(v, scale);
      span.style.color = `var(--cell-ink-${G.valueLevel(v, scale)})`;
      return span;
    }));
  };
  return { node: out, set };
}

export const TOY_STAGE = Object.freeze({ w: 580, h: 340 });

export function newFigure() {
  return G.svgEl('svg', { width: TOY_STAGE.w, height: TOY_STAGE.h, viewBox: `0 0 ${TOY_STAGE.w} ${TOY_STAGE.h}`, role: 'img' });
}

const FIGURE_LABEL = 'The followed token\'s layer: a 16 by 16 grid of cells it reads (tinted) and does not read (hatched), and the cache entries the layer stores.';
const STATE_LABEL = 'The linear-attention state S, the query for "sat" and the output S times q.';

// The pattern figure: grid with the followed row outlined, and the cache stack (merged entries are just counted tiles).
export function drawPattern(svg, view) {
  svg.replaceChildren();
  G.hatchFill(svg);
  svg.setAttribute('aria-label', FIGURE_LABEL);
  const holder = scene(svg);
  gridAxes(holder, 1, view.row);
  patternGrid(holder, { mask: view.mask });
  if (view.merged) {
    const store = linked(holder, 'store', { x: STACK.x, y: STACK.y, w: view.storedTiles * (TILE + 2), h: 2 * TILE + 2 });
    G.kvStack(store, { x: STACK.x, y: STACK.y, count: view.storedTiles, tile: TILE, label: 'stored in this layer' });
  } else {
    tileStack(holder, { veils: view.veils });
  }
}

const NUM = G.NUMBER_CELL;

// The linear figure: state S [4 × 4] at NUMBER_CELL, "× q_sat", "= o_sat".
export function drawState(svg, view) {
  svg.replaceChildren();
  svg.setAttribute('aria-label', STATE_LABEL);
  const s = { x: 40, y: 40 };
  const q = { x: s.x + 4 * NUM + 50, y: s.y };
  const o = { x: q.x + NUM + 60, y: s.y };
  const side = linked(svg, 'state', { x: s.x, y: s.y, w: 4 * NUM, h: 4 * NUM });
  G.matrix(side, { ...s, values: view.S.map(glyphValues), cell: NUM, maxAbs: 6, label: 'state S', format: formatFor('state') });
  G.vector(svg, { ...q, values: view.q, cell: NUM, orient: 'col', maxAbs: 6, label: 'q_sat', format: formatFor('state') });
  G.vector(svg, { ...o, values: view.output, cell: NUM, orient: 'col', maxAbs: 6, label: 'o_sat', format: formatFor('output') });
  [[s.x + 4 * NUM + 16, '×'], [q.x + NUM + 22, '=']].forEach(([x, text]) => {
    const t = G.svgEl('text', { x, y: s.y + 2 * NUM, 'text-anchor': 'middle', 'dominant-baseline': 'central', class: 'g-label' }, G.svgEl('g', { class: 'glyph g-note' }, svg));
    t.textContent = text;
  });
}

