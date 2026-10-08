// DOM helpers the attention toy draws with: number rows and weight heatmaps. Layout comes from theme.css
// (.toy-row, .toy-row-label, .cell, .cell--masked); the only inline values are a cell's value color.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { numberGrid, select, CELL, TOKENS, SCALE } from './stage.js';

// One labelled row of number cells: <output class="toy-row" data-readout="name"> with one .cell per value.
// set(cells): [{ text, v, scale, masked }]; a masked cell is hatched, its text on a solid chip.
export function numberRowView(name, text) {
  const out = el('output', { className: 'toy-row' });
  out.dataset.readout = name;
  const label = el('span', { className: 'toy-row-label', textContent: text });
  const set = (cells) => out.replaceChildren(label, ...cells.map((c) => {
    if (c.masked) {
      return el('span', { className: 'cell cell--masked', title: 'masked: excluded before softmax' }, [el('span', { className: 'cell-text', textContent: c.text })]);
    }
    const span = el('span', { className: 'cell', textContent: c.text });
    span.style.background = G.valueColor(c.v, c.scale);
    span.style.color = `var(--cell-ink-${G.valueLevel(c.v, c.scale)})`;
    return span;
  }));
  return { node: out, set };
}

const HEAT_POS = { x: 44, y: 40 };
const HEAT_SVG = { w: HEAT_POS.x + 4 * CELL + 6, h: HEAT_POS.y + 4 * CELL + 6 };

// One weight heatmap per shown head (numbers printed at NUMBER_CELL), the followed query outlined.
// A masked cell's weight is 0: printed and hatched.
export function heatmapView({ name, weights, mask, query }) {
  const svg = G.svgEl('svg', { width: HEAT_SVG.w, height: HEAT_SVG.h, viewBox: `0 0 ${HEAT_SVG.w} ${HEAT_SVG.h}`, role: 'img', 'aria-label': `Head ${name} attention weights; rows are queries, columns are keys` });
  const hatch = mask.map((row) => row.map((visible) => !visible));
  numberGrid(svg, { ...HEAT_POS, values: weights, kind: 'weight', maxAbs: SCALE.weight, hatch, label: `head ${name}: weights`, rowLabels: TOKENS, colLabels: TOKENS });
  select(svg, HEAT_POS.x, HEAT_POS.y + query * CELL, 4 * CELL, CELL);
  return svg;
}
