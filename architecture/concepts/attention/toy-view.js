// The attention toy's readouts: number rows, the weight heatmaps and the "Check my work" box.
// DOM only inside the functions; every number arrives already computed by math/attention.js.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { numberGrid, select, CELL, TOKENS } from './stage.js';

// theme.css styles the stage glyphs; these few readout rows carry their own layout inline.
const ROW_STYLE = 'display:flex;flex-wrap:nowrap;align-items:center;gap:3px;font-family:var(--font-mono);font-variant-numeric:tabular-nums;font-size:var(--fs-s)';
const LABEL_STYLE = 'flex:0 0 7.5em;color:var(--ink-muted);font-family:var(--font-body, inherit)';
const CELL_STYLE = 'flex:0 0 4.6em;text-align:center;padding:6px 0;border-radius:var(--radius-s)';
const HATCH = 'repeating-linear-gradient(45deg, color-mix(in oklab, var(--ink-muted) 55%, transparent) 0 1px, var(--surface) 1px 8px)';

// One labelled row of number cells: <output data-readout="name"> holding one .cell per value.
export function numberRowView(name, text) {
  const out = el('output', { className: 'toy-row' });
  out.dataset.readout = name;
  const wrap = el('div', {}, [el('span', { textContent: text, style: LABEL_STYLE }), out]);
  wrap.style.cssText = ROW_STYLE;
  out.style.cssText = 'display:flex;gap:3px';
  // cells: [{ text, v, scale, hatched }]; the fill encodes v on the value scale, the text prints it.
  const set = (cells) => out.replaceChildren(...cells.map((c) => {
    const span = el('span', { className: 'cell', textContent: c.text });
    const plain = c.hatched || !Number.isFinite(c.v);
    span.style.cssText = `${CELL_STYLE};background:${plain ? HATCH : G.valueColor(c.v, c.scale)};color:var(--cell-ink-${plain ? 0 : G.valueLevel(c.v, c.scale)})`;
    if (c.hatched) span.title = 'masked: excluded before softmax';
    return span;
  }));
  return { node: wrap, set };
}

const HEAT_POS = { x: 44, y: 40 };
const HEAT_SVG = { w: HEAT_POS.x + 4 * CELL + 6, h: HEAT_POS.y + 4 * CELL + 6 };

// One weight heatmap per shown head (numbers printed at NUMBER_CELL), the followed query outlined.
export function heatmapView({ name, weights, mask, query }) {
  const svg = G.svgEl('svg', { width: HEAT_SVG.w, height: HEAT_SVG.h, viewBox: `0 0 ${HEAT_SVG.w} ${HEAT_SVG.h}`, role: 'img', 'aria-label': `Head ${name} attention weights; rows are queries, columns are keys` });
  const values = weights.map((row, i) => row.map((v, j) => (mask[i][j] ? v : -Infinity)));
  numberGrid(svg, { ...HEAT_POS, values, kind: 'weight', maxAbs: 1, mask, label: `head ${name}: weights`, rowLabels: TOKENS, colLabels: TOKENS });
  select(svg, HEAT_POS.x, HEAT_POS.y + query * CELL, 4 * CELL, CELL);
  return svg;
}
