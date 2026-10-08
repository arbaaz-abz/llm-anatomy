// DOM helpers for the rope toy: number rows (HTML cells) and the four hands (one SVG of dials).
// Layout comes from theme.css (.toy-row, .toy-row-label, .cell); the only inline values are a cell's value color.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { SCALE } from './stage.js';
import { Q_PAIRS, K_PAIRS } from './numbers.js';

// One labelled row of number cells: <output class="toy-row" data-readout="name"> with one .cell per text.
export function numberRowView(name, text) {
  const out = el('output', { className: 'toy-row' });
  out.dataset.readout = name;
  const label = el('span', { className: 'toy-row-label', textContent: text });
  const set = (cells) => out.replaceChildren(label, ...cells.map(({ text: t, v, scale = SCALE.vector }) => {
    const span = el('span', { className: 'cell', textContent: t });
    if (v != null) {
      span.style.background = G.valueColor(v, scale);
      span.style.color = `var(--cell-ink-${G.valueLevel(v, scale)})`;
    }
    return span;
  }));
  return { node: out, set };
}

const DIAL = Object.freeze({ r: 28, x: Object.freeze([70, 220]), y: Object.freeze([60, 190]), w: 340, h: 262 });
const PAIRS = Object.freeze({ query: Q_PAIRS, key: K_PAIRS });

// The four hands: query pairs on the first row, key pairs on the second; each dial prints its angle in radians and degrees.
export function handsView() {
  const svg = G.svgEl('svg', { width: DIAL.w, height: DIAL.h, viewBox: `0 0 ${DIAL.w} ${DIAL.h}`, role: 'img', 'aria-label': 'The four hands: the query\'s and the key\'s two pairs, turned by their positions' });
  const paint = (hands) => {
    svg.replaceChildren();
    hands.forEach((h) => {
      const row = h.who === 'query' ? 0 : 1;
      G.dial(svg, { x: DIAL.x[h.i], y: DIAL.y[row], r: DIAL.r, vector: PAIRS[h.who][h.i], angle: h.angle, scale: SCALE.vector, label: h.label });
      const note = G.svgEl('g', { class: 'glyph g-note' }, svg);
      const t = G.svgEl('text', { x: DIAL.x[h.i], y: DIAL.y[row] + DIAL.r + 32, class: 'g-label', 'text-anchor': 'middle' }, note);
      t.textContent = h.text;
    });
  };
  return { node: svg, paint };
}
