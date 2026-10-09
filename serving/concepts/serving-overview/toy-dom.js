// serving-overview toy DOM helpers: the readout table and the to-scale timeline figure, repainted from toyView.
import * as G from '@shared/glyphs.js';
import { el } from '@shared/ui/dom.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { bracket } from './stage.js';

const FIG = Object.freeze({ x0: 92, y: 6, bracketY: 26 }); // room on the left for the bar's "your request" label

export const output = (name, props = {}) => {
  const o = el('output', props);
  o.dataset.readout = name;
  return o;
};

// One ruled table: what each part of the wait costs.
export function timelineTable(r) {
  const row = (label, cell, name) => ({ label, sub: cell.sub, cells: [{ value: cell.value, name }] });
  return readoutTable({
    head: ['Where the time goes', ''], name: 'timeline',
    rows: [
      row('Prefill time', r.prefill, 'prefill'),
      row('TTFT', r.ttft, 'ttft'),
      row('TPOT', r.tpot, 'tpot'),
      row('Total time', r.total, 'total'),
      row('Share of the wait after the first token', r.decodeShare, 'decode-share'),
    ],
  });
}

// Your request on one time axis, drawn to scale (full width = the whole request), with the TTFT span and the span after it.
export function paintFigure(svg, fig) {
  svg.replaceChildren();
  G.request(svg, { x: FIG.x0, y: FIG.y, steps: fig.steps, label: 'your request' });
  bracket(svg, { x0: FIG.x0, x1: FIG.x0 + fig.ttftX, y: FIG.bracketY, text: fig.ttftLabel, anchor: 'start' });
  if (fig.decodeLabel) bracket(svg, { x0: FIG.x0 + fig.ttftX, x1: FIG.x0 + fig.width, y: FIG.bracketY, text: fig.decodeLabel, anchor: 'end' });
  G.fitViewBox(svg, 4);
}
