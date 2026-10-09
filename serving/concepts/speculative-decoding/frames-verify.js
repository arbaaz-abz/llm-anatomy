// speculative-decoding frame 6: the zoom on one position. p (target) and q (drafter) over four words, the keep chance of the
// drafter's guess, the leftover it is redrawn from, and the result, which equals p. Every number comes from specdec.js.
import * as G from '@shared/glyphs.js';
import { acceptanceRate, outputDistribution, verifyToken } from '@math/specdec.js';
import { NUMBER_CELL } from '@shared/glyphs/core.js';
import { P, Q, ZOOM_GUESS, ZOOM_WORDS } from './numbers.js';
import { cellText, chanceText } from './format.js';
import { layer, leaving, linkedRow, note, seg, textLines } from './stage.js';
import { drawFrame5 } from './frames-draft.js';

const GRID = Object.freeze({ x: 110, rows: [64, 114, 164, 214], labelX: 100 });
const PANEL_X = 290;
const CHECK = verifyToken(P, Q, ZOOM_GUESS);
const RESULT = outputDistribution(P, Q);
const WORD = ZOOM_WORDS[ZOOM_GUESS];
const RANK = (list) => list.map((v, i) => [v, i]).filter(([v]) => v > 0).map(([v, i]) => `${ZOOM_WORDS[i]} ${v.toFixed(2)}`).join(', ');

const rowLabels = ['p (target)', 'q (drafter)', 'leftover', 'result'];

function labels(svg, shown) {
  ZOOM_WORDS.forEach((w, j) => note(svg, GRID.x + j * NUMBER_CELL + NUMBER_CELL / 2, GRID.rows[0] - 8, w, { anchor: 'middle' }));
  rowLabels.slice(0, shown).forEach((r, i) => note(svg, GRID.labelX, GRID.rows[i] + NUMBER_CELL / 2 + 4, r, { anchor: 'end' }));
}

function row(svg, i, letter, values, opacity = 1) {
  const g = layer(svg, opacity);
  const opts = { x: GRID.x, y: GRID.rows[i], values, cell: NUMBER_CELL, label: undefined, format: cellText };
  if (letter) linkedRow(g, letter, opts);
  else G.vector(g, { ...opts, orient: 'row', maxAbs: 1 });
}

function panel(svg, t) {
  const g = layer(svg, t);
  note(g, PANEL_X, 80, `keep ${WORD} with chance ${P[ZOOM_GUESS].toFixed(2)} / ${Q[ZOOM_GUESS].toFixed(2)} = ${chanceText(CHECK.accept)}`);
  G.shareBar(g, {
    x: PANEL_X, y: 94, w: 250, label: `chance to keep ${WORD}`, tail: 'none',
    parts: [{ name: 'keep', value: CHECK.accept, hue: 2 }, { name: 'reject', value: 1 - CHECK.accept, hue: 4 }],
  });
  textLines(g, PANEL_X, 178, [`reject mass ${CHECK.rejectMass.toFixed(2)} · acceptance rate ${chanceText(acceptanceRate(P, Q))}`]);
}

export function drawFrame6(svg, p) {
  if (p < 0.15) drawFrame5(layer(svg, leaving(p)), 1);
  const base = seg(p, 0, 0.15);
  note(layer(svg, base), 24, 22, 'the position after “The cat sat”: four candidate words');
  const g = layer(svg, base);
  labels(g, 2);
  row(g, 0, 'p', P);
  row(g, 1, 'q', Q);
  G.selectionMark(g, { x: GRID.x, y: GRID.rows[0], w: NUMBER_CELL, h: NUMBER_CELL });
  G.selectionMark(g, { x: GRID.x, y: GRID.rows[1], w: NUMBER_CELL, h: NUMBER_CELL });
  panel(svg, seg(p, 0.15, 0.4));
  const left = seg(p, 0.4, 0.65);
  note(layer(svg, left), GRID.labelX, GRID.rows[2] + NUMBER_CELL / 2 + 4, rowLabels[2], { anchor: 'end' });
  row(svg, 2, null, CHECK.residual, left);
  note(layer(svg, left), PANEL_X, 196, `leftover: ${RANK(CHECK.residual)}`);
  const result = seg(p, 0.65, 0.95);
  note(layer(svg, result), GRID.labelX, GRID.rows[3] + NUMBER_CELL / 2 + 4, rowLabels[3], { anchor: 'end' });
  row(svg, 3, 'p', RESULT, result);
  note(layer(svg, result), PANEL_X, 214, 'result = p, the target\'s own');
}
