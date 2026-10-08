// pretraining toy's SVG strip: the eight chips (targets 2–8 value-colored by their loss and selectable), the p printed
// under each, and a NUMBER_CELL loss row. The selection pattern (shared/ui/stage-select.js) keeps one <g> per target across repaints.
import * as G from '@shared/glyphs.js';
import { TOKENS, LOSS_MAX_ABS } from './numbers.js';
import { formatLoss } from './format.js';
import { mountStageSelect } from '@shared/ui/stage-select.js';

const CELL = G.NUMBER_CELL;
const SLOT = 45; // one token's column, chip above, loss cell below (the strip is 368 px: it fits a 400 px screen)
const PAD = 4;
const CHIP = Object.freeze({ y: PAD, h: 24 });
const P_Y = 52; // the p printed under each target chip
const LOSS_Y = 60;
const SIZE = Object.freeze({ w: 2 * PAD + TOKENS.length * SLOT, h: LOSS_Y + CELL + PAD });

const center = (i) => PAD + i * SLOT + SLOT / 2;
const chipX = (i) => center(i) - G.tokenWidth(TOKENS[i]) / 2;
const cellX = (i) => center(i) - CELL / 2;

function label(parent, x, y, text, anchor = 'middle') {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: 'g-label', 'text-anchor': anchor }, g);
  t.textContent = text;
}

// Token 1 is context only (it predicts but is never a target); the row labels sit under it.
function drawStatic(parent) {
  G.token(parent, { x: chipX(0), y: CHIP.y, text: TOKENS[0], index: 1 });
  label(parent, PAD + SLOT - 8, P_Y, 'p', 'end');
  label(parent, PAD + SLOT - 8, LOSS_Y + CELL / 2 + 4, 'loss', 'end');
}

// One target: its chip filled by its loss, the p it got, its loss cell; the selection outline on chip and cell.
function drawTarget(g, cell) {
  const i = cell.chip - 1;
  G.token(g, { x: chipX(i), y: CHIP.y, text: cell.word, index: cell.chip, fill: G.valueColor(cell.loss, LOSS_MAX_ABS) });
  label(g, center(i), P_Y, cell.pText);
  G.vector(g, { x: cellX(i), y: LOSS_Y, values: [cell.loss], cell: CELL, orient: 'row', maxAbs: LOSS_MAX_ABS, format: formatLoss });
  if (!cell.selected) return;
  G.selectionMark(g, { x: chipX(i), y: CHIP.y, w: G.tokenWidth(cell.word), h: CHIP.h });
  G.selectionMark(g, { x: cellX(i), y: LOSS_Y, w: CELL, h: CELL });
}

// → { paint(view, pos), destroy }. labelledBy: the id of the visible label that names the strip.
export function mountStrip(host, { view, pos, labelledBy, onSelect }) {
  const svg = G.svgEl('svg', { width: SIZE.w, height: SIZE.h, viewBox: `0 0 ${SIZE.w} ${SIZE.h}`, role: 'group', 'aria-labelledby': labelledBy });
  drawStatic(svg);
  const items = G.svgEl('g', {}, svg);
  items.dataset.readout = 'loss-strip';
  const select = mountStageSelect(items, { items: view.strip.map((c) => ({ value: c.pos, label: c.label })), value: pos, onSelect });
  host.append(svg);
  return {
    paint(next, selected) {
      next.strip.forEach((cell) => {
        const g = select.node(cell.pos);
        g.replaceChildren();
        drawTarget(g, cell);
      });
      select.sync(selected);
    },
    destroy() {
      select.destroy();
      svg.remove();
    },
  };
}
