// The toy's table: one non-stage <svg> built with the same glyphs as the stage (chips, verdicts, R and A cells).
// Chips are selectable <g role="button"> groups; the selection is drawn with G.selectionMark. No state lives here.
import * as G from '@shared/glyphs.js';
import { CELL, CHIP_H, CHIP_GAP } from './stage.js';
import { ADV_MAX_ABS } from './numbers.js';

const LAYOUT = Object.freeze({ index: 8, chips: 28, verdict: 0, header: 30, rowH: CELL, gapAfterChips: 18, cellGap: 4, pushW: 104 });
const MOVES = Object.freeze({ ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] });

// x positions of the columns, from the widest answer in the group (the table grows with its content).
function columns(rows) {
  const widest = Math.max(...rows.map((r) => r.chips.reduce((x, c) => x + G.tokenWidth(c.text) + CHIP_GAP, 0)));
  const verdict = LAYOUT.chips + widest + LAYOUT.gapAfterChips;
  const r = verdict + 20;
  const a = r + CELL + LAYOUT.cellGap;
  const push = a + CELL + 12;
  return { verdict, r, a, push, width: push + LAYOUT.pushW };
}

const rowTop = (i) => LAYOUT.header + i * LAYOUT.rowH;
const text = (parent, x, y, str, { anchor = 'start', cls = 'g-label' } = {}) => {
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor, 'dominant-baseline': 'central' }, parent);
  t.textContent = str;
  return t;
};

function chipButton(svg, view, { row, token, x, y, selected }) {
  const chip = view.rows[row].chips[token];
  const button = G.svgEl('g', { class: 'stage-item', role: 'button', tabindex: selected ? 0 : -1, 'data-tok': token + 1, 'aria-label': `row ${row + 1}, token ${token + 1}: ${chip.text}`, 'aria-pressed': String(selected) }, svg);
  G.token(button, { x, y, text: chip.text, fill: chip.fill, hatched: chip.hatched });
  return button;
}

function drawRow(svg, view, i, cols) {
  const group = G.svgEl('g', { 'data-row': i + 1 }, svg);
  const top = rowTop(i);
  const row = view.rows[i];
  const label = G.svgEl('g', { class: 'glyph' }, group);
  text(label, LAYOUT.index, top + CELL / 2, String(i + 1), { anchor: 'middle' });
  const y = top + (CELL - CHIP_H) / 2;
  let selection = null;
  row.chips.reduce((x, chip, j) => {
    const selected = view.selected.row === i && view.selected.token === j;
    chipButton(group, view, { row: i, token: j, x, y, selected });
    if (selected) selection = { x, y, w: G.tokenWidth(chip.text) };
    return x + G.tokenWidth(chip.text) + CHIP_GAP;
  }, LAYOUT.chips);
  G.verdict(group, { x: cols.verdict, y: top + CELL / 2, ok: row.ok });
  const reward = G.cell(group, { x: cols.r, y: top, size: CELL, v: Number(row.reward), maxAbs: 1, format: () => row.reward, fill: row.ok ? 'ok' : 'bad' });
  reward.dataset.col = 'r';
  const advantage = G.cell(group, { x: cols.a, y: top, size: CELL, v: view.group.advantages[i], maxAbs: ADV_MAX_ABS, format: () => row.advantage });
  advantage.dataset.col = 'a';
  const push = G.svgEl('g', { class: 'glyph', 'data-col': 'push' }, group);
  text(push, cols.push, top + CELL / 2, row.push, { cls: 'g-text' });
  return selection;
}

// Builds the table for one view. onSelect(row, token) fires from a click or an arrow key.
export function buildTable(view, onSelect) {
  const cols = columns(view.rows);
  const height = LAYOUT.header + view.rows.length * LAYOUT.rowH + 6;
  const svg = G.svgEl('svg', { class: 'toy-table', width: cols.width, height, viewBox: `0 0 ${cols.width} ${height}`, role: 'group', 'aria-label': 'The group: each row is one sampled answer; select a token to inspect its push' });
  G.hatchFill(svg);
  const heads = G.svgEl('g', { class: 'glyph' }, svg);
  [[cols.r + CELL / 2, 'R'], [cols.a + CELL / 2, 'A'], [cols.push, 'push per answer']].forEach(([x, str], k) => text(heads, x, 14, str, { anchor: k === 2 ? 'start' : 'middle' }));
  const selection = view.rows.map((_, i) => drawRow(svg, view, i, cols)).find(Boolean);
  G.selectionMark(svg, { ...selection, h: CHIP_H });
  svg.addEventListener('click', (event) => {
    const chip = event.target.closest('[data-tok]');
    if (chip) onSelect(Number(chip.closest('[data-row]').dataset.row) - 1, Number(chip.dataset.tok) - 1);
  });
  svg.addEventListener('keydown', (event) => {
    const chip = event.target.closest?.('[data-tok]');
    const move = MOVES[event.key] ?? (event.key === 'Enter' || event.key === ' ' ? [0, 0] : null);
    if (!chip || !move) return;
    event.preventDefault();
    const row = Number(chip.closest('[data-row]').dataset.row) - 1;
    const token = Number(chip.dataset.tok) - 1;
    const nextRow = Math.min(Math.max(row + move[0], 0), view.rows.length - 1);
    const nextToken = Math.min(Math.max(token + move[1], 0), view.rows[nextRow].chips.length - 1);
    onSelect(nextRow, nextToken);
  });
  return svg;
}
