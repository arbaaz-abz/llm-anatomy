// DOM helpers for the sft toy: the transcript as focusable chips (ruling P3-R17: role="button", tabindex, Enter / Space,
// arrow keys, visible focus, the selection drawn with G.selectionMark) and the readout tables.
import * as G from '@shared/glyphs.js';
import { readoutTable } from '@shared/ui/readout-table.js';
import { LINES } from './numbers.js';
import { CHIP_H, lineY, note } from './stage.js';

const KEYS = Object.freeze({ ArrowLeft: -1, ArrowRight: 1 });
const SVG_W = 580;

const label = (chip) => `${chip.text}, ${chip.kind}, ${chip.trained ? 'trained' : 'masked'}`;

// The chip that an arrow key moves to; ArrowUp / ArrowDown keep the position in the line.
export function neighbour(chips, flat, key) {
  if (key in KEYS) return Math.min(Math.max(flat + KEYS[key], 0), chips.length - 1);
  if (key === 'Home') return 0;
  if (key === 'End') return chips.length - 1;
  if (key !== 'ArrowUp' && key !== 'ArrowDown') return flat;
  const here = chips[flat];
  const line = here.line + (key === 'ArrowUp' ? -1 : 1);
  const inLine = chips.filter((c) => c.line === line);
  if (inLine.length === 0) return flat;
  const offset = flat - chips.find((c) => c.line === here.line).flat;
  return inLine[Math.min(offset, inLine.length - 1)].flat;
}

// Draws the transcript into `host`; `onSelect(flat)` fires for a click, Enter, Space or an arrow key. Returns the chip elements.
export function drawTranscript(host, chips, selected, onSelect) {
  const height = lineY(LINES.length - 1) + CHIP_H + 8;
  const svg = G.svgEl('svg', { width: SVG_W, height, viewBox: `0 0 ${SVG_W} ${height}`, role: 'group', 'aria-label': 'The 26-token transcript: choose a token to see whether it is trained' });
  G.hatchFill(svg);
  LINES.forEach((line, i) => { if (line.role) note(svg, 4, lineY(i) + CHIP_H / 2, line.role); });
  const nodes = chips.map((chip) => {
    const g = G.svgEl('g', { class: 'stage-item', role: 'button', tabindex: chip.flat === selected ? '0' : '-1', 'aria-label': label(chip), 'aria-pressed': String(chip.flat === selected), 'data-flat': String(chip.flat), 'data-trained': String(chip.trained), 'data-kind': chip.kind }, svg);
    G.token(g, { x: chip.x, y: chip.y, text: chip.text, state: chip.kind === 'template' ? 'dim' : 'idle', hatched: !chip.trained });
    if (chip.flat === selected) G.selectionMark(g, { x: chip.x, y: chip.y, w: chip.w, h: CHIP_H });
    g.addEventListener('click', () => onSelect(chip.flat, false));
    g.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(chip.flat, true); return; }
      const next = neighbour(chips, chip.flat, event.key);
      if (event.key.startsWith('Arrow') || event.key === 'Home' || event.key === 'End') { event.preventDefault(); onSelect(next, true); }
    });
    return g;
  });
  host.replaceChildren(svg);
  return nodes;
}

export const countsTable = (v) => readoutTable({
  name: 'counts',
  rows: [
    { label: 'Trained tokens', cells: [{ value: v.trained, name: 'trained' }] },
    { label: 'Masked tokens', cells: [{ value: v.masked, name: 'masked' }] },
    { label: 'Share trained', cells: [{ value: v.share, name: 'share' }] },
  ],
});

export const kindsTable = (v) => readoutTable({
  head: ['Kind', 'Trained / total'],
  name: 'kinds',
  rows: v.kinds.map((r) => ({ label: r.kind, cells: [{ value: r.text, name: `kind-${r.kind}` }] })),
});
