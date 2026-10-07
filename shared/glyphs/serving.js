// Glyphs added by the pilot storyboards: verdict badges, clip lines, KV block pools,
// block tables and memory bars.
import { svgEl, group, text, hatchRect } from './core.js';

const REQUEST_HUES = 4;

export function requestSlot(owner) {
  const code = String(owner ?? '').toUpperCase().charCodeAt(0);
  if (Number.isNaN(code)) return 1;
  return ((((code - 65) % REQUEST_HUES) + REQUEST_HUES) % REQUEST_HUES) + 1;
}

export function verdict(parent, { x, y, ok, r = 9 }) {
  const g = group(parent, `g-verdict g-verdict--${ok ? 'ok' : 'bad'}`, x, y, { role: 'img', 'aria-label': ok ? 'correct' : 'wrong' });
  svgEl('circle', { r }, g);
  const s = r / 9;
  svgEl('path', { d: ok ? `M${-4 * s} ${0.5 * s} l${3 * s} ${3 * s} l${5 * s} ${-6 * s}` : `M${-3.5 * s} ${-3.5 * s} l${7 * s} ${7 * s} M${3.5 * s} ${-3.5 * s} l${-7 * s} ${7 * s}` }, g);
  return g;
}

export function clipLine(parent, { x, y, w = 220, lo, hi, band, marker, label }) {
  if (!(hi > lo)) throw new RangeError(`glyphs.clipLine: hi (${hi}) must exceed lo (${lo})`);
  if (marker != null && !Number.isFinite(marker)) throw new RangeError(`glyphs.clipLine: marker must be a finite number, got ${marker}`);
  const scale = (v) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * w;
  const name = label ?? 'ratio';
  const g = group(parent, 'g-clip', x, y, { role: 'img', 'aria-label': marker == null ? name : `${name} ${marker.toFixed(2)}` });
  if (label) text(g, 0, -22, label, 'g-label', { 'text-anchor': 'start' });
  if (band) svgEl('rect', { class: 'g-band', x: scale(band[0]), y: -8, width: scale(band[1]) - scale(band[0]), height: 16, rx: 2 }, g);
  svgEl('line', { class: 'g-axis', x1: 0, y1: 0, x2: w, y2: 0 }, g);
  const ticks = [lo, ...(band ?? []), hi];
  ticks.forEach((v) => {
    svgEl('line', { class: 'g-tick', x1: scale(v), y1: -4, x2: scale(v), y2: 4 }, g);
    text(g, scale(v), 18, v.toFixed(2), 'g-label g-tick-label', { 'text-anchor': 'middle' });
  });
  if (marker != null) {
    const mx = scale(marker);
    svgEl('polygon', { class: 'g-marker', points: `${mx},-2 ${mx - 5},-11 ${mx + 5},-11` }, g);
    text(g, mx, -14, marker.toFixed(2), 'g-marker-label', { 'text-anchor': 'middle', 'font-weight': 600 });
  }
  return g;
}

const SLOT_STATES = new Set(['filled', 'reserved', 'free']);
const isCount = (n) => Number.isInteger(n) && n >= 1;

export function blockPool(parent, { x, y, blocks, blockSize, slots, cell = 14, perRow = 6 }) {
  if (!isCount(blocks) || !isCount(blockSize)) throw new RangeError(`glyphs.blockPool: blocks and blockSize must be integers ≥ 1, got ${blocks} × ${blockSize}`);
  if (!Array.isArray(slots) || slots.length !== blocks * blockSize) {
    throw new RangeError(`glyphs.blockPool: expected ${blocks * blockSize} slots (blocks × blockSize), got ${slots?.length}`);
  }
  slots.forEach((slot, i) => {
    if (!SLOT_STATES.has(slot?.state ?? 'free')) throw new RangeError(`glyphs.blockPool: slot ${i} has state "${slot.state}"; use filled, reserved or free`);
  });
  const vH = Math.round(cell * 0.6);
  const blockW = blockSize * (cell + 1) + 5;
  const blockH = cell + vH + 7;
  const stepX = blockW + 10;
  const stepY = blockH + 18;
  const g = group(parent, 'g-pool', x, y, { role: 'img', 'aria-label': `KV pool: ${blocks} blocks of ${blockSize}` });
  for (let b = 0; b < blocks; b += 1) {
    const bx = (b % perRow) * stepX;
    const by = Math.floor(b / perRow) * stepY + 12;
    text(g, bx, by - 4, b, 'g-sub g-block-index');
    svgEl('rect', { class: 'g-frame', x: bx, y: by, width: blockW, height: blockH, rx: 3 }, g);
    for (let s = 0; s < blockSize; s += 1) {
      const { owner = '', state = 'free' } = slots[b * blockSize + s];
      const sx = bx + 3 + s * (cell + 1);
      const sy = by + 3;
      const slot = svgEl('g', { class: `g-slot g-slot--${state}`, 'data-req': state === 'free' ? null : requestSlot(owner) }, g);
      svgEl('rect', { class: 'g-slot-k', x: sx, y: sy, width: cell, height: cell, rx: 2 }, slot);
      svgEl('rect', { class: 'g-slot-v', x: sx, y: sy + cell + 1, width: cell, height: vH, rx: 2 }, slot);
      if (state === 'reserved') hatchRect(slot, { x: sx, y: sy, width: cell, height: cell + 1 + vH, rx: 2 });
      text(slot, sx + cell / 2, sy + cell / 2, state === 'free' ? '·' : owner, 'g-text');
      svgEl('title', {}, slot).textContent = `block ${b}, slot ${s}: ${state}${owner ? ` (${owner})` : ''}`;
    }
  }
  return g;
}

export function blockTable(parent, { x, y, rows, title }) {
  const colW = 56;
  const rowH = 16;
  const hasRef = rows.some((r) => r.ref != null);
  const cols = hasRef ? ['logical', 'physical', 'ref'] : ['logical', 'physical'];
  const g = group(parent, 'g-table', x, y, { role: 'img', 'aria-label': `${title ?? 'block table'}: ${rows.length} rows` });
  if (title) text(g, 0, -14, title, 'g-label');
  cols.forEach((c, j) => text(g, j * colW, 0, c.toUpperCase(), 'g-sub g-th'));
  svgEl('line', { class: 'g-rule', x1: 0, y1: 5, x2: cols.length * colW - 8, y2: 5 }, g);
  rows.forEach((row, i) => {
    const ty = 5 + (i + 1) * rowH;
    text(g, 0, ty, row.logical);
    text(g, colW - 16, ty, '→', 'g-label');
    text(g, colW, ty, row.physical);
    if (hasRef) text(g, 2 * colW, ty, row.ref ?? '', 'g-label');
  });
  return g;
}
