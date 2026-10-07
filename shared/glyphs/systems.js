// Systems glyphs: KV cache stacks, GPUs, racks and request timelines.
import { svgEl, group, text } from './core.js';

const clamp01 = (v) => Math.min(Math.max(Number(v) || 0, 0), 1);

export function kvStack(parent, { x, y, count, tile = 14, highlight = [], label }) {
  const gap = 2;
  const g = group(parent, 'g-kv', x, y, { role: 'img', 'aria-label': `KV cache: ${count} tokens` });
  if (label) text(g, 0, -8, label, 'g-label');
  text(g, -8, tile / 2, 'K', 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  text(g, -8, tile + gap + tile / 2, 'V', 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  for (let i = 0; i < count; i += 1) {
    const cx = i * (tile + gap);
    svgEl('rect', { class: 'g-k', x: cx, y: 0, width: tile, height: tile, rx: 2 }, g);
    svgEl('rect', { class: 'g-v', x: cx, y: tile + gap, width: tile, height: tile, rx: 2 }, g);
  }
  highlight.forEach((i) => svgEl('rect', { class: 'g-hl', x: i * (tile + gap) - 1.5, y: -1.5, width: tile + 3, height: 2 * tile + gap + 3, rx: 3 }, g));
  return g;
}

export function gpu(parent, { x, y, w = 96, h = 72, memFill = 0, label }) {
  const fill = clamp01(memFill);
  const g = group(parent, 'g-gpu', x, y, { role: 'img', 'aria-label': `${label ?? 'GPU'}: memory ${Math.round(fill * 100)}% full` });
  const dieW = Math.round(w * 0.68);
  const dieH = h - 18;
  svgEl('rect', { class: 'g-frame', width: w, height: h, rx: 5 }, g);
  svgEl('rect', { class: 'g-frame', x: 6, y: 6, width: dieW, height: dieH - 12, rx: 2 }, g);
  const cols = 4;
  const rows = 3;
  const cw = (dieW - 10) / cols;
  const ch = (dieH - 22) / rows;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      svgEl('rect', { class: 'g-sm', x: 8 + c * cw + 1, y: 8 + r * ch + 1, width: cw - 2, height: ch - 2, rx: 1 }, g);
    }
  }
  const hbmX = dieW + 12;
  const hbmW = w - hbmX - 6;
  for (let i = 0; i < 2; i += 1) {
    svgEl('rect', { class: 'g-hbm', x: hbmX, y: 6 + i * ((dieH - 12) / 2), width: hbmW, height: (dieH - 12) / 2 - 3, rx: 2 }, g);
  }
  const barY = h - 11;
  svgEl('rect', { class: 'g-mem-track', x: 6, y: barY, width: w - 12, height: 6, rx: 3 }, g);
  svgEl('rect', { class: 'g-mem-fill', x: 6, y: barY, width: (w - 12) * fill, height: 6, rx: 3 }, g);
  if (label) text(g, w / 2, h + 12, label, 'g-label', { 'text-anchor': 'middle' });
  return g;
}

export function rack(parent, { x, y, gpus = 8, linkWidth = 2, label, cols = 4 }) {
  const tile = 16;
  const gap = 10;
  const rows = Math.ceil(gpus / cols);
  const w = cols * (tile + gap) + gap;
  const h = rows * (tile + gap) + gap;
  const g = group(parent, 'g-rack', x, y, { role: 'img', 'aria-label': `${label ?? 'node'}: ${gpus} GPUs` });
  svgEl('rect', { class: 'g-frame', width: w, height: h, rx: 6 }, g);
  for (let r = 0; r < rows; r += 1) {
    const cy = gap + r * (tile + gap) + tile / 2;
    svgEl('line', { class: 'g-link', x1: gap + tile / 2, y1: cy, x2: gap + (cols - 1) * (tile + gap) + tile / 2, y2: cy, 'stroke-width': linkWidth }, g);
    if (r > 0) {
      svgEl('line', { class: 'g-link', x1: gap + tile / 2, y1: cy - (tile + gap), x2: gap + tile / 2, y2: cy, 'stroke-width': linkWidth }, g);
    }
  }
  for (let i = 0; i < gpus; i += 1) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const gx = gap + c * (tile + gap);
    const gy = gap + r * (tile + gap);
    svgEl('rect', { class: 'g-frame', x: gx, y: gy, width: tile, height: tile, rx: 2 }, g);
    svgEl('rect', { class: 'g-sm', x: gx + 3, y: gy + 3, width: tile - 6, height: tile - 6, rx: 1 }, g);
  }
  if (label) text(g, w / 2, h + 12, label, 'g-label', { 'text-anchor': 'middle' });
  return g;
}

export function request(parent, { x, y, prefill, decode, unit = 6, label }) {
  const h = 10;
  const prefillW = Math.max(0, prefill) * unit;
  const total = prefillW + Math.max(0, decode) * unit;
  const g = group(parent, 'g-request', x, y, { role: 'img', 'aria-label': `${label ?? 'request'}: prefill ${prefill} tokens, decode ${decode} tokens` });
  if (label) text(g, -8, h / 2, label, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  svgEl('line', { class: 'g-rail', x1: 0, y1: h / 2, x2: total, y2: h / 2 }, g);
  if (prefillW > 0) svgEl('rect', { class: 'g-prefill', width: prefillW, height: h, rx: 2 }, g);
  for (let i = 1; i <= decode; i += 1) {
    const tx = prefillW + i * unit;
    svgEl('line', { class: 'g-decode', x1: tx, y1: 1, x2: tx, y2: h - 1 }, g);
  }
  return g;
}
