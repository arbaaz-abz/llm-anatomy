// training-memory stage: layout constants, timing helpers and the drawing pieces shared by the 11 frames. Every frame is a
// pure function of (index, progress): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { formatBytes } from '@math/core.js';
import { TRAINING_RECIPES, trainingBytesPerParam, zeroPerGpuBytes } from '@math/training-memory.js';
import { gbText } from './format.js';
import { GPT3, ZERO_DP, H100_HBM } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ADAM = TRAINING_RECIPES.adam;
export const PER_PARAM = trainingBytesPerParam(ADAM); // 2 + 2 + 8 + 4 = 16 B

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start', opacity = 1 } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

export const layer = (svg, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, svg);

// A group a math term can name (data-link) with an invisible frame the hover outline uses. Draw it on the stage, never inside a
// glyph: `.glyph .g-frame` is a filled, stroked box in the theme.
export function linkFrame(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 1, y: y - 1, width: w + 2, height: h + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// ---- frames 1–4: one parameter's bytes as a shareBar, then the counter and the GPU row ----
export const BAR = Object.freeze({ x: 20, y: 66, h: 14, perByte: 26.25 }); // 16 B fill 420 px
const BYTE_PARTS = Object.freeze([['weight', 'weight', 1, 'w'], ['grad', 'gradient', 2, 'g'], ['optimizer', 'Adam moments', 3, 'o'], ['master', 'master copy', 4, 'o']]);

// `shown` = bytes drawn so far per part (a partial part grows). Width is bytes × perByte, so a segment keeps its size.
export function byteBar(parent, shown) {
  const live = BYTE_PARTS.filter(([key]) => shown[key] > 1e-9);
  const total = live.reduce((s, [key]) => s + shown[key], 0);
  if (live.length === 0) return 0;
  const w = total * BAR.perByte;
  G.shareBar(parent, { x: BAR.x, y: BAR.y, w, h: BAR.h, tail: 'none', label: 'bytes per parameter',
    parts: live.map(([key, name, hue]) => ({ name, value: shown[key], hue })), format: (share) => formatBytes(share * total) });
  let at = 0;
  live.forEach(([key, , , letter]) => {
    linkFrame(parent, letter, { x: BAR.x + at * BAR.perByte, y: BAR.y, w: shown[key] * BAR.perByte, h: BAR.h });
    at += shown[key];
  });
  return total;
}

export const COUNTER = Object.freeze({ x: 20, y: 214 });
// The counting number: whole GB while it runs (the end states are whole), thousands separated.
export const countGb = (bytes) => `${Math.round(bytes / 1e9).toLocaleString('en-US')} GB`;
export const ROW = Object.freeze({ x: 26, y: 238, w: 96, h: 72, gap: 12 });

// "GPT-3 needs: 350 GB = 4.4 H100s" and the GPU row: at most 5 glyphs, otherwise 4 plus "+ N more" (README lesson 18).
export function gpuRow(parent, bytes) {
  const need = bytes / H100_HBM;
  const count = Math.max(1, Math.ceil(need - 1e-9));
  const collapsed = count > 5;
  const drawn = collapsed ? 4 : count;
  for (let i = 0; i < drawn; i += 1) {
    G.gpu(parent, { x: ROW.x + i * (ROW.w + ROW.gap), y: ROW.y, w: ROW.w, h: ROW.h, memFill: clamp01(need - i), label: 'H100 · 80 GB' });
  }
  if (collapsed) note(parent, ROW.x + 4 * (ROW.w + ROW.gap) + 6, ROW.y + ROW.h / 2, `+ ${count - 4} more`, { cls: 'g-text' });
}

// ---- frames 8–10: the 64-GPU row and GPU 1's composition ----
export const DP_ROW = Object.freeze({ y: 70, xs: [30, 160, 452], w: 96, h: 72, others: 354 });
export const COMPOSITION = Object.freeze({ x: 30, y: 222, w: 420, h: 14 });

export function zeroState(stage) {
  return zeroPerGpuBytes({ params: GPT3.params, recipe: ADAM, stage, dp: ZERO_DP });
}

export { gbText };
