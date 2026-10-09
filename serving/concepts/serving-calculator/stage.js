// serving-calculator stage layout: fixed positions shared by every frame, the timing helpers and the drawing helpers on the glyph library.
// The followed item is ONE GPU (G.selectionMark on the big GPU and on its cell in the rack), the same in every frame that shows it.
import * as G from '@shared/glyphs.js';
import { formatBytes } from '@math/core.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const LEFT = 8;
export const LINE = 16; // text line pitch
export const RACK_AT = Object.freeze({ x: 8, y: 4 });
export const BIG_GPU = Object.freeze({ x: 262, y: 6, w: 104, h: 76 });
export const COL_X = 392; // the right text column beside the GPU
export const MEM = Object.freeze({ x: 8, y: 120, w: 564, h: 14 }); // the followed GPU's memory bar (weights · KV · free)
export const FOOTER_Y = Object.freeze([346, 360]);
export const WEIGHTS_LABEL = 'weights';

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF] (template rule 4)
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15). cls 'g-label' is muted; '' prints in ink (a value the learner reads).
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  const t = G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor }, g);
  t.textContent = str;
  return g;
}

// Several lines from (x, y) down, one per `pitch`; `shown` lines are drawn.
export function lines(parent, x, y, texts, { shown = texts.length, cls = '', pitch = LINE } = {}) {
  texts.slice(0, shown).forEach((t, i) => note(parent, x, y + i * pitch, t, { cls }));
}

// A math-panel link target (template rule 8): <g data-link> with an invisible frame the hover outlines.
export function linkGroup(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// The conditions every frame carries (storyboard §4): two lines at the foot of the stage.
export function footer(svg, { context = '8K in / 1K out', opacity = 1 } = {}) {
  const g = layer(svg, opacity);
  note(g, LEFT, FOOTER_Y[0], `V4-Pro · GB300 NVL72 · EP 16 · ${context} · KV 4 kB/token (low) – 12 kB (high)`);
  note(g, LEFT, FOOTER_Y[1], 'floor = bytes and FLOPs only; it counts output tokens');
}

// The replica rack (16 cells, 8 per row, a bracket labeled "one replica, EP 16"); the followed GPU is cell 0.
const RACK_GROUPS = Object.freeze([{ from: 0, to: 15, label: 'one replica, EP 16' }]);
export function rackScene(parent, { at = RACK_AT } = {}) {
  G.rack(parent, { x: at.x, y: at.y, gpus: 16, cols: 8, groups: RACK_GROUPS });
  const cell = G.rackLayout({ gpus: 16, cols: 8, groups: RACK_GROUPS }).cells[0];
  G.selectionMark(parent, { x: at.x + cell.x, y: at.y + cell.y, w: 16, h: 16 });
}

// The followed GPU, drawn large, wearing the one selection mark.
export function bigGpu(parent, { label = 'GB300 · 288 GB nominal', memFill = 0 } = {}) {
  G.gpu(parent, { ...BIG_GPU, memFill, label, showMem: true });
  G.selectionMark(parent, { x: BIG_GPU.x, y: BIG_GPU.y, w: BIG_GPU.w, h: BIG_GPU.h });
}

// The followed GPU's memory: weights, KV and free in bytes on one fixed 288 GB scale, shares printed as bytes (formatBytes).
// The KV part carries the math panel's `kv` link. Parts of zero bytes stay in the legend.
export function memoryBar(parent, { y = MEM.y, weights, kv, total }) {
  const free = Math.max(total - weights - kv, 0);
  const parts = [{ name: WEIGHTS_LABEL, value: weights, hue: 1 }, { name: 'KV', value: kv, hue: 2 }, { name: 'free', value: free, hue: 3 }];
  const format = (share) => formatBytes(share * total);
  G.shareBar(parent, { x: MEM.x, y, w: MEM.w, h: MEM.h, parts, format, tail: 'none', label: 'followed GPU memory' });
  const kvSeg = G.shareBarLayout(parts, { w: MEM.w, minSegment: 0 }).main.find((s) => s.name === 'KV');
  if (kvSeg) linkGroup(parent, 'kv', { x: MEM.x + kvSeg.x, y, w: Math.max(kvSeg.width, 2), h: MEM.h });
}
