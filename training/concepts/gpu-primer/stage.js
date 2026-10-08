// gpu-primer stage layout: fixed positions shared by every frame, the timing helpers and drawing helpers on the glyph library.
// The plot keeps one place and one pair of axes from frame 6 to 11 (stage template: plot right, lanes below, text left).
import * as G from '@shared/glyphs.js';
import { matmulCost, arithmeticIntensity, rooflineTime } from '@math/roofline.js';
import { formatDuration } from '@math/core.js';
import { STAGE_CHIPS, TOKENS, SAT, D_REAL } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const CELL = G.NUMBER_CELL;
export const LEFT = 8; // the left text column's x
export const LINE = 16; // text line pitch
// Axes chosen so no glyph label collides (checked with rooflineLayout): the memory roof's label clears the y ticks, and the
// x ticks fit 46 px apart. The last x tick label reaches 23 px past the plot box, which the stage leaves free.
export const PLOT = Object.freeze({ x: 216, y: 4, w: 340, h: 232, xDomain: Object.freeze([0.1, 1e5]), yDomain: Object.freeze([1, 1e5]) });
export const LANES = Object.freeze({ x: 8, y: 270, w: 564, noteY: 262 });
export const GPU_BIG = Object.freeze({ x: 40, y: 56, w: 300, h: 190 }); // frames 1–3
export const GPU_SMALL = Object.freeze({ x: 360, y: 150, w: 212, h: 110 }); // frames 4–5
export const CHIP_ROW = Object.freeze({ x: LEFT, y: 4, gap: 8 });
export const CELLS = Object.freeze({ x: LEFT, y: 254, gap: 40, labelY: 246 }); // bytes · FLOPs · FLOPs per byte
export const NEUTRAL = 1e15; // a cell's value-scale maxAbs: readout cells stay neutral (the number is the reading)

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a));
export const lerp = (a, b, t) => a + (b - a) * t;
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

// Several lines from (x, y) down, one per LINE; `shown` lines are drawn (the rest wait to type in).
export function lines(parent, x, y, texts, { shown = texts.length, cls = '' } = {}) {
  texts.slice(0, shown).forEach((t, i) => note(parent, x, y + i * LINE, t, { cls }));
}

// The four token chips, "sat" carrying the one selection mark (template rule 5).
export function chips(parent, { opacity = 1 } = {}) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  let x = CHIP_ROW.x;
  TOKENS.forEach((text, i) => {
    G.token(g, { x, y: CHIP_ROW.y, text, index: i + 1 });
    if (i === SAT) G.selectionMark(g, { x, y: CHIP_ROW.y, w: G.tokenWidth(text), h: 24 });
    x += G.tokenWidth(text) + CHIP_ROW.gap;
  });
}

// A readout cell at NUMBER_CELL (neutral fill); null draws the empty slot it will type into. `link` wraps it for the math panel.
export function readCell(parent, { x, y, value, label, format = String, link = null }) {
  const host = link ? linkGroup(parent, link, { x, y, w: CELL, h: CELL }) : parent;
  if (label) note(parent, x, y - 8, label);
  if (value == null) {
    const g = G.svgEl('g', { class: 'glyph g-empty' }, host);
    G.svgEl('rect', { class: 'g-frame', x: 1.5, y: 1.5, width: CELL - 3, height: CELL - 3, rx: 3, transform: `translate(${x} ${y})` }, g);
    return;
  }
  G.cell(host, { x, y, size: CELL, v: value, maxAbs: NEUTRAL, format });
}

// A math-panel link target (template rule 8): <g data-link> with an invisible frame the hover outlines.
export function linkGroup(parent, letter, { x, y, w, h }) {
  const g = G.svgEl('g', { 'data-link': letter }, parent);
  G.svgEl('rect', { class: 'g-frame', x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 3, fill: 'none', stroke: 'none' }, g);
  return g;
}

// A shape-only matrix: its frame and its label, for sizes no grid can draw (X [4 × 8,192]).
export function shape(parent, { x, y, w, h, label }) {
  const g = G.svgEl('g', { class: 'glyph g-matrix g-shape', transform: `translate(${x} ${y})` }, parent);
  G.svgEl('rect', { class: 'g-frame', x: 0, y: 0, width: w, height: h, rx: 2 }, g);
  G.svgEl('text', { x: 0, y: -6, class: 'g-label' }, g).textContent = label;
  return g;
}

// The gpu glyph's own geometry (shared/glyphs/systems.js): die, HBM stacks and the gap between the two stacks.
export function gpuParts({ x, y, w, h }) {
  const dieW = Math.round(w * 0.68);
  const dieH = h - 18;
  return {
    die: { x: x + 6, y: y + 6, w: dieW, h: dieH - 12 },
    hbm: { x: x + dieW + 12, y: y + 6, w: w - dieW - 18, h: dieH - 12 },
    tile: { x: x + 9, y: y + 9, w: (dieW - 10) / 4 - 2, h: (dieH - 22) / 3 - 2 },
    arrowY: y + 6 + (dieH - 12) / 2 - 1.5,
  };
}

// The GPU; `parts` fades the die and the HBM side separately (frame 1) through two clip rects, never by editing the glyph.
export function gpu(parent, box, { litSms = null, die = 1, hbm = 1, label = 'H100' } = {}) {
  const P = gpuParts(box);
  const draw = (host) => G.gpu(host, { ...box, label, showMem: false, litSms });
  if (die >= 1 && hbm >= 1) return draw(parent);
  const svg = parent.ownerSVGElement ?? parent;
  const defs = svg.querySelector('defs');
  [['die', die, box.x - 2, P.hbm.x - 3 - box.x + 2], ['hbm', hbm, P.hbm.x - 3, box.x + box.w + 2 - (P.hbm.x - 3)]].forEach(([name, o, cx, cw]) => {
    if (o <= 0) return;
    const id = `${svg.dataset.hatchId}-gpu-${name}`;
    const clip = G.svgEl('clipPath', { id }, defs);
    G.svgEl('rect', { x: cx, y: box.y - 2, width: cw, height: box.h + 20 }, clip);
    draw(G.svgEl('g', { 'clip-path': `url(#${id})`, opacity: o < 1 ? o.toFixed(3) : null }, parent));
  });
  return null;
}

// Numbers moving between HBM and the SMs: a flow across the HBM stacks into the die (or back), linked to "BW".
export function hbmFlow(parent, box, { progress, carry = 'activation', back = false }) {
  const P = gpuParts(box);
  const [a, b] = [[P.hbm.x + P.hbm.w - 4, P.arrowY], [P.die.x + P.die.w - 4, P.arrowY]];
  const g = linkGroup(parent, 'bw', { x: b[0], y: P.arrowY - 6, w: a[0] - b[0], h: 12 });
  G.flow(g, { from: back ? b : a, to: back ? a : b, carry, progress });
}

// A chip preset for the stage, in chipPreset's shape (hardware.js), built from the stage constants (numbers.js).
export function stagePreset(id) {
  const c = STAGE_CHIPS[id];
  return {
    id, label: c.label, formats: { bf16: 'bf16', fp8: 'fp8_e4m3', fp4: 'nvfp4' },
    peak: { bf16: c.peak.bf16 ?? null, fp8: c.peak.fp8 ?? null, fp4: c.peak.fp4 ?? null }, bandwidths: c.bandwidths,
  };
}

// The roofline at its fixed place, the chip and format named over it, with math-panel links on its parts (template rule 8:
// invisible frames placed from rooflineLayout, drawn after the glyph, never editing it).
export function roof(parent, { peakTflops, bandwidthTBps, title, points = [], opacity = 1, xDomain = PLOT.xDomain, yDomain = PLOT.yDomain }) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  const spec = { w: PLOT.w, h: PLOT.h, peakTflops, bandwidthTBps, xDomain, yDomain, points };
  G.roofline(g, { x: PLOT.x, y: PLOT.y, ...spec, label: title });
  note(g, PLOT.x + PLOT.w, PLOT.y + 11, title, { cls: '', anchor: 'end' });
  const L = G.rooflineLayout(spec);
  const [memory, compute] = [L.plot.series.find((s) => s.tone === 'memory'), L.plot.series.find((s) => s.tone === 'compute')];
  const box = (pts) => ({ x: PLOT.x + Math.min(...pts.map((p) => p.x)), y: PLOT.y + Math.min(...pts.map((p) => p.y)) - 3, w: Math.max(...pts.map((p) => p.x)) - Math.min(...pts.map((p) => p.x)), h: Math.max(...pts.map((p) => p.y)) - Math.min(...pts.map((p) => p.y)) + 6 });
  if (memory) linkGroup(g, 'bw', box(memory.points));
  if (compute) linkGroup(g, 'peak', box(compute.points));
  if (L.ridge.label) linkGroup(g, 'ridge', { x: PLOT.x + L.ridge.x - 4 - L.ridge.label.length * 6.6, y: PLOT.y + L.ridge.y - 18, w: L.ridge.label.length * 6.6, h: 13 });
  L.points.filter((p) => p.followed).forEach((p) => linkGroup(g, 'int', { x: PLOT.x + p.x - 7, y: PLOT.y + p.y - 7, w: 14, h: 14 }));
}

// Two roof states crossing over (a chip, format or followed-dot change): every printed number is a real state.
export function roofFade(parent, from, to, t) {
  if (t < 1) roof(parent, { ...from, opacity: (from.opacity ?? 1) * (1 - t) });
  if (t > 0) roof(parent, { ...to, opacity: (to.opacity ?? 1) * t });
}

// The real-size multiply X [m × 8,192] · W on a chip: m may be fractional while a counter tweens between two real stops.
export function realCost(m, peakTflops, bandwidthTBps, bytesPerElem = 2) {
  const cost = matmulCost({ m, k: D_REAL, n: D_REAL, bytesPerElem });
  return { cost, intensity: arithmeticIntensity(cost), time: rooflineTime({ ...cost, peakTflops, bandwidthTBps }) };
}

// The memory and compute lanes for one real-size multiply (µs). `scale` fixes px per µs (frames 7 and 9 hold one axis);
// `grow` (0–1) extends the lanes from 0 (frame 7); labels print once a lane has reached its length.
export function lanes(parent, { time, scale = null, grow = 1, labels = true, opacity = 1, full = null }) {
  if (opacity <= 0 || grow <= 0) return;
  const mem = time.memoryS * 1e6;
  const comp = time.computeS * 1e6;
  const end = full ?? Math.max(mem, comp);
  const at = end * grow;
  const lane = (label, us, kind) => {
    const to = Math.min(us, at);
    const segs = [{ from: 0, to, kind, label: labels && grow >= 1 ? formatDuration(us / 1e6) : null }];
    const stepEnd = Math.min(Math.max(mem, comp), at);
    if (to < stepEnd) segs.push({ from: to, to: stepEnd, kind: 'idle' });
    return { label, segments: segs };
  };
  G.laneTimeline(layer(parent, opacity), { x: LANES.x, y: LANES.y, w: LANES.w, lanes: [lane('memory', mem, 'memory'), lane('compute', comp, 'compute')], scale, label: 'time lanes' });
}

// px per µs that makes `us` fill the lane track (laneTimeline's gutter: 6.6 px per label character + 6).
export const laneScale = (us) => (LANES.w - (Math.round('compute'.length * 6.6) + 6)) / us;
export const laneNote = (parent, us, opacity = 1) => {
  if (opacity > 0) note(layer(parent, opacity), LANES.x, LANES.noteY, `lanes share one time axis; full width = ${formatDuration(us / 1e6)}`);
};
