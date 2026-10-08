// training-memory frames 8–10: 64 data-parallel GPUs. Plain data parallelism keeps a full copy on each, ZeRO-1 shards the
// optimizer state, ZeRO-3 (FSDP) shards gradients and weights too. GPU 1 is the followed GPU: its composition is the bar.
import * as G from '@shared/glyphs.js';
import { formatBytes } from '@math/core.js';
import { activationBytes } from '@math/training-memory.js';
import { gbText, trafficText, activationArgs, INITIAL_STATE } from './format.js';
import { ZERO_DP, H100_HBM } from './numbers.js';
import { seg, lerp, note, layer, linkFrame, zeroState, DP_ROW, COMPOSITION } from './stage.js';

const GPUS = Object.freeze([{ x: DP_ROW.xs[0], name: 'GPU 1', batch: 'batch 1', slice: 1 }, { x: DP_ROW.xs[1], name: 'GPU 2', batch: 'batch 2', slice: 2 }, { x: DP_ROW.xs[2], name: 'GPU 64', batch: 'batch 64', slice: 64 }]);
const center = (x) => x + DP_ROW.w / 2;
const ZERO = Object.freeze([0, 1, 2, 3].map(zeroState));
const GB_TEXT_Y = DP_ROW.y + DP_ROW.h + 30;
const SEGMENT_LETTERS = Object.freeze(['w', 'g', 'o']);

// Over capacity: "needs 2,800 GB" / "of 80" on two lines (one line is wider than the 130 px between GPUs); otherwise one line.
const memLines = (bytes) => (bytes > H100_HBM ? [`needs ${gbText(bytes)}`, 'of 80'] : [`${gbText(bytes)} of 80`]);

// The row of three GPUs (and "61 others"), each printing its GB; GPU 1 carries the one selection mark.
function gpuRow(svg, bytes, { batch = 1, slices = 0 } = {}) {
  GPUS.forEach((gpu, i) => {
    G.gpu(svg, { x: gpu.x, y: DP_ROW.y, w: DP_ROW.w, h: DP_ROW.h, memFill: bytes / H100_HBM, label: gpu.name });
    memLines(bytes).forEach((text, k) => note(svg, center(gpu.x), GB_TEXT_Y + k * 14, text, { anchor: 'middle' }));
    if (slices > 0 && i > 0) note(svg, center(gpu.x), GB_TEXT_Y + 28, `slice ${gpu.slice} of ${ZERO_DP}`, { anchor: 'middle', opacity: slices });
    if (batch > 0) note(svg, center(gpu.x), DP_ROW.y - 24 - (1 - batch) * 14, gpu.batch, { anchor: 'middle', opacity: batch });
  });
  note(svg, DP_ROW.others, DP_ROW.y + DP_ROW.h / 2 + 4, '61 others', { cls: 'g-text', anchor: 'middle' });
  G.selectionMark(svg, { x: GPUS[0].x, y: DP_ROW.y, w: DP_ROW.w, h: DP_ROW.h });
}

// GPU 1's composition at full width: weights · gradients · optimizer (moments + master). Returns the optimizer segment.
function composition(svg, z) {
  const values = [z.weights, z.grads, z.optimizer];
  const total = values.reduce((s, v) => s + v, 0);
  G.shareBar(svg, { x: COMPOSITION.x, y: COMPOSITION.y, w: COMPOSITION.w, h: COMPOSITION.h, tail: 'none', label: 'GPU 1 state', format: (share) => gbText(share * total),
    parts: [{ name: 'weights', value: z.weights, hue: 1 }, { name: 'gradients', value: z.grads, hue: 2 }, { name: 'optimizer', value: z.optimizer, hue: 3 }] });
  const segs = G.barSegments(values, COMPOSITION.w);
  segs.forEach((s, i) => linkFrame(svg, SEGMENT_LETTERS[i], { x: COMPOSITION.x + s.x, y: COMPOSITION.y, w: s.width, h: COMPOSITION.h }));
  return { x: COMPOSITION.x + segs[2].x, w: segs[2].width };
}

const blend = (a, b, t) => ({ weights: lerp(a.weights, b.weights, t), grads: lerp(a.grads, b.grads, t), optimizer: lerp(a.optimizer, b.optimizer, t) });
const sum = (z) => z.weights + z.grads + z.optimizer;

export function drawFrame8(svg, p) {
  const z = ZERO[0];
  gpuRow(svg, sum(z), { batch: Math.max(seg(p, 0.1, 0.5), 0) });
  composition(svg, z);
  note(svg, COMPOSITION.x, 346, `per GPU ${gbText(sum(z))} · ${ZERO_DP} copies: ${formatBytes(sum(z) * ZERO_DP)} in total`, { opacity: seg(p, 0.5, 0.9) });
}

// Pieces of the optimizer segment flying off to the other GPUs: a handful of dots for the 63 slices.
function pieces(svg, from, t) {
  if (t <= 0 || t >= 1) return;
  const g = G.svgEl('g', { class: 'glyph g-flow g-flow--weight', opacity: (1 - t * 0.6).toFixed(3) }, svg);
  [GPUS[1], GPUS[2], GPUS[1], GPUS[2], GPUS[1], GPUS[2], GPUS[1]].forEach((target, i) => {
    const goal = { x: center(target.x) + (i - 3) * 6, y: DP_ROW.y + DP_ROW.h - 6 };
    G.svgEl('circle', { class: 'g-dot', cx: lerp(from.x, goal.x, t).toFixed(2), cy: lerp(from.y, goal.y, t).toFixed(2), r: 3 }, g);
  });
}

// "slice 1 of 64" with a leader line to the (18.8 px) optimizer slice of GPU 1's bar.
function sliceLabel(svg, mid, opacity) {
  if (opacity <= 0) return;
  const l = layer(svg, opacity);
  G.svgEl('line', { class: 'g-link', x1: mid, y1: COMPOSITION.y - 2, x2: mid, y2: COMPOSITION.y - 10 }, l);
  note(l, mid + 6, COMPOSITION.y - 14, `slice 1 of ${ZERO_DP}`, { anchor: 'end' });
}

export function drawFrame9(svg, p) {
  const t = seg(p, 0.2, 0.7);
  const z = blend(ZERO[0], ZERO[1], t);
  gpuRow(svg, sum(z), { batch: 1, slices: seg(p, 0.6, 0.9) });
  const slice = composition(svg, z);
  const mid = slice.x + slice.w / 2;
  pieces(svg, { x: mid, y: COMPOSITION.y + COMPOSITION.h / 2 }, t);
  sliceLabel(svg, mid, seg(p, 0.6, 0.9));
}

function arrows(svg, run) {
  if (run <= 0) return;
  const from = [center(GPUS[1].x), DP_ROW.y + DP_ROW.h / 2];
  G.flow(svg, { from: [GPUS[1].x, from[1]], to: [GPUS[0].x + DP_ROW.w + 2, from[1]], carry: 'weight', progress: run });
  G.flow(svg, { from: [center(GPUS[2].x), DP_ROW.y - 10], to: [center(GPUS[0].x), DP_ROW.y - 10], carry: 'weight', progress: run });
}

export function drawFrame10(svg, p) {
  const gradsT = seg(p, 0.1, 0.4);
  const weightsT = seg(p, 0.4, 0.7);
  const z = { weights: lerp(ZERO[1].weights, ZERO[3].weights, weightsT), grads: lerp(ZERO[1].grads, ZERO[3].grads, gradsT), optimizer: ZERO[1].optimizer };
  gpuRow(svg, sum(z), { batch: 1 });
  const slice = composition(svg, z);
  sliceLabel(svg, slice.x + slice.w / 2, 1 - seg(p, 0, 0.2));
  arrows(svg, seg(p, 0.5, 0.95));
  const done = seg(p, 0.7, 1);
  const act = activationBytes(activationArgs({ ...INITIAL_STATE, recompute: 'full' }));
  note(svg, 125, DP_ROW.y + DP_ROW.h + 68, 'all-gather before the layer, forward and again backward', { opacity: seg(p, 0.5, 0.8) });
  const stages = ZERO.map((s, i) => `ZeRO-${i} ${gbText(sum(s)).replace(' GB', '')}`).join(' · ');
  note(svg, COMPOSITION.x, 322, `${stages} GB`, { opacity: done });
  note(svg, COMPOSITION.x, 338, `+ activations ${gbText(act)} (full recompute) = ${gbText(sum(ZERO[3]) + act)} of 80: fits`, { opacity: done });
  note(svg, COMPOSITION.x, 354, `traffic: ${trafficText(3)} plain data parallel`, { opacity: done });
}
