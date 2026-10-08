// Frames 1–4: the five cuts, data parallelism and the two tensor-parallel matrices (storyboard §5). Each frame is a
// pure function of its progress p (0 → 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { formatBytes } from '@math/core.js';
import { ringAllReduceBytes } from '@math/parallel.js';
import { BYTES_PER_NUMBER, DP_GPUS, DP_SENT_BYTES, GPT3_DP_GPUS, GPT3_GRADIENT_BYTES, GRADIENT_BYTES, MLP_PARAMS_PER_GPU, PARTIAL_GPU1, PARTIAL_GPU2, MLP_OUT_SAT, SAT, TOKENS, TOY, TP_OUTPUT_BYTES, TP_SENT_BYTES, D_MODEL, MLP_HIDDEN } from './numbers.js';
import { bytesExact, int } from './format.js';
import { BLANK, chipRow, counter, ease, fade, gpuSwatch, label, lerp, linked, noNumber, rule, seg, select, zeros } from './stage.js';

// ---- frame 1: the cut map ----
const CHIPS = { x: 24, y: 72 };
const BLOCKS = { x: 230, w: 150, h: 40, attnY: 60, mlpY: 116 };
const CUTS = Object.freeze([
  { name: 'data', from: [16, 190], to: [564, 190], at: [564, 182], anchor: 'end' },
  { name: 'tensor', from: [268, 48], to: [268, 162], at: [268, 38], anchor: 'middle' },
  { name: 'pipeline', from: [222, 108], to: [388, 108], at: [394, 108], anchor: 'start' },
  { name: 'context', from: [102, 62], to: [102, 106], at: [102, 52], anchor: 'middle' },
  { name: 'expert', from: [343, 114], to: [343, 158], at: [343, 172], anchor: 'middle' },
]);

export function drawFrame1(svg, p) {
  chipRow(svg, CHIPS.x, CHIPS.y);
  G.block(svg, { x: BLOCKS.x, y: BLOCKS.attnY, w: BLOCKS.w, h: BLOCKS.h, label: 'attention' });
  G.block(svg, { x: BLOCKS.x, y: BLOCKS.mlpY, w: BLOCKS.w, h: BLOCKS.h, label: 'MLP' });
  G.flow(svg, { from: [192, 84], to: [BLOCKS.x - 2, 84], carry: 'activation', progress: 0.5 });
  label(svg, CHIPS.x, 204, 'another sequence', { cls: 'g-label' });
  chipRow(svg, CHIPS.x, 216, { opacity: 0.4, mark: false });
  const step = 1 / CUTS.length;
  CUTS.forEach((cut, i) => {
    const t = ease(seg(p, i * step, i * step + step * 0.7));
    if (t <= 0) return;
    rule(svg, cut.from[0], cut.from[1], lerp(cut.from[0], cut.to[0], t), lerp(cut.from[1], cut.to[1], t));
    if (t >= 1) label(svg, cut.at[0], cut.at[1], cut.name, { anchor: cut.anchor });
  });
  label(svg, 24, 330, 'degree: how many GPUs share one cut', { cls: 'g-label' });
  label(svg, 24, 352, `${CUTS.length} cuts`, { opacity: seg(p, 0.9, 1) });
}

// ---- frame 2: data parallelism ----
const DP = { gpu1: { x: 50, y: 130 }, gpu2: { x: 434, y: 130 }, chipY: 88 };

export function drawFrame2(svg, p) {
  const drop = ease(seg(p, 0, 0.25));
  const chipY = lerp(DP.chipY - 30, DP.chipY, drop);
  chipRow(svg, 16, chipY, { opacity: drop });
  chipRow(svg, 398, chipY, { opacity: drop * 0.4, mark: false });
  label(svg, 398, chipY - 12, 'another 4-token sequence', { cls: 'g-label', opacity: drop });
  [DP.gpu1, DP.gpu2].forEach((pos, i) => {
    G.gpu(svg, { ...pos, label: `GPU ${i + 1}`, showMem: false });
    label(svg, pos.x + 48, pos.y + 104, `toy model: ${int(TOY.total)} parameters`, { anchor: 'middle', cls: 'g-label' });
  });
  const pulse = seg(p, 0.25, 0.5);
  if (pulse > 0 && p < 0.5) label(svg, 290, 100, 'forward, then backward', { anchor: 'middle', opacity: Math.sin(pulse * Math.PI) });
  const t = seg(p, 0.5, 0.9);
  if (p >= 0.5) {
    G.flow(svg, { from: [160, 148], to: [428, 148], carry: 'gradient', progress: t });
    G.flow(svg, { from: [428, 182], to: [160, 182], carry: 'gradient', progress: t });
    label(svg, 294, 120, `gradients ${int(TOY.total)} × ${BYTES_PER_NUMBER} B = ${bytesExact(GRADIENT_BYTES)}`, { anchor: 'middle' });
  }
  const allreduce = `all-reduce over ${DP_GPUS} GPUs: 2 × ½ × ${int(GRADIENT_BYTES)} = ${bytesExact(DP_SENT_BYTES)} per GPU`;
  label(svg, 290, 262, allreduce, { anchor: 'middle', opacity: seg(p, 0.5, 0.7) });
  const gpt3 = formatBytes(ringAllReduceBytes(GPT3_GRADIENT_BYTES, GPT3_DP_GPUS));
  label(svg, 290, 286, `GPT-3 shape on ${GPT3_DP_GPUS} GPUs: ${gpt3} per GPU per step`, { anchor: 'middle', opacity: seg(p, 0.7, 0.9) });
  label(svg, 290, 302, '(plain data parallel, BF16 gradients, no sharding)', { anchor: 'middle', cls: 'g-label', opacity: seg(p, 0.7, 0.9) });
  label(svg, 290, 326, 'ZeRO shards this state: training-memory', { anchor: 'middle', cls: 'g-label' });
  counter(svg, `sent per GPU: ${bytesExact(Math.round(DP_SENT_BYTES * seg(p, 0.5, 1)))}`);
}

// ---- frame 3: tensor parallelism, columns ----
const CELL_S = 12; // shape-only grids: no numbers, so a small cell is fine
const HALF = 8 * CELL_S;
const W_AT = { x: 150, in: 78, gate: 204 };
const HIDDEN = { x: 410, gpu1: 78, gpu2: 204 };

function shape(svg, { x, y, rows, cols, gpu, cell = CELL_S, title }) {
  return G.matrix(svg, { x, y, values: zeros(rows, cols), cell, label: title, format: BLANK, shards: gpu ? [{ cols: [1, cols], gpu }] : null });
}

function splitMatrix(svg, { y, name, split }) {
  const gap = 6 * split;
  label(svg, W_AT.x, y - 12, `${name} [${D_MODEL} × ${MLP_HIDDEN}]`, { cls: 'g-label' });
  shape(svg, { x: W_AT.x - gap, y, rows: 8, cols: 8, gpu: 1 });
  shape(svg, { x: W_AT.x + HALF + gap, y, rows: 8, cols: 8, gpu: 2 });
}

export function drawFrame3(svg, p) {
  gpuSwatch(svg, 24, 14, 1);
  gpuSwatch(svg, 110, 14, 2);
  G.matrix(svg, { x: 62, y: 130, values: zeros(TOKENS.length, D_MODEL), cell: 10, label: 'X', rowLabels: TOKENS, format: BLANK });
  select(svg, 62, 130 + SAT * 10, D_MODEL * 10, 10);
  const split = ease(seg(p, 0.15, 0.55));
  splitMatrix(svg, { y: W_AT.in, name: 'W_in', split });
  splitMatrix(svg, { y: W_AT.gate, name: 'W_gate', split });
  const fill = seg(p, 0.55, 1);
  [[HIDDEN.gpu1, 1], [HIDDEN.gpu2, 2]].forEach(([y, gpu]) => {
    const g = shape(svg, { x: HIDDEN.x, y, rows: TOKENS.length, cols: 8, gpu, cell: 14, title: `GPU ${gpu} hidden` });
    fade(g, fill);
    select(svg, HIDDEN.x, y + SAT * 14, 8 * 14, 14, fill);
  });
  const cut = split > 0 ? 1 : 0;
  if (cut) label(svg, HIDDEN.x, 168, 'cut by columns', { cls: 'g-label', opacity: seg(p, 0.15, 0.3) });
  label(svg, 24, 322, `each GPU: W_in, W_gate [${D_MODEL} × ${D_MODEL}] · hidden [${TOKENS.length} × ${D_MODEL}]`, { opacity: fill });
  counter(svg, `sent per GPU: ${bytesExact(0)}`);
}

// ---- frame 4: tensor parallelism, rows, and the all-reduce of partial sums ----
const VEC_X = 232;
const ROW = { gpu1: 84, gpu2: 136, sum: 232 };

function partial(parent, y, values, name, opacity = 1) {
  const g = G.vector(parent, { x: VEC_X, y, values, cell: G.NUMBER_CELL, orient: 'row', maxAbs: 1, label: name, format: noNumber });
  fade(g, opacity);
  select(parent, VEC_X, y, values.length * G.NUMBER_CELL, G.NUMBER_CELL, opacity);
}

const typeIn = (row, t) => row.map((v, j) => (t * row.length >= j + 1 - 1e-9 ? v : Number.NaN));

export function drawFrame4(svg, p) {
  gpuSwatch(svg, 24, 14, 1);
  gpuSwatch(svg, 110, 14, 2);
  G.matrix(svg, { x: 24, y: 84, values: zeros(MLP_HIDDEN, D_MODEL), cell: 10, label: 'W_out', format: BLANK, shards: [{ rows: [1, 8], gpu: 1 }, { rows: [9, 16], gpu: 2 }] });
  label(svg, 24, 258, 'cut by rows: 1–8, 9–16', { cls: 'g-label' });
  const appear = ease(seg(p, 0, 0.3));
  partial(svg, ROW.gpu1, PARTIAL_GPU1, 'GPU 1 sat', appear);
  partial(svg, ROW.gpu2, PARTIAL_GPU2, 'GPU 2 sat', appear);
  if (p > 0.3) G.flow(svg, { from: [VEC_X + 160, ROW.gpu2 + 46], to: [VEC_X + 160, ROW.sum - 6], carry: 'activation', progress: ease(seg(p, 0.3, 0.6)) });
  if (p > 0.3) label(svg, VEC_X + 172, 205, 'all-reduce: add', { opacity: seg(p, 0.3, 0.45) });
  const sumLink = linked(svg, 'comm', { x: VEC_X, y: ROW.sum, w: MLP_OUT_SAT.length * G.NUMBER_CELL, h: G.NUMBER_CELL });
  partial(sumLink, ROW.sum, typeIn(MLP_OUT_SAT, seg(p, 0.6, 1)), 'sum');
  label(svg, 24, 300, `[${TOKENS.length} × ${D_MODEL}] × ${BYTES_PER_NUMBER} B = ${bytesExact(TP_OUTPUT_BYTES)}; all-reduce over 2: ${bytesExact(TP_SENT_BYTES)} per GPU`);
  label(svg, 24, 316, `per GPU ${MLP_PARAMS_PER_GPU} of ${TOY.perLayer.mlp} MLP parameters · 2 all-reduces per block in the forward pass`);
  label(svg, 8, 332, 'hand-picked stand-in partial sums; they add up to decoder-anatomy\'s MLP output for sat', { cls: 'g-label' });
  counter(svg, `sent per GPU: ${bytesExact(Math.round(TP_SENT_BYTES * seg(p, 0.3, 0.6)))}`);
}
