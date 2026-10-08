// scale-reliability frames 4–6: one training step's compute and exposed communication, overlap, and FP8 matmuls.
// The step is a stand-in (100 ms compute, 40 ms communication); the SM and FP8 facts are DeepSeek-V3's.
import * as G from '@shared/glyphs.js';
import { FORMATS } from '@math/roofline.js';
import { trainingStepMs } from '@math/scale.js';
import { STAND_IN, DEEPSEEK, FP8 } from './numbers.js';
import { pctOf } from './format.js';
import { MARGIN, seg, ease, lerp, layer, note } from './stage.js';

const LANES = Object.freeze({ x: 6, y: 40, w: 568, labels: ['compute', 'communication'] });
const GUTTER = Math.round(Math.max(...LANES.labels.map((l) => l.length)) * 6.6) + 6; // laneTimeline's lane-label gutter
const TOTAL_MS = STAND_IN.computeMs + STAND_IN.commMs;
const MS_SCALE = (LANES.w - GUTTER) / TOTAL_MS; // px per ms, held between frames 4 and 5
const TEXT_Y = 140;

function stepLanes(svg, { commFrom, commLen, idleLen }) {
  const idle = idleLen > 0.5 ? [{ from: STAND_IN.computeMs, to: STAND_IN.computeMs + idleLen, kind: 'idle', label: 'waiting' }] : [];
  const comm = commLen > 0.5 ? [{ from: commFrom, to: commFrom + commLen, kind: 'comm', label: `${STAND_IN.commMs} ms` }] : [];
  G.laneTimeline(svg, {
    x: LANES.x, y: LANES.y, w: LANES.w, scale: MS_SCALE, label: 'one training step',
    lanes: [
      { label: LANES.labels[0], segments: [{ from: 0, to: STAND_IN.computeMs, kind: 'compute', label: `${STAND_IN.computeMs} ms` }, ...idle] },
      { label: LANES.labels[1], segments: comm },
    ],
  });
}

const standIn = (svg) => note(svg, MARGIN, TEXT_Y, `stand-in: ${STAND_IN.computeMs} ms compute, ${STAND_IN.commMs} ms communication`);

export function drawFrame4(svg, p) {
  const comm = ease(seg(p, 0.1, 0.6));
  stepLanes(svg, { commFrom: STAND_IN.computeMs, commLen: STAND_IN.commMs * comm, idleLen: STAND_IN.commMs * comm });
  standIn(svg);
  const step = trainingStepMs({ computeMs: STAND_IN.computeMs, commMs: STAND_IN.commMs, overlap: false });
  const out = layer(svg, ease(seg(p, 0.6, 0.9)));
  note(out, MARGIN, TEXT_Y + 28, `step: ${STAND_IN.computeMs} + ${STAND_IN.commMs} = ${step} ms`, { cls: '' });
  note(out, MARGIN, TEXT_Y + 46, `tensor cores busy: ${STAND_IN.computeMs} ÷ ${step} = ${pctOf(STAND_IN.computeMs, step)}`, { cls: '' });
}

export function drawFrame5(svg, p) {
  const slide = ease(seg(p, 0.1, 0.6));
  stepLanes(svg, { commFrom: lerp(STAND_IN.computeMs, 0, slide), commLen: STAND_IN.commMs, idleLen: STAND_IN.commMs * (1 - slide) });
  standIn(svg);
  const step = trainingStepMs({ computeMs: STAND_IN.computeMs, commMs: STAND_IN.commMs, overlap: true });
  note(layer(svg, ease(seg(p, 0.6, 0.9))), MARGIN, TEXT_Y + 28, `overlapped step: max(${STAND_IN.computeMs}, ${STAND_IN.commMs}) = ${step} ms`, { cls: '' });
  const sms = layer(svg, ease(seg(p, 0.3, 0.7)));
  G.gpu(sms, { x: MARGIN, y: 214, w: 120, h: 84, showMem: false, label: 'one H100', litSms: [10, 11] });
  note(sms, 170, 240, `${DEEPSEEK.commSms} of ${DEEPSEEK.smsPerGpu} SMs run communication (DeepSeek-V3)`, { cls: '' });
  note(sms, 170, 258, `${DEEPSEEK.commSms} ÷ ${DEEPSEEK.smsPerGpu} = ${pctOf(DEEPSEEK.commSms, DEEPSEEK.smsPerGpu)} of the chip`);
}

const SCALE_GRID = Object.freeze({ cell: 40, y: 128, activations: 40, weights: 380, outlier: [1, 2], outlierScale: 8 });
const ones = () => Array.from({ length: 4 }, () => Array(4).fill(1));

export function drawFrame6(svg, p) {
  G.bitLayout(svg, { x: 110, y: 22, format: FORMATS.fp8_e4m3, label: 'FP8 E4M3' });
  G.bitLayout(svg, { x: 110, y: 66, format: FORMATS.bf16, label: 'BF16' });
  note(svg, 360, 36, 'E4M3 in the big matmuls', { cls: '' });
  note(svg, 360, 54, `loss error vs BF16 ${FP8.lossError}`);
  const { cell, y, outlier, outlierScale } = SCALE_GRID;
  const grow = Math.round(lerp(1, outlierScale, ease(seg(p, 0.2, 0.6))));
  const activations = ones();
  activations[outlier[0]][outlier[1]] = grow;
  G.matrix(svg, { x: SCALE_GRID.activations, y, values: activations, cell, maxAbs: outlierScale, label: 'activation scales' });
  G.matrix(svg, { x: SCALE_GRID.weights, y, values: ones(), cell, maxAbs: outlierScale, label: 'weight scales' });
  note(svg, SCALE_GRID.activations, y + 4 * cell + 18, 'one scale per 1 × 128 tile');
  note(svg, SCALE_GRID.weights, y + 4 * cell + 18, 'one scale per 128 × 128 block');
  const accumulate = ease(seg(p, 0.5, 0.95));
  G.block(svg, { x: 235, y: 150, w: 110, h: 32, label: 'GEMM in FP8' });
  G.flow(svg, { from: [290, 182], to: [290, 224], carry: 'activation', progress: accumulate });
  G.block(svg, { x: 220, y: 226, w: 140, h: 32, label: 'accumulate in FP32' });
  note(layer(svg, ease(seg(p, 0.3, 0.7))), MARGIN, 330, 'partial sums promoted to FP32 · tiles stand in for real shapes', { cls: '' });
  note(svg, MARGIN, 348, `kept in BF16 or FP32: ${FP8.kept}`);
}
