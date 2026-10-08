// scale-reliability frames 7–8: utilization against two different peaks, and 419 failures in 54 days on 16,384 GPUs.
import * as G from '@shared/glyphs.js';
import { achievedTflopsPerGpu, mfuFrom, trainingFlops, clusterMtbfHours } from '@math/scale.js';
import { LLAMA, DEEPSEEK, BEHEMOTH, H100, PER_GPU_MTBF_H, FAILURE_DAYS, FAILED_GPUS, RACK_GPUS, OTHER_NODES, HOURS_PER_YEAR } from './numbers.js';
import { int, pct, hoursText, gpuHours } from './format.js';
import { MARGIN, seg, ease, layer, note } from './stage.js';
import { formatCount } from '@math/core.js';

const COLUMN_W = 250;
const COLUMNS_X = Object.freeze([MARGIN, 310]);

// Frame 7: one column per run: its TFLOPS per GPU, then the share of each peak it is measured against.
function peakColumn(parent, x, { heading, sub, tflops, derivation, opacity }) {
  const g = layer(parent, opacity);
  note(g, x, 34, heading, { cls: '' });
  note(g, x, 52, sub);
  const shares = [tflops / H100.bf16, tflops / H100.fp8];
  G.bars(g, {
    x, y: 82, w: 210, h: 90, values: shares.map((s) => s * 100), max: 100, labels: [`BF16 ${int(H100.bf16)}`, `FP8 ${int(H100.fp8)}`],
    format: (v) => pct(v / 100, 0), label: `${heading} against two peaks`,
  });
  derivation.forEach((line, i) => note(g, x, 214 + i * 18, line, { cls: '' }));
  return g;
}

export function drawFrame7(svg, p) {
  const deepseekFlops = trainingFlops({ params: DEEPSEEK.params, tokens: DEEPSEEK.tokens });
  const deepseekTflops = achievedTflopsPerGpu({ flops: deepseekFlops, gpuHours: DEEPSEEK.pretrainGpuHours });
  const [bf16, fp8] = [H100.bf16, H100.fp8].map((peak) => mfuFrom({ flops: deepseekFlops, gpuHours: DEEPSEEK.pretrainGpuHours, peakTflops: peak }));
  peakColumn(svg, COLUMNS_X[0], {
    heading: 'Llama 4 Behemoth', sub: `${BEHEMOTH.tflopsPerGpu} TFLOPS per GPU (FP8)`, tflops: BEHEMOTH.tflopsPerGpu, opacity: ease(seg(p, 0.05, 0.45)),
    derivation: [`${BEHEMOTH.tflopsPerGpu} ÷ ${int(H100.bf16)} = ${pct(BEHEMOTH.tflopsPerGpu / H100.bf16)}`, `${BEHEMOTH.tflopsPerGpu} ÷ ${int(H100.fp8)} = ${pct(BEHEMOTH.tflopsPerGpu / H100.fp8)}`],
  });
  G.selectionMark(layer(svg, ease(seg(p, 0.05, 0.45))), { x: COLUMNS_X[0], y: 82, w: 210, h: 90 });
  peakColumn(svg, COLUMNS_X[1], {
    heading: 'DeepSeek-V3', sub: `${int(deepseekTflops)} TFLOPS per GPU (FP8), run-average`, tflops: deepseekTflops, opacity: ease(seg(p, 0.4, 0.8)),
    derivation: [`6 × ${formatCount(DEEPSEEK.params)} × ${formatCount(DEEPSEEK.tokens)} ÷ (${gpuHours(DEEPSEEK.pretrainGpuHours)} × 3,600 s)`, `= ${deepseekTflops.toFixed(1)} TFLOPS per GPU`, `${deepseekTflops.toFixed(1)} ÷ ${int(H100.bf16)} = ${pct(bf16)}`, `${deepseekTflops.toFixed(1)} ÷ ${int(H100.fp8)} = ${pct(fp8)}`],
  });
  note(layer(svg, ease(seg(p, 0.75, 1))), MARGIN, 330, 'measured against H100 peaks; neither lab gave an MFU');
}

const ROWS = 3; // 54 days as three 18-day rows: 419 ticks on one 500 px row would fuse into a solid bar
const ROW_DAYS = LLAMA.windowDays / ROWS;
const TRACK = Object.freeze({ x: 6, y: 118, w: 568, rowH: 34 });
const RACK = Object.freeze({ y: 26, xs: [MARGIN, 168] });
const GUTTER = Math.round('days 37–54'.length * 6.6) + 6;
const DAY_SCALE = (TRACK.w - GUTTER) / ROW_DAYS;

const rackLabels = (failed, show) => Array.from({ length: RACK_GPUS }, (_, i) => (show && i === failed ? '✕' : ''));

export function drawFrame8(svg, p) {
  const count = Math.max(1, Math.floor(LLAMA.interruptions * ease(seg(p, 0.1, 0.85))));
  RACK.xs.forEach((x, i) => G.rack(svg, { x, y: RACK.y, gpus: RACK_GPUS, label: `node ${i + 1}`, labels: rackLabels(FAILED_GPUS[i], p >= 0.3 + 0.3 * i) }));
  note(svg, 310, RACK.y + 24, `+ ${int(OTHER_NODES)} other nodes`, { cls: '' });
  note(svg, 310, RACK.y + 42, `${int(LLAMA.gpus)} GPUs in all, ${RACK_GPUS} per node`);
  note(svg, 310, RACK.y + 66, '✕ = one unexpected failure');
  for (let row = 0; row < ROWS; row += 1) {
    const [from, to] = [row * ROW_DAYS, (row + 1) * ROW_DAYS];
    const ticks = FAILURE_DAYS.slice(0, count).filter((d) => d >= from && d < to).map((d) => ({ t: d - from }));
    const lanes = [{ label: `days ${from + 1}–${to}`, segments: [{ from: 0, to: ROW_DAYS, kind: 'compute' }] }];
    G.laneTimeline(svg, { x: TRACK.x, y: TRACK.y + row * TRACK.rowH, w: TRACK.w, lanes, ticks: ticks.length ? ticks : [{ t: 0 }], scale: DAY_SCALE, label: `days ${from + 1} to ${to} of the run` });
  }
  G.selectionMark(svg, { x: TRACK.x + GUTTER, y: TRACK.y + 4, w: TRACK.w - GUTTER, h: (ROWS - 1) * TRACK.rowH + 24 });
  note(svg, MARGIN, 240, `${count} unexpected interruptions in ${LLAMA.windowDays} days`, { cls: '' });
  const mtbf = clusterMtbfHours({ perGpuMtbfHours: PER_GPU_MTBF_H, gpus: LLAMA.gpus });
  const lines = [
    `${LLAMA.windowDays} × 24 ÷ ${LLAMA.interruptions} = ${hoursText(mtbf)} between failures`,
    `× ${int(LLAMA.gpus)} GPUs = ${int(PER_GPU_MTBF_H)} h per GPU (${(PER_GPU_MTBF_H / HOURS_PER_YEAR).toFixed(1)} years)`,
    `${pct(LLAMA.hardwareShare, 0)} hardware, GPU issues ${pct(LLAMA.gpuShare)} of unexpected`,
  ];
  lines.forEach((line, i) => note(layer(svg, ease(seg(p, 0.85 + 0.05 * i, 0.95 + 0.05 * i))), MARGIN, 266 + i * 18, line, { cls: '' }));
}
