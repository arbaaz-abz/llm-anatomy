// scale-reliability frames 1–3: 6ND, the GPU-hours it needs at peak, and the gap to the model card's total.
// Frames 2 and 3 draw one bar on a shared scale (the 100,000-GPU run's GPU-hours = full width, as in frame 10).
import * as G from '@shared/glyphs.js';
import { formatCount, formatRatio } from '@math/core.js';
import { trainingFlops, FLOPS_PER_PARAM_TOKEN, gpuHoursAt, wallClockDays, mfuFrom } from '@math/scale.js';
import { LLAMA, H100 } from './numbers.js';
import { SAME_SCALE_GPU_HOURS } from './runs.js';
import { gpuHours, sci, int, daysText, pct, bandText } from './format.js';
import { MARGIN, seg, ease, layer, note, typed, linked, tick, barWidth } from './stage.js';

const FLOPS = trainingFlops({ params: LLAMA.params, tokens: LLAMA.tokens });
const AT_PEAK = gpuHoursAt({ flops: FLOPS, peakTflops: H100.bf16, mfu: 1 });
const AT_MFU = gpuHoursAt({ flops: FLOPS, peakTflops: H100.bf16, mfu: LLAMA.toyMfu });
const RUN_AVERAGE = mfuFrom({ flops: FLOPS, gpuHours: LLAMA.gpuHours, peakTflops: H100.bf16 });

const BAR_Y = 92;
const BAR_H = 14;
const LINE_Y = Object.freeze([222, 240, 258, 276]);

// Frame 1: the equation card 6 × N × D with Llama 3.1 405B's numbers dropping in.
const EQ = Object.freeze({ y: 52, h: 40, gap: 24, widths: [56, 112, 112, 176] });
const EQ_W = EQ.widths.reduce((s, w) => s + w, 0) + 3 * EQ.gap;
const eqX = (i) => MARGIN + EQ.widths.slice(0, i).reduce((s, w) => s + w + EQ.gap, 0);

export function drawFrame1(svg, p) {
  note(svg, MARGIN, 38, 'Llama 3.1 405B');
  const factors = [
    { label: String(FLOPS_PER_PARAM_TOKEN), sub: 'FLOPs', at: [0.0, 0.18] },
    { label: `N = ${formatCount(LLAMA.params)}`, sub: 'parameters', at: [0.14, 0.32] },
    { label: `D = ${formatCount(LLAMA.tokens)}`, sub: 'tokens', at: [0.28, 0.46] },
  ];
  const row = linked(svg, 'flops', { x: MARGIN, y: EQ.y, w: EQ_W, h: EQ.h });
  factors.forEach((f, i) => {
    const t = ease(seg(p, ...f.at));
    const g = layer(row, t);
    g.setAttribute('transform', `translate(0 ${(-(1 - t) * 18).toFixed(2)})`);
    G.block(g, { x: eqX(i), y: EQ.y, w: EQ.widths[i], h: EQ.h, label: f.label });
    if (i < 3) note(g, eqX(i) + EQ.widths[i] + EQ.gap / 2, EQ.y + EQ.h / 2 + 4, i < 2 ? '×' : '=', { cls: 'g-label', anchor: 'middle' });
  });
  const product = typed(sci(FLOPS), seg(p, 0.55, 0.9));
  G.block(layer(row, seg(p, 0.5, 0.58)), { x: eqX(3), y: EQ.y, w: EQ.widths[3], h: EQ.h, label: product });
  note(layer(svg, seg(p, 0.55, 0.7)), eqX(3) + EQ.widths[3] / 2, EQ.y + EQ.h + 16, 'FLOPs to train', { anchor: 'middle' });
  factors.forEach((f, i) => note(layer(svg, ease(seg(p, ...f.at))), eqX(i) + EQ.widths[i] / 2, EQ.y + EQ.h + 16, f.sub, { anchor: 'middle' }));
  G.selectionMark(layer(svg, seg(p, 0.5, 0.9)), { x: MARGIN, y: EQ.y, w: EQ_W, h: EQ.h });
  const split = layer(svg, ease(seg(p, 0.45, 0.7)));
  G.shareBar(split, {
    x: MARGIN, y: 150, w: 300, parts: [{ name: 'forward', value: 2, hue: 1 }, { name: 'backward', value: 4, hue: 2 }],
    format: (share) => String(Math.round(share * FLOPS_PER_PARAM_TOKEN)), label: 'FLOPs per parameter per token', tail: 'none',
  });
  const why = layer(svg, ease(seg(p, 0.6, 0.85)));
  note(why, MARGIN, 268, 'why 6: each multiply-add is 2 FLOPs;');
  note(why, MARGIN, 284, 'backward does two matmuls for each forward one');
  note(why, MARGIN, 312, 'ignores attention FLOPs, which grow at long context');
}

// The run bar of frames 2 and 3: GPU-hours as length on the shared scale, the model card's total as a plain tick.
function runBar(svg, parts, { link, hours, format }) {
  const w = Math.max(barWidth(hours, SAME_SCALE_GPU_HOURS), 1);
  const wrap = linked(svg, link, { x: MARGIN, y: BAR_Y, w, h: BAR_H });
  G.shareBar(wrap, { x: MARGIN, y: BAR_Y, w, h: BAR_H, parts, format, label: 'GPU-hours to train Llama 3.1 405B', tail: 'none' });
  G.selectionMark(svg, { x: MARGIN, y: BAR_Y, w, h: BAR_H });
}

const cardTick = (svg, opacity, dy = 0) => tick(svg, { x: MARGIN + barWidth(LLAMA.gpuHours, SAME_SCALE_GPU_HOURS), y: BAR_Y - 20, h: BAR_H + 26, label: `model card: ${gpuHours(LLAMA.gpuHours)}`, opacity, dy });

export function drawFrame2(svg, p) {
  note(svg, MARGIN, 48, 'GPU-hours to train Llama 3.1 405B');
  const grow = ease(seg(p, 0, 0.5));
  if (grow > 0.01) runBar(svg, [{ name: 'at peak', value: AT_PEAK * grow, hue: 1 }], { link: 'peak', hours: AT_PEAK * grow, format: () => gpuHours(AT_PEAK * grow) });
  const drop = ease(seg(p, 0.5, 0.85));
  if (drop > 0) cardTick(svg, drop, -(1 - drop) * 16);
  const days = daysText(wallClockDays({ gpuHours: AT_PEAK, gpus: LLAMA.gpus }));
  const lines = [
    `${sci(FLOPS)} ÷ (${int(H100.bf16)} TFLOPS × 3,600 s) = ${gpuHours(AT_PEAK)} GPU-hours`,
    `${gpuHours(AT_PEAK)} ÷ ${int(LLAMA.gpus)} GPUs ÷ 24 = ${days}`,
    `the model card's ${gpuHours(LLAMA.gpuHours)} is ${formatRatio(LLAMA.gpuHours / AT_PEAK)} that`,
  ];
  lines.forEach((line, i) => note(svg, MARGIN, LINE_Y[i], typed(line, seg(p, 0.15 + 0.2 * i, 0.5 + 0.2 * i)), { cls: '' }));
}

export function drawFrame3(svg, p) {
  note(svg, MARGIN, 48, 'GPU-hours to train Llama 3.1 405B');
  const grow = ease(seg(p, 0, 0.55));
  const below = (AT_MFU - AT_PEAK) * grow;
  runBar(svg, [{ name: 'useful (MFU)', value: AT_PEAK, hue: 1 }, { name: 'below peak', value: Math.max(below, 0), hue: 2 }], { link: 'mfu', hours: AT_PEAK + below });
  cardTick(svg, 1);
  const lines = [
    `MFU while training: ${bandText(LLAMA.mfuBand)} (paper)`,
    `run-average MFU: ${pct(RUN_AVERAGE)} or more (card; covers more stages than 6ND counts)`,
    `at ${pct(LLAMA.toyMfu, 0)} while training: ${gpuHours(AT_MFU)} GPU-hours, ${daysText(wallClockDays({ gpuHours: AT_MFU, gpus: LLAMA.gpus }))}`,
    `${pct(RUN_AVERAGE)} ÷ ${LLAMA.effective.toFixed(2)} = ${pct(RUN_AVERAGE / LLAMA.effective)}`,
  ];
  lines.forEach((line, i) => note(layer(svg, ease(seg(p, 0.3 + 0.15 * i, 0.55 + 0.15 * i))), MARGIN, LINE_Y[i], line, { cls: '' }));
}
