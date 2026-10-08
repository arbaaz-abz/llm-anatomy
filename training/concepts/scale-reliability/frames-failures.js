// scale-reliability frames 9–11: checkpoints and failures in a 90-minute window, the same run on 100,000 GPUs, and the bill.
import * as G from '@shared/glyphs.js';
import { formatCount } from '@math/core.js';
import { checkpointLossParts, runCost, wallClockDays } from '@math/scale.js';
import { LLAMA, DEEPSEEK, STAND_IN, BIG_GPUS, WINDOW_MIN, FAILURE_AT_MIN } from './numbers.js';
import { llamaRun, SAME_SCALE_GPU_HOURS } from './runs.js';
import { gpuHours, hoursText, minutesText, daysText, pct, int, dollarsExact } from './format.js';
import { MARGIN, seg, ease, layer, note, typed, linked, barWidth } from './stage.js';

const SECONDS_PER_HOUR = 3600;
const MIN_PER_HOUR = 60;
const LANE = Object.freeze({ x: 6, y: 84, w: 556, label: 'run' });
const GUTTER = Math.round(LANE.label.length * 6.6) + 6;
const MIN_SCALE = (LANE.w - GUTTER) / WINDOW_MIN; // px per minute
const MIN_LENGTH = 0.05; // a segment shorter than this many minutes is not drawn

// The window's run, clipped at `visible` minutes: compute between saves, the work since the last save turning `lost` at the
// failure, then the restart gap and compute again.
export function windowRun(visible, intervalMin, restartMin) {
  const saves = [];
  for (let t = intervalMin; t < FAILURE_AT_MIN; t += intervalMin) saves.push(t);
  const lastSave = saves.at(-1);
  const resumeAt = FAILURE_AT_MIN + restartMin;
  const failed = visible >= FAILURE_AT_MIN;
  const points = [0, ...saves];
  const raw = [
    ...points.slice(0, -1).map((from, i) => ({ from, to: points[i + 1], kind: 'compute' })),
    { from: lastSave, to: FAILURE_AT_MIN, kind: failed ? 'lost' : 'compute' },
    { from: resumeAt, to: WINDOW_MIN, kind: 'compute' },
  ];
  const segments = raw.filter((s) => s.from < visible - MIN_LENGTH).map((s) => ({ ...s, to: Math.min(s.to, visible) }));
  const ticks = [{ t: 0, label: ' ' }, ...[...saves, resumeAt + intervalMin].filter((t) => t <= visible && t < WINDOW_MIN).map((t) => ({ t, label: 'save' }))];
  const gaps = visible > FAILURE_AT_MIN + MIN_LENGTH ? [{ from: FAILURE_AT_MIN, to: Math.min(resumeAt, visible), label: 'restart' }] : [];
  return { segments, ticks, gaps, failed };
}

export function drawFrame9(svg, p) {
  const run = llamaRun();
  const intervalMin = run.intervalH * MIN_PER_HOUR;
  const parts = checkpointLossParts({ intervalH: run.intervalH, saveH: STAND_IN.saveS / SECONDS_PER_HOUR, restartH: STAND_IN.restartMin / MIN_PER_HOUR, mtbfH: run.mtbfH });
  const visible = WINDOW_MIN * ease(seg(p, 0.05, 0.85));
  const { segments, ticks, gaps, failed } = windowRun(visible, intervalMin, STAND_IN.restartMin);
  const wrap = linked(svg, 'loss', { x: LANE.x, y: LANE.y, w: LANE.w, h: 70 });
  G.laneTimeline(wrap, { ...LANE, lanes: [{ label: LANE.label, segments }], ticks, gaps, scale: MIN_SCALE, label: '90-minute window of the run' });
  G.selectionMark(svg, { x: LANE.x + GUTTER, y: LANE.y + 16, w: LANE.w - GUTTER, h: 24 });
  if (failed) note(svg, LANE.x + GUTTER + FAILURE_AT_MIN * MIN_SCALE, LANE.y - 8, '✕ failure', { anchor: 'middle' });
  note(svg, MARGIN, 196, `${WINDOW_MIN}-minute window of the run, not all ${LLAMA.windowDays} days`);
  note(svg, MARGIN, 214, `save ${STAND_IN.saveS} s, restart ${STAND_IN.restartMin} min (stand-ins)`, { cls: '' });
  note(svg, MARGIN, 230, 'chosen so Llama 3.1 lands near its reported >90% effective time');
  const out = layer(svg, ease(seg(p, 0.8, 1)));
  note(out, MARGIN, 262, `best interval ${minutesText(intervalMin)}: loss ${pct(parts.total)}`, { cls: '' });
  note(out, MARGIN, 280, `save ${pct(parts.save)} + lost work ${pct(parts.lostWork)} + restarts ${pct(parts.restart)}`, { cls: '' });
}

const BARS = Object.freeze({ headY: [60, 168], barY: [70, 178] });

function runHeading(run, gpus) {
  return `${int(gpus)} GPUs · MTBF ${hoursText(run.mtbfH)} · checkpoint every ${minutesText(run.intervalH * MIN_PER_HOUR)} · ${daysText(run.days)}`;
}

function lossBar(svg, run, { y, lostScale = 1, selected }) {
  const useful = run.usefulGpuHours;
  const lost = (run.gpuHours - useful) * lostScale;
  const w = Math.max(barWidth(useful + lost, SAME_SCALE_GPU_HOURS), 1);
  const wrap = linked(svg, 'loss', { x: MARGIN, y, w, h: 14 });
  G.shareBar(wrap, {
    x: MARGIN, y, w, parts: [{ name: `useful ${gpuHours(useful)}`, value: useful, hue: 1 }, { name: `lost to failures ${gpuHours(lost)}`, value: Math.max(lost, 0), hue: 2, hatched: true }],
    label: 'GPU-hours: useful and lost to failures', tail: 'none',
  });
  if (selected) G.selectionMark(svg, { x: MARGIN, y, w, h: 14 });
}

export function drawFrame10(svg, p) {
  const [small, big] = [llamaRun(), llamaRun({ gpus: BIG_GPUS })];
  note(svg, MARGIN, 30, `same scale: ${gpuHours(SAME_SCALE_GPU_HOURS)} GPU-h = full width`);
  note(svg, MARGIN, BARS.headY[0], runHeading(small, LLAMA.gpus), { cls: '' });
  lossBar(svg, small, { y: BARS.barY[0], selected: true });
  const grow = ease(seg(p, 0.1, 0.6));
  const typedAt = seg(p, 0.55, 0.9);
  note(svg, MARGIN, BARS.headY[1], typed(runHeading(big, BIG_GPUS), typedAt), { cls: '' });
  lossBar(svg, big, { y: BARS.barY[1], lostScale: grow });
  const out = layer(svg, ease(seg(p, 0.6, 0.9)));
  note(out, MARGIN, 280, `loss ${pct(small.loss)} → ${pct(big.loss)} · ${gpuHours(small.gpuHours)} → ${gpuHours(big.gpuHours)} GPU-hours`, { cls: '' });
  note(out, MARGIN, 298, `save ${STAND_IN.saveS} s, restart ${STAND_IN.restartMin} min (stand-ins chosen so Llama 3.1`);
  note(out, MARGIN, 314, 'lands near its reported >90% effective time)');
}

export function drawFrame11(svg, p) {
  const widths = [150, 130, 112];
  const gap = 28;
  const x0 = (580 - (widths.reduce((s, w) => s + w, 0) + 2 * gap)) / 2;
  const xs = [x0, x0 + widths[0] + gap, x0 + widths[0] + widths[1] + 2 * gap];
  const y = 70;
  const cost = runCost({ gpuHours: DEEPSEEK.totalGpuHours, dollarsPerGpuHour: DEEPSEEK.dollarsPerGpuHour });
  const blocks = [`${gpuHours(DEEPSEEK.totalGpuHours)} H800-hours`, `$${DEEPSEEK.dollarsPerGpuHour} per GPU-hour`];
  blocks.forEach((label, i) => {
    const g = layer(svg, ease(seg(p, 0.1 * i, 0.1 * i + 0.3)));
    G.block(g, { x: xs[i], y, w: widths[i], h: 40, label });
    note(g, xs[i] + widths[i] + gap / 2, y + 24, i === 0 ? '×' : '=', { anchor: 'middle' });
  });
  note(svg, xs[0], y - 14, 'DeepSeek-V3, final run', { cls: '' });
  G.block(layer(svg, seg(p, 0.45, 0.55)), { x: xs[2], y, w: widths[2], h: 40, label: typed(dollarsExact(cost), seg(p, 0.5, 0.9)) });
  G.selectionMark(layer(svg, seg(p, 0.5, 0.9)), { x: xs[2], y, w: widths[2], h: 40 });
  const perTrillion = DEEPSEEK.pretrainGpuHours / (DEEPSEEK.tokens / 1e12);
  const days = daysText(wallClockDays({ gpuHours: DEEPSEEK.pretrainGpuHours, gpus: DEEPSEEK.gpus }));
  const out = layer(svg, ease(seg(p, 0.7, 1)));
  note(out, MARGIN, 150, "DeepSeek's own price assumption; excludes research and ablations");
  note(out, MARGIN, 196, `pre-training alone: ${gpuHours(DEEPSEEK.pretrainGpuHours)} H800-hours = ${formatCount(perTrillion)} per trillion tokens`, { cls: '' });
  note(out, MARGIN, 214, `${days} on ${int(DEEPSEEK.gpus)} GPUs`, { cls: '' });
}
