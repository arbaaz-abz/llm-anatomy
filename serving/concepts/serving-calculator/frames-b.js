// serving-calculator frames 5–8: a million-token context, one user, the users a target allows, and the measured gap.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatDuration, formatInt, formatCount, formatRatio } from '@math/core.js';
import { V4, GB300, CONTEXT, TARGET_TOK_S_USER, MEASURED, MEASURED_DATE } from './numbers.js';
import {
  WEIGHTS_PER_GPU, FREE_PER_GPU, kvPerUser, usersFit, decodeStep, tokSUser, tokSGpu, atTarget, WEIGHT_READ_S, IMPLIED_S, FLOOR_TO_IMPLIED,
  MEASURED_OUTPUT_ONLY, TOKENS_PER_OUTPUT,
} from './figures.js';
import { LEFT, COL_X, seg, lerp, ease, arriving, leaving, layer, note, lines, footer } from './stage.js';
import { sceneTop, sceneMemory, frame4Rest } from './frames-a.js';
import { maxUsersPerGpu } from '@math/serving.js';

const STEP_BAR = Object.freeze({ x: 8, y: 212, w: 440, scaleS: 0.04 }); // 40 ms is the full width in frames 5 and 6
const BW = GB300.bandwidthTBps * 1e12;
const LONG = CONTEXT.long;
const SHORT = CONTEXT.short;

// One decode step as the step bar's rows: weights, KV and activations read, and the math (all seconds).
function stepParts(users, tokens) {
  const step = decodeStep(users, tokens);
  return { reading: [{ label: 'weights read', s: WEIGHTS_PER_GPU / BW }, { label: 'KV read', s: step.kvBytes / BW }, { label: 'activations', s: step.actBytes / BW }], mathS: step.computeS };
}

const mixParts = (a, b, t) => ({ reading: a.reading.map((part, i) => ({ ...part, s: lerp(part.s, b.reading[i].s, t) })), mathS: lerp(a.mathS, b.mathS, t) });

function drawStepBar(svg, parts, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  G.stepBar(g, { ...STEP_BAR, reading: parts.reading, mathS: parts.mathS, label: 'one decode step' });
}

// Frame 5's memory at a point t (0 → 1) between the 8K and the 1M context: the users that fit at that per-user KV.
function longMemory(svg, t, opacity = 1) {
  const per = lerp(kvPerUser(SHORT), kvPerUser(LONG), t);
  const users = maxUsersPerGpu(FREE_PER_GPU, per);
  sceneMemory(svg, { weights: WEIGHTS_PER_GPU, kv: users * per, opacity });
  return { per, users };
}

function frame5Column(svg, per, users, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  const high = V4.kvBytesPerToken.high;
  const step = decodeStep(usersFit(LONG), LONG);
  lines(g, COL_X, 22, [`${formatBytes(per)} per user`, `at ${formatInt(LONG)} tokens`], { cls: '' });
  note(g, COL_X, 54, `users that fit: ${formatInt(users)}`, { cls: '' });
  note(g, COL_X, 70, `${formatBytes(high)}/token: ${formatBytes(kvPerUser(LONG, 'high'))} → ${formatInt(usersFit(LONG, 'high'))}`);
  if (users === usersFit(LONG)) lines(g, COL_X, 90, [`step ${formatDuration(step.timeS)} · ${formatCount(1 / step.timeS)} tok/s`, `${formatInt(users / step.timeS)} tok/s per GPU`], { cls: '' });
}

function frame5Notes(svg, o) {
  if (o <= 0) return;
  const high = decodeStep(usersFit(LONG, 'high'), LONG, 'high');
  const g = layer(svg, o);
  lines(g, LEFT, 316, ['sparse attention reads less than the whole cache; this floor assumes it all,', `so ${Math.round(tokSUser(usersFit(LONG), LONG))} tok/s is a lower bound on per-user speed at 1M (${formatBytes(V4.kvBytesPerToken.high)}/token: ${formatDuration(high.timeS)}, ${formatCount(1 / high.timeS)})`], { pitch: 14 });
}

// Frame 5's end state, for frame 6 to fade out.
function frame5End(svg, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  const { per, users } = longMemory(g, 1);
  frame5Column(g, per, users, 1);
  frame5Notes(g, 1);
}

export function drawFrame5(svg, p) {
  frame4Rest(svg, leaving(p));
  const t = ease(seg(p, 0.1, 0.65));
  sceneTop(svg);
  const { per, users } = longMemory(svg, t);
  frame5Column(svg, per, users, seg(p, 0.1, 0.3));
  drawStepBar(svg, stepParts(usersFit(LONG), LONG), seg(p, 0.6, 0.85));
  frame5Notes(svg, seg(p, 0.8, 1));
  footer(svg, { context: '1M tokens' });
}

// Frame 6: back to 8K tokens and one user; the memory bar steps aside and the step is just the weights read plus a sliver.
function frame6Text(svg, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  lines(g, LEFT, 134, [
    `1 user on the followed GPU: ${formatBytes(kvPerUser(SHORT))} of KV beside ${formatBytes(WEIGHTS_PER_GPU)} of weights`,
    `weights read: ${formatBytes(WEIGHTS_PER_GPU)} ÷ ${GB300.bandwidthTBps} TB/s = ${formatDuration(WEIGHT_READ_S)}`,
    `→ ${formatCount(tokSUser(1, SHORT))} tok/s per user, at best (floor)`,
  ], { cls: '' });
}

export function drawFrame6(svg, p) {
  sceneTop(svg);
  frame5End(svg, 1 - seg(p, 0, 0.3));
  const e = ease(seg(p, 0.1, 0.7));
  drawStepBar(svg, mixParts(stepParts(usersFit(LONG), LONG), stepParts(1, SHORT), e), 1);
  frame6Text(svg, seg(p, 0.5, 0.8));
  footer(svg);
}

const PLOT = Object.freeze({ x: 8, y: 10, w: 392, h: 250 });
const SAMPLE_COUNT = 40;
// Users per GPU from 1 to the most that fit at 8K, evenly spaced on a log axis (integers, no repeats).
const SAMPLES = Object.freeze([...new Set(Array.from({ length: SAMPLE_COUNT }, (_, i) => Math.round(usersFit(SHORT) ** (i / (SAMPLE_COUNT - 1)))))]);

function perUserPlot(g, users) {
  const target = atTarget(SHORT).users;
  G.curvePlot(g, {
    ...PLOT, label: 'tokens per second per user against users per GPU',
    xAxis: { log: true, domain: [1, 10000], ticks: [1, 10, 100, 1000, 10000], label: 'users per GPU' },
    yAxis: { domain: [0, 160], ticks: [0, 50, 100, 150], label: 'tokens/s per user (floor)' },
    series: [{ label: '', points: SAMPLES.map((u) => [u, tokSUser(u, SHORT)]) }],
    markers: [{ x: users, y: tokSUser(users, SHORT), label: `${formatInt(users)} users`, followed: true }],
    refY: { value: TARGET_TOK_S_USER, label: `target ${TARGET_TOK_S_USER} tok/s` },
  });
  return target;
}

function frame7Column(svg, users, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  const step = decodeStep(users, SHORT);
  lines(g, 412, 34, [`${formatInt(users)} users`, `step ${formatDuration(step.timeS)}`, `${step.bound}-bound`], { cls: '' });
  lines(g, 412, 82, [`${formatInt(users / step.timeS)} tok/s`, 'per GPU (floor)'], { cls: '' });
}

export function drawFrame7(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    sceneTop(g);
    drawStepBar(g, stepParts(1, SHORT), 1);
    frame6Text(g, 1);
  }
  const target = atTarget(SHORT).users;
  const users = Math.max(1, Math.round(target ** ease(seg(p, 0.2, 0.9))));
  perUserPlot(layer(svg, arriving(p)), users);
  frame7Column(svg, users, seg(p, 0.15, 0.3));
  note(svg, LEFT, 284, `a target of ${TARGET_TOK_S_USER} tok/s leaves each step 1 ÷ ${TARGET_TOK_S_USER} = ${formatDuration(IMPLIED_S)}`, { cls: '' });
  note(svg, LEFT, 300, 'floor = bytes and FLOPs only; users per GPU on a log scale');
  footer(svg);
}

// Frame 8: the same floor as tokens/s per GPU against tokens/s per user, with InferenceX's measured points and the causes.
const LINE_8 = 16;
const CAUSES = Object.freeze(['all-to-all', 'attention', 'expert imbalance', 'prefill GPUs', 'scheduling']);

function frame8Plot(g, drop) {
  const floorAt = atTarget(SHORT).users;
  const markers = [{ x: tokSUser(floorAt, SHORT), y: tokSGpu(floorAt, SHORT), label: `floor ${formatInt(tokSGpu(floorAt, SHORT))}` }];
  if (drop > 0) {
    const m = MEASURED.gb300;
    markers.push({ x: m.tokSUser, y: m.tokSGpu, label: formatInt(m.tokSGpu) }, { x: m.peakTokSUser, y: m.peakTokSGpu, label: formatInt(m.peakTokSGpu) });
  }
  G.curvePlot(g, {
    ...PLOT, h: 232, label: 'tokens per second per GPU against tokens per second per user',
    xAxis: { log: true, domain: [5, 200], ticks: [5, 10, 20, 50, 100, 200], label: 'tokens/s per user' },
    yAxis: { log: true, domain: [100, 100000], ticks: [100, 1000, 10000, 100000], label: 'tokens/s per GPU' },
    series: [{ label: 'floor', points: SAMPLES.map((u) => [tokSUser(u, SHORT), tokSGpu(u, SHORT)]), labelAt: 'end' }], markers,
  });
}

function frame8Text(svg, p) {
  const col = layer(svg, seg(p, 0.5, 0.7));
  const m = MEASURED.gb300;
  lines(col, 412, 34, [`at ${m.tokSUser} tok/s per user:`, `floor ${formatInt(tokSGpu(atTarget(SHORT).users, SHORT))}`, `measured ${formatInt(m.tokSGpu)}`], { cls: '' });
  lines(col, 412, 88, ['dots: measured,', `InferenceX ${MEASURED_DATE}`]);
  CAUSES.forEach((cause, i) => note(layer(svg, seg(p, 0.6 + i * 0.07, 0.7 + i * 0.07)), 412, 130 + i * LINE_8, cause));
  const foot = layer(svg, seg(p, 0.75, 0.95));
  const gap = formatRatio(tokSGpu(atTarget(SHORT).users, SHORT) / m.tokSGpu);
  const gapIn = formatRatio(tokSGpu(atTarget(SHORT).users, SHORT) / MEASURED_OUTPUT_ONLY);
  lines(foot, LEFT, 262, [
    'InferenceX may count input tokens in tok/s/GPU; the source does not say.',
    `If it does, output-only is about ${formatInt(MEASURED_OUTPUT_ONLY)} (${formatInt(m.tokSGpu)} ÷ ${TOKENS_PER_OUTPUT}, 8K in / 1K out) and the gap`,
    `is ${gapIn}, not ${gap}: compare them as a range, not a ratio.`,
    `implied: 1 ÷ ${TARGET_TOK_S_USER} = ${formatDuration(IMPLIED_S)} per token vs the ${formatDuration(WEIGHT_READ_S)} weight-read floor = ${formatRatio(FLOOR_TO_IMPLIED)}`,
  ], { pitch: 14 });
}
// Frame 8's content at progress p; `plotOpacity` fades the plot in while frame 7's plot fades out.
export function frame8Content(svg, p, plotOpacity = 1) {
  frame8Plot(layer(svg, plotOpacity), seg(p, 0.4, 0.6));
  frame8Text(svg, p);
}

export function drawFrame8(svg, p) {
  const out = 1 - seg(p, 0, 0.3);
  if (out > 0) perUserPlot(layer(svg, out), atTarget(SHORT).users);
  frame8Content(svg, p, seg(p, 0.15, 0.45));
  footer(svg);
}
