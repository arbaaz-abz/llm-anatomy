// prefill-decode frames 1–6: the weights in memory, one decode step, one 1,000-token prefill, their intensities, the
// roofline, and the 217-token crossover. Decode keeps the top row and prefill the row under it from frame 2 to frame 6.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatDuration, formatInt } from '@math/core.js';
import { FORWARD_FLOPS_PER_PARAM_TOKEN } from '@math/serving.js';
import { MODEL, STATES, RIDGE, CROSSING, BF16_CROSSING, H100_CROSSING, WEIGHTS_READ_S, prefill, intensityOf } from './model.js';
import { H200, H100, PREFILL_TOKENS, CROSS_TOKENS } from './numbers.js';
import { fixed1, formatFlops, tbps, tflops } from './format.js';
import {
  LEFT, LINE, CELL, ROW_D, ROW_P, GPU_BIG, GPU_SMALL, NOTES_Y, seg, lerp, arriving, leaving, pulse, layer, note, lines,
  answerChip, promptChips, stepBarAt, readCell, gpu,
} from './stage.js';

const WEIGHT_SHARE = MODEL.weightBytesPerGpu / H200.hbmBytes;
const BW = tbps(MODEL.bandwidthTBps);
const PEAK = tflops(MODEL.peakTflops);
const D = STATES.decodeAlone;
const P = STATES.prefill;
const tokensText = (n) => `${formatInt(n)} token${n === 1 ? '' : 's'}`;

// ---- frame 1: the weights in memory ----
const FRAME1_LINES = Object.freeze([
  'Llama-3.1-70B, FP8: 1 byte per weight',
  '70B is rounded; the toy uses 70,000,000,000 parameters',
]);
const frame1Numbers = () => [
  `weights ${formatBytes(MODEL.weightBytesPerGpu)} of ${formatBytes(H200.hbmBytes)}, ${formatBytes(H200.hbmBytes - MODEL.weightBytesPerGpu)} free`,
  `${Number(((FORWARD_FLOPS_PER_PARAM_TOKEN * MODEL.activeParamsPerGpu) / 1e9).toPrecision(3))} billion operations per token: ${FORWARD_FLOPS_PER_PARAM_TOKEN} per weight`,
];

function frame1Text(parent, { numbers = true } = {}) {
  note(parent, LEFT, 268, FRAME1_LINES[0], { cls: '' });
  note(parent, LEFT, 268 + LINE, FRAME1_LINES[1]);
  if (numbers) lines(parent, LEFT, 268 + 3 * LINE, frame1Numbers());
}

const answerHeader = (parent, text) => note(parent, LEFT + G.tokenWidth('down') + 10, ROW_D.chipY + 16, text, { cls: '' });

// Frame 1: request A's next token, and the H200 whose memory bar fills with the 70 GB of weights.
export function drawFrame1(svg, p) {
  answerChip(svg, LEFT, ROW_D.chipY);
  answerHeader(layer(svg, arriving(p)), 'request A\'s next token');
  gpu(svg, GPU_BIG, { memFill: WEIGHT_SHARE * seg(p, 0.15, 0.7) });
  frame1Text(layer(svg, seg(p, 0.1, 0.3)), { numbers: false });
  lines(layer(svg, seg(p, 0.7, 0.85)), LEFT, 268 + 3 * LINE, frame1Numbers());
}

// ---- frames 2–6: the decode row and the prefill row ----
const decodeLines = () => [
  `read ${formatBytes(MODEL.weightBytesPerGpu)} ÷ ${BW} = ${formatDuration(WEIGHTS_READ_S)}`,
  `math ${formatFlops(D.flops)} ÷ ${PEAK} = ${formatDuration(D.computeS)}`,
  `step ${formatDuration(D.timeS)}: the step waits on memory`,
];
const FLOOR = 'floor: real engines reach less than this bandwidth, so real steps take longer';

function decodeRow(parent, { header = 'decode: 1 token', bar = 1 } = {}) {
  answerChip(parent, LEFT, ROW_D.chipY);
  answerHeader(parent, header);
  stepBarAt(parent, { y: ROW_D.barY, step: D, reveal: bar, name: 'decode' });
}

function prefillRow(parent, { tokens = PREFILL_TOKENS, step = P, bar = 1 } = {}) {
  const end = promptChips(parent, { y: ROW_P.chipY, tokens: Math.round(tokens) });
  note(parent, end + 12, ROW_P.chipY + 16, `prefill: ${tokensText(Math.round(tokens))}`, { cls: '' });
  stepBarAt(parent, { y: ROW_P.barY, step, reveal: bar, name: 'prefill' });
}

// The GPU at its small place, with the "weights read" beat: the label shows and the GPU steps back for a moment.
function smallGpu(parent, { beat = 0, opacity = 1 }) {
  gpu(parent, GPU_SMALL, { memFill: WEIGHT_SHARE, opacity: opacity * (1 - 0.5 * beat) });
  if (beat > 0) note(layer(parent, beat * opacity), GPU_SMALL.x + GPU_SMALL.w / 2, GPU_SMALL.y + GPU_SMALL.h + 28, 'weights read', { cls: '', anchor: 'middle' });
}

const lerpBox = (a, b, t) => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t) });

// Frame 2: one decode token; the bar shows reading 14.6 ms against 70.7 µs of math.
export function drawFrame2(svg, p) {
  const out = leaving(p);
  if (out > 0) frame1Text(layer(svg, out));
  answerChip(svg, LEFT, ROW_D.chipY);
  answerHeader(layer(svg, out), 'request A\'s next token');
  answerHeader(layer(svg, arriving(p)), 'decode: 1 token');
  const move = seg(p, 0, 0.3);
  if (move < 1) gpu(svg, lerpBox(GPU_BIG, GPU_SMALL, move), { memFill: WEIGHT_SHARE });
  else smallGpu(svg, { beat: pulse(p, 0.3, 0.7) });
  stepBarAt(svg, { y: ROW_D.barY, step: D, reveal: seg(p, 0.3, 0.8), name: 'decode' });
  const g = layer(svg, seg(p, 0.8, 1));
  lines(g, LEFT, NOTES_Y, decodeLines());
  note(g, LEFT, NOTES_Y + 3 * LINE + 6, FLOOR);
}

const prefillLines = () => [
  `math ${formatFlops(P.flops)} ÷ ${PEAK} = ${formatDuration(P.computeS)}`,
  `read ${formatBytes(MODEL.weightBytesPerGpu)} weights + ${formatBytes(P.actBytes)} activations = ${formatBytes(P.bytes)} ÷ ${BW} = ${formatDuration(P.memoryS)}`,
  `step ${formatDuration(P.timeS)}: now the step waits on math`,
];

// Frame 3: a 1,000-token prompt; the same weights, read once, serve every token, and the math bar outgrows the reading.
export function drawFrame3(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    lines(g, LEFT, NOTES_Y, decodeLines());
    note(g, LEFT, NOTES_Y + 3 * LINE + 6, FLOOR);
  }
  decodeRow(svg);
  smallGpu(svg, { beat: pulse(p, 0.15, 0.5) });
  const inP = arriving(p);
  const row = layer(svg, inP);
  const end = promptChips(row, { y: ROW_P.chipY, tokens: PREFILL_TOKENS });
  note(row, end + 12, ROW_P.chipY + 16, `prefill: ${tokensText(PREFILL_TOKENS)}`, { cls: '' });
  stepBarAt(svg, { y: ROW_P.barY, step: P, reveal: seg(p, 0.3, 0.85), name: 'prefill' });
  lines(layer(svg, seg(p, 0.85, 1)), LEFT, NOTES_Y, prefillLines());
}

// ---- frame 4: operations per byte read ----
const INTENSITY = Object.freeze({ decode: intensityOf(D), prefill: intensityOf(P) });
const intensityRows = () => [
  { y: 262, value: INTENSITY.decode, format: fixed1, label: `decode: ${formatInt(INTENSITY.decode)} operations per byte read`, formula: `${formatFlops(D.flops)} ÷ ${formatBytes(D.bytes)} = ${fixed1(INTENSITY.decode)}` },
  { y: 314, value: INTENSITY.prefill, format: formatInt, label: `prefill (${formatInt(PREFILL_TOKENS)}): about ${formatInt(Math.round(INTENSITY.prefill / 100) * 100)} operations per byte read`, formula: `${formatFlops(P.flops)} ÷ ${formatBytes(P.bytes)} = ${fixed1(INTENSITY.prefill)}` },
];

function intensityCells(parent, { decode = true, prefillShown = true } = {}) {
  intensityRows().forEach((r, i) => {
    const shown = i === 0 ? decode : prefillShown;
    readCell(parent, { x: LEFT, y: r.y, value: shown ? r.value : null, format: r.format });
    if (!shown) return;
    note(parent, LEFT + CELL + 10, r.y + 16, r.label, { cls: '' });
    note(parent, LEFT + CELL + 10, r.y + 32, r.formula);
  });
}

// Frame 4: the two ratios type in under the two bars: 2 for decode, about 1,600 for the prefill.
export function drawFrame4(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    smallGpu(svg, { opacity: out });
    lines(layer(svg, out), LEFT, NOTES_Y, prefillLines());
  }
  decodeRow(svg);
  prefillRow(svg);
  intensityCells(layer(svg, arriving(p)), { decode: p >= 0.35, prefillShown: p >= 0.65 });
}

// ---- frame 5: the roofline ----
const ROOF = Object.freeze({ x: 200, y: 4, w: 350, h: 256, xDomain: Object.freeze([1, 1e4]), yDomain: Object.freeze([1, 1e4]) });
const roofPoints = () => [
  { intensity: INTENSITY.decode, label: 'decode', followed: true },
  { intensity: INTENSITY.prefill, label: `prefill ${formatInt(PREFILL_TOKENS)}` },
];

function roof(parent, { points = [], opacity = 1 }) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  G.roofline(g, { ...ROOF, peakTflops: MODEL.peakTflops, bandwidthTBps: MODEL.bandwidthTBps, points, label: 'H200 roofline, FP8' });
  note(g, ROOF.x + ROOF.w, ROOF.y + 11, 'H200 · FP8 (dense, reported)', { cls: '', anchor: 'end' });
}

function roofLeft(parent) {
  answerChip(parent, LEFT, ROW_D.chipY);
  note(parent, LEFT, 50, 'decode: 1 token', { cls: '' });
  promptChips(parent, { y: 70, tokens: 1 });
  note(parent, LEFT + G.tokenWidth('The') + 8, 86, `+ ${formatInt(PREFILL_TOKENS - 1)} others`, { cls: '' });
  note(parent, LEFT, 114, `prefill: ${tokensText(PREFILL_TOKENS)}`, { cls: '' });
}

const roofLines = () => [
  `ridge = ${formatInt(MODEL.peakTflops)} ÷ ${MODEL.bandwidthTBps} = ${fixed1(RIDGE)} operations per byte`,
  `decode ${fixed1(INTENSITY.decode)}: memory-bound`,
  `prefill ${fixed1(INTENSITY.prefill)}: compute-bound`,
];

// Frame 5: on the H200's roofline the ridge sits at 412; decode lands on the slope, the prefill on the flat roof.
export function drawFrame5(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    decodeRow(g);
    prefillRow(g);
    intensityCells(g);
  }
  roofLeft(layer(svg, arriving(p)));
  const dots = seg(p, 0.45, 0.7);
  roof(svg, { opacity: seg(p, 0.1, 0.4) * (1 - dots) });
  roof(svg, { points: roofPoints(), opacity: dots });
  lines(layer(svg, seg(p, 0.75, 0.9)), LEFT, 300, roofLines());
}

// ---- frame 6: the crossover, 217 tokens per weight read ----
const CROSS_Y = 276; // under the prefill bar's arithmetic label
const crossLines = () => [
  `tokens per weight read to be compute-bound: ${fixed1(CROSSING)} on an H200 in FP8 (${fixed1(BF16_CROSSING)} in BF16)`,
  `${H100.label}: ${formatInt(H100_CROSSING)} (A GPU for LLM people); the H200's faster ${BW} memory`,
  'lowers the crossover',
  `prefill ${formatInt(CROSS_TOKENS)}: math ${formatDuration(STATES.cross.computeS)}, reading ${formatDuration(STATES.cross.memoryS)} · decode ${formatDuration(D.timeS)}`,
];

// Frame 6: the prompt shrinks from 1,000 to 217 tokens (each drawn count a real stepTime state) until math and reading meet.
export function drawFrame6(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    roofLeft(g);
    roof(g, { points: roofPoints() });
    lines(g, LEFT, 300, roofLines());
  }
  const t = seg(p, 0.3, 0.8);
  const tokens = Math.round(Math.exp(lerp(Math.log(PREFILL_TOKENS), Math.log(CROSS_TOKENS), t)));
  const g = layer(svg, arriving(p));
  decodeRow(g);
  prefillRow(g, { tokens, step: t >= 1 ? STATES.cross : prefill(tokens) });
  lines(layer(svg, seg(p, 0.8, 1)), LEFT, CROSS_Y, crossLines());
}

// Frame 6's end state, which frame 7 fades out.
export function frame6End(parent) {
  decodeRow(parent);
  prefillRow(parent, { tokens: CROSS_TOKENS, step: STATES.cross });
  lines(parent, LEFT, CROSS_Y, crossLines());
}
