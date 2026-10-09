// serving-calculator frames 1–4: the model card, the weights needing four GPUs, the 16-GPU replica, and users filling the KV.
// Every frame is a pure function of (svg, p); the end state (p = 1) of one frame is the start of the next.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatCount, formatInt } from '@math/core.js';
import { V4, GB300, GB200_HBM_BYTES, EP_SIZE, CONTEXT } from './numbers.js';
import {
  WEIGHTS, WEIGHTS_PER_GPU, FREE_PER_GPU, MIN_GPUS, MIN_GPUS_GB200, WEIGHTS_BF16, WEIGHTS_FP8, kvPerUser, usersFit,
} from './figures.js';
import {
  LEFT, COL_X, RACK_AT, BIG_GPU, MEM, seg, lerp, ease, arriving, leaving, layer, note, lines, footer, rackScene, bigGpu, memoryBar,
} from './stage.js';

const CARD = Object.freeze({ x: 8, y: 14, w: 316, h: 30, pitch: 38 });
const CARD_TEXT = Object.freeze([
  V4.label,
  `${formatCount(V4.totalParams)} total · ${formatCount(V4.activeParams)} active per token`,
  `${V4.experts} experts, ${V4.expertsPerToken} per token`,
  'FP4 experts + FP8 rest',
  `about ${formatBytes(WEIGHTS)} (reported)`,
]);
const SIZES = Object.freeze({ x: 352, y: 44, w: 216, h: 120 });

// Frame 1's content: the card fields "type in" one after another, then the three sizes of the same weights.
function modelCard(svg, { reveal = () => 1, opacity = 1 } = {}) {
  const g = layer(svg, opacity);
  CARD_TEXT.forEach((text, i) => {
    const o = reveal(i);
    if (o > 0) G.block(layer(g, o), { x: CARD.x, y: CARD.y + i * CARD.pitch, w: CARD.w, h: CARD.h, label: text });
  });
  const sizes = layer(g, reveal(CARD_TEXT.length));
  note(sizes, SIZES.x, 28, 'the same weights, three sizes');
  G.bars(sizes, {
    x: SIZES.x, y: SIZES.y, w: SIZES.w, h: SIZES.h, values: [WEIGHTS_BF16, WEIGHTS_FP8, WEIGHTS], labels: ['BF16', 'FP8', 'shipped'],
    max: WEIGHTS_BF16, format: formatBytes, label: 'weights in three sizes',
  });
  note(sizes, LEFT, 224, 'the 865 GB checkpoint size is reported, not confirmed by the model card');
}

export function drawFrame1(svg, p) {
  modelCard(svg, { reveal: (i) => seg(p, i * 0.12, i * 0.12 + 0.2) });
  footer(svg);
}

const GPU_ROW = Object.freeze({ x: [28, 170, 312, 454], y: 56, w: 96, h: 72 });

// Frame 2's content: GB300s appear one at a time, each full of weights until the last, which holds the final gigabyte.
function gpuRow(svg, { reveal = () => 1, opacity = 1 } = {}) {
  const g = layer(svg, opacity);
  const lastFill = (WEIGHTS - 3 * GB300.hbmBytes) / GB300.hbmBytes;
  GPU_ROW.x.forEach((x, i) => {
    const o = reveal(i);
    if (o <= 0) return;
    const slot = layer(g, o);
    const memFill = i < 3 ? 1 : lastFill;
    G.gpu(slot, { x, y: GPU_ROW.y, w: GPU_ROW.w, h: GPU_ROW.h, memFill, label: `GB300 · ${formatBytes(GB300.hbmBytes)}` });
    note(slot, x + GPU_ROW.w / 2, GPU_ROW.y + GPU_ROW.h + 30, i < 3 ? 'full of weights' : `${formatBytes(WEIGHTS - 3 * GB300.hbmBytes)} of weights`, { anchor: 'middle' });
  });
  return g;
}

function sumLines(svg, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  note(g, LEFT, 28, `weights: ${formatBytes(WEIGHTS)} (reported) · replica = one copy of the model spread over several GPUs`);
  lines(g, LEFT, 196, [
    `3 GPUs × ${formatBytes(GB300.hbmBytes)} = ${formatBytes(3 * GB300.hbmBytes)}: ${formatBytes(WEIGHTS - 3 * GB300.hbmBytes)} short of the weights`,
    `${formatBytes(WEIGHTS)} ÷ ${formatBytes(GB300.hbmBytes)}, rounded up = ${MIN_GPUS} GPUs for weights alone`,
    `GB200 NVL72: ${formatBytes(GB200_HBM_BYTES)} per GPU (the rack total over 72 GPUs) → ${MIN_GPUS_GB200} GPUs`,
  ], { cls: '' });
}

export function drawFrame2(svg, p) {
  const out = leaving(p);
  if (out > 0) modelCard(svg, { opacity: out });
  const slots = [0.0, 0.2, 0.4, 0.65];
  gpuRow(svg, { reveal: (i) => seg(p, slots[i], slots[i] + 0.2) * (i === 0 ? arriving(p) : 1) });
  sumLines(svg, seg(p, 0.7, 0.95));
  footer(svg);
}

// Frames 3–6 share this scene: the 16-GPU replica, the followed GPU drawn large, its memory bar and the text column beside it.
export function sceneTop(svg, opacity = 1) {
  const top = layer(svg, opacity);
  rackScene(top);
  bigGpu(top);
}

export function sceneMemory(svg, { weights, kv, opacity = 1 }) {
  const g = layer(svg, opacity);
  note(g, MEM.x, MEM.y - 8, `followed GPU memory: ${formatBytes(GB300.hbmBytes)} nominal`);
  memoryBar(g, { weights, kv, total: GB300.hbmBytes });
}

export function scene(svg, { weights, kv, rackOpacity = 1 }) {
  sceneTop(svg, rackOpacity);
  sceneMemory(svg, { weights, kv });
}

function frame3Text(svg, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  lines(g, COL_X, 28, [`${formatBytes(WEIGHTS)} ÷ ${EP_SIZE} = ${formatBytes(WEIGHTS_PER_GPU)}`, `${formatBytes(GB300.hbmBytes)} − ${formatBytes(WEIGHTS_PER_GPU)}`], { cls: '' });
  note(g, COL_X, 60, `= ${formatBytes(FREE_PER_GPU)} free`);
  note(g, LEFT, 250, 'upper bound: activations and runtime buffers are not subtracted');
}

export function drawFrame3(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    gpuRow(svg, { opacity: out });
    sumLines(svg, out);
  }
  const shrink = ease(seg(p, 0.2, 0.85));
  scene(svg, { weights: lerp(WEIGHTS / MIN_GPUS, WEIGHTS_PER_GPU, shrink), kv: 0, rackOpacity: arriving(p) });
  frame3Text(svg, seg(p, 0.75, 0.95));
  footer(svg);
}

// Frame 4: user chips A–D enter the followed GPU and its KV part grows by one user's cache at a time, up to the users that fit.
const USER_CHIPS = Object.freeze(['A', 'B', 'C', 'D']);

function userChips(svg, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  USER_CHIPS.forEach((owner, i) => G.token(g, { x: COL_X + i * 34, y: 66, text: owner, owner }));
  note(g, COL_X, 106, 'A to D: many more users');
}

export function frame4Lines(svg, users, o = 1) {
  if (o <= 0) return;
  const g = layer(svg, o);
  const per = kvPerUser(CONTEXT.short);
  lines(g, COL_X, 22, [`${formatInt(users)} users × ${formatBytes(per)}`, `= ${formatBytes(users * per)}`], { cls: '' });
  note(g, COL_X, 54, `users that fit: ${formatInt(usersFit(CONTEXT.short))}`, { cls: '' });
}

function frame4Notes(svg, o) {
  if (o <= 0) return;
  const g = layer(svg, o);
  const [low, high] = [kvPerUser(CONTEXT.short, 'low'), kvPerUser(CONTEXT.short, 'high')];
  lines(g, LEFT, 296, [
    `KV per token is a range: ${formatBytes(V4.kvBytesPerToken.low)} (low) – ${formatBytes(V4.kvBytesPerToken.high)} (high),`,
    'formula-derived from the config; the layer mix is uncertain',
    `${formatBytes(V4.kvBytesPerToken.high)}: ${formatBytes(high)} per user → ${formatInt(usersFit(CONTEXT.short, 'high'))} users fit (${formatBytes(V4.kvBytesPerToken.low)}: ${formatBytes(low)} → ${formatInt(usersFit(CONTEXT.short))})`,
  ], { pitch: 15 });
}

export function drawFrame4(svg, p) {
  const grow = ease(seg(p, 0.1, 0.9));
  const fit = usersFit(CONTEXT.short);
  const users = Math.floor(fit * grow);
  scene(svg, { weights: WEIGHTS_PER_GPU, kv: users * kvPerUser(CONTEXT.short) });
  frame3Text(svg, leaving(p));
  frame4Lines(svg, users, seg(p, 0.15, 0.3));
  userChips(svg, seg(p, 0, 0.2));
  frame4Notes(svg, seg(p, 0.8, 1));
  footer(svg);
}

// Frame 4's end state without the scene, for frame 5 to fade out.
export function frame4Rest(svg, o) {
  frame4Lines(svg, usersFit(CONTEXT.short), o);
  userChips(svg, o);
  frame4Notes(svg, o);
}
