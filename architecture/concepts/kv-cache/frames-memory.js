// Frames 6–9: what a cache costs in bytes, from the toy model to GPT-3, one GPU, and the 2026 models (storyboard §5).
// Each frame is a pure function of its progress p; frame n starts by drawing frame n − 1's last picture and fading it out.
import * as G from '@shared/glyphs.js';
import { formatBytes } from '@math/core.js';
import { kvBytesPerToken } from '@math/memory.js';
import { int, ratioText } from './format.js';
import {
  GPT3, TOY_SHAPE, TOY_CONTEXT, HEAD_B_ON, K_ON, V_ON, BYTES_PER_TOKEN, V4_PRO_BYTES, CACHE, AT_STOP, LLAMA_TOKENS, SLIDER_STOP, MAX_ABS,
  gpuShare, GPU_FILL,
} from './numbers.js';
import { seg, lerp, ease, arriving, layer, label, textBlock, linkedText, cacheStackAt, handOff, cellText } from './stage.js';
import { drawFrame5 } from './frames-cache.js';

const STACK = Object.freeze({ x: 20, y: 20, w: 200 });
const BLOCK_PITCH = 94; // blockStack: a 82 px block and a 12 px gap
const KV_X = 236;
const TEXT_X = 236;
const attentionCenter = (top, block) => top + block * BLOCK_PITCH + 29;
const TILE = 5; // the newest tile: token 5, "on"
const TILES_END = 5 * 16 - 2 + 4; // just past the last of five 14 px tiles, 2 px apart

// ---- frame 6: the toy model, 2 blocks × 2 heads ----
const TOY_NUMBERS = kvBytesPerToken({ ...TOY_SHAPE, bytesPerElem: 1 });
const TOY_BYTES = BYTES_PER_TOKEN.toy;
const HEAD_ROWS = Object.freeze([K_ON, V_ON, HEAD_B_ON.k, HEAD_B_ON.v]);
const HEAD_LABELS = Object.freeze(['K h1', 'V h1', 'K h2', 'V h2']);
const EXPAND = Object.freeze({ x: 392, y: 58 });

function toyScene(parent, p, opacity = 1) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  G.blockStack(g, { x: STACK.x, y: STACK.y, w: STACK.w, count: TOY_SHAPE.layers, shown: 2 });
  const lit = [p >= 0.2, p >= 0.5];
  [0, 1].forEach((block) => cacheStackAt(g, { x: KV_X, y: attentionCenter(STACK.y, block) - 15, count: TOY_CONTEXT, highlight: lit[block] ? [TILE - 1] : [], title: block === 0 ? 'KV cache' : undefined }));
  const grow = ease(seg(p, 0.25, 0.45));
  if (grow > 0) {
    const sub = layer(g, grow);
    G.matrix(sub, { x: lerp(EXPAND.x - 40, EXPAND.x, grow), y: EXPAND.y, values: HEAD_ROWS, cell: 20, maxAbs: MAX_ABS, label: 'block 1, token 5', rowLabels: HEAD_LABELS, format: cellText });
    G.flow(sub, { from: [KV_X + TILES_END, attentionCenter(STACK.y, 0)], to: [KV_X + TILES_END + 22, attentionCenter(STACK.y, 0)], carry: 'kv', progress: grow });
  }
  label(g, TEXT_X, 190, 'K and V × 2 heads × 4 numbers × 2 blocks', { cls: '', opacity: seg(p, 0.45, 0.55) });
  label(g, TEXT_X, 212, `= ${int(TOY_NUMBERS)} numbers per token`, { cls: '', opacity: seg(p, 0.6, 0.7) });
  linkedText(g, 'bytes', { x: TEXT_X, y: 234, str: `× ${TOY_SHAPE.bytesPerElem} bytes = ${int(TOY_BYTES)} bytes per token`, cls: '', opacity: seg(p, 0.75, 0.85) });
  linkedText(g, 'bytes', { x: TEXT_X, y: 256, str: `${TOY_CONTEXT} tokens: ${int(CACHE.toyFive)} B`, cls: '', opacity: seg(p, 0.88, 0.98) });
}
export function drawFrame6(svg, p) {
  handOff(svg, p, (g) => drawFrame5(g, 1));
  toyScene(svg, p, arriving(p));
}

// ---- frame 7: the same picture for GPT-3 ----
const GPT3_STACK = Object.freeze({ x: 20, y: 14 });
const GPT3_PER_TOKEN = BYTES_PER_TOKEN.gpt3;

function gptScene(parent, p, opacity = 1) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  G.blockStack(g, { x: GPT3_STACK.x, y: GPT3_STACK.y, w: STACK.w, count: GPT3.layers, shown: 2 });
  cacheStackAt(g, { x: KV_X, y: attentionCenter(GPT3_STACK.y, 0) - 15, count: TOY_CONTEXT, highlight: [] });
  label(g, KV_X + 5 * 16 + 6, attentionCenter(GPT3_STACK.y, 0), `⋯ × ${int(GPT3.context)} positions`, { opacity: seg(p, 0.35, 0.5) });
  label(g, TEXT_X, 110, `${GPT3.layers} blocks · ${GPT3.kvHeads} heads · ${GPT3.headDim} numbers per head`, { cls: '', opacity: seg(p, 0.3, 0.45) });
  label(g, TEXT_X, 140, `2 × ${GPT3.layers} × ${GPT3.kvHeads} × ${GPT3.headDim} × ${GPT3.bytesPerElem} B`, { cls: '', opacity: seg(p, 0.4, 0.5) });
  const counted = Math.round(GPT3_PER_TOKEN * ease(seg(p, 0.5, 0.7)));
  const perToken = counted >= GPT3_PER_TOKEN ? `= ${int(GPT3_PER_TOKEN)} B ≈ ${formatBytes(GPT3_PER_TOKEN)} per token` : `= ${int(counted)} B`;
  linkedText(g, 'bytes', { x: TEXT_X, y: 160, str: perToken, cls: '', opacity: seg(p, 0.5, 0.55) });
  label(g, TEXT_X, 200, `× ${int(GPT3.context)} tokens`, { cls: '', opacity: seg(p, 0.7, 0.8) });
  linkedText(g, 'bytes', { x: TEXT_X, y: 220, str: `= ${int(CACHE.gpt3Context)} B = ${formatBytes(CACHE.gpt3Context)}`, cls: '', opacity: seg(p, 0.8, 0.95) });
}

export function drawFrame7(svg, p) {
  const swap = ease(seg(p, 0, 0.35));
  toyScene(svg, 1, 1 - swap);
  gptScene(svg, p, swap);
}

// ---- frame 8: one conversation fills an H100 to 54 %; a second does not fit ----
const GPU_BOX = Object.freeze({ w: 96, h: 72, y: 56 });
const GPU_X = Object.freeze({ one: 60, two: 330 });
const LLAMA_SHARES = Object.freeze({ one: gpuShare(CACHE.llamaOne), two: gpuShare(CACHE.llamaTwo) });
const LLAMA_FILLS = Object.freeze({ one: GPU_FILL(CACHE.llamaOne), two: GPU_FILL(CACHE.llamaTwo) });

function gpuScene(parent, p, opacity = 1) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  label(g, 20, 20, `Llama-3.1-70B: ${int(BYTES_PER_TOKEN.llama)} B per token × ${int(LLAMA_TOKENS)} tokens per conversation`, { cls: '' });
  const rise = (from, to) => ease(seg(p, from, to));
  G.gpu(g, { x: GPU_X.one, y: GPU_BOX.y, ...GPU_BOX, memFill: LLAMA_FILLS.one * rise(0.15, 0.5), label: 'H100, 80 GB' });
  G.gpu(g, { x: GPU_X.two, y: GPU_BOX.y, ...GPU_BOX, memFill: Math.min(LLAMA_FILLS.two, 1) * rise(0.5, 0.85), label: 'H100, 80 GB' });
  const first = seg(p, 0.45, 0.55);
  label(g, GPU_X.one, 168, '1 conversation', { cls: '', opacity: first });
  linkedText(g, 'bytes', { x: GPU_X.one, y: 186, str: `${formatBytes(CACHE.llamaOne)} = ${LLAMA_SHARES.one}% of 80 GB`, cls: '', opacity: first });
  label(g, GPU_X.one, 202, `${int(CACHE.llamaOne)} B`, { opacity: first });
  const second = seg(p, 0.8, 0.9);
  label(g, GPU_X.two, 168, '2 conversations', { cls: '', opacity: second });
  linkedText(g, 'bytes', { x: GPU_X.two, y: 186, str: `${formatBytes(CACHE.llamaTwo)} = ${LLAMA_SHARES.two}% of 80 GB`, cls: '', opacity: second });
  label(g, GPU_X.two, 202, `${int(CACHE.llamaTwo)} B`, { opacity: second });
  const spill = seg(p, 0.88, 0.98);
  label(g, GPU_X.two, 232, 'does not fit', { cls: '', opacity: spill });
  textBlock(g, GPU_X.two, 250, ['(before the weights, which', 'prefill-decode adds)'], { opacity: spill });
}

export function drawFrame8(svg, p) {
  handOff(svg, p, (g) => drawFrame7(g, 1));
  gpuScene(svg, p, arriving(p));
}

// ---- frame 9: bytes per token, 2020 to 2026, and what one 1,048,576-token conversation does to a GPU ----
const ROWS = Object.freeze([
  { name: 'GPT-3', bytes: BYTES_PER_TOKEN.gpt3, gpu: null },
  { name: 'Llama-3.1-70B', bytes: BYTES_PER_TOKEN.llama, gpu: AT_STOP.llama },
  { name: 'DeepSeek-V3', bytes: BYTES_PER_TOKEN.v3, gpu: AT_STOP.v3 },
  { name: 'DeepSeek-V4-Pro', bytes: V4_PRO_BYTES, gpu: [AT_STOP.v4Low, AT_STOP.v4High] },
]);
const ROW_Y = (i) => 36 + i * 58;
const COLUMNS = Object.freeze({ name: 20, bytes: 170, gpu: 330, share: 402 });
const GPU_SMALL = Object.freeze({ w: 64, h: 48 });
const RATIO = Object.freeze([BYTES_PER_TOKEN.gpt3 / V4_PRO_BYTES[1], BYTES_PER_TOKEN.gpt3 / V4_PRO_BYTES[0]]);

function rowBytes(g, row, y, opacity) {
  const range = Array.isArray(row.bytes);
  linkedText(g, 'bytes', { x: COLUMNS.bytes, y: y + 12, str: range ? `${int(row.bytes[0])}–${int(row.bytes[1])} B` : `${int(row.bytes)} B`, cls: '', opacity });
  label(g, COLUMNS.bytes, y + 28, range ? 'reported estimate' : `= ${formatBytes(row.bytes)}`, { opacity });
}

function rowGpu(g, row, y, p, window) {
  if (!row.gpu) {
    label(g, COLUMNS.gpu, y + 24, `context ${int(GPT3.context)} tokens`);
    return;
  }
  const ends = [row.gpu].flat();
  const grow = ease(seg(p, ...window));
  G.gpu(g, { x: COLUMNS.gpu, y, ...GPU_SMALL, memFill: GPU_FILL(ends[0]) * grow });
  const shown = seg(p, window[1] - 0.1, window[1]);
  const range = ends.length > 1;
  const sizes = range ? ends.map(formatBytes).join('–') : formatBytes(ends[0]);
  linkedText(g, 'bytes', { x: COLUMNS.share, y: y + 12, str: sizes, cls: '', opacity: shown });
  const shares = ends.map(gpuShare);
  const second = range ? `${shares[0]}–${shares[1]}% of 80 GB` : (shares[0] > 100 ? `${(shares[0] / 100).toFixed(1)} H100s` : `${shares[0]}% of 80 GB`);
  label(g, COLUMNS.share, y + 28, second, { opacity: shown });
}

function tableScene(parent, p, opacity = 1) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  label(g, COLUMNS.bytes, 14, 'per token');
  label(g, COLUMNS.gpu, 14, `one conversation, ${int(SLIDER_STOP)} tokens`);
  label(g, COLUMNS.gpu, 26, 'a what-if at a slider stop');
  ROWS.forEach((row, i) => {
    const y = ROW_Y(i);
    const shown = seg(p, 0.15 * i, 0.15 * i + 0.15);
    label(g, COLUMNS.name, y + 12, row.name, { cls: '', opacity: shown });
    rowBytes(g, row, y, shown);
    if (shown > 0) rowGpu(layer(g, shown), row, y, p, [0.3 + 0.15 * (i - 1), 0.55 + 0.15 * (i - 1)]);
  });
  label(g, 20, 280, `GPT-3 ÷ DeepSeek-V4-Pro: ${ratioText(RATIO[0])} to ${ratioText(RATIO[1])} (an estimate)`, { cls: '', opacity: seg(p, 0.7, 0.85) });
  label(g, 20, 304, `${int(SLIDER_STOP)} (2²⁰) is a slider stop;`, { opacity: seg(p, 0.7, 0.85) });
  label(g, 20, 320, 'each model\'s own context comes from its data entry.', { opacity: seg(p, 0.7, 0.85) });
}

export function drawFrame9(svg, p) {
  handOff(svg, p, (g) => drawFrame8(g, 1));
  tableScene(svg, p, arriving(p));
}
