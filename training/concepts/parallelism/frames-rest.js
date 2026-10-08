// Frames 9–11: context parallelism, expert parallelism and the product of the degrees (storyboard §5).
import * as G from '@shared/glyphs.js';
import { gpuCount } from '@math/parallel.js';
import { formatBytes, formatCount } from '@math/core.js';
import {
  COPIES_PER_GPU, CP_SENT_BYTES, CP_TOKENS_SENT, CROSSING, DEEPSEEK_V3, EP_COPY_BYTES, EP_DISPATCH_BYTES, EXPERTS_PER_GPU, EP_GPUS,
  KV_BYTES_PER_TOKEN, KV_NUMBERS_PER_TOKEN, LLAMA, ROUTES, SAT, TOKENS, WEIGHT_BYTES_PER_PARAM, gpuOfExpert,
} from './numbers.js';
import { bytesExact, formatStateGB, int, stateView } from './format.js';
import { BLANK, chipRow, counter, ease, gpuSwatch, label, lerp, linked, seg, select, zeros } from './stage.js';

// ---- frame 9: context parallelism ----
const CAUSAL = TOKENS.map((_, i) => TOKENS.map((__, j) => j <= i));
const CELLS_PER_GPU = [[0, 1], [2, 3]].map((rows) => rows.reduce((n, i) => n + CAUSAL[i].filter(Boolean).length, 0)); // 3 and 7
const CP = { gpu1: { x: 40, y: 96 }, gpu2: { x: 444, y: 96 }, heat: { x: 190, y: 214 }, cell: 30 };

export function drawFrame9(svg, p) {
  [CP.gpu1, CP.gpu2].forEach((pos, i) => G.gpu(svg, { ...pos, label: `GPU ${i + 1}`, showMem: false }));
  chipRow(svg, 50, 58, { tokens: ['The', 'cat', null, null], mark: false });
  chipRow(svg, 414, 58, { tokens: [null, null, 'sat', 'down'], mark: true });
  G.flow(svg, { from: [152, 128], to: [438, 128], carry: 'kv', progress: ease(seg(p, 0.1, 0.8)) });
  const x = lerp(170, 360, ease(seg(p, 0.1, 0.8)));
  G.kvStack(svg, { x, y: 98, count: CP_TOKENS_SENT, tile: 14, label: 'The, cat' });
  label(svg, 295, 150, 'forward pass', { anchor: 'middle' });
  label(svg, 295, 168, 'backward: their gradients travel back', { anchor: 'middle', cls: 'g-label', opacity: seg(p, 0.7, 1) });
  const rowLabels = TOKENS;
  G.heatmap(svg, {
    ...CP.heat, values: zeros(4, 4), cell: CP.cell, mask: CAUSAL, rowLabels, colLabels: TOKENS, label: 'scores', format: BLANK,
    shards: [{ rows: [1, 2], gpu: 1 }, { rows: [3, 4], gpu: 2 }],
  });
  select(svg, CP.heat.x, CP.heat.y + SAT * CP.cell, 4 * CP.cell, CP.cell);
  label(svg, 24, 346, 'rows: queries · columns: keys', { cls: 'g-label' });
  gpuSwatch(svg, 24, 14, 1);
  gpuSwatch(svg, 110, 14, 2);
  const lines = [
    `K+V per token: ${KV_NUMBERS_PER_TOKEN} × 2 B = ${KV_BYTES_PER_TOKEN} B`,
    `${CP_TOKENS_SENT} tokens: ${bytesExact(CP_SENT_BYTES)} per layer, fwd`,
    `backward: same ${bytesExact(CP_SENT_BYTES)} of grads`,
    `score cells: GPU 1 ${CELLS_PER_GPU[0]}, GPU 2 ${CELLS_PER_GPU[1]}`,
    `the later half does ${CELLS_PER_GPU[1]} of ${CELLS_PER_GPU[0] + CELLS_PER_GPU[1]};`,
    'real schemes interleave chunks',
  ];
  lines.forEach((t, i) => label(svg, 372, 214 + i * 16, t, { opacity: seg(p, 0.1 + i * 0.1, 0.3 + i * 0.1) }));
  counter(svg, `sent per GPU: ${bytesExact(Math.round(CP_SENT_BYTES * seg(p, 0.1, 0.8)))}`);
}

// ---- frame 10: expert parallelism ----
const COL = (g) => 16 + 142 * g;
const EXPERT = { y: 112, w: 58, h: 28 };
const expertX = (e) => COL(gpuOfExpert(e)) + 4 + (e % EXPERTS_PER_GPU) * 64;
const chipX = (t) => COL(t) + 65 - G.tokenWidth(TOKENS[t]) / 2;
const BACK_OFFSET = 5; // results travel back beside the outbound copies
const CHIP_Y = 24;

export function drawFrame10(svg, p) {
  for (let g = 0; g < EP_GPUS; g += 1) {
    G.token(svg, { x: chipX(g), y: CHIP_Y, text: TOKENS[g], index: g + 1 });
    G.gpu(svg, { x: COL(g) + 17, y: 152, label: `GPU ${g + 1}`, showMem: false });
    label(svg, COL(g) + 65, 252, `copies: ${COPIES_PER_GPU[g]}`, { anchor: 'middle', opacity: seg(p, 0.3, 0.5) });
  }
  select(svg, chipX(SAT), CHIP_Y, G.tokenWidth(TOKENS[SAT]), 24);
  const lit = seg(p, 0.4, 0.55) > 0 && p < 0.95;
  ROUTES.flat().forEach((e) => G.block(svg, { x: expertX(e), y: EXPERT.y, w: EXPERT.w, h: EXPERT.h, label: `E${e + 1}`, state: lit ? 'active' : 'idle' }));
  for (let e = 0; e < EP_GPUS * EXPERTS_PER_GPU; e += 1) {
    if (!ROUTES.flat().includes(e)) G.block(svg, { x: expertX(e), y: EXPERT.y, w: EXPERT.w, h: EXPERT.h, label: `E${e + 1}`, state: 'idle' });
  }
  const out = seg(p, 0, 0.4);
  const back = seg(p, 0.55, 0.95);
  ROUTES.forEach((experts, k) => experts.forEach((e) => {
    const target = [expertX(e) + EXPERT.w / 2, EXPERT.y - 2];
    const source = [chipX(k) + G.tokenWidth(TOKENS[k]) / 2, CHIP_Y + 26];
    if (back <= 0) G.flow(svg, { from: source, to: target, carry: 'activation', progress: ease(out) });
    else G.flow(svg, { from: [target[0] + BACK_OFFSET, target[1]], to: [source[0] + BACK_OFFSET, source[1]], carry: 'activation', progress: ease(back) });
  }));
  const crossing = CROSSING.map(({ token, expert }) => `${TOKENS[token]} → E${expert + 1}`).join(', ');
  label(svg, 12, 290, `${ROUTES.flat().length} copies, ${CROSSING.length} cross GPUs (${crossing})`);
  label(svg, 12, 308, `${CROSSING.length} × ${EP_COPY_BYTES} B = ${bytesExact(EP_DISPATCH_BYTES)} dispatch + ${bytesExact(EP_DISPATCH_BYTES)} combine`);
  label(svg, 12, 326, 'routes from moe\'s router toy', { cls: 'g-label' });
  counter(svg, `sent, all GPUs: ${bytesExact(Math.round(EP_DISPATCH_BYTES * (out + back) ))}`);
}

// ---- frame 11: the product of the degrees ----
const FACTORS = Object.freeze([['tensor', 'tp'], ['context', 'cp'], ['pipeline', 'pp'], ['data', 'dp']]);
const FACTOR_X = Object.freeze([10, 134, 258, 382]);
const FACTOR = { y: 52, w: 108, h: 40, gap: 16 };

export function drawFrame11(svg, p) {
  const shown = Math.floor(seg(p, 0, 0.6) * FACTORS.length + 1e-9);
  FACTORS.forEach(([name, key], i) => {
    if (i >= shown) return;
    G.block(svg, { x: FACTOR_X[i], y: FACTOR.y, w: FACTOR.w, h: FACTOR.h, label: `${name} ${LLAMA[key]}` });
    if (i < FACTORS.length - 1) label(svg, FACTOR_X[i] + FACTOR.w + FACTOR.gap / 2, FACTOR.y + FACTOR.h / 2, '×', { anchor: 'middle' });
  });
  const total = gpuCount(LLAMA);
  const done = seg(p, 0.6, 0.8);
  const totalText = `= ${int(total)}`;
  const totalBox = linked(svg, 'comm', { x: 500, y: 62, w: totalText.length * 6.6, h: 20 });
  label(totalBox, 500, 72, totalText, { opacity: done });
  select(svg, 498, 60, totalText.length * 6.6 + 4, 24, done);
  label(svg, 12, 120, `Llama 3.1 405B: tensor ${LLAMA.tp} × context ${LLAMA.cp} × pipeline ${LLAMA.pp} × data ${LLAMA.dp} = ${int(total)} H100s`, { opacity: done });
  label(svg, 12, 148, `DeepSeek-V3: pipeline ${DEEPSEEK_V3.pp} × expert ${DEEPSEEK_V3.ep}, ZeRO-1 data parallel, no tensor parallelism`, { opacity: seg(p, 0.8, 1) });
  const state = stateView({ ...LLAMA, zero: LLAMA.zeroStage });
  const rest = seg(p, 0.8, 1);
  label(svg, 12, 196, `state per GPU: ${formatStateGB(state.parts.total)} (optimizer states and gradients sharded, as Llama 3 did)`, { opacity: rest });
  label(svg, 12, 230, `${formatCount(LLAMA.params)} parameters × ${WEIGHT_BYTES_PER_PARAM} B ÷ (${LLAMA.tp} × ${LLAMA.pp}) = ${formatBytes((LLAMA.params * WEIGHT_BYTES_PER_PARAM) / (LLAMA.tp * LLAMA.pp))} before sharding`, { cls: 'g-label', opacity: rest });
  const parts = state.parts;
  label(svg, 12, 248, `→ ${formatStateGB(parts.total)} sharded over ${LLAMA.dp}: weights ${num2(parts.weights)} + gradients ${num2(parts.grads)} + optimizer ${num2(parts.optimizer)}`, { cls: 'g-label', opacity: rest });
  counter(svg, `GPUs: ${int(total)}`);
}

const num2 = (bytes) => (bytes / 1e9).toFixed(2);
