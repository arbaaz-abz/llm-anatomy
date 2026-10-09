// disaggregation frames 6–8: 384 experts on one GPU starve; wide expert parallelism pools the tokens; DeepSeek's production layout.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatCount, formatInt, mulberry32 } from '@math/core.js';
import { tokensPerExpert, freeHbmPerGpu } from '@math/serving.js';
import { matmulCost, arithmeticIntensity, ridgePoint, tokensToComputeBound, bytesPerElement } from '@math/roofline.js';
import { EXPERT_COUNT, EP_SHOWN, USERS_SHOWN, ROUTING_SEED, V4_PRO, GB300, DEEPSEEK } from './numbers.js';
import { LEFT, CELL, LINE, seg, lerp, layer, note, lines, connector, scene } from './stage.js';
import { drawFrame5 } from './frames-transfer.js';

const { total: EXPERTS, active: PER_TOKEN, columns: COLS } = EXPERT_COUNT;

// Frame 6's stand-in routing: a seeded shuffle of the experts, dealt six per token, so 64 users cover all 384 once.
function routing() {
  const rng = mulberry32(ROUTING_SEED);
  const order = Array.from({ length: EXPERTS }, (_, i) => i);
  for (let i = EXPERTS - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return Array.from({ length: USERS_SHOWN }, (_, u) => order.slice(u * PER_TOKEN, (u + 1) * PER_TOKEN));
}
export const ROUTES = Object.freeze(routing().map((r) => Object.freeze(r)));
export const EXPERT0_TOKEN = ROUTES.findIndex((r) => r.includes(0));

const IDENT = (n) => n;
const EXPERT_MATRIX = Object.freeze({ x: 150, y: 60, cell: 12 });
const expertIntensity = (tokens) => arithmeticIntensity(matmulCost({ m: tokens, k: V4_PRO.dModel, n: V4_PRO.expertHidden, bytesPerElem: bytesPerElement('nvfp4') }));
const RIDGE = ridgePoint({ peakTflops: GB300.peakTflops, bandwidthTBps: GB300.bandwidthTBps });
const NEEDED = Math.ceil(tokensToComputeBound({ peakTflops: GB300.peakTflops, bandwidthTBps: GB300.bandwidthTBps, bytesPerElem: bytesPerElement('nvfp4'), k: V4_PRO.dModel, n: V4_PRO.expertHidden }));
const ONE_GPU = tokensPerExpert({ usersPerGpu: USERS_SHOWN, epSize: 1, expertsActive: PER_TOKEN, expertsTotal: EXPERTS });
const WIDE = tokensPerExpert({ usersPerGpu: USERS_SHOWN, epSize: EP_SHOWN, expertsActive: PER_TOKEN, expertsTotal: EXPERTS });

// Cell values: 0 = no token yet, 1 = one token received, 2 = the cells the token being routed right now flashes.
function routedValues(routed, flashing) {
  const values = Array.from({ length: EXPERT_COUNT.rows }, () => Array(COLS).fill(0));
  ROUTES.slice(0, routed).forEach((route) => route.forEach((e) => { values[Math.floor(e / COLS)][e % COLS] = 1; }));
  if (flashing) ROUTES[routed - 1].forEach((e) => { values[Math.floor(e / COLS)][e % COLS] = 2; });
  return values;
}

function counter(svg, { x, y, value, label }) {
  note(svg, x, y - 8, label, { cls: '' });
  G.cell(svg, { x, y, size: CELL, v: value, maxAbs: 1e15, format: IDENT });
  G.selectionMark(svg, { x, y, w: CELL, h: CELL });
}

export function drawFrame6(svg, p) {
  scene(svg, p, (g) => drawFrame5(g, 1), (g) => {
    const progress = seg(p, 0.05, 0.9);
    const routed = Math.floor(progress * USERS_SHOWN);
    const flashing = routed > 0 && progress < 1;
    const m = EXPERT_MATRIX;
    G.gpu(g, { x: LEFT, y: 150, w: 96, h: 72, label: 'one GPU', showMem: false });
    note(g, LEFT, 254, `${USERS_SHOWN} users`);
    note(g, LEFT, 270, `${PER_TOKEN} experts each`);
    G.matrix(g, { x: m.x, y: m.y, values: routedValues(routed, flashing), cell: m.cell, maxAbs: 2, format: () => '', label: 'experts' });
    G.selectionMark(g, { x: m.x, y: m.y, w: m.cell, h: m.cell });
    counter(g, { x: 80, y: 54, value: routed > EXPERT0_TOKEN ? 1 : 0, label: 'expert 0' });
    connector(g, [80 + CELL, 54 + CELL / 2], [m.x, m.y + m.cell / 2]);
    note(g, 80 + CELL, 108, 'tokens this step', { anchor: 'end' });
    note(g, 452, 80, 'cell = one expert');
    note(g, 452, 96, 'brighter: token');
    note(g, 452, 112, 'just routed');
    lines(g, LEFT, 290, [
      `${USERS_SHOWN} × ${PER_TOKEN} ÷ ${EXPERTS} = ${formatCount(ONE_GPU)} token per expert`,
      `expert intensity ${formatCount(expertIntensity(ONE_GPU))} FLOP/byte (FP4 weights)`,
    ]);
  });
}

const GPUS = Object.freeze({ y: 8, w: 88, h: 60, xs: [24, 164, 304, 444], stubY: 84, flowYs: [100, 110, 120] });
const WIDE_MATRIX = Object.freeze({ x: 24, y: 176, cell: 8 });
const RIGHT = 326; // frame 7's right column: the counter and the memory bar

function gpuRow(svg) {
  GPUS.xs.forEach((x, i) => {
    G.gpu(svg, { x, y: GPUS.y, w: GPUS.w, h: GPUS.h, label: `GPU ${i + 1}`, showMem: false });
    G.cell(svg, { x: x + GPUS.w / 2 + 24, y: GPUS.y + GPUS.h + 3, size: 14, v: 0, maxAbs: 1, gpu: i + 1 });
  });
}

function allToAll(svg, p) {
  const target = GPUS.xs[0] + GPUS.w / 2;
  connector(svg, [target, GPUS.stubY], [target, GPUS.flowYs.at(-1) + 4]);
  GPUS.xs.slice(1).forEach((x, k) => {
    const fromX = x + GPUS.w / 2;
    const y = GPUS.flowYs[k];
    connector(svg, [fromX, GPUS.stubY], [fromX, y]);
    G.flow(svg, { from: [fromX, y], to: [target, y], carry: 'activation', progress: (p * 3 + k / 3) % 1 });
  });
  note(svg, 24, 144, 'all-to-all: each token goes to the GPU that holds its experts');
}

function wideMemory(svg) {
  const weights = V4_PRO.checkpointBytes / EP_SHOWN;
  const free = freeHbmPerGpu({ hbmBytes: GB300.hbmBytes, weightBytes: V4_PRO.checkpointBytes, gpus: EP_SHOWN });
  note(svg, RIGHT, 234, `memory per GPU, ${formatBytes(GB300.hbmBytes)} (nominal)`, { cls: '' });
  G.shareBar(svg, { x: RIGHT, y: 242, w: 230, parts: [{ name: `weights ${formatBytes(weights)}`, value: weights, hue: 1 }, { name: `free ${formatBytes(free)}`, value: free, hue: 2 }], tail: 'none', label: 'memory per GPU' });
  return { weights, free };
}

export function drawFrame7(svg, p) {
  scene(svg, p, (g) => drawFrame6(g, 1), (g) => {
    const m = WIDE_MATRIX;
    gpuRow(g);
    allToAll(g, p);
    const shards = [1, 2, 3, 4].map((gpu) => ({ rows: [gpu, gpu], gpu }));
    G.matrix(g, { x: m.x, y: m.y, values: Array.from({ length: EXPERT_COUNT.rows }, () => Array(COLS).fill(0)), cell: m.cell, maxAbs: 1, format: () => '', label: 'experts', shards });
    G.selectionMark(g, { x: m.x, y: m.y, w: m.cell, h: m.cell });
    const rowsBottom = m.y + EXPERT_COUNT.rows * m.cell;
    connector(g, [m.x + COLS * m.cell + 6, m.y + 4 * m.cell], [m.x + COLS * m.cell + 6, rowsBottom]);
    note(g, m.x + COLS * m.cell + 12, (m.y + 4 * m.cell + rowsBottom) / 2 - 2, `+ ${EP_SHOWN - 4} others`);
    note(g, m.x + COLS * m.cell + 12, (m.y + 4 * m.cell + rowsBottom) / 2 + 14, `(EP ${EP_SHOWN})`);
    counter(g, { x: RIGHT, y: 172, value: Math.round(lerp(ONE_GPU, WIDE, seg(p, 0.1, 0.9))), label: 'expert 0: tokens this step' });
    const { weights, free } = wideMemory(g);
    lines(g, LEFT, 318, [
      `${USERS_SHOWN} × ${EP_SHOWN} × ${PER_TOKEN} ÷ ${EXPERTS} = ${formatCount(WIDE)} tokens per expert`,
      `${formatBytes(V4_PRO.checkpointBytes)} would not fit one ${formatBytes(GB300.hbmBytes)} GPU; over ${EP_SHOWN}: ${formatBytes(weights)} of weights, ${formatBytes(free)} free`,
      `expert intensity ${formatCount(expertIntensity(WIDE))}; FP4 ridge on GB300: ${formatInt(RIDGE)}, reached at ${formatInt(NEEDED)} tokens`,
    ]);
  });
}

// Frame 8: DeepSeek's production layout. The prefill pool is four 8-GPU nodes; the decode pool is 18 nodes, one of them drawn.
const LAYOUT = Object.freeze({ prefill: { x: 20, y: 44 }, decode: { x: 300, y: 56 }, panel: { x: 300, y: 128, w: 260, h: 84 } });
const NODE = DEEPSEEK.gpusPerNode;
const PREFILL_NODES = DEEPSEEK.prefillEp / NODE;
const DECODE_NODES = DEEPSEEK.decodeEp / NODE;
const ZOOMED_GPU = 3; // the decode GPU the zoom opens on
const lerpBox = (a, b, t) => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t) });

function zoom(svg, t, inside) {
  const cell = G.rackLayout({ gpus: NODE, cols: NODE }).cells[ZOOMED_GPU];
  const from = { x: LAYOUT.decode.x + cell.x, y: LAYOUT.decode.y + cell.y, w: 16, h: 16 };
  G.selectionMark(svg, from);
  if (t <= 0) return;
  const box = lerpBox(from, LAYOUT.panel, t);
  connector(svg, [from.x, from.y + 16], [box.x, box.y]);
  connector(svg, [from.x + 16, from.y + 16], [box.x + box.w, box.y]);
  G.svgEl('rect', { class: 'g-frame', x: box.x, y: box.y, width: box.w, height: box.h, rx: 4, fill: 'none' }, G.svgEl('g', { class: 'glyph g-zoom' }, svg));
  if (inside <= 0) return;
  const g = layer(svg, inside);
  const p = LAYOUT.panel;
  note(g, p.x + 8, p.y + 18, 'one decode GPU holds');
  [['routed', 'idle'], ['routed', 'idle'], ['shared', 'idle']].forEach(([label], i) => G.block(g, { x: p.x + 8 + i * 84, y: p.y + 30, w: 76, h: 32, label, state: 'idle' }));
  note(g, p.x + 8, p.y + 78, '2 routed + 1 shared expert', { cls: '' });
}

export function drawFrame8(svg, p) {
  scene(svg, p, (g) => drawFrame7(g, 1), (g) => {
    const groups = (n) => Array.from({ length: n }, (_, k) => ({ from: k * NODE, to: (k + 1) * NODE - 1, label: `node ${k + 1}` }));
    G.rack(g, { ...LAYOUT.prefill, gpus: DEEPSEEK.prefillEp, cols: NODE, groups: groups(PREFILL_NODES), label: `prefill: EP${DEEPSEEK.prefillEp} (${PREFILL_NODES} nodes)` });
    note(g, LAYOUT.decode.x, LAYOUT.decode.y - 12, `decode: EP${DEEPSEEK.decodeEp} (${DECODE_NODES} nodes, one drawn)`, { cls: '' });
    G.rack(g, { ...LAYOUT.decode, gpus: NODE, cols: NODE });
    note(g, LAYOUT.decode.x, 232, `+ ${DECODE_NODES - 1} more nodes, ${DEEPSEEK.decodeEp} GPUs in all`);
    zoom(g, seg(p, 0.2, 0.7), seg(p, 0.7, 0.95));
    lines(g, LEFT, 262, [`prefill: ${DEEPSEEK.prefillRoutedPerGpu} routed + ${DEEPSEEK.prefillSharedPerGpu} shared`, 'expert per GPU']);
    lines(g, 292, 262, [
      `${DEEPSEEK.routedExperts} routed + ${DEEPSEEK.redundant} redundant = ${DEEPSEEK.routedExperts + DEEPSEEK.redundant}`,
      `${DEEPSEEK.routedExperts + DEEPSEEK.redundant} ÷ ${DEEPSEEK.decodeEp} GPUs = ${(DEEPSEEK.routedExperts + DEEPSEEK.redundant) / DEEPSEEK.decodeEp} routed experts per GPU`,
      `${DEEPSEEK.activePerToken} experts active per token`,
      'measured, dated: Feb 2025',
    ]);
  });
}
