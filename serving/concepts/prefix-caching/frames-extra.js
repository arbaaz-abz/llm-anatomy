// prefix-caching frames 9-11: a router choosing between two replicas, the three memory tiers, and the price chart. Each starts by
// fading the previous scene out during [0, HANDOFF] and is a pure function of (p, text).
import * as G from '@shared/glyphs.js';
import { formatBytes, formatDuration } from '@math/core.js';
import { PREFIX_REQUESTS, routeHits } from '@math/prefix.js';
import { DEFAULT_MODEL, DEFAULT_PATHS, LOG, SIZE, shortLabel } from './scenes.js';
import { PREFILL_TITLE } from './facts.js';
import { SCALE_UP } from './numbers.js';
import { formatPrice } from './format.js';
import { CELL, arriving, layer, leaving, lerp, note, seg } from './stage.js';

const [A, B, C] = [0, 1, 2];
const nodeText = (id) => DEFAULT_MODEL.find((n) => n.id === id).text;

// Frame 9 ------------------------------------------------------------------------------------------------------------------

const REPLICA = Object.freeze({ y: 46, w: 250, h: 94, xs: [14, 316], treeDx: 12, treeDy: 10, mini: 1 });
const ROUTER = Object.freeze({ x: 240, y: 178, w: 100, h: 34 });
const MINI_WRAP = 1; // the mini trees wrap after one block, so a four-block path is 221 px wide

// One replica's four-block path as mini-tree nodes; `hits` blocks (from the first) are marked as hits once C is routed.
function miniNodes(path, hits, marked) {
  return path.map((id, i) => ({ id, parent: i === 0 ? null : path[i - 1], label: shortLabel(nodeText(id)), state: marked && i < hits ? 'hit' : 'cached', owner: 'A' }));
}

export function replicaHits() {
  const replicas = [DEFAULT_PATHS[A].path, DEFAULT_PATHS[B].path];
  return routeHits({ replicas, request: PREFIX_REQUESTS[C], blockSize: SIZE });
}

function replica(parent, x, title, path, hits, marked) {
  note(parent, x, REPLICA.y - 6, title);
  G.block(parent, { x, y: REPLICA.y, w: REPLICA.w, h: REPLICA.h, label: '', state: 'idle' });
  G.prefixTree(parent, { x: x + REPLICA.treeDx, y: REPLICA.y + REPLICA.treeDy, nodes: miniNodes(path, hits / SIZE, marked), wrapAfter: MINI_WRAP, label: title });
}

export function drawFrame9(svg, p, text, drawPrevious) {
  const hits = replicaHits();
  const paths = [DEFAULT_PATHS[A].path, DEFAULT_PATHS[B].path];
  if (leaving(p) > 0) drawPrevious(layer(svg, leaving(p)));
  const reps = layer(svg, seg(p, 0.1, 0.25));
  ['replica 1 ran A', 'replica 2 ran B'].forEach((title, i) => replica(reps, REPLICA.xs[i], title, paths[i], hits[i], p >= 0.6));
  const router = layer(svg, seg(p, 0.3, 0.42));
  G.token(router, { x: ROUTER.x - 44, y: ROUTER.y + 5, text: 'C', owner: 'C' });
  G.block(router, { ...ROUTER, label: 'router', state: 'idle' });
  const move = seg(p, 0.45, 0.75);
  if (move > 0) {
    G.flow(svg, { from: [ROUTER.x + 15, ROUTER.y], to: [150, REPLICA.y + REPLICA.h + 2], carry: 'token', progress: move });
    G.flow(svg, { from: [ROUTER.x + ROUTER.w - 15, ROUTER.y], to: [440, REPLICA.y + REPLICA.h + 2], carry: 'token', progress: move });
  }
  if (p >= 0.7) G.selectionMark(svg, { x: REPLICA.xs[0], y: REPLICA.y, w: REPLICA.w, h: REPLICA.h });
  const counts = layer(svg, seg(p, 0.7, 0.85));
  [['KV-aware sends C to replica 1', 0, 140], ['round-robin sends C to replica 2', 1, 440]].forEach(([label, i, cx]) => {
    note(counts, cx, 232, label, { anchor: 'middle' });
    G.vector(counts, { x: cx - CELL - 4, y: 242, values: [0], cell: CELL, orient: 'row', maxAbs: 1, format: () => String(hits[i]) });
    note(counts, cx + 4, 242 + CELL / 2 + 4, 'hit tokens');
  });
  note(layer(svg, seg(p, 0.85, 1)), 14, 330, 'routers also balance load, so they trade some hits for even queues', { cls: '' });
}

// Frame 10 -----------------------------------------------------------------------------------------------------------------

const TIER = Object.freeze({ xs: [30, 225, 420], y: 62, chipY: 188, pitch: 28 });
// The four blocks frame 7 evicted, in pop order; each one's x over the frame (HBM → CPU memory → storage, one back on a hit).
const EVICTED = LOG[3].evicted.map(shortLabel);
const MOVES = Object.freeze([ // per chip: [column at start, then (start, end, to) segments]
  [[0.3, 0.45, 1]], [[0.35, 0.5, 1], [0.8, 0.95, 0]], [[0.4, 0.55, 1], [0.6, 0.75, 2]], [[0.45, 0.6, 1], [0.65, 0.8, 2]],
]);

function chipX(chip, p) {
  return MOVES[chip].reduce((x, [from, to, col]) => lerp(x, TIER.xs[col], seg(p, from, to)), TIER.xs[0]);
}

export function drawFrame10(svg, p, text, drawPrevious) {
  if (leaving(p) > 0) drawPrevious(layer(svg, leaving(p)));
  const g = layer(svg, arriving(p));
  G.gpu(g, { x: TIER.xs[0], y: TIER.y - 10, w: 130, h: 88, memFill: 1, label: 'HBM' });
  G.block(g, { x: TIER.xs[1], y: TIER.y, w: 130, h: 60, label: 'CPU memory', state: 'idle' });
  G.block(g, { x: TIER.xs[2], y: TIER.y, w: 130, h: 60, label: 'storage', state: 'idle' });
  G.flow(g, { from: [TIER.xs[0] + 132, TIER.y + 30], to: [TIER.xs[1] - 2, TIER.y + 30], carry: 'kv', progress: seg(p, 0.3, 0.6) });
  G.flow(g, { from: [TIER.xs[1] + 132, TIER.y + 30], to: [TIER.xs[2] - 2, TIER.y + 30], carry: 'kv', progress: seg(p, 0.6, 0.8) });
  const chips = layer(svg, seg(p, 0.1, 0.25));
  EVICTED.forEach((label, i) => G.token(chips, { x: chipX(i, p), y: TIER.chipY + i * TIER.pitch, text: label, state: 'idle' }));
  note(layer(svg, seg(p, 0.1, 0.25)), TIER.xs[0], TIER.chipY - 12, 'blocks evicted in the last frame');
  const lines = layer(svg, seg(p, 0.85, 1));
  text.tiers.forEach((line, i) => note(lines, 14, 308 + i * 14, line, { cls: '' }));
}

// Frame 11 -----------------------------------------------------------------------------------------------------------------

const CHART = Object.freeze({ y: 76, h: 120 });
export function drawFrame11(svg, p, text, drawPrevious) {
  if (leaving(p) > 0) drawPrevious(layer(svg, leaving(p)));
  const providers = text.providers;
  if (!providers) { note(svg, 14, 100, 'prices unavailable: the data did not load'); return; }
  const [sonnet, , deepseek] = providers;
  const max = sonnet.base * sonnet.write;
  note(layer(svg, arriving(p)), 14, 22, 'price per million input tokens, both charts on one scale', { cls: '' });
  const one = layer(svg, seg(p, 0.15, 0.3));
  note(one, 30, CHART.y - 22, `${sonnet.label} (5-minute write)`);
  G.bars(one, { x: 30, y: CHART.y, h: CHART.h, w: 210, values: [sonnet.base, sonnet.base * sonnet.write, sonnet.base * sonnet.read], labels: ['plain', 'write', 'read'], max, format: formatPrice, label: 'Anthropic prices' });
  const two = layer(svg, seg(p, 0.35, 0.5));
  note(two, 330, CHART.y - 22, deepseek.label);
  G.bars(two, { x: 330, y: CHART.y, h: CHART.h, w: 140, values: [deepseek.base, deepseek.hitUsd], labels: ['miss', 'hit'], max, format: formatPrice, label: 'DeepSeek prices' });
  const out = layer(svg, seg(p, 0.6, 0.75));
  note(out, 14, 246, `what a hit saves: ${formatDuration(SCALE_UP.skippedS)} of GPU math for a ${SCALE_UP.tokens.toLocaleString('en-US')}-token prefix`, { cls: '' });
  note(out, 14, 262, `what it costs: ${formatBytes(SCALE_UP.heldBytes)} held until reuse (H200 example from ${PREFILL_TITLE})`, { cls: '' });
  note(layer(svg, seg(p, 0.8, 0.95)), 14, 300, `prices read ${text.readOn}; they change`);
}
