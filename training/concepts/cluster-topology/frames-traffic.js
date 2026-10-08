// cluster-topology frames 3, 4, 5, 9: what each parallelism sends, as a comm lane against a compute lane on one clock.
// Each drawFrameN(svg, p) is a pure function of p (template rule 4); the followed GPU is GPU 1 of server 1.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatRatio } from '@math/core.js';
import { epMinLinkGBps } from '@math/topology.js';
import { SYSTEMS, DEGREES, TOKENS_PER_REPLICA, LANES, CAP_UNITS } from './numbers.js';
import { int, ratioFor, pct0, pct1, tensorLayerWork } from './format.js';
import { tileAt, RACK8, seg, ease, label, select, selectGpu, linked, linkedText, wire, hop, laneLinks } from './stage.js';

const H100 = SYSTEMS.h100;
const GFLOP = 1e9;
const RING = Object.freeze([0, 1, 2, 3, 7, 6, 5, 4]); // a snake through the 4 × 2 rack, so every hop is a short line
const BASIS = 'one full training step; both lanes in the same time units';
const MIN_UNITS = 1e-6;

// A lane segment that exists only once it has length (the glyph rejects from ≥ to).
const piece = (from, to, kind, text) => (to - from > MIN_UNITS ? [{ from, to, kind, label: text }] : []);

function lanes(svg, { x, y, w, scale = LANES.pxPerUnit, cap = null, rows, letters }) {
  const spec = rows.map(({ name, segments }) => ({ label: name, segments }));
  G.laneTimeline(svg, { x, y, w, lanes: spec, scale, cap, label: 'comm vs compute' });
  return laneLinks(svg, { x, y, w, lanes: spec, scale, cap }, letters);
}

const commText = (grow, ratio) => (grow >= 1 ? pct1(ratio) : null);

function ringDots(host, rackX, rackY, p) {
  RING.forEach((tile, k) => {
    const a = tileAt(rackX, rackY, tile, 4);
    const b = tileAt(rackX, rackY, RING[(k + 1) % RING.length], 4);
    hop(host, [a.cx, a.cy], [b.cx, b.cy], 'activation', ease(seg(p, 0.05 + 0.07 * k, 0.3 + 0.07 * k)));
  });
}

// ---- frame 3: tensor parallelism, inside a server and spread over eight ----
const F3 = Object.freeze({ rack: { x: 10, y: 58 }, lanesA: { x: 150, y: 58, w: 52 + 140 }, serversY: 168, lanesB: { x: 4, y: 262, w: 572 }, pitch: 46, blockW: 38, switchY: 214 });

export function drawFrame3(svg, p) {
  const { rack, lanesA, serversY, lanesB, pitch, blockW, switchY } = F3;
  const rIn = ratioFor({ cut: 'tensor', degree: DEGREES.tensor, where: 'inside' }, H100);
  const rNet = ratioFor({ cut: 'tensor', degree: DEGREES.tensor, where: 'network' }, H100);
  const computeGrow = ease(seg(p, 0, 0.4));
  const grow = ease(seg(p, 0.15, 0.85));
  label(svg, 10, 10, BASIS, { cls: '' });
  label(svg, 10, 26, 'compute at dense BF16 peak; real kernels run slower, so real ratios are smaller');
  label(svg, 10, 46, 'inside one server', { cls: '' });
  G.rack(svg, { x: rack.x, y: rack.y, gpus: 8, cols: 4 });
  linked(svg, 'comm', { x: rack.x, y: rack.y, w: RACK8.w, h: RACK8.h }, (host) => ringDots(host, rack.x, rack.y, p));
  selectGpu(svg, rack.x, rack.y, 4);
  linkedText(svg, 'link', { x: rack.x, y: rack.y + RACK8.h + 14, str: `NVLink ${int(H100.nvlinkGBps)} GB/s each way` });
  lanes(svg, {
    ...lanesA,
    rows: [
      { name: 'compute', segments: piece(0, LANES.computeUnits * computeGrow, 'compute') },
      { name: 'comm', segments: piece(0, rIn * LANES.computeUnits * grow, 'comm', commText(grow, rIn)) },
    ],
    letters: ['flops', 'comm'],
  });
  label(svg, 10, serversY - 14, 'spread over 8 servers', { cls: '' });
  Array.from({ length: 8 }, (_, i) => i).forEach((i) => {
    const x = 10 + i * pitch;
    G.block(svg, { x, y: serversY, w: blockW, h: 22, label: String(i + 1), state: 'idle' });
    wire(svg, x + blockW / 2, serversY + 22, x + blockW / 2, switchY);
    hop(svg, [x + blockW / 2, serversY + 22], [x + blockW / 2, switchY], 'activation', ease(seg(p, 0.05 + 0.05 * i, 0.3 + 0.05 * i)));
  });
  select(svg, 10, serversY, blockW, 22);
  G.block(svg, { x: 10, y: switchY, w: 7 * pitch + blockW, h: 20, label: 'network switch', state: 'idle' });
  linkedText(svg, 'link', { x: 10, y: switchY + 36, str: `network ${int(H100.networkGBps)} GB/s each way` });
  const net = lanes(svg, {
    ...lanesB, cap: { at: CAP_UNITS, label: `continues: ${pct0(rNet)}` },
    rows: [
      { name: 'compute', segments: piece(0, LANES.computeUnits * computeGrow, 'compute') },
      { name: 'comm', segments: piece(0, rNet * LANES.computeUnits * grow, 'comm', commText(grow, rNet)) },
    ],
    letters: ['flops', 'comm'],
  });
  const { bytes, flops } = tensorLayerWork(DEGREES.tensor);
  label(svg, 10, lanesB.y + net.height + 16, `per layer per step (forward and backward): ${formatBytes(bytes)} sent per GPU vs ${int(flops / GFLOP)} GFLOP per GPU`);
}

// ---- frame 4: data parallelism, one gradient sync that hides under backward ----
const F4 = Object.freeze({ servers: { x: 10, y: 64, pitch: 46, w: 38 }, switchY: 118, lanes: { x: 4, y: 186, w: 52 + 380 }, scale: 3.6 });
const FORWARD_UNITS = 100 / 3; // the forward pass is a third of a full step: backward costs twice the forward matmul

export function drawFrame4(svg, p) {
  const { servers, switchY, lanes: at, scale } = F4;
  const ratio = ratioFor({ cut: 'data', degree: DEGREES.data, where: 'network', tokens: TOKENS_PER_REPLICA.default }, H100);
  const low = ratioFor({ cut: 'data', degree: DEGREES.data, where: 'network', tokens: TOKENS_PER_REPLICA.low }, H100);
  const fwd = seg(p, 0, 0.3);
  const bwd = seg(p, 0.3, 0.75);
  const sync = ease(seg(p, 0.4, 0.8));
  label(svg, 10, 10, BASIS, { cls: '' });
  label(svg, 10, 26, `tokens per replica per step: ${int(TOKENS_PER_REPLICA.default)} (stand-in)`);
  label(svg, 10, 46, 'GPU 1 of servers 1, 2, …, 64', { cls: '' });
  [0, 1, 2].forEach((i) => {
    const x = servers.x + i * servers.pitch;
    G.block(svg, { x, y: servers.y, w: servers.w, h: 22, label: String(i + 1), state: 'idle' });
    wire(svg, x + servers.w / 2, servers.y + 22, x + servers.w / 2, switchY);
    hop(svg, [x + servers.w / 2, servers.y + 22], [x + servers.w / 2, switchY], 'gradient', sync);
  });
  select(svg, servers.x, servers.y, servers.w, 22);
  label(svg, servers.x + 3 * servers.pitch, servers.y + 11, '… 61 others');
  G.block(svg, { x: servers.x, y: switchY, w: 2 * servers.pitch + servers.w, h: 20, label: 'network switch', state: 'idle' });
  linkedText(svg, 'link', { x: 10, y: switchY + 36, str: `network ${int(H100.networkGBps)} GB/s each way` });
  lanes(svg, {
    ...at, scale,
    rows: [
      { name: 'compute', segments: [...piece(0, FORWARD_UNITS * fwd, 'forward', fwd >= 1 ? 'forward' : null), ...piece(FORWARD_UNITS, FORWARD_UNITS + (100 - FORWARD_UNITS) * bwd, 'backward', bwd >= 1 ? 'backward' : null)] },
      { name: 'comm', segments: piece(FORWARD_UNITS, FORWARD_UNITS + ratio * 100 * sync, 'comm', sync >= 1 ? pct1(ratio) : null) },
    ],
    letters: ['flops', 'comm'],
  });
  label(svg, 10, at.y + 70, 'the sync starts while backward is still running, so it hides under it');
  label(svg, 10, at.y + 90, `ratio ${pct1(ratio)} over the network · rises to ${pct1(low)} at ${int(TOKENS_PER_REPLICA.low)} tokens per replica`);
}

// ---- frame 5: pipeline parallelism, a tiny hand-off at each boundary ----
const F5 = Object.freeze({ x0: 14, y: 76, pitch: 34, w: 20, h: 24, lanes: { x: 10, y: 174, w: 52 + 150 } });

export function drawFrame5(svg, p) {
  const { x0, y, pitch, w, h, lanes: at } = F5;
  const stages = DEGREES.pipeline;
  const net = ratioFor({ cut: 'pipeline', degree: stages, where: 'network' }, H100);
  const nv = ratioFor({ cut: 'pipeline', degree: stages, where: 'inside' }, H100);
  label(svg, 10, 10, BASIS, { cls: '' });
  label(svg, 10, 46, `pipeline: ${stages} stages, one per server`, { cls: '' });
  Array.from({ length: stages }, (_, i) => G.block(svg, { x: x0 + i * pitch, y, w, h, label: String(i + 1), state: 'idle' }));
  select(svg, x0, y, w, h);
  linked(svg, 'comm', { x: x0, y: y - 6, w: (stages - 1) * pitch + w, h: h + 12 }, (host) => {
    for (let i = 0; i < stages - 1; i += 1) {
      const a = x0 + i * pitch + w;
      const b = x0 + (i + 1) * pitch;
      hop(host, [a, y + 6], [b, y + 6], 'activation', ease(seg(p, 0.05 + 0.03 * i, 0.15 + 0.03 * i)));
      hop(host, [b, y + h - 6], [a, y + h - 6], 'gradient', ease(seg(p, 0.45 + 0.03 * (stages - 2 - i), 0.55 + 0.03 * (stages - 2 - i))));
    }
  });
  label(svg, 10, y + h + 22, 'activation forward (top arrows), its gradient back (bottom arrows)');
  const grow = ease(seg(p, 0.5, 0.95));
  lanes(svg, {
    ...at,
    rows: [
      { name: 'compute', segments: piece(0, LANES.computeUnits * ease(seg(p, 0, 0.4)), 'compute') },
      { name: 'comm', segments: piece(0, net * LANES.computeUnits * grow, 'comm', commText(grow, net)) },
    ],
    letters: ['flops', 'comm'],
  });
  label(svg, 10, at.y + 76, `network ${int(H100.networkGBps)} GB/s each way: ${pct1(net)} of the compute`, { cls: '' });
  label(svg, 10, at.y + 96, `NVLink ${int(H100.nvlinkGBps)} GB/s each way would be ${pct1(nv)}`);
}

// ---- frame 9: expert all-to-all against DeepSeek-V4's hiding line ----
const F9 = Object.freeze({ rack: { x: 10, y: 50 }, lanes: { x: 4, y: 168, w: 572 }, verdictX: 462 });

export function drawFrame9(svg, p) {
  const { rack, lanes: at, verdictX } = F9;
  const inside = ratioFor({ cut: 'expert', where: 'inside' }, H100);
  const over = ratioFor({ cut: 'expert', where: 'network' }, H100);
  const need = epMinLinkGBps({ peakTflops: H100.peakTflops });
  const computeGrow = ease(seg(p, 0, 0.35));
  const grow = ease(seg(p, 0.2, 0.8));
  label(svg, 10, 10, 'for DeepSeek-V4\'s expert shape', { cls: '' });
  label(svg, 10, 26, BASIS);
  G.rack(svg, { x: rack.x, y: rack.y, gpus: 8, cols: 4 });
  linked(svg, 'comm', { x: rack.x, y: rack.y, w: RACK8.w, h: RACK8.h }, (host) => {
    [[0, 6], [3, 5], [1, 4], [7, 2]].forEach(([a, b], k) => {
      const from = tileAt(rack.x, rack.y, a, 4);
      const to = tileAt(rack.x, rack.y, b, 4);
      hop(host, [from.cx, from.cy], [to.cx, to.cy], 'token', ease(seg(p, 0.05 + 0.08 * k, 0.3 + 0.08 * k)));
    });
  });
  selectGpu(svg, rack.x, rack.y, 4);
  label(svg, 150, rack.y + 10, 'expert all-to-all in one H100 server', { cls: '' });
  linkedText(svg, 'link', { x: 150, y: rack.y + 34, str: `${int(H100.peakTflops)} TFLOPS ÷ 6,144 FLOPs per byte = ${int(need)} GB/s of link needed` });
  label(svg, 150, rack.y + 56, 'DeepSeek-V4: expert traffic hides when compute per');
  label(svg, 150, rack.y + 72, 'byte of link is at most 6,144 FLOPs');
  const rows = [
    { name: 'compute', segments: piece(0, LANES.computeUnits * computeGrow, 'compute') },
    { name: 'NVLink', segments: piece(0, inside * 100 * grow, 'comm', commText(grow, inside)) },
    { name: 'network', segments: piece(0, over * 100 * grow, 'comm', commText(grow, over)) },
  ];
  const L = lanes(svg, { ...at, rows, letters: ['flops', 'comm', 'comm'] });
  const shown = seg(p, 0.8, 1);
  [[1, true, 'hides'], [2, false, 'overflows']].forEach(([lane, ok, text]) => {
    const cy = at.y + L.lanes[lane].y + 12;
    if (shown <= 0) return;
    const g = G.svgEl('g', { opacity: shown.toFixed(3) }, svg);
    G.verdict(g, { x: verdictX, y: cy, ok });
    label(g, verdictX + 16, cy, text);
  });
  label(svg, 10, at.y + L.height + 20, `NVLink ${int(H100.nvlinkGBps)} GB/s each way → ${pct1(inside)} of compute · network ${int(H100.networkGBps)} GB/s each way → ${pct1(over)}`, { cls: '' });
  label(svg, 10, at.y + L.height + 40, `hiding line: 100% of compute · ${formatRatio(need / H100.networkGBps)} more link than the network gives`);
}
