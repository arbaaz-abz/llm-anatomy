// cluster-topology frames 8 and 10: a rail-optimized network routing one token, and 72 GPUs merging into one NVLink domain.
// Each drawFrameN(svg, p) is a pure function of p (template rule 4); the followed GPU is GPU 1 of server 1.
import * as G from '@shared/glyphs.js';
import { formatRatio } from '@math/core.js';
import { epMinLinkGBps } from '@math/topology.js';
import { SYSTEMS, DEEPSEEK_V3, RAIL, NVL72 } from './numbers.js';
import { int, ratioFor, pct1 } from './format.js';
import { RACK8, RACK72, tileAt, seg, lerp, ease, label, textBlock, selectGpu, linked, linkedText, wire, hop, layer } from './stage.js';

const H100 = SYSTEMS.h100;
const GB200 = SYSTEMS.gb200;
const CHIP_TEXT = 'cat'; // the course's router-toy token (ROUTER_TOY), short enough not to hide the GPU numbers
const TOKEN_CHIP_H = 24;

// The point `t` (0 → 1) of the way along a polyline, by length.
function along(points, t) {
  const lengths = points.slice(1).map((pt, i) => Math.hypot(pt[0] - points[i][0], pt[1] - points[i][1]));
  let left = t * lengths.reduce((a, b) => a + b, 0);
  for (let i = 0; i < lengths.length; i += 1) {
    if (left <= lengths[i] || i === lengths.length - 1) {
      const f = lengths[i] === 0 ? 0 : Math.min(left / lengths[i], 1);
      return [lerp(points[i][0], points[i + 1][0], f), lerp(points[i][1], points[i + 1][1], f)];
    }
    left -= lengths[i];
  }
  return points[0];
}

// Progress (0 → 1) of each leg of a polyline when the whole path is `t` done.
function legProgress(points, t) {
  const lengths = points.slice(1).map((pt, i) => Math.hypot(pt[0] - points[i][0], pt[1] - points[i][1]));
  const total = lengths.reduce((a, b) => a + b, 0);
  let before = 0;
  return lengths.map((len) => {
    const f = Math.min(Math.max((t * total - before) / len, 0), 1);
    before += len;
    return f;
  });
}

// ---- frame 8: four servers, one rail switch, a token's two hops ----
const F8 = Object.freeze({ xs: [12, 156, 300, 444], rackY: 78, switchY: 8, switchH: 22 });
const SERVER_LABELS = Object.freeze(['1', '2', '3', '4', '5', '6', '7', '8']);

export function drawFrame8(svg, p) {
  const { xs, rackY, switchY, switchH } = F8;
  const rail = RAIL.gpuIndex - 1;
  const expert = RAIL.expertGpuIndex - 1;
  const railTop = (k) => tileAt(xs[k], rackY, rail, 4);
  G.block(svg, { x: xs[0] + 28, y: switchY, w: xs[3] - xs[0] + 84, h: switchH, label: `rail ${RAIL.gpuIndex} switch`, state: 'idle' });
  xs.forEach((x, k) => {
    G.rack(svg, { x, y: rackY, gpus: 8, cols: 4, labels: SERVER_LABELS, label: `server ${k + 1}` });
    const t = railTop(k);
    wire(svg, t.cx, t.y, t.cx, switchY + switchH);
  });
  selectGpu(svg, xs[0], rackY, 4);
  const start = railTop(0);
  const end = railTop(RAIL.servers - 1);
  const dest = tileAt(xs[RAIL.servers - 1], rackY, expert, 4);
  const lift = switchY + switchH;
  const path = [[start.cx, start.y], [start.cx, lift], [end.cx, lift], [end.cx, end.y], [dest.cx, dest.cy]];
  const t = ease(seg(p, 0.1, 0.9));
  const head = along(path, t);
  linked(svg, 'comm', { x: xs[0], y: switchY, w: xs[3] + RACK8.w - xs[0], h: rackY - switchY + RACK8.h }, (host) => {
    legProgress(path, t).forEach((leg, i) => hop(host, path[i], path[i + 1], 'token', leg));
  });
  if (t > 0) G.token(svg, { x: head[0] - G.tokenWidth(CHIP_TEXT) / 2, y: head[1] - TOKEN_CHIP_H / 2, text: CHIP_TEXT, state: 'active' });
  const shown = seg(p, 0.3, 0.7);
  linkedText(svg, 'link', { x: 12, y: rackY + RACK8.h + 40, str: `hop 1: network, ${int(H100.networkGBps)} GB/s each way, to GPU ${RAIL.gpuIndex} of server ${RAIL.servers}`, opacity: shown });
  linkedText(svg, 'link', { x: 12, y: rackY + RACK8.h + 58, str: `hop 2: NVLink, to the expert on GPU ${RAIL.expertGpuIndex}`, opacity: seg(p, 0.7, 1) });
  const ratio = DEEPSEEK_V3.nvlinkGBps / DEEPSEEK_V3.ibGBps;
  textBlock(svg, 12, rackY + RACK8.h + 92, [
    `DeepSeek-V3 on H800s: NVLink ${int(DEEPSEEK_V3.nvlinkGBps)} GB/s vs network ${int(DEEPSEEK_V3.ibGBps)} GB/s (${formatRatio(ratio)})`,
    'DeepSeek\'s stated effective rates; direction not given',
    `each token reaches at most ${int(DEEPSEEK_V3.maxNodesPerToken)} servers`,
  ], { gap: 18 });
}

// ---- frame 10: nine servers' worth of GPUs in one NVLink domain ----
const F10 = Object.freeze({ rack: { x: 12, y: 44 }, smallX: 12, smallYs: [44, 116, 188], textX: 282 });
const EXPERT_PAIRS = Object.freeze([[2, 40], [12, 66], [20, 53], [31, 70]]);

export function drawFrame10(svg, p) {
  const { rack, smallX, smallYs, textX } = F10;
  const merge = ease(seg(p, 0.1, 0.6));
  const need = epMinLinkGBps({ peakTflops: GB200.peakTflops });
  const inside = ratioFor({ cut: 'expert', where: 'inside' }, GB200);
  const over = ratioFor({ cut: 'expert', where: 'network' }, GB200);
  const tp16 = ratioFor({ cut: 'tensor', degree: 16, where: 'inside' }, GB200);
  if (merge < 1) {
    const small = layer(svg, 1 - merge);
    smallYs.forEach((y, k) => G.rack(small, { x: smallX, y, gpus: 8, cols: 4, label: `server ${k + 1}` }));
    selectGpu(small, smallX, smallYs[0], 4);
    label(small, smallX, smallYs[2] + RACK8.h + 34, '+ 6 other servers');
  }
  if (merge > 0) {
    const big = layer(svg, merge);
    G.rack(big, { x: rack.x, y: rack.y, gpus: NVL72.gpus, cols: 9 });
    linked(big, 'comm', { x: rack.x, y: rack.y, w: RACK72.w, h: RACK72.h }, (host) => {
      EXPERT_PAIRS.forEach(([a, b], k) => {
        const from = tileAt(rack.x, rack.y, a, 9);
        const to = tileAt(rack.x, rack.y, b, 9);
        hop(host, [from.cx, from.cy], [to.cx, to.cy], 'token', ease(seg(p, 0.6 + 0.07 * k, 0.9 + 0.03 * k)));
      });
    });
    selectGpu(big, rack.x, rack.y, 9);
    label(big, rack.x, rack.y + RACK72.h + 16, `= ${int(NVL72.gpus / H100.domain)} HGX servers`, { cls: '' });
  }
  const shown = seg(p, 0.5, 0.85);
  textBlock(svg, textX, 56, [
    `GB200 NVL72: ${int(NVL72.gpus)} GPUs + ${int(NVL72.graceCpus)} Grace CPUs,`,
    'one NVLink domain',
  ], { cls: '', gap: 18, opacity: shown });
  textBlock(svg, textX, 100, [
    `${int(GB200.nvlinkGBps)} GB/s each way per GPU`,
    `(${(GB200.nvlinkGBps * 2) / 1000} TB/s both directions)`,
    `${int(NVL72.rackNvlinkTbps)} TB/s total`,
  ], { gap: 18, opacity: shown });
  const late = seg(p, 0.75, 1);
  textBlock(svg, textX, 176, [
    'expert parallelism on GB200:',
    `${int(GB200.peakTflops)} TFLOPS BF16, needs ${int(need)} GB/s`,
    `NVLink ${int(GB200.nvlinkGBps)} GB/s each way → ${pct1(inside)}`,
    `network ${int(GB200.networkGBps)} GB/s each way → ${pct1(over)}`,
    '(800 Gb/s port, reported)',
    `tensor 16 inside the rack: ${pct1(tp16)}`,
  ], { gap: 18, opacity: late });
}
