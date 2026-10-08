// cluster-topology frames 1, 2, 6, 7: the two network layers, where Llama 3 put each cut, and Meta's pods.
// Each drawFrameN(svg, p) is a pure function of p (template rule 4); the followed GPU is GPU 1 of server 1.
import * as G from '@shared/glyphs.js';
import { formatRatio } from '@math/core.js';
import { SYSTEMS, LLAMA3_LAYOUT, META } from './numbers.js';
import { int, ratioFor, pct1, plural } from './format.js';
import {
  STAGE, RACK8, RACK16, tileAt, seg, lerp, ease, label, textBlock, select, selectGpu, linked, linkedText, wire, region, hop, fade, layer,
} from './stage.js';

const H100 = SYSTEMS.h100;
const BOTH_DIRECTIONS = 2;

// ---- frame 1: one server, eight GPUs on NVLink ----
const F1 = Object.freeze({ x: 40, y: 56, textX: 196 });

export function drawFrame1(svg, p) {
  const { x, y, textX } = F1;
  G.rack(svg, { x, y, gpus: 8, cols: 4, label: 'H100 server (HGX)' });
  const from = tileAt(x, y, 0, 4);
  linked(svg, 'comm', { x, y, w: RACK8.w, h: RACK8.h }, (host) => {
    for (let i = 1; i < 8; i += 1) {
      const to = tileAt(x, y, i, 4);
      hop(host, [from.cx, from.cy], [to.cx, to.cy], 'activation', ease(seg(p, 0.04 * i, 0.5 + 0.06 * i)));
    }
  });
  selectGpu(svg, x, y, 4);
  const shown = seg(p, 0.1, 0.5);
  textBlock(svg, textX, y + 8, [`${plural(8, 'GPU')} in one NVLink domain`], { cls: '', opacity: shown });
  linkedText(svg, 'link', { x: textX, y: y + 32, str: `NVLink: ${int(H100.nvlinkGBps * BOTH_DIRECTIONS)} GB/s per GPU, both directions`, opacity: shown });
  linkedText(svg, 'link', { x: textX, y: y + 50, str: `= ${int(H100.nvlinkGBps)} GB/s each way`, cls: '', opacity: shown });
  label(svg, textX, y + 96, 'this fast island is the scale-up domain', { opacity: seg(p, 0.5, 0.9) });
}

// ---- frame 2: four servers, a slower network, and the 9× ladder ----
const F2 = Object.freeze({ xs: [38, 168, 298, 428], rackY: 34, switchY: 126, ladder: { x: 38, y: 214, w: 140, h: 90 } });
const PORT_PITCH = 14;

export function drawFrame2(svg, p) {
  const { xs, rackY, switchY, ladder } = F2;
  const grow = ease(seg(p, 0.05, 0.5));
  const ratio = H100.nvlinkGBps / H100.networkGBps;
  xs.forEach((x, k) => {
    label(svg, x + RACK8.w / 2, rackY - 12, `server ${k + 1}`, { anchor: 'middle' });
    G.rack(svg, { x, y: rackY, gpus: 8, cols: 4 });
  });
  linked(svg, 'link', { x: xs[0], y: rackY + RACK8.h, w: xs[3] + RACK8.w - xs[0], h: switchY - rackY - RACK8.h }, (host) => {
    xs.forEach((x) => Array.from({ length: 8 }, (_, i) => x + 6 + i * PORT_PITCH).forEach((px) => wire(host, px, rackY + RACK8.h, px, lerp(rackY + RACK8.h, switchY, grow))));
  });
  G.block(svg, { x: xs[0], y: switchY, w: xs[3] + RACK8.w - xs[0], h: 24, label: 'network switch', state: 'idle' });
  selectGpu(svg, xs[0], rackY, 4);
  linkedText(svg, 'link', { x: xs[0], y: switchY + 44, str: `400 Gb/s = ${int(H100.networkGBps)} GB/s each way, one port per GPU`, opacity: seg(p, 0.3, 0.7) });
  const shrink = ease(seg(p, 0.45, 0.9));
  linked(svg, 'link', { x: ladder.x, y: ladder.y - 14, w: ladder.w, h: ladder.h + 30 }, (host) => {
    G.bars(host, { ...ladder, values: [H100.nvlinkGBps, lerp(H100.nvlinkGBps, H100.networkGBps, shrink)], labels: ['NVLink', 'network'], max: H100.nvlinkGBps, format: (v) => `${int(v)} GB/s`, label: 'link speed per GPU, each way' });
  });
  const note = seg(p, 0.7, 1);
  textBlock(svg, 214, ladder.y + 40, [`${int(H100.nvlinkGBps)} ÷ ${int(H100.networkGBps)} = ${formatRatio(ratio)}`], { cls: '', opacity: note });
  label(svg, 214, ladder.y + 58, 'per GPU, each way', { opacity: note });
}

// ---- frame 6: Llama 3's order, innermost first ----
const F6 = Object.freeze({ tagX: 10, tagW: 84, bodyX: 110, rows: { tensor: 24, pipeline: 112, data: 188 } });
const STAGE_BLOCK = Object.freeze({ w: 18, h: 24, pitch: 22 });

function tag(svg, name, y, lit) {
  G.block(svg, { x: F6.tagX, y: y + 4, w: F6.tagW, h: 26, label: name, state: lit ? 'active' : 'dim' });
}

export function drawFrame6(svg, p) {
  const { bodyX, rows } = F6;
  const { tp, pp, dp } = LLAMA3_LAYOUT;
  const net = H100.networkGBps;
  const lit = { tensor: p >= 0.2, pipeline: p >= 0.5, data: p >= 0.8 };
  tag(svg, `tensor ${tp}`, rows.tensor, lit.tensor);
  G.rack(svg, { x: bodyX, y: rows.tensor, gpus: 8, cols: 4 });
  selectGpu(svg, bodyX, rows.tensor, 4);
  textBlock(svg, bodyX + RACK8.w + 20, rows.tensor + 22, ['inside each server', `NVLink ${int(H100.nvlinkGBps)} GB/s each way`], { gap: 18 });
  tag(svg, `pipeline ${pp}`, rows.pipeline, lit.pipeline);
  Array.from({ length: pp }, (_, i) => G.block(svg, { x: bodyX + i * STAGE_BLOCK.pitch, y: rows.pipeline + 3, w: STAGE_BLOCK.w, h: STAGE_BLOCK.h, label: String(i + 1), state: lit.pipeline ? 'idle' : 'dim' }));
  label(svg, bodyX, rows.pipeline + 44, `across servers, network ${int(net)} GB/s each way`);
  tag(svg, `data ${dp}`, rows.data, lit.data);
  const groupLabels = ['group 1', 'group 2', `group ${dp}`];
  groupLabels.forEach((name, i) => G.block(svg, { x: bodyX + [0, 80, 280][i], y: rows.data + 3, w: 70, h: 24, label: name, state: lit.data ? 'idle' : 'dim' }));
  label(svg, bodyX + 80 + 70 + 60, rows.data + 15, `${int(dp - 3)} others`, { anchor: 'start' });
  label(svg, bodyX, rows.data + 44, 'groups of pipelines, across the network');
  select(svg, bodyX, rows.pipeline + 3, STAGE_BLOCK.w, STAGE_BLOCK.h, lit.pipeline ? 1 : 0);
  const total = tp * pp * dp;
  textBlock(svg, 10, 262, [`${tp} × ${pp} × ${dp} = ${int(total)} GPUs`, 'innermost → outermost: tensor, context, pipeline, data'], { cls: '', gap: 18 });
  const dpHigh = pct1(ratioFor({ cut: 'data', degree: dp, where: 'network', tokens: 262144 }, H100));
  const ppHigh = pct1(ratioFor({ cut: 'pipeline', degree: pp, where: 'network' }, H100));
  textBlock(svg, 10, 304, [
    `data parallelism sends more per unit of compute than pipeline here (${dpHigh} vs ${ppHigh}),`,
    'but it overlaps with backward; pipeline hand-offs sit between stages',
  ], { gap: 16 });
}

// ---- frame 7: Meta's 24K-GPU cluster, pods and the thin uplinks ----
const F7 = Object.freeze({ pod: { x: 10, y: 8, w: 360, h: 206 }, racks: [{ x: 30, y: 68 }, { x: 160, y: 68 }], leaf: { x: 110, y: 28, w: 130, h: 24 }, spineY: 238, podsY: 276, podW: 56, podPitch: 64 });

export function drawFrame7(svg, p) {
  const { pod, racks, leaf, spineY, podsY, podW, podPitch } = F7;
  const tiled = seg(p, 0, 0.5);
  region(svg, pod);
  label(svg, pod.x + 8, pod.y + 10, `pod 1: ${int(META.gpusPerPod)} GPUs, full bandwidth`, { cls: '' });
  G.block(svg, { ...leaf, label: 'leaf switch', state: 'idle' });
  racks.forEach((r, i) => {
    G.rack(svg, { x: r.x, y: r.y, gpus: META.gpusPerRack, cols: 4 });
    label(svg, r.x + RACK16.w / 2, r.y + RACK16.h + 14, `rack ${i + 1}: ${int(META.gpusPerRack)} GPUs`, { anchor: 'middle' });
    wire(svg, r.x + RACK16.w / 2, r.y, leaf.x + leaf.w * (0.3 + 0.4 * i), leaf.y + leaf.h);
  });
  selectGpu(svg, racks[0].x, racks[0].y, 4);
  label(svg, 330, 124, `${int(META.racksPerPod - racks.length)}`, { cls: '', anchor: 'middle' });
  label(svg, 330, 140, 'other racks', { anchor: 'middle' });
  const podsWidth = podPitch * META.pods - (podPitch - podW);
  G.block(svg, { x: 10, y: spineY - 22, w: podsWidth, h: 22, label: `aggregation layer, oversubscribed ${META.oversubscription}`, state: 'idle' });
  for (let i = 0; i < META.pods; i += 1) {
    const x = 10 + i * podPitch;
    const arrive = seg(tiled, i / META.pods, (i + 1) / META.pods + 0.2);
    fade(layer(svg, arrive), 1);
    const g = layer(svg, arrive);
    wire(g, x + podW / 2, spineY, x + podW / 2, podsY);
    region(g, { x, y: podsY, w: podW, h: 36 });
    if (i < 2) label(g, x + podW / 2, podsY + 18, `pod ${i + 1}`, { anchor: 'middle', cls: '' });
  }
  label(svg, 10 + 2 * podPitch, podsY + 52, `${int(META.pods - 2)} other pods`);
  linkedText(svg, 'comm', { x: 10, y: podsY + 52, str: `${int(META.perGpuGbps)} Gb/s per GPU`, opacity: seg(p, 0.5, 0.8) });
  hop(svg, [10 + podW / 2, spineY - 28 + 6], [10 + podPitch + podW / 2, spineY - 28 + 6], 'gradient', seg(p, 0.55, 0.95));
}
