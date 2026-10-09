// disaggregation frames 4–5: the KV cache crosses a link (to scale against the prefill it follows), then the pools get their own sizes.
import * as G from '@shared/glyphs.js';
import { formatCount, formatDuration, formatBytes } from '@math/core.js';
import { kvCacheBytes, sharePct } from '@math/memory.js';
import { RUNNING_EXAMPLE, kvTransferTime } from '@math/serving.js';
import { BRANCH_PROMPT, STAGE_LINKS, VLLM_GB200 } from './numbers.js';
import { LEFT, LINE, seg, lerp, layer, note, lines, linkGroup, scene } from './stage.js';
import { POOLS, POOL_PX_PER_MS, PREFILL_PX, TIMES, poolRows, drawFrame3 } from './frames-colocated.js';

const MS = 1000;
const ms = (value) => formatDuration(value / MS);
const GB = 1e9;
const KV_BYTES = kvCacheBytes({ bytesPerToken: RUNNING_EXAMPLE.kvBytesPerToken, tokens: BRANCH_PROMPT });
const transferMs = (id) => kvTransferTime(BRANCH_PROMPT, RUNNING_EXAMPLE.kvBytesPerToken, STAGE_LINKS[id].gbPerS * GB) * MS;
const LINKS = Object.freeze([
  { id: 'net400', label: 'network 400 Gb/s (50 GB/s)', ms: transferMs('net400') },
  { id: 'nvlink', label: 'NVLink5 (900 GB/s each way)', ms: transferMs('nvlink') },
]);
const share = (link) => `${sharePct(link.ms, TIMES.prefillMs).toFixed(1)}%`;
const CHART = Object.freeze({ x: 20, y: 250, w: 150, h: 56 });
const REST = 0.85; // a flow's dot at rest sits short of its head, so the arrowhead stays visible
const D_TICKS = 3; // D has three tokens left to decode

function dLane(svg, startMs) {
  const from = PREFILL_PX + startMs * POOL_PX_PER_MS;
  const steps = Array.from({ length: D_TICKS }, (_, k) => ({ from: from + k * POOLS.stepPx, to: from + (k + 1) * POOLS.stepPx, kind: 'decode' }));
  G.request(svg, { x: POOLS.x0, y: POOLS.decodeY.D, owner: 'D', label: 'D', steps });
  G.selectionMark(svg, { x: POOLS.x0 - 22, y: POOLS.decodeY.D, w: 18, h: 10 });
}

function kvArrow(svg, endMs, progress) {
  const from = [POOLS.x0 + PREFILL_PX, POOLS.prefillY + 5];
  const to = [POOLS.x0 + PREFILL_PX + endMs * POOL_PX_PER_MS, POOLS.decodeY.D + 5];
  const g = linkGroup(svg, 'kv', { x: Math.min(from[0], to[0]) - 4, y: from[1], w: Math.abs(to[0] - from[0]) + 8, h: to[1] - from[1] });
  G.flow(g, { from, to, carry: 'kv', progress });
}

function linkText(svg, link, opacity) {
  if (opacity <= 0) return;
  const g = layer(svg, opacity);
  const x = POOLS.x0 + PREFILL_PX - 10; // left of the arrow, in the free space between the pools
  note(g, x, 100, `KV of D: ${formatBytes(KV_BYTES)}`, { cls: '', anchor: 'end' });
  note(g, x, 116, link.label, { anchor: 'end' });
  note(g, x, 132, `${ms(link.ms)}, ${share(link)} of the prefill`, { cls: '', anchor: 'end' });
}

function chart(svg, opacity) {
  if (opacity <= 0) return;
  const g = layer(svg, opacity);
  G.bars(g, {
    x: CHART.x, y: CHART.y, w: CHART.w, h: CHART.h, values: LINKS.map((l) => l.ms), labels: ['400 Gb/s', 'NVLink5'], max: TIMES.prefillMs,
    reference: { value: TIMES.prefillMs, label: `prefill ${ms(TIMES.prefillMs)}` }, format: (v) => ms(v), label: 'KV transfer time by link',
  });
  note(g, CHART.x, CHART.y - 18, 'KV transfer time by link, against the prefill', { cls: '' });
}

export function drawFrame4(svg, p) {
  scene(svg, p, (g) => drawFrame3(g, 1), (g) => {
    poolRows(g, TIMES.prefillMs);
    const link = p < 0.5 ? LINKS[0] : LINKS[1];
    const start = lerp(LINKS[0].ms, LINKS[1].ms, seg(p, 0.45, 0.55));
    if (p >= 0.1) dLane(g, start);
    if (p >= 0.1) kvArrow(g, start, REST * (p < 0.5 ? seg(p, 0.1, 0.45) : seg(p, 0.55, 0.95)));
    linkText(g, link, seg(p, 0.1, 0.2));
    chart(g, seg(p, 0.1, 0.3));
    const summary = layer(g, seg(p, 0.85, 1));
    lines(summary, 300, 268, LINKS.map((l) => `${l.label.split(' (')[0]}: ${ms(l.ms)}, ${share(l)}`));
    note(summary, 300, 268 + 2 * LINE, 'any prompt past 217 tokens');
    note(g, LEFT, 358, 'all link speeds per GPU, each way: network 400 Gb/s = 50 GB/s; NVLink5 = 900 GB/s');
  });
}

// Frame 5: two pools sized separately: four 2-GPU prefill groups feeding one 8-GPU decode group (vLLM, DeepSeek-R1 on GB200).
const RACK = Object.freeze({ x: 40, prefillY: 60, decodeY: 170 });
const ARROW = Object.freeze({ from: RACK.prefillY + 72, to: RACK.decodeY - 2 }); // clears the rack's own label under the prefill pool
const GROUPS = Object.freeze(Array.from({ length: VLLM_GB200.prefillGroups }, (_, k) => ({ from: k * VLLM_GB200.prefillGpusEach, to: (k + 1) * VLLM_GB200.prefillGpusEach - 1, label: `${VLLM_GB200.prefillGpusEach} GPUs` })));

export function drawFrame5(svg, p) {
  scene(svg, p, (g) => drawFrame4(g, 1), (g) => {
    const total = VLLM_GB200.prefillGroups * VLLM_GB200.prefillGpusEach + VLLM_GB200.decodeGpus;
    G.rack(g, { x: RACK.x, y: RACK.prefillY, gpus: VLLM_GB200.prefillGroups * VLLM_GB200.prefillGpusEach, cols: 8, groups: GROUPS, label: 'prefill pool' });
    G.rack(g, { x: RACK.x, y: RACK.decodeY, gpus: VLLM_GB200.decodeGpus, cols: 8, groups: [{ from: 0, to: VLLM_GB200.decodeGpus - 1, label: `decode: ${VLLM_GB200.decodeGpus} GPUs` }], label: 'decode pool' });
    const centers = GROUPS.map((_, k) => RACK.x + 10 + k * 52 + 21);
    centers.forEach((x, k) => {
      const t = seg(p, 0.1 + k * 0.15, 0.4 + k * 0.15);
      if (t <= 0) return;
      const arrow = linkGroup(g, 'kv', { x: x - 6, y: ARROW.from, w: 12, h: ARROW.to - ARROW.from });
      G.flow(arrow, { from: [x, ARROW.from], to: [x, ARROW.to], carry: 'kv', progress: REST * t });
    });
    note(g, RACK.x + 200, (ARROW.from + ARROW.to) / 2 + 4, 'KV', { cls: '' });
    lines(g, 300, 70, [
      `${VLLM_GB200.prefillGroups} × ${VLLM_GB200.prefillGpusEach} + 1 × ${VLLM_GB200.decodeGpus} = ${total} GPUs`,
      `prefill GPUs: ${formatCount(VLLM_GB200.prefillTokSGpu)} prompt tok/s each`,
      `decode GPUs: ${formatCount(VLLM_GB200.decodeTokSGpu)} output tok/s each`,
      'at 2K tokens in, 2K out',
      'vLLM, 2026-02-03',
      'measured, dated',
    ]);
  });
}
