// disaggregation frames 9–10: wide EP wants one fast link domain (the 72-GPU rack against two nodes on a network), then the bill.
import * as G from '@shared/glyphs.js';
import { formatCount, formatRatio } from '@math/core.js';
import { GB300, GB300_GAIN, GB300_TOK_S_USER, STAGE_LINKS, PAPERS, EP_SHOWN } from './numbers.js';
import { LEFT, LINE, seg, layer, note, lines, scene } from './stage.js';
import { drawFrame8 } from './frames-experts.js';

const NVL = Object.freeze({ x: 20, y: 50, gpus: GB300.gpus, cols: 9 });
const NODES = Object.freeze([{ x: 330, y: 44 }, { x: 330, y: 172 }]);
const LINK = Object.freeze({ x: NODES[0].x + 57, from: 124, to: 170 });
const CROSS_PAIRS = Object.freeze([[0, 70], [8, 63], [4, 40], [27, 17], [60, 12], [35, 53]]);
const QUEUE = 5; // dots waiting at the link
const BULLET = 13; // a list item's second line sits under its text, not its bullet

function insideRack(svg, p) {
  const cells = G.rackLayout({ gpus: NVL.gpus, cols: NVL.cols }).cells;
  const centre = (i) => [NVL.x + cells[i].x + 8, NVL.y + cells[i].y + 8];
  CROSS_PAIRS.forEach(([a, b], k) => G.flow(svg, { from: centre(a), to: centre(b), carry: 'activation', progress: (p * 2 + k / CROSS_PAIRS.length) % 1 }));
}

function acrossLink(svg, p) {
  NODES.forEach((n) => G.rack(svg, { ...n, gpus: 8, cols: 4, label: '8-GPU node' }));
  const wait = seg(p, 0.1, 0.9) * 0.15;
  for (let i = 0; i < QUEUE; i += 1) G.flow(svg, { from: [LINK.x, LINK.from], to: [LINK.x, LINK.to], carry: 'activation', progress: i * 0.07 + wait });
  const net = STAGE_LINKS.net400;
  note(svg, LINK.x + 12, (LINK.from + LINK.to) / 2 - 2, 'network link:');
  note(svg, LINK.x + 12, (LINK.from + LINK.to) / 2 + 14, `${net.gbPerS} GB/s each way`);
}

export function drawFrame9(svg, p) {
  scene(svg, p, (g) => drawFrame8(g, 1), (g) => {
    G.rack(g, { x: NVL.x, y: NVL.y, gpus: NVL.gpus, cols: NVL.cols, label: `NVL72: one NVLink domain` });
    insideRack(g, p);
    acrossLink(g, p);
    const both = (2 * STAGE_LINKS.nvlink.gbPerS) / 1000;
    lines(g, 292, 268, [
      `NVL72: ${GB300.gpus} GPUs, about ${formatCount(GB300.rackNvlinkTbps)} TB/s NVLink`,
      `NVIDIA's figure: ${GB300.gpus} × ${both} TB/s, so it`,
      'counts both directions',
      `GB300's ${formatCount(GB300.hbmBytes / 1e9)} GB allowed EP ${EP_SHOWN}: ${formatRatio(GB300_GAIN)}`,
      'more tokens per GPU than GB200 at',
      `${GB300_TOK_S_USER} tok/s/user (InferenceX, 2026-05)`,
    ]);
  });
}

// Frame 10: the bill, one line at a time, beside the two pictures it is about.
const BILL = Object.freeze([
  ['two pools to size and', 'rescale as traffic shifts'],
  ['KV transfer per request'],
  ['all-to-all every MoE layer'],
  ['hot experts → extra copies (EPLB)'],
]);

function pictures(svg) {
  G.gpu(svg, { x: LEFT, y: 8, w: 64, h: 56, label: 'prefill', showMem: false });
  G.gpu(svg, { x: 136, y: 8, w: 64, h: 56, label: 'decode', showMem: false });
  G.flow(svg, { from: [76, 36], to: [130, 36], carry: 'kv', progress: 0.5 });
  [8, 92, 176].forEach((x) => G.gpu(svg, { x, y: 110, w: 64, h: 52, showMem: false }));
  [[72, 88], [156, 172]].forEach(([a, b]) => G.flow(svg, { from: [a, 136], to: [b, 136], carry: 'activation', progress: 0.5 }));
  note(svg, LEFT, 186, 'expert GPUs, tokens flying between them');
}

export function drawFrame10(svg, p) {
  scene(svg, p, (g) => drawFrame9(g, 1), (g) => {
    pictures(g);
    BILL.forEach((rows, k) => {
      const item = layer(g, seg(p, 0.1 + k * 0.2, 0.3 + k * 0.2));
      rows.forEach((r, i) => note(item, 290 + (i === 0 ? 0 : BULLET), 40 + k * 52 + i * LINE, i === 0 ? `• ${r}` : r, { cls: '' }));
    });
    note(g, LEFT, 316, `DistServe (${PAPERS.distserve.year}): ${formatRatio(PAPERS.distserve.goodputGain)} more requests or ${formatRatio(PAPERS.distserve.sloGain)} tighter targets (the paper's abstract)`);
    note(g, LEFT, 334, `Splitwise (${PAPERS.splitwise.year}): ${formatRatio(PAPERS.splitwise.throughputGain)} throughput at ${PAPERS.splitwise.costCutPct}% lower cost (the paper's abstract)`);
  });
}
