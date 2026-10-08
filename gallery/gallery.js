// Gallery page script: palette, glyph figures, and the control demos.
import * as G from '@shared/glyphs.js';
import { mountStepper } from '@shared/ui/stepper.js';
import { mountThemeToggle } from '@shared/ui/theme-toggle.js';
import { mountSlider } from '@shared/ui/slider.js';
import { mountMathPanel, linkMathToStage } from '@shared/ui/math-panel.js';
import { renderFact } from '@shared/facts.js';
import { loadJSON } from '@shared/data.js';
import { matmul, transpose, softmax, causalMask, randomMatrix, formatBytes, formatCount } from '@math/core.js';
import { TOY as ATTENTION_TOY } from '@math/attention.js';

const $ = (sel) => document.querySelector(sel);
const svg = (parent, w, h, attrs = {}) => G.svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, ...attrs }, parent);

// ---- theme toggle (shared with the lesson header) ----
$('#theme-toggle').replaceWith(mountThemeToggle());

// ---- palette ----
function swatches(root, list) {
  root.replaceChildren(...list.map(([name, note]) => {
    const el = document.createElement('div');
    el.className = 'swatch';
    el.innerHTML = `<i></i><b></b><small></small>`;
    el.querySelector('i').style.background = `var(${name})`;
    el.querySelector('b').textContent = name;
    el.querySelector('small').textContent = note;
    return el;
  }));
}
swatches($('#sw-base'), [['--bg', 'page'], ['--surface', 'panels'], ['--surface-2', 'inset'], ['--ink', 'text, 16:1 / 15:1'], ['--ink-muted', 'notes, 5.5:1 / 7.3:1'], ['--line', 'hairlines']]);
swatches($('#sw-accent'), [['--accent-arch', 'architecture'], ['--accent-train', 'training'], ['--accent-serve', 'serving']]);
swatches($('#sw-sem'), [['--sem-compute', 'compute-bound'], ['--sem-memory', 'memory-bound, KV'], ['--sem-comm', 'communication'], ['--sem-ok', 'pass, hit, useful'], ['--sem-bad', 'fail, miss, wasted'], ['--carry-weight', 'weights in motion (flow dot)']]);
swatches($('#sw-req'), [['--req-1', 'request A'], ['--req-2', 'request B'], ['--req-3', 'request C'], ['--req-4', 'request D']]);
for (let i = -10; i <= 10; i += 1) {
  const cellEl = document.createElement('i');
  cellEl.style.background = G.valueColor(i, 10);
  cellEl.title = `${i / 10}`;
  $('#value-scale').append(cellEl);
}

// ---- glyph figures ----
const TOKENS = ATTENTION_TOY.tokens;
const TOY = ATTENTION_TOY.heads.A; // the one copy of the hand-picked Q, K, V (attention §4)
const S = matmul(TOY.Q, transpose(TOY.K));
const SCALED = S.map((row) => row.map((v) => v / 2));
const MASK = causalMask(4);
const MASKED = SCALED.map((row, i) => row.map((v, j) => (MASK[i][j] ? v : -Infinity)));
const A = MASKED.map((row) => softmax(row));
const O = matmul(A, TOY.V);
const X = randomMatrix(4, 8, 1);
const VISIBLE_SCALED_SUM = SCALED[2].filter((_, j) => MASK[2][j]).reduce((acc, v) => acc + v, 0);
const poolSlots = [...'AAAAAAAA'.split('').map((o) => ({ owner: o, state: 'filled' })),
  ...'BBBBB'.split('').map((o) => ({ owner: o, state: 'filled' })), { owner: 'B', state: 'reserved' }, { owner: 'B', state: 'reserved' }, { owner: 'B', state: 'reserved' },
  ...'CCCCCCCCCC'.split('').map((o) => ({ owner: o, state: 'filled' })), { owner: 'C', state: 'reserved' }, { owner: 'C', state: 'reserved' },
  ...Array.from({ length: 20 }, () => ({ state: 'free' }))];

// ---- Shared prep S3: the Training glyphs and options (numbers from the Training storyboards; schedules and fits illustrative) ----
const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));
const LOSS_FIT = (n) => 1.8 + 400 / n ** 0.34; // illustrative power law, not a published fit
const DECADES = range(7, (i) => 10 ** (8 + i / 2));
const gpipe = (row) => row.map((c, t) => ({ from: t, to: t + 1, kind: c ? (c[0] === 'F' ? 'forward' : 'backward') : 'idle', label: c ?? undefined }));
const TRAINING_FIGURES = [
  ['curvePlot', 'a small line chart: shaded bands with printed labels, a solid series, a muted one (never dashed), a followed marker with the selection mark (illustrative schedule)', 300, 200, (s) => {
    G.curvePlot(s, { x: 0, y: 0, w: 276, h: 196, label: 'learning-rate schedule',
      xAxis: { label: 'step (thousands)', ticks: [0, 20, 40, 60, 80, 100] }, yAxis: { label: 'learning rate (×10⁻⁴)', ticks: [0, 1, 2, 3, 4] },
      bands: [{ from: 0, to: 10, label: 'warm' }, { from: 10, to: 80, label: 'stable' }, { from: 80, to: 100, label: 'decay' }],
      series: [{ points: [[0, 0], [10, 3], [80, 3], [100, 0.3]], label: 'WSD' }, { points: [[60, 3], [80, 0.3]], label: 'branch', style: 'muted' }],
      markers: [{ x: 80, y: 3, label: 'decay starts', followed: true }] });
  }],
  ['curvePlot refY', 'log x axis (each decade the same width), the dashed labeled reference line (drawn like the bars reference), printed marker labels (illustrative fit)', 300, 200, (s) => {
    G.curvePlot(s, { x: 0, y: 0, w: 276, h: 196, label: 'loss against parameters',
      xAxis: { label: 'parameters', log: true, ticks: [{ value: 1e8, label: '100M' }, { value: 1e9, label: '1B' }, { value: 1e10, label: '10B' }, { value: 1e11, label: '100B' }] },
      yAxis: { label: 'loss', ticks: [1.8, 2, 2.2, 2.4, 2.6] },
      series: [{ points: DECADES.map((n) => [n, LOSS_FIT(n)]), label: 'fit' }, { points: DECADES.map((n) => [n, LOSS_FIT(n) - 0.06]), label: '2× data', style: 'muted' }],
      refY: { value: 1.96, label: 'same loss 1.960' },
      markers: [{ x: 1e10, y: LOSS_FIT(1e10), label: `10B · ${LOSS_FIT(1e10).toFixed(3)}`, followed: true }] });
  }],
  ['roofline', 'H100 BF16 from two numbers: the sloped roof (memory) and the flat roof (compute) with both values printed, the ridge printed at the bend, points on the roof (gpu-primer frames 6–8)', 300, 230, (s) => {
    G.roofline(s, { x: 0, y: 0, w: 276, h: 226, peakTflops: 989, bandwidthTBps: 3.35, label: 'H100 BF16',
      points: [{ intensity: 2, label: 'toy', followed: true }, { intensity: 2048, label: '4,096 tokens' }] });
  }],
  ['roofline ridgeRange', 'when sources conflict on bandwidth, the ridge is a band with both ends printed (Rubin NVFP4, gpu-primer frame 11)', 300, 230, (s) => {
    G.roofline(s, { x: 0, y: 0, w: 276, h: 226, peakTflops: 35000, bandwidthTBps: 22, yDomain: [1, 1e5], ridgeRange: [35000 / 22, 35000 / 19.2], label: 'Rubin NVFP4', points: [] });
  }],
  ['laneTimeline', 'lanes on one clock; a segment prints its text inside from 36 px, beside it when shorter; idle hatched (gpu-primer frame 7)', 300, 70, (s) => {
    G.laneTimeline(s, { x: 0, y: 0, w: 280, label: 'one multiply', lanes: [
      { label: 'HBM', segments: [{ from: 0, to: 40.1, kind: 'memory', label: 'memory 40.1 µs' }] },
      { label: 'SMs', segments: [{ from: 0, to: 0.54, kind: 'compute', label: 'compute 0.54 µs' }, { from: 0.54, to: 40.1, kind: 'idle' }] },
    ] });
  }],
  ['laneTimeline schedule', 'forward and backward cells in the carry colors, idle hatched: a GPipe schedule, 2 stages × 2 micro-batches (parallelism frames 6–8)', 300, 70, (s) => {
    G.laneTimeline(s, { x: 0, y: 0, w: 286, label: 'GPipe', lanes: [
      { label: 'stage 1', segments: gpipe(['F1', 'F2', null, null, 'B1', 'B2']) },
      { label: 'stage 2', segments: gpipe([null, 'F1', 'F2', 'B1', 'B2', null]) },
    ] });
  }],
  ['laneTimeline cap, ticks, gaps', 'cap: a lane cut at the track ends in an arrow with its number printed (cluster-topology); ticks for saves, a hatched lost segment and a restart gap (scale-reliability frame 9)', 300, 150, (s) => {
    G.laneTimeline(s, { x: 0, y: 0, w: 280, scale: 0.6, cap: { at: 228 / 0.6, label: 'continues: 2,254%' }, label: 'tensor parallelism over the network', lanes: [
      { label: 'compute', segments: [{ from: 0, to: 100, kind: 'compute', label: '100%' }] },
      { label: 'comm', segments: [{ from: 0, to: 2254, kind: 'comm', label: 'comm' }] },
    ] });
    G.laneTimeline(s, { x: 0, y: 76, w: 280, label: '90 minutes of the run', ticks: [{ t: 13.6, label: 'save' }, { t: 27.2, label: 'save' }, { t: 58, label: 'save' }],
      gaps: [{ from: 40, to: 43, label: 'restart' }], lanes: [{ label: 'run', segments: [
        { from: 0, to: 27.2, kind: 'compute' }, { from: 27.2, to: 40, kind: 'lost', label: 'lost' }, { from: 43, to: 90, kind: 'compute' },
      ] }] });
  }],
  ['bitLayout', 'number formats as bit fields, colored by role (never by value) and labeled S / E / M; a shared block scale with its bracket (gpu-primer frame 10)', 300, 130, (s) => {
    G.bitLayout(s, { x: 56, y: 4, bitW: 11, format: { layout: '1/8/7' }, label: 'BF16' });
    G.bitLayout(s, { x: 56, y: 34, bitW: 11, format: { layout: '1/4/3' }, label: 'FP8 E4M3' });
    const four = G.bitFields('1/2/1');
    G.bitLayout(s, { x: 56, y: 64, bitW: 11, fields: [...four, ...four, ...four, { role: 'scale', bits: 8 }], sharedBy: 16, label: 'NVFP4' });
  }],
  ['gpu showMem, litSms', 'showMem: false drops the memory bar; litSms lights SM tiles and titles the count (20 of 132 SMs drawn as 2 of 12 tiles)', 300, 100, (s) => {
    G.gpu(s, { x: 10, y: 6, showMem: false, label: 'GPU 1' });
    G.gpu(s, { x: 150, y: 6, showMem: false, litSms: [0, 5], label: '20 of 132 SMs: comm' });
  }],
  ['vector fill', 'semantic fills for a pass/fail number (R: --sem-ok for 1, --sem-bad for 0) beside a value-scale column (A) (rlvr-grpo frame 2)', 300, 310, (s) => {
    G.vector(s, { x: 8, y: 18, values: [1, 0, 0, 0, 1, 0, 0, 0], cell: 36, label: 'R', fill: [1, 0, 0, 0, 1, 0, 0, 0].map((r) => (r ? 'ok' : 'bad')) });
    G.vector(s, { x: 56, y: 18, values: [1.73, -0.58, -0.58, -0.58, 1.73, -0.58, -0.58, -0.58], cell: 36, label: 'A', maxAbs: 1.73 });
  }],
  ['matrix shards', 'a matrix cut between GPUs: each GPU\'s columns (or rows) take its tint, shape only (parallelism frames 3–4)', 300, 230, (s) => {
    G.matrix(s, { x: 4, y: 24, values: range(4, () => Array(16).fill(0)), cell: 16, label: 'W_in, by columns', shards: [{ cols: [1, 8], gpu: 1 }, { cols: [9, 16], gpu: 2 }] });
    G.matrix(s, { x: 4, y: 124, values: range(8, () => Array(8).fill(0)), cell: 12, label: 'W_out, by rows', shards: [{ rows: [1, 4], gpu: 1 }, { rows: [5, 8], gpu: 2 }] });
  }],
  ['shareBar unknown', 'value null = not published: neutral, off the scale, never hatched (training-pipeline frame 8); a bar with nothing published prints "no published shares" (Kimi K3)', 300, 130, (s) => {
    G.shareBar(s, { x: 4, y: 6, w: 60, tail: 'none', label: 'GLM-5 tokens', parts: [{ name: 'pretrain', value: 27, hue: 1 }, { name: 'mid-train', value: 1.55, hue: 2 }, { name: 'post-training', value: null }] });
    G.shareBar(s, { x: 4, y: 96, w: 60, label: 'Kimi K3 tokens', parts: [{ name: 'pretrain', value: null }, { name: 'mid-train', value: null }, { name: 'post-training', value: null }] });
  }],
  ['shareBar hatched, tailBasis', 'hatched part = excluded / doesn\'t count (scale-reliability); tailBasis "tail": the zoomed bar\'s shares are of the tail and say so (midtraining frame 6)', 300, 290, (s) => {
    G.shareBar(s, { x: 4, y: 6, w: 280, label: 'run GPU-hours', parts: [{ name: 'useful', value: 26.62, hue: 3 }, { name: 'below peak', value: 4.22, hue: 4 }, { name: 'lost to failures', value: 2.62, hue: 5, hatched: true }] });
    G.shareBar(s, { x: 4, y: 104, w: 280, tailBasis: 'tail', tailLabel: 'last 5%', label: 'GLM-5 by context length', parts: [{ name: '4K', value: 27, hue: 1 }, { name: '32K', value: 1, hue: 2 }, { name: '128K', value: 0.5, hue: 3 }, { name: '200K', value: 0.05, hue: 4 }] });
  }],
];

const FIGURES = [
  ['token', 'idle · active · dim · value fill · hatched (gradient off)', 300, 44, (s) => {
    G.token(s, { x: 4, y: 8, text: 'The', index: 1 });
    G.token(s, { x: 52, y: 8, text: 'cat', index: 2, state: 'active' });
    G.token(s, { x: 100, y: 8, text: 'sat', index: 3, state: 'dim' });
    G.token(s, { x: 150, y: 8, text: '56', fill: G.valueColor(1.73, 2.65) });
    G.token(s, { x: 192, y: 8, text: '48', fill: G.valueColor(-0.58, 2.65), hatched: true });
    G.token(s, { x: 236, y: 8, text: '?', state: 'idle' });
  }],
  ['vector', 'column and row; numbers print in cells of 36 px or more (40 px here), and every cell has its value as a tooltip', 480, 260, (s) => {
    // Laid out to fit a 400 px screen: the 8-cell row (320 px) takes the full width with its label above it.
    G.vector(s, { x: 64, y: 24, values: TOY.Q[2], cell: G.NUMBER_CELL, orient: 'row', label: 'q_sat' });
    G.vector(s, { x: 270, y: 24, values: O[2], cell: G.NUMBER_CELL, label: 'o_sat' });
    G.svgEl('text', { x: 6, y: 196, class: 'g-label' }, G.svgEl('g', { class: 'glyph' }, s)).textContent = 'x_sat'; // glyph label style
    G.vector(s, { x: 4, y: 204, values: X[2], cell: G.NUMBER_CELL, orient: 'row', maxAbs: 1 });
  }],
  ['matrix', 'shape label [rows × cols]; row labels by token', 300, 130, (s) => {
    G.matrix(s, { x: 44, y: 30, values: TOY.K, cell: 22, label: 'K', rowLabels: TOKENS, maxAbs: 3 });
    G.matrix(s, { x: 180, y: 30, values: TOY.V, cell: 22, label: 'V', maxAbs: 3 });
  }],
  ['heatmap', 'rows = queries, columns = keys; masked cells hatched', 300, 140, (s) => {
    G.heatmap(s, { x: 44, y: 38, values: MASKED, mask: MASK, cell: 22, label: 'S / 2', rowLabels: TOKENS, colLabels: TOKENS.map((t) => t[0]), maxAbs: 1.5 });
    G.heatmap(s, { x: 180, y: 38, values: A, cell: 22, label: 'A', maxAbs: 1 });
  }],
  ['heatmap format', 'format: the page picks the printed precision, here weights at 3 d.p. (default: formatCell, 2 d.p.); hatch: a masked weight is 0 and counts for nothing, so it is hatched and still printed', 300, 70, (s) => {
    G.heatmap(s, { x: 60, y: 22, values: [A[2]], cell: G.NUMBER_CELL, label: 'weights', rowLabels: ['sat'], maxAbs: 1, format: (v) => (v === 0 ? '0' : v.toFixed(3)), hatch: [MASK[2].map((visible) => !visible)] });
  }],
  ['block', 'head / expert / layer: active = filled accent, idle = outline, dim = faded (never dashed: dashed means draft)', 300, 50, (s) => {
    G.block(s, { x: 4, y: 5, w: 88, h: 40, label: 'head A', state: 'active' });
    G.block(s, { x: 104, y: 5, w: 88, h: 40, label: 'head B' });
    G.block(s, { x: 204, y: 5, w: 88, h: 40, label: 'W_O [8×8]', state: 'dim' });
  }],
  ['kvStack', 'one K and one V tile per token, growing right; highlight = this step', 300, 50, (s) => {
    G.kvStack(s, { x: 24, y: 20, count: 12, tile: 14, highlight: [11], label: 'KV cache, 12 tokens' });
  }],
  ['gpu', 'die with SM grid, HBM stacks at the side, memory fill bar', 300, 100, (s) => {
    G.gpu(s, { x: 10, y: 6, memFill: 0.3, label: 'H100 · 30 %' });
    G.gpu(s, { x: 130, y: 6, memFill: 0.85, label: 'GB300 · 85 %' });
  }],
  ['rack', 'GPUs in a frame; every link at the default width (thickness never encodes bandwidth: print it, P3-R8); labels number the GPUs', 300, 100, (s) => {
    G.rack(s, { x: 10, y: 6, gpus: 8, label: 'NVLink 450 GB/s', labels: ['1', '2', '3', '4', '5', '6', '7', '8'] });
    G.rack(s, { x: 150, y: 6, gpus: 8, label: 'network 50 GB/s' });
  }],
  ['request', 'prefill segment (compute) then decode ticks (memory)', 300, 70, (s) => {
    G.request(s, { x: 30, y: 8, prefill: 8, decode: 4, label: 'A' });
    G.request(s, { x: 30, y: 28, prefill: 5, decode: 2, label: 'B' });
    G.request(s, { x: 30, y: 48, prefill: 10, decode: 6, label: 'C' });
  }],
  ['flow', 'arrow with a moving dot; dot color = what it carries (weight: gpu-primer, training-memory, parallelism)', 300, 120, (s) => {
    [['activation', 0.25], ['gradient', 0.5], ['kv', 0.75], ['token', 1], ['weight', 0.4]].forEach(([carry, p], i) => {
      const label = G.svgEl('text', { x: 8, y: 16 + i * 22, class: 'g-label' }, G.svgEl('g', { class: 'glyph' }, s));
      label.textContent = carry;
      G.flow(s, { from: [80, 12 + i * 22], to: [290, 12 + i * 22], carry, progress: p });
    });
  }],
  ['verdict', 'checker decision on one answer; aria-label "correct" / "wrong"', 300, 36, (s) => {
    G.token(s, { x: 4, y: 6, text: '7 × 8 = 56' });
    G.verdict(s, { x: 110, y: 18, ok: true });
    G.token(s, { x: 150, y: 6, text: '7 + 8 = 15' });
    G.verdict(s, { x: 256, y: 18, ok: false });
  }],
  ['clipLine', 'number line, shaded safe band, marker at the current ratio', 300, 70, (s) => {
    G.clipLine(s, { x: 30, y: 40, w: 240, lo: 0.6, hi: 1.6, band: [0.8, 1.2], marker: 1.25, label: 'r = π_θ / π_old' });
  }],
  ['blockPool', 'fixed-size KV blocks; filled · reserved (hatched) · free', 310, 150, (s) => {
    G.blockPool(s, { x: 4, y: 4, blocks: 12, blockSize: 4, slots: poolSlots, cell: 14, perRow: 4 });
  }],
  ['blockTable', 'logical → physical lookup, optional ref count', 300, 100, (s) => {
    G.blockTable(s, { x: 10, y: 24, title: 'A', rows: [{ logical: 0, physical: 0 }, { logical: 1, physical: 1 }, { logical: 2, physical: 7 }] });
    G.blockTable(s, { x: 146, y: 24, title: 'D₁ (shared)', rows: [{ logical: 0, physical: 8, ref: 2 }, { logical: 1, physical: 9, ref: 2 }] });
  }],
  ['selectionMark', 'the one selection outline: "the item we follow", in every frame; never an amount', 400, 70, (s) => {
    G.vector(s, { x: 70, y: 14, values: TOY.Q[2], cell: G.NUMBER_CELL, orient: 'row', maxAbs: 3, label: 'q_sat' });
    G.selectionMark(s, { x: 70, y: 14, w: 4 * G.NUMBER_CELL, h: G.NUMBER_CELL });
  }],
  ['patch', 'one image patch as an input piece (4 × 4 crop of a cat\'s ear) · idle · active · dim', 300, 50, (s) => {
    const ear = [0.18, 0.22, 0.78, 0.9, 0.2, 0.55, 0.86, 0.95, 0.42, 0.76, 0.9, 0.84, 0.7, 0.86, 0.8, 0.62];
    G.patch(s, { x: 8, y: 8, pixels: ear, index: 5 });
    G.patch(s, { x: 60, y: 8, pixels: ear, index: 5, state: 'active' });
    G.patch(s, { x: 112, y: 8, pixels: ear, state: 'dim' });
  }],
  ['adder', 'the residual add: a junction on the stream lane, no quantity', 300, 40, (s) => {
    G.adder(s, { x: 20, y: 20 });
  }],
  ['blockStack', '"this block × N" with the residual lane; count is printed, never a height; lastLabel names the last block ("block N")', 300, 200, (s) => {
    G.blockStack(s, { x: 8, y: 8, w: 240, count: 61, active: { block: 1, half: 0 }, lastLabel: 'block N' });
  }],
  ['token draft', 'the draft state: a guess that may still count (dashed, muted), never hatched · idle · active · draft', 300, 50, (s) => {
    G.token(s, { x: 8, y: 10, text: 'on', index: 5 });
    G.token(s, { x: 70, y: 10, text: 'on', index: 5, state: 'active' });
    G.token(s, { x: 132, y: 10, text: 'on', index: 5, state: 'draft' });
  }],
  ['memBar', 'useful · reserved-empty (hatched) · free, with percentages', 300, 60, (s) => {
    G.memBar(s, { x: 4, y: 6, w: 280, useful: 23, reserved: 5, free: 20 });
  }],
  ['dial', 'one pair of numbers as a clock hand: length = the pair\'s size (printed), direction = its turn; pale wedge = angles seen in training; ghost hand = the largest angle reached; past one turn prints "turns"', 300, 200, (s) => {
    G.dial(s, { x: 55, y: 60, vector: [0, 2], angle: 3, scale: 2, seen: [0, 15], label: 'pair 1' }); // rope frames 3 and 8
    G.dial(s, { x: 165, y: 60, vector: [0.5, 0], angle: 0.3, scale: 2, seen: [0, 1.5], reached: [0, 6.3], label: 'pair 2' });
    G.dial(s, { x: 55, y: 172, r: 26, angle: 15, label: 'no vector: unit hand' });
  }],
  ['bars', 'side-by-side amounts (not parts of a whole): the value printed above each bar, a neutral fill, an explicit max so the scale holds between frames, a dashed reference with its own label', 300, 160, (s) => {
    G.bars(s, { x: 4, y: 22, w: 200, h: 100, values: [96, 51, 12, 16, 25, 9, 21, 26], labels: ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'], max: 96, reference: { value: 32, label: 'fair share 32' }, label: 'loads per expert' });
  }],
  ['shareBar', 'categorical parts: five hues; parts under 18 px fold into "others" and a bracketed zoomed bar; "not published" is neutral, off the scale, never hatched', 360, 210, (s) => {
    G.shareBar(s, { x: 4, y: 6, w: 172, label: 'toy model', parts: [
      { name: 'embedding', value: 128, hue: 1 }, { name: 'attention', value: 512, hue: 2 }, { name: 'MLP', value: 768, hue: 3 },
      { name: 'other (norms)', value: 40, hue: 4 }, { name: 'head', value: 128, hue: 5 }, { name: 'not published', value: 60, unknown: true },
    ] });
  }],
  ...TRAINING_FIGURES,
];
$('#figures').replaceChildren(...FIGURES.map(([name, note, w, h, draw]) => {
  const fig = document.createElement('figure');
  if (w > 400) fig.dataset.wide = ''; // an 8-cell row at 40 px needs two columns of the figure grid
  const box = document.createElement('div');
  box.className = 'scroll-x';
  // role="group", not "img": children such as verdict badges carry their own labels.
  const figureSvg = svg(box, w, h, { role: 'group', 'aria-labelledby': `cap-${name.replace(/\W+/g, '-')}` });
  draw(figureSvg);
  G.fitViewBox(figureSvg, 8);
  document.fonts?.ready.then(() => G.fitViewBox(figureSvg, 8));
  const cap = document.createElement('figcaption');
  cap.id = `cap-${name.replace(/\W+/g, '-')}`;
  cap.innerHTML = '<b></b><span></span>';
  cap.querySelector('b').textContent = `${name}()`;
  cap.querySelector('span').textContent = note;
  fig.append(box, cap);
  return fig;
}));

// ---- stepper demo: token → embedding → attention row → weighted sum ----
const lerp = (a, b, t) => a + (b - a) * t;
const STEPS = [
  { caption: 'Four tokens. We follow "sat", token 3.' },
  { caption: 'Each token starts as its embedding, a vector of numbers. Here d_model = 8.' },
  { caption: 'A learned matrix turns the embedding into a query. Its dot product with every key gives one score per token: "cat" scores highest.' },
  { caption: 'Divide by √d_head = 2, mask the future to −∞, softmax. The row now adds to 1: 70 % of "sat" goes to "cat".' },
  { caption: 'Output = weights × values: 70 % of cat\'s value, 20 % of sat\'s, 10 % of The\'s.' },
];
const STAGE_W = 580;
const STAGE_H = 366;
const CELL = G.NUMBER_CELL; // 40: numbered rows
const GRID = 20; // un-numbered K and V matrices
const ROW_X = 236;
const ROWS = { x: 60, q: 118, s: 176, a: 244, o: 316 };
const LEFT = { k: { x: 48, y: 118 }, v: { x: 48, y: 246 } };
const TOKEN_X = [8, 60, 112, 164];
const QUERY_ROW = 2; // "sat": the token we follow in every frame
const WEIGHT_MAX = 1.5; // one scale for the weights row and the "× w" chips, so their colors match
// Selection marks (storyboard rule: outlines and accent marks mean "the one we're following", never a quantity).
function queryLabel(s, x, y) {
  const t = G.svgEl('text', { x, y, class: 'g-sub', 'text-anchor': 'middle' }, G.svgEl('g', { class: 'glyph' }, s));
  t.textContent = 'query';
  t.style.fill = 'var(--accent)';
  return t;
}
function rowMarker(s, x, y) {
  const marker = G.svgEl('polygon', { class: 'glyph g-row-marker', points: `${x},${y - 4} ${x + 6},${y} ${x},${y + 4}` }, s);
  marker.style.fill = 'var(--accent)'; // fills via style, per the glyph CSS contract
  return marker;
}
const easeOut = (p) => 1 - (1 - p) ** 2;
function renderDemo(index, progress, stage) {
  // One <svg> per stage, cleared each frame: the DOM of a frame is then a pure function of (index, progress).
  const s = stage.querySelector('svg') ?? svg(stage, STAGE_W, STAGE_H, { role: 'img' });
  G.hatchFill(s); // claims the svg's hatch id up front, so frames with and without hatching share one <svg> element state
  s.replaceChildren();
  s.setAttribute('aria-label', STEPS[index].caption);
  TOKENS.forEach((t, i) => {
    const visible = index > 0 ? 1 : Math.min(Math.max(progress * 4 - i, 0), 1);
    const g = G.token(s, { x: TOKEN_X[i], y: 12, text: t, index: i + 1, state: i === 2 ? 'active' : 'idle' });
    g.style.opacity = String(visible);
  });
  queryLabel(s, TOKEN_X[2] + 18, 50).style.opacity = String(index > 0 ? 1 : Math.min(Math.max(progress * 4 - 2, 0), 1));
  if (index === 0) return;
  // step 2: the embedding row appears as the dot arrives
  const p1 = index === 1 ? progress : 1;
  const xg = G.vector(s, { x: ROW_X, y: ROWS.x, values: X[2], cell: CELL, orient: 'row', maxAbs: 1, label: 'x_sat [1×8]' });
  xg.dataset.link = 'x';
  xg.style.opacity = String(Math.min(p1 * 2, 1));
  if (index === 1) G.flow(s, { from: [TOKEN_X[2] + 40, 24], to: [150, ROWS.x + 14], carry: 'token', progress: p1 });
  if (index === 1) return;
  // step 3: query, keys, one score per key
  const p2 = index === 2 ? progress : 1;
  const qg = G.vector(s, { x: ROW_X, y: ROWS.q, values: TOY.Q[2], cell: CELL, orient: 'row', maxAbs: 3, label: 'q_sat' });
  qg.dataset.link = 'q';
  qg.style.opacity = String(Math.min(p2 * 3, 1));
  const kg = G.matrix(s, { x: LEFT.k.x, y: LEFT.k.y, values: TOY.K, cell: GRID, label: 'K', rowLabels: TOKENS, maxAbs: 3 });
  kg.dataset.link = 'k';
  rowMarker(s, LEFT.k.x - 44, LEFT.k.y + 2 * GRID + GRID / 2);
  const shown = Math.floor(p2 * 4 + 1e-9);
  const sg = G.vector(s, { x: ROW_X, y: ROWS.s, values: S[2].slice(0, shown), cell: CELL, orient: 'row', maxAbs: 4.5, label: 'q·k' });
  sg.dataset.link = 's';
  if (index === 2 && p2 < 1) G.flow(s, { from: [LEFT.k.x + 94, LEFT.k.y + 10], to: [LEFT.k.x + 94, LEFT.k.y + 80], carry: 'activation', progress: p2 });
  if (index === 2) return;
  // step 4: scale, mask and softmax in place; only the numbers and colors change
  const p3 = index === 3 ? progress : 1;
  const rowVals = MASKED[2].map((v, j) => (j === 3 ? -Infinity : lerp(SCALED[2][j], A[2][j], p3)));
  const ag = G.heatmap(s, { x: ROW_X, y: ROWS.a, values: [rowVals], mask: [[true, true, true, false]], cell: CELL, maxAbs: WEIGHT_MAX, label: p3 < 0.5 ? '÷ 2, masked' : 'weights' });
  ag.dataset.link = 'a';
  const sumT = G.svgEl('text', { x: ROW_X + 4 * CELL + 14, y: ROWS.a + CELL / 2, class: 'g-label', 'dominant-baseline': 'central' }, G.svgEl('g', { class: 'glyph' }, s));
  sumT.textContent = `Σ = ${lerp(VISIBLE_SCALED_SUM, 1, p3).toFixed(2)}`; // 1.25 (scaled row) → 1.00 (softmax)
  if (index === 3) return;
  // step 5: V stays plain; each row's weight is a value-colored "× w" chip (same scale as the weights row);
  // the weighted rows slide together into o_sat, whose cells fill.
  const p4 = progress;
  const vg = G.matrix(s, { x: LEFT.v.x, y: LEFT.v.y, values: TOY.V, cell: GRID, label: 'V', rowLabels: TOKENS, maxAbs: 3 });
  vg.dataset.link = 'v';
  rowMarker(s, LEFT.v.x - 44, LEFT.v.y + QUERY_ROW * GRID + GRID / 2);
  const slide = easeOut(Math.min(p4 / 0.6, 1)); // rows travel during the first 60 % of the step…
  const merge = Math.min(Math.max((p4 - 0.6) / 0.3, 0), 1); // …then dissolve into the output as its cells fill
  const target = { x: ROW_X, y: ROWS.o + (CELL - GRID) / 2 };
  TOY.V.forEach((row, i) => {
    const w = A[QUERY_ROW][i];
    const y = LEFT.v.y + i * GRID;
    const tag = G.token(s, { x: LEFT.v.x + 4 * GRID + 10, y: y + 1, text: w === 0 ? '× 0' : `× ${w.toFixed(2)}`, fill: G.valueColor(w, WEIGHT_MAX), hatched: w === 0 });
    tag.classList.add('g-tag');
    // Chips fit the matrix's row pitch: 18px tall on 20px rows, so they stack without touching.
    tag.querySelectorAll('rect').forEach((r) => { r.setAttribute('height', GRID - 2); r.setAttribute('rx', 5); });
    tag.querySelector('.g-text').setAttribute('y', (GRID - 2) / 2);
  });
  const og = G.vector(s, { x: ROW_X, y: ROWS.o, values: O[QUERY_ROW].map((v) => v * Math.min(Math.max((p4 - 0.5) / 0.5, 0), 1)), cell: CELL, orient: 'row', maxAbs: 1.5, label: 'o_sat' });
  og.dataset.link = 'o';
  // The travelling copies are drawn last, on top of o_sat, so the dissolve (0.6–0.9) stays visible while its cells fill.
  TOY.V.forEach((row, i) => {
    const w = A[QUERY_ROW][i];
    if (w === 0 || p4 === 0 || merge === 1) return;
    const ghost = G.vector(s, { x: lerp(LEFT.v.x, target.x, slide), y: lerp(LEFT.v.y + i * GRID, target.y, slide), values: row, cell: GRID, orient: 'row', maxAbs: 3 });
    ghost.classList.add('g-ghost');
    ghost.style.opacity = String(1 - merge);
  });
}
const stepper = mountStepper($('#stepper-root'), { steps: STEPS, render: renderDemo, label: 'One row of attention' });

// ---- slider demo ----
const BYTES_PER_TOKEN = 2 * 36 * 8 * 64 * 2; // 2 (K,V) · layers · kv heads · head dim · bytes (BF16)
const CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const kvOut = $('#kv-readout');
const showKv = (tokens) => { kvOut.value = formatBytes(tokens * BYTES_PER_TOKEN); };
mountSlider($('#slider-root'), { id: 'ctx', label: 'Context length', values: CONTEXTS, value: 131072, unit: 'tokens', format: formatCount, onInput: showKv });
showKv(131072);

// ---- math panel ----
const math = mountMathPanel($('#math-root'), { summary: 'Show me the math', blocks: [
  { tex: '\\htmlClass{hl-q}{Q} = X\\,W_Q,\\qquad \\htmlClass{hl-k}{K} = X\\,W_K,\\qquad \\htmlClass{hl-v}{V} = X\\,W_V', note: 'X [n × d_model] (4 × 8); W [d_model × d_head] (8 × 4); Q, K, V [n × d_head] (4 × 4).' },
  { tex: '\\htmlClass{hl-a}{A} = \\operatorname{softmax}\\!\\left(\\frac{\\htmlClass{hl-q}{Q}\\,\\htmlClass{hl-k}{K}^{\\top}}{\\sqrt{d_{\\text{head}}}} + M\\right),\\qquad \\htmlClass{hl-o}{O} = \\htmlClass{hl-a}{A}\\,\\htmlClass{hl-v}{V}', note: 'M is 0 where j ≤ i and −∞ where j > i. Worked row: softmax([−0.5, 1.5, 0.25, −∞]) = [0.095, 0.703, 0.202, 0].' },
] });
linkMathToStage($('#math-root'), stepper.stage);

// ---- fact ----
const FACT = { entry: 'deepseek-v4-pro', key: 'kv_bytes_per_token' };
loadJSON('../data/models.json').then((models) => {
  const entry = models.entries.find((e) => e.id === FACT.entry);
  const fact = entry?.facts[FACT.key];
  if (!fact) {
    $('#fact-label').textContent = `fact not found: ${FACT.entry}.${FACT.key} is missing from data/models.json`;
    return;
  }
  $('#fact-label').textContent = `${entry.name} · KV cache per token (as of ${models.as_of})`;
  renderFact($('#fact-root'), fact, (v) => formatBytes(v));
}).catch((error) => {
  console.error('Could not load data/models.json', error);
  $('#fact-label').textContent = 'data/models.json did not load';
});
