// quantization frames 8–11: what fewer bits do to a GPU (storyboard §5). Every number is computed from math/serving.js and
// math/roofline.js on the stage constants (numbers.js, pinned to data/*.json by the page test).
import * as G from '@shared/glyphs.js';
import { bitsPerElement } from '@math/roofline.js';
import { formatBytes, formatInt, formatDuration } from '@math/core.js';
import { CONTEXT, PREFILL_TOKENS, VLLM, MODEL_CARDS } from './numbers.js';
import { memoryPlan, kvBytesFor, shrink, bitsText } from './format.js';
import { stagePreset } from './hardware.js';
import { COUNT_FORMAT } from './model-text.js';
import { LEFT, LINE, seg, arriving, leaving, layer, note, lines } from './stage.js';

const H200 = stagePreset('h200');
const B200 = stagePreset('b200');
const PART = Object.freeze({ weights: 1, kv: 2, free: 3 });

// ---- frame 8: formats on the H200's memory, one bar that swaps, and the four formats in a table ----
const STORAGE = Object.freeze([
  { label: 'BF16', key: 'bf16', at: 0 }, { label: 'FP8', key: 'fp8_e4m3', at: 0.33 }, { label: 'MXFP4', key: 'mxfp4', at: -1 }, { label: 'NVFP4', key: 'nvfp4', at: 0.66 },
]);
const SWAPS = Object.freeze(STORAGE.filter((f) => f.at >= 0)); // the bar steps BF16 → FP8 → NVFP4; MXFP4 sits in the table
const BAR8 = Object.freeze({ x: LEFT, y: 104, w: 300, table: 292 });

const usersText = (n) => `${formatInt(n)} user${n === 1 ? '' : 's'}`;
const planFor = (f, preset = H200) => memoryPlan({ bits: bitsPerElement(f.key), hbmBytes: preset.hbmBytes, kvBytesPerToken: kvBytesFor('bf16') });

// weights / the KV caches of the users that fit / what is left, on one chip.
function memoryParts(plan, hbmBytes) {
  const kv = plan.users * plan.kvPerUser;
  return [
    { name: 'weights', value: plan.weights, hue: PART.weights },
    { name: 'KV cache', value: kv, hue: PART.kv },
    { name: 'free', value: Math.max(hbmBytes - plan.weights - kv, 0), hue: PART.free },
  ];
}

// One memory bar with its header; parts under 2 px fold into the zoomed bar below it (the glyph's rule), and the KV note
// ("KV 671 MB (1 user)") prints beside that zoomed bar, so a sliver is never the only place its size is written.
function memoryBar(parent, { x, y, w, header, plan, preset, kvNote, noteY, freeNote = null }) {
  note(parent, x, y - 8, header, { cls: '' });
  G.shareBar(parent, { x, y, w, parts: memoryParts(plan, preset.hbmBytes), label: `${header}: memory` });
  if (kvNote) note(parent, noteY.x, noteY.y, kvNote);
  if (freeNote) note(parent, freeNote.x, freeNote.y, freeNote.text);
}

function drawBar8(parent, f) {
  const plan = planFor(f);
  const kv = plan.users * plan.kvPerUser;
  // Only when the KV part folds into the zoomed bar does the note belong beside that bar; otherwise the zoom holds the free
  // sliver, so the KV size sits on the legend's "KV cache" row and the zoom says what it holds.
  const kvFolded = (kv / H200.hbmBytes) * BAR8.w < 2;
  const free = Math.max(H200.hbmBytes - plan.weights - kv, 0);
  const zoomY = BAR8.y + 14 + 40 + 11;
  memoryBar(parent, {
    x: BAR8.x, y: BAR8.y, w: BAR8.w, plan, preset: H200,
    header: `${f.label} · ${bitsText(bitsPerElement(f.key))} bits per weight · weights ${formatBytes(plan.weights)}`,
    kvNote: `KV ${formatBytes(kv)} (${usersText(plan.users)})`,
    noteY: kvFolded ? { x: BAR8.x + BAR8.w + 12, y: zoomY } : { x: BAR8.x + 130, y: BAR8.y + 14 + 40 + 14 + 32 + 15 },
    freeNote: kvFolded ? null : { text: `free ${formatBytes(free)}`, x: BAR8.x + BAR8.w + 12, y: zoomY },
  });
}

const tableRow = (f) => {
  const plan = planFor(f);
  return `${f.label.padEnd(6)} ${bitsText(bitsPerElement(f.key)).padStart(4)} bits  ${formatBytes(plan.weights).padStart(10)}  ${usersText(plan.users)}`;
};

function drawFrame8Scene(parent, p) {
  G.gpu(parent, { x: LEFT, y: 6, w: 84, h: 56, showMem: false, label: H200.label });
  lines(parent, 108, 24, [`${H200.label} · ${H200.hbmGb} GB of HBM, ${H200.basis}`], { cls: '' });
  lines(parent, 108, 44, [`${H200.label} has no FP4 tensor cores:`, '4-bit weights are expanded before the math'], { cls: 'g-label' });
  const shown = SWAPS.filter((f) => p >= f.at && p >= 0).at(-1) ?? SWAPS[0];
  drawBar8(parent, p >= 1 ? SWAPS.at(-1) : shown);
  note(parent, LEFT, BAR8.table - 18, 'format  bits         weights  users at 2,048 tokens', { pre: true });
  STORAGE.forEach((f, i) => {
    if (p >= Math.max(f.at, 0.05) + (f.at < 0 ? 0.5 : 0)) note(parent, LEFT, BAR8.table + i * LINE, tableRow(f), { cls: '', pre: true });
  });
}

export function drawFrame8(svg, p, drawPrevEnd) {
  if (leaving(p) > 0) drawPrevEnd(layer(svg, leaving(p)));
  drawFrame8Scene(layer(svg, arriving(p)), p);
}
export const drawFrame8End = (parent) => drawFrame8Scene(parent, 1);

// ---- frame 9: decode and prefill step bars on the H200 ----
const STEP = Object.freeze({ cols: [8, 300], w: 190, h: 12, top: 22, pitch: 100, scaleS: [0.03, 0.6] });
const GROUPS = Object.freeze([
  { label: 'BF16', modelFormat: 'bf16' }, { label: 'FP8 weights, FP8 math', modelFormat: 'fp8' }, { label: '4-bit weights, 16-bit math', modelFormat: 'w4a16' },
]);

// The parts of a step's memory reading (each a time) over its arithmetic, from one stepTime result.
function stepParts(step, preset) {
  const perByte = 1 / (preset.bandwidthTBps * 1e12);
  const reading = [
    { label: 'weights read', s: (step.bytes - step.kvBytes - step.actBytes) * perByte },
    { label: 'KV read and activations', s: (step.kvBytes + step.actBytes) * perByte },
  ].filter((q) => q.s > 0);
  return { reading, mathS: step.computeS };
}

function stepGroup(parent, g, k) {
  const m = shrink({ modelFormat: g.modelFormat, kv: 'bf16' }, H200);
  [m.decode, m.prefill].forEach((step, col) => {
    const y = STEP.top + k * STEP.pitch;
    note(parent, STEP.cols[col], y + 10, g.label);
    G.stepBar(parent, { x: STEP.cols[col], y: y + 16, w: STEP.w, h: STEP.h, scaleS: STEP.scaleS[col], ...stepParts(step, H200), label: `${g.label}, ${col ? 'prefill' : 'decode'}` });
  });
}

function drawFrame9Scene(parent, p) {
  note(parent, STEP.cols[0], 14, `Decode, 1 user, ${formatInt(CONTEXT)} tokens of context`, { cls: '' });
  note(parent, STEP.cols[1], 14, `Prefill, ${formatInt(PREFILL_TOKENS)} tokens`, { cls: '' });
  GROUPS.forEach((g, k) => {
    const t = seg(p, 0.1 + k * 0.25, 0.4 + k * 0.25);
    if (t > 0) stepGroup(layer(parent, t), g, k);
  });
  const fp4 = shrink({ modelFormat: 'nvfp4', kv: 'bf16' }, B200);
  lines(parent, LEFT, 340, [
    'decode and prefill rows use different time scales: compare within a row',
    `each bar is one forward pass on one GPU · B200 native FP4: prefill ${formatDuration(fp4.prefill.timeS)}`,
  ], { cls: 'g-label' });
}

export function drawFrame9(svg, p) {
  if (leaving(p) > 0) drawFrame8End(layer(svg, leaving(p)));
  drawFrame9Scene(layer(svg, arriving(p)), p);
}
export const drawFrame9End = (parent) => drawFrame9Scene(parent, 1);

// ---- frame 10: the KV cache in FP8, and the needle test ----
const NEEDLE = Object.freeze({ y: 8, cell: 28, at: 8, count: 12 });
const BAR10 = Object.freeze({ x: 250, y: 84, w: 300 });

function needleRow(parent) {
  const tokenW = G.tokenWidth('needle');
  let x = LEFT;
  for (let i = 0; i < NEEDLE.count; i += 1) {
    if (i === NEEDLE.at) {
      G.token(parent, { x, y: NEEDLE.y - 2, text: 'needle' });
      G.selectionMark(parent, { x, y: NEEDLE.y - 2, w: tokenW, h: 24 });
      x += tokenW + 4;
    } else {
      G.cell(parent, { x, y: NEEDLE.y - 6, size: NEEDLE.cell, v: 0, maxAbs: 1e15, format: String });
      x += NEEDLE.cell;
    }
  }
  note(parent, x + 4, NEEDLE.y + 12, '128K tokens of context');
}

function drawFrame10Scene(parent, p) {
  const bf = shrink({ modelFormat: 'fp8', kv: 'bf16' }, H200);
  const fp = shrink({ modelFormat: 'fp8', kv: 'fp8' }, H200);
  needleRow(parent);
  const half = seg(p, 0.05, 0.4);
  if (half > 0) {
    // The same users with each cache halved: the plan keeps the BF16 user count, the cache per user is the FP8 one.
    const g = layer(parent, half);
    const halved = { ...bf, kvPerUser: fp.kvPerUser };
    memoryBar(g, {
      x: BAR10.x, y: BAR10.y, w: BAR10.w, preset: H200, plan: halved, header: `the same ${formatInt(bf.users)} users, KV in FP8`,
      kvNote: `KV ${formatBytes(bf.users * halved.kvPerUser)} (${usersText(bf.users)})`, noteY: { x: BAR10.x + 130, y: BAR10.y + 14 + 32 + 15 },
    });
    lines(g, BAR10.x, 196, [`KV per user: ${formatBytes(bf.kvPerUser)} → ${formatBytes(fp.kvPerUser)}`, `users that fit: ${formatInt(bf.users)} → ${formatInt(fp.users)}`], { cls: '' });
  }
  const chart = seg(p, 0.4, 0.7);
  if (chart > 0) {
    const g = layer(parent, chart);
    note(g, 24, 70, 'needle recall at 128K tokens', { cls: '' });
    G.bars(g, { x: 24, y: 84, w: 160, h: 100, values: [VLLM.baselinePct, VLLM.naivePct], labels: ['BF16 KV', 'naive FP8 KV'], max: 100, format: (v) => `${v}%`, label: 'needle-in-a-haystack recall' });
  }
  const fixed = seg(p, 0.7, 1);
  if (fixed > 0) lines(layer(parent, fixed), 24, 226, ['fixed:', `${VLLM.recoveryPct.join('–')}% of baseline AUC`], { cls: '' });
  lines(parent, LEFT, 336, ['test: Llama-3.3-70B-Instruct, 128K-token needle-in-a-haystack (vLLM)', 'bars show recall; the fix is reported as a share of the baseline AUC'], { cls: 'g-label' });
}

export function drawFrame10(svg, p) {
  if (leaving(p) > 0) drawFrame9End(layer(svg, leaving(p)));
  drawFrame10Scene(layer(svg, arriving(p)), p);
}
export const drawFrame10End = (parent) => drawFrame10Scene(parent, 1);

// ---- frame 11: models that ship 4-bit weights ----
const CARD = Object.freeze({ w: 176, h: 36, y: 70, xs: [8, 202, 396] });

function cards() {
  const { gptOss, v4Pro } = MODEL_CARDS;
  return [
    { name: 'gpt-oss-120b', lines: [`${COUNT_FORMAT.total(gptOss.totalParams)} parameters`, 'MXFP4 MoE weights', `fits one ${gptOss.fitsGpuGb} GB GPU`] },
    { name: 'DeepSeek-V4-Pro', lines: ['FP4 experts, FP8 the rest', `checkpoint about ${v4Pro.checkpointGb} GB`, '(size reported)'] },
    { name: 'Kimi K2.x', lines: ['INT4 weights', 'quantization-aware', '(reported)'] },
  ];
}

export function drawFrame11(svg, p) {
  if (leaving(p) > 0) drawFrame10End(layer(svg, leaving(p)));
  cards().forEach((c, i) => {
    const t = seg(p, 0.1 + i * 0.25, 0.4 + i * 0.25);
    if (t <= 0) return;
    const g = layer(svg, t);
    G.block(g, { x: CARD.xs[i], y: CARD.y, w: CARD.w, h: CARD.h, label: c.name });
    lines(g, CARD.xs[i] + 4, CARD.y + CARD.h + 22, c.lines, { cls: '' });
  });
}
