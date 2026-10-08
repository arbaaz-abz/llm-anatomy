// gpu-primer frames 1–5: the GPU (compute next to HBM), its SMs and SRAM, then the toy multiply's bytes, FLOPs and intensity.
// Each draw is a pure function of progress p; frame N at p = 1 is frame N + 1 at p = 0.
import * as G from '@shared/glyphs.js';
import { matmulCost, arithmeticIntensity } from '@math/roofline.js';
import { formatRatio } from '@math/core.js';
import { STAGE_CHIPS, H100_SMS, SRAM_PER_SM, SPARSE_FACTOR, TOKENS, SAT, D_TOY } from './numbers.js';
import { GPU_BIG, GPU_SMALL, CELLS, CELL, LEFT, seg, lerp, arriving, leaving, layer, note, chips, readCell, gpu, gpuParts, hbmFlow } from './stage.js';
import { int } from './format.js';

const H100 = STAGE_CHIPS.h100;
const ALL_SMS = Object.freeze(Array.from({ length: 12 }, (_, i) => i));
const TOY = matmulCost({ m: TOKENS.length, k: D_TOY, n: D_TOY, bytesPerElem: 2 }); // { flops: 512, bytes: 256 }
const BYTES = 2;
const PARTS = Object.freeze({ x: TOKENS.length * D_TOY, w: D_TOY * D_TOY, y: TOKENS.length * D_TOY }); // numbers in X, W_O, Y
const FLOPS_PER_ROW = TOY.flops / TOKENS.length; // 128
const SMALL = 20; // shapes-only cells (README: small grids show no numbers)
const MAT = Object.freeze({ y: 52, x: LEFT, w: 192, yOut: 376 }); // X, W_O, Y anchors (frames 4–6)
const RIGHT = 352; // frames 1–3 right column
const ZOOM_BLOCKS = Object.freeze({ x: 60, w: 260, h: 54, tensor: 92, sram: 164 });
const HBM_BLOCK = Object.freeze({ x: 420, y: 92, w: 150, h: 126 });

function gpuLabels(svg, { hbmText = 1, compute = 1 }) {
  const P = gpuParts(GPU_BIG);
  if (compute > 0) note(layer(svg, compute), P.die.x + P.die.w / 2, 48, 'compute', { anchor: 'middle' });
  if (hbmText > 0) {
    const g = layer(svg, hbmText);
    note(g, RIGHT, 118, 'HBM: high-bandwidth memory');
    note(g, RIGHT, 136, `${H100.hbmGb} GB HBM3 (nominal)`, { cls: '' });
  }
}

// Frame 1: the die, then the HBM stacks, fade in.
export function drawFrame1(svg, p) {
  const die = seg(p, 0, 0.45);
  const hbm = seg(p, 0.45, 0.9);
  gpu(svg, GPU_BIG, { die, hbm });
  gpuLabels(svg, { compute: die, hbmText: hbm });
}

function frame2Marks(svg, t) {
  if (t <= 0) return;
  const g = layer(svg, t);
  note(g, GPU_BIG.x, 14, `${H100_SMS} SMs (streaming multiprocessors), each with tensor cores`);
  note(g, GPU_BIG.x, 30, `${int(H100.peak.bf16)} trillion ops/s (BF16, dense)`, { cls: '' });
  note(g, RIGHT, 172, 'HBM → SMs:');
  note(g, RIGHT, 188, `${H100.bandwidths[0]} TB/s`, { cls: '' });
  note(g, GPU_BIG.x, 300, `dense numbers; vendor sparse figures are ${formatRatio(SPARSE_FACTOR)} and not used`);
}

// Frame 2: the SM grid lights; the HBM → SM arrow draws; both readouts type in.
export function drawFrame2(svg, p) {
  const lit = p >= 0.3;
  gpu(svg, GPU_BIG, { litSms: lit ? ALL_SMS : null });
  gpuLabels(svg, {});
  const arrow = seg(p, 0.3, 0.75);
  if (arrow > 0) hbmFlow(svg, GPU_BIG, { progress: arrow });
  frame2Marks(svg, seg(p, 0.15, 0.4));
}

const lerpBox = (a, b, t) => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t) });

// The zoom frame grows from one SM tile to the panel (t); its blocks fade in only once it has landed (inside), so they never
// overhang a frame still growing.
function zoomPanel(svg, t, inside, sram) {
  const P = gpuParts(GPU_BIG);
  const box = lerpBox(P.tile, GPU_BIG, t);
  const g = G.svgEl('g', { class: 'glyph g-zoom' }, svg);
  G.svgEl('rect', { class: 'g-frame', x: box.x, y: box.y, width: box.w, height: box.h, rx: 4 }, g);
  if (inside <= 0) return;
  const b = layer(svg, inside);
  note(b, GPU_BIG.x + 8, GPU_BIG.y + 20, 'one SM (streaming multiprocessor)');
  G.block(b, { x: ZOOM_BLOCKS.x, y: ZOOM_BLOCKS.tensor, w: ZOOM_BLOCKS.w, h: ZOOM_BLOCKS.h, label: 'tensor cores', state: 'idle' });
  G.block(b, { x: ZOOM_BLOCKS.x, y: ZOOM_BLOCKS.sram, w: ZOOM_BLOCKS.w, h: ZOOM_BLOCKS.h, label: 'SRAM: hundreds of KB, on chip', state: sram ? 'active' : 'idle' });
}

function frame3Marks(svg, t) {
  if (t <= 0) return;
  const g = layer(svg, t);
  G.block(g, { ...HBM_BLOCK, label: `HBM: ${H100.hbmGb} GB, off chip`, state: 'idle' });
  note(g, GPU_BIG.x, 270, `SRAM: ${SRAM_PER_SM} per SM (rough)`);
  note(g, GPU_BIG.x, 286, `HBM: ${H100.hbmGb} GB per GPU`);
}

// Frame 3: zoom from one SM tile to the panel; tensor cores and SRAM appear, then SRAM lights; HBM stays at the edge.
export function drawFrame3(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    gpu(g, GPU_BIG, { litSms: ALL_SMS });
    gpuLabels(g, {});
    hbmFlow(g, GPU_BIG, { progress: 1 });
    frame2Marks(g, 1);
  }
  zoomPanel(svg, seg(p, 0.1, 0.5), seg(p, 0.5, 0.65), p >= 0.75);
  frame3Marks(svg, seg(p, 0.5, 0.75));
}

// ---- frames 4–5: X [4 × 8] · W_O [8 × 8] → Y [4 × 8] ----
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const X_VALUES = zeros(TOKENS.length, D_TOY);
const W_VALUES = zeros(D_TOY, D_TOY);
const rowY = (r) => MAT.y + r * SMALL;

// Phases of frame 4: X in, W_O in, Y's rows computed, Y back out.
const PHASE = Object.freeze({ x: [0.15, 0.3], w: [0.3, 0.45], rows: [0.45, 0.8], back: [0.8, 0.95] });

function frame4State(p) {
  const xIn = seg(p, ...PHASE.x);
  const wIn = seg(p, ...PHASE.w);
  const rowsDone = Math.floor(seg(p, ...PHASE.rows) * TOKENS.length + 1e-9);
  const back = seg(p, ...PHASE.back);
  let bytes = null;
  if (xIn >= 1) bytes = PARTS.x * BYTES;
  if (wIn >= 1) bytes += PARTS.w * BYTES;
  if (back >= 1) bytes += PARTS.y * BYTES;
  return { xIn, wIn, rowsDone, back, bytes, flops: rowsDone ? rowsDone * FLOPS_PER_ROW : null };
}

// X [4 × 8], its "sat" row outlined; dim until it has travelled into the SMs.
export function xMatrix(svg, loaded = true) {
  G.matrix(layer(svg, loaded ? 1 : 0.45), { x: MAT.x, y: MAT.y, values: X_VALUES, cell: SMALL, maxAbs: 1, label: 'X' });
  G.selectionMark(svg, { x: MAT.x, y: rowY(SAT), w: D_TOY * SMALL, h: SMALL });
}

function wAndY(svg, s) {
  G.matrix(layer(svg, s.wIn >= 1 ? 1 : 0.45), { x: MAT.w, y: MAT.y, values: W_VALUES, cell: SMALL, maxAbs: 1, label: 'W_O' });
  const yOut = G.svgEl('g', { class: 'glyph g-matrix g-empty', transform: `translate(${MAT.yOut} ${MAT.y})` }, svg);
  G.svgEl('rect', { class: 'g-frame', x: -1, y: -1, width: D_TOY * SMALL + 2, height: TOKENS.length * SMALL + 2, rx: 3 }, yOut);
  G.svgEl('text', { x: 0, y: -10, class: 'g-label' }, yOut).textContent = `Y [${TOKENS.length} × ${D_TOY}]`;
  for (let r = 0; r < s.rowsDone; r += 1) G.vector(svg, { x: MAT.yOut, y: rowY(r), values: X_VALUES[r], cell: SMALL, orient: 'row', maxAbs: 1 });
  if (s.rowsDone > SAT) G.selectionMark(svg, { x: MAT.yOut, y: rowY(SAT), w: D_TOY * SMALL, h: SMALL });
}

const TRANSFER = Object.freeze({ x: GPU_SMALL.x, y: 292 });
function transfer(svg, s) {
  if (s.xIn > 0 && s.xIn < 1) { hbmFlow(svg, GPU_SMALL, { progress: s.xIn }); note(svg, TRANSFER.x, TRANSFER.y, `HBM → SMs: X, ${PARTS.x} numbers, ${PARTS.x * BYTES} B`); }
  if (s.wIn > 0 && s.wIn < 1) { hbmFlow(svg, GPU_SMALL, { progress: s.wIn, carry: 'weight' }); note(svg, TRANSFER.x, TRANSFER.y, `HBM → SMs: W_O, ${PARTS.w} numbers, ${PARTS.w * BYTES} B`); }
  if (s.back > 0 && s.back < 1) { hbmFlow(svg, GPU_SMALL, { progress: s.back, back: true }); note(svg, TRANSFER.x, TRANSFER.y, `SMs → HBM: Y, ${PARTS.y} numbers, ${PARTS.y * BYTES} B`); }
}

export const BYTES_LINE = `bytes (BF16, ${BYTES} per number): X ${PARTS.x} + W_O ${PARTS.w} + Y ${PARTS.y} = ${PARTS.x + PARTS.w + PARTS.y} numbers → ${TOY.bytes}`;
export const FLOPS_LINE = `FLOPs: ${TOKENS.length} · ${D_TOY} outputs × (${D_TOY} multiplies + ${D_TOY} adds) = ${TOY.flops} · row "sat": ${FLOPS_PER_ROW}`;
export const INTENSITY = arithmeticIntensity(TOY); // 2
export const INTENSITY_LINE = `intensity: ${TOY.flops} ÷ ${TOY.bytes} = ${INTENSITY.toFixed(2)} FLOPs per byte`;
const cellX = (i) => CELLS.x + i * (CELL + CELLS.gap);

// The three readout cells (bytes, FLOPs, FLOPs per byte), typed in; the third is linked to I in the math panel.
export function readCells(svg, { bytes, flops, intensity = undefined }) {
  readCell(svg, { x: cellX(0), y: CELLS.y, value: bytes, label: 'bytes' });
  readCell(svg, { x: cellX(1), y: CELLS.y, value: flops, label: 'FLOPs' });
  if (intensity !== undefined) readCell(svg, { x: cellX(2), y: CELLS.y, value: intensity, label: 'FLOPs per byte', format: (v) => v.toFixed(2), link: 'int' });
}

// What frame 6 lets go of: W_O, Y, the GPU, the transfer and the two count lines.
export function restScene(svg, s) {
  wAndY(svg, s);
  gpu(svg, GPU_SMALL, {});
  transfer(svg, s);
  if (s.flops === TOY.flops) note(svg, LEFT, 312, FLOPS_LINE);
  if (s.bytes === TOY.bytes) note(svg, LEFT, 328, BYTES_LINE);
}

export function toyScene(svg, s) {
  chips(svg);
  xMatrix(svg, s.xIn >= 1);
  restScene(svg, s);
}

const DONE = Object.freeze(frame4State(1));

// Frame 4: back to the whole GPU; X and W_O travel in, Y's rows fill (FLOPs count by row), Y travels back (bytes complete).
export function drawFrame4(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    zoomPanel(g, 1, 1, true);
    frame3Marks(g, 1);
  }
  const g = layer(svg, arriving(p));
  const s = frame4State(p);
  toyScene(g, s);
  readCells(g, s);
}

// Frame 5: the third cell types FLOPs per byte.
export function drawFrame5(svg, p) {
  toyScene(svg, DONE);
  const shown = seg(p, 0.2, 0.5) >= 1;
  readCells(svg, { ...DONE, intensity: shown ? INTENSITY : null });
  if (shown) note(svg, LEFT, 344, INTENSITY_LINE);
}

export { DONE as FRAME4_END, cellX };
