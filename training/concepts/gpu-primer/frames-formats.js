// gpu-primer frames 10–11: number formats as bit fields (FP4 needs a shared block scale), then the ridge ladder across chips.
import * as G from '@shared/glyphs.js';
import { FORMATS, bitsPerElement, bytesPerElement, ridgePoint, tokensToComputeBound } from '@math/roofline.js';
import { formatRatio } from '@math/core.js';
import { STAGE_CHIPS, BLOCK_SHOWN, TOKENS, D_REAL } from './numbers.js';
import { CELL, LEFT, LINE, NEUTRAL, PLOT, seg, lerp, arriving, leaving, layer, note, lines, roof, lanes, laneNote, realCost } from './stage.js';
import { FRAME9_END, frame9Left } from './frames-roof.js';
import { int } from './format.js';

const BIT_W = 14;
const BITS = Object.freeze({ x: 100, cellX: 530, headY: 18 });
const ROWS = Object.freeze({ bf16: 40, fp8: 88, nvfp4: 150, mxfp4: 250 });
const OTHERS_X = BITS.x + BLOCK_SHOWN * 4 * BIT_W + 8;
const SCALE_X = Object.freeze({ from: 410, to: 364 });
const NUMBER_FIELDS = Object.freeze(Array.from({ length: BLOCK_SHOWN }, () => G.bitFields(FORMATS.nvfp4.layout)).flat());

function bitsCell(parent, rowY, value) {
  G.cell(parent, { x: BITS.cellX, y: rowY + BIT_W / 2 - CELL / 2, size: CELL, v: value, maxAbs: NEUTRAL, format: String });
}

function plainRows(svg) {
  note(svg, BITS.cellX + CELL, BITS.headY, 'bits per number', { anchor: 'end' });
  G.bitLayout(svg, { x: BITS.x, y: ROWS.bf16, format: FORMATS.bf16, bitW: BIT_W, label: 'BF16' });
  bitsCell(svg, ROWS.bf16, bitsPerElement('bf16'));
  G.bitLayout(svg, { x: BITS.x, y: ROWS.fp8, format: FORMATS.fp8_e4m3, bitW: BIT_W, label: 'FP8 E4M3' });
  bitsCell(svg, ROWS.fp8, bitsPerElement('fp8_e4m3'));
}

// One block-scaled format: three drawn 4-bit numbers, "… N others", and the 8-bit scale sliding in under its bracket.
function blockRow(svg, name, label, t) {
  const f = FORMATS[name];
  const y = ROWS[name];
  G.bitLayout(svg, { x: BITS.x, y, fields: NUMBER_FIELDS, bitW: BIT_W, label });
  note(svg, OTHERS_X, y + BIT_W / 2 + 4, `… ${f.blockSize - BLOCK_SHOWN} others`);
  if (t > 0) G.bitLayout(layer(svg, t), { x: lerp(SCALE_X.from, SCALE_X.to, t), y, fields: [{ role: 'scale', bits: f.scaleBits }], bitW: BIT_W, sharedBy: f.blockSize });
  const landed = t >= 1;
  bitsCell(svg, y, landed ? bitsPerElement(name) : f.bits);
  if (landed) note(svg, BITS.x, y + 64, `${f.bits} + ${f.scaleBits} ÷ ${f.blockSize} = ${bitsPerElement(name)} bits per number`, { cls: '' });
}

function frame10Scene(svg, p) {
  plainRows(svg);
  blockRow(svg, 'nvfp4', 'NVFP4', seg(p, 0.2, 0.55));
  blockRow(svg, 'mxfp4', 'MXFP4', seg(p, 0.55, 0.9));
}

// Frame 10: BF16 and FP8 bit rows; NVFP4 and MXFP4 blocks, each scale sliding into place; bits per number 4 → 4.5 / 4.25.
export function drawFrame10(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    roof(g, FRAME9_END.roof);
    laneNote(g, realCost(TOKENS.length, STAGE_CHIPS.h100.peak.bf16, STAGE_CHIPS.h100.bandwidths[0]).time.memoryS * 1e6);
    lanes(g, { time: FRAME9_END.time, scale: FRAME9_END.scale });
    frame9Left(g);
  }
  frame10Scene(layer(svg, arriving(p)), p);
}

// ---- frame 11: the ridge ladder (its plot spans x 1–100,000 and y 10–100,000: the B300's flat roof needs the room for its
// label beside the ridge's; frame 10 has no plot, so the axes change out of sight) ----
const NV4 = bytesPerElement('nvfp4');
const roofOf = (id, fmt, bw) => ({ peakTflops: STAGE_CHIPS[id].peak[fmt], bandwidthTBps: bw });
const bytesOf = { bf16: 2, fp4: NV4 };
function ladderRow(id, fmt, label) {
  const ends = STAGE_CHIPS[id].bandwidths.map((bw) => {
    const r = roofOf(id, fmt, bw);
    return { ridge: ridgePoint(r), cross: tokensToComputeBound({ ...r, bytesPerElem: bytesOf[fmt], k: D_REAL, n: D_REAL }) };
  }).sort((a, b) => a.ridge - b.ridge);
  const span = (k) => [...new Set(ends.map((e) => int(e[k])))].join('–');
  return { label, ridge: span('ridge'), cross: span('cross') };
}
export const LADDER = Object.freeze([
  ladderRow('h100', 'bf16', 'H100 BF16'), ladderRow('b200', 'bf16', 'B200 BF16'), ladderRow('b300', 'bf16', 'B300 BF16'),
  ladderRow('b300', 'fp4', 'B300 NVFP4'), ladderRow('rubin', 'fp4', 'Rubin NVFP4'),
]);
export const SMALL_ROWS = Object.freeze([ladderRow('h200', 'bf16', 'H200 BF16'), ladderRow('b200', 'fp4', 'B200 NVFP4')]);
const LADDER_X = Object.freeze({ ridge: 160, cross: 212, top: 40 }); // right-aligned columns, clear of the plot's tick labels
const B300 = STAGE_CHIPS.b300;
const B300_4 = realCost(TOKENS.length, B300.peak.fp4, B300.bandwidths[0], NV4);
export const ROOF11 = Object.freeze({ ...roofOf('b300', 'fp4', B300.bandwidths[0]), title: 'B300 · NVFP4', xDomain: [1, 1e5], yDomain: [10, 1e5], points: [{ intensity: B300_4.intensity, label: '4 tokens', followed: true }] });

function ladder(svg, shown) {
  note(svg, LADDER_X.ridge, 20, 'ridge', { anchor: 'end' });
  note(svg, LADDER_X.cross, 20, 'tokens', { anchor: 'end' });
  LADDER.slice(0, shown).forEach((r, i) => {
    const y = LADDER_X.top + i * (LINE + 4);
    note(svg, LEFT, y, r.label, { cls: '' });
    note(svg, LADDER_X.ridge, y, r.ridge, { cls: '', anchor: 'end' });
    note(svg, LADDER_X.cross, y, r.cross, { cls: '', anchor: 'end' });
  });
  if (shown >= LADDER.length) lines(svg, LEFT, LADDER_X.top + LADDER.length * (LINE + 4) + 8, ['ridge: FLOPs per byte;', 'tokens: to cross the ridge'], { cls: 'g-label' });
}

const FP4_X = formatRatio(B300.peak.fp4 / B300.peak.bf16);
const BYTES_X = formatRatio(2 / NV4);
function bottomRows(svg) {
  const [h200, b200] = SMALL_ROWS;
  lines(svg, LEFT, 262, [
    `${h200.label} ${h200.ridge} / ${h200.cross}: a bandwidth refresh lowers the crossing`,
    `${b200.label} ${b200.ridge} / ${b200.cross}: on a B200 the FP4 crossing barely moves`,
  ], { cls: 'g-label' });
  lines(svg, LEFT, 304, [
    `B300: FP4 ${int(B300.peak.fp4)} vs BF16 ${int(B300.peak.bf16)} TFLOPS (${FP4_X});`,
    `NVFP4 bytes ${NV4} vs 2 per number (${BYTES_X})`,
  ]);
  note(svg, LEFT, 344, `Rubin bandwidth: sources conflict, ${[...STAGE_CHIPS.rubin.bandwidths].sort((a, b) => a - b).join(' or ')} TB/s`, { cls: 'g-label' });
}

// Frame 11: the ladder rows type in top to bottom; the B300 NVFP4 roofline appears with the four tokens' dot.
export function drawFrame11(svg, p) {
  const out = leaving(p);
  if (out > 0) frame10Scene(layer(svg, out), 1);
  const shown = Math.floor(seg(p, 0.15, 0.75) * LADDER.length + 1e-9);
  const g = layer(svg, arriving(p));
  ladder(g, shown);
  roof(svg, { ...ROOF11, opacity: seg(p, 0.6, 0.8) });
  if (p >= 0.8) bottomRows(svg);
}

export { PLOT };
