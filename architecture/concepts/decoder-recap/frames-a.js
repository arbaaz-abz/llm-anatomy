// decoder-recap panels 1–4: GPT-3 in numbers, LayerNorm against RMSNorm, the RoPE dial, the gated MLP.
import * as G from '@shared/glyphs.js';
import { PANEL, ROWS_X, CELL, seg, layer, note, lines, linkedRow, typed } from './stage.js';
import { GPT3_FACTS, MODERN_FACTS, X_SAT, ROW_NORMS } from './numbers.js';
import { int, norm2, plain } from './format.js';

const MAX_ABS = 2;
const NOTE_X = PANEL.x + 6;

// Panel 1: GPT-3 in numbers (the reference everything else is compared with).
export function panel1(g, p) {
  const o = layer(g, seg(p, 0.5, 0.9));
  note(o, NOTE_X, 60, 'GPT-3 (2020), the reference', { cls: '' });
  lines(o, NOTE_X, 84, [
    `${GPT3_FACTS.layers} blocks`, `d_model ${int(GPT3_FACTS.dModel)}`, `${GPT3_FACTS.totalText} parameters`,
    `${int(GPT3_FACTS.cacheBytes)} B (${GPT3_FACTS.cacheText}) of cache per token`,
  ]);
}

// Panel 2: the row "sat" through LayerNorm and RMSNorm (NUMBER_CELL rows across the right 360 px).
const ROW2 = Object.freeze({ input: 78, layer: 158, rms: 266 });
export function panel2(g, p) {
  note(g, ROWS_X, 22, "why normalize: keeps the row's size steady");
  note(g, ROWS_X, 36, 'as the stack gets deep');
  note(g, ROWS_X, ROW2.input - 8, 'x_sat, the row going in');
  G.vector(g, { x: ROWS_X, y: ROW2.input, values: X_SAT, cell: CELL, orient: 'row', maxAbs: MAX_ABS, format: plain });
  note(g, ROWS_X, ROW2.layer - 8, 'LayerNorm');
  G.vector(g, { x: ROWS_X, y: ROW2.layer, values: typed(ROW_NORMS.layer, seg(p, 0.2, 0.5)), cell: CELL, orient: 'row', maxAbs: MAX_ABS, format: norm2 });
  lines(layer(g, seg(p, 0.45, 0.55)), ROWS_X, ROW2.layer + CELL + 15, [`subtract the mean ${ROW_NORMS.mean},`, `divide by the spread ${ROW_NORMS.spread.toFixed(3)}`]);
  note(g, ROWS_X, ROW2.rms - 8, 'RMSNorm');
  linkedRow(g, 'rms', { x: ROWS_X, y: ROW2.rms, values: typed(ROW_NORMS.rms, seg(p, 0.5, 0.8)), maxAbs: MAX_ABS, format: norm2 });
  lines(layer(g, seg(p, 0.78, 0.9)), ROWS_X, ROW2.rms + CELL + 15, [`divide by the root mean square ${ROW_NORMS.rootMeanSquare.toFixed(3)};`, 'zeros stay zero']);
}

// Panel 3: the position table is gone; a dial turns once inside attention.
const DIAL = Object.freeze({ x: 330, y: 150, r: 42 });
export function panel3(g, p) {
  note(g, NOTE_X, 22, 'RoPE turns q and k by their position');
  const turn = seg(p, 0.3, 0.9) * 2 * Math.PI;
  G.dial(layer(g, seg(p, 0.15, 0.3)), { ...DIAL, vector: [0, 2], angle: turn, label: 'one pair of q' });
  lines(layer(g, seg(p, 0.5, 0.7)), 400, 130, ['why: depends on distance,', `no ${int(GPT3_FACTS.positions)} limit`]);
  lines(layer(g, seg(p, 0.7, 1)), NOTE_X, 250, [
    `GPT-3's table: ${int(GPT3_FACTS.positions)} × ${int(GPT3_FACTS.dModel)}`, `= ${int(GPT3_FACTS.positionTable)} parameters,`,
    `and nothing for position ${int(GPT3_FACTS.positions + 1)}`, 'RoPE: 0 parameters',
  ]);
}

// Panel 4: GELU's two matrices become SwiGLU's three, with a gate that multiplies two branches.
const SW = Object.freeze({ top: 30, wOut: 72, title2: 176, branch: 184, out: 222 });
export function panel4(g, p) {
  const geluHidden = int(GPT3_FACTS.geluHidden);
  note(g, NOTE_X, SW.top - 8, 'GELU MLP (GPT-3)', { cls: '' });
  G.block(g, { x: NOTE_X, y: SW.top, w: 150, h: 22, label: 'W_in [d × 4d]' });
  note(g, NOTE_X + 60, SW.wOut - 7, 'GELU', { anchor: 'middle' });
  G.block(g, { x: NOTE_X, y: SW.wOut, w: 150, h: 22, label: 'W_out [4d × d]' });
  lines(g, NOTE_X, 118, [`GELU: 2 × ${int(GPT3_FACTS.dModel)} × ${geluHidden}`, `= ${int(2 * GPT3_FACTS.dModel * GPT3_FACTS.geluHidden)} parameters`]);
  const hidden = int(MODERN_FACTS.swigluHidden);
  note(layer(g, seg(p, 0.1, 0.3)), NOTE_X, SW.title2 - 8, 'SwiGLU MLP (2026)', { cls: '' });
  const branches = G.svgEl('g', { 'data-link': 'glu' }, g);
  G.block(layer(branches, seg(p, 0.1, 0.3)), { x: NOTE_X, y: SW.branch, w: 130, h: 22, label: 'W_in [d × 8/3 d]' });
  const gate = layer(branches, seg(p, 0.3, 0.6));
  note(gate, NOTE_X + 130 + 14, SW.branch + 16, '×', { anchor: 'middle', cls: '' });
  G.block(gate, { x: NOTE_X + 130 + 28, y: SW.branch, w: 130, h: 22, label: 'W_gate [d × 8/3 d]' });
  G.block(layer(g, seg(p, 0.55, 0.75)), { x: NOTE_X + 40, y: SW.out, w: 170, h: 22, label: 'W_out [8/3 d × d]' });
  lines(layer(g, seg(p, 0.75, 1)), NOTE_X, 268, [`SwiGLU: 3 × ${int(GPT3_FACTS.dModel)} × ${hidden}`, `= ${int(3 * GPT3_FACTS.dModel * MODERN_FACTS.swigluHidden)} parameters (no biases)`]);
}
