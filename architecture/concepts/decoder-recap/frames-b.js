// decoder-recap panels 5–7: the router and 64 experts, 96 KV tiles collapsing to 8, QK-norm on the "sat" row.
import * as G from '@shared/glyphs.js';
import { PANEL, CELL, seg, lerp, layer, note, lines, tile, linkedRow } from './stage.js';
import { GPT3_FACTS, MODERN_FACTS, LIT_EXPERTS, KV_FORMULA, Q_SAT, KEYS, KEY_NAMES, Q_MULTIPLES, QK } from './numbers.js';
import { fixed3 } from './format.js';

const NOTE_X = PANEL.x + 6;

// Panel 5: the MLP box becomes a router and 64 experts, 8 lit (illustrative: not a real model).
const GRID = Object.freeze({ x: NOTE_X, y: 70, stride: 16, tile: 14, side: 8 });
export function panel5(g, p) {
  note(g, NOTE_X, 22, 'most layers in 2026 models', { cls: '' });
  G.block(g, { x: GRID.x, y: 32, w: GRID.side * GRID.stride - 2, h: 24, label: 'router' });
  Array.from({ length: GRID.side * GRID.side }, (_, i) => {
    const order = LIT_EXPERTS.indexOf(i);
    const lit = order >= 0 && seg(p, 0.3 + order * 0.04, 0.4 + order * 0.04) >= 0.5;
    return tile(g, { x: GRID.x + (i % GRID.side) * GRID.stride, y: GRID.y + Math.floor(i / GRID.side) * GRID.stride, w: GRID.tile, h: GRID.tile, state: lit ? 'active' : 'idle' });
  });
  note(g, NOTE_X, GRID.y + GRID.side * GRID.stride + 14, `${MODERN_FACTS.experts} experts, ${MODERN_FACTS.topK} used per token`);
  const out = layer(g, seg(p, 0.7, 0.9));
  lines(out, 412, 70, [`stored: ${MODERN_FACTS.stored}`, `active: ${MODERN_FACTS.active}`, `dense version: ${MODERN_FACTS.dense}`, 'active = dense', 'plus the router']);
  lines(out, 412, 150, ['illustrative:', 'not a real model']);
}

// Panel 6: 96 key/value tiles in 8 groups of 12 query heads merge to 8 shared tiles.
const KV = Object.freeze({ x: NOTE_X + 6, y: 56, colGap: 150, rowGap: 46, tile: 6, step: 8, merged: 14 });
function kvGroup(g, index, t) {
  const gx = KV.x + (index % 2) * KV.colGap;
  const gy = KV.y + Math.floor(index / 2) * KV.rowGap;
  const heads = MODERN_FACTS.group;
  note(g, gx, gy - 6, `heads ${index * heads + 1}–${(index + 1) * heads}`);
  const body = G.svgEl('g', { class: 'glyph g-kv', role: 'img', 'aria-label': `${heads} key-value tiles, shared by query heads ${index * heads + 1} to ${(index + 1) * heads}` }, g);
  Array.from({ length: heads }, (_, i) => {
    const w = i === 0 ? lerp(KV.tile, KV.merged, t) : KV.tile;
    const o = layer(body, i === 0 ? 1 : 1 - t);
    const x = gx + lerp(i * KV.step, 0, t);
    G.svgEl('rect', { class: 'g-k', x, y: gy, width: w, height: KV.tile, rx: 1.5 }, o);
    G.svgEl('rect', { class: 'g-v', x, y: gy + KV.tile + 2, width: w, height: KV.tile, rx: 1.5 }, o);
  });
}
export function panel6(g, p) {
  lines(g, NOTE_X, 18, ['one K, V tile per head,', 'grouped by who will share it']);
  Array.from({ length: MODERN_FACTS.sharedKv }, (_, k) => kvGroup(g, k, seg(p, 0.25, 0.65)));
  const out = layer(g, seg(p, 0.7, 0.9));
  lines(out, NOTE_X, 262, ['cache per token', `${KV_FORMULA(GPT3_FACTS.layers).split(' = ')[1]} → ${KV_FORMULA(MODERN_FACTS.sharedKv).split(' = ')[1]}`, `(${GPT3_FACTS.cacheText} → ${MODERN_FACTS.cacheText})`, 'or a small latent (MLA)']);
}

// Panel 7: three weight rows for "sat" against The, cat, sat: plain, q grown 10×, and with QK-norm.
const Q7 = Object.freeze({ x: 322, y: 34, keys: 96, keyCell: 18, keyGap: 12, rows: [182, 230, 278], rowsX: 340 });
const MAX_Q = Q_MULTIPLES[2] * Math.max(...Q_SAT.map(Math.abs)); // the value scale holds q × 100
const fmtQ = (v) => (Number.isInteger(v) ? String(v) : (Math.abs(v) >= 10 ? String(Math.round(v)) : v.toFixed(2)));

function queryRow(g, p) {
  const grow = seg(p, 0.15, 0.4);
  const again = seg(p, 0.75, 0.95);
  const factor = again > 0 ? lerp(Q_MULTIPLES[1], Q_MULTIPLES[2], again) : lerp(Q_MULTIPLES[0], Q_MULTIPLES[1], grow);
  note(g, Q7.x - 8, Q7.y + 25, 'q_sat', { anchor: 'end' });
  G.vector(g, { x: Q7.x, y: Q7.y, values: Q_SAT.map((v) => v * factor), cell: CELL, orient: 'row', maxAbs: MAX_Q, format: fmtQ });
  const shown = again > 0 ? (again >= 0.5 ? Q_MULTIPLES[2] : Q_MULTIPLES[1]) : (grow >= 0.5 ? Q_MULTIPLES[1] : Q_MULTIPLES[0]);
  note(g, Q7.x + 4 * CELL + 8, Q7.y + 25, `q × ${shown}`);
}

function keyRows(g) {
  KEYS.forEach((key, i) => {
    const x = Q7.x + i * (4 * Q7.keyCell + Q7.keyGap);
    note(g, x, Q7.y + CELL + 24, KEY_NAMES[i]);
    G.vector(g, { x, y: Q7.y + CELL + 28, values: key, cell: Q7.keyCell, orient: 'row', maxAbs: 2 });
  });
}

function weightRow(g, i, name, values, opacity, link = null) {
  const o = layer(g, opacity);
  const y = Q7.rows[i];
  note(o, Q7.rowsX - 8, y + 25, name, { anchor: 'end' });
  if (link) linkedRow(o, link, { x: Q7.rowsX, y, values, maxAbs: 1, format: fixed3 });
  else G.vector(o, { x: Q7.rowsX, y, values, cell: CELL, orient: 'row', maxAbs: 1, format: fixed3 });
}

export function panel7(g, p) {
  note(g, NOTE_X, 20, 'q_sat against the keys The, cat, sat');
  const ring = seg(p, 0.5, 0.6);
  if (ring > 0) {
    const o = layer(g, ring);
    tile(o, { x: Q7.x - 6, y: Q7.y - 6, w: 252, h: CELL + 6 + 28 + Q7.keyCell + 8, state: 'idle' });
    note(o, Q7.x + 242, Q7.y - 10, 'norm', { anchor: 'end' });
  }
  queryRow(g, p);
  keyRows(g);
  KEY_NAMES.forEach((name, i) => note(g, Q7.rowsX + i * CELL + CELL / 2, Q7.rows[0] - 8, name, { anchor: 'middle' }));
  weightRow(g, 0, 'plain', QK.plainWeights(1), 1);
  weightRow(g, 1, 'q grew 10×', QK.plainWeights(10), seg(p, 0.4, 0.5));
  weightRow(g, 2, 'with QK-norm', QK.normedWeights(1), seg(p, 0.6, 0.75), 'qk');
}
