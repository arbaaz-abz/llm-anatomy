// Frames 5–7: multi-head latent attention on the same 8-head layer: the latent, the position key, and the rebuild folded away.
import * as G from '@shared/glyphs.js';
import {
  Q, QUERY_HEADS, HEADER, READOUT, LINES, REST_DOT, qx, qcx, seg, lerp, ease, fade, key, label, swapKey, linked,
  queryRow, dimBlock, ghost,
} from './stage.js';
import { drawFrame4 } from './frames-pattern.js';
import { LATENTS, POSITION_KEYS, VALUE_SCALE, FOUR_TOKENS, TOY_SHAPE } from './numbers.js';

const heads = Array.from({ length: QUERY_HEADS }, (_, h) => h);
const REBUILD = Object.freeze({ y: 104, h: 24 });
const STORE = Object.freeze({ x: 170, y: 178, cell: 14, pitch: 16 }); // four token rows, one latent each
const LATENT_W = TOY_SHAPE.dLatent * STORE.cell;
const POS_X = STORE.x + LATENT_W + 8;
const STORE_H = 3 * STORE.pitch + STORE.cell;
const STORE_TOP_CENTER = STORE.x + LATENT_W / 2;
const W_O = Object.freeze({ x: 410, y: 198, w: 110, h: 28 });
const HALF = Object.freeze({ w: 28, gap: 4 });
const NOTE_Y = STORE.y + STORE_H + 14;
const rowY = (i) => STORE.y + i * STORE.pitch;
const readout = (svg, n, opacity = 1) => key(svg, READOUT.x, READOUT.y, `stored per token per layer: ${n} numbers`, { anchor: 'end', opacity });

// The stored latent, one row per token (rowOpacity(i)); `posX` places the position keys (null = not shown).
function store(svg, { rowOpacity = () => 1, posX = null, posOpacity = 1 }) {
  FOUR_TOKENS.forEach((token, i) => {
    const opacity = rowOpacity(i);
    if (opacity <= 0) return;
    const row = linked(svg, 'lat', { x: STORE.x, y: rowY(i), w: LATENT_W, h: STORE.cell });
    G.vector(row, { x: STORE.x, y: rowY(i), values: LATENTS[i], cell: STORE.cell, orient: 'row', maxAbs: VALUE_SCALE });
    fade(row, opacity);
    label(svg, STORE.x - 10, rowY(i) + STORE.cell / 2, token, { anchor: 'end', opacity });
    if (posX === null) return;
    const pos = linked(svg, 'rope', { x: posX, y: rowY(i), w: 2 * STORE.cell, h: STORE.cell });
    G.vector(pos, { x: posX, y: rowY(i), values: POSITION_KEYS[i], cell: STORE.cell, orient: 'row', maxAbs: VALUE_SCALE });
    fade(pos, posOpacity * opacity);
  });
}

// Wires: store → the head's rebuild block (leg 1, `toY` is where it lands) → its query (leg 2).
function legs(svg, { leg1 = 0, leg2 = 0, toY = REBUILD.y + REBUILD.h + 3, dot1 = REST_DOT, dot2 = REST_DOT }) {
  heads.forEach((h) => {
    if (leg1 > 0) fade(G.flow(svg, { from: [STORE_TOP_CENTER, STORE.y - 4], to: [qcx(h), toY], carry: 'kv', progress: dot1 }), leg1);
    if (leg2 > 0) fade(G.flow(svg, { from: [qcx(h), REBUILD.y - 3], to: [qcx(h), Q.y + Q.h + 3], carry: 'kv', progress: dot2 }), leg2);
  });
}

const rebuildBlocks = (svg, opacity) => heads.forEach((h) => dimBlock(svg, { x: qx(h), y: REBUILD.y, w: Q.w, h: REBUILD.h, label: 'rebuild' }, opacity));
const posLabels = (svg, opacity) => {
  label(svg, POS_X, STORE.y - 12, 'position key', { opacity });
  label(svg, POS_X, NOTE_Y, 'carries position (rope)', { opacity });
};

export function drawFrame5(svg, p) {
  ghost(svg, drawFrame4, 1 - seg(p, 0, 0.25));
  key(svg, HEADER.x, HEADER.y, 'MLA', { opacity: seg(p, 0.25, 0.45) });
  label(svg, HEADER.x + 40, HEADER.y, 'back on the 8-head layer', { opacity: seg(p, 0.25, 0.45) });
  queryRow(svg, seg(p, 0.15, 0.4));
  rebuildBlocks(svg, seg(p, 0.3, 0.5));
  store(svg, { rowOpacity: (i) => seg(p, 0.4 + 0.08 * i, 0.5 + 0.08 * i) });
  legs(svg, { leg1: seg(p, 0.6, 0.75), leg2: seg(p, 0.7, 0.85), dot1: REST_DOT * seg(p, 0.7, 0.88), dot2: REST_DOT * seg(p, 0.85, 1) });
  label(svg, STORE.x, NOTE_Y, 'latent', { opacity: seg(p, 0.6, 0.8) });
  readout(svg, TOY_SHAPE.dLatent, seg(p, 0.5, 0.7));
  key(svg, LINES.x, LINES.y[0], 'latent = 8 numbers per token per layer', { opacity: seg(p, 0.75, 0.95) });
  key(svg, LINES.x, LINES.y[1], 'rebuilds 8 heads × (4 + 4) = 64 numbers on the fly', { opacity: seg(p, 0.8, 1) });
}

// The two position-key cells slide in beside each latent; the readout ticks 8 → 10.
export function drawFrame6(svg, p) {
  const slide = ease(seg(p, 0.1, 0.7));
  key(svg, HEADER.x, HEADER.y, 'MLA');
  label(svg, HEADER.x + 40, HEADER.y, 'back on the 8-head layer');
  queryRow(svg);
  rebuildBlocks(svg, 1);
  store(svg, { posX: lerp(POS_X + 24, POS_X, slide), posOpacity: seg(p, 0.1, 0.4) });
  legs(svg, { leg1: 1, leg2: 1 });
  label(svg, STORE.x, NOTE_Y, 'latent');
  posLabels(svg, seg(p, 0.55, 0.85));
  readout(svg, Math.round(lerp(TOY_SHAPE.dLatent, TOY_SHAPE.dLatent + TOY_SHAPE.dRope, slide)));
  swapKey(svg, LINES.x, LINES.y[0], 'latent = 8 numbers per token per layer', '8 + 2 = 10 numbers per token per layer', p, [0, 0.2, 0.75]);
  key(svg, LINES.x, LINES.y[1], 'rebuilds 8 heads × (4 + 4) = 64 numbers on the fly', { opacity: 1 - seg(p, 0, 0.2) });
}

// The "rebuild K" halves slide up into the query blocks, the "rebuild V" halves down into W_O; the stored latent never moves.
export function drawFrame7(svg, p) {
  const move = ease(seg(p, 0.25, 0.75));
  const split = seg(p, 0.05, 0.2);
  const vX = (h) => lerp(qx(h) + HALF.w + HALF.gap, W_O.x + (W_O.w - HALF.w) / 2, move);
  key(svg, HEADER.x, HEADER.y, 'MLA');
  label(svg, HEADER.x + 40, HEADER.y, 'back on the 8-head layer');
  queryRow(svg);
  rebuildBlocks(svg, 1 - split);
  heads.forEach((h) => {
    dimBlock(svg, { x: qx(h), y: lerp(REBUILD.y, Q.y + 8, move), w: HALF.w, h: REBUILD.h, label: 'K' }, split * (1 - seg(p, 0.55, 0.8)));
    dimBlock(svg, { x: vX(h), y: lerp(REBUILD.y, W_O.y + 2, move), w: HALF.w, h: REBUILD.h, label: 'V' }, split * (1 - seg(p, 0.6, 0.85)));
  });
  fade(G.block(svg, { ...W_O, label: 'W_O' }), seg(p, 0.15, 0.35));
  store(svg, { posX: POS_X });
  legs(svg, { leg1: 1, leg2: 1 - seg(p, 0.45, 0.75), toY: lerp(REBUILD.y + REBUILD.h + 3, Q.y + Q.h + 3, move) });
  label(svg, STORE.x, NOTE_Y, 'latent');
  posLabels(svg, 1);
  label(svg, HEADER.x, 38, 'queries with the K-rebuild folded in', { opacity: seg(p, 0.7, 0.95) });
  label(svg, W_O.x + W_O.w / 2, W_O.y - 14, 'V-rebuild folded in', { anchor: 'middle', opacity: seg(p, 0.7, 0.95) });
  readout(svg, TOY_SHAPE.dLatent + TOY_SHAPE.dRope);
  swapKey(svg, LINES.x, LINES.y[0], '8 + 2 = 10 numbers per token per layer', '0 rebuilt K or V rows stored', p, [0, 0.2, 0.7]);
  key(svg, LINES.x, LINES.y[1], 'q · (c W_UK) = (q W_UKᵀ) · c', { opacity: seg(p, 0.7, 0.95) });
  key(svg, LINES.x, LINES.y[2], 'more compute per decode step', { opacity: seg(p, 0.75, 1) });
}
