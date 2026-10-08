// decoder-anatomy frames 4–6: inside block 1, following row "sat": attention, the MLP, and the Mixture-of-Experts branch.
import * as G from '@shared/glyphs.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { TOKENS, X, SAT, A_SAT, M_SAT, X1_SAT, X2_SAT, WEIGHTS_SAT } from './stream.js';
import { CELL, SMALL, STREAM, LANE, DETAIL, STRIP, seg, lerp, dip, layer, note, chips, stream, streamRowY, lane, linkedRow } from './layout.js';
import { int } from './format.js';

const ADDER = Object.freeze({ x: 30, y: (STRIP.a + CELL + STRIP.b) / 2 });
const CENTER_X = DETAIL.x + DETAIL.w / 2;
const DENSE = paramBreakdown(PRESETS.toy);
const MOE_CONFIG = Object.freeze({ ...PRESETS.toy, moe: { routed: 8, shared: 0, topK: 2, hidden: 8, denseLayers: 0 } });
const MOE = paramBreakdown(MOE_CONFIG);
const ACTIVE_EXPERTS = [2, 5]; // E3 and E6, 0-based

export const rowsWith = (r, row) => X.map((x, i) => (i === r ? row : x));
const plain = (v) => (v === 0 ? '0' : `${v < 0 ? '−' : ''}${Math.abs(v)}`);

function normLabel(svg, opacity) {
  note(layer(svg, opacity), STRIP.x, STRIP.a - 8, 'normalize: rescale the row to a standard size');
}

// The followed row in the strip, always with the selection outline.
function followed(svg, { y, values, cell = CELL, label, opacity = 1 }) {
  linkedRow(layer(svg, opacity), 'x', { x: STRIP.x, y, values, cell, label });
  G.selectionMark(svg, { x: STRIP.x, y, w: values.length * cell, h: cell });
}

// A delta (a_sat or m_sat) leaving the detail box and landing in strip row b.
function delta(svg, letter, values, label, t) {
  if (t === 0) return;
  const from = { x: DETAIL.x + 8, y: DETAIL.y + DETAIL.h - SMALL - 6 };
  linkedRow(svg, letter, { x: lerp(from.x, STRIP.x, t), y: lerp(from.y, STRIP.b, t), values, cell: lerp(SMALL, CELL, t), label: t === 1 ? label : undefined });
}

// The sum types in cell by cell in strip row c.
function sum(svg, values, label, t) {
  if (t === 0) return;
  linkedRow(svg, 'x', { x: STRIP.x, y: STRIP.c, values: values.slice(0, Math.ceil(values.length * t)), cell: CELL, label });
}

function attentionBox(svg, flowT) {
  G.block(svg, { x: DETAIL.x, y: DETAIL.y, w: DETAIL.w, h: DETAIL.h, label: '' }).setAttribute('data-link', 'a');
  note(svg, CENTER_X, DETAIL.y + 15, 'attention', { anchor: 'middle', cls: '' });
  const hx = DETAIL.x + 11;
  const hy = DETAIL.y + 40;
  G.heatmap(svg, { x: hx, y: hy, values: [WEIGHTS_SAT], cell: CELL, maxAbs: 1, colLabels: TOKENS });
  // "down" comes after "sat": masked, so its weight is 0 and the cell is hatched (excluded, README lesson 24).
  const hatch = G.svgEl('rect', { class: 'g-hatch', x: hx + 3 * CELL + 1.5, y: hy + 1.5, width: CELL - 3, height: CELL - 3, rx: 3, fill: G.hatchFill(svg), 'pointer-events': 'none' }, svg);
  G.svgEl('title', {}, hatch).textContent = 'masked: "down" comes after "sat"';
  note(svg, CENTER_X, DETAIL.y + 96, 'head A of 2; you\'ll compute', { anchor: 'middle' });
  note(svg, CENTER_X, DETAIL.y + 109, 'this row in attention', { anchor: 'middle' });
  if (flowT === 0) return;
  [0, 1, 2].forEach((r) => {
    const y = streamRowY(r) + SMALL / 2;
    G.flow(svg, { from: [STREAM.x + 8 * SMALL + 2, y], to: [DETAIL.x - 2, y], carry: 'activation', progress: flowT });
  });
}

export function mlpBox(parent, { widen = 0, opacity = 1, title = 'MLP' } = {}) {
  const g = layer(parent, opacity);
  G.block(g, { x: DETAIL.x, y: DETAIL.y, w: DETAIL.w, h: DETAIL.h, label: '' }).setAttribute('data-link', 'm');
  note(g, CENTER_X, DETAIL.y + 15, title, { anchor: 'middle', cls: '' });
  G.block(g, { x: DETAIL.x + 16, y: DETAIL.y + 24, w: 150, h: 22, label: 'W_in, W_gate [8 × 16]', state: 'dim' });
  const wide = Math.sin(Math.PI * widen);
  const hw = lerp(80, 160, wide);
  G.block(g, { x: CENTER_X - hw / 2, y: DETAIL.y + 52, w: hw, h: 18, label: `${Math.round(lerp(8, 16, wide))} numbers` });
  G.block(g, { x: DETAIL.x + 16, y: DETAIL.y + 76, w: 150, h: 22, label: 'W_out [16 × 8]', state: 'dim' });
  note(g, CENTER_X, DETAIL.y + 110, 'MLP 8 → 16 → 8', { anchor: 'middle' });
  return g;
}

function perBlockReadout(svg) {
  note(svg, LANE.x, 236, 'per block:');
  note(svg, LANE.x, 252, `attention ${int(DENSE.perLayer.attention)}`);
  note(svg, LANE.x, 268, `MLP ${int(DENSE.perLayer.mlp)} · norms ${int(DENSE.perLayer.norms)}`);
}

// Frame 4: sat lifts out, is normalized, reads the other rows through attention, and the result is added back.
export function drawFrame4(svg, p) {
  chips(svg);
  const cs = lerp(CELL, SMALL, seg(p, 0, 0.25));
  stream(svg, rowsWith(SAT, p === 1 ? X1_SAT : X[SAT]), { cell: cs, selected: [SAT] });
  lane(svg, { active: { block: 1, half: 0 } });
  attentionBox(svg, seg(p, 0.5, 0.65));
  const lift = seg(p, 0.15, 0.4);
  normLabel(svg, seg(p, 0.3, 0.4));
  G.adder(svg, ADDER);
  followed(svg, { y: lerp(streamRowY(SAT, cs), STRIP.a, lift), values: X[SAT], cell: lerp(cs, CELL, lift), label: lift === 1 ? 'x_sat' : undefined, opacity: dip(seg(p, 0.4, 0.5)) });
  delta(svg, 'a', A_SAT, 'a_sat', seg(p, 0.65, 0.85));
  const typed = seg(p, 0.85, 1);
  sum(svg, X1_SAT, 'x′_sat', typed);
  if (typed === 1) note(svg, STRIP.x + 4.5 * CELL, STRIP.c + CELL + 14, `${plain(X[SAT][4])} + ${plain(A_SAT[4])} = ${plain(X1_SAT[4])}`, { anchor: 'middle' });
}

// Frame 5: the same row goes through the MLP alone (8 → 16 → 8) and its result is added back.
export function drawFrame5(svg, p) {
  chips(svg);
  stream(svg, rowsWith(SAT, p === 1 ? X2_SAT : X1_SAT), { selected: [SAT] });
  lane(svg, { active: { block: 1, half: 1 } });
  mlpBox(svg, { widen: seg(p, 0.3, 0.6) });
  normLabel(svg, 1);
  G.adder(svg, ADDER);
  followed(svg, { y: lerp(STRIP.c, STRIP.a, seg(p, 0, 0.15)), values: X1_SAT, label: 'x′_sat', opacity: dip(seg(p, 0.15, 0.3)) });
  delta(svg, 'm', M_SAT, 'm_sat', seg(p, 0.6, 0.8));
  sum(svg, X2_SAT, 'x″_sat', seg(p, 0.8, 1));
  perBlockReadout(svg);
}

export function moeBox(parent, { opacity = 1, route = 1 } = {}) {
  const g = layer(parent, opacity);
  G.block(g, { x: DETAIL.x, y: DETAIL.y, w: DETAIL.w, h: DETAIL.h, label: '' }).setAttribute('data-link', 'm');
  note(g, CENTER_X, DETAIL.y + 15, 'in most 2026 models:', { anchor: 'middle', cls: '' });
  const router = { y: DETAIL.y + 24, h: 20 };
  G.block(g, { x: DETAIL.x + 4, y: router.y, w: DETAIL.w - 8, h: router.h, label: 'router' });
  const ey = DETAIL.y + 62;
  const ex = (k) => DETAIL.x + 4 + k * 22;
  const landed = ACTIVE_EXPERTS.map((_, i) => route >= (i + 1) / ACTIVE_EXPERTS.length);
  Array.from({ length: 8 }, (_, k) => G.block(g, { x: ex(k), y: ey, w: 20, h: 26, label: `E${k + 1}`, state: landed[ACTIVE_EXPERTS.indexOf(k)] ? 'active' : 'idle' }));
  ACTIVE_EXPERTS.forEach((k, i) => {
    const t = seg(route, i / ACTIVE_EXPERTS.length, (i + 1) / ACTIVE_EXPERTS.length);
    if (t > 0) G.flow(g, { from: [ex(k) + 10, router.y + router.h + 1], to: [ex(k) + 10, ey - 2], carry: 'activation', progress: t });
  });
  note(g, CENTER_X, DETAIL.y + 106, '8 small MLPs, hidden 8 each', { anchor: 'middle' });
  return g;
}

function moeReadout(svg, t) {
  if (t === 0) return;
  const { routed, topK } = MOE_CONFIG.moe;
  const up = (n) => int(Math.round(n * t));
  note(svg, STRIP.x, 246, `total: ${routed} × ${MOE.perLayer.expert} = ${up(MOE.parts.experts / MOE_CONFIG.layers)}`);
  note(svg, STRIP.x, 264, `active: ${topK} × ${MOE.perLayer.expert} = ${up(topK * MOE.perLayer.expert)}, the same as the dense MLP (${int(DENSE.perLayer.mlp)})`);
  note(svg, STRIP.x, 282, `router: ${MOE_CONFIG.dModel} × ${routed} = ${up(MOE.parts.router / MOE_CONFIG.layers)} more per block`);
}

// Frame 6 (a branch): the MLP becomes a router and eight small experts; a dot lands on E3, then E6.
export function drawFrame6(svg, p) {
  chips(svg);
  stream(svg, rowsWith(SAT, X1_SAT), { selected: [SAT] });
  lane(svg, { halves: ['attention', 'MoE'], active: { block: 1, half: 1 } });
  const swap = seg(p, 0, 0.3);
  if (swap < 1) mlpBox(svg, { opacity: 1 - swap });
  moeBox(svg, { opacity: swap, route: seg(p, 0.3, 0.7) });
  followed(svg, { y: STRIP.a, values: X1_SAT, label: 'x′_sat' });
  moeReadout(svg, seg(p, 0.7, 1));
}
