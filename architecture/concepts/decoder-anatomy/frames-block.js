// decoder-anatomy frames 4–6: inside block 1, following row "sat": attention, the MLP, and the Mixture-of-Experts branch.
import * as G from '@shared/glyphs.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { TOKENS, X, SAT, A_SAT, M_SAT, X1_SAT, X2_SAT, WEIGHTS_SAT } from './numbers.js';
import { CELL, SMALL, STREAM, LANE, DETAIL, STRIP, seg, lerp, dip, arriving, leaving, layer, note, chips, stream, streamRowY, laneBlend, linkedRow, blendRows } from './stage.js';
import { int } from './format.js';

const ADDER = Object.freeze({ x: 30, y: (STRIP.a + CELL + STRIP.b) / 2 });
const CENTER_X = DETAIL.x + DETAIL.w / 2;
const DENSE = paramBreakdown(PRESETS.toy);
const MOE_CONFIG = Object.freeze({ ...PRESETS.toy, moe: { routed: 8, shared: 0, topK: 2, hidden: 8, denseLayers: 0 } });
const MOE = paramBreakdown(MOE_CONFIG);
const ACTIVE_EXPERTS = [2, 5]; // E3 and E6, 0-based
const W_TO_3DP = (v) => (v === 0 ? '0' : v.toFixed(3)); // the weights row at the storyboard's (and attention's) 3 d.p.
const SAT_CHIP = [SAT];


export const rowsWith = (r, row) => X.map((x, i) => (i === r ? row : x));
const plain = (v) => (v === 0 ? '0' : `${v < 0 ? '−' : ''}${Math.abs(v)}`);

function normLabel(svg, opacity) {
  note(layer(svg, opacity), STRIP.x, STRIP.a - 8, 'normalize: rescale the row to a standard size');
}

// The followed row in the strip, always with the selection outline.
function followed(svg, { y, values, cell = CELL, label, opacity = 1, selOpacity = 1 }) {
  linkedRow(layer(svg, opacity), 'x', { x: STRIP.x, y, values, cell, label });
  if (selOpacity > 0) G.selectionMark(layer(svg, selOpacity), { x: STRIP.x, y, w: values.length * cell, h: cell });
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
  // "down" comes after "sat": masked, so its weight is 0, printed, and hatched (excluded, README lesson 24).
  G.heatmap(svg, { x: hx, y: hy, values: [WEIGHTS_SAT], cell: CELL, maxAbs: 1, colLabels: TOKENS, format: W_TO_3DP, hatch: [[false, false, false, true]] });
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
  G.block(g, { x: DETAIL.x + 16, y: DETAIL.y + 24, w: 150, h: 22, label: 'W_in, W_gate [8 × 16]' });
  const wide = widen >= 1 ? 0 : Math.sin(Math.PI * widen); // widens to 16 and back to 8
  const hw = lerp(80, 160, wide);
  G.block(g, { x: CENTER_X - hw / 2, y: DETAIL.y + 52, w: hw, h: 18, label: `${Math.round(lerp(8, 16, wide))} numbers` });
  G.block(g, { x: DETAIL.x + 16, y: DETAIL.y + 76, w: 150, h: 22, label: 'W_out [16 × 8]' });
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
  const sel = arriving(p);
  chips(svg, TOKENS, { follow: SAT_CHIP, selOpacity: sel });
  note(layer(svg, leaving(p)), STREAM.x, 246, 'each block reads X and adds to it; nothing is overwritten');
  const cs = lerp(CELL, SMALL, seg(p, 0, 0.25));
  stream(svg, rowsWith(SAT, p === 1 ? X1_SAT : X[SAT]), { cell: cs, selected: [SAT], selOpacity: sel });
  laneBlend(svg, { opacity: 0.55 }, { active: { block: 1, half: 0 } }, sel);
  const box = layer(svg, sel);
  attentionBox(box, seg(p, 0.5, 0.65));
  G.adder(box, ADDER);
  normLabel(svg, seg(p, 0.3, 0.4));
  const lift = seg(p, 0.15, 0.4);
  if (lift > 0) followed(svg, { y: lerp(streamRowY(SAT, cs), STRIP.a, lift), values: X[SAT], cell: lerp(cs, CELL, lift), label: lift === 1 ? 'x_sat' : undefined, opacity: dip(seg(p, 0.4, 0.5)) });
  delta(svg, 'a', A_SAT, 'a_sat', seg(p, 0.65, 0.85));
  const typed = seg(p, 0.85, 1);
  sum(svg, X1_SAT, 'x′_sat', typed);
  if (typed === 1) cellNote(svg);
}

function cellNote(parent) {
  note(parent, STRIP.x + 4.5 * CELL, STRIP.c + CELL + 14, `${plain(X[SAT][4])} + ${plain(A_SAT[4])} = ${plain(X1_SAT[4])}`, { anchor: 'middle' });
}

// Frame 5: the same row goes through the MLP alone (8 → 16 → 8) and its result is added back.
export function drawFrame5(svg, p) {
  const out = leaving(p);
  const inn = arriving(p);
  chips(svg, TOKENS, { follow: SAT_CHIP });
  stream(svg, rowsWith(SAT, p === 1 ? X2_SAT : X1_SAT), { selected: [SAT] });
  laneBlend(svg, { active: { block: 1, half: 0 } }, { active: { block: 1, half: 1 } }, inn);
  if (out > 0) {
    const old = layer(svg, out);
    attentionBox(old, 1);
    followed(old, { y: STRIP.a, values: X[SAT], label: 'x_sat' });
    linkedRow(old, 'a', { x: STRIP.x, y: STRIP.b, values: A_SAT, cell: CELL, label: 'a_sat' });
    cellNote(old);
  }
  mlpBox(svg, { widen: seg(p, 0.3, 0.6), opacity: inn });
  normLabel(svg, 1);
  G.adder(svg, ADDER);
  followed(svg, { y: lerp(STRIP.c, STRIP.a, inn), values: X1_SAT, label: 'x′_sat', opacity: dip(seg(p, 0.15, 0.3)), selOpacity: inn });
  delta(svg, 'm', M_SAT, 'm_sat', seg(p, 0.6, 0.8));
  sum(svg, X2_SAT, 'x″_sat', seg(p, 0.8, 1));
  perBlockReadout(layer(svg, inn));
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

// The readout types in line by line (template rule 7: no count-up from 0).
export function moeReadout(svg, t) {
  const { routed, topK } = MOE_CONFIG.moe;
  const lines = [
    `total: ${routed} × ${MOE.perLayer.expert} = ${int(MOE.parts.experts / MOE_CONFIG.layers)}`,
    `active: ${topK} × ${MOE.perLayer.expert} = ${int(topK * MOE.perLayer.expert)}, the same as the dense MLP (${int(DENSE.perLayer.mlp)})`,
    `router: ${MOE_CONFIG.dModel} × ${routed} = ${int(MOE.parts.router / MOE_CONFIG.layers)} more per block`,
  ];
  lines.slice(0, Math.ceil(lines.length * t)).forEach((line, i) => note(svg, STRIP.x, 246 + 18 * i, line));
}

// Frame 6 (a branch): the MLP becomes a router and eight small experts; a dot lands on E3, then E6.
export function drawFrame6(svg, p) {
  const out = leaving(p);
  chips(svg, TOKENS, { follow: SAT_CHIP });
  // The branch replaces the MLP half, so sat's row goes back to x′ (before the MLP): two real states, blended.
  stream(svg, rowsWith(SAT, blendRows(X2_SAT, X1_SAT, arriving(p))), { selected: [SAT] });
  const swap = seg(p, 0, 0.3);
  laneBlend(svg, { active: { block: 1, half: 1 } }, { halves: ['attention', 'MoE'], active: { block: 1, half: 1 } }, swap);
  if (swap < 1) mlpBox(svg, { opacity: 1 - swap, widen: 1 });
  moeBox(svg, { opacity: swap, route: seg(p, 0.3, 0.7) });
  if (out > 0) {
    const old = layer(svg, out);
    normLabel(old, 1);
    G.adder(old, ADDER);
    linkedRow(old, 'm', { x: STRIP.x, y: STRIP.b, values: M_SAT, cell: CELL, label: 'm_sat' });
    linkedRow(old, 'x', { x: STRIP.x, y: STRIP.c, values: X2_SAT, cell: CELL, label: 'x″_sat' });
    perBlockReadout(old);
  }
  followed(svg, { y: STRIP.a, values: X1_SAT, label: 'x′_sat' });
  moeReadout(svg, seg(p, 0.7, 1));
}
