// decoder-recap's two block stacks: "GPT-3 (2020)" on the left, "2026" on the right, one block each (storyboard §4).
// A stack is drawn from a parts description; between two descriptions each swapped part cross-fades, so a frame
// is a pure function of (frame, progress) and its start equals the previous frame's end.
import * as G from '@shared/glyphs.js';
import { STACK_W, LEFT_X, RIGHT_X, layer, note, select, pulse } from './stage.js';
import { GPT3_FACTS, MODERN_FACTS } from './numbers.js';

const Y = Object.freeze({ title: 14, pos: 22, frameTop: 52, norm1: 60, attn: 84, subAttn: 123, norm2: 156, mlp: 180, subMlp: 219, frameBottom: 242, times: 258, summary: 276 });
const LANE = 10;
const BOX = Object.freeze({ x: 24, w: 88 });
const SUB_X = 14;
const SUB_GAP = 12;
const TAG_WIDTH = 16; // characters per sub-label line (88 px box, 11 px labels)
export const PART = Object.freeze({
  pos: { x: 10, y: Y.pos, w: 100, h: 22 },
  norm1: { x: BOX.x, y: Y.norm1, w: BOX.w, h: 18 },
  attn: { x: BOX.x, y: Y.attn, w: BOX.w, h: 26 },
  norm2: { x: BOX.x, y: Y.norm2, w: BOX.w, h: 18 },
  mlp: { x: BOX.x, y: Y.mlp, w: BOX.w, h: 26 },
  times: { x: 6, y: Y.times - 11, w: 96, h: 16 },
});
export const partRect = (side, name) => ({ ...PART[name], x: PART[name].x + (side === 'left' ? LEFT_X : RIGHT_X) });

// What the right stack looks like once `k` frames are done (k = 1 is the unchanged copy of GPT-3's block).
export const partsAfter = (k) => ({
  norm: k >= 2 ? 'RMSNorm' : 'LayerNorm',
  table: k < 3,
  rope: k >= 3,
  mlp: k >= 5 ? 'experts' : (k >= 4 ? 'swiglu' : 'gelu'),
  biases: k < 4,
  kv: k >= 6 ? 'shared' : 'own',
  qknorm: k >= 7,
  sink: k >= 8,
});

const eq = (a, b) => a === b || JSON.stringify(a) === JSON.stringify(b);
// Draws `draw(parent, value)` for a value that changes from a to b; both cross-fade while t is between 0 and 1.
function mix(parent, a, b, t, draw) {
  if (eq(a, b) || t <= 0) return draw(layer(parent), a);
  if (t >= 1) return draw(layer(parent), b);
  draw(layer(parent, 1 - t), a);
  return draw(layer(parent, t), b);
}

function wrap(tags) {
  return tags.reduce((out, tag) => {
    const last = out.length - 1;
    if (last >= 0 && `${out[last]} · ${tag}`.length <= TAG_WIDTH) return [...out.slice(0, last), `${out[last]} · ${tag}`];
    return [...out, tag];
  }, []);
}
const attentionLines = (parts) => [
  parts.kv === 'shared' ? `${MODERN_FACTS.sharedKv} shared KV` : 'own KV per head',
  ...wrap([parts.biases && 'biases', parts.rope && 'RoPE', parts.qknorm && 'QK-norm', parts.sink && 'sink'].filter(Boolean)),
];
const MLP_LINES = Object.freeze({ gelu: ['GELU, 4 × wider'], swiglu: ['SwiGLU, 8/3 × d'], experts: ['SwiGLU experts', `top-${MODERN_FACTS.topK} of ${MODERN_FACTS.experts}`] });
const mlpLines = (parts) => [...MLP_LINES[parts.mlp], ...(parts.biases && parts.mlp === 'gelu' ? ['biases'] : [])];
const attentionSwapped = (parts) => parts.kv === 'shared' || parts.rope || parts.qknorm || parts.sink;

function mixLines(g, y, a, b, t) {
  Array.from({ length: Math.max(a.length, b.length) }, (_, i) => mix(g, a[i] ?? '', b[i] ?? '', t, (parent, s) => (s ? note(parent, SUB_X, y + i * SUB_GAP, s) : null)));
}
const box = (g, rect, label, active) => G.block(g, { ...rect, label, state: active ? 'active' : 'idle' });

function frameAndLane(g) {
  G.svgEl('rect', { class: 'g-stack-frame', x: 0, y: Y.frameTop, width: STACK_W, height: Y.frameBottom - Y.frameTop, rx: 6 }, g);
  G.svgEl('line', { class: 'g-lane', x1: LANE, y1: Y.frameTop + 8, x2: LANE, y2: Y.frameBottom - 8 }, g);
  [PART.attn, PART.mlp].forEach((r) => {
    const cy = r.y + r.h / 2;
    G.svgEl('line', { class: 'g-link', x1: LANE + 8, y1: cy, x2: r.x, y2: cy }, g);
    G.adder(g, { x: LANE, y: cy, r: 8 });
  });
}

// One block with its position slot, drawn from `from` toward `to` at cross-fade t. `summary`: { lines, opacity } under the block.
export function drawStack(parent, { x, title, from, to = from, t = 1, layers, summary = null, opacity = 1 }) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-stack', transform: `translate(${x} 0)`, role: 'img', 'aria-label': `${title}: one block, repeated ${layers} times`, opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  note(g, 0, Y.title, title, { cls: '' });
  mix(g, from.table, to.table, t, (p, table) => (table ? G.block(p, { ...PART.pos, label: 'position table' }) : note(p, PART.pos.x + PART.pos.w / 2, Y.pos + 15, '(no table)', { anchor: 'middle' })));
  frameAndLane(g);
  mix(g, from.norm, to.norm, t, (p, name) => { box(p, PART.norm1, name, name === 'RMSNorm'); box(p, PART.norm2, name, name === 'RMSNorm'); });
  mix(g, attentionSwapped(from), attentionSwapped(to), t, (p, swapped) => box(p, PART.attn, 'attention', swapped));
  mixLines(g, Y.subAttn, attentionLines(from), attentionLines(to), t);
  mix(g, from.mlp, to.mlp, t, (p, kind) => box(p, PART.mlp, kind === 'experts' ? 'experts' : 'MLP', kind !== 'gelu'));
  mixLines(g, Y.subMlp, mlpLines(from), mlpLines(to), t);
  note(g, PART.times.x + 4, Y.times, `× ${layers} blocks`);
  if (summary && summary.opacity > 0) summary.lines.forEach((s, i) => note(layer(g, summary.opacity), 0, Y.summary + i * 14, s));
  return g;
}

export const GPT3_PARTS = partsAfter(1);

// ---- frame 9: the right stack grows to four blocks, three of them with a cheaper kind of attention ----
const HYBRID = Object.freeze({ top: 22, step: 72, h: 64, kinds: ['linear', 'linear', 'linear', 'full'] });
export const hybridAttnRect = (k) => ({ x: RIGHT_X + BOX.x, y: HYBRID.top + k * HYBRID.step + 16, w: BOX.w, h: 22 });

function miniBlock(g, k, relabel) {
  const y0 = HYBRID.top + k * HYBRID.step;
  G.svgEl('rect', { class: 'g-stack-frame', x: 0, y: y0, width: STACK_W, height: HYBRID.h, rx: 6 }, g);
  G.svgEl('line', { class: 'g-lane', x1: LANE, y1: y0 + 6, x2: LANE, y2: y0 + HYBRID.h - 6 }, g);
  note(g, 14, y0 + 11, `block ${k + 1}`, { cls: 'g-label' });
  const rows = [{ y: y0 + 16, h: 22 }, { y: y0 + 42, h: 16 }];
  rows.forEach((r) => {
    G.svgEl('line', { class: 'g-link', x1: LANE + 7, y1: r.y + r.h / 2, x2: BOX.x, y2: r.y + r.h / 2 }, g);
    G.adder(g, { x: LANE, y: r.y + r.h / 2, r: 7 });
  });
  mix(g, 'attention', HYBRID.kinds[k], relabel, (p, label) => G.block(p, { x: BOX.x, y: rows[0].y, w: BOX.w, h: rows[0].h, label, state: label === 'linear' ? 'active' : 'idle' }));
  G.block(g, { x: BOX.x, y: rows[1].y, w: BOX.w, h: rows[1].h, label: 'MLP' });
}

// `relabel` runs 0 → 1 as the attention boxes of blocks 1–3 change from "attention" to "linear".
export function drawHybrid(parent, { opacity = 1, relabel = 1 }) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-stack', transform: `translate(${RIGHT_X} 0)`, role: 'img', 'aria-label': '2026: four blocks with mixed attention kinds', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  note(g, 0, Y.title, '2026', { cls: '' });
  HYBRID.kinds.forEach((_, k) => miniBlock(g, k, relabel));
  note(g, 14, HYBRID.top + 4 * HYBRID.step + 14, '⋮ and so on, N blocks');
  return g;
}

// ---- the selection outlines: which part of the right stack is being swapped in each frame ----
export const SELECTED = Object.freeze({ 2: ['norm1', 'norm2'], 3: ['pos', 'attn'], 4: ['mlp'], 5: ['mlp'], 6: ['attn'], 7: ['attn'], 8: ['attn'] });
export function selectParts(parent, names, opacity) {
  names.forEach((name) => select(parent, partRect('right', name), opacity));
}
export function selectHybrid(parent, opacity) {
  [0, 1, 2].forEach((k) => select(parent, hybridAttnRect(k), opacity));
}

// Frame 10: each swapped box pulses once, in the order of the frames (2 → 9). The ninth is the "× N blocks" label's "some window or linear".
const PULSES = Object.freeze([['norm1', 'norm2'], ['pos'], ['mlp'], ['mlp'], ['attn'], ['attn'], ['attn'], ['times']]);
export function pulseParts(parent, p) {
  PULSES.forEach((names, i) => {
    const t = pulse(p, 0.15 + i * 0.1, 0.15 + (i + 1) * 0.1 + 0.02);
    selectParts(parent, names, t);
  });
}

export { LEFT_X, RIGHT_X };
