// prefill-decode stage layout: fixed positions shared by every frame, the timing helpers and drawing helpers on the glyph
// library. Frames 2–6 hold two rows (decode above, prefill below); frames 7–9 hold the batch bar left, the plots right.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatInt } from '@math/core.js';
import { STAGE_SCALE_S, readParts } from './model.js';
import { ANSWER_WORD, PROMPT_WORDS, CHIPS_SHOWN, BATCH_OWNERS, H200 } from './numbers.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const LEFT = 16;
export const LINE = 16;
export const CELL = G.NUMBER_CELL;
export const NEUTRAL = 1e15; // a readout cell's value-scale maxAbs: the cell stays neutral (the number is the reading)
// One seconds scale for every step bar on the stage (P4-R8): the full width BAR.w is the 1,000-token prefill step.
export const BAR = Object.freeze({ x: LEFT, w: 440, scaleS: STAGE_SCALE_S });
export const ROW_D = Object.freeze({ chipY: 6, barY: 40 }); // frames 2–6: decode
export const ROW_P = Object.freeze({ chipY: 134, barY: 166 }); // frames 3–6: prefill
export const BATCH = Object.freeze({ chipY: 6, contextY: 46, barY: 74, secondBarY: 198 }); // frames 7–9
export const GPU_BIG = Object.freeze({ x: 150, y: 70, w: 280, h: 170 }); // frame 1
export const GPU_SMALL = Object.freeze({ x: 440, y: 8, w: 124, h: 84 }); // frames 2–3
export const NOTES_Y = 270; // frames 2–7: the numbers under the bars
export const CHIP_GAP = 6;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a));
export const lerp = (a, b, t) => a + (b - a) * t;
export const HANDOFF = 0.15; // what leaves fades out and what arrives fades in during [0, HANDOFF] (template rule 4)
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);
// A pulse: 0 → 1 → 0 across [a, b] (the memory fill's "weights read" beat).
export const pulse = (p, a, b) => Math.sin(Math.PI * seg(p, a, b));

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15). cls 'g-label' is muted; '' prints in ink (a value the learner reads).
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start' } = {}) {
  const g = G.svgEl('g', { class: 'glyph g-note' }, parent);
  G.svgEl('text', { x, y, class: cls || null, 'text-anchor': anchor }, g).textContent = str;
  return g;
}

export function lines(parent, x, y, texts, { cls = '' } = {}) {
  texts.forEach((t, i) => note(parent, x, y + i * LINE, t, { cls }));
}

// One token chip; `followed` adds the one selection mark (template rule 5). Returns the chip's right edge.
export function chip(parent, { x, y, text, owner = null, followed = false }) {
  G.token(parent, { x, y, text, owner });
  const w = G.tokenWidth(text);
  if (followed) G.selectionMark(parent, { x, y, w, h: 24 });
  return x + w;
}

const othersText = (n) => `+ ${formatInt(n)} others`;

// Request A's answer chip (frames 1–6), framed.
export const answerChip = (parent, x, y) => chip(parent, { x, y, text: ANSWER_WORD, owner: 'A', followed: true });

// Request A's prompt: eight chips (the first framed) and "+ N others" for the rest (README lesson 18). Returns the right edge.
export function promptChips(parent, { x = LEFT, y, tokens }) {
  let cx = x;
  PROMPT_WORDS.slice(0, Math.min(CHIPS_SHOWN, tokens)).forEach((text, i) => {
    cx = chip(parent, { x: cx, y, text, owner: 'A', followed: i === 0 }) + CHIP_GAP;
  });
  if (tokens <= CHIPS_SHOWN) return cx;
  note(parent, cx, y + 16, othersText(tokens - CHIPS_SHOWN), { cls: '' });
  return cx + othersText(tokens - CHIPS_SHOWN).length * 6.6;
}

// The decode batch: chips A–D in their request hues (A framed) and "+ N others" (frames 7–9). Returns the right edge.
export function batchChips(parent, { x = LEFT, y = BATCH.chipY, users }) {
  let cx = x;
  BATCH_OWNERS.slice(0, Math.min(users, BATCH_OWNERS.length)).forEach((owner, i) => {
    cx = chip(parent, { x: cx, y, text: owner, owner, followed: i === 0 }) + CHIP_GAP;
  });
  if (users <= BATCH_OWNERS.length) return cx;
  note(parent, cx, y + 16, othersText(users - BATCH_OWNERS.length), { cls: '' });
  return cx + othersText(users - BATCH_OWNERS.length).length * 6.6;
}

// A left-to-right wipe: everything drawn into the returned group shows only left of x0 + width (no glyph is edited).
export function wipe(parent, name, { x, y, w, h }) {
  const svg = parent.ownerSVGElement ?? parent;
  const id = `${svg.dataset.hatchId}-wipe-${name}`;
  const clip = G.svgEl('clipPath', { id }, svg.querySelector('defs'));
  G.svgEl('rect', { x, y, width: Math.max(0, w), height: h }, clip);
  return G.svgEl('g', { 'clip-path': `url(#${id})` }, parent);
}

// A step's bar at y on the stage's one scale: reading parts (weights, KV, activations) over the arithmetic row, linked
// to the math panel (hl-memory, hl-compute). `reveal` (0–1) wipes it in from the left; the numbers it prints are final.
export function stepBarAt(parent, { y, step, reveal = 1, name = 'bar', label = 'step time' }) {
  if (reveal <= 0) return;
  const reading = readParts(step);
  const L = G.stepBarLayout({ w: BAR.w, scaleS: BAR.scaleS, reading, mathS: step.computeS });
  const host = reveal >= 1 ? parent : wipe(parent, name, { x: BAR.x - 4, y: y - 4, w: (L.extent + 8) * reveal, h: L.height + 8 });
  // The theme outlines the reading segments for hl-memory and the arithmetic segment for hl-compute.
  const compute = G.svgEl('g', { 'data-link': 'compute' }, G.svgEl('g', { 'data-link': 'memory' }, host));
  G.stepBar(compute, { x: BAR.x, y, w: BAR.w, scaleS: BAR.scaleS, reading, mathS: step.computeS, label });
}

// A readout cell at NUMBER_CELL with its label above (template rule 7). null draws the empty slot it will type into.
export function readCell(parent, { x, y, value, format = String, label = null }) {
  if (label) note(parent, x, y - 8, label);
  if (value == null) {
    const g = G.svgEl('g', { class: 'glyph g-empty' }, parent);
    G.svgEl('rect', { class: 'g-frame', x: x + 1.5, y: y + 1.5, width: CELL - 3, height: CELL - 3, rx: 3 }, g);
    return;
  }
  G.cell(parent, { x, y, size: CELL, v: value, maxAbs: NEUTRAL, format });
}

// The H200 with its memory bar (the weights' share of its nominal HBM) and its label.
export function gpu(parent, box, { memFill, opacity = 1 }) {
  if (opacity <= 0) return;
  G.gpu(layer(parent, opacity), { ...box, memFill, label: `${H200.label} · ${formatBytes(H200.hbmBytes)} ${H200.basis}` });
}
