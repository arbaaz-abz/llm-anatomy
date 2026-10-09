// batching stage: fixed 580 × 366 layout (three lane slots, the step axis) and the drawing helpers every frame shares.
// Every frame is a pure function of (frame, progress): no module state, no clock, no randomness (template rule 4).
import * as G from '@shared/glyphs.js';
import { spanLabel } from './format.js';

export const STAGE = Object.freeze({ w: 580, h: 366 });
export const X0 = 72; // x of step 0 (seat labels and request letters live in the gutter to its left)
export const STEP_PX = 40; // one step on the stage; 11 steps fit in 440 px
export const AXIS_PX = 440;
export const AXIS_Y = 14;
export const SLOT_TOP = Object.freeze([24, 128, 232]); // static lane · continuous lane · the 4-seat lane (frame 10)
export const ROW_PITCH = 16;
export const REQ_H = 10; // the request glyph's height
export const CHIP_W = 28; // tokenWidth('A')
export const CHIP_H = 24;
export const CHIP_PITCH = 36;
const QUEUE_DY = 17; // chip row, below the lane header
const SEATS_DY_QUEUE = 50; // first seat row below the lane top when the lane has a queue box
const SEATS_DY_PLAIN = 16;
export const EDGE_RIGHT = STAGE.w - 8;
const EPS = 1e-6;

export const clamp01 = (t) => Math.min(Math.max(t, 0), 1);
export const seg = (p, a, b) => clamp01((p - a) / (b - a)); // progress p remapped to the sub-phase [a, b]
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
// Each frame starts from the previous frame's end state: what leaves fades out and what arrives fades in during [0, HANDOFF].
export const HANDOFF = 0.15;
export const arriving = (p) => seg(p, 0, HANDOFF);
export const leaving = (p) => 1 - seg(p, 0, HANDOFF);

export const layer = (parent, opacity = 1) => G.svgEl('g', { opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);

// A plain labeled text mark (README lesson 15), styled like glyph labels.
export function note(parent, x, y, str, { cls = 'g-label', anchor = 'start', opacity = 1, halo = false } = {}) {
  if (opacity <= 0) return null;
  const g = G.svgEl('g', { class: 'glyph g-note', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  const t = G.svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, g);
  t.textContent = str;
  if (halo) {
    t.style.paintOrder = 'stroke';
    t.style.stroke = 'var(--surface)';
    t.style.strokeWidth = '3px';
    t.style.strokeLinejoin = 'round';
  }
  return g;
}

// A solid 1 px --ink-muted line (a rule or a connector; never dashed, P4-R15).
export function rule(parent, x1, y1, x2, y2, { opacity = 1 } = {}) {
  const line = G.svgEl('line', { x1, y1, x2, y2 }, G.svgEl('g', { class: 'glyph g-note', opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent));
  line.style.stroke = 'var(--ink-muted)';
  line.style.strokeWidth = '1';
  return line;
}

// A flat tinted bar (a seat track, a step block): fill only, no outline (outlines mean selection).
export function tint(parent, { x, y, w, h, rx = 2, opacity = 1, fill = 'var(--surface-2)' }) {
  if (w <= EPS || opacity <= 0) return null;
  const r = G.svgEl('rect', { x, y, width: w, height: h, rx, opacity: opacity < 1 ? opacity.toFixed(3) : null }, parent);
  r.style.fill = fill;
  return r;
}

export const headerY = (slot) => SLOT_TOP[slot] + 10;
export const chipY = (slot) => SLOT_TOP[slot] + QUEUE_DY;
export const chipX = (index) => X0 + index * CHIP_PITCH;
export const seatY = (slot, seat, hasQueue) => SLOT_TOP[slot] + (hasQueue ? SEATS_DY_QUEUE : SEATS_DY_PLAIN) + seat * ROW_PITCH;

// The step axis: cell numbers, or millisecond ticks (frames 8-9), with its unit name in the gutter.
export function axisSteps(parent, opacity = 1, count = 11) {
  if (opacity <= 0) return;
  note(parent, X0 - 14, AXIS_Y, 'step', { anchor: 'end', opacity });
  for (let s = 0; s < count; s += 1) note(parent, X0 + s * STEP_PX + STEP_PX / 2, AXIS_Y, String(s), { anchor: 'middle', opacity });
}

export function axisMs(parent, maxMs, opacity = 1, tickMs = 100) {
  if (opacity <= 0) return;
  note(parent, X0 - 14, AXIS_Y, 'ms', { anchor: 'end', opacity });
  for (let t = 0; t <= maxMs; t += tickMs) note(parent, X0 + (t * AXIS_PX) / maxMs, AXIS_Y, String(t), { anchor: 'middle', opacity });
}

// A request's bar as G.request segments (px relative to X0): prefill, one segment per decode step, then the idle hold.
// Bars are revealed up to `rb` px and the idle hatch up to `ri` px; a segment with no visible width is dropped.
export function rowSegments(row, rb = Infinity, ri = Infinity) {
  const segs = [];
  const push = (kind, from, to, limit) => {
    const end = Math.min(to, limit);
    if (end - from > EPS) segs.push({ kind, from, to: end });
  };
  push('prefill', row.prefill.from, row.prefill.to, rb);
  let previous = row.prefill.to;
  row.ticks.forEach((tick) => {
    push('decode', previous, tick, rb);
    previous = tick;
  });
  if (row.idle) push('idle', row.idle.from, row.idle.to, ri);
  return segs;
}

// Model rows (axis units) → px rows relative to X0.
export const pxRow = (row, scale) => ({
  ...row,
  prefill: { from: row.prefill.from * scale, to: row.prefill.to * scale },
  ticks: row.ticks.map((t) => t * scale),
  idle: row.idle ? { from: row.idle.from * scale, to: row.idle.to * scale } : null,
});

// Two px-row sets of the same requests, blended (frames 9, 10): positions, seats and ticks move; counts match by construction.
export function blendRows(a, b, t) {
  return a.map((ra) => {
    const rb = b.find((r) => r.id === ra.id);
    const idle = ra.idle && rb.idle ? { from: lerp(ra.idle.from, rb.idle.from, t), to: lerp(ra.idle.to, rb.idle.to, t) } : (t < 1 ? ra.idle : rb.idle);
    return {
      ...(t < 1 ? ra : rb),
      seat: lerp(ra.seat, rb.seat, t),
      prefill: { from: lerp(ra.prefill.from, rb.prefill.from, t), to: lerp(ra.prefill.to, rb.prefill.to, t) },
      ticks: ra.ticks.map((x, i) => lerp(x, rb.ticks[i], t)),
      idle,
    };
  });
}

const drawLetter = (parent, letter, x, y, opacity) => note(parent, x - 6, y + REQ_H / 2 + 4, letter, { cls: 'g-label', anchor: 'end', opacity, halo: true });

// The extent a row occupies on screen, for the followed request's selection mark.
function extent(segs) {
  return segs.length ? { from: segs[0].from, to: segs.at(-1).to } : null;
}

// One lane: header, optional queue box (stage) or wait bars (toy), seat tracks, request bars with their letters, queued chips.
//   rows: px rows (relative to X0)    chips: [{ id, x, y, opacity, label, labelOpacity }]    follow: the followed request's id
//   waits: [{ id, from, to }] px, drawn as grey queue bars above the seats (the toy's queue)
export function drawLane(parent, spec) {
  const { slot, title, right = '', rows = [], seats, hasQueue = true, rb = Infinity, ri = Infinity, chips = [], follow = 'D', opacity = 1, trackPx = AXIS_PX, seatLabels = true, markExtent = true, endLabel = false, extraSeatOpacity = 1, waits = [] } = spec;
  const top = spec.top ?? SLOT_TOP[slot];
  const seatsTop = spec.seatsTop ?? top + (hasQueue ? SEATS_DY_QUEUE : SEATS_DY_PLAIN);
  const yOf = (seat) => seatsTop + seat * ROW_PITCH;
  const g = layer(parent, opacity);
  note(g, 8, top + 10, title);
  if (right) note(g, EDGE_RIGHT, top + 10, right, { anchor: 'end' });
  if (hasQueue) {
    note(g, X0 - 8, top + QUEUE_DY + CHIP_H / 2 + 4, 'queue', { anchor: 'end' });
    tint(g, { x: X0 - 4, y: top + QUEUE_DY - 3, w: 4 * CHIP_PITCH + 6, h: CHIP_H + 6, rx: 4 });
  }
  drawWaits(g, waits, top);
  for (let s = 0; s < seats; s += 1) {
    const fade = s === seats - 1 ? extraSeatOpacity : 1;
    tint(g, { x: X0, y: yOf(s), w: trackPx, h: REQ_H, opacity: fade });
    if (seatLabels) note(g, X0 - 22, yOf(s) + REQ_H / 2 + 4, `seat ${s + 1}`, { anchor: 'end', opacity: fade });
  }
  rows.forEach((row) => drawRow(g, row, { y: yOf(row.seat), rb, ri, follow, markExtent, endLabel }));
  chips.forEach((chip) => drawChip(g, chip, follow));
  return g;
}

function drawRow(g, row, { y, rb, ri, follow, markExtent, endLabel }) {
  const segs = rowSegments(row, rb, ri);
  if (segs.length === 0) return;
  G.request(g, { x: X0, y, steps: segs, owner: row.id });
  drawLetter(g, row.id, X0 + segs[0].from, y, 1);
  if (row.id !== follow || !markExtent) return;
  const box = extent(segs);
  G.selectionMark(g, { x: X0 + box.from, y, w: box.to - box.from, h: REQ_H });
  if (endLabel && box.to >= row.ticks.at(-1) - EPS) note(g, X0 + box.to + 8, y + REQ_H / 2 + 4, spanLabel(row));
}

// The toy's queue: one thin bar per waiting request, from its arrival to its seat, two bars deep at most.
function drawWaits(g, waits, top) {
  const lanes = [];
  waits.forEach((w) => {
    const row = lanes.findIndex((end) => end <= w.from);
    const k = row === -1 ? lanes.length : row;
    lanes[k] = w.to;
    const y = top + 14 + k * ROW_PITCH;
    G.request(g, { x: X0, y, steps: [{ kind: 'queue', from: w.from, to: w.to }], owner: w.id });
    drawLetter(g, w.id, X0 + w.from, y, 1);
  });
}

function drawChip(parent, chip, follow) {
  const holder = layer(parent, chip.opacity ?? 1);
  G.token(holder, { x: chip.x, y: chip.y, text: chip.id, owner: chip.id });
  if (chip.id === follow) G.selectionMark(holder, { x: chip.x, y: chip.y, w: CHIP_W, h: CHIP_H });
  if (chip.label) note(parent, chip.x + CHIP_W + 6, chip.y + CHIP_H / 2 + 4, chip.label, { opacity: chip.labelOpacity ?? 1 });
}

// "Waiting" pulses once per step; at every whole step it is fully shown, so a frame at rest never depends on the phase.
export const pulse = (t) => 0.6 + 0.4 * Math.cos(Math.PI * t) ** 2;

// A lane run's clock: `pauses` are steps where admitted chips slide to their seats before time moves on.
// Returns { t, slide: { at, f } | null, slidDone: [steps whose slide finished] }.
export const SLIDE_LEN = 0.8; // virtual steps a slide takes
export function runClock(p, { from, to, pauses = [] }) {
  const list = pauses.filter((at) => at >= from && at <= to).sort((a, b) => a - b);
  let left = clamp01(p) * (to - from + list.length * SLIDE_LEN);
  let t = from;
  const slidDone = [];
  for (const at of list) {
    const run = Math.min(left, at - t);
    t += run;
    left -= run;
    if (t < at - EPS) return { t, slide: null, slidDone };
    if (left < SLIDE_LEN) return { t, slide: left > 0 ? { at, f: left / SLIDE_LEN } : null, slidDone };
    left -= SLIDE_LEN;
    slidDone.push(at);
  }
  return { t: t + Math.min(left, to - t), slide: null, slidDone };
}

// Queue chips for a lane at lane time `t`: queued (with D's "arrives" / "waiting" label), sliding to a seat, or gone.
//   seatedAt: admission steps whose chips are already seated before this frame.
export function queueChips({ rows, t, slot, seatedAt = [], slide = null, slidDone = [], seatsHasQueue = true }) {
  return rows.flatMap((row, i) => {
    if (seatedAt.includes(row.admitted) || slidDone.includes(row.admitted)) return [];
    const queued = { id: row.id, x: chipX(i), y: chipY(slot), opacity: 1 };
    const label = row.arrives > 0 ? (t < row.arrives ? `arrives at step ${row.arrives}` : 'waiting') : '';
    const labelOpacity = label === 'waiting' ? pulse(t) : 1;
    if (slide && slide.at === row.admitted) {
      const f = ease(slide.f);
      const toX = X0 + row.prefill.from * STEP_PX - CHIP_W / 2;
      const toY = seatY(slot, row.seat, seatsHasQueue) + REQ_H / 2 - CHIP_H / 2;
      return [{ ...queued, x: lerp(queued.x, toX, f), y: lerp(queued.y, toY, f), opacity: 1 - 0.8 * seg(slide.f, 0.7, 1), label: '' }];
    }
    return [{ ...queued, label, labelOpacity }];
  });
}
