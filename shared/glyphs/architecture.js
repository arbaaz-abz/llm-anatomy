// Glyphs accepted with the decoder-anatomy storyboard (§4, §13): the selection mark, image patches,
// the residual adder and the block stack; plus rope's dial (shared patch S1). Pure DOM builders; nothing runs at import time.
import { svgEl, group, text, block } from './core.js';

const STATES = new Set(['idle', 'active', 'dim']);
const finite = (...values) => values.every(Number.isFinite);

// The one selection outline: "the item we're following". Never a quantity.
export function selectionMark(parent, { x, y, w, h, pad = 2, rx = 5 }) {
  if (!finite(x, y, w, h) || !(w > 0) || !(h > 0)) throw new RangeError(`glyphs.selectionMark: x, y, w, h must be finite with w, h > 0, got ${x}, ${y}, ${w}, ${h}`);
  const g = group(parent, 'g-select', 0, 0, { 'aria-hidden': 'true' });
  svgEl('rect', { x: x - pad, y: y - pad, width: w + 2 * pad, height: h + 2 * pad, rx }, g);
  return g;
}

export function pixelFill(p) {
  if (typeof p === 'number' && p >= 0 && p <= 1) {
    const v = Math.round(p * 255);
    return `rgb(${v} ${v} ${v})`;
  }
  if (Array.isArray(p) && p.length === 3 && p.every((c) => Number.isInteger(c) && c >= 0 && c <= 255)) return `rgb(${p.join(' ')})`;
  throw new RangeError(`glyphs.pixelFill: a pixel is a grey level 0–1 or an [r, g, b] triple of 0–255, got ${JSON.stringify(p)}`);
}

// One image patch as an input piece: a 4 × 4 crop of real pixels (image content, not values).
export function patch(parent, { x, y, pixels, index, state = 'idle', size = 24 }) {
  if (!Array.isArray(pixels) || pixels.length !== 16) throw new RangeError(`glyphs.patch: pixels must hold 16 entries (a 4 × 4 crop), got ${pixels?.length}`);
  if (!STATES.has(state)) throw new RangeError(`glyphs.patch: state must be idle, active or dim, got ${state}`);
  const fills = pixels.map(pixelFill);
  const g = group(parent, `g-patch g-patch--${state}`, x, y, { role: 'img', 'aria-label': index == null ? 'image patch' : `image patch ${index}` });
  svgEl('rect', { class: 'g-frame', width: size, height: size, rx: 5 }, g);
  const px = (size - 6) / 4;
  fills.forEach((fill, i) => {
    const r = svgEl('rect', { class: 'g-pixel', x: 3 + (i % 4) * px, y: 3 + Math.floor(i / 4) * px, width: px, height: px }, g);
    r.style.fill = fill;
  });
  if (index != null) text(g, size, size + 9, index, 'g-sub', { 'text-anchor': 'end' });
  return g;
}

// The residual add: a ⊕ on the stream lane. It marks where a sub-layer's result is added back.
export function adder(parent, { x, y, r = 9 }) {
  const g = group(parent, 'g-adder', x, y, { role: 'img', 'aria-label': 'add back into the residual stream' });
  svgEl('circle', { r }, g);
  const arm = r * 0.55;
  svgEl('path', { d: `M${-arm} 0H${arm}M0 ${-arm}V${arm}` }, g);
  return g;
}

export function blockStackLayout({ count, shown = 2 }) {
  if (!Number.isInteger(count) || count < 1) throw new RangeError(`glyphs.blockStack: count must be an integer ≥ 1, got ${count}`);
  if (!Number.isInteger(shown) || shown < 1) throw new RangeError(`glyphs.blockStack: shown must be an integer ≥ 1, got ${shown}`);
  const blocks = (n) => Array.from({ length: n }, (_, i) => ({ kind: 'block', index: i + 1 }));
  if (count <= shown + 1) return blocks(count);
  return [...blocks(shown), { kind: 'collapse', hidden: count - shown - 1 }, { kind: 'block', index: count }];
}

const STACK = { lane: 12, inset: 26, padTop: 16, pad: 8, halfH: 26, halfGap: 6, gap: 12, collapseH: 24, adderR: 8 };

function drawStackBlock(g, { top, w, blockH, index, label, halves, active, residual }) {
  const left = STACK.inset - 8;
  svgEl('rect', { class: 'g-stack-frame', x: left, y: top, width: w - left, height: blockH, rx: 6 }, g);
  text(g, left + 6, top + 11, label, 'g-sub');
  halves.forEach((name, i) => {
    const hy = top + STACK.padTop + i * (STACK.halfH + STACK.halfGap);
    const cy = hy + STACK.halfH / 2;
    const state = active?.block === index && active?.half === i ? 'active' : 'idle';
    block(g, { x: STACK.inset, y: hy, w: w - STACK.inset - STACK.pad, h: STACK.halfH, label: name, state });
    if (!residual) return;
    svgEl('line', { class: 'g-link', x1: STACK.lane + STACK.adderR, y1: cy, x2: STACK.inset, y2: cy }, g);
    adder(g, { x: STACK.lane, y: cy, r: STACK.adderR });
  });
}

// "This block repeated N times, with a residual lane through all of them." Count is printed, never a height.
// `lastLabel` (optional) names the last drawn block, e.g. 'block N' for a stack that stands for any model.
export function blockStack(parent, { x, y, w = 220, count, shown = 2, residual = true, halves = ['attention', 'MLP'], active = null, countLabel = `⋮ × ${count}`, lastLabel = `block ${count}` }) {
  const rows = blockStackLayout({ count, shown });
  const blockH = STACK.padTop + halves.length * STACK.halfH + (halves.length - 1) * STACK.halfGap + STACK.pad;
  const heights = rows.map((row) => (row.kind === 'block' ? blockH : STACK.collapseH));
  const tops = heights.map((_, i) => heights.slice(0, i).reduce((sum, h) => sum + h + STACK.gap, 0));
  const height = tops[tops.length - 1] + heights[heights.length - 1];
  const g = group(parent, 'g-stack', x, y, { role: 'img', 'aria-label': `${count} blocks, each ${halves.join(' then ')}` });
  if (residual) svgEl('line', { class: 'g-lane', x1: STACK.lane, y1: 0, x2: STACK.lane, y2: height }, g);
  rows.forEach((row, i) => {
    const label = row.index === count ? lastLabel : `block ${row.index}`;
    if (row.kind === 'block') drawStackBlock(g, { top: tops[i], w, blockH, index: row.index, label, halves, active, residual });
    else text(g, STACK.inset, tops[i] + STACK.collapseH / 2, countLabel, 'g-label g-collapse', { 'dominant-baseline': 'central' });
  });
  return g;
}

// ---- dial (rope §4, accepted with conditions a–j; reused by decoder-recap and midtraining) ----
const TURN = 2 * Math.PI;
const TICK_TITLES = ['0 rad', 'π/2 rad', 'π rad', '3π/2 rad'];
const TICK_INNER = 0.82; // ticks run from 82 % of the radius to the rim
const num = (v) => String(Number(v.toFixed(3)) || 0); // short SVG coordinates, never "-0"
const isRange = (v) => Array.isArray(v) && v.length === 2 && finite(...v) && v[1] >= v[0];
// Screen point at `angle` radians (counter-clockwise from 3 o'clock; SVG y grows downward).
const polar = (len, angle) => ({ x: len * Math.cos(angle), y: -len * Math.sin(angle) });

function checkDial({ r, vector, angle, seen, reached, scale }) {
  if (!(Number.isFinite(r) && r > 0)) throw new RangeError(`glyphs.dial: r must be a finite number > 0, got ${r}`);
  if (vector != null && !(Array.isArray(vector) && vector.length === 2 && finite(...vector))) throw new RangeError(`glyphs.dial: vector must be a pair of finite numbers, got ${JSON.stringify(vector)}`);
  if (!Number.isFinite(angle)) throw new RangeError(`glyphs.dial: angle must be finite (radians), got ${angle}`);
  if (seen != null && !isRange(seen)) throw new RangeError(`glyphs.dial: seen must be [from, to] radians with from ≤ to, got ${JSON.stringify(seen)}`);
  if (reached != null && !isRange(reached)) throw new RangeError(`glyphs.dial: reached must be [from, to] radians with from ≤ to, got ${JSON.stringify(reached)}`);
  if (scale != null && !(Number.isFinite(scale) && scale > 0)) throw new RangeError(`glyphs.dial: scale must be a finite number > 0, got ${scale}`);
}

function sectorPath(r, from, to) {
  const a = polar(r, from);
  const b = polar(r, to);
  const large = to - from > Math.PI ? 1 : 0;
  return `M0 0L${num(a.x)} ${num(a.y)}A${r} ${r} 0 ${large} 0 ${num(b.x)} ${num(b.y)}Z`;
}

function turnsText(angle) {
  if (Math.abs(angle) <= TURN) return null;
  const turns = (Math.abs(angle) / TURN).toFixed(1);
  return `${angle < 0 ? '−' : ''}${turns} turns`;
}

// Pure geometry of a dial: the hand (length = |pair| on `scale`, direction = the pair's own direction + angle),
// the four ticks, the "seen in training" wedge, the ghost hand at the largest angle reached.
export function dialLayout({ r = 34, vector = null, angle = 0, seen = null, reached = null, scale = null }) {
  checkDial({ r, vector, angle, seen, reached, scale });
  const magnitude = vector ? Math.hypot(vector[0], vector[1]) : 1;
  const base = vector ? Math.atan2(vector[1], vector[0]) : 0;
  const length = magnitude === 0 ? 0 : (r * magnitude) / (scale ?? magnitude);
  const full = seen ? seen[1] - seen[0] >= TURN : false;
  return {
    r,
    hand: polar(length, base + angle),
    magnitudeLabel: vector ? `|·| = ${magnitude.toFixed(2)}` : null,
    turnsLabel: turnsText(angle),
    ticks: TICK_TITLES.map((title, i) => ({ ...polar(r, (i * TURN) / 4), inner: polar(r * TICK_INNER, (i * TURN) / 4), title })),
    seen: seen ? { full, path: full ? null : sectorPath(r, base + seen[0], base + seen[1]) } : null,
    ghost: reached ? polar(length, base + reached[1]) : null,
    neverSeen: Boolean(seen && reached && !full && reached[1] > seen[1]),
  };
}

function drawDialFace(g, layout) {
  svgEl('circle', { class: 'g-frame', r: layout.r }, g);
  if (layout.seen?.full) svgEl('circle', { class: 'g-dial-seen', r: layout.r }, g);
  else if (layout.seen) svgEl('path', { class: 'g-dial-seen', d: layout.seen.path }, g);
  layout.ticks.forEach((t) => {
    const tick = svgEl('line', { class: 'g-dial-tick', x1: num(t.inner.x), y1: num(t.inner.y), x2: num(t.x), y2: num(t.y) }, g);
    svgEl('title', {}, tick).textContent = t.title;
  });
}

// One pair of numbers as a clock hand (rope §4). Hand in --ink; the seen wedge is a pale fill, never an outline;
// the selection outline is the page's G.selectionMark. Pages interpolate `angle` / `reached` with progress.
export function dial(parent, { x, y, r = 34, vector = null, angle = 0, seen = null, reached = null, scale = null, label }) {
  const layout = dialLayout({ r, vector, angle, seen, reached, scale });
  const spoken = [label, layout.magnitudeLabel, `turned ${angle.toFixed(2)} rad`].filter(Boolean).join(', ');
  const g = group(parent, 'g-dial', x, y, { role: 'img', 'aria-label': spoken });
  drawDialFace(g, layout);
  if (layout.ghost) {
    svgEl('line', { class: 'g-dial-ghost', x1: 0, y1: 0, x2: num(layout.ghost.x), y2: num(layout.ghost.y) }, g);
    if (layout.neverSeen) {
      const at = polar(r + 6, Math.atan2(-layout.ghost.y, layout.ghost.x)); // just outside the rim, on the ghost's line
      text(g, num(at.x), num(at.y), 'never seen', 'g-label g-dial-never', { 'text-anchor': at.x >= 0 ? 'start' : 'end', 'dominant-baseline': 'central' });
    }
  }
  svgEl('line', { class: 'g-dial-hand', x1: 0, y1: 0, x2: num(layout.hand.x), y2: num(layout.hand.y) }, g);
  svgEl('circle', { class: 'g-dial-hub', r: 2.5 }, g);
  if (label) text(g, 0, -r - 8, label, 'g-label', { 'text-anchor': 'middle' });
  if (layout.magnitudeLabel) text(g, 0, r + 14, layout.magnitudeLabel, 'g-label', { 'text-anchor': 'middle' });
  if (layout.turnsLabel) text(g, r + 6, 0, layout.turnsLabel, 'g-label g-dial-turns', { 'dominant-baseline': 'central' });
  return g;
}
