// Systems glyphs: KV cache stacks, GPUs, racks and request timelines.
import { svgEl, group, text, hatchRect } from './core.js';

const clamp01 = (v) => Math.min(Math.max(Number(v) || 0, 0), 1);

export function kvStack(parent, { x, y, count, tile = 14, highlight = [], label }) {
  const gap = 2;
  const g = group(parent, 'g-kv', x, y, { role: 'img', 'aria-label': `KV cache: ${count} tokens` });
  if (label) text(g, 0, -8, label, 'g-label');
  text(g, -8, tile / 2, 'K', 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  text(g, -8, tile + gap + tile / 2, 'V', 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  for (let i = 0; i < count; i += 1) {
    const cx = i * (tile + gap);
    svgEl('rect', { class: 'g-k', x: cx, y: 0, width: tile, height: tile, rx: 2 }, g);
    svgEl('rect', { class: 'g-v', x: cx, y: tile + gap, width: tile, height: tile, rx: 2 }, g);
  }
  highlight.forEach((i) => svgEl('rect', { class: 'g-hl', x: i * (tile + gap) - 1.5, y: -1.5, width: tile + 3, height: 2 * tile + gap + 3, rx: 3 }, g));
  return g;
}

export function gpu(parent, { x, y, w = 96, h = 72, memFill = 0, label }) {
  const fill = clamp01(memFill);
  const g = group(parent, 'g-gpu', x, y, { role: 'img', 'aria-label': `${label ?? 'GPU'}: memory ${Math.round(fill * 100)}% full` });
  const dieW = Math.round(w * 0.68);
  const dieH = h - 18;
  svgEl('rect', { class: 'g-frame', width: w, height: h, rx: 5 }, g);
  svgEl('rect', { class: 'g-frame', x: 6, y: 6, width: dieW, height: dieH - 12, rx: 2 }, g);
  const cols = 4;
  const rows = 3;
  const cw = (dieW - 10) / cols;
  const ch = (dieH - 22) / rows;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      svgEl('rect', { class: 'g-sm', x: 8 + c * cw + 1, y: 8 + r * ch + 1, width: cw - 2, height: ch - 2, rx: 1 }, g);
    }
  }
  const hbmX = dieW + 12;
  const hbmW = w - hbmX - 6;
  for (let i = 0; i < 2; i += 1) {
    svgEl('rect', { class: 'g-hbm', x: hbmX, y: 6 + i * ((dieH - 12) / 2), width: hbmW, height: (dieH - 12) / 2 - 3, rx: 2 }, g);
  }
  const barY = h - 11;
  svgEl('rect', { class: 'g-mem-track', x: 6, y: barY, width: w - 12, height: 6, rx: 3 }, g);
  svgEl('rect', { class: 'g-mem-fill', x: 6, y: barY, width: (w - 12) * fill, height: 6, rx: 3 }, g);
  if (label) text(g, w / 2, h + 12, label, 'g-label', { 'text-anchor': 'middle' });
  return g;
}

export function rack(parent, { x, y, gpus = 8, linkWidth = 2, label, cols = 4 }) {
  const tile = 16;
  const gap = 10;
  const rows = Math.ceil(gpus / cols);
  const w = cols * (tile + gap) + gap;
  const h = rows * (tile + gap) + gap;
  const g = group(parent, 'g-rack', x, y, { role: 'img', 'aria-label': `${label ?? 'node'}: ${gpus} GPUs` });
  svgEl('rect', { class: 'g-frame', width: w, height: h, rx: 6 }, g);
  for (let r = 0; r < rows; r += 1) {
    const cy = gap + r * (tile + gap) + tile / 2;
    svgEl('line', { class: 'g-link', x1: gap + tile / 2, y1: cy, x2: gap + (cols - 1) * (tile + gap) + tile / 2, y2: cy, 'stroke-width': linkWidth }, g);
    if (r > 0) {
      svgEl('line', { class: 'g-link', x1: gap + tile / 2, y1: cy - (tile + gap), x2: gap + tile / 2, y2: cy, 'stroke-width': linkWidth }, g);
    }
  }
  for (let i = 0; i < gpus; i += 1) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const gx = gap + c * (tile + gap);
    const gy = gap + r * (tile + gap);
    svgEl('rect', { class: 'g-frame', x: gx, y: gy, width: tile, height: tile, rx: 2 }, g);
    svgEl('rect', { class: 'g-sm', x: gx + 3, y: gy + 3, width: tile - 6, height: tile - 6, rx: 1 }, g);
  }
  if (label) text(g, w / 2, h + 12, label, 'g-label', { 'text-anchor': 'middle' });
  return g;
}

export function request(parent, { x, y, prefill, decode, unit = 6, label }) {
  const h = 10;
  const prefillW = Math.max(0, prefill) * unit;
  const total = prefillW + Math.max(0, decode) * unit;
  const g = group(parent, 'g-request', x, y, { role: 'img', 'aria-label': `${label ?? 'request'}: prefill ${prefill} tokens, decode ${decode} tokens` });
  if (label) text(g, -8, h / 2, label, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  svgEl('line', { class: 'g-rail', x1: 0, y1: h / 2, x2: total, y2: h / 2 }, g);
  if (prefillW > 0) svgEl('rect', { class: 'g-prefill', width: prefillW, height: h, rx: 2 }, g);
  for (let i = 1; i <= decode; i += 1) {
    const tx = prefillW + i * unit;
    svgEl('line', { class: 'g-decode', x1: tx, y1: 1, x2: tx, y2: h - 1 }, g);
  }
  return g;
}

// ---- laneTimeline (gpu-primer §4, ruling P3-R9) ----
// Horizontal lanes on one shared time axis. Kinds fill with the course's meanings: compute, memory and comm
// (--sem-*), forward and backward (--carry-activation, --carry-gradient: what travels in each pass), idle
// (hatched: the resource waits) and lost (compute fill under the hatch: work done that doesn't count).
const KINDS = ['compute', 'memory', 'comm', 'idle', 'forward', 'backward', 'lost'];
const KIND_TEXT = { compute: 'compute', memory: 'memory', comm: 'communication', idle: 'idle', forward: 'forward', backward: 'backward', lost: 'lost work' };
const HATCHED = new Set(['idle', 'lost']);
const LANE = { h: 24, gap: 8, tickRow: 16, gapRow: 22 };
const SEG_GAP = 3; // between touching segments, as between cells
const CHAR_W = 6.6;
const MIN_INSIDE = 36; // a segment prints its text inside from 36 px (gpu-primer §4), when the text fits
const CAP_ARROW = 8;
const TIME_EPS = 1e-9;
const isNum = Number.isFinite;
const failLane = (msg) => { throw new RangeError(`glyphs.laneTimeline: ${msg}`); };

function checkLanes({ w, lanes, scale, cap, ticks, gaps }) {
  if (!(isNum(w) && w > 0)) failLane(`w must be a finite number > 0, got ${w}`);
  if (!Array.isArray(lanes) || lanes.length === 0) failLane('lanes must be a non-empty array');
  if (scale != null && !(isNum(scale) && scale > 0)) failLane(`scale must be a finite number > 0 (px per time unit), got ${scale}`);
  if (cap != null && !(isNum(cap.at) && cap.at > 0 && typeof cap.label === 'string')) failLane('cap must be { at > 0, label }');
  lanes.forEach((lane) => (lane.segments ?? []).forEach((s) => {
    if (!KINDS.includes(s.kind)) failLane(`kind must be one of ${KINDS.join(', ')}, got ${s.kind}`);
    if (!(isNum(s.from) && isNum(s.to) && s.from >= 0 && s.from < s.to)) failLane(`a segment needs 0 ≤ from < to, got ${s.from} → ${s.to}`);
  }));
  if (!Array.isArray(ticks) || !Array.isArray(gaps)) failLane('ticks and gaps must be arrays');
  gaps.forEach((gp) => { if (!(isNum(gp.from) && isNum(gp.to) && gp.from >= 0 && gp.from < gp.to)) failLane(`a gap needs 0 ≤ from < to, got ${gp.from} → ${gp.to}`); });
}

function fitScale({ lanes, cap, ticks, gaps }, track) {
  const ends = [...lanes.flatMap((l) => (l.segments ?? []).map((s) => (cap ? Math.min(s.to, cap.at) : s.to))), ...ticks.map((t) => t.t), ...gaps.map((gp) => gp.to)];
  return track / Math.max(...ends.filter(isNum), TIME_EPS);
}

function segmentText(seg, { x, width, capped, labels, w }) {
  if (labels === false || !seg.label) return { text: null, textX: null, textAnchor: null, textInside: false };
  const text = String(seg.label);
  const tw = text.length * CHAR_W;
  if (width >= MIN_INSIDE && tw + 8 <= width) return capped ? { text, textX: x + 6, textAnchor: 'start', textInside: true } : { text, textX: x + width / 2, textAnchor: 'middle', textInside: true };
  const right = x + width + 4;
  return right + tw <= w ? { text, textX: right, textAnchor: 'start', textInside: false } : { text, textX: x - 4, textAnchor: 'end', textInside: false };
}

function placeSegment(seg, ctx) {
  const { gutter, scale, capX, lane, y, w, labels, starts } = ctx;
  const x = gutter + seg.from * scale;
  const capped = capX != null && gutter + seg.to * scale > capX + TIME_EPS;
  const touches = starts.some((s) => Math.abs(s - seg.to) < TIME_EPS);
  const end = capped ? capX - CAP_ARROW : gutter + seg.to * scale - (touches ? SEG_GAP : 0);
  const width = Math.max(1, end - x);
  const kindText = KIND_TEXT[seg.kind];
  const title = labels !== false && seg.label ? `${lane.label}: ${seg.label} (${kindText})` : `${lane.label}: ${kindText}`;
  return {
    lane: ctx.index, kind: seg.kind, hatched: HATCHED.has(seg.kind), x, y, width, height: LANE.h, title, capped,
    arrow: capped ? { base: capX - CAP_ARROW, tip: capX } : null, capLabel: capped ? ctx.capLabel : null,
    ...segmentText(seg, { x, width, capped, labels, w }),
  };
}

// Geometry only (pure). `scale` is px per time unit (default: the latest end fills the track); the lane labels
// take a gutter on the left; `cap` cuts every lane at `cap.at` with an arrow and its printed label.
export function laneTimelineLayout({ w, lanes, scale = null, cap = null, ticks = [], gaps = [], labels = true }) {
  checkLanes({ w, lanes, scale, cap, ticks, gaps });
  const gutter = Math.round(Math.max(0, ...lanes.map((l) => String(l.label ?? '').length)) * CHAR_W) + 6;
  const track = w - gutter;
  const k = scale ?? fitScale({ lanes, cap, ticks, gaps }, track);
  const capX = cap ? gutter + cap.at * k : null;
  if (capX != null && capX > w + TIME_EPS) failLane(`cap at ${cap.at} lies past the track`);
  const top = ticks.some((t) => t.label) ? LANE.tickRow : ticks.length ? 4 : 0;
  const lanesBottom = top + lanes.length * (LANE.h + LANE.gap) - LANE.gap;
  const placed = lanes.map((lane, index) => ({ label: String(lane.label ?? ''), y: top + index * (LANE.h + LANE.gap) }));
  const segments = lanes.flatMap((lane, index) => {
    const segs = (lane.segments ?? []).filter((s) => !(cap && s.from >= cap.at));
    if (!cap && segs.some((s) => s.to * k > track + 1e-6)) failLane(`lane "${lane.label}" runs past the track; pass cap or a smaller scale`);
    const starts = segs.map((s) => s.from);
    return segs.map((seg) => placeSegment(seg, { gutter, scale: k, capX, capLabel: cap?.label, lane: placed[index], index, y: placed[index].y, w, labels, starts }));
  });
  const onTrack = (t) => isNum(t) && t >= 0 && gutter + t * k <= w + 1e-6;
  ticks.forEach((t) => { if (!onTrack(t.t)) failLane(`tick t must lie on the track, got ${t.t}`); });
  gaps.forEach((gp) => { if (!onTrack(gp.to)) failLane(`gap ${gp.from} → ${gp.to} runs past the track`); });
  return {
    w, gutter, track, scale: k, top, lanesBottom,
    height: lanesBottom + (gaps.some((gp) => gp.label) ? LANE.gapRow : 0) + (ticks.length ? 4 : 0),
    lanes: placed, segments,
    ticks: ticks.map((t) => ({ x: gutter + t.t * k, label: t.label ?? '' })),
    gaps: gaps.map((gp) => ({ x: gutter + gp.from * k, width: (gp.to - gp.from) * k, label: gp.label ?? '' })),
  };
}

function drawSegment(g, s) {
  const r = svgEl('rect', { class: `g-lane-seg g-lane--${s.kind}`, x: num(s.x), y: s.y, width: num(s.width), height: s.height, rx: 2 }, g);
  svgEl('title', {}, r).textContent = s.title;
  if (s.hatched) hatchRect(g, { x: num(s.x), y: s.y, width: num(s.width), height: s.height, rx: 2 });
  if (s.arrow) {
    const mid = s.y + s.height / 2;
    svgEl('polygon', { class: `g-lane-arrow g-lane--${s.kind}`, points: `${num(s.arrow.base)},${s.y} ${num(s.arrow.tip)},${mid} ${num(s.arrow.base)},${s.y + s.height}` }, g);
    if (s.capLabel) text(g, num(s.arrow.base - 4), mid, s.capLabel, 'g-label g-lane-text g-lane-cap', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  }
  if (s.text) text(g, num(s.textX), s.y + s.height / 2, s.text, `g-lane-text${s.textInside ? '' : ' g-label'}`, { 'text-anchor': s.textAnchor, 'dominant-baseline': 'central' });
}

const num = (v) => Number(v.toFixed(2));

// Lanes sharing one clock (gpu-primer frame 7, parallelism frames 6–8, cluster-topology, scale-reliability).
// Every segment carries a <title>; with labels: false (dense grids) nothing is printed on the lanes.
export function laneTimeline(parent, { x, y, w = 568, lanes, scale = null, cap = null, ticks = [], gaps = [], labels = true, label = 'timeline' }) {
  const L = laneTimelineLayout({ w, lanes, scale, cap, ticks, gaps, labels });
  const g = group(parent, 'g-lanes', x, y, { role: 'img', 'aria-label': `${label}: ${L.lanes.map((l) => l.label).join(', ')}` });
  L.lanes.forEach((lane) => {
    svgEl('rect', { class: 'g-lane-track', x: L.gutter, y: lane.y, width: num(L.track), height: LANE.h, rx: 2 }, g);
    if (lane.label) text(g, L.gutter - 5, lane.y + LANE.h / 2, lane.label, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  });
  L.segments.forEach((s) => drawSegment(g, s));
  L.ticks.forEach((t) => {
    svgEl('line', { class: 'g-lane-tick', x1: num(t.x), y1: L.top - 4, x2: num(t.x), y2: L.lanesBottom + 4 }, g);
    if (t.label) text(g, num(t.x), L.top - 6, t.label, 'g-label', { 'text-anchor': 'middle' });
  });
  L.gaps.forEach((gp) => {
    const b = L.lanesBottom + 3;
    svgEl('path', { class: 'g-bracket', d: `M${num(gp.x)} ${b}v4H${num(gp.x + gp.width)}v-4` }, g);
    if (gp.label) text(g, num(gp.x + gp.width / 2), b + 16, gp.label, 'g-label', { 'text-anchor': 'middle' });
  });
  return g;
}
