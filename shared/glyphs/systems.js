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

const SM_TILES = 12; // the drawn SM grid: 4 × 3 tiles standing for the chip's SMs

function checkLit(litSms) {
  if (litSms == null) return null;
  if (!Array.isArray(litSms) || !litSms.every((i) => Number.isInteger(i) && i >= 0 && i < SM_TILES)) throw new RangeError(`glyphs.gpu: litSms must be SM tile indices 0–${SM_TILES - 1}, got ${JSON.stringify(litSms)}`);
  return new Set(litSms);
}

// `showMem: false` (S3) drops the memory bar for pages that show no fullness; `litSms` (S3) lights those SM tiles
// (indices 0–11 of the drawn 4 × 3 grid) and titles the count, e.g. "20 of 132 SMs" drawn as 2 of 12 tiles.
export function gpu(parent, { x, y, w = 96, h = 72, memFill = 0, label, showMem = true, litSms = null }) {
  const fill = clamp01(memFill);
  const lit = checkLit(litSms);
  const g = group(parent, `g-gpu${lit ? ' g-gpu--lit' : ''}`, x, y, { role: 'img', 'aria-label': showMem ? `${label ?? 'GPU'}: memory ${Math.round(fill * 100)}% full` : (label ?? 'GPU') });
  if (lit) svgEl('title', {}, g).textContent = `${label ?? 'GPU'}: SM grid, ${lit.size} of ${SM_TILES} tiles lit`;
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
      svgEl('rect', { class: lit?.has(r * cols + c) ? 'g-sm g-sm--lit' : 'g-sm', x: 8 + c * cw + 1, y: 8 + r * ch + 1, width: cw - 2, height: ch - 2, rx: 1 }, g);
    }
  }
  const hbmX = dieW + 12;
  const hbmW = w - hbmX - 6;
  for (let i = 0; i < 2; i += 1) {
    svgEl('rect', { class: 'g-hbm', x: hbmX, y: 6 + i * ((dieH - 12) / 2), width: hbmW, height: (dieH - 12) / 2 - 3, rx: 2 }, g);
  }
  if (showMem) {
    const barY = h - 11;
    svgEl('rect', { class: 'g-mem-track', x: 6, y: barY, width: w - 12, height: 6, rx: 3 }, g);
    svgEl('rect', { class: 'g-mem-fill', x: 6, y: barY, width: (w - 12) * fill, height: 6, rx: 3 }, g);
  }
  if (label) text(g, w / 2, h + 12, label, 'g-label', { 'text-anchor': 'middle' });
  return g;
}

// `labels` (S3, cluster-topology frame 8): one string per GPU, printed in its tile in place of the SM square
// ('' keeps the square). Link thickness never encodes an amount (P3-R8): pages keep the default linkWidth.
export function rack(parent, { x, y, gpus = 8, linkWidth = 2, label, cols = 4, labels = null }) {
  if (labels != null && !(Array.isArray(labels) && labels.length === gpus && labels.every((t) => typeof t === 'string'))) throw new RangeError(`glyphs.rack: labels must be ${gpus} strings, one per GPU`);
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
    if (labels?.[i]) text(g, gx + tile / 2, gy + tile / 2, labels[i], 'g-rack-label', { 'text-anchor': 'middle', 'dominant-baseline': 'central' });
    else svgEl('rect', { class: 'g-sm', x: gx + 3, y: gy + 3, width: tile - 6, height: tile - 6, rx: 1 }, g);
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
}

// Drawn after every segment, so a label printed beside a short segment is never covered by its neighbor.
function drawSegmentText(g, s) {
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
  L.segments.forEach((s) => drawSegmentText(g, s));
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

// ---- bitLayout (gpu-primer §4): a number format as bit cells grouped and labeled by role ----
const ROLES = ['sign', 'exponent', 'mantissa', 'scale'];
const failBits = (msg) => { throw new RangeError(`glyphs.bitLayout: ${msg}`); };
const fieldText = ({ role, bits }) => ({ sign: 'S', exponent: `E${bits}`, mantissa: `M${bits}`, scale: 'scale' })[role];

// '1/8/7' (a FORMATS entry's layout) → sign, exponent and mantissa fields.
export function bitFields(layout) {
  const parts = String(layout).split('/').map(Number);
  if (parts.length !== 3 || !parts.every((n) => Number.isInteger(n) && n >= 0) || parts[0] !== 1) failBits(`layout must read "sign/exponent/mantissa" with a 1-bit sign, got "${layout}"`);
  return parts.map((bits, i) => ({ role: ROLES[i], bits })).filter((f) => f.bits > 0);
}

// The bracket's text is centered under the scale field, slid left or right so it never runs past the row.
function bracketOf(scale, sharedBy, width) {
  const text = `shared by ${sharedBy} numbers`;
  const half = (text.length * CHAR_W) / 2;
  const center = scale.x + scale.width / 2;
  return { x0: scale.x, x1: scale.x + scale.width, text, textX: Math.max(Math.min(center, width - half), Math.min(half, width / 2)) };
}

// Geometry only (pure): one cell per bit at `bitW` px; per field its x, width and printed text (S, E8, M7, scale);
// with `sharedBy`, the bracket under the last scale field.
export function bitLayoutLayout({ fields = null, format = null, bitW = 14, sharedBy = null }) {
  if (!fields && !format) failBits('pass fields or format (a FORMATS entry)');
  const list = fields ?? bitFields(format.layout);
  if (!(isNum(bitW) && bitW > 0)) failBits(`bitW must be a finite number > 0, got ${bitW}`);
  list.forEach((f) => {
    if (!ROLES.includes(f.role)) failBits(`role must be one of ${ROLES.join(', ')}, got ${f.role}`);
    if (!(Number.isInteger(f.bits) && f.bits >= 1)) failBits(`bits must be an integer ≥ 1, got ${f.bits}`);
  });
  const starts = list.map((_, i) => list.slice(0, i).reduce((s, f) => s + f.bits, 0));
  // A field's text prints when it fits its field (+ 2 px); otherwise only its <title> names it (e.g. M1 at bitW 11).
  const fits = (f) => fieldText(f).length * CHAR_W <= f.bits * bitW + 2;
  const placed = list.map((f, i) => ({ role: f.role, bits: f.bits, x: starts[i] * bitW, width: f.bits * bitW, text: fits(f) ? fieldText(f) : null }));
  const scale = placed.filter((f) => f.role === 'scale').at(-1);
  if (sharedBy != null && !scale) failBits('sharedBy needs a scale field');
  const width = placed.reduce((s, f) => s + f.width, 0);
  return {
    bitW, fields: placed, width,
    cells: placed.flatMap((f) => Array.from({ length: f.bits }, (_, b) => ({ x: f.x + b * bitW, role: f.role }))),
    bracket: sharedBy != null ? bracketOf(scale, sharedBy, width) : null,
  };
}

// A row of bit cells colored by role (never by value: formats are bit fields, not numbers), each field labeled
// under its cells; `label` (e.g. "BF16") at the left; `sharedBy` draws the "shared by N numbers" bracket.
export function bitLayout(parent, { x, y, fields = null, format = null, bitW = 14, label, sharedBy = null }) {
  const L = bitLayoutLayout({ fields, format, bitW, sharedBy });
  const spoken = L.fields.map((f) => `${f.role} ${f.bits}`).join(', ');
  const g = group(parent, 'g-bits', x, y, { role: 'img', 'aria-label': `${label ? `${label}: ` : ''}${spoken}${L.bracket ? `; ${L.bracket.text}` : ''}` });
  if (label) text(g, -8, bitW / 2, label, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  L.fields.forEach((f) => {
    const field = svgEl('g', { class: `g-bit-field g-bit--${f.role}` }, g);
    svgEl('title', {}, field).textContent = `${f.role}: ${f.bits} bit${f.bits === 1 ? '' : 's'}`;
    L.cells.filter((c) => c.x >= f.x && c.x < f.x + f.width).forEach((c) => svgEl('rect', { class: 'g-bit', x: c.x + 0.5, y: 0.5, width: bitW - 1, height: bitW - 1, rx: 1.5 }, field));
    if (f.text) text(g, f.x + f.width / 2, bitW + 11, f.text, 'g-label g-bit-text', { 'text-anchor': 'middle' });
  });
  if (L.bracket) {
    const b = bitW + 16;
    svgEl('path', { class: 'g-bracket', d: `M${L.bracket.x0} ${b}v4H${L.bracket.x1}v-4` }, g);
    text(g, num(L.bracket.textX), b + 16, L.bracket.text, 'g-label', { 'text-anchor': 'middle' });
  }
  return g;
}
