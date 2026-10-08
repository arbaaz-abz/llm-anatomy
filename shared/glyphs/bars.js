// Stacked bars: shareBar (categorical parts, decoder-anatomy §4) and memBar (paged-attention).
// One geometry (barSegments) under both, so the two never drift apart. Plus `bars` (moe §4, S1): side-by-side counts.
import { svgEl, group, text, hatchRect } from './core.js';
import { sharePct } from '../../math/memory.js';

const PART_HUES = 5;
const MIN_PCT_WIDTH = 30; // narrower segments print their share in the legend instead
const LEGEND_ROW = 15;
const UNKNOWN_W = 24; // "not published": a fixed width, off the scale (decoder-anatomy §4)
const UNKNOWN_GAP = 6;
const UNKNOWN_LABEL_GAP = 4; // the printed "not published" label sits this far right of the off-scale segment
const SEGMENT_GAP = 2; // barSegments' default gap between neighbors
const TAIL_GAP = 40; // main bar top → zoomed tail bar top: room for the printed shares and the bracket

export function barSegments(values, w, gap = SEGMENT_GAP) {
  if (!Array.isArray(values) || values.length === 0) throw new RangeError('glyphs.barSegments: values must be a non-empty array');
  if (values.some((n) => typeof n !== 'number' || !Number.isFinite(n) || n < 0)) throw new RangeError(`glyphs.barSegments: values must be finite numbers ≥ 0, got ${values.join(' / ')}`);
  const total = values.reduce((a, b) => a + b, 0);
  if (!(total > 0)) throw new RangeError('glyphs.barSegments: values must sum to more than 0');
  const starts = values.map((_, i) => (values.slice(0, i).reduce((a, b) => a + b, 0) / total) * w);
  return values.map((n, i) => ({
    x: starts[i],
    width: Math.max(0, (n / total) * w - (i < values.length - 1 ? gap : 0)),
    share: n / total,
  }));
}

// The shareBar label: sharePct (math/memory.js, the one definition), one decimal from 1 %, two below.
export const formatShare = (share) => {
  const decimals = share >= 0.01 ? 1 : 2;
  return `${sharePct(share, 1, { decimals }).toFixed(decimals)}%`;
};

function checkParts(parts) {
  if (!Array.isArray(parts) || parts.length === 0) throw new RangeError('glyphs.shareBar: parts must be a non-empty array');
  parts.forEach((p, i) => {
    if (typeof p?.name !== 'string' || p.name.trim() === '') throw new RangeError(`glyphs.shareBar: part ${i} needs a name`);
    if (!p.unknown && !(Number.isInteger(p.hue) && p.hue >= 1 && p.hue <= PART_HUES)) throw new RangeError(`glyphs.shareBar: part "${p.name}" needs hue 1–${PART_HUES} or unknown: true`);
    if (!(Number.isFinite(p.value) && p.value >= 0)) throw new RangeError(`glyphs.shareBar: part "${p.name}" value must be a finite number ≥ 0, got ${p.value}`);
  });
  if (new Set(parts.map((p) => p.name)).size !== parts.length) throw new RangeError('glyphs.shareBar: part names must be unique');
}

const sumOf = (list) => list.reduce((acc, p) => acc + p.value, 0);

// Geometry only (pure). Known parts narrower than minSegment px fold into one "others" segment at the end of
// the main bar and are redrawn on a zoomed tail bar; unknown parts sit off the scale at a fixed width.
// Every share is of the whole (unknown parts included), so printed percentages read the same on both bars.
export function shareBarLayout(parts, { w = 240, minSegment = 18 } = {}) {
  checkParts(parts);
  if (!(minSegment >= 0)) throw new RangeError(`glyphs.shareBar: minSegment must be a number ≥ 0, got ${minSegment}`);
  const whole = sumOf(parts);
  const known = parts.filter((p) => !p.unknown && p.value > 0);
  if (known.length === 0) throw new RangeError('glyphs.shareBar: at least one known part must be more than 0');
  const knownTotal = sumOf(known);
  // Judged on the drawn width (after the gap), so a kept segment is never drawn under minSegment.
  const isNarrow = (p) => (p.value / knownTotal) * w - SEGMENT_GAP < minSegment;
  const folded = known.filter(isNarrow);
  const kept = known.filter((p) => !isNarrow(p));
  const mainParts = folded.length ? [...kept, { name: 'others', value: sumOf(folded), others: true }] : kept;
  const place = (list) => barSegments(list.map((p) => p.value), w)
    .map((seg, i) => ({ ...list[i], x: seg.x, width: seg.width, share: list[i].value / whole }));
  return {
    main: place(mainParts),
    tail: folded.length ? place(folded) : [],
    unknown: parts.filter((p) => p.unknown)
      .map((p, i) => ({ ...p, x: w + UNKNOWN_GAP + i * (UNKNOWN_W + 2), width: UNKNOWN_W, share: p.value / whole })),
  };
}

const partAttrs = (part, cls) => ({
  class: `${cls}${part.unknown ? ' g-part-none' : ''}${part.others ? ' g-part-others' : ''}`,
  'data-part': part.unknown || part.others ? null : part.hue,
});

function drawBar(g, segs, { y, h, format }) {
  segs.forEach((seg) => {
    const rect = svgEl('rect', { ...partAttrs(seg, 'g-seg'), x: seg.x, y, width: seg.width, height: h, rx: 2 }, g);
    svgEl('title', {}, rect).textContent = `${seg.name}: ${format(seg.share)}`;
    if (seg.width >= MIN_PCT_WIDTH) text(g, seg.x + seg.width / 2, y + h + 13, format(seg.share), 'g-pct', { 'text-anchor': 'middle' });
  });
}

// Off-scale parts get a printed label beside them (decoder-anatomy §4: "the printed label 'not published'").
function drawUnknownLabel(g, unknown, { h }) {
  if (!unknown.length) return;
  const last = unknown[unknown.length - 1];
  text(g, last.x + last.width + UNKNOWN_LABEL_GAP, h / 2, unknown.map((p) => p.name).join(' · '), 'g-label g-unknown-label', { 'dominant-baseline': 'central' });
}

// The bracket joins the "others" segment to the zoomed bar under it, below the main bar's printed shares.
function drawBracket(g, others, { h, w, tailY }) {
  const below = h + 17;
  const d = `M${others.x} ${h + 2}V${below}L0 ${tailY - 3}M${others.x + others.width} ${h + 2}V${below}L${w} ${tailY - 3}`;
  svgEl('path', { class: 'g-bracket', d }, g);
}

function legendRows(parts, layout, format) {
  const printed = new Set([...layout.main, ...layout.tail].filter((s) => s.width >= MIN_PCT_WIDTH).map((s) => s.name));
  const whole = sumOf(parts);
  const rows = parts.map((p) => ({ ...p, text: printed.has(p.name) ? p.name : `${p.name} · ${format(p.value / whole)}` }));
  const others = layout.main.find((s) => s.others);
  return others ? [...rows, { ...others, text: 'others: zoomed in the bar below' }] : rows;
}

function drawLegend(g, rows, { y }) {
  rows.forEach((row, i) => {
    const ly = y + i * LEGEND_ROW;
    svgEl('rect', { ...partAttrs(row, 'g-swatch'), x: 0, y: ly - 9, width: 10, height: 10, rx: 2 }, g);
    text(g, 16, ly, row.text, 'g-label');
  });
}

// Categorical stacked bar: segment length plus a printed share (under the segment, or in the legend when narrow).
export function shareBar(parent, { x, y, w = 240, h = 14, parts, format = formatShare, label = 'shares', minSegment = 18, tail = 'zoom' }) {
  if (tail !== 'zoom' && tail !== 'none') throw new RangeError(`glyphs.shareBar: tail must be "zoom" or "none", got ${tail}`);
  const layout = shareBarLayout(parts, { w, minSegment: tail === 'zoom' ? minSegment : 0 });
  const whole = sumOf(parts);
  const spoken = parts.map((p) => `${p.name} ${format(p.value / whole)}`).join(', ');
  const g = group(parent, 'g-share', x, y, { role: 'img', 'aria-label': `${label}: ${spoken}` });
  drawBar(g, [...layout.main, ...layout.unknown], { y: 0, h, format });
  drawUnknownLabel(g, layout.unknown, { h });
  const tailY = h + TAIL_GAP;
  if (layout.tail.length) {
    drawBracket(g, layout.main.find((s) => s.others), { h, w, tailY });
    drawBar(g, layout.tail, { y: tailY, h, format });
  }
  drawLegend(g, legendRows(parts, layout, format), { y: (layout.tail.length ? tailY : 0) + h + 32 });
  return g;
}

// useful · reserved-but-empty (hatched: it holds nothing) · free. API and output unchanged since Plan 1.
export function memBar(parent, { x, y, w = 240, h = 14, useful, reserved, free }) {
  if ([useful, reserved, free].some((n) => !(n >= 0))) throw new RangeError(`glyphs.memBar: parts must be numbers ≥ 0, got ${useful} / ${reserved} / ${free}`);
  const total = useful + reserved + free;
  if (!(total > 0)) throw new RangeError('glyphs.memBar: useful + reserved + free must be > 0');
  const pct = (n) => `${Math.round((n / total) * 100)}%`;
  const g = group(parent, 'g-membar', x, y, { role: 'img', 'aria-label': `memory: useful ${pct(useful)}, reserved but empty ${pct(reserved)}, free ${pct(free)}` });
  const values = [useful, reserved, free];
  const segs = barSegments(values, w);
  ['useful', 'reserved', 'free'].forEach((name, i) => {
    if (values[i] === 0) return;
    const { x: sx, width } = segs[i];
    if (name === 'reserved') {
      svgEl('rect', { class: 'g-reserved-bg', x: sx, width, height: h, rx: 2 }, g);
      hatchRect(g, { x: sx, width, height: h, rx: 2 });
    } else {
      svgEl('rect', { class: `g-${name}`, x: sx, width, height: h, rx: 2 }, g);
    }
    if (width > 26) text(g, sx + width / 2, h + 13, pct(values[i]), 'g-pct', { 'text-anchor': 'middle' });
  });
  text(g, 0, h + 28, `useful ${useful} · reserved ${reserved} · free ${free}`, 'g-label');
  return g;
}

// ---- bars (moe §4, accepted with conditions a–f; reused by sampling) ----
const BAR_PITCH = 43; // NUMBER_CELL + 3: eight bars fill 344 px (condition d)
const BAR_GAP = 3;
const VALUE_LABEL_RISE = 5; // a value label's baseline sits this far above its bar
const REFERENCE_CLEARANCE = 8; // a label baseline closer than this to the reference line is moved clear of it
const LABEL_LIFT = 4; // a moved label's baseline sits this far above the line (or above its old place if the line is lower)

function checkBars({ values, max, h, labels, reference, w }) {
  if (!Array.isArray(values) || values.length === 0) throw new RangeError('glyphs.bars: values must be a non-empty array');
  if (values.some((v) => typeof v !== 'number' || !Number.isFinite(v) || v < 0)) throw new RangeError(`glyphs.bars: values must be finite numbers ≥ 0, got ${values.join(' / ')}`);
  if (!(Number.isFinite(max) && max > 0)) throw new RangeError(`glyphs.bars: max must be a finite number > 0 (it holds the scale between frames), got ${max}`);
  const over = values.find((v) => v > max);
  if (over !== undefined) throw new RangeError(`glyphs.bars: value ${over} exceeds max ${max}`);
  if (!(Number.isFinite(h) && h > 0)) throw new RangeError(`glyphs.bars: h must be a finite number > 0, got ${h}`);
  if (w != null && !(Number.isFinite(w) && w > 0)) throw new RangeError(`glyphs.bars: w must be a finite number > 0, got ${w}`);
  if (labels != null && (!Array.isArray(labels) || labels.length !== values.length)) throw new RangeError('glyphs.bars: labels must match values one to one');
  if (reference != null && !(Number.isFinite(reference.value) && reference.value >= 0 && reference.value <= max && typeof reference.label === 'string')) {
    throw new RangeError('glyphs.bars: reference must be { value ≤ max, label }');
  }
}

// Geometry only (pure): bar boxes inside [0, w] × [0, h] (baseline at h), each with its value label's baseline
// `labelY` (lifted above the dashed reference line when it would sit on it), and the reference line's y.
export function barsLayout({ values, max, h, w = null, labels = null, reference = null }) {
  checkBars({ values, max, h, labels, reference, w });
  const width = w ?? values.length * BAR_PITCH;
  const pitch = width / values.length;
  const referenceY = reference ? h - (reference.value / max) * h : null;
  const bars = values.map((v, i) => {
    const height = (v / max) * h;
    const y = h - height;
    const labelY = y - VALUE_LABEL_RISE;
    const onLine = referenceY !== null && Math.abs(labelY - referenceY) < REFERENCE_CLEARANCE;
    return { x: i * pitch, width: pitch - BAR_GAP, height, y, value: v, labelY: onLine ? Math.min(labelY, referenceY) - LABEL_LIFT : labelY };
  });
  return { bars, w: width, referenceY };
}

// Vertical bars, one per label, each printing its value above it (length plus number); neutral fill, never the
// accent or the value scale; an optional dashed reference line with its own printed label (ruling S1-R8).
export function bars(parent, { x, y, w = null, h = 120, values, labels = null, max, reference = null, format = String, label = 'counts' }) {
  const layout = barsLayout({ values, max, h, w, labels, reference });
  const names = labels ?? values.map((_, i) => String(i + 1));
  const spoken = names.map((n, i) => `${n} ${format(values[i])}`).join(', ');
  const g = group(parent, 'g-bars', x, y, { role: 'img', 'aria-label': `${label}: ${spoken}${reference ? `; ${reference.label}` : ''}` });
  svgEl('line', { class: 'g-bars-axis', x1: 0, y1: h, x2: layout.w - BAR_GAP, y2: h }, g);
  layout.bars.forEach((b, i) => {
    const rect = svgEl('rect', { class: 'g-bar', x: b.x, y: b.y, width: b.width, height: b.height, rx: 2 }, g);
    svgEl('title', {}, rect).textContent = `${names[i]}: ${format(b.value)}`;
    text(g, b.x + b.width / 2, b.labelY, format(b.value), 'g-bar-value', { 'text-anchor': 'middle' });
    if (labels) text(g, b.x + b.width / 2, h + 14, labels[i], 'g-label', { 'text-anchor': 'middle' });
  });
  if (reference) {
    svgEl('line', { class: 'g-bars-ref', x1: 0, y1: layout.referenceY, x2: layout.w - BAR_GAP, y2: layout.referenceY }, g);
    text(g, layout.w + 2, layout.referenceY, reference.label, 'g-label g-bars-ref-label', { 'dominant-baseline': 'central' });
  }
  return g;
}
