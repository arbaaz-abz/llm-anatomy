// Stacked bars: shareBar (categorical parts, decoder-anatomy §4) and memBar (paged-attention).
// One geometry (barSegments) under both, so the two never drift apart. Plus `bars` (moe §4, S1): side-by-side counts.
import { svgEl, group, text, hatchRect } from './core.js';
import { sharePct } from '../../math/memory.js';
import { formatDuration } from '../../math/core.js';

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

// Unknown: `value: null` (a share that is not published, P3-R6) or the older `unknown: true` (off the scale, with a value).
const isUnknown = (p) => p.unknown === true || p.value === null;
const NOT_PUBLISHED = 'not published';
const ALL_UNKNOWN = 'no published shares';

function checkParts(parts, tailBasis) {
  if (!Array.isArray(parts) || parts.length === 0) throw new RangeError('glyphs.shareBar: parts must be a non-empty array');
  parts.forEach((p, i) => {
    if (typeof p?.name !== 'string' || p.name.trim() === '') throw new RangeError(`glyphs.shareBar: part ${i} needs a name`);
    if (!isUnknown(p) && !(Number.isInteger(p.hue) && p.hue >= 1 && p.hue <= PART_HUES)) throw new RangeError(`glyphs.shareBar: part "${p.name}" needs hue 1–${PART_HUES} or unknown: true`);
    if (p.value !== null && !(Number.isFinite(p.value) && p.value >= 0)) throw new RangeError(`glyphs.shareBar: part "${p.name}" value must be a finite number ≥ 0 (or null: not published), got ${p.value}`);
    if (p.hatched && isUnknown(p)) throw new RangeError(`glyphs.shareBar: part "${p.name}" is unknown and cannot be hatched (unknown is neutral, hatched means excluded: README lessons 19, 24)`);
  });
  if (new Set(parts.map((p) => p.name)).size !== parts.length) throw new RangeError('glyphs.shareBar: part names must be unique');
  if (tailBasis !== 'whole' && tailBasis !== 'tail') throw new RangeError(`glyphs.shareBar: tailBasis must be "whole" or "tail", got ${tailBasis}`);
}

const sumOf = (list) => list.reduce((acc, p) => acc + (p.value ?? 0), 0);

// The printed text beside the off-scale parts: the older unknown parts print their names; value-null parts add
// "not published"; a bar with nothing known prints "no published shares" (P3-R6).
function unknownText(unknown, allUnknown) {
  if (allUnknown) return ALL_UNKNOWN;
  const named = unknown.filter((p) => p.value !== null).map((p) => p.name);
  const missing = unknown.filter((p) => p.value === null).map((p) => p.name);
  return [...named, ...(missing.length ? [`${missing.join(' · ')}: ${NOT_PUBLISHED}`] : [])].join(' · ');
}

function placeUnknown(parts, { x0, whole }) {
  return parts.filter(isUnknown).map((p, i) => ({ ...p, x: x0 + i * (UNKNOWN_W + 2), width: UNKNOWN_W, share: p.value === null ? null : p.value / whole }));
}

// Geometry only (pure). Known parts narrower than minSegment px fold into one "others" segment at the end of
// the main bar and are redrawn on a zoomed tail bar; unknown parts sit off the scale at a fixed width.
// Every share is of the whole (unknown parts with a value included), so printed percentages read the same on
// both bars, unless tailBasis is 'tail': then the zoomed bar's shares are of the tail and add to 100 % (P3-R7).
export function shareBarLayout(parts, { w = 240, minSegment = 18, tailBasis = 'whole' } = {}) {
  checkParts(parts, tailBasis);
  if (!(minSegment >= 0)) throw new RangeError(`glyphs.shareBar: minSegment must be a number ≥ 0, got ${minSegment}`);
  const whole = sumOf(parts);
  const known = parts.filter((p) => !isUnknown(p) && p.value > 0);
  if (known.length === 0 && parts.every(isUnknown)) {
    const unknown = placeUnknown(parts, { x0: 0, whole });
    return { main: [], tail: [], unknown, unknownLabel: unknownText(unknown, true), allUnknown: true };
  }
  if (known.length === 0) throw new RangeError('glyphs.shareBar: at least one known part must be more than 0');
  const knownTotal = sumOf(known);
  // Judged on the drawn width (after the gap), so a kept segment is never drawn under minSegment.
  const isNarrow = (p) => (p.value / knownTotal) * w - SEGMENT_GAP < minSegment;
  const folded = known.filter(isNarrow);
  const kept = known.filter((p) => !isNarrow(p));
  const mainParts = folded.length ? [...kept, { name: 'others', value: sumOf(folded), others: true }] : kept;
  const place = (list, basis) => barSegments(list.map((p) => p.value), w)
    .map((seg, i) => ({ ...list[i], x: seg.x, width: seg.width, share: list[i].value / basis }));
  const unknown = placeUnknown(parts, { x0: w + UNKNOWN_GAP, whole });
  return {
    main: place(mainParts, whole),
    tail: folded.length ? place(folded, tailBasis === 'tail' ? sumOf(folded) : whole) : [],
    unknown,
    unknownLabel: unknownText(unknown, false),
    allUnknown: false,
  };
}

const partAttrs = (part, cls) => ({
  class: `${cls}${isUnknown(part) ? ' g-part-none' : ''}${part.others ? ' g-part-others' : ''}`,
  'data-part': isUnknown(part) || part.others ? null : part.hue,
});

// `basisNote` (tail basis only) follows each share in the segment's tooltip, e.g. "64.5% of last 5%".
function drawBar(g, segs, { y, h, format, basisNote = '', pctOthers = true }) {
  segs.forEach((seg) => {
    const rect = svgEl('rect', { ...partAttrs(seg, 'g-seg'), x: seg.x, y, width: seg.width, height: h, rx: 2 }, g);
    svgEl('title', {}, rect).textContent = `${seg.name}: ${seg.share === null ? NOT_PUBLISHED : format(seg.share) + basisNote}`;
    if (seg.hatched) hatchRect(g, { x: seg.x, y, width: seg.width, height: h, rx: 2 });
    if (seg.share !== null && seg.width >= MIN_PCT_WIDTH && (pctOthers || !seg.others)) text(g, seg.x + seg.width / 2, y + h + 13, format(seg.share), 'g-pct', { 'text-anchor': 'middle' });
  });
}

// Off-scale parts get a printed label beside them (decoder-anatomy §4: "the printed label 'not published'").
function drawUnknownLabel(g, layout, { h }) {
  if (!layout.unknown.length) return;
  const last = layout.unknown[layout.unknown.length - 1];
  text(g, last.x + last.width + UNKNOWN_LABEL_GAP, h / 2, layout.unknownLabel, 'g-label g-unknown-label', { 'dominant-baseline': 'central' });
}

// The bracket joins the "others" segment to the zoomed bar under it, below the main bar's printed shares.
function drawBracket(g, others, { h, w, tailY, tailLabel }) {
  const below = h + 17;
  const d = `M${others.x} ${h + 2}V${below}L0 ${tailY - 3}M${others.x + others.width} ${h + 2}V${below}L${w} ${tailY - 3}`;
  svgEl('path', { class: 'g-bracket', d }, g);
  if (tailLabel) text(g, others.x - 4, h + 13, tailLabel, 'g-label g-tail-label', { 'text-anchor': 'end' });
}

function legendText(p, { printed, whole, format, tailFormat, tailNames, tailShares, basisNote }) {
  if (p.value === null) return `${p.name} · ${NOT_PUBLISHED}`;
  if (printed.has(p.name)) return p.name;
  if (!tailNames.has(p.name)) return `${p.name} · ${format(p.value / whole)}`;
  return basisNote ? `${p.name} · ${tailFormat(tailShares.get(p.name))}${basisNote}` : `${p.name} · ${tailFormat(p.value / whole)}`;
}

function legendRows(parts, layout, { format, tailFormat, tailBasis, tailLabel }) {
  const printed = new Set([...layout.main, ...layout.tail].filter((s) => s.width >= MIN_PCT_WIDTH).map((s) => s.name));
  const basisNote = tailBasis === 'tail' ? ` of ${tailLabel ?? 'the others'}` : '';
  const ctx = { printed, whole: sumOf(parts), format, tailFormat, tailNames: new Set(layout.tail.map((s) => s.name)), tailShares: new Map(layout.tail.map((s) => [s.name, s.share])), basisNote };
  const rows = parts.map((p) => ({ ...p, text: legendText(p, ctx) }));
  const others = layout.main.find((s) => s.others);
  const othersText = basisNote ? `others: zoomed below, shares${basisNote}` : 'others: zoomed in the bar below';
  return others ? [...rows, { ...others, text: othersText }] : rows;
}

function drawLegend(g, rows, { y }) {
  rows.forEach((row, i) => {
    const ly = y + i * LEGEND_ROW;
    svgEl('rect', { ...partAttrs(row, 'g-swatch'), x: 0, y: ly - 9, width: 10, height: 10, rx: 2 }, g);
    if (row.hatched) hatchRect(g, { x: 0, y: ly - 9, width: 10, height: 10, rx: 2 });
    text(g, 16, ly, row.text, 'g-label');
  });
}

const spokenShare = (p, whole, format) => (p.value === null ? NOT_PUBLISHED : format(p.value / whole));

// Categorical stacked bar: segment length plus a printed share (under the segment, or in the legend when narrow).
// Parts: { name, value, hue } · { name, value: null } (not published, neutral, off the scale) · hatched: true
// (excluded / doesn't count, README lesson 24). tailBasis 'tail' + tailLabel: the zoomed bar's shares are of the
// tail, and the bracket and legend say so (P3-R7). tailFormat (default: format) prints the zoomed bar's shares, in the
// bar, its tooltips and the legend (S3.5, midtraining frame 6: main "94.57%", tail "64.5%").
export function shareBar(parent, { x, y, w = 240, h = 14, parts, format = formatShare, tailFormat = format, label = 'shares', minSegment = 18, tail = 'zoom', tailBasis = 'whole', tailLabel = null }) {
  if (tail !== 'zoom' && tail !== 'none') throw new RangeError(`glyphs.shareBar: tail must be "zoom" or "none", got ${tail}`);
  const layout = shareBarLayout(parts, { w, minSegment: tail === 'zoom' ? minSegment : 0, tailBasis });
  const whole = sumOf(parts);
  const spoken = parts.map((p) => `${p.name} ${spokenShare(p, whole, format)}`).join(', ');
  const g = group(parent, 'g-share', x, y, { role: 'img', 'aria-label': `${label}: ${spoken}` });
  drawBar(g, [...layout.main, ...layout.unknown], { y: 0, h, format, pctOthers: !tailLabel });
  drawUnknownLabel(g, layout, { h });
  const tailY = h + TAIL_GAP;
  if (layout.tail.length) {
    const basisNote = tailBasis === 'tail' ? ` of ${tailLabel ?? 'the others'}` : '';
    drawBracket(g, layout.main.find((s) => s.others), { h, w, tailY, tailLabel });
    drawBar(g, layout.tail, { y: tailY, h, format: tailFormat, basisNote });
  }
  drawLegend(g, legendRows(parts, layout, { format, tailFormat, tailBasis, tailLabel }), { y: (layout.tail.length ? tailY : 0) + h + 32 });
  return g;
}

const MEM_NAMES = ['useful', 'reserved', 'free'];
const MEM_BRACKET_DROP = 2; // a widened segment's bracket mark sits this far under the bar
const MEM_NARROW_PCT_Y = 17; // …and its percentage under the bracket (other percentages sit at h + 13)

// Geometry only (pure). The default (minSegment 0) is the Plan 1 drawing. With minSegment > 0 (S6, P4-R12) a nonzero
// segment drawn narrower than minSegment px widens to it and the others rescale so the bar keeps its total width;
// `value` stays the true amount, and `widened` segments get a bracket. Zero parts are not drawn.
export function memBarLayout({ useful, reserved, free, w = 240, minSegment = 0 }) {
  if (!(Number.isFinite(minSegment) && minSegment >= 0)) throw new RangeError(`glyphs.memBar: minSegment must be a number ≥ 0, got ${minSegment}`);
  const values = [useful, reserved, free];
  const base = barSegments(values, w);
  const shown = MEM_NAMES.map((name, i) => ({ name, value: values[i], x: base[i].x, width: base[i].width, widened: false })).filter((s) => s.value > 0);
  if (shown.every((s) => s.width >= minSegment)) return { w, segments: shown, brackets: [] };
  const gaps = SEGMENT_GAP * (shown.length - 1);
  const fixed = new Set();
  for (let again = true; again;) {
    const free_ = shown.filter((s) => !fixed.has(s.name));
    const room = w - gaps - fixed.size * minSegment;
    const sum = free_.reduce((a, s) => a + s.value, 0);
    const narrow = free_.filter((s) => (s.value / sum) * room < minSegment);
    narrow.forEach((s) => fixed.add(s.name));
    again = narrow.length > 0 && narrow.length < free_.length;
  }
  const room = w - gaps - fixed.size * minSegment;
  const sum = shown.filter((s) => !fixed.has(s.name)).reduce((a, s) => a + s.value, 0);
  let cursor = 0;
  const segments = shown.map((s) => {
    const widened = fixed.has(s.name);
    const width = widened ? minSegment : (s.value / sum) * room;
    const placed = { ...s, x: cursor, width, widened };
    cursor += width + SEGMENT_GAP;
    return placed;
  });
  return { w, segments, brackets: segments.filter((s) => s.widened).map((s) => ({ name: s.name, x0: s.x, x1: s.x + s.width })) };
}

// useful · reserved-but-empty (hatched: it holds nothing) · free. Default output unchanged since Plan 1;
// `minSegment` (S6): see memBarLayout.
export function memBar(parent, { x, y, w = 240, h = 14, useful, reserved, free, minSegment = 0 }) {
  if ([useful, reserved, free].some((n) => !(n >= 0))) throw new RangeError(`glyphs.memBar: parts must be numbers ≥ 0, got ${useful} / ${reserved} / ${free}`);
  const total = useful + reserved + free;
  if (!(total > 0)) throw new RangeError('glyphs.memBar: useful + reserved + free must be > 0');
  const L = memBarLayout({ useful, reserved, free, w, minSegment });
  const pct = (n) => `${Math.round((n / total) * 100)}%`;
  const g = group(parent, 'g-membar', x, y, { role: 'img', 'aria-label': `memory: useful ${pct(useful)}, reserved but empty ${pct(reserved)}, free ${pct(free)}` });
  L.segments.forEach(({ name, value, x: sx, width, widened }) => {
    if (name === 'reserved') {
      svgEl('rect', { class: 'g-reserved-bg', x: sx, width, height: h, rx: 2 }, g);
      hatchRect(g, { x: sx, width, height: h, rx: 2 });
    } else {
      svgEl('rect', { class: `g-${name}`, x: sx, width, height: h, rx: 2 }, g);
    }
    if (widened) {
      svgEl('path', { class: 'g-bracket', d: `M${sx} ${h + MEM_BRACKET_DROP}v4H${sx + width}v-4` }, g);
      text(g, sx + width / 2, h + MEM_NARROW_PCT_Y, pct(value), 'g-pct', { 'text-anchor': 'middle' });
    } else if (width > 26) text(g, sx + width / 2, h + 13, pct(value), 'g-pct', { 'text-anchor': 'middle' });
  });
  text(g, 0, h + 28, `useful ${useful} · reserved ${reserved} · free ${free}`, 'g-label');
  return g;
}

// ---- stepBar (prefill-decode §4, P4-R8): a step's time as reading against arithmetic, on one held seconds scale ----
const STEP_MIN_SEGMENT = 18; // a part the learner must read is at least this wide (README lesson 19)
const STEP_GAP = 3; // between the parts of the reading row
const STEP_CHAR_W = 6.6;
const STEP_INSIDE_PAD = 8;
const STEP_TOTAL_GAP = 6; // the step total and "overlapped" print this far right of their row
const STEP_BELOW = 17; // first below-bar label baseline, under the bar
const STEP_LINE = 12; // further below-bar lines
const STEP_ROW_GAP = 8;
const STEP_EPS = 1e-12;
const failStep = (msg) => { throw new RangeError(`glyphs.stepBar: ${msg}`); };

function checkStep({ w, h, scaleS, reading, mathS }, format) {
  if (!(Number.isFinite(w) && w > 0 && Number.isFinite(h) && h > 0)) failStep(`w and h must be finite numbers > 0, got ${w} × ${h}`);
  if (!(Number.isFinite(scaleS) && scaleS > 0)) failStep(`scaleS must be a finite number > 0 (the seconds the full width stands for), got ${scaleS}`);
  if (!Array.isArray(reading) || reading.length === 0) failStep('reading must be a non-empty array of { label, s }');
  reading.forEach((p, i) => {
    if (typeof p?.label !== 'string' || p.label.trim() === '') failStep(`reading part ${i} needs a label`);
    if (!(Number.isFinite(p.s) && p.s >= 0)) failStep(`reading part "${p.label}" s must be a finite number ≥ 0, got ${p.s}`);
  });
  if (!(Number.isFinite(mathS) && mathS >= 0)) failStep(`mathS must be a finite number ≥ 0, got ${mathS}`);
  const readingS = reading.reduce((a, p) => a + p.s, 0);
  if (readingS > scaleS * (1 + STEP_EPS)) failStep(`reading ${readingS} s runs past scaleS ${scaleS} s; hold one scale across frames`);
  if (mathS > scaleS * (1 + STEP_EPS)) failStep(`arithmetic ${mathS} s runs past scaleS ${scaleS} s; hold one scale across frames`);
  if (typeof format !== 'function') failStep('format must be a function');
}

// One row's segments on the scale. Parts keep a 3 px gap; a nonzero part narrower than 18 px widens to 18 (the rest
// of the row shifts right) and gets a bracket; its true value stays in `s`, `text` and `title`.
function placeStepRow(parts, { w, h, scaleS, format, y }) {
  const nonzero = parts.filter((p) => p.s > 0);
  let cursor = 0;
  const segments = parts.map((p) => {
    const last = nonzero.length > 0 && p === nonzero.at(-1);
    const pw = (p.s / scaleS) * w;
    const raw = pw - (last ? 0 : STEP_GAP);
    const widened = p.s > 0 && raw < STEP_MIN_SEGMENT;
    const width = p.s === 0 ? 0 : widened ? STEP_MIN_SEGMENT : raw;
    const seg = { label: p.label, s: p.s, x: cursor, width, widened, y, h, title: `${p.label}: ${format(p.s)}`, text: `${p.label} ${format(p.s)}` };
    cursor += p.s === 0 ? 0 : widened ? STEP_MIN_SEGMENT + (last ? 0 : STEP_GAP) : pw;
    return seg;
  });
  return { segments, end: cursor };
}

// Text goes inside a segment that fits it; otherwise below the bar, centered under the segment, on the first line
// where it clears the labels already there.
function placeStepText(segments, w) {
  const lineEnds = [];
  segments.filter((s) => s.width > 0).forEach((seg) => {
    const len = seg.text.length * STEP_CHAR_W;
    if (len + STEP_INSIDE_PAD <= seg.width) {
      Object.assign(seg, { textInside: true, textX: seg.x + seg.width / 2, line: null });
      return;
    }
    const half = len / 2;
    const textX = Math.min(Math.max(seg.x + seg.width / 2, half), Math.max(w - half, half));
    let line = 0;
    while ((lineEnds[line] ?? -Infinity) + STEP_INSIDE_PAD > textX - half) line += 1;
    lineEnds[line] = textX + half;
    Object.assign(seg, { textInside: false, textX, line });
  });
  return lineEnds.length;
}

// Geometry only (pure). `scaleS` is the seconds the full width `w` stands for, held across frames; the longer row is
// the step, the shorter row is faint and labeled "overlapped" (the roofline's max); a tie fades neither.
export function stepBarLayout({ w = 300, h = 14, scaleS, reading, mathS, format = formatDuration }) {
  checkStep({ w, h, scaleS, reading, mathS }, format);
  const readingS = reading.reduce((a, p) => a + p.s, 0);
  const stepS = Math.max(readingS, mathS);
  const tied = Math.abs(readingS - mathS) <= STEP_EPS * stepS;
  const longer = tied ? 'tie' : readingS > mathS ? 'reading' : 'arithmetic';
  const first = placeStepRow(reading, { w, h, scaleS, format, y: 0 });
  const firstLines = placeStepText(first.segments, w);
  const secondY = h + (STEP_BELOW + STEP_LINE * Math.max(firstLines - 1, 0)) + STEP_ROW_GAP;
  const second = placeStepRow([{ label: 'arithmetic', s: mathS }], { w, h, scaleS, format, y: secondY });
  const secondLines = placeStepText(second.segments, w);
  const rows = [
    { kind: 'reading', y: 0, faint: longer === 'arithmetic', end: first.end, segments: first.segments },
    { kind: 'arithmetic', y: secondY, faint: longer === 'reading', end: second.end, segments: second.segments },
  ];
  const stepRow = longer === 'arithmetic' ? rows[1] : rows[0];
  const shortRow = longer === 'reading' ? rows[1] : longer === 'arithmetic' ? rows[0] : null;
  const total = { text: format(stepS), x: stepRow.end + STEP_TOTAL_GAP, y: stepRow.y + h / 2 };
  const overlapped = shortRow ? { text: 'overlapped', x: shortRow.end + STEP_TOTAL_GAP, y: shortRow.y + h / 2 } : null;
  const brackets = rows.flatMap((r) => r.segments.filter((s) => s.widened).map((s) => ({ row: r.kind, x0: s.x, x1: s.x + s.width, y: r.y + h + 2 })));
  const right = Math.max(w, total.x + total.text.length * STEP_CHAR_W, overlapped ? overlapped.x + overlapped.text.length * STEP_CHAR_W : 0);
  const height = secondY + h + (secondLines > 0 ? STEP_BELOW + STEP_LINE * (secondLines - 1) : 0);
  return { w, h, scaleS, stepS, longer, rows, total, overlapped, brackets, extent: right, height };
}

function drawStepRow(g, row, h) {
  const rowG = svgEl('g', { class: `g-step-row g-step-row--${row.kind}${row.faint ? ' g-faint' : ''}` }, g);
  const fill = row.kind === 'reading' ? 'g-step--memory' : 'g-step--compute';
  row.segments.filter((s) => s.width > 0).forEach((s) => {
    const rect = svgEl('rect', { class: `g-step-seg ${fill}`, x: num2(s.x), y: s.y, width: num2(s.width), height: h, rx: 2 }, rowG);
    svgEl('title', {}, rect).textContent = s.title;
  });
}

const num2 = (v) => Number(v.toFixed(2));

function drawStepTexts(g, row, h) {
  row.segments.filter((s) => s.width > 0).forEach((s) => {
    if (s.textInside) text(g, num2(s.textX), s.y + h / 2, s.text, 'g-step-text', { 'text-anchor': 'middle', 'dominant-baseline': 'central' });
    else text(g, num2(s.textX), s.y + h + STEP_BELOW + STEP_LINE * s.line, s.text, 'g-label g-step-text', { 'text-anchor': 'middle' });
  });
}

// Two rows on one seconds scale: the reading parts (--sem-memory, each with its name and time) over the arithmetic
// (--sem-compute). The step is the longer row and prints its total at the row's right end; the shorter row is faint
// and says "overlapped". No animation: pages interpolate the seconds they pass in.
export function stepBar(parent, { x, y, w = 300, h = 14, scaleS, reading, mathS, format = formatDuration, label = 'step time' }) {
  const L = stepBarLayout({ w, h, scaleS, reading, mathS, format });
  const parts = reading.map((p) => `${p.label} ${format(p.s)}`).join(', ');
  const g = group(parent, 'g-stepbar', x, y, { role: 'img', 'aria-label': `${label}: reading ${parts}; arithmetic ${format(mathS)}; step ${L.total.text}` });
  L.rows.forEach((row) => drawStepRow(g, row, h));
  L.brackets.forEach((b) => svgEl('path', { class: 'g-bracket', d: `M${num2(b.x0)} ${b.y}v4H${num2(b.x1)}v-4` }, g));
  L.rows.forEach((row) => drawStepTexts(g, row, h));
  text(g, num2(L.total.x), L.total.y, L.total.text, 'g-step-total', { 'dominant-baseline': 'central' });
  if (L.overlapped) text(g, num2(L.overlapped.x), L.overlapped.y, L.overlapped.text, 'g-label g-step-overlapped', { 'dominant-baseline': 'central' });
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
