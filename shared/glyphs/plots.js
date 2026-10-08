// Plots (shared prep S3): curvePlot, a small line chart with optional log axes (prefill-decode §4, P3-R10,
// with scaling-laws' bands), and roofline, a thin wrapper that draws a chip's roof from two numbers
// (gpu-primer §4). Pure layouts (…Layout) plus DOM builders; neither animates nor outlines anything except
// a `followed` marker, which gets the one selection mark (G.selectionMark).
import { svgEl, group, text } from './core.js';
import { selectionMark } from './architecture.js';

const CHAR_W = 6.6; // JetBrains Mono at the 11 px label size
const TICK = 4; // tick mark length
const PAD = { top: 22, right: 8, bottom: 34 }; // y-axis label row above the plot; tick labels + x-axis label below
const MARKER_R = 4;
const MARKER_LABEL = { dx: 7, dy: 15 }; // a marker's label sits below-right of its dot (below a rising line)
const BAND_INSET = 1; // touching bands keep a 2 px gap
const STYLES = new Set(['solid', 'muted']);
const TONES = ['ink', 'compute', 'memory', 'comm', 'ok', 'bad'];
const EPS = 1e-9;

const isNum = Number.isFinite;
const textWidth = (str) => String(str).length * CHAR_W;
const fail = (msg) => { throw new RangeError(`glyphs.curvePlot: ${msg}`); };

// Tick and readout numbers: thousands separators from 1,000, three significant figures below, a real minus.
export function formatTick(v) {
  const a = Math.abs(v);
  const body = a >= 1000 ? Math.round(a).toLocaleString('en-US') : String(Number(a.toPrecision(3)));
  return v < 0 ? `−${body}` : body;
}

const tickOf = (t) => (typeof t === 'number' ? { value: t, label: formatTick(t) } : { value: t.value, label: String(t.label) });

function axisDomain(axis, name, values) {
  const ticks = (axis.ticks ?? []).map(tickOf);
  const fromTicks = ticks.length >= 2 ? [Math.min(...ticks.map((t) => t.value)), Math.max(...ticks.map((t) => t.value))] : null;
  const finite = values.filter(isNum);
  const fromData = finite.length >= 2 ? [Math.min(...finite), Math.max(...finite)] : null;
  const domain = axis.domain ?? fromTicks ?? fromData;
  if (!domain || !(isNum(domain[0]) && isNum(domain[1]) && domain[0] < domain[1])) fail(`${name} axis needs a domain: pass domain [min, max] or at least two ticks`);
  if (axis.log && !(domain[0] > 0)) fail(`a log axis needs a domain > 0, got [${domain.join(', ')}]`);
  return { domain, ticks, log: Boolean(axis.log), label: axis.label ?? '' };
}

// Maps a value onto [from, to] px (to < from for the y axis).
function scaler({ domain, log }, from, to) {
  const f = log ? Math.log10 : (v) => v;
  const [d0, d1] = domain.map(f);
  return (v) => from + ((f(v) - d0) / (d1 - d0)) * (to - from);
}

const inside = ({ domain }, v) => v >= domain[0] - EPS * Math.abs(domain[0] || 1) && v <= domain[1] + EPS * Math.abs(domain[1] || 1);

function checkPoint(ax, ay, x, y) {
  if (!(isNum(x) && isNum(y)) || !inside(ax, x) || !inside(ay, y)) fail(`point (${x}, ${y}) is outside the domain x [${ax.domain.join(', ')}], y [${ay.domain.join(', ')}]`);
}

function checkInput({ w, h, series, bands, markers }) {
  if (!(isNum(w) && w > 0 && isNum(h) && h > 0)) fail(`w and h must be finite numbers > 0, got ${w} × ${h}`);
  if (!Array.isArray(series) || !Array.isArray(markers) || !Array.isArray(bands)) fail('series, markers and bands must be arrays');
  series.forEach((s) => {
    if (!STYLES.has(s.style ?? 'solid')) fail(`series style must be solid or muted (no dashed series, P3-R11), got ${s.style}`);
    if (!TONES.includes(s.tone ?? 'ink')) fail(`series tone must be one of ${TONES.join(', ')}, got ${s.tone}`);
    if (!Array.isArray(s.points)) fail('each series needs points: [[x, y], …]');
    if (!['end', 'mid'].includes(s.labelAt ?? 'end')) fail(`series labelAt must be end or mid, got ${s.labelAt}`);
  });
  bands.forEach((b) => {
    if (!(isNum(b.from) && isNum(b.to) && b.from < b.to)) fail(`band from must be < to, got ${b.from} → ${b.to}`);
    if (!['top', 'bottom'].includes(b.labelAt ?? 'top')) fail(`band labelAt must be top or bottom, got ${b.labelAt}`);
  });
}

function margins(ay) {
  const labels = ay.ticks.map((t) => t.label);
  const left = Math.ceil(Math.max(0, ...labels.map(textWidth))) + TICK + 6;
  return { left, ...PAD };
}

function placeMarker(m, toX, toY, plot) {
  const x = toX(m.x);
  const y = toY(m.y);
  const label = m.label == null ? '' : String(m.label);
  const flip = x + MARKER_LABEL.dx + textWidth(label) > plot.right;
  const below = y + MARKER_LABEL.dy <= plot.bottom - 2;
  return {
    x, y, label, followed: Boolean(m.followed),
    labelX: flip ? x - MARKER_LABEL.dx : x + MARKER_LABEL.dx,
    labelY: below ? y + MARKER_LABEL.dy : y - MARKER_LABEL.dy + 4,
    labelAnchor: flip ? 'end' : 'start',
  };
}

// Geometry only (pure). w × h is the whole figure: the plot box sits inside it, leaving room for the
// tick labels (left, below), the x-axis label (below) and the y-axis label (top-left, horizontal).
export function curvePlotLayout({ w, h, xAxis = {}, yAxis = {}, series = [], markers = [], bands = [], refY = null }) {
  checkInput({ w, h, series, bands, markers });
  const xs = [...series.flatMap((s) => s.points.map((p) => p[0])), ...markers.map((m) => m.x), ...bands.flatMap((b) => [b.from, b.to])];
  const ys = [...series.flatMap((s) => s.points.map((p) => p[1])), ...markers.map((m) => m.y), ...(refY ? [refY.value] : [])];
  const ax = axisDomain(xAxis, 'x', xs);
  const ay = axisDomain(yAxis, 'y', ys);
  const m = margins(ay);
  const plot = { left: m.left, top: m.top, right: w - m.right, bottom: h - m.bottom };
  if (!(plot.right - plot.left > 20 && plot.bottom - plot.top > 20)) fail(`w × h (${w} × ${h}) leaves no room for the plot`);
  const toX = scaler(ax, plot.left, plot.right);
  const toY = scaler(ay, plot.bottom, plot.top);
  if (refY && !(isNum(refY.value) && inside(ay, refY.value) && typeof refY.label === 'string')) fail('refY must be { value inside the y domain, label }');
  bands.forEach((b) => { if (!inside(ax, b.from) || !inside(ax, b.to)) fail(`band ${b.from} → ${b.to} is outside the x domain [${ax.domain.join(', ')}]`); });
  return {
    w, h, plot, toX, toY,
    xLabel: ax.label, yLabel: ay.label,
    xTicks: ax.ticks.filter((t) => inside(ax, t.value)).map((t) => ({ ...t, x: toX(t.value) })),
    yTicks: ay.ticks.filter((t) => inside(ay, t.value)).map((t) => ({ ...t, y: toY(t.value) })),
    bands: bands.map((b) => ({ label: b.label ?? '', labelY: (b.labelAt ?? 'top') === 'top' ? plot.top + 11 : plot.bottom - 5, x: toX(b.from) + BAND_INSET, width: Math.max(1, toX(b.to) - toX(b.from) - 2 * BAND_INSET) })),
    series: series.map((s) => {
      s.points.forEach(([x, y]) => checkPoint(ax, ay, x, y));
      const pts = s.points.map(([x, y]) => ({ x: toX(x), y: toY(y) }));
      return { label: s.label ?? '', style: s.style ?? 'solid', tone: s.tone ?? 'ink', points: pts, path: pts.map((p, i) => `${i ? 'L' : 'M'}${round(p.x)} ${round(p.y)}`).join(''), labelPos: seriesLabel(pts, s.labelAt ?? 'end') };
    }),
    markers: markers.map((mk) => { checkPoint(ax, ay, mk.x, mk.y); return placeMarker(mk, toX, toY, plot); }),
    refY: refY ? { y: toY(refY.value), x1: plot.left, x2: plot.right, label: refY.label } : null,
  };
}

const round = (v) => Number(v.toFixed(2));

// The point halfway along a polyline (screen length), for a label placed mid-series.
function midpoint(pts) {
  const lens = pts.slice(1).map((p, i) => Math.hypot(p.x - pts[i].x, p.y - pts[i].y));
  let left = lens.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < lens.length; i += 1) {
    if (left <= lens[i] && lens[i] > 0) return { x: pts[i].x + ((pts[i + 1].x - pts[i].x) * left) / lens[i], y: pts[i].y + ((pts[i + 1].y - pts[i].y) * left) / lens[i] };
    left -= lens[i];
  }
  return pts[pts.length - 1] ?? null;
}

// A series label sits above its end (anchor end), or above-left of its midpoint (labelAt 'mid', anchor end).
function seriesLabel(pts, labelAt) {
  const at = labelAt === 'mid' ? midpoint(pts) : pts[pts.length - 1];
  if (!at) return null;
  return labelAt === 'mid' ? { x: at.x - 6, y: at.y - 6 } : { x: at.x, y: at.y - 6 };
}

function drawAxes(g, L) {
  const { plot } = L;
  L.yTicks.forEach((t) => {
    svgEl('line', { class: 'g-plot-grid', x1: plot.left, y1: round(t.y), x2: plot.right, y2: round(t.y) }, g);
    text(g, plot.left - TICK - 3, round(t.y), t.label, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  });
  L.xTicks.forEach((t) => {
    svgEl('line', { class: 'g-plot-tick', x1: round(t.x), y1: plot.bottom, x2: round(t.x), y2: plot.bottom + TICK }, g);
    text(g, round(t.x), plot.bottom + TICK + 11, t.label, 'g-label', { 'text-anchor': 'middle' });
  });
  svgEl('path', { class: 'g-plot-axis', d: `M${plot.left} ${plot.top}V${plot.bottom}H${plot.right}` }, g);
  if (L.xLabel) text(g, (plot.left + plot.right) / 2, L.h - 3, L.xLabel, 'g-label', { 'text-anchor': 'middle' });
  if (L.yLabel) text(g, 0, 11, L.yLabel, 'g-label');
}

function drawSeries(g, L) {
  L.series.forEach((s) => {
    const path = svgEl('path', { class: `g-series g-series--${s.style} g-tone--${s.tone}`, d: s.path }, g);
    if (s.label) svgEl('title', {}, path).textContent = s.label;
    if (s.label && s.labelPos) text(g, round(s.labelPos.x), round(s.labelPos.y), s.label, `g-label g-series-label g-tone--${s.tone}`, { 'text-anchor': 'end' });
  });
}

function drawMarkers(g, L) {
  L.markers.forEach((m) => {
    const dot = svgEl('circle', { class: 'g-marker', cx: round(m.x), cy: round(m.y), r: MARKER_R }, g);
    if (m.label) {
      svgEl('title', {}, dot).textContent = m.label;
      text(g, round(m.labelX), round(m.labelY), m.label, 'g-label g-marker-label', { 'text-anchor': m.labelAnchor });
    }
    if (m.followed) selectionMark(g, { x: m.x - MARKER_R, y: m.y - MARKER_R, w: 2 * MARKER_R, h: 2 * MARKER_R, pad: 3, rx: MARKER_R + 3 });
  });
}

// A small line chart (P3-R10): optional log axes with labeled ticks; series solid or muted (never dashed);
// shaded x-range bands with printed labels; a dashed, labeled horizontal reference (refY, drawn like the bars
// reference line); markers with printed labels. Every number a learner must read is printed, never only drawn.
export function curvePlot(parent, { x, y, w, h, xAxis, yAxis, series = [], markers = [], bands = [], refY = null, label = 'plot' }) {
  const L = curvePlotLayout({ w, h, xAxis, yAxis, series, markers, bands, refY });
  const g = group(parent, 'g-plot', x, y, { role: 'img', 'aria-label': [label, ...L.markers.map((m) => m.label).filter(Boolean), L.refY?.label].filter(Boolean).join('; ') });
  L.bands.forEach((b) => {
    svgEl('rect', { class: 'g-band', x: round(b.x), y: L.plot.top, width: round(b.width), height: L.plot.bottom - L.plot.top }, g);
    if (b.label) text(g, round(b.x + b.width / 2), b.labelY, b.label, 'g-label g-band-label', { 'text-anchor': 'middle' });
  });
  drawAxes(g, L);
  if (L.refY) {
    svgEl('line', { class: 'g-plot-ref', x1: L.refY.x1, y1: round(L.refY.y), x2: L.refY.x2, y2: round(L.refY.y) }, g);
    text(g, L.refY.x1 + 4, round(L.refY.y) - 4, L.refY.label, 'g-label g-plot-ref-label'); // left end: series labels sit at the right
  }
  drawSeries(g, L);
  drawMarkers(g, L);
  return g;
}

// ---- roofline (gpu-primer §4): two series on log-log axes, the ridge printed at the bend ----
const ROOF_X = [0.1, 1e4];
const decades = ([lo, hi]) => {
  const out = [];
  for (let e = Math.ceil(Math.log10(lo) - EPS); e <= Math.floor(Math.log10(hi) + EPS); e += 1) out.push(10 ** e);
  return out;
};
const failRoof = (msg) => { throw new RangeError(`glyphs.roofline: ${msg}`); };

function checkRoof({ peakTflops, bandwidthTBps, xDomain, ridgeRange, points }) {
  if (!(isNum(peakTflops) && peakTflops > 0 && isNum(bandwidthTBps) && bandwidthTBps > 0)) failRoof(`peakTflops and bandwidthTBps must be finite numbers > 0, got ${peakTflops}, ${bandwidthTBps}`);
  if (!(Array.isArray(xDomain) && xDomain[0] > 0 && xDomain[1] > xDomain[0])) failRoof(`xDomain must be [low, high] with 0 < low < high, got ${JSON.stringify(xDomain)}`);
  if (ridgeRange != null && !(Array.isArray(ridgeRange) && isNum(ridgeRange[0]) && ridgeRange[0] > 0 && ridgeRange[1] > ridgeRange[0])) failRoof(`ridgeRange must be [low, high] with 0 < low < high, got ${JSON.stringify(ridgeRange)}`);
  points.forEach((p) => { if (!(isNum(p.intensity) && p.intensity > 0)) failRoof(`point intensity must be a finite number > 0, got ${p.intensity}`); });
}

function roofSeries({ peakTflops: peak, bandwidthTBps: bw, xDomain, yDomain, ridge }) {
  const [x0, x1] = xDomain;
  const start = Math.max(x0, yDomain[0] / bw);
  const bend = Math.min(ridge, x1);
  const memory = { points: [[start, bw * start], [bend, bw * bend]], label: `${formatTick(bw)} TB/s`, tone: 'memory', labelAt: 'mid' };
  const compute = { points: [[Math.max(ridge, x0), peak], [x1, peak]], label: `${formatTick(peak)} TFLOPS`, tone: 'compute' };
  return [...(ridge > x0 ? [memory] : []), ...(ridge < x1 ? [compute] : [])];
}

// Geometry only (pure): the curvePlot input (`spec`) and layout (`plot`) of the roof, the ridge (value, x, y, printed
// label) and each point on the roof with its attainable TFLOPS and bound ('memory' left of the ridge, 'compute' from it on).
export function rooflineLayout({ w, h, peakTflops, bandwidthTBps, xDomain = ROOF_X, yDomain = null, ridgeRange = null, points = [] }) {
  checkRoof({ peakTflops, bandwidthTBps, xDomain, ridgeRange, points });
  const ridge = peakTflops / bandwidthTBps;
  const yd = yDomain ?? [10 ** Math.floor(Math.log10(bandwidthTBps * xDomain[0])), 10 ** Math.ceil(Math.log10(peakTflops * 2))];
  const placed = points.map((p) => {
    const attainable = Math.min(peakTflops, p.intensity * bandwidthTBps);
    return { ...p, attainable, bound: p.intensity >= ridge ? 'compute' : 'memory' };
  });
  const spec = {
    w, h,
    xAxis: { label: 'arithmetic intensity (FLOPs per byte)', log: true, domain: xDomain, ticks: decades(xDomain) },
    yAxis: { label: 'TFLOPS', log: true, domain: yd, ticks: decades(yd) },
    series: roofSeries({ peakTflops, bandwidthTBps, xDomain, yDomain: yd, ridge }),
    markers: placed.map((p) => ({ x: p.intensity, y: p.attainable, followed: p.followed, label: p.label ? `${p.label} · ${formatTick(p.intensity)}` : formatTick(p.intensity) })),
    bands: ridgeRange ? [{ from: ridgeRange[0], to: ridgeRange[1], label: `ridge ${formatTick(ridgeRange[0])}–${formatTick(ridgeRange[1])}`, labelAt: 'bottom' }] : [],
  };
  const plot = curvePlotLayout(spec);
  return {
    spec, plot,
    ridge: { value: ridge, x: plot.toX(Math.min(Math.max(ridge, xDomain[0]), xDomain[1])), y: plot.toY(peakTflops), label: ridgeRange ? null : `ridge point ${formatTick(Math.round(ridge))}` },
    points: placed.map((p, i) => ({ ...p, x: plot.markers[i].x, y: plot.markers[i].y })),
  };
}

// A chip's roofline from two numbers (gpu-primer §4): the sloped roof in --sem-memory, the flat roof in
// --sem-compute, the ridge printed at the bend (or a printed band for ridgeRange), each point on the roof.
export function roofline(parent, { x, y, w = 360, h = 240, peakTflops, bandwidthTBps, xDomain = ROOF_X, yDomain = null, ridgeRange = null, points = [], label = 'roofline' }) {
  const L = rooflineLayout({ w, h, peakTflops, bandwidthTBps, xDomain, yDomain, ridgeRange, points });
  const spoken = [...L.plot.series.map((s) => s.label), L.ridge.label ?? L.plot.bands[0]?.label].filter(Boolean).join(', ');
  const g = curvePlot(parent, { x, y, ...L.spec, label: `${label}: ${spoken}` });
  g.classList.add('g-roofline');
  // Above-left of the bend: the slope comes up from the lower left, the flat roof's label sits at the right end.
  if (L.ridge.label) text(g, round(L.ridge.x) - 4, round(L.ridge.y) - 8, L.ridge.label, 'g-label g-ridge-label', { 'text-anchor': 'end' });
  return g;
}
