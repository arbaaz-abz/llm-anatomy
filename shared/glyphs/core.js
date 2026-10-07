// Core glyphs: SVG helpers, the value scale, cells, tokens, vectors, matrices, heatmaps,
// blocks and flows. Pure DOM builders: every function appends to `parent` (inside an <svg>)
// and returns the <g> it made. Coordinates are SVG user units. No global state is read.

export const SVG_NS = 'http://www.w3.org/2000/svg';
const MIN_TOKEN_WIDTH = 28;
const MIN_CELL_FOR_TEXT = 36; // narrower cells keep the number in their tooltip only
const CELL_GAP = 3; // surface gap between cells (dataviz spacer); the cell stride stays `size`
export const NUMBER_CELL = 40; // the cell size lessons use when numbers must be readable ("−0.87" + padding)
const LABEL_FONT = 11; // the one diagram label size (spec §5.4); subscripts use .g-sub
const TOKEN_HEIGHT = 24;

export function svgEl(tag, attrs = {}, parent = null) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value != null) el.setAttribute(key, String(value));
  }
  parent?.append(el);
  return el;
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function assertMaxAbs(maxAbs) {
  if (!(maxAbs > 0) || !Number.isFinite(maxAbs)) throw new RangeError(`glyphs: maxAbs must be a finite number > 0, got ${maxAbs}`);
}

// Diverging value scale as a theme-aware color-mix() string. Apply with el.style.fill.
export function valueColor(v, maxAbs) {
  assertMaxAbs(maxAbs);
  const ratio = v / maxAbs; // NaN for undefined, strings or NaN itself; ±Infinity clamps to the poles
  if (Number.isNaN(ratio)) return 'var(--val-zero)';
  const t = clamp(ratio, -1, 1);
  const pct = Math.round(Math.abs(t) * 100);
  if (pct === 0) return 'var(--val-zero)';
  return `color-mix(in oklab, var(--val-zero), var(--val-${t < 0 ? 'neg' : 'pos'}) ${pct}%)`;
}

// 0..10 bucket of |v|/maxAbs; theme.css picks the cell's text ink per bucket.
export function valueLevel(v, maxAbs) {
  assertMaxAbs(maxAbs);
  const ratio = Math.abs(v) / maxAbs;
  if (Number.isNaN(ratio)) return 0;
  return Math.round(clamp(ratio, 0, 1) * 10);
}

// Recovers the 0..10 ink level from a valueColor() string, so a token filled with one
// can switch its text ink exactly like a cell. Any other fill reads as level 0.
export function levelFromFill(fill) {
  const pct = Number(String(fill ?? '').match(/ (\d+)%\)$/)?.[1]);
  return Number.isFinite(pct) ? Math.round(clamp(pct, 0, 100) / 10) : 0;
}

// Integers as they are; |v| ≥ 10 rounded; otherwise two decimals with a leading zero ("−0.87").
// At most five characters, which fits a NUMBER_CELL with padding at the 11px label size.
export function formatCell(v) {
  if (v === -Infinity) return '−∞';
  if (v === Infinity) return '∞';
  if (!Number.isFinite(v)) return '·';
  const a = Math.abs(v);
  const sign = v < 0 ? '−' : '';
  if (Number.isInteger(v)) return sign + String(a);
  if (a >= 10) return sign + String(Math.round(a));
  return sign + a.toFixed(2);
}

export const tokenWidth = (text, fontSize = LABEL_FONT) =>
  Math.max(MIN_TOKEN_WIDTH, Math.round(String(text).length * fontSize * 0.62 + 16));

export const maxAbsOf = (values) => {
  const finite = values.flat().filter(Number.isFinite).map(Math.abs);
  return finite.length ? Math.max(...finite, 1e-9) : 1;
};

export function group(parent, cls, x = 0, y = 0, attrs = {}) {
  return svgEl('g', { class: `glyph ${cls}`, transform: `translate(${x} ${y})`, ...attrs }, parent);
}

export function text(parent, x, y, str, cls = '', attrs = {}) {
  const el = svgEl('text', { x, y, class: cls || null, ...attrs }, parent);
  el.textContent = String(str);
  return el;
}

// One <pattern> per <svg>, created on first use. The id is remembered on the <svg>
// (dataset survives replaceChildren), so redrawing into the same <svg> reproduces the same DOM.
let hatchCount = 0;
export function hatchFill(parent) {
  const svg = parent.ownerSVGElement ?? (parent.tagName === 'svg' ? parent : null);
  if (!svg) throw new Error('glyphs: parent must be an <svg> or inside one');
  let pattern = svg.querySelector('pattern.g-hatch-pattern');
  if (!pattern) {
    svg.dataset.hatchId ??= `g-hatch-${hatchCount += 1}`;
    const id = svg.dataset.hatchId;
    const defs = svg.querySelector('defs') ?? svgEl('defs', {}, svg);
    pattern = svgEl('pattern', { id, class: 'g-hatch-pattern', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 6 }, pattern);
  }
  return `url(#${pattern.id})`;
}

// Sizes an <svg> to its drawn content plus `pad` on every side (pad must cover half the
// widest stroke). For figures that are drawn once; a stepper stage keeps a fixed size instead.
export function fitViewBox(svg, pad = 4) {
  const b = svg.getBBox();
  const box = { x: Math.floor(b.x - pad), y: Math.floor(b.y - pad), width: Math.ceil(b.width + 2 * pad), height: Math.ceil(b.height + 2 * pad) };
  svg.setAttribute('viewBox', `${box.x} ${box.y} ${box.width} ${box.height}`);
  svg.setAttribute('width', box.width);
  svg.setAttribute('height', box.height);
  return box;
}

export function hatchRect(parent, attrs) {
  return svgEl('rect', { ...attrs, fill: hatchFill(parent), class: 'g-hatch' }, parent);
}

// One value cell. Masked cells (mask === false or v === −∞) are hatched.
export function cell(parent, { x, y, size, v, maxAbs, masked = false }) {
  const isMasked = masked || v === -Infinity;
  const g = svgEl('g', { class: `g-cell${isMasked ? ' g-cell--masked' : ''}`, transform: `translate(${x} ${y})`, 'data-level': isMasked ? 0 : valueLevel(v, maxAbs) }, parent);
  const inset = { x: CELL_GAP / 2, y: CELL_GAP / 2, width: size - CELL_GAP, height: size - CELL_GAP, rx: 3 };
  const rect = svgEl('rect', inset, g);
  rect.style.fill = isMasked ? 'var(--surface)' : valueColor(v, maxAbs);
  if (isMasked) hatchRect(g, inset);
  const title = svgEl('title', {}, g);
  title.textContent = isMasked ? 'masked (−∞)' : Number(v).toFixed(3);
  if (size >= MIN_CELL_FOR_TEXT) text(g, size / 2, size / 2, isMasked ? '−∞' : formatCell(v), 'g-text');
  return g;
}

export function token(parent, { x, y, text: label, index, state = 'idle', fill, hatched = false }) {
  const w = tokenWidth(label);
  const g = group(parent, `g-token g-token--${state}`, x, y, { role: 'img', 'aria-label': index == null ? String(label) : `${label}, token ${index}`, 'data-level': fill ? levelFromFill(fill) : null });
  const rect = svgEl('rect', { class: 'g-frame', width: w, height: TOKEN_HEIGHT, rx: 7 }, g);
  if (fill) rect.style.fill = fill;
  if (hatched) hatchRect(g, { width: w, height: TOKEN_HEIGHT, rx: 7 });
  text(g, w / 2, TOKEN_HEIGHT / 2, label, 'g-text');
  if (index != null) text(g, w - 4, TOKEN_HEIGHT - 3, index, 'g-sub', { 'text-anchor': 'end' });
  return g;
}

export function vector(parent, { x, y, values, cell: size = 18, orient = 'col', maxAbs = maxAbsOf(values), label }) {
  const g = group(parent, `g-vector g-vector--${orient}`, x, y);
  if (label) {
    if (orient === 'col') text(g, size / 2, -10, label, 'g-label', { 'text-anchor': 'middle' });
    else text(g, -10, size / 2, label, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' });
  }
  values.forEach((v, i) => cell(g, { x: orient === 'row' ? i * size : 0, y: orient === 'col' ? i * size : 0, size, v, maxAbs }));
  return g;
}

function grid(parent, cls, { x, y, values, mask, cell: size = 18, maxAbs = maxAbsOf(values), label, rowLabels = [], colLabels = [] }) {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const g = group(parent, cls, x, y);
  const frame = svgEl('rect', { class: 'g-frame', x: -1, y: -1, width: cols * size + 2, height: rows * size + 2, rx: 3 }, g);
  svgEl('title', {}, frame).textContent = `${label ?? ''} [${rows} × ${cols}]`.trim();
  if (label) text(g, 0, -10 - (colLabels.length ? 14 : 0), `${label} [${rows} × ${cols}]`, 'g-label');
  rowLabels.forEach((r, i) => text(g, -10, i * size + size / 2, r, 'g-label', { 'text-anchor': 'end', 'dominant-baseline': 'central' }));
  colLabels.forEach((c, j) => text(g, j * size + size / 2, -8, c, 'g-label', { 'text-anchor': 'middle' }));
  values.forEach((row, i) => row.forEach((v, j) => cell(g, { x: j * size, y: i * size, size, v, maxAbs, masked: mask ? !mask[i]?.[j] : false })));
  return g;
}

export const matrix = (parent, opts) => grid(parent, 'g-matrix', opts);
export const heatmap = (parent, opts) => grid(parent, 'g-heatmap', opts);

export function block(parent, { x, y, w = 96, h = 40, label = '', state = 'idle' }) {
  const g = group(parent, `g-block g-block--${state}`, x, y, { role: 'img', 'aria-label': `${label} (${state})` });
  svgEl('rect', { class: 'g-frame', width: w, height: h, rx: 6 }, g);
  text(g, w / 2, h / 2, label, 'g-text');
  return g;
}

const CARRIES = new Set(['activation', 'gradient', 'kv', 'token']);

export function flow(parent, { from, to, carry = 'activation', progress = 0 }) {
  if (!CARRIES.has(carry)) throw new RangeError(`glyphs.flow: carry must be one of ${[...CARRIES].join(', ')}`);
  const [x1, y1] = from;
  const [x2, y2] = to;
  const g = group(parent, `g-flow g-flow--${carry}`, 0, 0);
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 7;
  const ex = x2 - Math.cos(angle) * headLen;
  const ey = y2 - Math.sin(angle) * headLen;
  svgEl('line', { class: 'g-path', x1, y1, x2: ex, y2: ey }, g);
  const wing = (sign) => `${x2 - Math.cos(angle - sign * 0.5) * headLen},${y2 - Math.sin(angle - sign * 0.5) * headLen}`;
  svgEl('polygon', { class: 'g-head', points: `${x2},${y2} ${wing(1)} ${wing(-1)}` }, g);
  const p = clamp(Number(progress) || 0, 0, 1);
  svgEl('circle', { class: 'g-dot', cx: x1 + (ex - x1) * p, cy: y1 + (ey - y1) * p, r: 5 }, g);
  return g;
}
