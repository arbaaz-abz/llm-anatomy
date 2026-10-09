// Glyphs added by the pilot storyboards: verdict badges, clip lines, KV block pools,
// block tables and memory bars.
import { svgEl, group, text, hatchRect, requestSlot, REQUEST_OWNER } from './core.js';
import { selectionMark } from './architecture.js';

export { requestSlot }; // lives in core.js (token needs it); still exported from here and from glyphs.js

export function verdict(parent, { x, y, ok, r = 9 }) {
  const g = group(parent, `g-verdict g-verdict--${ok ? 'ok' : 'bad'}`, x, y, { role: 'img', 'aria-label': ok ? 'correct' : 'wrong' });
  svgEl('circle', { r }, g);
  const s = r / 9;
  svgEl('path', { d: ok ? `M${-4 * s} ${0.5 * s} l${3 * s} ${3 * s} l${5 * s} ${-6 * s}` : `M${-3.5 * s} ${-3.5 * s} l${7 * s} ${7 * s} M${3.5 * s} ${-3.5 * s} l${-7 * s} ${7 * s}` }, g);
  return g;
}

// `format` (S7): prints the tick labels, the marker label and the aria-label (default two decimals).
export function clipLine(parent, { x, y, w = 220, lo, hi, band, marker, label, format = (v) => v.toFixed(2) }) {
  if (!(hi > lo)) throw new RangeError(`glyphs.clipLine: hi (${hi}) must exceed lo (${lo})`);
  if (marker != null && !Number.isFinite(marker)) throw new RangeError(`glyphs.clipLine: marker must be a finite number, got ${marker}`);
  const scale = (v) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * w;
  const name = label ?? 'ratio';
  const g = group(parent, 'g-clip', x, y, { role: 'img', 'aria-label': marker == null ? name : `${name} ${format(marker)}` });
  if (label) text(g, 0, -22, label, 'g-label', { 'text-anchor': 'start' });
  if (band) svgEl('rect', { class: 'g-band', x: scale(band[0]), y: -8, width: scale(band[1]) - scale(band[0]), height: 16, rx: 2 }, g);
  svgEl('line', { class: 'g-axis', x1: 0, y1: 0, x2: w, y2: 0 }, g);
  const ticks = [lo, ...(band ?? []), hi];
  ticks.forEach((v) => {
    svgEl('line', { class: 'g-tick', x1: scale(v), y1: -4, x2: scale(v), y2: 4 }, g);
    text(g, scale(v), 18, format(v), 'g-label g-tick-label', { 'text-anchor': 'middle' });
  });
  if (marker != null) {
    const mx = scale(marker);
    svgEl('polygon', { class: 'g-marker', points: `${mx},-2 ${mx - 5},-11 ${mx + 5},-11` }, g);
    text(g, mx, -14, format(marker), 'g-marker-label', { 'text-anchor': 'middle', 'font-weight': 600 });
  }
  return g;
}

const SLOT_STATES = new Set(['filled', 'reserved', 'cached', 'free']);
const isCount = (n) => Number.isInteger(n) && n >= 1;
const REFS_RISE = 4; // a block's reference count baseline sits this far above its frame, beside the block number

function checkRefs(refs, blocks) {
  if (refs == null) return;
  if (!(Array.isArray(refs) && refs.length === blocks && refs.every((n) => Number.isInteger(n) && n >= 0))) throw new RangeError(`glyphs.blockPool: refs must be ${blocks} integers ≥ 0, one per block`);
}

// States: filled (a request holds it, its --req hue), reserved (hatched: held but empty), cached (S6, P4-R11: filled,
// no request holds it, kept for reuse: a muted fill, its letter kept), free (a faint fill, never dashed or outlined).
// `refs` (S6): one reference count per block, printed small beside the block number when above 1.
export function blockPool(parent, { x, y, blocks, blockSize, slots, cell = 14, perRow = 6, refs = null }) {
  if (!isCount(blocks) || !isCount(blockSize)) throw new RangeError(`glyphs.blockPool: blocks and blockSize must be integers ≥ 1, got ${blocks} × ${blockSize}`);
  if (!Array.isArray(slots) || slots.length !== blocks * blockSize) {
    throw new RangeError(`glyphs.blockPool: expected ${blocks * blockSize} slots (blocks × blockSize), got ${slots?.length}`);
  }
  slots.forEach((slot, i) => {
    if (!SLOT_STATES.has(slot?.state ?? 'free')) throw new RangeError(`glyphs.blockPool: slot ${i} has state "${slot.state}"; use filled, reserved, cached or free`);
  });
  checkRefs(refs, blocks);
  const vH = Math.round(cell * 0.6);
  const blockW = blockSize * (cell + 1) + 5;
  const blockH = cell + vH + 7;
  const stepX = blockW + 10;
  const stepY = blockH + 18;
  const g = group(parent, 'g-pool', x, y, { role: 'img', 'aria-label': `KV pool: ${blocks} blocks of ${blockSize}` });
  for (let b = 0; b < blocks; b += 1) {
    const bx = (b % perRow) * stepX;
    const by = Math.floor(b / perRow) * stepY + 12;
    text(g, bx, by - 4, b, 'g-sub g-block-index');
    if (refs && refs[b] > 1) {
      const mark = svgEl('g', { class: 'g-refs' }, g); // the title sits beside the number, not inside its <text>
      svgEl('title', {}, mark).textContent = `block ${b}: ${refs[b]} references`;
      text(mark, bx + blockW, by - REFS_RISE, refs[b], 'g-sub g-block-refs', { 'text-anchor': 'end' });
    }
    svgEl('rect', { class: 'g-frame', x: bx, y: by, width: blockW, height: blockH, rx: 3 }, g);
    for (let s = 0; s < blockSize; s += 1) {
      const { owner = '', state = 'free' } = slots[b * blockSize + s];
      const sx = bx + 3 + s * (cell + 1);
      const sy = by + 3;
      const hued = state === 'filled' || state === 'reserved';
      const slot = svgEl('g', { class: `g-slot g-slot--${state}`, 'data-req': hued ? requestSlot(owner) : null }, g);
      svgEl('rect', { class: 'g-slot-k', x: sx, y: sy, width: cell, height: cell, rx: 2 }, slot);
      svgEl('rect', { class: 'g-slot-v', x: sx, y: sy + cell + 1, width: cell, height: vH, rx: 2 }, slot);
      if (state === 'reserved') hatchRect(slot, { x: sx, y: sy, width: cell, height: cell + 1 + vH, rx: 2 });
      text(slot, sx + cell / 2, sy + cell / 2, state === 'free' ? '·' : owner, 'g-text');
      svgEl('title', {}, slot).textContent = `block ${b}, slot ${s}: ${state}${owner ? ` (${owner})` : ''}`;
    }
  }
  return g;
}

export function blockTable(parent, { x, y, rows, title }) {
  const colW = 56;
  const rowH = 16;
  const hasRef = rows.some((r) => r.ref != null);
  const cols = hasRef ? ['logical', 'physical', 'ref'] : ['logical', 'physical'];
  const g = group(parent, 'g-table', x, y, { role: 'img', 'aria-label': `${title ?? 'block table'}: ${rows.length} rows` });
  if (title) text(g, 0, -14, title, 'g-label');
  cols.forEach((c, j) => text(g, j * colW, 0, c.toUpperCase(), 'g-sub g-th'));
  svgEl('line', { class: 'g-rule', x1: 0, y1: 5, x2: cols.length * colW - 8, y2: 5 }, g);
  rows.forEach((row, i) => {
    const ty = 5 + (i + 1) * rowH;
    text(g, 0, ty, row.logical);
    text(g, colW - 16, ty, '→', 'g-label');
    text(g, colW, ty, row.physical);
    if (hasRef) text(g, 2 * colW, ty, row.ref ?? '', 'g-label');
  });
  return g;
}

// ---- prefixTree (prefix-caching §4, P4-R14): blocks keyed by their tokens, one node per block ----
const PT = { minW: 64, h: 26, edge: 14, rowGap: 14, pad: 12, charW: 6.6, wrapAfter: 4 };
const PT_STATES = ['hit', 'new', 'cached', 'evicted'];
const failTree = (msg) => { throw new RangeError(`glyphs.prefixTree: ${msg}`); };

function checkTree(nodes, wrapAfter) {
  if (!Array.isArray(nodes) || nodes.length === 0) failTree('nodes must be a non-empty array');
  if (!(Number.isInteger(wrapAfter) && wrapAfter >= 1)) failTree(`wrapAfter must be an integer ≥ 1, got ${wrapAfter}`);
  const ids = new Set();
  nodes.forEach((n) => {
    if (typeof n?.id !== 'string' || n.id === '') failTree('every node needs an id');
    if (ids.has(n.id)) failTree(`duplicate node id "${n.id}"`);
    ids.add(n.id);
  });
  nodes.forEach((n) => {
    if (n.parent != null && !ids.has(n.parent)) failTree(`node "${n.id}" has unknown parent "${n.parent}"`);
    if (!PT_STATES.includes(n.state)) failTree(`node "${n.id}" state must be one of ${PT_STATES.join(', ')}, got ${n.state}`);
    if (typeof n.label !== 'string') failTree(`node "${n.id}" needs a label`);
    if (n.state === 'new' && !REQUEST_OWNER.test(n.owner ?? '')) failTree(`node "${n.id}" is new and needs an owner A–D`);
  });
}

// Depth-first from the roots in input order. A leaf takes the next lane; a lane is one row, or two when its leaf lies
// deeper than `wrapAfter` blocks (the rest of the path wraps to a second row, starting under the first block column).
function placeTree(nodes, wrapAfter) {
  const byId = new Map(nodes.map((n) => [n.id, { ...n, children: [] }]));
  byId.forEach((n) => { if (n.parent != null) byId.get(n.parent).children.push(n); });
  const lanes = [];
  const visit = (n, depth) => {
    n.depth = depth;
    if (n.children.length === 0) { lanes.push(depth > wrapAfter ? 2 : 1); n.lane = lanes.length - 1; return; }
    n.children.forEach((c) => visit(c, depth + 1));
    n.lane = n.children[0].lane;
  };
  const roots = [...byId.values()].filter((n) => n.parent == null);
  roots.forEach((r) => visit(r, 0));
  if ([...byId.values()].some((n) => n.depth === undefined)) failTree('the nodes form a cycle');
  const laneTop = lanes.map((_, i) => lanes.slice(0, i).reduce((a, b) => a + b, 0));
  byId.forEach((n) => {
    const wrapped = n.depth > wrapAfter;
    n.col = wrapped ? n.depth - wrapAfter : n.depth;
    n.row = laneTop[n.lane] + (wrapped ? 1 : 0);
  });
  return { list: [...byId.values()], rows: lanes.reduce((a, b) => a + b, 0) };
}

const edgePath = (p, c) => {
  const cy = c.y + PT.h / 2;
  if (c.x > p.x) {
    const mid = p.x + p.w + PT.edge / 2;
    return `M${p.x + p.w} ${p.y + PT.h / 2}H${mid}V${cy}H${c.x}`;
  }
  const midY = p.y + PT.h + PT.rowGap / 2;
  return `M${p.x + p.w / 2} ${p.y + PT.h}V${midY}H${c.x - PT.edge / 2}V${cy}H${c.x}`;
};

// Geometry only (pure): a left-to-right tree, one column per depth (a column is as wide as its widest node, at least
// 64 px; a node is wide enough for its label), siblings stacked, edges as elbows from parent to child.
export function prefixTreeLayout({ nodes, wrapAfter = PT.wrapAfter }) {
  checkTree(nodes, wrapAfter);
  const { list, rows } = placeTree(nodes, wrapAfter);
  const nodeW = (n) => Math.max(PT.minW, Math.ceil(n.label.length * PT.charW) + PT.pad);
  const cols = Math.max(...list.map((n) => n.col)) + 1;
  const colW = Array.from({ length: cols }, (_, c) => Math.max(...list.filter((n) => n.col === c).map(nodeW), 0));
  const colX = colW.map((_, c) => colW.slice(0, c).reduce((a, b) => a + b + PT.edge, 0));
  const placed = list.map((n) => ({
    id: n.id, parent: n.parent ?? null, label: n.label, state: n.state, owner: n.owner ?? null, followed: Boolean(n.followed),
    depth: n.depth, col: n.col, row: n.row, x: colX[n.col], y: n.row * (PT.h + PT.rowGap), w: colW[n.col], h: PT.h,
  }));
  const at = new Map(placed.map((n) => [n.id, n]));
  const edges = placed.filter((n) => n.parent !== null).map((n) => ({ from: n.parent, to: n.id, d: edgePath(at.get(n.parent), n) }));
  return { nodes: placed, edges, width: Math.max(...placed.map((n) => n.x + n.w)), height: rows * PT.h + (rows - 1) * PT.rowGap };
}

const treeSpoken = (nodes) => PT_STATES.map((st) => [st, nodes.filter((n) => n.state === st).length]).filter(([, n]) => n > 0).map(([st, n]) => `${n} ${st}`).join(', ');

// One node per block: hit (reused: --sem-ok), new (computed for its `owner`: that request's --req hue), cached (kept,
// unused: muted) or evicted (hatched: no longer counts). `followed` nodes wear the selection mark. Pass the label
// you want printed (the page shortens non-followed ones); nodes widen to fit it. Edges are plain lines, not flows.
export function prefixTree(parent, { x, y, nodes, wrapAfter = PT.wrapAfter, label = 'prefix tree' }) {
  const L = prefixTreeLayout({ nodes, wrapAfter });
  const g = group(parent, 'g-ptree', x, y, { role: 'img', 'aria-label': `${label}: ${L.nodes.length} blocks: ${treeSpoken(L.nodes)}` });
  L.edges.forEach((e) => svgEl('path', { class: 'g-ptree-edge', d: e.d }, g));
  L.nodes.forEach((n) => {
    const node = svgEl('g', { class: `g-ptree-node g-ptree--${n.state}`, 'data-req': n.state === 'new' ? requestSlot(n.owner) : null }, g);
    svgEl('title', {}, node).textContent = `${n.label}: ${n.state}`;
    svgEl('rect', { class: 'g-frame', x: n.x, y: n.y, width: n.w, height: n.h, rx: 5 }, node);
    if (n.state === 'evicted') hatchRect(node, { x: n.x, y: n.y, width: n.w, height: n.h, rx: 5 });
    text(node, n.x + n.w / 2, n.y + n.h / 2, n.label, 'g-text');
  });
  L.nodes.filter((n) => n.followed).forEach((n) => selectionMark(g, { x: n.x, y: n.y, w: n.w, h: n.h }));
  return g;
}
