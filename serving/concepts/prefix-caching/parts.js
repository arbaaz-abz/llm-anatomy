// prefix-caching drawing parts shared by the frames and the toy: the prefix tree, the KV pool, the free queue, the block table
// strip and the counters. Each takes the state to draw (built by scenes.js) and an origin; none keeps state.
import * as G from '@shared/glyphs.js';
import { formatInt } from '@math/core.js';
import { CELL, COUNTERS, STRIP, TABLE, TREE, POOL, WALK, QUEUE_Y, layer, note, wrapLines } from './stage.js';
import { queueText } from './scenes.js';

const HIT_FILL = 'color-mix(in oklab, var(--sem-ok) 40%, var(--surface))'; // the tree's hit fill, on a table chip
const ROW = 'row';

// The tree at (x, y). `owner` puts that request's letter chip at the head of the walk lines (the tree is too wide for a leaf chip).
export function drawTree(parent, nodes, { x = TREE.x, y = TREE.y, wrapAfter = 6, label = 'prefix tree' } = {}) {
  return G.prefixTree(parent, { x, y, nodes, wrapAfter, label });
}

// The followed request's blocks in full: its letter chip, then `parts` (already worded) wrapped to two lines.
export function drawWalk(parent, owner, parts, opacity = 1) {
  if (!owner || parts.length === 0 || opacity <= 0) return;
  const g = layer(parent, opacity);
  G.token(g, { x: TREE.x, y: WALK.chipY, text: owner, owner });
  wrapLines(parts, WALK.max).slice(0, WALK.y.length).forEach((line, i) => note(g, WALK.x, WALK.y[i], line, { cls: '' }));
}

// The KV pool: `blocks` = [{ state, owner }] per block; `slots` overrides single slots ({ block, slot, state }).
export function drawPool(parent, blocks, { x = POOL.x, y = POOL.y, blockSize = POOL.blockSize, perRow = POOL.perRow, overrides = [] } = {}) {
  const slots = blocks.flatMap((b, bi) => Array.from({ length: blockSize }, (_, si) => overrides.find((o) => o.block === bi && o.slot === si) ?? b));
  return G.blockPool(parent, { x, y, blocks: blocks.length, blockSize, slots, perRow });
}

export function drawQueue(parent, queue, { x = POOL.x, y = QUEUE_Y, opacity = 1, suffix = '' } = {}) {
  if (opacity <= 0) return;
  const line = `free queue (evict from the left): ${queue.length ? queueText(queue) : 'empty'}`;
  note(layer(parent, opacity), x, y, suffix ? `${line}  ${suffix}` : line);
}

// Frame 6: a bracket joining two nodes that print the same words (A's and D's "Where did you sit"), labeled; plain 1 px lines.
export function drawTwin(parent, nodes, idA, idD, { x = TREE.x, y = TREE.y, wrapAfter = 6 } = {}) {
  const L = G.prefixTreeLayout({ nodes, wrapAfter });
  const a = L.nodes.find((n) => n.id === idA);
  const d = L.nodes.find((n) => n.id === idD);
  if (!a || !d) return;
  const edgeX = x + a.x + a.w;
  const g = G.svgEl('g', { class: 'glyph g-ptree' }, parent);
  G.svgEl('path', { class: 'g-ptree-edge', d: `M${edgeX} ${y + d.y + d.h / 2}h8V${y + a.y + a.h / 2}h-8` }, g);
  note(g, edgeX + 16, y + d.y + d.h / 2 + 4, 'same words, different key');
}

// The active request's block table: one chip per block, the hit ones in the hit fill and the new ones in the request's hue.
// Returns each chip's centre so a flow can end on it.
export function drawStrip(parent, title, chips, { x = STRIP.x, y = STRIP.y, opacity = 1 } = {}) {
  const g = layer(parent, opacity);
  const centres = [];
  let cursor = x;
  note(g, cursor, y - 3, title);
  cursor += title.length * 6.6 + 10;
  chips.forEach((c) => {
    const text = String(c.block);
    const host = c.kind === 'hit' ? G.svgEl('g', { 'data-link': 'hit' }, g) : g; // the math panel's hit terms outline these
    G.token(host, { x: cursor, y, text, owner: c.kind === 'new' ? c.owner : undefined, fill: c.kind === 'hit' ? HIT_FILL : undefined });
    centres.push(cursor + G.tokenWidth(text) / 2);
    cursor += G.tokenWidth(text) + STRIP.gap;
  });
  return centres;
}

// One counter cell: a one-cell vector at NUMBER_CELL; `null` leaves the cell empty until the number is shown.
function counterCell(parent, x, y, text) {
  const blank = text === null;
  return G.vector(parent, { x, y, values: [blank ? 0 : 1], cell: CELL, orient: ROW, maxAbs: 1, format: () => (blank ? '' : text) });
}

export const pctText = (hit, prompt) => G.formatShare(hit / prompt);

// "request B" over four counters: prompt tokens, from cache, computed, hit rate. A null value is not shown yet.
export function drawCounters(parent, title, values, opacity = 1) {
  const g = layer(parent, opacity);
  note(g, COUNTERS.labelX, COUNTERS.headY, title, { cls: '', anchor: 'end' });
  const labels = ['prompt tokens', 'from cache', 'computed', 'hit rate'];
  const texts = values.map((v, i) => (v === null ? null : (i === 3 ? v : formatInt(v))));
  labels.forEach((label, i) => {
    const y = COUNTERS.y + i * COUNTERS.pitch;
    note(g, COUNTERS.labelX, y + CELL / 2 + 4, label, { anchor: 'end' });
    counterCell(g, COUNTERS.cellX, y, texts[i]);
  });
}

// Frame 8's table: one row per request, three numbers each, the letter chip at the left.
export function drawTable(parent, rows, opacity = 1) {
  const g = layer(parent, opacity);
  ['prompt', 'cache', 'computed'].forEach((h, i) => note(g, TABLE.x + i * TABLE.step + CELL / 2, TABLE.headY, h, { anchor: 'middle' }));
  rows.forEach((r, ri) => {
    const y = TABLE.y + ri * TABLE.pitch;
    G.token(g, { x: TABLE.letterX - 14, y: y + (CELL - 24) / 2, text: r.id, owner: r.id });
    [r.prompt, r.cache, r.computed].forEach((v, i) => counterCell(g, TABLE.x + i * TABLE.step - 0, y, formatInt(v)));
  });
}
