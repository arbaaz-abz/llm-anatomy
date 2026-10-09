// Pieces every frame is assembled from: the A–D request bars, the two kinds of pool (strips and blocks), the followed
// request's block table and the usage line. Each draws one state; the frames decide which state, at which progress.
import * as G from '@shared/glyphs.js';
import { BARS, STAGE, POOL_Y, BEFORE_X, AFTER_X, TABLE_X, TABLE_Y, BAR_X, BAR_W, layer, label, select, linked } from './stage.js';
import { drawPool, poolGeometry, usageBar, markBlocks } from './pool.js';
import { IDS, ADMITTED, POOL_SLOTS, MAX_LEN, BLOCK_SIZE, request } from './numbers.js';

// ---- the A–D request bars (they follow the after-lane's timeline: D starts at step 1) ----
export const barDecode = (id, step, scheme = 'after') => Math.min(Math.max(step - ADMITTED[scheme][id], 0), request(id).output);
export const barEnd = (id, step, scheme = 'after') => (request(id).prompt + barDecode(id, step, scheme)) * BARS.unit;
const barY = (id) => BARS.y + IDS.indexOf(id) * BARS.gap;

// ids: the requests drawn (a request not yet arrived is left out); slide 0 → 1 brings the bars in from the left.
export function requestBars(parent, { step, ids, follow = null, opacity = 1, slide = 1, scheme = 'after', override = null }) {
  const g = layer(parent, opacity);
  ids.forEach((id) => {
    const x = BARS.x - (1 - slide) * 18;
    const decode = override?.[id] ?? barDecode(id, step, scheme);
    G.request(g, { x, y: barY(id), prefill: request(id).prompt, decode, unit: BARS.unit, label: id, owner: id });
    if (follow === id) select(g, x, barY(id), (request(id).prompt + decode) * BARS.unit, BARS.h);
  });
  return g;
}

export const stepTag = (parent, step, opacity = 1) => label(parent, STAGE.w - 4, BARS.y + 4, `step ${step}`, { anchor: 'end', cls: 'g-label', opacity });

// ---- the before lane: three strips of 16, drawn as three blocks of 16 slots ----
export const STRIP = Object.freeze({ cell: 12, blocks: POOL_SLOTS / MAX_LEN, blockSize: MAX_LEN, perRow: 1 });
export const stripGeo = poolGeometry(STRIP);

export function beforePool(parent, slots, { opacity = 1, marks = true } = {}) {
  const g = layer(parent, opacity);
  drawPool(g, { x: BEFORE_X, y: POOL_Y, ...STRIP, slots });
  if (marks) {
    for (let i = 0; i < STRIP.blocks; i += 1) {
      const at = stripGeo.blockAt(i);
      label(g, BEFORE_X + at.w + 12, POOL_Y + at.y + at.h / 2, `slots ${i * MAX_LEN}–${(i + 1) * MAX_LEN - 1}`);
    }
  }
  return g;
}

// ---- the after lane: 12 blocks of 4, six to a row ----
export const BLOCKS = Object.freeze({ cell: 12, blocks: POOL_SLOTS / BLOCK_SIZE, blockSize: BLOCK_SIZE, perRow: 6 });
export const blockGeo = poolGeometry(BLOCKS);

export function afterPool(parent, slots, { refs = null, opacity = 1, follow = [] } = {}) {
  const g = layer(parent, opacity);
  drawPool(g, { x: AFTER_X, y: POOL_Y, ...BLOCKS, slots, refs });
  markBlocks(g, blockGeo, AFTER_X, POOL_Y, follow);
  return g;
}

// ---- the followed request's block table, on the left of the pool ----
export function tableView(parent, { rows, title, opacity = 1, y = TABLE_Y, withRef = false }) {
  if (opacity <= 0) return null;
  const g = layer(parent, opacity);
  const box = { x: TABLE_X - 2, y: y - 16, w: (withRef ? 168 : 112) - 6, h: 24 + rows.length * 16 };
  linked(g, 'table', box, (host) => G.blockTable(host, { x: TABLE_X, y, title, rows: rows.map((r) => ({ logical: r.logical, physical: r.physical, ...(withRef ? { ref: r.ref } : {}) })) }));
  return g;
}

// A table row's arrow starts after the physical number and ends at the near edge of its block.
export function pointer(parent, { row, y = TABLE_Y, withRef = false, block, progress, opacity = 1 }) {
  const to = blockGeo.blockAt(block);
  const fromX = TABLE_X + 56 + 24;
  const fromY = y + 5 + (row + 1) * 16 - 4;
  const g = layer(parent, opacity);
  G.flow(g, { from: [fromX, fromY], to: [AFTER_X + to.x - 1, POOL_Y + to.y + to.h / 2], carry: 'kv', progress });
  return g;
}

// ---- the usage line under a pool ----
export function usage(parent, { y, useful, reserved, free, opacity = 1 }) {
  const g = layer(parent, opacity);
  const link = G.svgEl('g', { 'data-link': 'waste' }, g);
  usageBar(link, { x: BAR_X, y, w: BAR_W, useful, reserved, free, minSegment: 18 });
  return g;
}

