// The KV pool as the stage and the toy draw it: slot states from the simulators, a geometry that mirrors G.blockPool
// (so marks can sit on a block or a slot), progressive flips between two states, and the usage bar with one-decimal shares.
import * as G from '@shared/glyphs.js';
import { shareText } from './format.js';

const POOL_RISE = 12; // G.blockPool puts block numbers above each block: the first row starts this far down
const POOL_PAD = 3; // slot inset inside a block

// Mirrors the constants inside G.blockPool (shared/glyphs/serving.js), so a selection mark or a chip can land on a block or a slot.
export function poolGeometry({ blocks, blockSize, cell, perRow }) {
  const vH = Math.round(cell * 0.6);
  const blockW = blockSize * (cell + 1) + 5;
  const blockH = cell + vH + 7;
  const stepX = blockW + 10;
  const stepY = blockH + 18;
  const rows = Math.ceil(blocks / perRow);
  const blockAt = (b) => ({ x: (b % perRow) * stepX, y: Math.floor(b / perRow) * stepY + POOL_RISE, w: blockW, h: blockH });
  const slotAt = (b, s) => { const at = blockAt(b); return { x: at.x + POOL_PAD + s * (cell + 1), y: at.y + POOL_PAD, size: cell }; };
  return { blockW, blockH, width: Math.min(blocks, perRow) * stepX - 10, height: POOL_RISE + (rows - 1) * stepY + blockH, blockAt, slotAt };
}

const FREE = Object.freeze({ state: 'free' });

// entries: [{ id, table, tokens }] → one slot per (physical block, position). A block shared by several tables is drawn once,
// as its first holder's (a full shared block reads the same for every holder).
export function slotsOf(entries, { blocks, blockSize }) {
  const slots = new Array(blocks * blockSize).fill(FREE);
  const taken = new Set();
  entries.forEach(({ id, table, tokens }) => table.forEach((b, i) => {
    if (taken.has(b)) return;
    taken.add(b);
    for (let s = 0; s < blockSize; s += 1) slots[b * blockSize + s] = { owner: id, state: i * blockSize + s < tokens ? 'filled' : 'reserved' };
  }));
  return slots;
}

const sameSlot = (a, b) => a.state === b.state && a.owner === b.owner;

// Slots that differ between two states flip one after another, in slot order, between progress `from` and `to`:
// the pool is a pure function of (a, b, p), so scrubbing backward is exact. p = 0 is `a`, p = 1 is `b`.
export function mixSlots(a, b, p, { from = 0.2, to = 0.95 } = {}) {
  const changed = a.map((slot, i) => (sameSlot(slot, b[i]) ? -1 : i)).filter((i) => i >= 0);
  const rank = new Map(changed.map((i, k) => [i, k]));
  return a.map((slot, i) => {
    if (!rank.has(i)) return slot;
    const at = changed.length === 1 ? to : from + ((to - from) * rank.get(i)) / (changed.length - 1);
    return p >= at ? b[i] : slot;
  });
}

export function drawPool(parent, { x, y, blocks, blockSize, cell, perRow, slots, refs = null }) {
  return G.blockPool(parent, { x, y, blocks, blockSize, slots, cell, perRow, refs });
}

// A selection mark on each of a request's blocks.
export function markBlocks(parent, geo, ox, oy, tableBlocks, opacity = 1) {
  tableBlocks.forEach((b) => {
    const at = geo.blockAt(b);
    const m = G.selectionMark(parent, { x: ox + at.x, y: oy + at.y, w: at.w, h: at.h });
    if (opacity < 1) m.setAttribute('opacity', opacity.toFixed(3));
  });
}

// The usage bar, with every percentage printed at one decimal through the glyph's own share formatter.
export function usageBar(parent, { x, y, w, useful, reserved, free, minSegment = 18 }) {
  const g = G.memBar(parent, { x, y, w, useful, reserved, free, minSegment, format: G.formatShare });
  const total = useful + reserved + free;
  g.setAttribute('aria-label', `memory: useful ${shareText(useful, total)}, reserved but empty ${shareText(reserved, total)}, free ${shareText(free, total)}`);
  return g;
}

