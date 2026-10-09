// The toy's picture: both lanes of the same 48-slot pool, one above the other, each with its usage bar.
// Drawn from the two simulations the toy already ran (toy-view.js), so the picture and the readouts cannot disagree.
import * as G from '@shared/glyphs.js';
import { label, layer } from './stage.js';
import { drawPool, markBlocks, slotsOf, usageBar } from './pool.js';
import { poolGeometry } from './pool.js';
import { POOL_SLOTS, MAX_LEN, entriesOfContiguous, entriesOfPaged } from './numbers.js';

export const TOY_SVG = Object.freeze({ w: 580, h: 282 });
const CELL = 10;
const PER_ROW = Object.freeze({ 2: 12, 4: 6, 8: 3, 16: 3 }); // blocks per row, so the widest pool is 563 px
const X = 10;
const BEFORE = Object.freeze({ titleY: 14, pool: 22, bar: 70 });
const AFTER = Object.freeze({ titleY: 130, pool: 136, bar: 226 });
const BAR = Object.freeze({ x: 10, w: 500 });

function lane(svg, { title, titleY, pool, bar, blocks, blockSize, perRow, sim, entries, follow = [] }) {
  label(svg, X, titleY, title);
  const slots = slotsOf(entries, { blocks, blockSize });
  const geo = poolGeometry({ blocks, blockSize, cell: CELL, perRow });
  const g = layer(svg);
  drawPool(g, { x: X, y: pool, blocks, blockSize, cell: CELL, perRow, slots });
  markBlocks(g, geo, X, pool, follow);
  const link = G.svgEl('g', { 'data-link': 'waste' }, g);
  usageBar(link, { x: BAR.x, y: bar, w: BAR.w, useful: sim.useful, reserved: sim.waste, free: sim.free, minSegment: 18 });
}

export function paintLanes(svg, { sims, blockSize, follow }) {
  svg.replaceChildren();
  G.hatchFill(svg);
  const strips = POOL_SLOTS / MAX_LEN;
  lane(svg, { ...BEFORE, title: 'Before: one strip per request', blocks: strips, blockSize: MAX_LEN, perRow: strips, sim: sims.before, entries: entriesOfContiguous(sims.before) });
  const followed = sims.after.live.find((r) => r.id === follow);
  lane(svg, {
    ...AFTER, title: `After: blocks of ${blockSize}`, blocks: POOL_SLOTS / blockSize, blockSize, perRow: PER_ROW[blockSize], sim: sims.after, entries: entriesOfPaged(sims.after), follow: followed?.running ? followed.table : [],
  });
}
