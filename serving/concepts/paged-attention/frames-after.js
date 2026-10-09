// Frames 4–7: paging. Blocks are handed out on demand, a block table says where they are, a finished request gives its blocks back.
import * as G from '@shared/glyphs.js';
import { AFTER_X, LANE_TITLE_Y, POOL_Y, TABLE_X, HANDOFF, arriving, leaving, layer, label, textBlock, seg, lerp } from './stage.js';
import { mixSlots, slotsOf } from './pool.js';
import {
  requestBars, stepTag, afterPool, blockGeo, BLOCKS, tableView, pointer, usage, barEnd,
} from './scene.js';
import {
  pagedAt, entriesOfPaged, addressOf, finishStep, ADMITTED, POOL_SLOTS, BLOCK_SIZE,
} from './numbers.js';
import { drawFrame3 } from './frames-before.js';

const geo = { blocks: BLOCKS.blocks, blockSize: BLOCKS.blockSize };
const slotsFor = (entries) => slotsOf(entries, geo);
const FREE = slotsFor([]);
const USAGE_Y = POOL_Y + blockGeo.height + 14;
const NOTE_Y = USAGE_Y + 60;
const usageOf = (sim) => ({ useful: sim.useful, reserved: sim.waste, free: sim.free });
const tableRows = (table) => table.map((physical, logical) => ({ logical, physical }));
const A_TOKEN = 9;

// A token chip travelling from a request's bar to the slot it is written into, then gone.
function travellingChip(parent, { id, index, from, to, p, from0 = 0, to0 = 0.5 }) {
  const t = seg(p, from0, to0);
  if (t <= 0 || p > to0 + 0.1) return;
  const g = layer(parent, 1 - seg(p, to0, to0 + 0.1));
  const w = G.tokenWidth(id);
  G.token(g, { x: lerp(from[0], to[0] - w / 2, t), y: lerp(from[1], to[1] - 28, t), text: id, owner: id, index });
}

const slotCentre = (block, slot) => {
  const at = blockGeo.slotAt(block, slot);
  return [AFTER_X + at.x + at.size / 2, POOL_Y + at.y + at.size / 2];
};
const barTip = (id, step) => [34 + barEnd(id, step) + 4, 8 + ['A', 'B', 'C', 'D'].indexOf(id) * 14 + 5];

const legend = (svg, opacity = 1) => textBlock(svg, TABLE_X, POOL_Y + 14, ['1 slot = one token\'s', 'K and V, for every', 'layer'], { opacity });

export function drawFrame4(svg, p) {
  const sim = pagedAt(0);
  if (p < HANDOFF) drawFrame3(layer(svg, leaving(p)), 1);
  const g = layer(svg, arriving(p));
  requestBars(g, { step: 0, ids: ['A', 'B', 'C'] });
  label(g, AFTER_X, LANE_TITLE_Y, `After: blocks of ${BLOCK_SIZE}`);
  legend(g);
  afterPool(g, mixSlots(FREE, slotsFor(entriesOfPaged(sim)), p, { from: 0.15, to: 0.85 }));
  usage(g, { y: USAGE_Y, ...usageOf(sim), opacity: seg(p, 0.75, 1) });
  label(g, TABLE_X, NOTE_Y, `${sim.blocksUsed} blocks = ${sim.blocksUsed * BLOCK_SIZE} slots`, { cls: '', opacity: seg(p, 0.75, 1) });
}

// Frame 5 holds step 0 and adds only A's ninth token: block 7 opens.
function frame5Entries() {
  const [s0, s1] = [pagedAt(0), pagedAt(1)];
  const a1 = s1.live.find((r) => r.id === 'A');
  return entriesOfPaged(s0).map((e) => (e.id === 'A' ? { id: 'A', table: a1.table, tokens: a1.tokens } : e));
}

export function drawFrame5(svg, p) {
  const sim0 = pagedAt(0);
  const end = frame5Entries();
  const a = end.find((e) => e.id === 'A');
  const landed = p >= 0.55;
  const table = landed ? a.table : a.table.slice(0, 2);
  const at = addressOf(a.table, A_TOKEN, BLOCK_SIZE);
  requestBars(svg, { step: 0, ids: ['A', 'B', 'C'], follow: 'A', override: { A: landed ? 1 : 0 } });
  label(svg, AFTER_X, LANE_TITLE_Y, `After: blocks of ${BLOCK_SIZE}`);
  afterPool(svg, mixSlots(slotsFor(entriesOfPaged(sim0)), slotsFor(end), p, { from: 0.55, to: 0.6 }), { follow: table });
  tableView(svg, { rows: tableRows(table), title: 'block table of A' });
  table.forEach((physical, row) => pointer(svg, { row, block: physical, progress: row < 2 ? 1 : seg(p, 0.6, 0.95) }));
  travellingChip(svg, { id: 'A', index: A_TOKEN, from: barTip('A', 0), to: slotCentre(at.block, at.slot), p, to0: 0.5 });
  textBlock(svg, TABLE_X, POOL_Y + blockGeo.height + 22, ['tokens count from 1; blocks and slots count from 0,', 'like memory addresses'], { opacity: 1 });
  label(svg, TABLE_X, POOL_Y + blockGeo.height + 62, `A: ${a.tokens} tokens → ${a.table.length} blocks · table ${a.table.map((b, i) => `${i}→${b}`).join(', ')} · token ${A_TOKEN} → block ${at.block}, slot ${at.slot}`, { cls: '', opacity: seg(p, 0.7, 1) });
}

export function drawFrame6(svg, p) {
  const s0 = frame5Entries();
  const sim1 = pagedAt(1);
  const end = entriesOfPaged(sim1);
  requestBars(svg, { step: 1, ids: ['A', 'B', 'C', 'D'], override: p < 0.55 ? { B: 0, C: 0 } : null });
  stepTag(svg, 1);
  label(svg, AFTER_X, LANE_TITLE_Y, `After: blocks of ${BLOCK_SIZE}`);
  legend(svg, 0);
  afterPool(svg, mixSlots(slotsFor(s0), slotsFor(end), p, { from: 0.55, to: 0.95 }));
  ['B', 'C'].forEach((id) => {
    const r = sim1.live.find((q) => q.id === id);
    const at = addressOf(r.table, r.tokens, BLOCK_SIZE);
    travellingChip(svg, { id, index: r.tokens, from: barTip(id, 0), to: slotCentre(at.block, at.slot), p, to0: 0.5 });
  });
  usage(svg, { y: USAGE_Y, ...usageOf(sim1), opacity: seg(p, 0.85, 1) });
  label(svg, TABLE_X, NOTE_Y, `${sim1.blocksUsed} of ${sim1.poolBlocks} blocks · D starts at step ${ADMITTED.after.D}`, { cls: '', opacity: seg(p, 0.85, 1) });
}

// Frame 7 plays steps 2 and 3. B finishes at the end of step 2 and its blocks go back; C then takes the lowest free block, 2.
export function drawFrame7(svg, p) {
  const sims = [1, 2, 3].map((s) => pagedAt(s));
  const states = sims.map((sim) => slotsFor(entriesOfPaged(sim)));
  const stage = Math.min(Math.floor(p * 2), 1);
  const local = p * 2 - stage;
  const done = local >= 0.9 ? stage + 1 : stage;
  const sim = sims[done];
  const step = done + 1;
  const c = sim.live.find((r) => r.id === 'C');
  requestBars(svg, { step, ids: ['A', 'B', 'C', 'D'], follow: 'C' });
  stepTag(svg, step);
  label(svg, AFTER_X, LANE_TITLE_Y, `After: blocks of ${BLOCK_SIZE}`);
  afterPool(svg, mixSlots(states[stage], states[stage + 1], local, { from: 0.15, to: 0.9 }), { follow: c.table });
  tableView(svg, { rows: tableRows(c.table), title: 'block table of C' });
  usage(svg, { y: USAGE_Y, ...usageOf(sim) });
  label(svg, TABLE_X, NOTE_Y, `${sim.blocksUsed} of ${sim.poolBlocks} blocks`, { cls: '' });
  const d = sim.live.find((r) => r.id === 'D');
  label(svg, TABLE_X, NOTE_Y + 16, `D started at step ${d.admitted} and finishes at step ${finishStep('after', 'D')} (before: step ${finishStep('before', 'D')})`, { cls: '', opacity: step >= 3 ? 1 : 0 });
}
