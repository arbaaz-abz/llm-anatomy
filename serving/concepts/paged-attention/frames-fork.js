// Frames 8–9: a "What if?" branch. D asks for two answers; both share the prompt's blocks, and the first write into the
// shared partial block copies it (copy-on-write).
import * as G from '@shared/glyphs.js';
import { AFTER_X, LANE_TITLE_Y, POOL_Y, layer, label, seg } from './stage.js';
import { mixSlots, slotsOf } from './pool.js';
import { requestBars, stepTag, afterPool, blockGeo, BLOCKS, tableView } from './scene.js';
import { forkAt, entriesOfPaged, pagedAt, blocksInUse, refCounts, BLOCK_SIZE } from './numbers.js';
import { forkBlocks } from '@math/paging.js';

const geo = { blocks: BLOCKS.blocks, blockSize: BLOCKS.blockSize };
const slotsFor = (entries) => slotsOf(entries, geo);
const TABLE_Y1 = 106;
const TABLE_Y2 = 186;
const NOTE_Y = POOL_Y + blockGeo.height + 130;
const DIM = 0.35; // the main timeline, dimmed while the branch plays
const D_PROMPT = 6;
const SUB = { 1: 'D₁', 2: 'D₂' };

function branchFrame(svg, { step, slots, refs, tables }) {
  requestBars(svg, { step, ids: ['A', 'B', 'C', 'D'], follow: 'D', opacity: DIM });
  stepTag(svg, step, DIM);
  label(svg, AFTER_X, LANE_TITLE_Y, `What if? D asks for two answers (n = 2) · blocks of ${BLOCK_SIZE}`, { cls: '' });
  const dBlocks = [...new Set(tables.flatMap((t) => t.map((r) => r.physical)))];
  afterPool(svg, slots, { refs, follow: dBlocks });
  [TABLE_Y1, TABLE_Y2].forEach((y, i) => {
    const rows = tables[i].map((r) => ({ ...r, ref: refs[r.physical] }));
    tableView(svg, { rows, title: SUB[i + 1], y, withRef: true });
  });
}

const tableOf = (physicalBlocks) => physicalBlocks.map((physical, logical) => ({ logical, physical }));

export function drawFrame8(svg, p) {
  const before = slotsFor(entriesOfPaged(pagedAt(3)));
  const fork = forkAt(1);
  const end = slotsFor(fork.entries);
  const refsNow = p >= 0.6 ? refCounts(fork.entries, BLOCKS.blocks) : refCounts(entriesOfPaged(pagedAt(3)), BLOCKS.blocks);
  branchFrame(svg, { step: 1, slots: mixSlots(before, end, p, { from: 0.15, to: 0.85 }), refs: refsNow, tables: [tableOf(fork.d1), tableOf(fork.d2)] });
  const saved = forkBlocks({ prompt: D_PROMPT, generated: 0, samples: 2, blockSize: BLOCK_SIZE });
  label(svg, 6, NOTE_Y, `${saved.physical} physical blocks instead of ${saved.withoutSharing}`, { cls: '', opacity: seg(p, 0.7, 1) });
  const cells = fork.d1.map((b) => `block ${b}: ${Math.min(BLOCK_SIZE, D_PROMPT - BLOCK_SIZE * fork.d1.indexOf(b))}/${BLOCK_SIZE}, ref ${refCounts(fork.entries, BLOCKS.blocks)[b]}`);
  label(svg, 6, NOTE_Y + 16, cells.join(' · '), { cls: '', opacity: seg(p, 0.7, 1) });
}

export function drawFrame9(svg, p) {
  const one = forkAt(1);
  const two = forkAt(2);
  const copying = p >= 0.6;
  const refs = refCounts((copying ? two : one).entries, BLOCKS.blocks);
  const slots = mixSlots(slotsFor(one.entries), slotsFor(two.entries), p, { from: 0.5, to: 0.95 });
  const d1 = copying ? two.d1 : one.d1;
  branchFrame(svg, { step: copying ? 2 : 1, slots, refs, tables: [tableOf(d1), tableOf(two.d2)] });
  const { from, to } = two.copied;
  const a = blockGeo.blockAt(from);
  const b = blockGeo.blockAt(to);
  const y = POOL_Y + a.y + a.h + 8;
  G.flow(layer(svg, seg(p, 0.1, 0.2)), { from: [AFTER_X + a.x + a.w / 2, y], to: [AFTER_X + b.x + b.w / 2, y], carry: 'kv', progress: seg(p, 0.15, 0.6) });
  const saved = forkBlocks({ prompt: D_PROMPT, generated: 1, samples: 2, blockSize: BLOCK_SIZE });
  label(svg, 6, NOTE_Y, `${saved.physical} physical blocks instead of ${saved.withoutSharing}`, { cls: '', opacity: seg(p, 0.8, 1) });
  label(svg, 6, NOTE_Y + 16, `block ${one.d1[0]} stays shared (ref ${refs[one.d1[0]]}) · ${blocksInUse(two.entries)} of ${BLOCKS.blocks} blocks in use`, { cls: '', opacity: seg(p, 0.8, 1) });
}
