// Frames 1–3: the old scheme. Three requests arrive, each reserves a strip of 16 slots, and the fourth must wait.
import * as G from '@shared/glyphs.js';
import { BEFORE_X, LANE_TITLE_Y, POOL_Y, HANDOFF, arriving, leaving, layer, label, seg } from './stage.js';
import { mixSlots, slotsOf, drawPool } from './pool.js';
import {
  requestBars, stepTag, beforePool, stripGeo, STRIP, BLOCKS, blockGeo, usage,
} from './scene.js';
import { contiguousAt, entriesOfContiguous, finishStep, ADMITTED, POOL_SLOTS, MAX_LEN, request } from './numbers.js';
import { shareText } from './format.js';

const TOP_IDS = ['A', 'B', 'C'];
const FREE_SLOTS = new Array(POOL_SLOTS).fill({ state: 'free' });
const sumPrompts = TOP_IDS.reduce((n, id) => n + request(id).prompt, 0);
const below = (geo, gap = 18) => POOL_Y + geo.height + gap;

// Frame 1's pool: 48 empty slots as 12 blocks of 4, drawn where frame 2 draws its strips.
function neutralPool(parent, opacity = 1) {
  const g = layer(parent, opacity);
  drawPool(g, { x: BEFORE_X, y: POOL_Y, ...BLOCKS, slots: FREE_SLOTS });
  return g;
}

const SLOT_NOTE = 'slot = one token\'s K and V, for every layer';

export function drawFrame1(svg, p) {
  requestBars(svg, { step: 0, ids: TOP_IDS, slide: seg(p, 0, 0.6), opacity: 0.15 + 0.85 * seg(p, 0, 0.3) });
  label(svg, BEFORE_X, LANE_TITLE_Y, 'The KV pool: 48 slots in 12 blocks of 4');
  neutralPool(svg);
  const y = below(blockGeo);
  label(svg, BEFORE_X, y, `1 ${SLOT_NOTE}`);
  label(svg, BEFORE_X, y + 16, 'sizes are hand-picked so you can count; real prompts are thousands of tokens');
  label(svg, BEFORE_X, y + 38, `prompts ${TOP_IDS.map((id) => request(id).prompt).join(' + ')} = ${sumPrompts} tokens · pool ${POOL_SLOTS} slots`, { cls: '', opacity: seg(p, 0.5, 0.9) });
}

// Strip i holds one request; `reserve` and `fill` (0 → 1) say how much of the strip is reserved and how much of the prompt is in.
function stripSlots(entries, reserve, fill) {
  const reserved = Math.ceil(reserve * MAX_LEN);
  const slots = new Array(POOL_SLOTS).fill({ state: 'free' });
  entries.forEach(({ id, table: [strip], tokens }) => {
    for (let j = 0; j < MAX_LEN; j += 1) {
      if (j >= reserved) continue;
      slots[strip * MAX_LEN + j] = { owner: id, state: j < Math.round(fill * tokens) ? 'filled' : 'reserved' };
    }
  });
  return slots;
}

const usageOf = (sim) => ({ useful: sim.useful, reserved: sim.waste, free: sim.free });

export function drawFrame2(svg, p) {
  const sim = contiguousAt(0);
  requestBars(svg, { step: 0, ids: TOP_IDS, opacity: arriving(p) });
  label(svg, BEFORE_X, LANE_TITLE_Y, 'Before: one strip per request');
  if (p < HANDOFF) neutralPool(svg, leaving(p));
  beforePool(svg, stripSlots(entriesOfContiguous(sim), seg(p, 0.1, 0.5), seg(p, 0.5, 0.9)), { opacity: arriving(p), marks: p > 0.4 });
  const y = below(stripGeo, 14);
  usage(svg, { y, ...usageOf(sim), opacity: seg(p, 0.8, 1) });
  label(svg, BEFORE_X, y + 60, 'only reserved-but-empty waste is shown; real systems also lost gaps between strips', { opacity: seg(p, 0.85, 1) });
}

// Frame 3 plays steps 1 and 2: a state is complete when its slots have all flipped, and the numbers follow the last complete state.
export function drawFrame3(svg, p) {
  const sims = [0, 1, 2].map(contiguousAt);
  const states = sims.map((sim) => slotsOf(entriesOfContiguous(sim), { blocks: STRIP.blocks, blockSize: STRIP.blockSize }));
  const stage = Math.min(Math.floor(p * 2), 1);
  const local = p * 2 - stage;
  const done = local >= 0.9 ? stage + 1 : stage;
  const slots = mixSlots(states[stage], states[stage + 1], local, { from: 0.15, to: 0.9 });
  requestBars(svg, { step: done, ids: TOP_IDS });
  stepTag(svg, done);
  label(svg, BEFORE_X, LANE_TITLE_Y, 'Before: one strip per request');
  beforePool(svg, slots);
  const y = below(stripGeo, 14);
  usage(svg, { y, ...usageOf(sims[done]) });
  const dStart = ADMITTED.before.D;
  if (done >= 1) {
    const pulse = 0.7 + 0.3 * Math.cos(local * Math.PI * 2);
    G.token(layer(svg, pulse), { x: BEFORE_X, y: y + 44, text: 'D', owner: 'D' });
    label(svg, BEFORE_X + 36, y + 56, `waiting until step ${dStart}`, { cls: '' });
  }
  if (done >= 2) {
    label(svg, BEFORE_X, y + 84, `D starts at step ${dStart} (in B's old strip) and finishes at step ${finishStep('before', 'D')}`);
  }
  label(svg, BEFORE_X, y + 104, `step ${done}: the pool is ${shareText(POOL_SLOTS, POOL_SLOTS)} reserved and ${shareText(sims[done].waste, POOL_SLOTS)} of it is empty`, { opacity: done >= 2 ? 1 : 0 });
}

