// paged-attention's fixed scenario (storyboard §5): the four requests come from math/serving.js, the pool and the
// 16-token limit are hand-picked stand-ins so every number can be counted. Pure, no DOM.
import { simulateContiguous, simulatePaged } from '@math/paging.js';
import { TOY_REQUESTS } from '@math/serving.js';

export { TOY_REQUESTS };
export const POOL_SLOTS = 48;
export const MAX_LEN = 16; // the toy model's context limit: what the old scheme reserved per request
export const BLOCK_SIZE = 4; // the stage's block size; the toy slides it
export const LAST_STEP = 6;
export const SYSTEM_PROMPT = 4; // the toy's shared system prompt, in tokens
export const IDS = Object.freeze(TOY_REQUESTS.map((r) => r.id));
export const request = (id) => TOY_REQUESTS.find((r) => r.id === id);

export const contiguousAt = (step) => simulateContiguous({ requests: TOY_REQUESTS, poolSlots: POOL_SLOTS, maxLen: MAX_LEN, step });
export const pagedAt = (step, blockSize = BLOCK_SIZE, sharedPrefix = 0) => simulatePaged({ requests: TOY_REQUESTS, poolSlots: POOL_SLOTS, blockSize, step, sharedPrefix });

// The step each request starts at, in each scheme (the last step has seen every admission).
export const ADMITTED = Object.freeze({
  before: Object.freeze(Object.fromEntries(contiguousAt(LAST_STEP).live.map((r) => [r.id, r.admitted]))),
  after: Object.freeze(Object.fromEntries(pagedAt(LAST_STEP).live.map((r) => [r.id, r.admitted]))),
});
export const finishStep = (scheme, id) => ADMITTED[scheme][id] + request(id).output;

// What each simulation says a pool holds: [{ id, table, tokens }], running requests only (the drawing's input).
export const entriesOfPaged = (sim) => sim.live.filter((r) => r.running).map((r) => ({ id: r.id, table: r.table, tokens: r.tokens }));
export const entriesOfContiguous = (sim) => sim.live.filter((r) => r.running).map((r) => ({ id: r.id, table: [r.strip], tokens: r.tokens }));

// Frames 8 and 9: D asks for two answers (n = 2). Pure: built from the main timeline's step 1 or 2.
// Step 1: both tables hold the prompt's blocks. Step 2: D1 writes into the shared partial block, so it is copied to the
// lowest free block first; D2 then writes into the original in place.
export function forkAt(step) {
  if (step !== 1 && step !== 2) throw new RangeError(`forkAt: step must be 1 or 2, got ${step}`);
  const sim = pagedAt(step);
  const others = entriesOfPaged(sim).filter((e) => e.id !== 'D');
  const d = sim.live.find((r) => r.id === 'D');
  const used = new Set(sim.live.flatMap((r) => r.table));
  const copyTo = [...Array(sim.poolBlocks).keys()].find((b) => !used.has(b));
  const shared = d.table;
  const d1 = step === 1 ? shared : [...shared.slice(0, -1), copyTo];
  const entries = [...others, { id: 'D', table: d1, tokens: d.tokens, sample: 1 }, { id: 'D', table: shared, tokens: d.tokens, sample: 2 }];
  return { sim, entries, d1, d2: shared, copied: step === 2 ? { from: shared.at(-1), to: copyTo } : null };
}

export const blocksInUse = (entries) => new Set(entries.flatMap((e) => e.table)).size;
export const refCounts = (entries, poolBlocks) => {
  const refs = new Array(poolBlocks).fill(0);
  entries.forEach((e) => e.table.forEach((b) => { refs[b] += 1; }));
  return refs;
};

// Where token t (counting from 1) sits: block = table[⌊(t − 1) / B⌋], slot = (t − 1) mod B (§7; blocks and slots count from 0).
export function addressOf(table, token, blockSize) {
  return { block: table[Math.floor((token - 1) / blockSize)], slot: (token - 1) % blockSize };
}
