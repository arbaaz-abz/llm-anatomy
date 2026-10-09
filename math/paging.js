// PagedAttention block allocator (paged-attention §5–§6). Pure: no DOM, inputs never mutated, every simulator
// recomputes the whole state from step 0 for the requested step, so scrubbing backward is exact.
// TOY_REQUESTS lives in math/serving.js; kvCacheBytes, kvBytesPerToken(Mla) and sharePct live in math/memory.js.
import { kvCacheBytes } from './memory.js';

const isInt = (n, min) => Number.isInteger(n) && n >= min;
const need = (fn, name, ok, rule) => { if (!ok) throw new RangeError(`${fn}: ${name} must be ${rule}`); };

// A request holds memory from the step it is admitted through the step it finishes (admitted + output), and is free the step after.
const tokensAt = (r, admitted, step) => r.prompt + (step - admitted);
const finishesAfter = (r, admitted) => admitted + r.output;
const lowest = (taken, limit) => { for (let i = 0; i < limit; i += 1) if (!taken.has(i)) return i; return -1; };

function checkRequests(fn, requests) {
  need(fn, 'requests', Array.isArray(requests) && requests.length > 0, 'a non-empty array');
  requests.forEach((r) => {
    need(fn, `request ${r?.id} arrives`, isInt(r?.arrives, 0), 'an integer ≥ 0');
    need(fn, `request ${r?.id} prompt`, isInt(r?.prompt, 1), 'an integer ≥ 1');
    need(fn, `request ${r?.id} output`, isInt(r?.output, 0), 'an integer ≥ 0');
  });
}

export function blocksNeeded(tokens, blockSize) {
  need('blocksNeeded', 'tokens', isInt(tokens, 0), 'an integer ≥ 0');
  need('blocksNeeded', 'blockSize', isInt(blockSize, 1), 'an integer ≥ 1');
  return Math.ceil(tokens / blockSize);
}

export function internalWaste(tokens, blockSize) {
  return blocksNeeded(tokens, blockSize) * blockSize - tokens;
}

// ---- one strip per request ----

export function simulateContiguous({ requests, poolSlots, maxLen, step }) {
  checkRequests('simulateContiguous', requests);
  need('simulateContiguous', 'poolSlots', isInt(poolSlots, 1), 'an integer ≥ 1');
  need('simulateContiguous', 'maxLen', isInt(maxLen, 1) && maxLen <= poolSlots && requests.every((r) => r.prompt + r.output <= maxLen), 'an integer between every request\'s final length and poolSlots');
  need('simulateContiguous', 'step', isInt(step, 0), 'an integer ≥ 0');
  const strips = Math.floor(poolSlots / maxLen);
  const admitted = new Map();
  const stripOf = new Map();
  for (let t = 0; t <= step; t += 1) {
    const taken = new Set([...admitted].filter(([id, a]) => finishesAfter(requests.find((r) => r.id === id), a) >= t).map(([id]) => stripOf.get(id)));
    for (const r of requests) {
      if (admitted.has(r.id) || r.arrives > t) continue;
      const strip = lowest(taken, strips);
      if (strip < 0) break; // first come, first served: a request that does not fit holds the queue
      admitted.set(r.id, t);
      stripOf.set(r.id, strip);
      taken.add(strip);
    }
  }
  const live = requests.map((r) => {
    const a = admitted.get(r.id) ?? null;
    const running = a !== null && step <= finishesAfter(r, a);
    return {
      id: r.id, admitted: a, running, waiting: a === null && r.arrives <= step, done: a !== null && !running,
      tokens: running ? tokensAt(r, a, step) : 0, reserved: running ? maxLen : 0, strip: running ? stripOf.get(r.id) : null,
    };
  });
  const useful = live.reduce((n, r) => n + r.tokens, 0);
  const reserved = live.reduce((n, r) => n + r.reserved, 0);
  return { live, useful, waste: reserved - useful, free: poolSlots - reserved, poolSlots };
}

// ---- blocks on demand ----

function newPool(poolBlocks) {
  const refs = new Array(poolBlocks).fill(0);
  const take = (shared = null) => {
    const b = shared ?? refs.findIndex((n) => n === 0);
    if (b < 0) return -1;
    refs[b] += 1;
    return b;
  };
  return { refs, take, free: () => refs.filter((n) => n === 0).length, release: (b) => { refs[b] -= 1; } };
}

// Reaches `step` by replaying every step from 0: free the finished, extend the running (in admission order), then admit
// the arrivals (in arrival order), always handing out the lowest free block.
function runPaged({ requests, poolBlocks, blockSize, step, sharedPrefix }) {
  const pool = newPool(poolBlocks);
  const sharedBlocks = Math.floor(sharedPrefix / blockSize);
  const shared = []; // physical ids of the shared prefix blocks
  const tables = new Map();
  const admitted = new Map();
  const order = [];
  for (let t = 0; t <= step; t += 1) {
    order.filter((id) => !tables.get(id).released && finishesAfter(requests.find((r) => r.id === id), admitted.get(id)) < t).forEach((id) => {
      const entry = tables.get(id);
      entry.table.forEach((b) => pool.release(b));
      entry.released = true;
    });
    order.filter((id) => !tables.get(id).released).forEach((id) => {
      const r = requests.find((q) => q.id === id);
      const entry = tables.get(id);
      while (entry.table.length < blocksNeeded(tokensAt(r, admitted.get(id), t), blockSize)) {
        const b = pool.take();
        if (b < 0) throw new Error(`pool exhausted at step ${t}`);
        entry.table.push(b);
      }
    });
    for (const r of requests) {
      if (admitted.has(r.id) || r.arrives > t) continue;
      const own = Math.min(sharedBlocks, Math.floor(r.prompt / blockSize));
      const reuse = Array.from({ length: own }, (_, j) => (shared[j] !== undefined && pool.refs[shared[j]] > 0 ? shared[j] : null));
      const fresh = blocksNeeded(r.prompt, blockSize) - reuse.filter((b) => b !== null).length;
      if (fresh > pool.free()) break; // first come, first served
      const table = reuse.map((b, j) => {
        const taken = pool.take(b);
        if (b === null) shared[j] = taken;
        return taken;
      });
      while (table.length < blocksNeeded(r.prompt, blockSize)) table.push(pool.take());
      admitted.set(r.id, t);
      tables.set(r.id, { table, released: false });
      order.push(r.id);
    }
  }
  return { admitted, tables, order };
}

export function simulatePaged({ requests, poolSlots, blockSize, step, sharedPrefix = 0 }) {
  checkRequests('simulatePaged', requests);
  need('simulatePaged', 'blockSize', isInt(blockSize, 1), 'an integer ≥ 1');
  need('simulatePaged', 'poolSlots', isInt(poolSlots, 1) && poolSlots % blockSize === 0, 'a multiple of blockSize');
  need('simulatePaged', 'step', isInt(step, 0), 'an integer ≥ 0');
  need('simulatePaged', 'sharedPrefix', isInt(sharedPrefix, 0), 'an integer ≥ 0');
  const poolBlocks = poolSlots / blockSize;
  const { admitted, tables } = runPaged({ requests, poolBlocks, blockSize, step, sharedPrefix });
  const live = requests.map((r) => {
    const a = admitted.get(r.id) ?? null;
    const entry = a === null ? null : tables.get(r.id);
    const running = entry !== null && !entry.released;
    const tokens = running ? tokensAt(r, a, step) : 0;
    const table = running ? [...entry.table] : [];
    return {
      id: r.id, admitted: a, running, waiting: a === null && r.arrives <= step, done: a !== null && !running,
      tokens, blocks: table.length, waste: running ? table.length * blockSize - tokens : 0, table,
    };
  });
  const held = new Map(); // physical block → slots holding a token (a shared block counts once)
  live.filter((r) => r.running).forEach((r) => r.table.forEach((b, i) => held.set(b, Math.max(held.get(b) ?? 0, Math.min(blockSize, r.tokens - i * blockSize)))));
  const useful = [...held.values()].reduce((a, n) => a + n, 0);
  const blocksUsed = held.size;
  const logicalBlocks = live.reduce((n, r) => n + r.blocks, 0);
  return {
    live, useful, logicalTokens: live.reduce((n, r) => n + r.tokens, 0), waste: blocksUsed * blockSize - useful, free: poolSlots - blocksUsed * blockSize,
    blocksUsed, poolBlocks, blocksSaved: logicalBlocks - blocksUsed,
  };
}

// ---- sharing ----

export function sharedPrefixBlocksSaved({ running, prefixLen, blockSize }) {
  need('sharedPrefixBlocksSaved', 'running', isInt(running, 0), 'an integer ≥ 0');
  need('sharedPrefixBlocksSaved', 'prefixLen', isInt(prefixLen, 0), 'an integer ≥ 0');
  need('sharedPrefixBlocksSaved', 'blockSize', isInt(blockSize, 1), 'an integer ≥ 1');
  return Math.max(running - 1, 0) * Math.floor(prefixLen / blockSize);
}

// Parallel sampling: `samples` continuations of one prompt. Full prompt blocks are shared; the partial block is copied the
// first time a sample writes into it (all but the last writer copy; the last one finds ref 1 and writes in place).
export function forkBlocks({ prompt, generated, samples, blockSize }) {
  need('forkBlocks', 'prompt', isInt(prompt, 1), 'an integer ≥ 1');
  need('forkBlocks', 'generated', isInt(generated, 0), 'an integer ≥ 0');
  need('forkBlocks', 'samples', isInt(samples, 1), 'an integer ≥ 1');
  need('forkBlocks', 'blockSize', isInt(blockSize, 1), 'an integer ≥ 1');
  const withoutSharing = samples * blocksNeeded(prompt + generated, blockSize);
  const full = Math.floor(prompt / blockSize);
  const physical = generated === 0 ? blocksNeeded(prompt, blockSize) : full + samples * (blocksNeeded(prompt + generated, blockSize) - full);
  const copies = generated >= 1 && prompt % blockSize !== 0 ? samples - 1 : 0;
  return { physical, withoutSharing, saved: withoutSharing - physical, copies };
}

// ---- bytes ----

export function kvBytesPerBlock(bytesPerToken, blockSize) {
  need('kvBytesPerBlock', 'bytesPerToken', Number.isFinite(bytesPerToken) && bytesPerToken > 0, 'a finite number > 0');
  need('kvBytesPerBlock', 'blockSize', isInt(blockSize, 1), 'an integer ≥ 1');
  return bytesPerToken * blockSize;
}

// The old scheme's reservation for one request: the one definition of "cache for N tokens" (math/memory.js).
export function reserveMaxBytes(bytesPerToken, maxLen) {
  return kvCacheBytes({ bytesPerToken, tokens: maxLen });
}
