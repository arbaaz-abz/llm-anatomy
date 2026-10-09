// prefix-caching scenes (pure, no DOM): the tree nodes and pool slots the stage and the toy draw, derived from math/prefix.js.
// The animation runs blocks of 4 in a pool of 8; the toy passes its own block size, pool and requests. Nothing here is typed in:
// every id, label, state and slot comes from blockKeys / blockTexts / simulatePrefixCache.
import { PREFIX_REQUESTS, DEFAULT_BLOCK_SIZE, blockKeys, blockTexts, simulatePrefixCache } from '@math/prefix.js';

export const ROOT_ID = 'root';
export const POOL_BLOCKS = 8;
export const SIZE = DEFAULT_BLOCK_SIZE;
export const SIM = simulatePrefixCache({ requests: PREFIX_REQUESTS, blockSize: SIZE, poolBlocks: POOL_BLOCKS });
export const LOG = SIM.log;
export const [IDX_A, IDX_B, IDX_C, IDX_D] = [0, 1, 2, 3];

// "You are a cat" prints as "You…cat": the first and the last word keep every block of the page distinct at 64 px.
export function shortLabel(text) {
  const words = text.split(' ');
  return words.length === 1 ? text : `${words[0]}…${words.at(-1)}`;
}

const tokensOf = (r) => [...r.prompt, ...r.output];

// Every distinct block of the given requests, in the order they first appear, as tree nodes (the root is added by the caller).
export function treeModel(requests, blockSize) {
  const nodes = new Map();
  requests.forEach((r, index) => {
    const keys = blockKeys(tokensOf(r), blockSize);
    const texts = blockTexts(tokensOf(r), blockSize);
    keys.forEach((key, depth) => {
      if (!nodes.has(key)) nodes.set(key, { id: key, parent: depth === 0 ? ROOT_ID : keys[depth - 1], text: texts[depth], creator: r.id, creatorIndex: index });
    });
  });
  return [...nodes.values()];
}

// Per request: the node ids its prompt hit, the ids it computed (new), and its whole path, from the log.
export function requestPaths(requests, blockSize, log) {
  return requests.map((r, i) => {
    const path = blockKeys(tokensOf(r), blockSize);
    const hit = path.slice(0, log[i].hitBlocks);
    return { id: r.id, path, hit, fresh: path.slice(log[i].hitBlocks) };
  });
}

export const DEFAULT_MODEL = treeModel(PREFIX_REQUESTS, SIZE);
export const DEFAULT_PATHS = requestPaths(PREFIX_REQUESTS, SIZE, LOG);
export const pathOf = (index) => DEFAULT_PATHS[index];

// The prefixTree nodes at one moment. `visible(node)` hides blocks not yet drawn; `state(node)` names each state;
// `followed` marks the request whose path wears the selection. The root is the page's "start" node (muted, like a cached block).
export function treeNodes(model, { visible = () => true, state, followed = new Set() }) {
  const shown = model.filter(visible);
  return [
    { id: ROOT_ID, parent: null, label: 'start', state: 'cached', followed: false },
    ...shown.map((n) => ({
      id: n.id, parent: n.parent, label: shortLabel(n.text), state: state(n), owner: n.creator, followed: followed.has(n.id),
    })),
  ];
}

// The state a node has while request `active` runs: its first `hitCount` hits are `hit`, its fresh blocks `new`; with none
// running every node is `cached`. A node in `evicted` is hatched whatever else it is.
export function nodeState({ paths, active, hitCount = Infinity, evicted = new Set() }) {
  const mine = active === null ? null : paths[active];
  return (node) => {
    if (evicted.has(node.id)) return 'evicted';
    if (mine && mine.hit.indexOf(node.id) >= 0 && mine.hit.indexOf(node.id) < hitCount) return 'hit';
    if (mine?.fresh.includes(node.id)) return 'new';
    return 'cached';
  };
}

// The nodes whose label a row's pops evicted, in pop order: the model's blocks (made before that request) the pool no longer holds.
export function evictedIds(model, row, requestIndex) {
  const held = new Set(row.cachedKeys);
  const gone = model.filter((n) => n.creatorIndex < requestIndex && !held.has(n.id));
  return row.evicted.map((text) => gone.find((n) => n.text === text)?.id).filter((id) => id !== undefined);
}

const EMPTY_SLOT = Object.freeze({ state: 'free' });
const slotsOf = (blockSize, slot) => Array.from({ length: blockSize }, () => slot);

// One entry per physical block: { state: 'cached' | 'free', owner } as the pool stands after `row` finished (null = all free).
export function poolAfter(row, nBlocks) {
  const owners = new Map((row?.cached ?? []).map((b, i) => [b, row.owners[i]]));
  return Array.from({ length: nBlocks }, (_, b) => (owners.has(b) ? { state: 'cached', owner: owners.get(b) } : EMPTY_SLOT));
}

// The pool while `row`'s request runs: the new blocks are filled by it, the hit blocks stay cached under their first owner.
export function poolDuring(prevRow, row, nBlocks) {
  const base = poolAfter(prevRow, nBlocks);
  const fresh = row.blocks.slice(row.hitBlocks);
  return base.map((b, i) => (fresh.includes(i) ? { state: 'filled', owner: row.id } : b));
}

// Blocks → the slot array blockPool takes (every slot of a block shares the block's state).
export function slotsFor(blocks, blockSize) {
  return blocks.flatMap((b) => slotsOf(blockSize, b));
}

// The free queue as the page prints it: block numbers, head first.
export const queueText = (queue) => queue.join(' ');

// The request's own block numbers as the table strip lists them, with whether each was a hit.
export function tableChips(row) {
  return row.blocks.map((block, i) => ({ block, kind: i < row.hitBlocks ? 'hit' : 'new', owner: row.id }));
}

// The queue while a request runs: its hit blocks touched (removed) and its new blocks popped.
export function liveQueue(prevRow, row, nBlocks) {
  const before = prevRow ? prevRow.freeQueue : Array.from({ length: nBlocks }, (_, i) => i);
  const gone = new Set(row.blocks);
  return before.filter((b) => !gone.has(b));
}

export const ANIM_COUNTERS = LOG.map((row) => Object.freeze({
  prompt: row.promptTokens, cache: row.hitTokens, computed: row.computed, rate: (100 * row.hitTokens) / row.promptTokens,
}));
