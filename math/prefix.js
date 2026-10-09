// Prefix caching (prefix-caching §6): block keys, prefix matching, a least-recently-used pool simulator, cache-aware routing and the
// blended input price. Pure and deterministic (no DOM, no clock): the same input gives a deep-equal output, so a scrubbed frame equals
// the same frame reached forward. Inputs are never mutated. Block ids count from 0; token positions on screen count from 1.
import { deepFreeze } from './core.js';

export const DEFAULT_BLOCK_SIZE = 4; // the page's blocks (vLLM uses 16); the pool presets are counted in blocks of this size
const TOKEN_JOIN = ' ';
const KEY_JOIN = '\u001f'; // between a parent's key and the block's own tokens

const words = (text) => deepFreeze(text.split(' '));
const request = (id, prompt, output) => ({ id, prompt: words(prompt), output: words(output) });
const YOU_ARE_A_CAT = 'You are a cat . Reply in rhyme';
const A_PROMPT = `${YOU_ARE_A_CAT} Where did you sit`;

// Four hand-picked requests, each finished before the next arrives. C is A's second turn; D repeats A's question after another start.
export const PREFIX_REQUESTS = deepFreeze([
  request('A', A_PROMPT, 'The cat sat down'),
  request('B', `${YOU_ARE_A_CAT} Do you like fish ?`, 'Yes , fish'),
  request('C', `${A_PROMPT} The cat sat down Why down there ?`, 'It was warm .'),
  request('D', 'You are a dog . Reply in prose Where did you sit', 'On the mat .'),
]);

const isPositiveInt = (x) => Number.isInteger(x) && x >= 1;

function requireBlockSize(fn, blockSize) {
  if (!isPositiveInt(blockSize)) throw new RangeError(`${fn}: blockSize must be a positive integer, got ${blockSize}`);
}

function requireTokens(fn, tokens) {
  if (!Array.isArray(tokens)) throw new RangeError(`${fn}: tokens must be an array of strings`);
}

function fullBlocks(tokens, blockSize) {
  return Array.from({ length: Math.floor(tokens.length / blockSize) }, (_, i) => tokens.slice(i * blockSize, (i + 1) * blockSize));
}

// One key per FULL block: the parent block's key plus this block's tokens, so the same words after another start are another key.
export function blockKeys(tokens, blockSize) {
  requireTokens('blockKeys', tokens);
  requireBlockSize('blockKeys', blockSize);
  const keys = [];
  fullBlocks(tokens, blockSize).forEach((block, i) => keys.push(`${i === 0 ? '' : keys[i - 1]}${KEY_JOIN}${block.join(TOKEN_JOIN)}`));
  return keys;
}

// The block's own tokens as text, one per full block ("You are a cat"): what the tree and the eviction list print.
export function blockTexts(tokens, blockSize) {
  requireTokens('blockTexts', tokens);
  requireBlockSize('blockTexts', blockSize);
  return fullBlocks(tokens, blockSize).map((block) => block.join(TOKEN_JOIN));
}

// Walk the prompt's blocks from the first and stop at the first miss. cachedKeys is a Set or an array of keys.
export function matchPrefix(cachedKeys, tokens, blockSize) {
  const held = cachedKeys instanceof Set ? cachedKeys : new Set(cachedKeys);
  const keys = blockKeys(tokens, blockSize);
  let hitBlocks = 0;
  while (hitBlocks < keys.length && held.has(keys[hitBlocks])) hitBlocks += 1;
  return { hitBlocks, hitTokens: hitBlocks * blockSize };
}

// The pool presets are counted in blocks of DEFAULT_BLOCK_SIZE; at another block size the pool keeps the same token slots
// (rounded up to a whole block), so the smallest preset still holds the longest request at every block size.
export function poolBlocksFor(presetBlocks, blockSize) {
  requireBlockSize('poolBlocksFor', blockSize);
  if (!isPositiveInt(presetBlocks)) throw new RangeError(`poolBlocksFor: presetBlocks must be a positive integer, got ${presetBlocks}`);
  return Math.ceil((presetBlocks * DEFAULT_BLOCK_SIZE) / blockSize);
}

function requireRequests(requests) {
  if (!Array.isArray(requests) || requests.length === 0) throw new RangeError('simulatePrefixCache: requests must be a non-empty array');
  requests.forEach((r, i) => {
    const ok = r && typeof r.id === 'string' && Array.isArray(r.prompt) && r.prompt.length > 0 && Array.isArray(r.output);
    if (!ok) throw new RangeError(`simulatePrefixCache: request ${i} needs an id, a non-empty prompt and an output`);
  });
}

function requirePool(poolBlocks) {
  if (poolBlocks !== Infinity && !isPositiveInt(poolBlocks)) throw new RangeError(`simulatePrefixCache: poolBlocks must be a positive integer or Infinity, got ${poolBlocks}`);
}

// A pool of labeled blocks. Plain mutable state kept inside one simulation; nothing escapes except copies in the log.
function makePool(poolBlocks) {
  return {
    finite: poolBlocks !== Infinity,
    queue: poolBlocks === Infinity ? [] : Array.from({ length: poolBlocks }, (_, i) => i), // free queue, head first
    fresh: 0, // the next never-used block of an unbounded pool
    label: new Map(), // block -> { key, text }
    byKey: new Map(), // key -> block
    owner: new Map(), // block -> the request letter that computed it
  };
}

function takeBlock(pool) {
  const block = pool.finite ? pool.queue.shift() : pool.fresh++;
  const old = pool.label.get(block);
  if (old) {
    pool.label.delete(block);
    pool.byKey.delete(old.key);
  }
  return { block, evictedText: old?.text ?? null };
}

function labelBlock(pool, block, key, text) {
  const twin = pool.byKey.get(key);
  if (twin !== undefined && twin !== block) pool.label.delete(twin); // a stale duplicate of a re-computed block
  pool.label.set(block, { key, text });
  pool.byKey.set(key, block);
}

function runRequest(pool, r, blockSize) {
  const all = [...r.prompt, ...r.output];
  const promptKeys = blockKeys(r.prompt, blockSize);
  const keys = blockKeys(all, blockSize);
  const texts = blockTexts(all, blockSize);
  const { hitBlocks, hitTokens } = matchPrefix(new Set(pool.byKey.keys()), r.prompt, blockSize);
  const hitIds = promptKeys.slice(0, hitBlocks).map((k) => pool.byKey.get(k));
  const touched = new Set(hitIds);
  pool.queue = pool.queue.filter((b) => !touched.has(b)); // a hit "touches" its blocks: they leave the free queue
  const need = Math.ceil(all.length / blockSize) - hitBlocks;
  if (pool.finite && need > pool.queue.length) throw new Error(`pool exhausted at ${r.id}`);
  const taken = Array.from({ length: need }, () => takeBlock(pool));
  const blocks = [...hitIds, ...taken.map((t) => t.block)];
  taken.forEach((t) => pool.owner.set(t.block, r.id));
  keys.forEach((key, j) => { if (j >= hitBlocks) labelBlock(pool, blocks[j], key, texts[j]); });
  pool.queue = [...pool.queue, ...[...blocks].reverse()]; // finished: the blocks join the queue's tail in reverse order
  return { r, hitBlocks, hitTokens, blocks, taken };
}

function logRow(pool, { r, hitBlocks, hitTokens, blocks, taken }) {
  const ids = [...pool.label.keys()].sort((a, b) => a - b);
  return {
    id: r.id,
    promptTokens: r.prompt.length,
    hitTokens,
    hitBlocks,
    computed: r.prompt.length - hitTokens,
    blocks,
    evicted: taken.filter((t) => t.evictedText !== null).map((t) => t.evictedText),
    evictedBlocks: taken.filter((t) => t.evictedText !== null).map((t) => t.block),
    cached: ids,
    cachedKeys: ids.map((b) => pool.label.get(b).key),
    cachedTexts: ids.map((b) => pool.label.get(b).text),
    owners: ids.map((b) => pool.owner.get(b)),
    freeQueue: [...pool.queue],
  };
}

// Requests run one after another. A hit touches its blocks; new blocks pop the queue's head (evicting its label if it had one);
// a block is labeled once full (prompt or answer tokens). Throws Error('pool exhausted at <id>') when a request needs more
// blocks than the pool can free. hitRatePct is unrounded (round once with sharePct).
export function simulatePrefixCache({ requests, blockSize, poolBlocks = Infinity }) {
  requireBlockSize('simulatePrefixCache', blockSize);
  requirePool(poolBlocks);
  requireRequests(requests);
  const pool = makePool(poolBlocks);
  const log = requests.map((r) => logRow(pool, runRequest(pool, r, blockSize)));
  const promptTokens = log.reduce((s, row) => s + row.promptTokens, 0);
  const hitTokens = log.reduce((s, row) => s + row.hitTokens, 0);
  return { log, promptTokens, hitTokens, hitRatePct: (100 * hitTokens) / promptTokens, cachedBlocks: pool.label.size };
}

// Hit tokens per replica for one request. Each replica is a Set (or array) of the keys it holds.
export function routeHits({ replicas, request: r, blockSize }) {
  if (!Array.isArray(replicas) || replicas.length === 0) throw new RangeError('routeHits: replicas must be a non-empty array');
  return replicas.map((keys) => matchPrefix(keys, r.prompt, blockSize).hitTokens);
}

// Price per million input tokens: misses pay the write multiple, hits the read multiple.
export function blendedInputPrice({ hitRate, basePrice, readMult, writeMult = 1 }) {
  if (!(hitRate >= 0 && hitRate <= 1)) throw new RangeError(`blendedInputPrice: hitRate must be between 0 and 1, got ${hitRate}`);
  if (!(basePrice > 0)) throw new RangeError(`blendedInputPrice: basePrice must be positive, got ${basePrice}`);
  if (!(readMult >= 0)) throw new RangeError(`blendedInputPrice: readMult must be at least 0, got ${readMult}`);
  if (!(writeMult > 0)) throw new RangeError(`blendedInputPrice: writeMult must be positive, got ${writeMult}`);
  return (1 - hitRate) * basePrice * writeMult + hitRate * basePrice * readMult;
}
