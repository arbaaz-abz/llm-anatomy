import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RUNNING_EXAMPLE, stepTime } from '../math/serving.js';
import { kvCacheBytes } from '../math/memory.js';
import {
  PREFIX_REQUESTS, DEFAULT_BLOCK_SIZE, blockKeys, blockTexts, matchPrefix, simulatePrefixCache, routeHits, blendedInputPrice, poolBlocksFor,
} from '../math/prefix.js';

const [A, B, C, D] = PREFIX_REQUESTS;
const BAGAIN = B;
const sim = (opts) => simulatePrefixCache({ requests: PREFIX_REQUESTS, blockSize: 4, ...opts });
const whole = (r) => [...r.prompt, ...r.output];
const close = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

test('PREFIX_REQUESTS is the storyboard table: lengths, answers, deep-frozen', () => {
  assert.deepEqual(PREFIX_REQUESTS.map((r) => [r.id, r.prompt.length, r.output.length]), [['A', 12, 4], ['B', 13, 3], ['C', 20, 4], ['D', 12, 4]]);
  assert.equal(A.output.join(' '), 'The cat sat down');
  assert.equal(B.output.join(' '), 'Yes , fish');
  assert.equal(C.output.join(' '), 'It was warm .');
  assert.equal(D.output.join(' '), 'On the mat .');
  assert.deepEqual(C.prompt, [...A.prompt, ...A.output, 'Why', 'down', 'there', '?']);
  assert.ok(Object.isFrozen(PREFIX_REQUESTS) && Object.isFrozen(A) && Object.isFrozen(A.prompt));
  assert.equal(DEFAULT_BLOCK_SIZE, 4);
});

test('blockKeys: one key per full block, each key chains its parent; texts are the block\'s own tokens', () => {
  const keys = blockKeys(whole(A), 4);
  assert.equal(keys.length, 4);
  assert.equal(new Set(keys).size, 4);
  assert.deepEqual(blockTexts(whole(A), 4), ['You are a cat', '. Reply in rhyme', 'Where did you sit', 'The cat sat down']);
  assert.equal(blockKeys(B.prompt, 4).length, 3, 'B\'s 13th prompt token sits in a partial block');
  assert.equal(blockKeys(whole(B), 4).length, 4, 'B\'s answer fills it');
  assert.deepEqual(blockKeys(whole(C), 4).slice(0, 4), keys, 'C\'s first four blocks are A\'s');
  assert.equal(blockKeys(['x', 'y', 'z'], 4).length, 0);
});

test('blockKeys: a block\'s key depends on everything before it (D repeats A\'s question and shares nothing)', () => {
  const a = blockKeys(A.prompt, 4);
  const d = blockKeys(D.prompt, 4);
  assert.equal(blockTexts(A.prompt, 4)[2], blockTexts(D.prompt, 4)[2], 'same words');
  assert.notEqual(a[2], d[2], 'different keys');
  assert.equal(a.filter((k) => d.includes(k)).length, 0);
  const first = blockKeys(['p', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 4);
  const other = blockKeys(['q', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 4);
  assert.equal(first.filter((k) => other.includes(k)).length, 0, 'prompts that differ in the first token share no key');
});

test('matchPrefix: walks from the first block and stops at the first miss', () => {
  const cache = new Set(blockKeys(whole(A), 4));
  assert.deepEqual(matchPrefix(cache, B.prompt, 4), { hitBlocks: 2, hitTokens: 8 });
  assert.deepEqual(matchPrefix(cache, C.prompt, 4), { hitBlocks: 4, hitTokens: 16 });
  assert.deepEqual(matchPrefix(cache, D.prompt, 4), { hitBlocks: 0, hitTokens: 0 });
  assert.deepEqual(matchPrefix([...cache], C.prompt, 4), { hitBlocks: 4, hitTokens: 16 }, 'an array works too');
  const gap = new Set([blockKeys(whole(A), 4)[1], blockKeys(whole(A), 4)[2]]);
  assert.deepEqual(matchPrefix(gap, C.prompt, 4), { hitBlocks: 0, hitTokens: 0 }, 'a later block alone is no hit');
  assert.deepEqual(matchPrefix(cache, ['a', 'b'], 4), { hitBlocks: 0, hitTokens: 0 });
});

test('simulatePrefixCache: the storyboard\'s worked example, request by request (blocks of 4, pool of 8)', () => {
  const s = sim({ poolBlocks: 8 });
  const row = (i) => s.log[i];
  assert.deepEqual([row(0).promptTokens, row(0).hitTokens, row(0).computed, row(0).blocks, row(0).freeQueue], [12, 0, 12, [0, 1, 2, 3], [4, 5, 6, 7, 3, 2, 1, 0]]);
  assert.deepEqual([row(1).promptTokens, row(1).hitTokens, row(1).computed, row(1).blocks, row(1).freeQueue], [13, 8, 5, [0, 1, 4, 5], [6, 7, 3, 2, 5, 4, 1, 0]]);
  assert.deepEqual([row(2).promptTokens, row(2).hitTokens, row(2).computed, row(2).blocks, row(2).freeQueue], [20, 16, 4, [0, 1, 2, 3, 6, 7], [5, 4, 7, 6, 3, 2, 1, 0]]);
  assert.deepEqual([row(3).promptTokens, row(3).hitTokens, row(3).computed, row(3).blocks, row(3).freeQueue], [12, 0, 12, [5, 4, 7, 6], [3, 2, 1, 0, 6, 7, 4, 5]]);
  assert.deepEqual(row(3).evicted, ['? Yes , fish', 'Do you like fish', 'It was warm .', 'Why down there ?']);
  assert.deepEqual(row(3).evictedBlocks, [5, 4, 7, 6]);
  assert.deepEqual([0, 1, 2].map((i) => row(i).evicted), [[], [], []]);
  assert.deepEqual([s.promptTokens, s.hitTokens], [57, 24]);
  close(s.hitRatePct, 42.105, 1e-3);
  assert.equal(s.cachedBlocks, 8);
  assert.deepEqual(row(3).cached, [0, 1, 2, 3, 4, 5, 6, 7]);
});

test('simulatePrefixCache: "B again" evicts A\'s turn, and keeps B\'s 8-token system prompt', () => {
  const s = sim({ requests: [...PREFIX_REQUESTS, BAGAIN], poolBlocks: 8 });
  const last = s.log.at(-1);
  assert.equal(last.hitTokens, 8);
  assert.deepEqual(last.blocks, [0, 1, 3, 2]);
  assert.deepEqual(last.evicted, ['The cat sat down', 'Where did you sit']);
});

test('simulatePrefixCache: a pool of 6 blocks', () => {
  const s = sim({ poolBlocks: 6 });
  assert.deepEqual([...s.log[2].evicted].sort(), ['? Yes , fish', 'Do you like fish'], 'C evicts B\'s two private blocks');
  assert.deepEqual(s.log[3].evicted.sort(), ['It was warm .', 'Why down there ?', 'The cat sat down', 'Where did you sit'].sort());
  assert.equal(s.cachedBlocks, 6);
});

test('simulatePrefixCache: hit rate by block size with an unlimited pool', () => {
  const expected = { 1: [27, 47.368], 2: [26, 45.614], 4: [24, 42.105], 8: [24, 42.105], 16: [16, 28.070] };
  for (const [bs, [hit, pct]] of Object.entries(expected)) {
    const s = sim({ blockSize: Number(bs) });
    assert.equal(s.promptTokens, 57);
    assert.equal(s.hitTokens, hit, `block size ${bs}`);
    close(s.hitRatePct, pct, 1e-3);
  }
  assert.deepEqual(sim({ blockSize: 16 }).log.map((r) => r.hitTokens), [0, 0, 16, 0]);
  assert.deepEqual(sim({ blockSize: 1 }).log.map((r) => r.hitTokens), [0, 8, 16, 3]);
});

test('the pool scaled to the block size (poolBlocksFor) keeps the try-this numbers at pool 8', () => {
  assert.deepEqual([1, 2, 4, 8, 16].map((bs) => poolBlocksFor(8, bs)), [32, 16, 8, 4, 2]);
  assert.deepEqual([6, 8, 12].map((p) => poolBlocksFor(p, 16)), [2, 2, 3]);
  const rates = [1, 2, 4, 8, 16].map((bs) => sim({ blockSize: bs, poolBlocks: poolBlocksFor(8, bs) }).hitTokens);
  assert.deepEqual(rates, [27, 26, 24, 24, 16]);
  for (const pool of [6, 8, 12]) {
    for (const bs of [1, 2, 4, 8, 16]) assert.doesNotThrow(() => sim({ blockSize: bs, poolBlocks: poolBlocksFor(pool, bs) }), `pool ${pool}, block size ${bs}`);
  }
});

test('simulatePrefixCache: invariants over every block size and pool', () => {
  for (const bs of [1, 2, 4, 8, 16]) {
    for (const pool of [Infinity, 6, 8, 12]) {
      const s = sim({ blockSize: bs, poolBlocks: pool === Infinity ? pool : poolBlocksFor(pool, bs), requests: [...PREFIX_REQUESTS, BAGAIN] });
      s.log.forEach((r) => {
        assert.equal(r.hitTokens % bs, 0);
        assert.ok(r.hitTokens <= r.promptTokens);
        assert.equal(r.computed, r.promptTokens - r.hitTokens);
        if (pool !== Infinity) assert.ok(r.cached.length <= poolBlocksFor(pool, bs));
      });
    }
  }
});

test('simulatePrefixCache: a request that needs more blocks than the pool throws', () => {
  assert.throws(() => sim({ poolBlocks: 3 }), { message: 'pool exhausted at A' });
  assert.throws(() => sim({ poolBlocks: 5 }), { message: 'pool exhausted at C' });
  assert.doesNotThrow(() => sim({ poolBlocks: 6 }));
});

test('simulatePrefixCache: pure, deterministic, inputs not mutated, scrubbing recomputes the same log', () => {
  const frozen = Object.freeze(PREFIX_REQUESTS.map((r) => Object.freeze({ ...r })));
  const before = JSON.stringify(frozen);
  const one = simulatePrefixCache({ requests: frozen, blockSize: 4, poolBlocks: 8 });
  const two = simulatePrefixCache({ requests: frozen, blockSize: 4, poolBlocks: 8 });
  assert.deepEqual(one, two);
  assert.equal(JSON.stringify(frozen), before);
  const prefix = simulatePrefixCache({ requests: frozen.slice(0, 2), blockSize: 4, poolBlocks: 8 });
  assert.deepEqual(prefix.log, one.log.slice(0, 2), 'frame n does not depend on later requests');
});

test('simulatePrefixCache: errors name the argument', () => {
  assert.throws(() => sim({ blockSize: 0 }), { name: 'RangeError', message: /simulatePrefixCache: blockSize must be a positive integer/ });
  assert.throws(() => sim({ poolBlocks: 0 }), { name: 'RangeError', message: /poolBlocks/ });
  assert.throws(() => sim({ poolBlocks: 2.5 }), RangeError);
  assert.throws(() => sim({ requests: [] }), { name: 'RangeError', message: /requests must be a non-empty array/ });
  assert.throws(() => sim({ requests: [{ id: 'X', prompt: [], output: [] }] }), RangeError);
  assert.throws(() => blockKeys(['a'], 0), { name: 'RangeError', message: /blockKeys: blockSize must be a positive integer/ });
  assert.throws(() => blockKeys('abc', 2), RangeError);
});

test('routeHits: sent where A ran, C reuses 16 tokens; sent to B\'s replica, 8', () => {
  const replicas = [new Set(blockKeys(whole(A), 4)), new Set(blockKeys(whole(B), 4))];
  assert.deepEqual(routeHits({ replicas, request: C, blockSize: 4 }), [16, 8]);
  assert.deepEqual(routeHits({ replicas: [[], replicas[1]], request: C, blockSize: 4 }), [0, 8], 'arrays work too');
  assert.throws(() => routeHits({ replicas: [], request: C, blockSize: 4 }), RangeError);
});

test('blendedInputPrice: the storyboard\'s prices', () => {
  const sonnet = (h, writeMult = 1.25) => blendedInputPrice({ hitRate: h, basePrice: 2, readMult: 0.1, writeMult });
  close(sonnet(0), 2.5, 1e-9);
  close(sonnet(24 / 57), 1.5316, 1e-4);
  assert.equal(sonnet(0.421).toFixed(2), '1.53');
  assert.equal(sonnet(0.563).toFixed(2), '1.21');
  assert.equal(sonnet(0.563, 1).toFixed(2), '0.99');
  assert.equal(blendedInputPrice({ hitRate: 0.563, basePrice: 4, readMult: 0.05, writeMult: 1.25 }).toFixed(2), '2.30');
  const deepseek = (h) => blendedInputPrice({ hitRate: h, basePrice: 0.66, readMult: 0.022 / 0.66 });
  assert.equal(deepseek(0).toFixed(2), '0.66');
  assert.equal(deepseek(0.563).toFixed(4), '0.3008');
});

test('blendedInputPrice: at h = 1 it is base times read; without a write premium it never exceeds the base', () => {
  assert.equal(blendedInputPrice({ hitRate: 1, basePrice: 2, readMult: 0.1, writeMult: 1.25 }), 0.2);
  for (const h of [0, 0.25, 0.5, 1]) assert.ok(blendedInputPrice({ hitRate: h, basePrice: 2, readMult: 0.1 }) <= 2);
  assert.throws(() => blendedInputPrice({ hitRate: 1.1, basePrice: 2, readMult: 0.1 }), { name: 'RangeError', message: /blendedInputPrice: hitRate must be between 0 and 1/ });
  assert.throws(() => blendedInputPrice({ hitRate: 0.5, basePrice: -1, readMult: 0.1 }), RangeError);
  assert.throws(() => blendedInputPrice({ hitRate: 0.5, basePrice: 2, readMult: -0.1 }), RangeError);
  assert.throws(() => blendedInputPrice({ hitRate: 0.5, basePrice: 2, readMult: 0.1, writeMult: 0 }), RangeError);
});

test('the scale-up line: a 10,000-token prefix skips 707 ms of prefill and holds 3.28 GB', () => {
  const skipped = stepTime({ ...RUNNING_EXAMPLE, tokens: 10000, seqs: 0, context: 0 });
  close(skipped.timeS * 1000, 707.4, 0.05);
  assert.equal(kvCacheBytes({ bytesPerToken: RUNNING_EXAMPLE.kvBytesPerToken, tokens: 10000 }), 3_276_800_000);
});
