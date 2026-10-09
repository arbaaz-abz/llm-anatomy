import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TOY_REQUESTS } from '../math/serving.js';
import { kvCacheBytes, kvBytesPerToken, kvBytesPerTokenMla, sharePct } from '../math/memory.js';
import {
  blocksNeeded, internalWaste, simulateContiguous, simulatePaged, sharedPrefixBlocksSaved, forkBlocks, kvBytesPerBlock, reserveMaxBytes,
} from '../math/paging.js';

const R = TOY_REQUESTS;
const contig = (step) => simulateContiguous({ requests: R, poolSlots: 48, maxLen: 16, step });
const paged = (blockSize, step, sharedPrefix = 0) => simulatePaged({ requests: R, poolSlots: 48, blockSize, step, sharedPrefix });
const row = (sim) => [sim.useful, sim.waste, sim.free];
const tables = (sim) => Object.fromEntries(sim.live.filter((r) => r.running).map((r) => [r.id, r.table]));
const STEPS = [0, 1, 2, 3, 4, 5, 6];

test('blocksNeeded and internalWaste: the storyboard worked examples', () => {
  assert.deepEqual([blocksNeeded(10, 4), blocksNeeded(5, 2), blocksNeeded(8, 16), blocksNeeded(0, 4)], [3, 3, 1, 0]);
  assert.deepEqual([internalWaste(10, 4), internalWaste(5, 2), internalWaste(8, 16), internalWaste(8, 4)], [2, 1, 8, 0]);
});

test('simulateContiguous: steps 0, 2, 3 and 6 of the fixed scenario', () => {
  const s0 = contig(0);
  assert.deepEqual(row(s0), [23, 25, 0]);
  assert.deepEqual(s0.live.slice(0, 3).map((r) => r.strip), [0, 1, 2]);
  assert.equal(s0.poolSlots, 48);
  const s2 = contig(2);
  assert.deepEqual(row(s2), [29, 19, 0]);
  assert.deepEqual([s2.live[3].id, s2.live[3].waiting, s2.live[3].admitted], ['D', true, null]);
  const s3 = contig(3);
  assert.deepEqual(row(s3), [30, 18, 0]);
  assert.deepEqual([s3.live[3].admitted, s3.live[3].strip], [3, 1]);
  assert.deepEqual(row(contig(6)), [25, 7, 16]);
});

test('simulateContiguous: live entries carry tokens, reserved, and status for every request', () => {
  const s = contig(1);
  assert.deepEqual(s.live.map((r) => [r.id, r.tokens, r.reserved, r.running, r.waiting]), [['A', 9, 16, true, false], ['B', 6, 16, true, false], ['C', 11, 16, true, false], ['D', 0, 0, false, true]]);
  const early = contig(0);
  assert.deepEqual([early.live[3].waiting, early.live[3].running, early.live[3].admitted], [false, false, null]);
  const late = contig(5);
  assert.deepEqual([late.live[0].done, late.live[0].running, late.live[0].tokens, late.live[0].strip], [true, false, 0, null]);
  assert.equal(contig(4).live[0].tokens, 12);
  assert.equal(contig(4).live[0].running, true);
});

test('simulatePaged, block size 4: steps 0, 1, 3 and 4', () => {
  const s0 = paged(4, 0);
  assert.deepEqual([...row(s0), s0.blocksUsed, s0.poolBlocks], [23, 5, 20, 7, 12]);
  assert.deepEqual(s0.live.slice(0, 3).map((r) => [r.id, r.tokens, r.blocks, r.waste, r.table]), [['A', 8, 2, 0, [0, 1]], ['B', 5, 2, 3, [2, 3]], ['C', 10, 3, 2, [4, 5, 6]]]);
  const s1 = paged(4, 1);
  assert.deepEqual([...row(s1), s1.blocksUsed], [32, 8, 8, 10]);
  assert.deepEqual(tables(s1), { A: [0, 1, 7], B: [2, 3], C: [4, 5, 6], D: [8, 9] });
  assert.equal(s1.live[3].admitted, 1);
  const s3 = paged(4, 3);
  assert.deepEqual([...row(s3), s3.blocksUsed], [32, 4, 12, 9]);
  assert.deepEqual(tables(s3), { A: [0, 1, 7], C: [4, 5, 6, 2], D: [8, 9] });
  assert.deepEqual(tables(paged(4, 4)), { A: [0, 1, 7], C: [4, 5, 6, 2], D: [8, 9, 3] });
});

test('simulatePaged: block size 16 is the contiguous scheme, step by step', () => {
  const s1 = paged(16, 1);
  assert.deepEqual([...row(s1), s1.blocksUsed, s1.poolBlocks], [26, 22, 0, 3, 3]);
  assert.equal(s1.live[3].waiting, true);
  STEPS.forEach((step) => {
    const c = contig(step);
    const p = paged(16, step);
    assert.deepEqual(row(p), row(c), `step ${step}`);
    assert.deepEqual(p.live.map((r) => r.admitted), c.live.map((r) => r.admitted), `admitted at step ${step}`);
    assert.deepEqual(p.live.map((r) => (r.running ? r.table[0] : null)), c.live.map((r) => r.strip), `strips at step ${step}`);
  });
});

test('simulatePaged with a shared 4-token system prompt', () => {
  const four = paged(4, 1, 4);
  assert.deepEqual([four.useful, four.logicalTokens, four.waste, four.free, four.blocksUsed, four.blocksSaved], [20, 32, 8, 20, 7, 3]);
  const shared = Object.values(tables(four)).map((t) => t[0]);
  assert.equal(new Set(shared).size, 1);
  const eight = paged(8, 1, 4);
  assert.deepEqual([eight.useful, eight.logicalTokens, eight.waste, eight.free, eight.blocksUsed, eight.poolBlocks, eight.blocksSaved], [32, 32, 16, 0, 6, 6, 0]);
  assert.deepEqual(paged(2, 1, 4).blocksSaved, sharedPrefixBlocksSaved({ running: 4, prefixLen: 4, blockSize: 2 }));
  assert.equal(four.blocksSaved, sharedPrefixBlocksSaved({ running: 4, prefixLen: 4, blockSize: 4 }));
});

test('a shared block stays until its last holder finishes: 2 saved while A, C and D run (step 3), none once only C is left (step 5)', () => {
  const heldBy = (sim) => sim.live.filter((r) => r.running).map((r) => r.table[0]);
  const at3 = paged(4, 3, 4);
  assert.deepEqual([new Set(heldBy(at3)).size, at3.blocksSaved, at3.useful + at3.waste + at3.free], [1, 2, 48]);
  const at5 = paged(4, 5, 4);
  assert.deepEqual([heldBy(at5).length, at5.blocksSaved, at5.useful + at5.waste + at5.free], [1, 0, 48]);
});

test('time-averaged waste over steps 0 to 6: contiguous 34.2%, paged 2.7 / 8.0 / 17.6 / 34.2% for block size 2 / 4 / 8 / 16', () => {
  const mean = (sims) => sharePct(sims.reduce((a, s) => a + s.waste, 0), 48 * STEPS.length);
  assert.equal(mean(STEPS.map(contig)), 34.2);
  assert.deepEqual([2, 4, 8, 16].map((b) => mean(STEPS.map((s) => paged(b, s)))), [2.7, 8.0, 17.6, 34.2]);
});

test('peak paged use at block size 4 is 10 of 12 blocks; the pool is 24 / 12 / 6 / 3 blocks for block size 2 / 4 / 8 / 16', () => {
  assert.deepEqual([1, 2, 4].map((s) => paged(4, s).blocksUsed), [10, 10, 10]);
  assert.equal(Math.max(...STEPS.map((s) => paged(4, s).blocksUsed)), 10);
  assert.deepEqual([2, 4, 8, 16].map((b) => paged(b, 0).poolBlocks), [24, 12, 6, 3]);
});

test('property: useful + waste + free is the pool, nothing exceeds it, waste per request is under one block, no setting runs dry', () => {
  [2, 4, 8, 16].forEach((blockSize) => [0, 4, 8].forEach((prefix) => STEPS.forEach((step) => {
    const s = paged(blockSize, step, prefix);
    const label = `B ${blockSize}, prefix ${prefix}, step ${step}`;
    assert.equal(s.useful + s.waste + s.free, 48, label);
    assert.ok(s.blocksUsed <= s.poolBlocks, label);
    s.live.filter((r) => r.running).forEach((r) => assert.ok(r.waste >= 0 && r.waste <= blockSize - 1, `${label} ${r.id}`));
  })));
  STEPS.forEach((step) => { const c = contig(step); assert.equal(c.useful + c.waste + c.free, 48); });
});

test('simulatePaged throws "pool exhausted at step 1" for a crafted overflow (poolSlots 8, block size 4)', () => {
  assert.doesNotThrow(() => simulatePaged({ requests: R, poolSlots: 8, blockSize: 4, step: 0 }));
  assert.throws(() => simulatePaged({ requests: R, poolSlots: 8, blockSize: 4, step: 1 }), /^Error: pool exhausted at step 1$/);
});

test('forkBlocks: parallel sampling with copy-on-write on the partial block', () => {
  assert.deepEqual(forkBlocks({ prompt: 6, generated: 0, samples: 2, blockSize: 4 }), { physical: 2, withoutSharing: 4, saved: 2, copies: 0 });
  assert.deepEqual(forkBlocks({ prompt: 6, generated: 1, samples: 2, blockSize: 4 }), { physical: 3, withoutSharing: 4, saved: 1, copies: 1 });
  assert.deepEqual(forkBlocks({ prompt: 6, generated: 3, samples: 2, blockSize: 4 }), { physical: 5, withoutSharing: 6, saved: 1, copies: 1 });
  assert.deepEqual(forkBlocks({ prompt: 8, generated: 1, samples: 3, blockSize: 4 }), { physical: 5, withoutSharing: 9, saved: 4, copies: 0 });
  assert.deepEqual(forkBlocks({ prompt: 6, generated: 1, samples: 1, blockSize: 4 }), { physical: 2, withoutSharing: 2, saved: 0, copies: 0 });
});

test('sharedPrefixBlocksSaved: (running - 1) whole blocks, none below one block, none for a single request', () => {
  assert.deepEqual([[4, 4, 4], [4, 4, 8], [1, 4, 4], [0, 4, 4], [3, 8, 4]].map(([running, prefixLen, blockSize]) => sharedPrefixBlocksSaved({ running, prefixLen, blockSize })), [3, 0, 0, 0, 4]);
});

test('bytes: per block, the reservation for one request, and the shared definition of "cache for N tokens"', () => {
  assert.deepEqual([327_680, 70_272, 4_718_592].map((b) => kvBytesPerBlock(b, 16)), [5_242_880, 1_124_352, 75_497_472]);
  assert.deepEqual([[327_680, 131_072], [70_272, 131_072], [4_718_592, 2048]].map(([b, c]) => reserveMaxBytes(b, c)), [42_949_672_960, 9_210_691_584, 9_663_676_416]);
  [[327_680, 131_072], [70_272, 131_072], [4_718_592, 2048]].forEach(([b, c]) => assert.equal(reserveMaxBytes(b, c), kvCacheBytes({ bytesPerToken: b, tokens: c })));
  assert.equal(sharePct(reserveMaxBytes(327_680, 131_072), 80e9), 53.7);
  assert.equal(kvBytesPerToken({ layers: 80, kvHeads: 8, headDim: 128, bytesPerElem: 2 }), 327_680);
  assert.equal(kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 }), 70_272);
});

test('bad arguments throw RangeError naming the function and the argument', () => {
  assert.throws(() => blocksNeeded(-1, 4), /blocksNeeded: tokens must be/);
  assert.throws(() => blocksNeeded(3, 0), /blocksNeeded: blockSize must be/);
  assert.throws(() => internalWaste(2.5, 4), RangeError);
  assert.throws(() => simulatePaged({ requests: R, poolSlots: 50, blockSize: 4, step: 0 }), /simulatePaged: poolSlots must be a multiple of blockSize/);
  assert.throws(() => simulatePaged({ requests: R, poolSlots: 48, blockSize: 4, step: -1 }), /simulatePaged: step must be/);
  assert.throws(() => simulatePaged({ requests: R, poolSlots: 48, blockSize: 4, step: 0, sharedPrefix: -1 }), RangeError);
  assert.throws(() => simulateContiguous({ requests: R, poolSlots: 48, maxLen: 8, step: 0 }), /simulateContiguous: maxLen/);
  assert.throws(() => simulateContiguous({ requests: R, poolSlots: 8, maxLen: 16, step: 0 }), /simulateContiguous: maxLen/);
  assert.throws(() => simulateContiguous({ requests: [], poolSlots: 48, maxLen: 16, step: 0 }), /requests/);
  assert.throws(() => forkBlocks({ prompt: 6, generated: -1, samples: 2, blockSize: 4 }), /forkBlocks: generated must be/);
  assert.throws(() => forkBlocks({ prompt: 6, generated: 0, samples: 0, blockSize: 4 }), /forkBlocks: samples must be/);
  assert.throws(() => sharedPrefixBlocksSaved({ running: -1, prefixLen: 4, blockSize: 4 }), RangeError);
  assert.throws(() => kvBytesPerBlock(0, 16), /kvBytesPerBlock: bytesPerToken must be/);
  assert.throws(() => reserveMaxBytes(327_680, 0), RangeError);
});

test('simulators are pure: frozen inputs, the same input twice gives deepEqual output, and frame n is the same reached in any order', () => {
  const frozen = JSON.stringify(R);
  const a = paged(4, 3);
  const b = paged(4, 3);
  assert.deepEqual(a, b);
  assert.deepEqual(contig(3), contig(3));
  [6, 0, 6, 2, 3].forEach((s) => paged(4, s));
  assert.deepEqual(paged(4, 3), a);
  assert.equal(JSON.stringify(R), frozen);
  assert.ok(Object.isFrozen(R));
  const out = paged(4, 3);
  out.live[0].table.push(99);
  assert.deepEqual(paged(4, 3).live[0].table, [0, 1, 7]);
});
