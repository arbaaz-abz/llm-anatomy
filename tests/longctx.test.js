// math/longctx.js (long-context-attention §6): every worked example of the storyboard, the invariants it names, error cases.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { attentionPattern, readsAndStores, linearState, linearRead } from '../math/longctx.js';
import { randomMatrix, deepFreeze } from '../math/core.js';
import { TOY } from '../math/attention.js';

const S7 = randomMatrix(16, 16, 7);
const readTokens = (mask, row) => mask[row].map((read, j) => (read ? j + 1 : 0)).filter(Boolean);
const round2 = (xs) => xs.map((x) => Math.round(x * 100) / 100);

test('the seeded indexer scores: row 16 of randomMatrix(16, 16, 7), 2 d.p., equals the storyboard', () => {
  assert.deepEqual(round2(S7[15]), [0.85, 0.35, 0.59, -0.33, 0.82, 0.94, 0.69, -0.99, -0.67, 0.51, -0.79, -0.83, 0.82, -0.46, 0.18, -0.48]);
});

test('attentionPattern full: row 16 reads 16, 136 cells in all, stores 16', () => {
  const p = attentionPattern({ n: 16, kind: 'full' });
  assert.equal(p.readsPerRow[15], 16);
  assert.equal(p.cellsRead, 136);
  assert.equal(p.stored, 16);
  assert.deepEqual(readTokens(p.mask, 15), Array.from({ length: 16 }, (_, j) => j + 1));
});

test('attentionPattern window 4: row 16 reads 13–16, 58 cells, stores 4', () => {
  const p = attentionPattern({ n: 16, kind: 'window', window: 4 });
  assert.equal(p.readsPerRow[15], 4);
  assert.equal(p.cellsRead, 58);
  assert.equal(p.stored, 4);
  assert.deepEqual(readTokens(p.mask, 15), [13, 14, 15, 16]);
});

test('attentionPattern sparse top-4 on the seeded scores: row 16 reads 1, 5, 6, 13; 58 cells; stores all 16', () => {
  const p = attentionPattern({ n: 16, kind: 'sparse', topK: 4, scores: S7 });
  assert.deepEqual(readTokens(p.mask, 15), [1, 5, 6, 13]);
  assert.equal(p.readsPerRow[15], 4);
  assert.equal(p.cellsRead, 58);
  assert.equal(p.stored, 16);
});

test('attentionPattern compressed (merge 4, top 1, window 4): row 16 reads tokens 1–4 as one entry plus 13–16; 5 entries; stores 8', () => {
  const p = attentionPattern({ n: 16, kind: 'compressed', window: 4, topK: 1, merge: 4, scores: S7 });
  assert.deepEqual(readTokens(p.mask, 15), [1, 2, 3, 4, 13, 14, 15, 16]);
  assert.equal(p.readsPerRow[15], 5);
  assert.equal(p.stored, 8);
  assert.equal(p.cellsRead, 94);
});

test('compressed: a row whose window covers every earlier token reads no merged entry', () => {
  const p = attentionPattern({ n: 16, kind: 'compressed', window: 4, topK: 1, merge: 4, scores: S7 });
  assert.equal(p.readsPerRow[2], 3);
  assert.deepEqual(readTokens(p.mask, 2), [1, 2, 3]);
  assert.equal(p.readsPerRow[4], 4, 'row 5: the window is 2–5 and no group ends before it');
});

test('invariants: every row reads only keys at or before its query (causal), the diagonal is always read', () => {
  const kinds = [{ kind: 'full' }, { kind: 'window', window: 4 }, { kind: 'sparse', topK: 4, scores: S7 }, { kind: 'compressed', window: 4, topK: 1, merge: 4, scores: S7 }];
  for (const spec of kinds) {
    const { mask } = attentionPattern({ n: 16, ...spec });
    mask.forEach((row, i) => row.forEach((read, j) => { if (j > i) assert.equal(read, false, `${spec.kind}: cell (${i + 1}, ${j + 1}) is in the future`); }));
    if (spec.kind !== 'sparse') mask.forEach((row, i) => assert.equal(row[i], true, `${spec.kind}: row ${i + 1} reads itself`));
  }
});

test('the misconception-1 claim as a test: sparse stores every token, a window stores min(n, window)', () => {
  for (const n of [1, 3, 4, 16, 40]) {
    assert.equal(attentionPattern({ n, kind: 'sparse', topK: 4, scores: randomMatrix(n, n, 3) }).stored, n);
    assert.equal(attentionPattern({ n, kind: 'window', window: 4 }).stored, Math.min(n, 4));
    assert.equal(attentionPattern({ n, kind: 'full' }).cellsRead, (n * (n + 1)) / 2);
  }
});

test('sparse: ties go to the lower index; a row shorter than k reads all it has', () => {
  const flat = Array.from({ length: 6 }, () => Array(6).fill(0.5));
  const p = attentionPattern({ n: 6, kind: 'sparse', topK: 3, scores: flat });
  assert.deepEqual(readTokens(p.mask, 5), [1, 2, 3]);
  assert.deepEqual(readTokens(p.mask, 1), [1, 2]);
  assert.deepEqual(p.readsPerRow, [1, 2, 3, 3, 3, 3]);
});

test('attentionPattern throws a RangeError naming the argument, and never mutates its input', () => {
  assert.throws(() => attentionPattern({ n: 0, kind: 'full' }), /attentionPattern: n must be a positive integer/);
  assert.throws(() => attentionPattern({ n: 16, kind: 'dense' }), /attentionPattern: kind must be one of/);
  assert.throws(() => attentionPattern({ n: 16, kind: 'window' }), /window must be a positive integer/);
  assert.throws(() => attentionPattern({ n: 16, kind: 'sparse', topK: 4 }), /scores must be a 16 × 16 matrix/);
  assert.throws(() => attentionPattern({ n: 16, kind: 'sparse', topK: 0, scores: S7 }), /topK must be a positive integer/);
  assert.throws(() => attentionPattern({ n: 16, kind: 'compressed', window: 4, topK: 1, scores: S7 }), /merge must be a positive integer/);
  const scores = deepFreeze(S7);
  assert.doesNotThrow(() => attentionPattern({ n: 16, kind: 'sparse', topK: 4, scores }));
  assert.throws(() => attentionPattern({ n: 16, kind: 'sparse', topK: 4, scores: [[1]] }), RangeError);
});

test('readsAndStores at real scale: the storyboard table at 1,048,576 tokens', () => {
  const n = 1_048_576;
  assert.deepEqual(readsAndStores({ kind: 'full', n }), { reads: 1_048_576, stored: 1_048_576 });
  assert.deepEqual(readsAndStores({ kind: 'window', n, window: 128 }), { reads: 128, stored: 128 });
  assert.deepEqual(readsAndStores({ kind: 'sparse', n, topK: 2048 }), { reads: 2048, stored: 1_048_576, indexed: 1_048_576 });
  assert.deepEqual(readsAndStores({ kind: 'msa', n, topBlocks: 16, blockSize: 128 }), { reads: 2048, stored: 1_048_576, indexed: 8192 });
  assert.deepEqual(readsAndStores({ kind: 'compressed', n, merge: 4, topK: 1024, window: 128 }), { reads: 1152, stored: 262_272 });
  assert.deepEqual(readsAndStores({ kind: 'compressed', n, merge: 128, topK: Infinity, window: 128 }), { reads: 8320, stored: 8320 });
  assert.deepEqual(readsAndStores({ kind: 'linear', n }), { fixed: true });
});

test('readsAndStores: sparse stores every token for every n; a short context caps reads at n', () => {
  for (const n of [100, 2048, 131_072, 1_048_576]) assert.equal(readsAndStores({ kind: 'sparse', n, topK: 2048 }).stored, n);
  assert.equal(readsAndStores({ kind: 'sparse', n: 100, topK: 2048 }).reads, 100);
  assert.equal(readsAndStores({ kind: 'window', n: 50, window: 128 }).stored, 50);
});

test('readsAndStores throws a RangeError naming the argument', () => {
  assert.throws(() => readsAndStores({ kind: 'full', n: 1.5 }), /readsAndStores: n must be a positive integer/);
  assert.throws(() => readsAndStores({ kind: 'moe', n: 8 }), /readsAndStores: kind must be one of/);
  assert.throws(() => readsAndStores({ kind: 'msa', n: 1024, topBlocks: 16 }), /blockSize must be a positive integer/);
  assert.throws(() => readsAndStores({ kind: 'compressed', n: 1024, merge: 4, topK: 0, window: 128 }), /topK must be a positive integer or Infinity/);
});

const { A } = TOY.heads;
const KEYS = A.K.slice(0, 3);
const VALUES = A.V.slice(0, 3);
const Q_SAT = A.Q[2];

test('linearState after The, cat, sat equals the storyboard state exactly; linearRead gives [−1.5, 6, 1.5, 3.25]', () => {
  const S = linearState({ keys: KEYS, values: VALUES });
  assert.deepEqual(S, [[1.5, -0.5, -1, 0], [0, 3, 0, -1], [-1.5, 0.5, 1, 0], [-0.25, 1.5, 0.5, -0.25]]);
  assert.deepEqual(linearRead(S, Q_SAT), [-1.5, 6, 1.5, 3.25]);
});

test('linearState with gate 0.5 halves the state before each new token; the read is [−0.75, 3, 0.75, 1.75]', () => {
  const S = linearState({ keys: KEYS, values: VALUES, gate: 0.5 });
  assert.deepEqual(S, [[0.75, -0.125, -1, -0.375], [0, 1.5, 0, -0.5], [-0.75, 0.125, 1, 0.375], [-0.25, 0.75, 0.5, 0]]);
  assert.deepEqual(linearRead(S, Q_SAT), [-0.75, 3, 0.75, 1.75]);
});

test('linearState with gate 1 is the sum of v kᵀ, and its size does not depend on how many tokens were read', () => {
  const outer = (v, k) => v.map((vi) => k.map((kj) => vi * kj));
  const sum = KEYS.map((k, t) => outer(VALUES[t], k)).reduce((a, b) => a.map((row, i) => row.map((x, j) => x + b[i][j])));
  assert.deepEqual(linearState({ keys: KEYS, values: VALUES }), sum);
  const long = linearState({ keys: [...A.K, ...A.K, ...A.K], values: [...A.V, ...A.V, ...A.V] });
  assert.equal(long.length, 4);
  assert.ok(long.every((row) => row.length === 4));
});

test('linearState with no tokens is the zero state; a gate of 0 keeps only the newest token', () => {
  assert.deepEqual(linearState({ keys: [], values: [], dKey: 2, dValue: 2 }), [[0, 0], [0, 0]]);
  const newest = VALUES[2].map((v) => KEYS[2].map((k) => v * k + 0)); // + 0 turns −0 into 0
  assert.deepEqual(linearState({ keys: KEYS, values: VALUES, gate: 0 }), newest);
});

test('linear functions throw RangeError on bad shapes or gates and never mutate their inputs', () => {
  assert.throws(() => linearState({ keys: KEYS, values: VALUES.slice(0, 2) }), /linearState: keys and values must pair up/);
  assert.throws(() => linearState({ keys: KEYS, values: VALUES, gate: 1.5 }), /linearState: gate must be a number in \[0, 1\]/);
  assert.throws(() => linearState({ keys: [[1, 2], [1]], values: [[1], [1]] }), /linearState: every key must have the same length/);
  assert.throws(() => linearRead([[1, 2], [3, 4]], [1]), /linearRead: q has 1 entries but the state has 2 columns/);
  const keys = deepFreeze(KEYS);
  const values = deepFreeze(VALUES);
  const S = linearState({ keys, values });
  assert.doesNotThrow(() => linearRead(deepFreeze(S), deepFreeze(Q_SAT)));
});
